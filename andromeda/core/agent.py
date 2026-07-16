from textwrap import dedent
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Type
from andromeda.config.config import AgentConfig, AgentState
from langchain.tools import BaseTool
from langchain.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables.config import ensure_config, merge_configs

from andromeda.utils.langtils import get_chat_model
from andromeda.core.middleware import build_middleware

try:
    from langgraph_codeact import create_codeact
except ImportError:
    create_codeact = None
from andromeda.utils.sandbox import mamba_eval
from pyeztrace.tracer import trace
from andromeda.core.workflow import (
    WorkflowBuilder,
    Command
)
from andromeda import HumanMessage, AIMessage, BaseMessage
from andromeda.utils.logger import log_agent
from andromeda.core.middleware.tooling import EnsureToolCallIdsMiddleware
from andromeda.utils.langtils import (
    normalize_message_list,
    _extract_reasoning_text as _reasoning_text_from_content_block,
)


class Agent:
    """
    Represents an intelligent agent capable of processing tasks, invoking tools, and generating responses
    using a language model. The Agent is designed for integration in multi-agent workflows and supports
    both ReAct and CodeAct agent types.

    Core Responsibilities:
        - Initializes and manages a language model for response generation.
        - Manages a set of tools that can be invoked during task execution.
        - Maintains internal memory of conversation history or intermediate states.
        - Supports custom prompts and evaluation functions.
        - Integrates with workflow systems for stepwise execution.

    Attributes:
        name (str): The name of the agent.
        model (BaseChatModel): The initialized language model used for generating responses.
        tools (List[Tool]): List of tools available to the agent for task execution.
        memory (List[BaseMessage]): Internal memory for storing conversation history or intermediate states.
        agent (CompiledStateGraph): The underlying agent instance created with the specified model, tools, and prompt.
        workflow (WorkflowBuilder): The workflow builder instance for this agent.

    Example:
        agent = Agent(agent_config)
        result = agent.workflow.run({"messages": [HumanMessage(content="Hello")]})
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        """
        Initializes the Agent with the specified configuration.

        Args:
            agent_config (AgentConfig): Configuration for the agent.
        """
        self.name: str = agent_config.name
        self.model: BaseChatModel = get_chat_model(model_config=agent_config.model)
        self.tools: List[BaseTool] = agent_config.tools
        self.memory: List[BaseMessage] = []
        self.debug: int = getattr(agent_config, "debug", 0)
        self.recursion_limit: int = agent_config.recursion_limit
        self.state_schema: Type[AgentState] = agent_config.state_schema or AgentState
        self.middleware: List[Any] = [
            EnsureToolCallIdsMiddleware(),
            *build_middleware(
                agent_config.middleware,
                fallback_model=self.model,
            ),
        ]
        self.output_standard: str = agent_config.output_standard
        self.checkpointer = agent_config.checkpointer
        self.agent: CompiledStateGraph = (
            create_agent(
                model=self.model,
                tools=self.tools,
                system_prompt=agent_config.prompt,
                middleware=self.middleware,
                checkpointer=None,
                # response_format is optional; wrap in ToolStrategy only if provided
                # This maintains backwards compatibility with agents that don't use structured output
                state_schema=self.state_schema,
                response_format=ToolStrategy(agent_config.response_format) if agent_config.response_format else None,
            )
            if agent_config.type == "react"
            else (
                create_codeact(
                    model=self.model,
                    tools=self.tools,
                    eval_fn=agent_config.eval_fn or mamba_eval,
                    prompt=agent_config.prompt,
                ).compile(debug=self.debug == 3)
                if create_codeact
                else None
            )
        )
        if not self.agent:
            raise ValueError("CodeAct is not installed")

        # Enable deep tracing only when debug == 2
        if self.debug == 2:
            self.invoke = trace(exclude=["log_*"])(self.invoke)
            self.ainvoke = trace(exclude=["log_*"])(self.ainvoke)
            self.stream = trace(exclude=["log_*"])(self.stream)
            self.astream = trace(exclude=["log_*"])(self.astream)
            self.chat = trace(exclude=["log_*"])(self.chat)
            self.achat = trace(exclude=["log_*"])(self.achat)
            self.task = trace(exclude=["log_*"])(self.task)
            self.atask = trace(exclude=["log_*"])(self.atask)
            self.research = trace(exclude=["log_*"])(self.research)
            self.aresearch = trace(exclude=["log_*"])(self.aresearch)
            self.astream_structured_events = trace(exclude=["log_*"])(
                self.astream_structured_events
            )

        self.workflow = self._build_workflow()
        self._thread_id: Optional[str] = None
        self._metadata: Optional[Mapping[str, Any]] = None

    def set_thread_id(self, thread_id: str) -> None:
        self._thread_id = thread_id
    
    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._metadata = metadata

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------
    def _build_workflow(self) -> WorkflowBuilder:
        workflow = WorkflowBuilder(name=f"{self.name}", checkpointer=self.checkpointer)
        (
            workflow
            .start(self.name)
                .run(self._invoke_agent_step_dispatch)
        )
        return workflow

    def _agent_runnable_config(
        self,
        explicit_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        inherited_config = ensure_config()
        config = (
            merge_configs(inherited_config, explicit_config)
            if explicit_config
            else dict(inherited_config)
        )
        config["recursion_limit"] = self.recursion_limit
        return config

    def _invoke_agent_step_dispatch(
        self,
        payload: Dict[str, Any],
        context: Any = None,
    ) -> Any:
        if context is not None and context.metadata.get("__async__"):
            return self._ainvoke_agent_step(payload)
        return self._invoke_agent_step(payload)

    def _invoke_agent_step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[BaseMessage] = payload.get("messages", [])
        config = self._agent_runnable_config(payload.get("config"))
        state: Any = payload.get("state_kwargs", {})
        # Debug level 1: Log agent input messages
        if self.debug == 1:
            try:
                # input_preview = [getattr(m, "content", str(m)) for m in messages]
                log_agent(self.name, f"Input -> {messages}")
            except Exception:
                pass
        invoke_payload: Dict[str, Any] = {"messages": messages, **state}

        response_state = self.agent.invoke(invoke_payload, config=config)

        return self._format_agent_step_result(response_state)

    async def _ainvoke_agent_step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[BaseMessage] = payload.get("messages", [])
        config = self._agent_runnable_config(payload.get("config"))
        state: Any = payload.get("state_kwargs", {})
        if self.debug == 1:
            try:
                log_agent(self.name, f"Input -> {messages}")
            except Exception:
                pass
        invoke_payload: Dict[str, Any] = {"messages": messages, **state}
        response_state = await self.agent.ainvoke(invoke_payload, config=config)
        return self._format_agent_step_result(response_state)

    def _format_agent_step_result(self, response_state: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "messages": response_state["messages"],
            "state": response_state,
        }
        # Promote structured_response to top-level if present (LangGraph structured output)
        if isinstance(response_state, dict) and "structured_response" in response_state:
            result["structured_response"] = response_state["structured_response"]
        # Debug level 1: Log agent output messages
        if self.debug == 1:
            try:
                output_msgs = (
                    response_state.get("messages", [])
                    if isinstance(response_state, dict)
                    else []
                )
                # output_preview = [m for m in output_msgs]
                output_preview = output_msgs[-1]
                log_agent(self.name, f"Output <- {getattr(output_preview, 'content', output_preview)}")
            except Exception:
                pass
        return result

    def _store_messages(self, messages: List[BaseMessage], remember: str) -> None:
        if remember == "all":
            self.memory.extend(messages)
        elif remember == "last" and messages:
            self.memory.append(messages[-1])

    def _run_workflow(
        self,
        messages: List[BaseMessage],
        *,
        config: Dict[str, Any] | None = None,
        remember: str = "all",
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        state_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"messages": messages, "state_kwargs": state_kwargs or {}}
        if config:
            payload["config"] = config

        self._thread_id = thread_id
        self._metadata = metadata
            
        result = self.workflow.execute(
            state=payload,
            monitor=self.debug >= 1,
            debug=self.debug == 3,
            thread_id=thread_id,
            metadata=metadata,
        )
        execution_data = result if isinstance(result, dict) else {}
        if self.output_standard == "andromeda":
            execution_data["messages"] = normalize_message_list(execution_data.get("messages", []))
        output_messages: List[BaseMessage] = execution_data.get("messages", [])


        self._store_messages(output_messages, remember)

        return execution_data if isinstance(execution_data, dict) else {}

    async def _arun_workflow(
        self,
        messages: List[BaseMessage],
        *,
        config: Dict[str, Any] | None = None,
        remember: str = "all",
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        state_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {"messages": messages, "state_kwargs": state_kwargs or {}}
        if config:
            payload["config"] = config

        self._thread_id = thread_id
        self._metadata = metadata

        result = await self.workflow.aexecute(
            state=payload,
            monitor=self.debug >= 1,
            debug=self.debug == 3,
            thread_id=thread_id,
            metadata=metadata,
        )
        execution_data = result if isinstance(result, dict) else {}
        if self.output_standard == "andromeda":
            execution_data["messages"] = normalize_message_list(execution_data.get("messages", []))
        output_messages: List[BaseMessage] = execution_data.get("messages", [])

        self._store_messages(output_messages, remember)

        return execution_data if isinstance(execution_data, dict) else {}

    def invoke(
        self,
        messages: List[BaseMessage],
        *,
        config: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        state_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> List[BaseMessage]:
        """
        Invokes the agent with the given messages.

        Args:
            messages (List[BaseMessage]): The messages to invoke the agent with.
        """
        result = self._run_workflow(messages, remember="all", config=config, thread_id=thread_id, metadata=metadata, state_kwargs=state_kwargs)
        structured_response: Any = result.get("structured_response", None)
        if structured_response:
            return structured_response
        return result.get("messages", [])

    async def ainvoke(
        self, 
        messages: List[BaseMessage], 
        *, 
        config: Optional[Dict[str, Any]] = None, 
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        state_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> List[BaseMessage]:
        """
        Asynchronously invokes the agent with the given messages.

        Args:
            messages (List[BaseMessage]): The messages to invoke the agent with.
        """
        result = await self._arun_workflow(messages, config=config, remember="all", thread_id=thread_id, metadata=metadata, state_kwargs=state_kwargs)
        structured_response: Any = result.get("structured_response", None)
        if structured_response:
            return structured_response
        return result.get("messages", [])

    def stream(
        self,
        messages: List[BaseMessage],
        *,
        config: Optional[Dict[str, Any]] = None,
        remember: str = "all",
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        metadata: Optional[Mapping[str, Any]] = None,
        state_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Any]:
        """
        Streams the agent.

        Args:
            messages (List[BaseMessage]): The messages to stream the agent.

        Returns:
            Iterator[Any]: The streamed agent.
        """
        if stream_mode not in ["updates", "values", "messages"]:
            raise ValueError(f"Invalid stream_mode: {stream_mode}")

        payload: Dict[str, Any] = {"messages": messages, "state_kwargs": state_kwargs or {}}
        if config:
            payload["config"] = config

        self._thread_id = thread_id
        self._metadata = metadata

        last_messages: List[BaseMessage] | None = None
        try:
            for chunk in self.workflow.stream(
                state=payload,
                thread_id=thread_id,
                stream_mode=stream_mode,
                metadata=metadata,
            ):
                def _normalize_msg_lists_in_dict(d: dict):
                    """Recursively normalize any list of BaseMessages found under 'messages' keys in nested dicts."""
                    for key, value in list(d.items()):
                        # Always normalize any key named 'messages' if it's a list of BaseMessages.
                        if key == "messages" and isinstance(value, list) and all(isinstance(m, BaseMessage) for m in value):
                            d[key] = normalize_message_list(value)
                            nonlocal last_messages
                            last_messages = d[key]
                        elif isinstance(value, dict):
                            _normalize_msg_lists_in_dict(value)
                        elif isinstance(value, tuple):
                            # This will not update parent, but log for clarity
                            # Tuple of BaseMessages normalization (untouched in place)
                            pass
                        elif isinstance(value, BaseMessage):
                            # Single BaseMessage under non-messages key (not normalized)
                            pass
                        elif isinstance(value, list) and all(isinstance(m, BaseMessage) for m in value):
                            # List of BaseMessages under non-messages key (not normalized)
                            pass

                if isinstance(chunk, dict):
                    # This will normalize any nested 'messages' lists found in the chunk's dict structure.
                    _normalize_msg_lists_in_dict(chunk)

                    # Always track last top-level messages for remember=last/all.
                    chunk_messages = chunk.get("messages")
                    if isinstance(chunk_messages, list):
                        last_messages = chunk_messages

                    # Normalize message-like payloads inside traced event data,
                    # e.g. {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(...)}}
                    # and {'event': 'on_chat_model_end', 'data': {'output': AIMessage(...)}}
                    if "data" in chunk and isinstance(chunk["data"], dict):
                        for data_key in ("chunk", "output"):
                            inner = chunk["data"].get(data_key)
                            if isinstance(inner, list) and all(isinstance(m, BaseMessage) for m in inner):
                                chunk["data"][data_key] = normalize_message_list(inner)
                            elif isinstance(inner, BaseMessage):
                                chunk["data"][data_key] = normalize_message_list([inner]).pop()

                # The below already handle chunk values as lists/tuple/BaseMessage.
                if isinstance(chunk, list) and all(isinstance(m, BaseMessage) for m in chunk):
                    chunk = normalize_message_list(chunk)
                    last_messages = chunk

                if isinstance(chunk, BaseMessage):
                    chunk = normalize_message_list([chunk])
                    for m in chunk:
                        yield m
                    continue

                if isinstance(chunk, tuple):
                    # Tuples of BaseMessages are normalized but replaced in-place only if needed.
                    chunk = tuple(normalize_message_list(chunk))
                    last_messages = chunk
                yield chunk
        finally:
            if isinstance(last_messages, list):
                self._store_messages(last_messages, remember)

    async def astream(
        self,
        messages: List[BaseMessage],
        *,
        config: Optional[Dict[str, Any]] = None,
        remember: str = "all",
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        metadata: Optional[Mapping[str, Any]] = None,
        state_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        """
        Asynchronously streams the agent.

        Args:
            messages (List[BaseMessage]): The messages to stream the agent.

        Returns:
            AsyncIterator[Any]: The streamed agent.
        """
        if stream_mode not in ["updates", "values", "messages", "events", "tasks", "checkpoints"]:
            raise ValueError(f"Invalid stream_mode: {stream_mode}")

        payload: Dict[str, Any] = {"messages": messages, "state_kwargs": state_kwargs or {}}
        if config:
            payload["config"] = config

        last_messages: list[Any] | None = None

        self._thread_id = thread_id
        self._metadata = metadata

        try:
            async for chunk in self.workflow.astream(
                state=payload,
                thread_id=thread_id,
                stream_mode=stream_mode,
                metadata=metadata,
            ):
                def _normalize_msg_lists_in_dict(d: dict):
                    """Recursively normalize any list of BaseMessages found under 'messages' keys in nested dicts."""
                    for key, value in list(d.items()):
                        # Always normalize any key named 'messages' if it's a list of BaseMessages.
                        if key == "messages" and isinstance(value, list) and all(isinstance(m, BaseMessage) for m in value):
                            d[key] = normalize_message_list(value)
                            nonlocal last_messages
                            last_messages = d[key]
                        elif isinstance(value, dict):
                            _normalize_msg_lists_in_dict(value)
                        elif isinstance(value, tuple):
                            # This will not update parent, but log for clarity
                            # Tuple of BaseMessages normalization (untouched in place)
                            pass
                        elif isinstance(value, BaseMessage):
                            # Single BaseMessage under non-messages key (not normalized)
                            pass
                        elif isinstance(value, list) and all(isinstance(m, BaseMessage) for m in value):
                            # List of BaseMessages under non-messages key (not normalized)
                            pass

                if isinstance(chunk, dict):
                    # This will normalize any nested 'messages' lists found in the chunk's dict structure.
                    _normalize_msg_lists_in_dict(chunk)

                    # Always track last top-level messages for remember=last/all.
                    chunk_messages = chunk.get("messages")
                    if isinstance(chunk_messages, list):
                        last_messages = chunk_messages

                    # Normalize message-like payloads inside traced event data,
                    # e.g. {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(...)}}
                    # and {'event': 'on_chat_model_end', 'data': {'output': AIMessage(...)}}
                    if "data" in chunk and isinstance(chunk["data"], dict):
                        for data_key in ("chunk", "output"):
                            inner = chunk["data"].get(data_key)
                            if isinstance(inner, list) and all(isinstance(m, BaseMessage) for m in inner):
                                chunk["data"][data_key] = normalize_message_list(inner)
                            elif isinstance(inner, BaseMessage):
                                chunk["data"][data_key] = normalize_message_list([inner]).pop()

                # The below already handle chunk values as lists/tuple/BaseMessage.
                if isinstance(chunk, list) and all(isinstance(m, BaseMessage) for m in chunk):
                    chunk = normalize_message_list(chunk)
                    last_messages = chunk

                if isinstance(chunk, BaseMessage):
                    chunk = normalize_message_list([chunk])
                    for m in chunk:
                        yield m
                    continue

                if isinstance(chunk, tuple):
                    # Tuples of BaseMessages are normalized but replaced in-place only if needed.
                    chunk = tuple(normalize_message_list(chunk))
                    last_messages = chunk

                yield chunk
        finally:
            if isinstance(last_messages, list):
                self._store_messages(last_messages, remember)

    def handoff(self, messages: List[BaseMessage], thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> List[BaseMessage]:
        """
        Hands off the agent to a supervisor.

        Args:
            messages (List[BaseMessage]): The messages to hand off the agent.

        Returns:
            List[BaseMessage]: The messages from the handoff.
        """
        result = self._run_workflow(messages, remember="none", thread_id=thread_id, metadata=metadata)
        return result.get("messages", [])

    def chat(self, message: str, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> List[BaseMessage]:
        """
        Chats with the agent.

        Args:
            message (str): The message to chat with the agent.

        Returns:
            List[BaseMessage]: The messages from the chat.
        """
        messages = self.memory + [HumanMessage(content=message)]
        result = self._run_workflow(messages, remember="last", thread_id=thread_id, metadata=metadata)
        return result.get("messages", [])

    async def achat(self, message: str, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> List[BaseMessage]:
        """
        Asynchronously chats with the agent.

        Args:
            message (str): The message to chat with the agent.

        Returns:
            List[BaseMessage]: The messages from the chat.
        """
        messages = self.memory + [HumanMessage(content=message)]
        result = await self._arun_workflow(messages, remember="last", thread_id=thread_id, metadata=metadata)
        return result.get("messages", [])

    def task(self, task: str, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """
        Runs the task workflow.

        Args:
            task (str): The task to run.

        Returns:
            str: The content of the last message from the task workflow.
        """
        system_prompt = dedent(
            """
            Current Goal: Task
            Task: {task}
            Approach: Use the tools provided to complete the task when tools are useful or required.
            Effort: Work carefully and iterate as needed, but stop once the task is complete.
            Output: Follow any output format requested in the task. If none is specified, provide a concise markdown result with what you did, what you found, and any limitations."""
        ).format(task=task)
        result = self._run_workflow(
            [HumanMessage(content=system_prompt)],
            remember="last",
            thread_id=thread_id,
            metadata=metadata,
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else None

    async def atask(self, task: str, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """
        Asynchronously runs the task workflow.

        Args:
            task (str): The task to run.

        Returns:
            str: The content of the last message from the task workflow.
        """
        system_prompt = dedent(
            """
            Current Goal: Task
            Task: {task}
            Approach: Use the tools provided to complete the task when tools are useful or required.
            Effort: Work carefully and iterate as needed, but stop once the task is complete.
            Output: Follow any output format requested in the task. If none is specified, provide a concise markdown result with what you did, what you found, and any limitations.
        """).format(task=task)
        
        result = await self._arun_workflow(
            [HumanMessage(content=system_prompt)],
            remember="last",
            thread_id=thread_id,
            metadata=metadata,
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else None

    def research(self, task: str, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """
        Runs the research workflow.

        Args:
            task (str): The task to research.

        Returns:
            str: The content of the last message from the research workflow.
        """
        system_prompt = dedent(
            """
            Current Goal: Research
            Research Task: {task}
            Approach: Use the tools provided to collect information when tools are useful or required.
            Effort: Work carefully and iterate as needed, but stop once the research task is complete.
            Output: Follow any output format requested in the task. If none is specified, provide a concise markdown research result with findings, evidence, and limitations."""
        ).format(task=task)
        result = self._run_workflow(
            [HumanMessage(content=system_prompt)],
            remember="last",
            thread_id=thread_id,
            metadata=metadata,
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else None

    async def aresearch(self, task: str, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """
        Asynchronously runs the research workflow.

        Args:
            task (str): The task to research.

        Returns:
            str: The content of the last message from the research workflow.
        """
        system_prompt = dedent(
            """
            Current Goal: Research
            Research Task: {task}
            Approach: Use the tools provided to collect information when tools are useful or required.
            Effort: Work carefully and iterate as needed, but stop once the research task is complete.
            Output: Follow any output format requested in the task. If none is specified, provide a concise markdown research result with findings, evidence, and limitations."""
        ).format(task=task)
        result = await self._arun_workflow(
            [HumanMessage(content=system_prompt)],
            remember="last",
            thread_id=thread_id,
            metadata=metadata,
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else None

    async def astream_structured_events(
        self, messages: List[BaseMessage], thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None, state_kwargs: Optional[Mapping[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams the agent's structured events.

        Args:
            messages (List[BaseMessage]): The messages to stream the agent's structured events with.

        Returns:
            AsyncGenerator[Dict[str, Any], None]: The structured events.
        """
        def _content_to_text(message: Any) -> Optional[str]:
            if isinstance(message, dict):
                msgs = message.get("messages")
                if isinstance(msgs, list) and msgs:
                    return _content_to_text(msgs[-1])
                return None

            if isinstance(message, BaseMessage):
                content = message.content
            elif hasattr(message, "content"):
                content = getattr(message, "content")
            else:
                return None

            if isinstance(content, str):
                return content

            output_standard = getattr(self, "output_standard", "andromeda")
            if output_standard == "andromeda" and isinstance(content, list):
                ak: Dict[str, Any] = {}
                if isinstance(message, BaseMessage):
                    raw_ak = getattr(message, "additional_kwargs", None)
                    if isinstance(raw_ak, dict):
                        ak = dict(raw_ak)
                normalized = normalize_message_list([AIMessage(content=content, additional_kwargs=ak)])
                if normalized and isinstance(normalized[0].content, str):
                    return normalized[0].content

            return str(content)

        def _extract_reasoning_text(message: Any) -> Optional[str]:
            output_standard = getattr(self, "output_standard", "andromeda")
            if not output_standard == "andromeda":
                return None
            if isinstance(message, dict):
                msgs = message.get("messages")
                if isinstance(msgs, list) and msgs:
                    return _extract_reasoning_text(msgs[-1])
                return None
            if isinstance(message, BaseMessage):
                ak = getattr(message, "additional_kwargs", None) or {}
                if isinstance(ak, dict):
                    for key in ("reasoning_content", "reasoning"):
                        val = ak.get(key)
                        if isinstance(val, str) and val.strip():
                            return val
                # LiteLLM / providers may put thinking only in block list content, not kwargs.
                blk_content = message.content
                if isinstance(blk_content, list):
                    parts: List[str] = []
                    for block in blk_content:
                        if isinstance(block, dict):
                            t = _reasoning_text_from_content_block(block)
                            if t:
                                parts.append(t)
                    if parts:
                        return "".join(parts)
                return None
            additional_kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                for key in ("reasoning_content", "reasoning"):
                    val = additional_kwargs.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
            return None

        new_message = ""
                
        async for event in self.astream(messages, stream_mode="events", thread_id=thread_id, metadata=metadata, state_kwargs=state_kwargs):
            if event["event"] == "on_chat_model_stream" and event.get("metadata", {}).get("langgraph_node", "") != "tools":
                chunk_text = _content_to_text(event.get("data", {}).get("chunk"))
                reasoning_text = _extract_reasoning_text(event.get("data", {}).get("chunk"))
                if reasoning_text:
                    yield {
                        "type": "reasoning_chunk",
                        "content": reasoning_text,
                        "agent": event.get("metadata", {}).get("checkpoint_ns", "").split("|")[-1].split(":")[0]
                    }
                if chunk_text:
                    new_message += chunk_text
                    yield {
                        "type": "response_chunk",
                        "content": chunk_text,
                        "agent": event.get("metadata", {}).get("checkpoint_ns", "").split("|")[-1].split(":")[0]
                    }
            if event["event"] == "on_tool_start":
                yield {
                    "type": "tool_call",
                    "content": f"Using {event['name']}",
                    "raw": {
                        "name": event["name"],
                        "input": event.get("data", {}).get("input", {}),
                    }
                }
            if event["event"] == "on_tool_end":
                output = event.get("data", {}).get("output", None)
                output_text = _content_to_text(output)
                yield {
                    "type": "tool_result",
                    "content": f"Tool {event['name']} finished",
                    "raw": {
                        "name": event["name"],
                        "output": output_text,
                    }
                }
            
            if event["event"] == "on_chat_model_end" and event.get("metadata", {}).get("langgraph_node", "") != "tools":
                output = event.get("data", {}).get("output", None)
                output_text = _content_to_text(output)
                yield {
                    "type": "response_end",
                    "content": output_text,
                }
        self.memory.append(AIMessage(content=new_message))
        yield {"type": "end"}


if __name__ == "__main__":
    agent = Agent(AgentConfig(name="test", model="gpt-4o-mini", type="react", tools=[]))
