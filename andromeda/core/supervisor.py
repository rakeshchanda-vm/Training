import asyncio
import contextvars
from concurrent.futures import Future
from dataclasses import dataclass
from hashlib import sha256
from textwrap import dedent
import threading
import time
from typing import Dict, List, Literal, OrderedDict, Tuple, Union, Optional, Mapping, Any
import uuid

from pydantic import BaseModel, Field

from andromeda.tools import tool, BaseTool

from andromeda.config.config import AgentConfig, SupervisorConfig
from andromeda.core.agent import Agent
from andromeda import HumanMessage, AIMessage
from andromeda.utils.logger import log_supervisor
from andromeda.utils.prompts import supervisor_task_routing_prompt_v2


_ASYNC_TASK_SCOPE: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "andromeda_supervisor_async_task_scope",
    default=None,
)
_SUPERVISOR_THREAD_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "andromeda_supervisor_thread_id",
    default=None,
)
_SUPERVISOR_METADATA: contextvars.ContextVar[Optional[Mapping[str, Any]]] = (
    contextvars.ContextVar(
        "andromeda_supervisor_metadata",
        default=None,
    )
)


@dataclass
class _BackgroundTaskRecord:
    task_id: str
    scope_id: str
    agent_name: str
    prompt: str
    created_at: float
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    cancel_requested: bool = False
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    future: Optional[Future] = None


class Supervisor(Agent):
    def __init__(
        self,
        agents: List[Union[AgentConfig, Agent]],
        config: SupervisorConfig,
    ):
        """
        Supervisor is the router/orchestrator agent for specialist agents.
        It can:
            - Respond directly to the user.
            - Route specific tasks/questions to the most appropriate specialist agent.
            - Optionally manage and track a high-level todo list (when ``config.enable_planning``
              is enabled), exposing tools to inspect and update todos.

        Args:
            agents (List[Union[AgentConfig, Agent]]): The agents to coordinate.
            config (SupervisorConfig): The configuration for the supervisor.

        Returns:
            None
        """

        self.agents: List[Agent] = []
        for entry in agents:
            agent = entry if isinstance(entry, Agent) else Agent(entry)
            self.agents.append(agent)

        self.agent_map: Dict[str, Agent] = {agent.name: agent for agent in self.agents}
        # Build an instance-local config so the final system prompt and supervisor tools do
        # not leak back into a caller-owned config object and get wrapped again elsewhere.
        self.config: SupervisorConfig = config.model_copy(
            update={"tools": list(config.tools)}
        )
        self.allowed_route_types: List[Literal["chat", "task", "research", "handoff"]] = self.config.allowed_route_types
        self.allow_parallel_agents: bool = self.config.allow_parallel_agents
        self.allow_async_tasks: bool = self.config.allow_async_tasks
        self._async_task_lock = threading.RLock()
        self._async_tasks: Dict[str, _BackgroundTaskRecord] = {}
        self._async_task_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_task_loop_thread: Optional[threading.Thread] = None
        supervisor_tools = self.make_supervisor_tools(
            self.agents, required_plan=self.config.enable_planning
        )
        self.config.tools = list(self.config.tools) + supervisor_tools
        base_prompt = self._build_system_prompt(extended_prompt=self.config.prompt or "")
        self.config.prompt = base_prompt
        super().__init__(agent_config=self.config)

        self.plan: OrderedDict[str, Dict[str, str]] = OrderedDict()

    def _build_system_prompt(self, *, extended_prompt: str) -> str:
        return supervisor_task_routing_prompt_v2(
            agents=self.agent_map,
            extended_prompt=extended_prompt,
            enable_planning=self.config.enable_planning,
            allow_parallel_agents=self.allow_parallel_agents,
            allow_async_tasks=self.allow_async_tasks,
            allowed_route_types=list(self.allowed_route_types),
        )

    def extract_plan(self, state: dict) -> dict:
        base_plan = state.get("plan", [])

        if isinstance(base_plan, dict):
            self.plan = OrderedDict(base_plan)
            return self.plan

        for item in base_plan:
            plan_id = sha256(item.encode()).hexdigest()[:5]
            if plan_id in self.plan:
                continue
            self.plan[plan_id] = {"status": "pending", "item": item}
        return self.plan

    def _cleanup_async_tasks(
        self,
        *,
        cancel_running: bool = False,
        scope_id: Optional[str] = None,
    ) -> None:
        loop_to_stop: Optional[asyncio.AbstractEventLoop] = None
        loop_thread: Optional[threading.Thread] = None
        futures_to_wait: List[Future] = []
        with self._async_task_lock:
            terminal = {"completed", "failed", "cancelled"}
            matching_task_ids = [
                task_id
                for task_id, record in self._async_tasks.items()
                if scope_id is None or record.scope_id == scope_id
            ]
            if cancel_running:
                for task_id in matching_task_ids:
                    record = self._async_tasks[task_id]
                    if record.status in terminal:
                        continue
                    record.cancel_requested = True
                    if record.future is not None and record.future.cancel():
                        futures_to_wait.append(record.future)
                    elif record.future is not None:
                        futures_to_wait.append(record.future)
                    if record.status in {"pending", "running"}:
                        record.status = "cancelling"
                for task_id in matching_task_ids:
                    self._async_tasks.pop(task_id, None)
                if not self._async_tasks:
                    loop_to_stop = self._async_task_loop
                    loop_thread = self._async_task_loop_thread
                    self._async_task_loop = None
                    self._async_task_loop_thread = None
            else:
                self._async_tasks = {
                    task_id: record
                    for task_id, record in self._async_tasks.items()
                    if record.status not in terminal
                    or (scope_id is not None and record.scope_id != scope_id)
                }
                if not self._async_tasks:
                    loop_to_stop = self._async_task_loop
                    loop_thread = self._async_task_loop_thread
                    self._async_task_loop = None
                    self._async_task_loop_thread = None

        for future in futures_to_wait:
            try:
                future.result(timeout=1)
            except BaseException:
                pass

        if loop_to_stop is not None and loop_to_stop.is_running():
            loop_to_stop.call_soon_threadsafe(loop_to_stop.stop)
        if (
            loop_thread is not None
            and loop_thread.is_alive()
            and loop_thread is not threading.current_thread()
        ):
            loop_thread.join(timeout=1)

    def _current_thread_id(self) -> Optional[str]:
        return _SUPERVISOR_THREAD_ID.get()

    def _current_metadata(self) -> Optional[Mapping[str, Any]]:
        return _SUPERVISOR_METADATA.get()

    def _incomplete_async_tasks_summary(self) -> Optional[str]:
        """Return a short summary if any background tasks are not finished for this run scope."""
        scope_id = _ASYNC_TASK_SCOPE.get()
        if scope_id is None:
            return None
        terminal = {"completed", "failed", "cancelled"}
        lines: List[str] = []
        with self._async_task_lock:
            for task_id, record in self._async_tasks.items():
                if record.scope_id != scope_id:
                    continue
                if record.status in terminal:
                    continue
                lines.append(
                    f"  - {task_id}: status={record.status}, agent={record.agent_name}"
                )
        if not lines:
            return None
        return "Incomplete background tasks:\n" + "\n".join(lines)

    def _ensure_async_task_loop(self) -> asyncio.AbstractEventLoop:
        with self._async_task_lock:
            if self._async_task_loop is not None and self._async_task_loop.is_running():
                return self._async_task_loop

            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run_loop() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.close()

            thread = threading.Thread(
                target=run_loop,
                name="andromeda-supervisor-async",
                daemon=True,
            )
            thread.start()
            ready.wait()
            self._async_task_loop = loop
            self._async_task_loop_thread = thread
            return loop

    def make_supervisor_tools(
        self, agents: List[Agent], required_plan: bool = False
    ) -> List[BaseTool]:
        def random_id() -> str:
            return str(uuid.uuid4())[:8]

        description = dedent(
            """Route a task to a specialized agent to help you.
            Available agents and their tools: {agents}
            The agents do not share your memory unless you handoff to them, so you cannot ask follow up questions without providing the agent with the necessary context.
            Use task, chat, research when you want the agent to perform an independent task.
            Use handoff when you want to transfer the conversation to another agent. This is useful when you want want the agent to use information that you have gathered from other agents.
            Args:
            agent_name: The name of the agent to route the task to.
            prompt: The prompt to give the LLM agent. Include necessary context and instructions.
            Returns:
            The response from the agent."""
        ).format(agents=', '.join([f'{agent.name}: {agent.tools}' for agent in agents]))

        @tool(description=description)
        def route_to_agent(
            agent_name: str,
            prompt: str,
            invocation_type: Literal[*self.allowed_route_types],
        ) -> str:
            if agent_name not in [agent.name for agent in agents]:
                raise ValueError(
                    f"Agent {agent_name} not found. Use one of the following agents: {', '.join([agent.name for agent in agents])}"
                )
            if invocation_type not in self.allowed_route_types:
                if len(self.allowed_route_types) == 1 and invocation_type == "":
                    invocation_type = self.allowed_route_types[0]
                else:
                    raise ValueError(
                        f"Invocation type {invocation_type} not allowed. Use one of the following types: {', '.join(self.allowed_route_types)}"
                    )

            agent = next(a for a in agents if a.name == agent_name)
            if self.debug == 1:
                log_supervisor(f"Routing {invocation_type} to {agent_name}: {prompt}")

            thread_id = self._current_thread_id()
            metadata = self._current_metadata()
            if invocation_type == "chat":
                messages = agent.chat(prompt, thread_id=thread_id, metadata=metadata)
                self.memory.extend(messages)
                return messages[-1].content if messages else None
            if invocation_type == "task":
                message = agent.task(prompt, thread_id=thread_id, metadata=metadata)
                if message is not None:
                    self.memory.append(AIMessage(content=message))
                return message
            if invocation_type == "research":
                message = agent.research(prompt, thread_id=thread_id, metadata=metadata)
                if message is not None:
                    self.memory.append(AIMessage(content=message))
                return message
            if invocation_type == "handoff":
                state_messages = self.memory + [HumanMessage(content=prompt)]
                messages = agent.handoff(state_messages, thread_id=thread_id, metadata=metadata)
                self.memory.extend(messages)
                return messages[-1].content if messages else None

        parallel_task_description = dedent(
            """Assign tasks to multiple agents in parallel.
            Available agents and their tools: {agents}
            Use this tool for independent tasks that do not require coordination or collaboration with other agents.
            Ensure you account for your token limits when using this tool.
            Args:
            tasks: A list of task objects. Each task object must include agent_name and prompt.
            Returns:
            String of the results from each agent."""
        ).format(agents=', '.join([f'{agent.name}: {agent.tools}' for agent in agents]))

        class _ParallelTaskSpec(BaseModel):
            agent_name: str = Field(description="The name of the agent to execute the task.")
            prompt: str = Field(description="The prompt to give the agent.")

        class _ParallelTaskArgs(BaseModel):
            tasks: List[_ParallelTaskSpec] = Field(
                description="Independent agent tasks to execute in parallel.",
            )

        @tool(args_schema=_ParallelTaskArgs, description=parallel_task_description)
        def parallel_task(tasks: List[_ParallelTaskSpec]) -> str:
            """
            Assign tasks to multiple agents in parallel and collect their results.
            Args:
                tasks: A list of task objects with agent_name and prompt fields.
            Returns:
                String representation of results from each agent.
            """
            if not self.allow_parallel_agents:
                raise ValueError("Parallel agents are not allowed. Set allow_parallel_agents to True in the supervisor config to allow parallel agents.")

            if not tasks:
                raise ValueError("At least one task is required.")

            agent_lookup = {agent.name: agent for agent in agents}
            agent_names = list(agent_lookup.keys())
            normalized_tasks: List[Tuple[str, str]] = []

            for task in tasks:
                agent_name = task.agent_name
                prompt = task.prompt
                if agent_name not in agent_lookup:
                    raise ValueError(
                        f"Agent {agent_name} not found. Use one of the following agents: {', '.join(agent_names)}"
                    )
                if not prompt or prompt.strip() == "":
                    raise ValueError("Task prompt cannot be empty.")
                normalized_tasks.append((agent_name, prompt))

            results: List[Optional[str]] = [None] * len(normalized_tasks)
            # Execute tasks in parallel (threaded).
            import concurrent.futures
            from andromeda.core.workflow.eval_scheduler import get_safe_process_count

            thread_id = self._current_thread_id()
            metadata = self._current_metadata()
            def run_task(agent_name: str, prompt: str) -> Tuple[str, Any]:
                agent = agent_lookup[agent_name]
                result = agent.task(prompt, thread_id=thread_id, metadata=metadata)
                return (agent_name, result)

            max_workers = max(1, min(len(normalized_tasks), get_safe_process_count()))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="andromeda-supervisor"
            ) as executor:
                future_to_task = {
                    executor.submit(run_task, agent_name, prompt): (idx, agent_name, prompt)
                    for idx, (agent_name, prompt) in enumerate(normalized_tasks)
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    idx, agent_name, _prompt = future_to_task[future]
                    try:
                        agent_name, result = future.result()
                    except Exception as exc:
                        result = f"Error ({exc.__class__.__name__}): {exc}"
                    results[idx] = f"Agent '{agent_name}': {result}"

            return "\n".join(r for r in results if r is not None)

        def run_and_track_task(agent: Agent, prompt: str) -> str:
            if not prompt or prompt.strip() == "":
                raise ValueError("Task prompt cannot be empty.")
            
            if len([task for task in self._async_tasks.values() if task.status in {"pending", "running"}]) >= 10:
                raise ValueError("Maximum number of async tasks (10) reached. Please wait for a task to complete before starting a new one.")

            with self._async_task_lock:
                task_id = random_id()
                while task_id in self._async_tasks:
                    task_id = random_id()
                scope_id = _ASYNC_TASK_SCOPE.get() or f"supervisor:{id(self)}"
                thread_id = self._current_thread_id()
                current_metadata = self._current_metadata()
                metadata = dict(current_metadata) if current_metadata is not None else None
                record = _BackgroundTaskRecord(
                    task_id=task_id,
                    scope_id=scope_id,
                    agent_name=agent.name,
                    prompt=prompt,
                    created_at=time.monotonic(),
                )
                self._async_tasks[task_id] = record

            async def runner() -> Optional[str]:
                with self._async_task_lock:
                    if record.cancel_requested:
                        record.status = "cancelled"
                        record.finished_at = time.monotonic()
                        return None
                    record.status = "running"
                    record.started_at = time.monotonic()

                try:
                    result = await agent.atask(
                        prompt,
                        thread_id=thread_id,
                        metadata=metadata,
                    )
                except asyncio.CancelledError:
                    with self._async_task_lock:
                        record.status = "cancelled"
                        record.result = None
                        record.finished_at = time.monotonic()
                    raise
                except BaseException as exc:  # noqa: BLE001
                    with self._async_task_lock:
                        record.status = "failed"
                        record.error = f"{exc.__class__.__name__}: {exc}"
                        record.finished_at = time.monotonic()
                    return None

                with self._async_task_lock:
                    if record.cancel_requested:
                        record.status = "cancelled"
                        record.result = None
                    else:
                        record.status = "completed"
                        record.result = result
                    record.finished_at = time.monotonic()
                return result

            loop = self._ensure_async_task_loop()
            try:
                future = asyncio.run_coroutine_threadsafe(runner(), loop)
            except BaseException:
                with self._async_task_lock:
                    self._async_tasks.pop(task_id, None)
                raise

            with self._async_task_lock:
                record.future = future

            return task_id

        def _task_elapsed(record: _BackgroundTaskRecord) -> float:
            end = record.finished_at if record.finished_at is not None else time.monotonic()
            return max(0.0, end - record.created_at)

        def _format_task_status(record: _BackgroundTaskRecord) -> str:
            elapsed = _task_elapsed(record)
            lines = [
                f"Task {record.task_id}",
                f"status: {record.status}",
                f"agent: {record.agent_name}",
                f"elapsed_seconds: {elapsed:.2f}",
            ]
            if record.status == "completed":
                lines.append(f"result: {record.result}")
            elif record.status == "failed":
                lines.append(f"error: {record.error}")
            elif record.status == "cancelling":
                lines.append("message: cancellation requested; waiting for the async task to stop.")
            elif record.status == "cancelled":
                lines.append("message: task was cancelled and no result will be returned.")
            return "\n".join(lines)

        async_task_description = dedent(
            """Start an independent agent task on a background event loop and return immediately.
            Use this when the task is likely to take a while and you can continue useful coordination, planning, or inspection while it runs.
            Always check the task with get_task_status, wait, or cancel_task before relying on its result or finishing the user request.
            Args:
            agent_name: The name of the agent to execute the task.
            prompt: A complete, self-contained prompt for the agent.
            Returns:
            A task id and initial status."""
        )

        @tool(description=async_task_description)
        def async_task(agent_name: str, prompt: str) -> str:
            """
            Execute an agent task asynchronously on a background event loop.
            Use when you need to continue working on other things while the task is running.
            Typically, if a task is expected to take more than 15 seconds to complete and you have other work to do, you should use this tool.
            Args:
                agent_name: The name of the agent to execute the task.
                prompt: The prompt to give the agent.
            Returns:
                Task ID."""

            if agent_name not in [agent.name for agent in agents]:
                raise ValueError(
                    f"Agent {agent_name} not found. Use one of the following agents: {', '.join([agent.name for agent in agents])}"
                )
            agent = next(a for a in agents if a.name == agent_name)
            task_id = run_and_track_task(agent, prompt)

            return f"task_id: {task_id}\nstatus: started\nagent: {agent_name}"

        get_task_status_description = dedent(
            """Get the current status and result, if available, for a background async task.
            Use this before relying on async work or before giving a final answer when async tasks may still be running.
            Args:
            task_id: The id returned by async_task.
            Returns:
            The task status, elapsed time, and result or error when available."""
        )

        @tool(description=get_task_status_description)
        def get_task_status(task_id: str) -> str:
            """
            Get the status of a task.
            Args:
                task_id: The id of the task to get the status of.
            Returns:
                The status of the task."""

            with self._async_task_lock:
                record = self._async_tasks.get(task_id)
                if record is None:
                    return f"Task {task_id} not found."
                return _format_task_status(record)

        cancel_task_description = dedent(
            """Cancel a background async task that is no longer needed.
            Use this when the user's request can be completed without that task or when the task should stop before finalizing.
            Args:
            task_id: The id returned by async_task.
            Returns:
            The updated task status."""
        )

        @tool(description=cancel_task_description)
        def cancel_task(task_id: str) -> str:
            """
            Cancel a task.
            Args:
                task_id: The id of the task to cancel.
            Returns:
                A message indicating the status of the task."""

            with self._async_task_lock:
                record = self._async_tasks.get(task_id)
                if record is None:
                    return f"Task {task_id} not found."

                if record.status in {"completed", "failed", "cancelled"}:
                    return _format_task_status(record)

                record.cancel_requested = True
                future = record.future
                if future is None:
                    record.status = "cancelled"
                    record.finished_at = time.monotonic()
                elif future.cancel():
                    record.status = "cancelled"
                    record.finished_at = time.monotonic()
                elif future.done():
                    if record.status not in {"completed", "failed", "cancelled"}:
                        record.status = "cancelled"
                        record.finished_at = time.monotonic()
                else:
                    record.status = "cancelling"

                return _format_task_status(record)

        wait_description = dedent(
            """Wait for a short period before checking background task status again.
            Use this after async_task when the background task needs more time and its result is still required.
            Args:
            seconds: Number of seconds to wait. Minimum 0. Maximum 300.
            Returns:
            A short wait confirmation."""
        )

        @tool(description=wait_description)
        def wait(seconds: float) -> str:
            """
            Wait for a given number of seconds.
            Use this tool when you need to wait for a sub agent to complete its task before continuing.
            Args:
                seconds: The number of seconds to wait. Minimum 15 seconds. Maximum 300 seconds.
            Returns:
                A message indicating the status of the wait."""
            if seconds < 0:
                raise ValueError("seconds must be >= 0.")
            if seconds > 300:
                raise ValueError("seconds must be <= 300.")
            time.sleep(seconds)
            return f"Waited for {seconds} seconds."

        tools: List[BaseTool] = [route_to_agent]
        if self.allow_parallel_agents:
            tools.append(parallel_task)
        if self.allow_async_tasks:
            tools.append(async_task)
            tools.append(get_task_status)
            tools.append(cancel_task)
            tools.append(wait)

        if required_plan:
            current_todos_description = dedent(
                """Get the current todo list. Use this tool to get your todo list for the current task if any.
                Use this tool when working on complex tasks that require a todo list to stay on track.
                Args: None
                Returns: A dictionary of todo items with their status and item."""
            )
            mark_todo_complete_description = dedent(
                """Use this tool to update the status of the todo item when you have completed it.
                Be sure to complete the item before marking as complete.
                Args:
                plan_id: The id of the todo item to mark as complete.
                Returns:
                A message indicating the status of the todo item."""
            )
            new_todo_description = dedent(
                """Record a new todo list. Overwrites existing todo list - beware of this.
                Use get_todos tool to get the current todos if it exists.
                Args:
                List[str]: A list of todo items.
                Returns:
                A message indicating the status of the todo items recorded and their ids."""
            )

            @tool(description=current_todos_description)
            def get_todos() -> Dict[str, Dict[str, str]]:
                self.debug == 1 and log_supervisor("getting current todos")
                return self.plan

            @tool(description=mark_todo_complete_description)
            def mark_todo_complete(plan_id: str) -> str:
                self.debug == 1 and log_supervisor(
                    f"marking todo {plan_id} complete"
                )
                if plan_id not in self.plan:
                    raise ValueError(f"Todo {plan_id} not found. Use get_todos tool to get todo list if it exists.")
                self.plan[plan_id]["status"] = "complete"
                return f"Todo {plan_id} marked as complete."

            @tool(description=new_todo_description)
            def new_todos(todos: List[str]) -> str:
                plan = {random_id(): {"status": "pending", "item": todo} for todo in todos}
                self.plan = self.extract_plan(plan)
                return f"New todos recorded: {self.plan}"

            tools.extend([get_todos, mark_todo_complete, new_todos])

        return tools
        

    def supervise(self, state: dict, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> dict:
        """Run the supervisor and clean up any run-scoped background tasks."""
        scope_id = f"supervisor:{id(self)}:{uuid.uuid4()}"
        scope_token = _ASYNC_TASK_SCOPE.set(scope_id)
        thread_token = _SUPERVISOR_THREAD_ID.set(thread_id)
        metadata_token = _SUPERVISOR_METADATA.set(metadata)
        try:
            return self._supervise_once(state, thread_id=thread_id, metadata=metadata)
        finally:
            self._cleanup_async_tasks(cancel_running=True, scope_id=scope_id)
            _SUPERVISOR_METADATA.reset(metadata_token)
            _SUPERVISOR_THREAD_ID.reset(thread_token)
            _ASYNC_TASK_SCOPE.reset(scope_token)

    def _supervise_once(self, state: dict, thread_id: Optional[str] = None, metadata: Optional[Mapping[str, Any]] = None) -> dict:
        """
        Run the supervisor as a router over the current conversation state.

        The supervisor can answer directly or route work to specialist agents
        using the tools exposed in :meth:`make_supervisor_tools`. When a plan
        is present and ``config.enable_planning`` is True, it will also encourage
        the model to keep the plan updated via plan tools.
        """
        #TODO: Check memory transfer and allow using supervisor

        # Refresh in-memory plan from state.
        self.extract_plan(state)
        if self.debug == 1:
            log_supervisor(f"Todos: {self.plan}" if self.plan else "No defined todos")

        attempts = 0
        while attempts < self.config.validation.skip_after_attempts:
            messages = list(state.get("messages", []))

            # If we have a plan and it's required, remind the supervisor about it.
            if self.plan and self.config.enable_planning:
                messages.extend(
                    [
                        HumanMessage(content=f"Todos:\n{self.plan}"),
                        HumanMessage(
                            content=(
                                "You have been given tools to get and update the todos. "
                                "Use them appropriately to ensure comprehensive coverage "
                                "of the task. Make sure you check and promptly update the "
                                "todo status when steps are completed."
                            )
                        ),
                    ]
                )

            response_messages = self.invoke(
                messages,
                thread_id=self._current_thread_id(),
                metadata=self._current_metadata(),
            )
            response = {
                "messages": response_messages,
                "plan": self.plan,
            }

            todos_incomplete = False
            if self.config.enable_planning and self.plan:
                for item_id in list(self.plan.keys()):
                    if self.plan[item_id]["status"] == "pending":
                        if self.debug == 1:
                            log_supervisor(
                                "Todos are incomplete. Please update the status or complete the todos. Check for completion if required."
                            )
                            log_supervisor(f"Todos: {self.plan}")
                        state.setdefault("messages", []).append(
                            HumanMessage(
                                content=(
                                    "Todos are incomplete. Please update the todo status "
                                    "using the available tools or complete the remaining steps. Check for completion if required."
                                )
                            )
                        )
                        todos_incomplete = True
                        break

            if todos_incomplete:
                attempts += 1
                continue

            async_summary = self._incomplete_async_tasks_summary()
            if self.allow_async_tasks and async_summary is not None:
                if self.debug == 1:
                    log_supervisor(
                        "Background async tasks are still active. Use get_task_status, wait, or cancel_task before finishing."
                    )
                    log_supervisor(async_summary)
                state.setdefault("messages", []).append(
                    HumanMessage(
                        content=(
                            f"{async_summary}\n\n"
                            "Resolve these using get_task_status, wait to allow completion, "
                            "or cancel_task if the work is no longer needed. "
                            "Do not treat the turn as finished while tasks are still pending or running."
                        )
                    )
                )
                attempts += 1
                continue

            return response

        return response
