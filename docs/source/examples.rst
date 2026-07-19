Examples
========

This page provides end-to-end examples, starting from the most basic building blocks and progressing to advanced multi-agent patterns. Every example uses the real Andromeda APIs and can be adapted to your own projects.

1. Chat Models and Embeddings
-----------------------------

Use ``get_chat_model`` and ``get_embedding_model`` when you want direct access to LangChain models without agents or teams.

**Basic chat model**

.. code-block:: python

   from andromeda.utils import get_chat_model, get_embedding_model
   from andromeda.config import ModelConfig
   from andromeda import HumanMessage

   # Chat model (LangChain BaseChatModel)
   chat_model = get_chat_model(
       ModelConfig(
           name="gpt-oss:20b",
           provider="litellm",
           other_args={
            "base_url": "http://localhost:11434", #optional
            "temperature": 0.3 #optional
            },
       )
   )

   messages = [HumanMessage(content="Say hello succinctly.")]
   chat_response = chat_model.invoke(messages)
   print(chat_response)

**Embedding model**

.. code-block:: python

   # Embedding model (provider-specific LangChain embeddings)
   embedding_model = get_embedding_model(
       ModelConfig(
           name="nomic-embed-text:latest",
           provider="litellm",
       )
   )

   vector = embedding_model.embed_query("Andromeda makes workflows easy.")
   print(len(vector), "dimensions")

What these return and how to use them:

- ``get_chat_model``: returns a LangChain ``BaseChatModel``. Call ``invoke([...messages])`` (sync) or use LangChain streaming APIs.
- ``get_embedding_model``: returns a provider-specific embeddings object. Use ``embed_query(text)`` or ``embed_documents(list[str])``.

2. Workflow Examples
--------------------

Workflows are the foundation for deterministic control flow and orchestration. They execute over a shared state dict and can be run synchronously, asynchronously, or in streaming mode.

2.1 Simple Linear Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder
   from andromeda import HumanMessage
   from andromeda.core import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from typing import TypedDict

   class StateModel(TypedDict):
       messages: list[BaseMessage] # is mandatory field for agents
       query: str
       raw: list[str]
       summary: str

   def ingest(state: Dict[str, Any]) -> Dict[str, Any]:
       query = state.get("query", "")
       return {"raw": [f"Result for {query} from web", f"Result for {query} from docs"]}

   def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
       agent = Agent(AgentConfig(
           name="summarizer",
           model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
           prompt="Summarize the following text:",
       ))
       messages = [HumanMessage(content=state["raw"])]
       result = agent.invoke(messages)
       text = result[-1].content
       return {"summary": text, "messages": result}

   workflow = WorkflowBuilder(name="SimpleLinearWorkflow", state_schema=StateModel)
   (
       workflow
       .start("ingest").run(ingest)
       .finish("summarize").run(summarize)
   )

   result = workflow.execute(state={"query": "Andromeda features"})
   print("Summary:", result["summary"])

2.2 Conditional Routing and Error Handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder

   def risky_step(state: Dict[str, Any]) -> Dict[str, Any]:
       if state.get("should_fail"):
           raise RuntimeError("Simulated failure")
       return {"status": "ok"}

   def on_failure(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"status": "failed", "handled": True}

   def on_success(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"status": "completed"}

   wf = WorkflowBuilder(name="RoutingExample", state_schema=StateModel)
   (
       wf
       .start("do_risky_thing")
           .run(risky_step)
           .if_fails().goto("failure_handler")
           .if_succeeds().goto("success_handler")
       .then("failure_handler")
           .run(on_failure)
       .then("success_handler")
           .run(on_success)
   )

   final_state = wf.execute(state={"should_fail": True})
   print(final_state["status"], final_state.get("handled"))

2.3 Parallel Branches
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder

   def compute_sum(state: Dict[str, Any]) -> Dict[str, Any]:
       numbers = state.get("numbers", [])
       return {"sum": sum(numbers)}

   def compute_count(state: Dict[str, Any]) -> Dict[str, Any]:
       numbers = state.get("numbers", [])
       return {"count": len(numbers)}

   wf = WorkflowBuilder(name="ParallelExample", state_schema=StateModel)
   (
       wf
       .start("prepare")
           .run(lambda s: {"numbers": [1, 2, 3, 4, 5]})
       .branch("analytics")
           .parallel([
               ("sum_step", compute_sum),
               ("count_step", compute_count),
           ])
           .merge_results()
       .finish("done")
           .run(lambda s: s)
   )

   out = wf.execute(state={})
   print("Sum:", out["sum"], "Count:", out["count"])

2.4 Operator Workflows
^^^^^^^^^^^^^^^^^^^^^^

Operator workflows focus on transforming a single "current value" and work well for functional composition:

.. code-block:: python

   from typing import List
   from andromeda.core.workflow import WorkflowBuilder, task, conditional, parallel

   def fetch_articles(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"articles": [f"Article about {state['query']} - A", f"Article about {state['query']} - B"]}

   def filter_relevant(state: Dict[str, Any]) -> Dict[str, Any]:
       articles = state["articles"]
       return [a for a in articles if "Article" in a]

   def summarize_list(state: Dict[str, Any]) -> Dict[str, Any]:
       articles = state["articles"]
       return {"summary": " | ".join(articles)}

   flow = (
       fetch_articles
       >> filter_relevant
       >> conditional(
           true_branch=summarize_list,
           false_branch=lambda state: {"summary": "No articles found."},
           condition=lambda state: len(state["articles"]) > 0,
       )
   )

   builder = WorkflowBuilder.from_expression(flow, name="ArticlePipeline")
   summary = builder.execute(state={"query": "Andromeda workflows"})
   print(summary)

2.5 Streaming Workflow Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder

   def step_one(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"a": 1}

   def step_two(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"b": state["a"] + 1}

   workflow = WorkflowBuilder(name="StreamingExample")
   (
       workflow
       .start("one").run(step_one)
       .finish("two").run(step_two)
   )

   for chunk in workflow.stream(state={}):
       print("Chunk:", chunk)

2.6 Workflow Evaluation and Langfuse Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Attach evaluators to workflow nodes to ingest Langfuse scores (non-blocking, background).
Scores are automatically scoped by node name (for example ``refine:correctness``).

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, correctness, hallucination, relevance
   from andromeda import HumanMessage

   wf = WorkflowBuilder(name="ScoringExample")
   (
       wf.start("generate")
         .run(lambda s: s)
         .with_evaluators(
             [correctness(), hallucination(), relevance()],
             scheduler={"max_workers": 2, "max_pending": 128},
             model={"name": "gpt-oss:20b", "provider": "litellm", "temperature": 0.0, "other_args": {"base_url": "http://localhost:11434" #optional}},
         )
   )

   out = wf.execute(state={"messages": [HumanMessage(content="Write a haiku about databases.")]})

Custom evaluators can also read arbitrary keys from the workflow state. Use the ``raw_payload``
preset to pass the full state dict into ``inp.input`` / ``inp.output`` (or read from
``inp.before`` / ``inp.after`` directly).

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, LangfuseEvaluator

   def has_answer(inp):
       # Because output_preset="raw_payload", inp.output is the full state dict after the step.
       answer = (inp.output or {}).get("answer")
       ok = bool(answer and str(answer).strip())
       return (1.0 if ok else 0.0, "answer present" if ok else "missing/empty answer")

   evaluator = LangfuseEvaluator(
       name="has_answer",
       eval_fn=has_answer,
       data_type="NUMERIC",
       input_preset="raw_payload",
       output_preset="raw_payload",
   )

   wf = WorkflowBuilder(name="CustomEvaluatorExample")
   (
       wf.start("generate")
         .run(lambda s: {"answer": "hello"})
         .with_evaluators([evaluator])
   )
   wf.execute(state={})

See also the runnable example script in ``andromeda/examples/ollama_workflow_scoring.py``.

3. Agent Examples
-----------------

Agents wrap LangGraph ReAct/CodeAct agents with a clean API for messages, memory, tools, and streaming.

3.1 Basic Agent
^^^^^^^^^^^^^^^

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   agent_config = AgentConfig(
       name="assistant",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       prompt="You are a helpful assistant. Keep answers brief.",
   )

   agent = Agent(agent_config)
   msgs = agent.invoke([HumanMessage(content="Summarize Andromeda in one line.")])
   print(msgs[-1].content)

3.2 Agent with Tools
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Dict, Any
   from andromeda.tools import tool
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   @tool
   def echo_tool(text: str) -> Dict[str, Any]:
       """Echo text back; stand-in for a real tool."""
       return {"echo": text}

   cfg = AgentConfig(
       name="tool_agent",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       tools=[echo_tool],
       prompt="Use tools when helpful.",
   )

   agent = Agent(cfg)
   messages = [HumanMessage(content="Use the echo tool to repeat 'Andromeda'.")]
   result = agent.invoke(messages)
   print(result[-1].content)

3.3 Streaming Agent Responses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import asyncio
   from typing import Dict, Any
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   async def stream_agent():
       agent = Agent(
           AgentConfig(
               name="streamer",
               model=ModelConfig(name="qwen3:8b", provider="litellm"),
           )
       )
       history = [HumanMessage(content="Explain Andromeda in a streaming fashion.")]

       async for event in agent.astream(history, stream_mode="events"):
           print(event)

       print()

   asyncio.run(stream_agent())

4. Supervisor Examples
----------------------

The ``Supervisor`` coordinates multiple agents, optionally with a plan of action and planning tools.

4.1 Simple Supervisor with Two Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from andromeda.core.supervisor import Supervisor
   from andromeda.config import AgentConfig, SupervisorConfig, ModelConfig
   from andromeda import HumanMessage

   researcher_cfg = AgentConfig(
       name="researcher",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   writer_cfg = AgentConfig(
       name="writer",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
       prompt="Coordinate agents to research and then write a short brief.",
       enable_planning=True,
   )

   supervisor = Supervisor(agents=[researcher_cfg, writer_cfg], config=supervisor_cfg)

   state = {
       "messages": [HumanMessage(content="Research EV trends and draft a short brief.")],
       "plan": ["Gather EV market data", "Draft a 200-word brief"],
   }

   result = supervisor.supervise(state)
   print("Plan:", result["plan"])
   print("Final message:", result["messages"][-1].content)

5. Team Examples
----------------

``Team`` combines planner, supervisor, agents, and optional reporting into a single orchestrator for long-horizon tasks.

5.1 Minimal Team
^^^^^^^^^^^^^^^^

.. code-block:: python

   from andromeda.config import AndromedaConfig, AgentConfig, SupervisorConfig, ModelConfig
   from andromeda.core.team import Team

   research_agent = AgentConfig(
       name="researcher",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   supervisor = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
       prompt="Plan, route, and ensure coverage for the task.",
   )

   config = AndromedaConfig(
       agents={"researcher": research_agent},
       supervisor=supervisor,
   )

   team = Team(config)
   result = team.begin("Analyze the impact of AI on logistics in 2025.")
   print(result["messages"][-1].content)

5.2 Team with Planner and Reporting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from andromeda.config import (
       AndromedaConfig,
       AgentConfig,
       SupervisorConfig,
       PlannerConfig,
       ReportConfig,
       ModelConfig,
       ValidationConfig,
       CitationConfig,
   )
   from andromeda.core.team import Team

   base_model = ModelConfig(name="llama3.1:8b", provider="litellm")

   researcher = AgentConfig(
       name="researcher",
       model=base_model,
   )

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=base_model,
       prompt="Coordinate agents and keep the plan up to date.",
       enable_planning=True,
   )

   planner_cfg = PlannerConfig(
       model=base_model,
       task_type="research",
       report_structure="Introduction, Current Landscape, Key Players, Risks, Conclusion",
   )

   report_cfg = ReportConfig(
       enabled=True,
       model=base_model,
       format="markdown",
       citations=CitationConfig(required=True, min_density=0.2),
       validation=ValidationConfig(
           enabled=True,
           min_sufficiency_score=0.8,
           model=base_model,
       ),
       output_mode="state",
   )

   cfg = AndromedaConfig(
       agents={"researcher": researcher},
       supervisor=supervisor_cfg,
       planner=planner_cfg,
       report=report_cfg,
   )

   team = Team(cfg)
   state = team.begin("Deep dive on AI chip shortages and their impact on cloud providers.")
   print("Report:\n", state.get("report_output"))

6. Configuration via YAML
-------------------------

Define your agents, supervisor, planner, and report in a YAML file and load it with ``AndromedaConfig.load_from_file`` for clean separation of config and code. Tools can be referenced by name if they're registered in the global tool registry.

**config.yaml**

.. code-block:: yaml

   agents:
     researcher:
       name: researcher
       model:
         name: llama3.1:8b
         provider: litellm
         other_args:
           base_url: http://localhost:11434 #optional
       tools:
         - web_search          # Built-in tools referenced by name
         - news_search
   supervisor:
     name: supervisor
     model:
       name: llama3.1:8b
       provider: litellm
     enable_planning: true
   planner:
     model:
       name: llama3.1:8b
       provider: litellm
     task_type: research
   report:
     enabled: true
     model:
       name: llama3.1:8b
       provider: litellm
     format: markdown
     output_mode: state

**Loading and running**

When loading from YAML, built-in tools are automatically registered. String tool names in the config are resolved to actual tool instances:

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.core.team import Team

   cfg = AndromedaConfig.load_from_file("config.yaml")
   # Tools specified as strings in YAML are now resolved to BaseTool instances
   team = Team(cfg)
   result = team.begin("Summarize the competitive landscape for open-source LLM orchestration tools.")
   print(result.get("report_output") or result["messages"][-1].content)

**Using custom tools in YAML:**

If you have custom tools, register them before loading the config:

.. code-block:: python

   from andromeda.tools import tool
   from andromeda.tools.toolkit import register_tool
   from andromeda.config import AndromedaConfig

   @tool
   def my_custom_tool(query: str) -> str:
       """Custom tool for specific processing."""
       return f"Custom processing: {query}"

   # Register before loading config
   register_tool(my_custom_tool)

   # Now use it in config.yaml:
   # tools:
   #   - my_custom_tool

   cfg = AndromedaConfig.load_from_file("config.yaml")

   # If not using config.yaml, this step is not needed

7. Advanced Patterns
--------------------

7.1 Structured Responses
^^^^^^^^^^^^^^^^^^^^^^^^

Use ``response_format`` on ``AgentConfig`` to request structured output (e.g., Pydantic model):

.. code-block:: python

   from typing import Optional
   from pydantic import BaseModel
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   class ResearchSummary(BaseModel):
       topic: str
       key_points: list[str]
       risk_rating: Optional[str]

   cfg = AgentConfig(
       name="structured_agent",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       response_format=ResearchSummary,
       prompt="Return a JSON-compatible summary following the schema.",
   )

   agent = Agent(cfg)
   messages = [HumanMessage(content="Summarize risks of AI chip shortages for cloud providers.")]
   structured = agent.invoke(messages)
   print(structured)

   # using chat model directly
   chat_model = get_chat_model(ModelConfig(name="qwen3:8b", provider="litellm"))
   response = chat_model.with_structured_output(ResearchSummary).invoke("Summarize risks of AI chip shortages for cloud providers.")
   print(response)

7.2 State Schema and Optional Middleware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``state_schema`` when tools/middleware need extra state keys. Middleware activation is inferred
from configured middleware blocks (or can be forced off with ``enabled=False``).

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentState, AgentConfig, ModelConfig, MiddlewareConfig

   class CustomAgentState(AgentState):
       user_id: str

   cfg = AgentConfig(
       name="stateful_agent",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       state_schema=CustomAgentState,
       middleware=MiddlewareConfig(
           tool_error_handler=True,
           summarization=MiddlewareConfig.SummarizationOptions(trigger_tokens=1200),
           guardrails=MiddlewareConfig.GuardrailOptions(input=True, output=True, tool=False),
           masking=MiddlewareConfig.MaskingOptions(
               output=True,
               strategy="tokenize",
               token_prefix="pii",
               token_ttl_seconds=3600,
           ),
       ),
   )
   agent = Agent(cfg)

You can also configure these from YAML:

.. code-block:: yaml

   agents:
     worker:
       name: worker
       model:
         name: qwen3:8b
         provider: litellm
       state_schema: my_project.schemas.CustomAgentState
       middleware:
         tool_error_handler: true
         summarization:
           trigger_tokens: 1200
         guardrails:
           input: true
           output: true
           tool: false
         masking:
           output: true
           strategy: tokenize
           token_prefix: pii
           token_ttl_seconds: 3600
         custom:
           - my_project.middleware.custom_tool_wrapper

For ``strategy: tokenize``, set one of:

.. code-block:: bash

   export ANDROMEDA_ENCRYPTION_KEY="$(python - <<'PY'
   from cryptography.fernet import Fernet
   print(Fernet.generate_key().decode())
   PY
   )"
   # or
   export ANDROMEDA_ENCRYPTION_SECRET="$(python - <<'PY'
   import secrets
   print(secrets.token_urlsafe(48))
   PY
   )"

You can also use the ``detokenize_value`` function to detokenize the values:

.. code-block:: python
    
   import re
   from andromeda.config import ModelConfig, AgentConfig, MiddlewareConfig, DataPatternsConfig
   from andromeda.core.agent import Agent
   from andromeda import HumanMessage
   from andromeda.utils.secure_store import detokenize_value
   import os

   from cryptography.fernet import Fernet
   os.environ["ANDROMEDA_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
   os.environ["ANDROMEDA_ENCRYPTION_SECRET"] = Fernet.generate_key().decode("utf-8")

   model_config = ModelConfig(
       name="qwen3:8b",
       provider="litellm",
       other_args={
           "base_url": "http://localhost:11434",
           "temperature": 0.2,
       },
   )

   agent_config = AgentConfig(
       name="assistant",
       model=model_config,
       debug=3,
       middleware=MiddlewareConfig(
           tool_error_handler=True,
           masking=MiddlewareConfig.MaskingOptions(
               output=True,
               input=True,
               strategy="tokenize",
               token_prefix="masked",
               token_ttl_seconds=3600,
               data_patterns=DataPatternsConfig(),
           ),
       ),
   )

   agent = Agent(agent_config)
   messages = agent.invoke([HumanMessage(content="My name is John Doe and my email is john.doe@example.com. My ssn is 123-45-6789. Repeat it.")])
   response = messages[-1].content
   print(response)
   mask_pattern = r"masked_\w+" 
   # find all masked tokens
   masked_tokens = re.findall(mask_pattern, response)
   print("masked_tokens: ", masked_tokens)
   for masked_token in masked_tokens:
       print("masked_token: ", masked_token)
       print("detokenized_value: ", detokenize_value(masked_token))

7.3 Combining Workflows and Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can embed agents inside workflows for precise orchestration over agent calls:

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda.core.workflow import WorkflowBuilder, HumanMessage

   analysis_agent = Agent(
       AgentConfig(
           name="analysis",
           model=ModelConfig(name="llama3.1:8b", provider="litellm"),
           prompt="Analyze the given text and list three key risks.",
       )
   )
   

   def analyze_step(state: Dict[str, Any]) -> Dict[str, Any]:
       text = state.get("text", "")
       msgs = analysis_agent.invoke([HumanMessage(content=text)])
       return {"analysis": msgs[-1].content}

   workflow = WorkflowBuilder(name="AgentInWorkflow")
   (
       workflow
       .start("analyze").run(analyze_step)
       .finish("postprocess").run(lambda s: {"report": s["analysis"]})
   )

   result = workflow.execute(state={"text": "Risk analysis for a new AI product launch."})
   print(result["report"])

7.4 Using MCP Servers
^^^^^^^^^^^^^^^^^^^^^

You can connect to external Model Context Protocol (MCP) servers as tool providers using the 
``mcp_servers`` section in your configuration and referencing their tools by name in agents 
or supervisors. Here's a demonstration of using an MCP tool within an agent, and inside a workflow:

**1. Using MCP tools in an agent**

.. code-block:: python

   from andromeda.config import AgentConfig, ModelConfig, AndromedaConfig
   from andromeda.core.agent import Agent
   from andromeda import HumanMessage

   # Assume config.yaml contains an mcp_servers section as shown in the config docs.
   cfg = AndromedaConfig.load_from_file("config.yaml")

   # The tools (like "github.get_issue") become available by name automatically.
   agent_cfg = AgentConfig(
       name="github_research",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       tools=["github.get_issue"],  # Refers to MCP tool registered from config
       prompt="Fetch details for a GitHub issue using the provided repo and number.",
   )

   agent = Agent(agent_cfg)
   msgs = [
       HumanMessage(content="Get GitHub issue for langchain-ai/langgraph, #123")
   ]
   result = agent.invoke(msgs)
   print(result[-1].content)

**2. Invoking an MCP tool directly in a workflow**

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage
   from andromeda.core.agent import Agent

   def fetch_github_issue(state: Dict[str, Any]) -> Dict[str, Any]:
   mcp_cfg = {
        "github": {"transport": "http", "url": "https://api.githubcopilot.com/mcp/", "authorization": "Bearer <token>"},
        "langgraph-docs-mcp": {"transport": "stdio", "command": "uvx", "args": ["--from", "mcpdoc", "mcpdoc", "--urls", "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt LangChain:https://python.langchain.com/llms.txt", "--transport", "stdio"]}
    }
    # this option keeps the server open for the duration of the workflow to reduce latency
    # tools = register_mcp(mcp_cfg) can be used to get the list of tools
    # which opens server connections for each tool call
    async with open_mcp_sessions(mcp_cfg) as sessions:
        tools = await load_from_sessions(sessions)
        agent = Agent(
            AgentConfig(
                name="github_tool_agent",
                model=ModelConfig(name="qwen3:8b", provider="ollama"),
                tools=tools
            )
        )
        repo = state.get("repo", "langchain-ai/langgraph")
        number = state.get("issue_number", 123)
        message = f"Get GitHub issue for {repo}, #{number}"
        msgs = [HumanMessage(content=message)]
        result = agent.invoke(msgs)
        return {"issue_result": result[-1].content}

   workflow = WorkflowBuilder(name="MCPWorkflowExample")
   (
       workflow
       .start("fetch_github_issue").run(fetch_github_issue)
       .finish("report_issue").run(lambda s: {"report": s["issue_result"]})
   )

   output = workflow.execute(state={"repo": "langchain-ai/langgraph", "issue_number": 123})
   print(output["report"])

**3. Referencing MCP tools in agent configuration**

.. code-block:: yaml

   agents:
     - name: github_agent
       model:
         name: "qwen3:8b"
         provider: "ollama"
       tools:
         - github.get_issue
         - langgraph-docs-mcp.list_doc_sources
    mcp_servers:
      - github:
          transport: http
          url: https://api.githubcopilot.com/mcp/
          authorization: Bearer <token>
      - langgraph-docs-mcp:
          transport: stdio
          command: uvx
          args: ["--from", "mcpdoc", "mcpdoc", "--urls", "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt LangChain:https://python.langchain.com/llms.txt", "--transport", "stdio"]

MCP tools behave exactly like built-in Andromeda/LangChain tools and can be called via agent ``invoke`` or inside workflows and teams.

8. Workspace Agent Examples
---------------------------

A ``WorkspaceAgent`` is a long-horizon supervisor bound to an isolated workspace session. It can read and edit files, run shell commands, and iterate on a plan until a task is done. See :doc:`basic_concepts/workspace-agents` for the full guide.

8.1 Zero-config Workspace Agent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Omit the session and let the agent auto-create (and clean up) its own sandbox. Used as a context manager, the workspace is released on exit.

.. code-block:: python

   from andromeda.core import WorkspaceAgent
   from andromeda.config.config import WorkspaceAgentConfig
   from andromeda.config import ModelConfig

   config = WorkspaceAgentConfig(
       name="builder_agent",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   with WorkspaceAgent(config) as agent:
       report = agent.run(
           "The workspace is empty. Create a CLI tool `wc_tool.py` that prints the "
           "line, word, and character counts of a file, add a unittest suite under "
           "`tests/`, run `python3 -m unittest discover -v`, and make sure it passes. "
           "Report the files you created and the final test output."
       )
       print(report)

8.2 Seeded Workspace with a Bring-Your-Own Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build a session yourself to control the backend, seeding, and policy. The caller owns the session lifecycle.

.. code-block:: python

   from andromeda.core import WorkspaceAgent
   from andromeda.config.config import WorkspaceAgentConfig
   from andromeda.config import ModelConfig
   from andromeda.workspace import DirectorySeed, WorkspacePolicy, WorkspaceSession

   session = WorkspaceSession.create(
       backend="bubblewrap_process",
       seed=DirectorySeed(source_dir="./my_project"),
       policy=WorkspacePolicy(read_only=False, enable_shell=True),
   )

   config = WorkspaceAgentConfig(
       name="fixer_agent",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   agent = WorkspaceAgent(config, session=session)
   try:
       report = agent.run(
           "Running `python3 -m unittest discover -v` fails. Investigate, fix the "
           "bugs in the source (do not weaken the tests), and iterate until the whole "
           "suite passes. Report each bug and paste the final test output."
       )
       print(report)
   finally:
       session.cleanup()  # you own the session you created

See :doc:`basic_concepts/workspace-agents` for the full workspace-agent guide,
including backend selection, the coworker team, and session lifecycle.

These examples should give you a complete tour of Andromeda’s capabilities—from direct model access and workflows to rich multi-agent orchestration and configuration management.
