Quickstart
==========

This page shows the **minimal steps** to get started with Andromeda, from a single chat model up to a multi-agent team. It assumes you have Python 3.11+ and the package installed.

Install and Configure
---------------------

Install Andromeda from source or via requirements:

.. code-block:: bash

   pip install git+https://github.com/AI-Emerging-Tech/ET-Agentify.git

Set any provider-specific environment variables you need (for example, for Tavily or OpenAI). For local models like Ollama, ensure the service is running and the models are pulled.

Chat Models
-----------

The fastest way to start is with `get_chat_model`, which returns a LangChain ``BaseChatModel``:

.. code-block:: python

   from andromeda.utils import get_chat_model
   from andromeda.config import ModelConfig
   from andromeda import HumanMessage

   chat_model = get_chat_model(
       ModelConfig(
           name="llama3.1:8b",
           provider="litellm",
           other_args={"base_url": "http://localhost:11434", "temperature": 0.2},
       )
   )

   response = chat_model.invoke([HumanMessage(content="One-sentence summary of Andromeda.")])
   print(response)

Workflow Builder
----------------

Use ``WorkflowBuilder`` when you want to compose multiple Python functions into a robust, resumable pipeline over shared state:

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder

   class StateModel(TypedDict):
       # messages: list[BaseMessage] is mandatory field for agents
       # but we don't use an agent in this example
       query: str
       raw: list[str]
       summary: str

   def ingest(state: Dict[str, Any]) -> Dict[str, Any]:
       query = state.get("query", "")
       return {"raw": [f"Result for {query}"]}

   def summarize(state: Dict[str, Any]) -> Dict[str, Any]:
       text = " | ".join(state.get("raw", []))
       return {"summary": text}

   workflow = WorkflowBuilder(name="QuickWorkflow", state_schema=StateModel)
   (
       workflow
       .start("ingest").run(ingest)
       .finish("summarize").run(summarize)
   )

   result = workflow.execute(state={"query": "Andromeda overview"})
   print(result["summary"])

Agents
------

``Agent`` wraps a LangGraph ReAct or CodeAct agent with a simple API. It uses the same ``ModelConfig`` and supports tools, recursion limits, and memory:

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   agent = Agent(
       AgentConfig(
           name="assistant",
           model=ModelConfig(name="llama3.1:8b", provider="litellm"),
           prompt="You are a concise assistant.",
       )
   )

   messages = agent.invoke([HumanMessage(content="What is Andromeda?")])
   print(messages[-1].content)


Supervisor and Team
-------------------

Use a ``Supervisor`` to coordinate multiple agents, and a ``Team`` to bundle planner, supervisor, agents, and (optionally) reporting into a single orchestrator:

.. code-block:: python

   from andromeda.core.supervisor import Supervisor
   from andromeda.core.team import Team
   from andromeda.config import AgentConfig, SupervisorConfig, AndromedaConfig, ModelConfig
   from andromeda import HumanMessage

   # Worker agents
   researcher = AgentConfig(
       name="researcher",
       model=ModelConfig(name="llama3.1:8b", provider="litellm"),
   )
   writer = AgentConfig(
       name="writer",
       model=ModelConfig(name="llama3.1:8b", provider="litellm"),
   )

   # Supervisor for routing and planning
   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="llama3.1:8b", provider="litellm"),
       prompt="Plan and coordinate agents to fully cover the task.",
   )

   # Team configuration
   config = AndromedaConfig(
       agents={"researcher": researcher, "writer": writer},
       supervisor=supervisor_cfg,
   )

   team = Team(config)
   result_state = team.begin("Research EV trends and draft a short brief.")
   print(result_state["messages"][-1].content)

Loading from config.yaml
------------------------

For reusable deployments, define agents and supervisor in YAML and load them via ``AndromedaConfig.load_from_file``:

.. code-block:: yaml

   # config.yaml
   agents:
     researcher:
       name: researcher
       model:
         name: llama3.1:8b
         provider: litellm
       tools:
         - web_search
         - news_search
   supervisor:
     name: supervisor
     model:
       name: llama3.1:8b
       provider: litellm
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
         token_ttl_seconds: 86400

You can also reference custom objects via dotted paths:

.. code-block:: yaml

   agents:
     researcher:
       name: researcher
       model:
         name: llama3.1:8b
         provider: litellm
       state_schema: my_project.schemas.CustomAgentState
       middleware:
         tool_error_handler: true
         custom:
           - my_project.middleware.my_custom_tool_wrapper

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.core.team import Team

   config = AndromedaConfig.load_from_file("config.yaml")
   team = Team(config)
   result = team.begin("Summarize current AI tooling trends.")
   print(result["messages"][-1].content)



Langfuse Integration
--------------------

Andromeda can be integrated with Langfuse to track the execution of workflows.

This requires an additional dependency:

.. code-block:: bash

   pip install langfuse

Basic Setup
~~~~~~~~~~~

Set environment variables for Langfuse:

.. code-block:: bash

   export LANGFUSE_SECRET_KEY=your_secret_key
   export LANGFUSE_PUBLIC_KEY=your_public_key
   export LANGFUSE_HOST=your_host

Andromeda automatically integrates with Langfuse and traces the execution of workflows from their root.

Advanced Setup
~~~~~~~~~~~~~~

For detailed tracing and evaluation/scoring options, see :doc:`integrations/langfuse`
