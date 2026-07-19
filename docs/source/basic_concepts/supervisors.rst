Supervisors
===========

Use ``Supervisor`` when one agent is not enough and you need a coordinator to
route work across multiple specialist agents.

In Andromeda, ``Supervisor`` extends ``Agent`` and adds routing/orchestration
tools automatically (such as ``route_to_agent`` and optional planning tools).

When to Use a Supervisor
------------------------

Use a supervisor when you want:

- a central agent that delegates to specialists
- explicit routing modes (``chat``, ``task``, ``research``, ``handoff``)
- optional plan tracking for multi-step tasks
- optional parallel task delegation

Quick Start
-----------

This example creates two worker agents and one supervisor, then runs one
supervision cycle over state.

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.core.supervisor import Supervisor
   from andromeda.config import AgentConfig, SupervisorConfig, ModelConfig
   from andromeda import HumanMessage

   # Specialist worker configs
   researcher = AgentConfig(
       name="researcher",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       prompt="Focus on evidence gathering and source quality.",
   )
   writer = AgentConfig(
       name="writer",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       prompt="Produce clear, concise writing from verified notes.",
   )

   # Supervisor config
   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       prompt="Plan and coordinate agents to fully cover the task.",
       enable_planning=True,
   )

   supervisor = Supervisor(agents=[researcher, writer], config=supervisor_cfg)

   state = {
       "messages": [HumanMessage(content="Research EV trends and draft a short brief.")],
       "plan": [],
   }
   result = supervisor.supervise(state)
   print(result["messages"][-1].content)

How Supervision Works
---------------------

``supervisor.supervise(state)`` does the following:

1. Reads the current conversation from ``state["messages"]``.
2. Optionally reads/updates ``state["plan"]`` when planning is enabled.
3. Decides whether to respond directly or route work to specialist agents.
4. Returns updated state, including ``messages`` and current ``plan``.

Required state keys in practice:

- ``messages``: list of chat messages
- ``plan``: optional plan data (list or dict)

Routing Modes
-------------

Supervisor routing behavior is controlled by ``allowed_route_types`` in
``SupervisorConfig``.

Supported route types:

- ``chat``
- ``task``
- ``research``
- ``handoff``

You can restrict routing modes if you want tighter behavior.

.. code-block:: python

   from andromeda.config import SupervisorConfig, ModelConfig

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       allowed_route_types=["task", "research"],
   )

Planning Controls
-----------------

Planning is enabled by default for supervisors. You can disable it for lighter,
single-pass routing behavior.

.. code-block:: python

   from andromeda.config import SupervisorConfig, ModelConfig

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       enable_planning=False,
   )

With ``enable_planning=True``, supervisor exposes additional planning tools
internally (for example, reading and updating todo status).

Parallel Delegation
-------------------

Enable ``allow_parallel_agents`` when you want the supervisor to run multiple
worker tasks concurrently.

.. code-block:: python

   from andromeda.config import SupervisorConfig, ModelConfig

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       allow_parallel_agents=True,
   )

This adds parallel routing capability through supervisor tools.

Background Delegation
---------------------

Enable ``allow_async_tasks`` when long-running worker tasks should run in the
background while the supervisor continues other work.

.. code-block:: python

   from andromeda.config import SupervisorConfig, ModelConfig

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       allow_async_tasks=True,
   )

This adds synchronous tools for starting a background task, checking
task status, requesting cancellation, and waiting briefly. Background tasks are
scheduled on an internal asyncio loop so cancellation is propagated to the
running agent coroutine. As with any asyncio cancellation, blocking synchronous
work inside that coroutine can only stop once control returns to the event loop.

Passing Thread Context
----------------------

Use ``thread_id`` and ``metadata`` when supervising sessions that need traceable
run context.

.. code-block:: python

   result = supervisor.supervise(
       state,
       thread_id="session-42",
       metadata={"tenant": "acme", "request_id": "req-1001"},
   )

Supervisor with Existing Agent Objects
--------------------------------------

You can pass either ``AgentConfig`` entries or fully initialized ``Agent``
instances.

.. code-block:: python

   worker_agent = Agent(
       AgentConfig(
           name="researcher",
           model=ModelConfig(name="qwen3:8b", provider="litellm"),
       )
   )

   supervisor = Supervisor(
       agents=[worker_agent],
       config=SupervisorConfig(
           name="supervisor",
           model=ModelConfig(name="qwen3:8b", provider="litellm"),
       ),
   )

Relationship to Team
--------------------

Use ``Supervisor`` directly when you want explicit control over state and
routing.

Use ``Team`` when you want a higher-level orchestration wrapper that combines
planner + supervisor + agents (+ optional reporting).
