Streaming Patterns
==================

Streaming lets you consume results incrementally instead of waiting for a full final response.
This is useful for:

- real-time UI updates
- progress reporting for long workflows
- reacting to tool calls as they happen
- reducing perceived latency

In Andromeda, you can stream at two layers:

- ``WorkflowBuilder`` / workflow layer
- ``Agent`` layer

When to use which
-----------------

- Use workflow streaming when you want state-level progress across workflow steps.
- Use agent streaming when you want message/model/tool event streams from an agent run.
- Use ``astream_structured_events`` when you want a normalized, UI-friendly event format.

Workflow Streaming
------------------

Workflow sync streaming (``stream``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from typing import Any, Dict
   from andromeda.core.workflow import WorkflowBuilder

   def step_one(state: Dict[str, Any]) -> Dict[str, Any]:
       return state | {"messages": state.get("messages", []) + ["first"]}

   def step_two(state: Dict[str, Any]) -> Dict[str, Any]:
       return state | {"messages": state.get("messages", []) + ["second"]}

   builder = WorkflowBuilder(name="stream_values")
   (
       builder.start("first").run(step_one)
       .finish("second").run(step_two)
   )

   for chunk in builder.stream(state={"messages": []}):
       print(chunk)

   # Typical values-mode chunks:
   # {"messages": []}
   # {"messages": ["first"]}
   # {"messages": ["first", "second"]}

Workflow async streaming (``astream``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import asyncio
   from typing import Any, Dict
   from andromeda.core.workflow import WorkflowBuilder

   def step_one(state: Dict[str, Any]) -> Dict[str, Any]:
       return state | {"messages": state.get("messages", []) + ["first"]}

   def step_two(state: Dict[str, Any]) -> Dict[str, Any]:
       return state | {"messages": state.get("messages", []) + ["second"]}

   async def main() -> None:
       builder = WorkflowBuilder(name="astream_values")
       (
           builder.start("first").run(step_one)
           .finish("second").run(step_two)
       )

       async for chunk in builder.astream(state={"messages": []}):
           print(chunk)

   asyncio.run(main())

Resume with checkpoints
^^^^^^^^^^^^^^^^^^^^^^^

You can resume a paused workflow stream using a ``Command`` and ``thread_id``.

.. code-block:: python

   from andromeda.core.workflow import Command, WorkflowBase

   initial = WorkflowBase.run(builder, state={"messages": []})
   # ... approval or external signal occurs ...
   resumed = builder.stream(
       resume=Command(goto="second"),
       thread_id=initial.context.thread_id,
   )
   for chunk in resumed:
       print(chunk)

Agent Streaming
---------------

Agent sync streaming (``stream``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Agent.stream`` supports ``stream_mode`` values:

- ``values``
- ``updates``
- ``messages``

.. code-block:: python

   from andromeda import HumanMessage
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig

   agent = Agent(
       AgentConfig(
           name="streamer_sync",
           model=ModelConfig(name="qwen3:8b", provider="litellm"),
       )
   )

   history = [HumanMessage(content="Explain streaming in simple terms.")]

   for chunk in agent.stream(history, stream_mode="values", remember="all"):
       print(chunk)

Agent async streaming (``astream``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Agent.astream`` supports:

- ``values``
- ``updates``
- ``messages``
- ``events``
- ``tasks``
- ``checkpoints``

.. code-block:: python

   import asyncio
   from andromeda import HumanMessage
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig

   async def main() -> None:
       agent = Agent(
           AgentConfig(
               name="streamer_async",
               model=ModelConfig(name="qwen3:8b", provider="litellm"),
           )
       )

       history = [HumanMessage(content="Explain Andromeda in streaming mode.")]

       async for chunk in agent.astream(history, stream_mode="values", remember="last"):
           print(chunk)

   asyncio.run(main())

Raw Event Streaming
-------------------

Use ``stream_mode=\"events\"`` with ``agent.astream(...)`` when you need low-level model/tool lifecycle events.

.. code-block:: python

   async for event in agent.astream(history, stream_mode="events"):
       etype = event.get("event")
       if etype == "on_chat_model_stream":
           print("model chunk", event.get("data", {}))
       elif etype == "on_tool_start":
           print("tool start", event.get("name"))
       elif etype == "on_tool_end":
           print("tool end", event.get("name"))

Structured Event Streaming (UI-friendly)
-----------------------------------------

``agent.astream_structured_events(...)`` wraps raw events into a normalized stream that is easier for frontends.

Emitted event types include:

- ``reasoning_chunk``
- ``response_chunk``
- ``tool_call``
- ``tool_result``
- ``response_end``
- ``end``

.. code-block:: python

   import asyncio

   async def main() -> None:
       async for ev in agent.astream_structured_events(history):
           if ev["type"] == "response_chunk":
               print(ev["content"], end="", flush=True)
           elif ev["type"] == "tool_call":
               print(f"\n[tool] {ev['raw']['name']}")
           elif ev["type"] == "tool_result":
               print(f"\n[tool done] {ev['raw']['name']}")
           elif ev["type"] == "response_end":
               print("\n--- done ---")

   asyncio.run(main())

Memory Behavior While Streaming
-------------------------------

Both ``stream`` and ``astream`` support ``remember``:

- ``remember=\"all\"``: store all streamed messages
- ``remember=\"last\"``: store only last message
- ``remember=\"none\"``: do not store

This is helpful when building chat UIs where you control history persistence manually.

Common Patterns
---------------

Token-by-token UI output
^^^^^^^^^^^^^^^^^^^^^^^^

Use async events and print only model chunks:

.. code-block:: python

   async for event in agent.astream(history, stream_mode="events"):
       if event.get("event") == "on_chat_model_stream":
           chunk = event.get("data", {}).get("chunk")
           print(chunk, end="")

Tool progress timeline
^^^^^^^^^^^^^^^^^^^^^^

Track tool start/end events for progress bars or logs:

.. code-block:: python

   async for event in agent.astream(history, stream_mode="events"):
       if event.get("event") == "on_tool_start":
           print("START", event.get("name"))
       if event.get("event") == "on_tool_end":
           print("DONE", event.get("name"))

Workflow step progress
^^^^^^^^^^^^^^^^^^^^^^

Use workflow streaming for multi-step orchestration status:

.. code-block:: python

   for state in builder.stream(state={"messages": []}):
       print("current state:", state)

Tips
----

- Prefer ``astream`` for web servers and async apps.
- Prefer ``astream_structured_events`` for frontend clients that need a stable event contract.
- Use ``thread_id`` if you need resumable execution.
- Use small chunk handlers; do not block inside event loops.