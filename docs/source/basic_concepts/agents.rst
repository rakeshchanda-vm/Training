Agents
======

Andromeda ``Agent`` is the core single-agent runtime. It combines:

- a chat model
- optional tools
- optional middleware
- memory-aware helper methods
- sync and async execution APIs

Use ``Agent`` when one model-driven worker is enough and you do not need a
multi-agent team or supervisor.

Quick Start (Single Prompt)
---------------------------

This is the smallest complete example using ``Agent.invoke``.

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   agent = Agent(
       AgentConfig(
           name="assistant",
           model=ModelConfig(name="qwen3:8b", provider="litellm"),
           prompt="You are a concise technical assistant.",
       )
   )

   messages = [HumanMessage(content="What is Andromeda in one sentence?")]
   response_messages = agent.invoke(messages)
   print(response_messages[-1].content)

Agent with Tools
----------------

Attach tools through ``AgentConfig.tools`` so the model can call them.

.. code-block:: python

   from typing import Dict, Any

   from andromeda.tools import tool
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage


   @tool
   def echo_tool(text: str) -> Dict[str, Any]:
       """Echo text back. Replace with your real tool logic."""
       return {"echo": text}


   agent = Agent(
       AgentConfig(
           name="tool_agent",
           model=ModelConfig(name="qwen3:8b", provider="litellm"),
           tools=[echo_tool],
           prompt="Use tools when they improve correctness.",
       )
   )

   result = agent.invoke([
       HumanMessage(content="Use echo_tool to repeat: Andromeda")
   ])
   print(result[-1].content)

Asynchronous Agent Example
--------------------------

Use ``ainvoke`` when your application is async.

.. code-block:: python

   import asyncio

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage


   async def main() -> None:
       agent = Agent(
           AgentConfig(
               name="async_assistant",
               model=ModelConfig(name="qwen3:8b", provider="litellm"),
           )
       )

       reply = await agent.ainvoke([
           HumanMessage(content="Give me 3 bullet points about workflow automation.")
       ])
       print(reply[-1].content)


   if __name__ == "__main__":
       asyncio.run(main())

Conversation Helpers and Memory
-------------------------------

For chat-style interactions, ``chat`` and ``achat`` reuse ``agent.memory``.

.. code-block:: python

    from andromeda.core.agent import Agent
    from andromeda.config import AgentConfig, ModelConfig

    agent = Agent(
        AgentConfig(name="chat_agent", model=ModelConfig(name="qwen3:8b", provider="litellm"))
    )

    agent.chat("My name is Omar.")
    second = agent.chat("What is my name?")

    print("Assistant answer:", second[-1].content)  # use -1, not -2
    print("Stored messages:", len(agent.memory))
    print("Memory snapshot:", [getattr(m, "content", str(m)) for m in agent.memory])

Task and Research Helpers
-------------------------

``task`` and ``research`` are convenience wrappers that shape a stronger
goal-oriented prompt and return final text output.

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig

   agent = Agent(
       AgentConfig(name="worker", model=ModelConfig(name="qwen3:8b", provider="litellm"))
   )

   report = agent.task("Summarize the top 5 risks in deploying a new API.")
   print(report)

Streaming Responses
-------------------

Use streaming for real-time UI output and tool progress visibility.

.. code-block:: python

   import asyncio

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage


   async def main() -> None:
       agent = Agent(
           AgentConfig(name="streamer", model=ModelConfig(name="qwen3:8b", provider="litellm"))
       )
       history = [HumanMessage(content="Explain streaming in plain English.")]

       async for event in agent.astream(history, stream_mode="events"):
           if event.get("event") == "on_chat_model_stream":
               print(event.get("data", {}))


   if __name__ == "__main__":
       asyncio.run(main())

Useful Agent Methods
--------------------

- ``invoke(messages, ...)``: sync invocation.
- ``ainvoke(messages, ...)``: async invocation.
- ``stream(messages, stream_mode=...)``: sync streaming.
- ``astream(messages, stream_mode=...)``: async streaming.
- ``chat(message, ...)`` / ``achat(message, ...)``: memory-based chat helpers.
- ``task(task, ...)`` / ``research(task, ...)``: report-oriented helper methods.
- ``set_thread_id(...)`` and ``set_metadata(...)``: set run context values.

Common Configuration Fields
---------------------------

Important ``AgentConfig`` fields used most often:

- ``name``
- ``model`` (``ModelConfig``)
- ``tools``
- ``prompt``
- ``middleware``
- ``response_format``
- ``type`` (``react`` or ``codeact``)
- ``output_standard`` (``andromeda`` or ``langchain``)