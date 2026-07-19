Structured Outputs
==================

Structured responses let you force outputs into a predictable schema instead of free-form text.
In Andromeda, you can do this in two main places:

- Agent level: set ``response_format`` on ``AgentConfig``.
- Chat-model level: call ``with_structured_output(...)`` on a model from ``get_chat_model(...)``.

Why use this?

- Easier downstream parsing (no fragile regex/JSON extraction).
- Stronger contracts between agents and tools.
- Better validation and debugging in production workflows.

TypedDict vs BaseModel
----------------------

Both work, but they serve slightly different needs.

Use ``TypedDict`` when:

- You want a lightweight schema with minimal overhead.
- You mostly need dictionary-style access.
- You do not need Pydantic validators/field-level constraints.

Use ``BaseModel`` when:

- You want validation, defaults, aliases, and richer schema behavior.
- You prefer attribute access and model methods.
- You want a stricter contract for larger systems.

Typical return shape:

- ``TypedDict`` schema: usually a ``dict``.
- ``BaseModel`` schema: usually a Pydantic model instance.

Schema Definitions
------------------

.. code-block:: python

   from typing import Optional, TypedDict
   from pydantic import BaseModel, Field

   class ResearchSummaryTD(TypedDict):
       topic: str
       key_points: list[str]
       risk_rating: Optional[str]

   class ResearchSummaryModel(BaseModel):
       topic: str = Field(description="Topic that was summarized")
       key_points: list[str]
       risk_rating: Optional[str] = None

Agent: Structured Output (Sync + Async)
---------------------------------------

When you set ``response_format`` on ``AgentConfig``, ``Agent.invoke(...)`` and
``Agent.ainvoke(...)`` return the parsed structured object directly.

Agent sync example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Optional, TypedDict
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   class RiskBrief(TypedDict):
       topic: str
       key_points: list[str]
       risk_rating: Optional[str]

   cfg = AgentConfig(
       name="structured_agent_sync",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       response_format=RiskBrief,
       prompt="Always return the requested schema.",
   )

   agent = Agent(cfg)
   messages = [
       HumanMessage(content="Summarize risks of AI chip shortages for cloud providers.")
   ]

   out = agent.invoke(messages)
   print(type(out))
   print(out)
   # Example shape:
   # {
   #   "topic": "AI chip shortages for cloud providers",
   #   "key_points": ["...", "..."],
   #   "risk_rating": "medium"
   # }

Agent async example
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from typing import Optional
   from pydantic import BaseModel
   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda import HumanMessage

   class RiskBriefModel(BaseModel):
       topic: str
       key_points: list[str]
       risk_rating: Optional[str] = None

   async def main() -> None:
       cfg = AgentConfig(
           name="structured_agent_async",
           model=ModelConfig(name="qwen3:8b", provider="litellm"),
           response_format=RiskBriefModel,
           prompt="Return output that matches RiskBriefModel.",
       )

       agent = Agent(cfg)
       out = await agent.ainvoke([
           HumanMessage(content="Summarize AI chip supply-chain risk in 3 points.")
       ])
       print(type(out))
       print(out)
       # Often a RiskBriefModel instance:
       # RiskBriefModel(topic="...", key_points=[...], risk_rating="...")

   asyncio.run(main())

Chat Model: Structured Output (Sync + Async)
---------------------------------------------

Use this when you want structured output directly from the model without building a full agent.

.. code-block:: python

   from andromeda.utils.langtils import get_chat_model
   from andromeda.config import ModelConfig

   chat_model = get_chat_model(
       ModelConfig(name="qwen3:8b", provider="litellm")
   )

Chat sync example
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import TypedDict, Optional
   from andromeda import HumanMessage

   class SummaryTD(TypedDict):
       topic: str
       key_points: list[str]
       risk_rating: Optional[str]

   structured_llm = chat_model.with_structured_output(SummaryTD, method="json_schema")
   out = structured_llm.invoke([
       HumanMessage(content="Summarize AI chip shortage impact for cloud providers.")
   ])
   print(type(out))
   print(out)
   # Usually a dict matching SummaryTD

Chat async example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from typing import Optional
   from pydantic import BaseModel
   from andromeda import HumanMessage

   class SummaryModel(BaseModel):
       topic: str
       key_points: list[str]
       risk_rating: Optional[str] = None

   async def main() -> None:
       structured_llm = chat_model.with_structured_output(SummaryModel)
       out = await structured_llm.ainvoke([
           HumanMessage(content="Summarize AI chip shortage impact in 3 bullets.")
       ])
       print(type(out))
       print(out)
       # Usually a SummaryModel instance

   asyncio.run(main())

Agent vs Chat Output Differences
--------------------------------

- Agent path (``Agent.invoke`` / ``Agent.ainvoke``):
  - If ``response_format`` is set and parsing succeeds, returns structured object directly.
  - If structured output is not produced, agent falls back to returning message list.
- Chat path (``with_structured_output``):
  - Returns parsed structured object directly by default.
  - You can request raw + parsed for debugging.

Debug parsed vs raw (chat)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   debug_out = chat_model.with_structured_output(
       SummaryTD,
       method="json_schema",
       include_raw=True,
   ).invoke([HumanMessage(content="Summarize in schema format")])

   print(debug_out.keys())
   # dict_keys(["raw", "parsed", "parsing_error"])
   print(debug_out["parsed"])

Best Practices
--------------

- Prefer ``TypedDict`` for fast, simple machine-readable contracts.
- Prefer ``BaseModel`` for stricter validation and long-term maintainability.
- Keep schema fields small and explicit.
- Add clear prompt instructions like: ``Return output that strictly matches the schema.``
- For diagnostics, use chat ``include_raw=True`` to inspect parser behavior.