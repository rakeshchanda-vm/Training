Models
======

This page explains how to use Andromeda model helpers when you want to work with
models directly, without creating a full agent or team.

Andromeda exposes two utility functions:

- ``get_chat_model`` for chat/completion models
- ``get_embedding_model`` for embeddings

Both functions accept a ``ModelConfig`` and return LangChain-compatible model
objects.

Chat Model (Synchronous)
------------------------

Use this when you want a simple request/response flow in normal Python code.

.. code-block:: python

   from andromeda.utils import get_chat_model
   from andromeda.config import ModelConfig
   from andromeda import HumanMessage

   chat_model = get_chat_model(
       ModelConfig(
           name="gpt-oss:20b",
           provider="litellm",
           other_args={
               "base_url": "http://localhost:11434",
               "temperature": 0.3,
           },
       )
   )

   messages = [HumanMessage(content="Explain Andromeda in one sentence.")]
   response = chat_model.invoke(messages)
   print(response.content)

Chat Model (Asynchronous)
-------------------------

Use this in async applications (FastAPI endpoints, background workers, async
pipelines) to avoid blocking the event loop.

.. code-block:: python

   import asyncio

   from andromeda.utils import get_chat_model
   from andromeda.config import ModelConfig
   from andromeda import HumanMessage


   async def main() -> None:
       chat_model = get_chat_model(
           ModelConfig(
               name="gpt-oss:20b",
               provider="litellm",
               other_args={"base_url": "http://localhost:11434"},
           )
       )

       messages = [HumanMessage(content="List three use cases for tool-enabled agents.")]
       response = await chat_model.ainvoke(messages)
       print(response.content)


   if __name__ == "__main__":
       asyncio.run(main())

OpenAI-Compatible Providers with Responses API
----------------------------------------------

When a provider exposes OpenAI-compatible ``/v1/chat/completions`` or
``/v1/responses`` endpoints, use Andromeda's ``openai`` provider path so
LangChain constructs ``ChatOpenAI``. This applies to OpenAI itself and to
compatible gateways or serving layers such as LiteLLM proxies, vLLM servers,
and other API-compatible providers.

Use ``provider="litellm"`` only when you want the ``langchain-litellm`` chat
wrapper. That wrapper targets LiteLLM chat completions and does not select
OpenAI's Responses API.

Install the OpenAI integration extra, then point ``OPENAI_BASE_URL`` at the
OpenAI-compatible endpoint:

.. code-block:: bash

   pip install "andromeda[openai]"
   export OPENAI_BASE_URL=http://localhost:8100/v1
   export OPENAI_API_KEY=your_proxy_key

Use ``other_args`` for LangChain ``ChatOpenAI`` options. ``model_kwargs`` is
useful for standard endpoint fields that LangChain does not expose as explicit
constructor arguments.

.. code-block:: python

   from andromeda.config import ModelConfig
   from andromeda.utils import get_chat_model

   chat_model = get_chat_model(
       ModelConfig(
           name="gpt-oss:20b",
           provider="openai",
           output_version="v1",
           other_args={
               "use_responses_api": True,
               "temperature": 1.0,
               "top_p": 1.0,
               "reasoning": {"effort": "low"},
               "model_kwargs": {
                   "max_output_tokens": 800,
               },
           },
       )
   )

   response = chat_model.invoke("Reply exactly: OK")
   print(response.text)

The same configuration works in YAML:

.. code-block:: yaml

   model:
     name: gpt-oss:20b
     provider: openai
     output_version: v1
     other_args:
       base_url: ${OPENAI_BASE_URL}
       use_responses_api: true
       temperature: 1.0
       top_p: 1.0
       reasoning:
         effort: low
       model_kwargs:
         max_output_tokens: 800

Endpoint Parameter Names
~~~~~~~~~~~~~~~~~~~~~~~~

OpenAI-compatible providers usually validate the request body against either a
Chat Completions schema or a Responses schema. Use the parameter names for the
endpoint you are calling. LangChain's ``ChatOpenAI`` converts some legacy chat
names when ``use_responses_api=True``; for example, ``max_tokens`` and
``max_completion_tokens`` are sent as ``max_output_tokens`` to the Responses
API. For explicit endpoint-native fields, pass them through ``model_kwargs``.
Provider wrappers may still normalize or drop fields before sending the final
request, so verify wrapper behavior for parameters that are critical to your
serving stack.

These commonly use the same names across Chat Completions and Responses
schemas when the backend supports them:

``temperature``, ``top_p``, ``top_k``, ``frequency_penalty``,
``presence_penalty``, ``repetition_penalty``, ``seed``, ``stop``,
``logit_bias``, ``top_logprobs``, ``structured_outputs``, ``tools``,
``tool_choice``, ``parallel_tool_calls``, ``stream``, ``ignore_eos``, and
``skip_special_tokens``.

Some concepts use different names by endpoint:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Concept
     - Chat Completions
     - Responses API
   * - Max output length
     - ``max_completion_tokens`` or legacy ``max_tokens``
     - ``max_output_tokens``
   * - Reasoning effort
     - ``reasoning_effort: "high"``
     - ``reasoning: {"effort": "high"}``
   * - Structured output
     - ``response_format`` plus provider-specific ``structured_outputs``, if supported
     - ``text: {"format": ...}`` plus provider-specific ``structured_outputs``, if supported

Do not send chat-only fields to Responses API backends unless your provider
documents support for them. Common chat-only fields include ``min_p``,
``min_tokens``, ``stop_token_ids``, ``logprobs`` as an enable flag, ``n``,
``stream_options``, ``prompt_logprobs``, ``length_penalty``,
``allowed_token_ids``, ``add_special_tokens``, ``truncate_prompt_tokens``,
``thinking_token_budget``, and ``include_reasoning``.

Responses-only fields include ``max_output_tokens``, ``reasoning``, ``text``,
``max_tool_calls``, ``store``, ``previous_response_id``, ``instructions``, and
``include``.

Provider and Model Caveats
~~~~~~~~~~~~~~~~~~~~~~~~~~

Accepted does not always mean honored. Some serving layers accept the same
field names as OpenAI but only implement a subset for a specific model family.
For reasoning models, prefer the reasoning-effort values documented by the
backend. For ``gpt-oss`` Harmony-backed models, ``low``, ``medium``, and
``high`` are the portable values; values such as ``minimal``, ``xhigh``, or
``max`` may be accepted by a schema but fail during rendering.

Sampling defaults also matter. For ``gpt`` reasoning models, ``temperature: 1.0`` and
``top_p: 1.0`` are safer defaults than low-temperature decoding; overly low
temperature can make repetition-loop truncation more likely. If your provider
injects model-specific defaults, prefer those defaults unless you need to tune
sampling deliberately.

Operational notes:

- Basic calls, full-history follow-up messages, streaming, structured output,
  and tool calls work through ``ChatOpenAI`` with ``use_responses_api=True``.
- Prefer full message history for conversation state. ``previous_response_id``
  can fail if the proxy/backend does not persist or route response IDs
  consistently.
- Keep tool choice at ``auto`` or ``none`` for Harmony-backed ``gpt-oss`` models;
  forced tool choice may be rejected by the backend.
- Use enough ``max_output_tokens`` for structured output. Reasoning models can
  otherwise spend the budget on reasoning and return an incomplete response
  without a parsed payload.
- If a proxy has fallbacks across providers, ensure shared parameters are
  supported by every fallback model or disable incompatible parameters for that
  route.

Embedding Model
---------------

Use embeddings for semantic search, retrieval, clustering, and similarity
scoring.

.. code-block:: python

   from andromeda.utils import get_embedding_model
   from andromeda.config import ModelConfig

   embedding_model = get_embedding_model(
       ModelConfig(
           name="nomic-embed-text:latest",
           provider="litellm",
           other_args={"base_url": "http://localhost:11434"},
       )
   )

   # Single text to vector
   query_vector = embedding_model.embed_query("Andromeda makes workflows easy.")
   print("Query vector dimensions:", len(query_vector))

   # Batch embedding for multiple texts
   docs = [
       "Andromeda supports supervisors and teams.",
       "Middleware enables guardrails and privacy controls.",
       "MCP servers expose external tools.",
   ]
   doc_vectors = embedding_model.embed_documents(docs)
   print("Number of vectors:", len(doc_vectors))
   print("Each vector dimensions:", len(doc_vectors[0]))

Embedding Model (Asynchronous)
------------------------------

Use this pattern when your application is already async. Async embedding APIs
are useful in web servers and concurrent pipelines.

.. code-block:: python

   import asyncio

   from andromeda.utils import get_embedding_model
   from andromeda.config import ModelConfig


   async def main() -> None:
       embedding_model = get_embedding_model(
           ModelConfig(
               name="nomic-embed-text:latest",
               provider="litellm",
               other_args={"base_url": "http://localhost:11434"},
           )
       )

       query_vector = await embedding_model.aembed_query(
           "Andromeda makes multi-agent workflows easier."
       )
       print("Async query vector dimensions:", len(query_vector))

       docs = [
           "Andromeda supports teams and supervisors.",
           "Middleware can enforce guardrails and privacy masking.",
       ]
       doc_vectors = await embedding_model.aembed_documents(docs)
       print("Async document vectors:", len(doc_vectors))


   if __name__ == "__main__":
       asyncio.run(main())

.. note::

   Async methods are available through the LangChain embeddings interface.
   Runtime behavior can vary by provider (native async vs wrapped sync calls).

What These Helpers Return
-------------------------

- ``get_chat_model`` returns a LangChain ``BaseChatModel``.
- ``get_embedding_model`` returns a LangChain ``Embeddings`` implementation.

Common methods:

- Chat model: ``invoke(...)``, ``ainvoke(...)``, and provider-supported streaming.
- Embedding model: ``embed_query(text)``, ``embed_documents(list[str])``,
  ``aembed_query(text)``, and ``aembed_documents(list[str])``.
