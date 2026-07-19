ChatGPT / Codex Models
=======================

Andromeda can use an existing **ChatGPT subscription** (Plus/Pro/Team) as a
model provider, authenticated with OAuth instead of an ``OPENAI_API_KEY``. It
talks to OpenAI's Codex backend (``https://chatgpt.com/backend-api/codex``),
so the resulting model is a real LangChain ``ChatOpenAI`` subclass (Responses
API) and works with all existing Andromeda orchestration, tools, memory,
callbacks and streaming — including native tool calling and
``with_structured_output``.

.. note::

   This integration builds on ``langchain-openai``'s experimental Codex
   support (``_ChatOpenAICodex`` / ``chatgpt_oauth``, shipped since
   ``langchain-openai`` 1.3.1). It is unofficial: use it only where your
   OpenAI account, workspace, plan, and OpenAI's terms permit
   ChatGPT-authenticated Codex access.

Install
-------

The provider lives behind an optional extra:

.. code-block:: bash

   pip install "andromeda[openai-codex]"

Sign in
-------

Authentication is OAuth, separate from any ``OPENAI_API_KEY``, via the
**browser-based Authorization Code + PKCE flow**: it binds a local loopback
callback server (default ``localhost:1455``) and opens your system browser to
the OpenAI sign-in page (the URL is also output as a fallback if a browser
can't be launched). The token bundle is written to
``~/.langchain/chatgpt-auth.json`` — a different path from the Codex CLI / VS
Code session (``~/.codex/auth.json``), so signing in here never invalidates
those — and is refreshed automatically after that.

You do not have to call this yourself: constructing ``ChatOpenAICodex`` with
no token on disk runs it for you automatically (see `Automatic sign-in`_
below). To sign in ahead of time instead:

.. code-block:: python

   from andromeda.utils.openai_codex import login_device

   login_device()

Check whether you're already signed in, and inspect the stored account/plan:

.. code-block:: python

   from andromeda.utils.openai_codex import logged_in, auth_status

   print(logged_in())     # True/False
   print(auth_status())   # {"logged_in": True, "account_id": ..., "plan_type": "plus", ...}

Automatic sign-in
~~~~~~~~~~~~~~~~~

When ``ChatOpenAICodex`` is constructed and no token is found on disk, it
runs :func:`login_device` for you instead of deferring to a
``FileNotFoundError`` on the first request. This is on by default only in an
interactive terminal, so it never blocks a non-interactive process (CI, a
daemon, a test run) waiting on a browser sign-in nobody is there to complete.
Resolution order:

1. Explicit ``other_args={"auto_login": True/False}`` — wins if set.
2. The ``ANDROMEDA_CODEX_AUTO_LOGIN`` environment variable (``1``/``true``/
   ``yes``/``on`` to force on, anything else to force off) — if set.
3. Otherwise it **autodetects**: on when running in an interactive terminal
   (stdin and stdout are a TTY), off otherwise.

In other words, ``auto_login`` is **not** simply ``True`` by default; with no
explicit setting it is on only in an interactive terminal. And even when on,
it only fires when nothing is already signed in — an existing (even expired)
token is refreshed transparently instead.

.. code-block:: python

   ModelConfig(
       provider="openai_codex",
       name="gpt-5.5",
       other_args={"auto_login": True},  # force on; use False to force off
   )

Passing your own ``token_provider`` object (rather than ``token_store_path``)
opts out of automatic sign-in entirely — you're managing auth yourself at that point.

Quick start
-----------

Set ``provider="openai_codex"`` on a ``ModelConfig``. The ``name`` must be a
**Codex model slug** (see `Discovering available models`_ below) — ordinary
OpenAI API model names like ``gpt-4`` or ``gpt-4o`` are rejected by this
backend:

.. code-block:: python

   from andromeda.config import AgentConfig, ModelConfig
   from andromeda.core import Agent

   agent = Agent(
       AgentConfig(
           name="codex-agent",
           model=ModelConfig(provider="openai_codex", name="gpt-5.5"),
       )
   )
   reply = agent.chat("Hello!")   # returns a list of messages
   print(reply[-1].content)

Chat models
-----------

Use ``get_chat_model`` directly when you don't need a full agent:

.. code-block:: python

   from andromeda.utils import get_chat_model
   from andromeda.config import ModelConfig
   from andromeda import HumanMessage

   model = get_chat_model(ModelConfig(provider="openai_codex", name="gpt-5.5"))
   print(model.invoke([HumanMessage(content="Explain Andromeda in one sentence.")]).content)

Tool calling, streaming (``stream`` / ``astream``) and async (``ainvoke``) all
work as they do for any other ``ChatOpenAI``-based model, since
``ChatOpenAICodex`` *is* one. There is no embeddings counterpart — the
subscription does not expose an embeddings endpoint.

Discovering available models
-----------------------------

The Codex backend only accepts its own model slugs, and which ones you can
use depends on your ChatGPT plan. Query it directly instead of guessing:

.. code-block:: python

   from andromeda.utils.openai_codex import print_models, list_models

   print_models()          # readable table: slug, plan access, context window, reasoning efforts
   print_models(show_hidden=True)  # also show internal/UI-only entries

   models = list_models()  # raw dicts, for programmatic use

``print_models`` marks each model's ``PLAN OK`` column by comparing its
``available_in_plans`` against your signed-in account's plan. A free ChatGPT
account, for example, typically sees ``gpt-5.5`` and ``gpt-5.4-mini``.

System prompts (``instructions``)
----------------------------------

The Codex backend is a Responses-API endpoint that rejects system-role chat
turns; any ``SystemMessage`` you pass is lifted automatically into the
top-level ``instructions`` field instead:

.. code-block:: python

   from langchain_core.messages import SystemMessage, HumanMessage

   model.invoke([
       SystemMessage(content="You are a terse assistant."),
       HumanMessage(content="Why is the sky blue?"),
   ])

You can also set a default via ``other_args``, or override per call with
``instructions=...``:

.. code-block:: python

   ModelConfig(
       provider="openai_codex",
       name="gpt-5.5",
       other_args={"instructions": "You are a senior Python reviewer. Be terse."},
   )

Configuration overrides
-----------------------

All overrides flow through ``ModelConfig.other_args`` — there is no separate
schema:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Purpose
   * - ``instructions``
     - Default system prompt (see above).
   * - ``token_store_path``
     - Use an alternate OAuth token file instead of the default
       ``~/.langchain/chatgpt-auth.json``.
   * - ``reasoning``
     - Reasoning effort, e.g. ``{"effort": "high"}`` (see a model's supported
       efforts via ``print_models()``).
   * - ``originator``
     - Telemetry header identifying the client; defaults to ``"andromeda"``.
   * - ``auto_login``
     - Force automatic browser-based sign-in on/off (see `Automatic sign-in`_).

Note that ``api_key`` and ``base_url`` are **not** valid overrides here — both
are managed by the OAuth layer and rejected if you try to set them; use
``ChatOpenAI`` directly for API-key auth instead.

Example:

.. code-block:: python

   ModelConfig(
       provider="openai_codex",
       name="gpt-5.5",
       other_args={
           "instructions": "You are a senior Python reviewer. Be terse.",
           "reasoning": {"effort": "high"},
       },
   )

Troubleshooting
---------------

- **"The '<model>' model is not supported when using Codex with a ChatGPT
  account."** — you passed a regular OpenAI API model name. Run
  ``print_models()`` and use one of the listed slugs (e.g. ``gpt-5.5``)
  instead.
- **429 "usage_limit_reached"** — your ChatGPT plan's Codex quota is
  exhausted; the error includes ``resets_at`` / ``resets_in_seconds``.
- **"requires langchain-openai>=1.3.1"** — install/upgrade the extra:
  ``pip install -U "andromeda[openai-codex]"``.
- **``FileNotFoundError: No ChatGPT OAuth token found``** — nothing is signed
  in and automatic sign-in didn't fire (non-interactive process, or
  ``auto_login`` explicitly off). Run ``login_device()`` once, or set
  ``ANDROMEDA_CODEX_AUTO_LOGIN=1``.
- **The process seems to hang after constructing the model** — you're likely
  mid sign-in: check the console (or your logging output) for the authorize
  URL printed by automatic sign-in / ``login_device()``, open it in a browser,
  and complete it — the call blocks until you do (up to 5 minutes by default).
- **No browser opens / running on a headless or remote box** — the browser
  flow needs to bind a local loopback port and launch a system browser; on a
  headless box, copy the printed authorize URL to a machine with a browser,
  complete sign-in there, then copy the resulting token file
  (``~/.langchain/chatgpt-auth.json``)
- **``RuntimeError: ChatGPT refresh token is no longer valid``
  (``invalid_grant``)** — the stored refresh token was revoked; run
  ``login_device()`` again.

See also
--------

- :doc:`../basic_concepts/models` — using ``get_chat_model`` / ``get_embedding_model`` directly.
- :doc:`github-copilot` — a similar subscription-backed provider, for GitHub Copilot.
- :doc:`../installation` — optional extras.
