GitHub Copilot Models
=====================

Andromeda can use an existing **GitHub Copilot** subscription (including free tier) as a model
provider, for both chat and embeddings. It talks to Copilot's
OpenAI-compatible APIs (``https://api.githubcopilot.com``), so the resulting
models are ordinary LangChain ``BaseChatModel`` / ``Embeddings`` objects and
work with all existing Andromeda orchestration, tools, memory, callbacks and
streaming.

The intended use case is Andromeda running as a CLI agent/subagent on a machine
where a developer is already signed in to Copilot (for example inside VS Code):
it transparently reuses that login, so there are no API keys to manage.

.. note::

   This integration uses Copilot's internal token-exchange endpoint and editor
   headers, which are reverse-engineered rather than a sanctioned public API. It
   can break if GitHub changes those internals, and programmatic use sits in a
   grey area of the Copilot terms of service. Intended for personal,
   subscription-backed use.

Install
-------

The provider lives behind an optional extra (it pulls in ``langchain-openai``):

.. code-block:: bash

   pip install "andromeda[github-copilot]"

Quick start
-----------

Set ``provider="github_copilot"`` on a ``ModelConfig`` and use the agent as
usual. Authentication is resolved automatically (see `Authentication`_ below):

.. code-block:: python

   from andromeda.config import AgentConfig, ModelConfig
   from andromeda.core import Agent

   agent = Agent(
       AgentConfig(
           name="copilot-agent",
           model=ModelConfig(provider="github_copilot", name="gpt-4o"),
       )
   )
   reply = agent.chat("Hello!")   # returns a list of messages
   print(reply[-1].content)

Authentication
--------------

The provider discovers a usable Copilot token automatically, trying these
sources in order:

1. An explicit ready Copilot token — ``other_args["copilot_token"]``.
2. A ready Copilot token from ``GITHUB_COPILOT_TOKEN`` or ``COPILOT_API_KEY``.
3. A valid Copilot token in Andromeda's own cache (``~/.andromeda-copilot.json``).
4. A GitHub OAuth/PAT token, exchanged for a short-lived Copilot token — tried as
   ``other_args["github_token"]``, then ``GITHUB_TOKEN`` / ``GH_TOKEN``, then the
   editor's Copilot auth files (``~/.config/github-copilot/{apps,hosts}.json`` on
   Linux/macOS, ``%LOCALAPPDATA%/github-copilot`` on Windows), then the cache.

So an explicit ``copilot_token`` wins outright, but an explicit ``github_token``
is only consulted at step 4 — it does not outrank a ready token in the
environment or a valid cached token.

Short-lived Copilot tokens are refreshed automatically: proactively just before
expiry, and reactively if a request fails with an auth error mid-run. The
long-lived GitHub login is reused to mint fresh tokens, so you do not have to
sign in again.

Device-flow login (headless / remote)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On a machine with no Copilot login to discover (a remote server, a plain
terminal), you will be prompted to proceed with GitHub's one-time authentication device flow.

This can also be forcefully invoked:

.. code-block:: python

   from andromeda.utils.github_copilot import device_login

   device_login()  # prints a URL + code; authorize in a browser, then it caches the token

After this, the cached token is reused and refreshed automatically.

You can also disable the provider triggering device login automatically when it
can't find any auth, which will cause it to throw an error. ``auto_login`` is resolved in this order:

1. Explicit ``other_args={"auto_login": True/False}`` — wins if set.
2. The ``ANDROMEDA_COPILOT_AUTO_LOGIN`` environment variable (``1``/``true``/
   ``yes``/``on`` to force on, anything else to force off) — if set.
3. Otherwise it **autodetects**: on when running in an interactive terminal
   (stdin and stdout are a TTY), off otherwise — so the blocking prompt never
   hangs CI, daemons, or other non-interactive processes.

In other words, ``auto_login`` is **not** simply ``True`` by default; with no
explicit setting it is on only in an interactive terminal. And even when on, it
fires only as a last resort — if a token is already discoverable (editor login,
environment variable, or a valid cache) that is used and no prompt appears.

.. code-block:: python

   ModelConfig(
       provider="github_copilot",
       name="gpt-4o",
       other_args={"auto_login": True},  # force on; use False to force off
   )

Chat models
-----------

Use ``get_chat_model`` directly when you don't need a full agent:

.. code-block:: python

   from andromeda.utils import get_chat_model
   from andromeda.config import ModelConfig
   from andromeda import HumanMessage

   chat_model = get_chat_model(
       ModelConfig(provider="github_copilot", name="gpt-4o")
   )
   print(chat_model.invoke([HumanMessage(content="Explain Andromeda in one sentence.")]).content)

Tool calling, streaming (``stream`` / ``astream``) and async (``ainvoke``) all
work as they do for any other chat model.

Embeddings
----------

Copilot also exposes an embeddings endpoint:

.. code-block:: python

   from andromeda.utils import get_embedding_model
   from andromeda.config import ModelConfig

   embeddings = get_embedding_model(
       ModelConfig(provider="github_copilot", name="text-embedding-3-small")
   )
   vector = embeddings.embed_query("Andromeda makes workflows easy.")
   print("dimensions:", len(vector))

``embed_documents`` / ``aembed_query`` / ``aembed_documents`` are available too.

Discovering models and quota
----------------------------

Helpers in ``andromeda.utils.github_copilot`` show what your account can use:

.. code-block:: python

   from andromeda.utils.github_copilot import print_models, list_models, get_copilot_limits

   print_models()                 # models your plan can actually call, as a table
   print_models(chat_only=True)   # chat models only
   print_models(check_access=False)  # GitHub's raw, unfiltered catalog instead, including models your plan can't call.

   models = list_models(check_access=True)  # raw dicts, filtered to your access
   print(get_copilot_limits())    # plan, chat-enabled flag, quota snapshots / reset date

``print_models`` groups models by capability (chat / embeddings) and shows each
model's id, display name, vendor, context window and capability flags
(``tool_calls``, ``vision``, ``streaming``, ...).

GitHub's ``/models`` catalog lists every model it knows about, not just the
ones your subscription can invoke — a plan-based restriction (e.g. the free
tier) isn't reflected in the catalog's own metadata, so ``print_models`` /
``list_models(check_access=True)`` verify access with one lightweight live
call per remaining candidate model (parallelized, and skipped for models
GitHub already reports as policy-disabled).

Configuration overrides
-----------------------

All overrides flow through ``ModelConfig.other_args`` — there is no separate
schema:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Purpose
   * - ``github_token``
     - GitHub OAuth/PAT token to exchange for a Copilot token (overrides discovery).
   * - ``copilot_token``
     - A ready Copilot token to use as-is (skips the exchange).
   * - ``base_url``
     - Override the API base URL (defaults to ``https://api.githubcopilot.com``).
   * - ``editor_version``
     - Override the ``Editor-Version`` header sent to Copilot.
   * - ``default_headers``
     - Extra HTTP headers, merged with the required Copilot headers.
   * - ``auto_login``
     - Force device-flow login on/off when no auth is discovered.

Example:

.. code-block:: python

   ModelConfig(
       provider="github_copilot",
       name="gpt-4o",
       temperature=0.2,
       other_args={"github_token": "gho_...", "editor_version": "vscode/1.104.1"},
   )

Troubleshooting
---------------

- **"Could not authenticate with GitHub Copilot"** — no auth was found. Sign in
  to Copilot in your editor, run ``device_login()`` once, set ``GITHUB_TOKEN``,
  or set ``ANDROMEDA_COPILOT_AUTO_LOGIN=1`` to log in automatically on first use.
- **"... requires langchain-openai"** — install the extra:
  ``pip install "andromeda[github-copilot]"``.
- **An embedding request fails with a 400** — Copilot's embeddings endpoint
  expects string inputs; the provider already disables tiktoken token-array
  input for you, so leave ``check_embedding_ctx_length`` at its default unless
  you have a specific reason to change it.

See also
--------

- :doc:`../basic_concepts/models` — using ``get_chat_model`` / ``get_embedding_model`` directly.
- :doc:`../installation` — optional extras.
