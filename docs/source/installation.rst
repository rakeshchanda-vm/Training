Installation
============

Prerequisites
-------------

- Python 3.11+

Local Installation
------------------

.. code-block:: bash

   # Option 1: Clone the repository
   git clone <repository-url>
   cd andromeda
   pip install -e .

   # Option 2: Install with pip (recommended for development)
   pip install git+https://github.com/AI-Emerging-Tech/ET-Agentify.git

   # Option 3: Install optional integrations
   # openai is required for provider="openai" and ChatOpenAI, including
   # LiteLLM proxies used through OpenAI-compatible endpoints.
   # cli is required for the `andromeda` command-line tool (see Verification below).
   pip install "andromeda[openai,mcp,ops,github-copilot,openai-codex,cli]"


Environment Setup
-----------------

Create a ``.env`` file in the project root:

.. code-block:: bash

   # Required: Tavily API key for web search
   TAVILY_API_KEY=your_tavily_api_key_here

   # Optional: Custom Ollama base URL (defaults to http://localhost:11434)
   OLLAMA_BASE_URL=http://localhost:11434

   # Optional: OpenAI-compatible model provider or LiteLLM proxy
   OPENAI_API_KEY=your_openai_or_proxy_key_here
   OPENAI_BASE_URL=http://localhost:8100/v1
   
   # Optional: Langfuse API key for tracing
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
   LANGFUSE_HOST=your_langfuse_host_here

   # Optional: GitHub Copilot model provider (provider="github_copilot")
   # Auth is auto-discovered from an editor login if present; these let you
   # supply or override it explicitly. Set at most one token variable.
   #   A ready-to-use Copilot token (used as-is, no exchange):
   GITHUB_COPILOT_TOKEN=your_copilot_token_here
   #   ...or a GitHub OAuth/PAT token to exchange for a Copilot token:
   GITHUB_TOKEN=your_github_token_here
   #   Override the Editor-Version header sent to Copilot (defaults to a pinned value):
   GITHUB_COPILOT_EDITOR_VERSION=vscode/1.104.1
   #   Force a one-time device login when no auth is found (1/true or 0/false).
   #   Default when unset: on only in an interactive terminal, off otherwise.
   ANDROMEDA_COPILOT_AUTO_LOGIN=1
   # Aliases also accepted: COPILOT_API_KEY (for GITHUB_COPILOT_TOKEN), GH_TOKEN (for GITHUB_TOKEN)

   # Optional: ChatGPT/Codex model provider (provider="openai_codex")
   # Force auto-login off with:
   ANDROMEDA_CODEX_AUTO_LOGIN=0

Verification
------------

Test your installation (requires the ``cli`` extra, see Option 3 above):

.. code-block:: bash

   andromeda --version
