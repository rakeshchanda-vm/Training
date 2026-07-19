Model Context Protocol (MCP)
============================

MCP (Model Context Protocol) integration for connecting external tools and services as native Andromeda tools.

.. note:: 
   MCP integration requires installing the optional ``mcp`` package:
   
   ``pip install mcp`` or when installing Andromeda: ``pip install andromeda[mcp]``

Overview
--------

The MCP adapter provides a thin bridge between MCP servers and Andromeda's tool registry.
MCP tools are discovered, converted into Andromeda tools, and then referenced by name in agent/supervisor ``tools``.

What you get:

* **Connect External Services**: GitHub, filesystem, database, APIs, and custom MCP servers
* **Native Tool Experience**: MCP tools become regular Andromeda tools
* **Flexible Transport**: ``stdio`` (local process) and ``http`` (remote endpoint)
* **Tool Governance**: Include/exclude filters and optional naming prefix

How it works at load time:

1. Andromeda reads ``mcp_servers`` from config.
2. It registers MCP servers/tools before resolving ``tools`` names.
3. Your agents can reference discovered tools by their registered names.

Prerequisites
-------------

Install MCP support:

.. code-block:: bash

  pip install mcp
  # or
  pip install andromeda[mcp]

For ``stdio`` servers, also ensure:

* The server command exists in PATH or uses an absolute path.
* Required runtime (Python/Node/etc.) is installed.
* Working directory and permissions are valid.

For ``http`` servers, also ensure:

* URL is reachable from your runtime environment.
* Required auth headers (for example ``Authorization``) are provided.
* TLS/cert/network policy allows outbound connection.

Configuration
-------------

MCP servers are configured in your ``config.yaml`` file:

.. code-block:: yaml

   mcp_servers:
     filesystem:
       transport: stdio
       command: ["python", "servers/fs_server.py"]
       cwd: "./servers"
       env:
         DEBUG: "true"
     
     github:
       transport: http
       url: "https://api.githubcopilot.com/mcp/"
       headers:
         Authorization: "Bearer ${GITHUB_TOKEN}"
   
   agents:
     - name: researcher
       model: 
         name: llama3:8b
         provider: litellm
       tools:
        - filesystem_read_file   # default registered form is <server>_<tool>
        - github_search_repos

.. important::
   By default MCP tool names are registered as ``<server>_<tool>``.
   Example: ``filesystem_read_file`` (not ``filesystem.read_file``).

Environment Variables and Tokens
--------------------------------

Andromeda supports environment-variable interpolation in config values using ``${VAR_NAME}``.

Example:

.. code-block:: yaml

  mcp_servers:
    github:
      transport: http
      url: "${MCP_GITHUB_URL}"
      headers:
        Authorization: "Bearer ${GITHUB_TOKEN}"
        User-Agent: "Andromeda/1.0"

Before running:

.. code-block:: bash

  export MCP_GITHUB_URL="https://api.githubcopilot.com/mcp/"
  export GITHUB_TOKEN="<your_token>"

If a variable is missing, config loading fails with a clear error that includes:

* config source
* missing variable name
* location in config
* original expression (for example ``${GITHUB_TOKEN}``)

Token/Auth patterns
~~~~~~~~~~~~~~~~~~~

Common header patterns for HTTP MCP endpoints:

.. code-block:: yaml

  # Bearer token
  headers:
    Authorization: "Bearer ${API_TOKEN}"

.. code-block:: yaml

  # API key header
  headers:
    X-API-Key: "${SERVICE_API_KEY}"

.. code-block:: yaml

  # Multiple headers
  headers:
    Authorization: "Bearer ${API_TOKEN}"
    X-Org-Id: "${ORG_ID}"
    User-Agent: "Andromeda/1.0"

Transport Types
---------------

**Stdio Transport** (Local Processes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For local MCP servers running as separate processes:

.. code-block:: yaml

   mcp_servers:
     local_server:
       transport: stdio
       command: ["python", "my_server.py"]
       args: ["--config", "server.json"]
       cwd: "/path/to/server"
       env:
         API_KEY: "${LOCAL_SERVER_API_KEY}"
         DEBUG: "true"

Notes:

* ``command`` is required for ``stdio`` transport.
* ``env`` values are passed to the MCP process.
* ``cwd`` controls working directory for the spawned process.

**HTTP Transport** (Remote Services)  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For remote MCP endpoints:

.. code-block:: yaml

   mcp_servers:
     remote_api:
       transport: http
       url: "${REMOTE_MCP_URL}"
       headers:
         Authorization: "Bearer ${REMOTE_MCP_TOKEN}"
         User-Agent: "Andromeda/1.0"

Notes:

* ``url`` is required for ``http`` transport.
* ``headers`` must be a key-value map.
* If ``transport`` is omitted, Andromeda defaults to ``stdio``.

Tool Management
---------------

**Tool Filtering**

Include only specific tools:

.. code-block:: yaml

   mcp_servers:
     filesystem:
       command: ["python", "fs_server.py"]
       include_tools:
         - read_file
         - write_file
         - list_directory

Exclude specific tools:

.. code-block:: yaml

   mcp_servers:
     large_service:
       url: "https://api.service.com/mcp/"
       exclude_tools:
         - dangerous_operation
         - deprecated_tool

**Tool Prefixing**

Customize tool names with prefixes:

.. code-block:: yaml

   mcp_servers:
     github:
       url: "https://api.githubcopilot.com/mcp/"
       prefix: "gh"  # Tools become: gh_search_repos, gh_create_issue

.. note::
   Tool registration format uses underscore separators: ``<prefix>_<tool>``.

Usage in Agents
---------------

Once configured, MCP tools can be used like any other Andromeda tool:

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.core.agent import Agent
   from andromeda import HumanMessage
   
  # Load config with MCP servers
   config = AndromedaConfig.load_from_file("config.yaml")
   
   # Create agent with MCP tools
   agent = Agent(config.agents["researcher"])
   
   # Use MCP tools in conversation
     response = await agent.ainvoke([
       HumanMessage("Read the contents of README.md using filesystem tools")
   ])

  Minimal sync usage:

  .. code-block:: python

     from andromeda import HumanMessage

     response = agent.invoke([
       HumanMessage(content="Search repositories related to LangGraph")
     ])
     print(response)

Common MCP Servers
------------------

**Filesystem Server**
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   mcp_servers:
     filesystem:
       command: ["python", "-m", "mcp.server.filesystem"]
       args: ["--base-path", "./workspace"]

**Database Server**  
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   mcp_servers:
     database:
       command: ["python", "-m", "mcp.server.postgres"]
       env:
         DATABASE_URL: "postgresql://user:pass@localhost/db"

**Web Search Server**
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   mcp_servers:
     search:
       command: ["python", "search_server.py"]
       env:
         SEARCH_API_KEY: "your-key"

Troubleshooting
---------------

**Common Issues:**

* **Missing MCP Package**: Install with ``pip install mcp``
* **Server Not Found**: Check command path and working directory
* **Permission Errors**: Ensure proper file permissions for stdio servers
* **Network Issues**: Verify URL and authentication for HTTP servers
* **Missing Env Var**: Ensure all ``${VAR_NAME}`` placeholders are exported
* **Bad Header Shape**: ``headers`` must be a mapping, not a list/string

**Debug Mode:**

Enable debug logging to troubleshoot MCP connections:

.. code-block:: yaml

   mcp_servers:
     debug_server:
       command: ["python", "server.py"]
       env:
         DEBUG: "true"
         LOG_LEVEL: "DEBUG"

**Validation:**

Test MCP server connection independently:

.. code-block:: bash

   # Test stdio server
   python your_server.py
   
   # Test HTTP server  
   curl -X POST https://api.example.com/mcp/

  Fail-fast config loader (recommended)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  .. code-block:: python

     from andromeda.config import AndromedaConfig

     def load_config_or_exit(path: str) -> AndromedaConfig:
       try:
         return AndromedaConfig.load_from_file(path)
       except FileNotFoundError as exc:
         raise SystemExit(f"Config not found: {exc}")
       except ValueError as exc:
         raise SystemExit(f"Invalid config: {exc}")
       except RuntimeError as exc:
         # Includes MCP discovery/connectivity failures at load time.
         raise SystemExit(f"MCP setup failed: {exc}")

Advanced Configuration
----------------------

**List-Style Configuration**

Alternative YAML syntax for multiple server definitions:

.. code-block:: yaml

   mcp_servers:
     - filesystem:
         command: ["python", "fs_server.py"]
     - github:
         transport: http
         url: "https://api.githubcopilot.com/mcp/"

**Environment Variables**

Use environment variables in configuration:

.. code-block:: yaml

   mcp_servers:
     secure_api:
       transport: http
       url: "${MCP_API_URL}"
       headers:
         Authorization: "Bearer ${API_TOKEN}"

Long-lived sessions (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, registered MCP tools use short-lived sessions per call.
For advanced workloads, you can open long-lived sessions explicitly:

.. code-block:: python

   from andromeda.tools.mcp_adapter import open_mcp_sessions

   async with open_mcp_sessions(config.mcp_servers) as sessions:
       fs_session = sessions["filesystem"]["session"]
       result = await fs_session.call_tool("read_file", {"path": "README.md"})

Execution-scoped tool isolation (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default MCP flow registers discovered tools into Andromeda's global tool
registry. That is the simplest option and is the recommended default for most
applications.

For advanced service architectures, you may want per-request or per-tenant MCP
isolation instead. A common example is a long-running service where multiple
agents may run concurrently but should not share MCP tool registrations or MCP
runtime state.

For that use case, Andromeda exposes a low-level API based on:

* ``ExecutionContext`` for request-scoped state
* ``Toolkit`` for request-scoped tool registration
* ``MCPRuntime`` for long-lived MCP sessions bound to that execution

Minimal config-driven example:

.. code-block:: python

   import asyncio

   from andromeda import HumanMessage
   from andromeda.config import AndromedaConfig
   from andromeda.core.agent import Agent
   from andromeda.core.workflow import ExecutionContext
   from andromeda.tools.mcp_adapter import MCPRuntime
   from andromeda.tools.toolkit import Toolkit

   async def main() -> None:
       config_data = {
           "mcp_servers": {
               "github": {
                   "transport": "http",
                   "url": "https://example.com/mcp/",
                   "prefix": "github",
               }
           },
           "agents": {
               "researcher": {
                   "name": "researcher",
                   "model": {"name": "fake-model", "provider": "litellm"},
                   "prompt": "Use MCP tools when helpful.",
                   "tools": ["github_search_repos"],
               }
           },
           "supervisor": {
               "name": "supervisor",
               "model": {"name": "fake-model", "provider": "litellm"},
           },
           "planner": {
               "model": {"name": "fake-model", "provider": "litellm"},
           },
       }

       context = ExecutionContext(name="request-123")
       context.toolkit = Toolkit()
       context.mcp_runtime = await MCPRuntime.open(
           config_data["mcp_servers"],
           execution_context=context,
       )

       try:
           cfg = AndromedaConfig.load_from_config(
               config_data,
               execution_context=context,
           )
           agent = Agent(cfg.agents["researcher"])
           result = await agent.ainvoke(
               [HumanMessage(content="Search for LangGraph repositories")]
           )
           print(result)
       finally:
           await context.mcp_runtime.aclose() # important cleanup to avoid resource leaks

   asyncio.run(main())

How this works:

* MCP tools are discovered and registered into ``context.toolkit`` instead of the
  global registry.
* Tool names in agent config are resolved against that scoped toolkit.
* Tool calls are executed through ``context.mcp_runtime`` instead of opening a
  fresh MCP connection for every call.

Important notes:

* This is a low-level API. You are responsible for opening and closing the runtime.
* Prefer one fresh ``ExecutionContext`` and scoped ``Toolkit`` per isolated request,
  tenant, or service execution.
* If you reuse the same scoped toolkit across different MCP runtimes or different
  MCP configurations, previously-registered tool objects may be retained.
* This advanced path is optional. If you do not need isolation, the default
  global-registry flow remains simpler.

Security recommendations
------------------------

* Never hardcode tokens in YAML committed to git.
* Use secret managers or environment variables for all credentials.
* Scope tokens to minimum permissions required by MCP tools.
* Rotate tokens regularly and revoke leaked credentials immediately.
* Use ``include_tools`` to reduce exposed tool surface area.

For more information on MCP protocol details, visit the `Model Context Protocol specification <https://modelcontextprotocol.io/specification/>`_.
