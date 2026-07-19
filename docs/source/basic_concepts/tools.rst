Tools
=====

Tools let agents interact with systems outside the model itself, for example:

- web search APIs
- URL crawling
- local filesystem workspaces
- MCP server tools

In Andromeda, there are two ways to attach tools:

- pass tool objects directly in Python (``AgentConfig.tools=[...]``)
- reference tool names in YAML via the global registry

Most users start with direct Python objects, then move to YAML tool names when
they need config-driven deployment.

Built-in Research Tools
-----------------------

Andromeda's default tool registry includes:

- ``web_search``
- ``news_search``
- ``search_historical``
- ``crawl_url``

These are available from ``andromeda.tools``.

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda.tools import web_search, news_search
   from andromeda import HumanMessage

   agent = Agent(
     AgentConfig(
       name="researcher",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       tools=[web_search, news_search],
       prompt="Use tools when information must be verified.",
     )
   )

   result = agent.invoke([
     HumanMessage(content="Find recent updates about Model Context Protocol.")
   ])
   print(result[-1].content)

When to use each built-in:

- ``web_search``: general web information (non-news focus)
- ``news_search``: recent or time-sensitive reporting
- ``search_historical``: bounded search by date range
- ``crawl_url``: extract content from a known URL

Example: historical search

.. code-block:: python

   from andromeda.tools import search_historical

   data = search_historical.invoke(
     {
       "query": "Open-source AI agent frameworks",
       "start_date": "2024-01-01",
       "end_date": "2024-12-31",
       "topic": "general",
     }
   )
   print(data)

Example: crawl a URL

.. code-block:: python

   from andromeda.tools import crawl_url

   content = crawl_url.invoke(
     {
       "url": "https://example.com/docs",
       "max_chars": 3000,
     }
   )
   print(content)

Using Tool Names in ``config.yaml``
-----------------------------------

When you load YAML config, tools can be referenced by registered names.

.. code-block:: yaml

   agents:
     - name: researcher
       model:
         name: qwen3:8b
         provider: litellm
       tools:
         - web_search
         - news_search

Custom Tools with Registry (YAML)
---------------------------------

If your YAML references a custom tool by string name, register it before
``AndromedaConfig.load_from_file(...)``.

.. code-block:: python

   from andromeda.tools import tool
   from andromeda.tools.toolkit import register_tool
   from andromeda.config import AndromedaConfig


   @tool
   def my_custom_tool(query: str) -> str:
     """Custom tool for domain-specific processing."""
     return f"Processed: {query}"


   register_tool(my_custom_tool)
   cfg = AndromedaConfig.load_from_file("config.yaml")

Register multiple tools:

.. code-block:: python

   from andromeda.tools.toolkit import register_tools
   from andromeda.tools import tool


   @tool
   def summarize_text(text: str) -> str:
     return text[:200]


   @tool
   def classify_priority(text: str) -> str:
     return "high" if "urgent" in text.lower() else "normal"


   register_tools([summarize_text, classify_priority])

.. note::

   If you pass tool objects directly in Python, explicit registry
   registration is not required.

Filesystem Tools
----------------

Andromeda includes a filesystem tool factory in
``andromeda.tools.filesystem.make_filesystem_tools``.

This is important behavior to understand:

- filesystem tools are created per allowed directory scope
- paths are validated so tools can only operate inside allowed directories
- these tools are not auto-registered globally by default

Create filesystem tools:

.. code-block:: python

   from andromeda.core.agent import Agent
   from andromeda.config import AgentConfig, ModelConfig
   from andromeda.tools.filesystem import make_filesystem_tools
   from andromeda import HumanMessage

   fs_tools = make_filesystem_tools(["./workspace"])

   agent = Agent(
     AgentConfig(
       name="file_agent",
       model=ModelConfig(name="qwen3:8b", provider="litellm"),
       tools=list(fs_tools.values()),
       prompt="Use filesystem tools carefully and keep edits minimal.",
     )
   )

   response = agent.invoke([
     HumanMessage(content="List files in the workspace and read README.md")
   ])
   print(response[-1].content)

Available filesystem tools returned by ``make_filesystem_tools``:

- ``read_file``
- ``edit_file``
- ``list_directory``
- ``list_allowed_directories``
- ``directory_tree``
- ``write_file``
- ``grep_file``
- ``search_files``
- ``search_and_replace_file_edit``
- ``create_directory``
- ``append_to_file``
- ``delete_file_or_directory``

Example: inspect and then edit a file

.. code-block:: python

   from andromeda.tools.filesystem import make_filesystem_tools

   fs_tools = make_filesystem_tools(["./workspace"])

   print(fs_tools["list_directory"].invoke({"path": "."}))
   print(fs_tools["read_file"].invoke({"path": "README.md", "start_line": 0, "end_line": 40}))

   edit_result = fs_tools["search_and_replace_file_edit"].invoke(
     {
       "path": "README.md",
       "search": "Old title",
       "replace": "New title",
     }
   )
   print(edit_result)

Example: register filesystem tools for YAML usage

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.tools.filesystem import make_filesystem_tools
   from andromeda.tools.toolkit import register_tools

   fs_tools = make_filesystem_tools(["./workspace"])
   register_tools(list(fs_tools.values()))

   cfg = AndromedaConfig.load_from_file("config.yaml")

.. note::

   If ``tools: [read_file, edit_file, ...]`` is declared in YAML, those names
   must be registered before loading config.

.. _workspace-sessions:

Workspace sessions
------------------

``WorkspaceSession`` is the recommended abstraction for new agent workspaces.
It binds filesystem tools, optional shell tools, seeding, policy, and lifecycle
to a single materialized workspace root.

To drive a long-horizon agent loop on top of a session (edit files, run tests,
iterate), bind it to a :doc:`Workspace Agent <workspace-agents>`.

.. code-block:: python

   from andromeda.workspace import FileSeed, WorkspacePolicy, WorkspaceSession

   session = WorkspaceSession.create(
       backend="local_fs",
       root="./workspace",
       seed=[FileSeed(path="AGENTS.md", content="Work only in this workspace.")],
       policy=WorkspacePolicy(read_only=False, enable_shell=True),
   )

   tools = session.tools()

``WorkspaceSession`` uses smaller tool profiles than the compatibility factory.
When no profile is specified, shell-enabled sessions default to
``shell_enabled`` and shell-disabled sessions default to ``shell_disabled``.

Available profiles:

- ``minimal``: ``read_file`` and ``apply_patch``
- ``read_only``: ``read_file``, ``list_directory``, ``directory_tree``,
  ``grep_file``, ``search_files``, and ``list_allowed_directories``
- ``full_compatibility``: the current v1 ``make_filesystem_tools`` set
- ``shell_enabled``: ``read_file`` and ``apply_patch`` plus shell tools
- ``shell_disabled``: ``minimal`` plus ``read_only``

``write_file``, ``edit_file``, ``append_to_file``, and
``search_and_replace_file_edit`` are compatibility-only tools. New coding-agent
workspaces should prefer ``apply_patch`` for structured mutations and shell for
framework commands, package commands, tests, ``mkdir``, ``ls``, and similar
operations.

.. code-block:: python

   from andromeda.workspace import ShellPolicy, WorkspacePolicy, WorkspaceSession

   session = WorkspaceSession.create(
       backend="ephemeral_fs",
       policy=WorkspacePolicy(
           enable_shell=True,
           shell=ShellPolicy(enable_background_shell=True),
       ),
   )

   tools = session.tools()
   task = tools["shell_start"].invoke({"command": "pytest"})
   # Later: shell_status, shell_output, shell_kill, or shell_list.

``enable_shell=True`` only returns shell tools for backends that expose a real
materialized directory or provider-backed execution environment. Local
``local_fs`` and generated ``ephemeral_fs`` sessions run shell commands in the
workspace root. ``bubblewrap_process`` sandboxes each command with ``bwrap``.
``gvisor_container`` runs commands in a long-lived Docker container using the
``runsc`` runtime. ``s3_snapshot`` materializes into an agent home before tools are
created. ``postgres_vfs`` is a native Postgres virtual filesystem backend and
is intentionally file-only.

Generated workspaces use a backend-neutral agent home layout:

.. code-block:: text

   ~/.andromeda/agent-home/<session_id>/workspace
   ~/.andromeda/agent-home/<session_id>/metadata.json

The metadata file records the backend, session id, root, seed types, ownership,
and policy summary. Explicit ``root=`` values for ``local_fs`` continue to use
the caller-provided directory.

Provider examples:

.. code-block:: python

   # Explicit local workspace.
   local_session = WorkspaceSession.create(
       backend="local_fs",
       root="./workspace",
       policy=WorkspacePolicy(enable_shell=True),
   )

   # Ephemeral local workspace under ~/.andromeda/agent-home.
   ephemeral_session = WorkspaceSession.create(
       backend="ephemeral_fs",
       policy=WorkspacePolicy(enable_shell=True),
   )

   # Production microVM workspace through a sandbox control plane.
   from andromeda.workspace import ContainerdKataSettings, PostgresVFSSettings, NerdctlDevSettings
   from andromeda.workspace import BubblewrapProcessSettings, GVisorContainerSettings

   microvm_session = WorkspaceSession.create(
       backend="microvm", # Currently untested due to OS limitations and should be considered experimental.
       settings=ContainerdKataSettings(
           control_plane_url="https://sandbox-control-plane.internal",
           image="andromeda-agent:latest",
           runtime="io.containerd.kata.v2",
           ttl_seconds=3600,
       ),
       policy=WorkspacePolicy(enable_shell=True),
   )

   # Local dev microVM via nerdctl + Kata (requires /dev/kvm on the host).
   nerdctl_session = WorkspaceSession.create(
       backend="microvm",
       settings=NerdctlDevSettings(image="andromeda-agent:latest"),
       policy=WorkspacePolicy(enable_shell=True),
   )

   # Postgres VFS workspace. This backend is file-only.
   postgres_session = WorkspaceSession.create(
       backend="postgres_vfs",
       settings=PostgresVFSSettings(
           connection_string="postgresql://user:pass@host/db",
           namespace_key="run-123",
           ensure_schema=True,
       ),
       policy=WorkspacePolicy(enable_shell=False),
   )

   # Bubblewrap process sandbox.
   bubblewrap_session = WorkspaceSession.create(
       backend="bubblewrap_process",
       root="./workspace",
       settings=BubblewrapProcessSettings(),
       policy=WorkspacePolicy(enable_shell=True),
   )

   # gVisor container sandbox via Docker + runsc.
   gvisor_session = WorkspaceSession.create(
       backend="gvisor_container",
       settings=GVisorContainerSettings(image="python:3.12-slim"),
       policy=WorkspacePolicy(enable_shell=True),
   )

For production ``microvm`` workspaces, Andromeda uses a sandbox control plane
client. The control plane owns host-level ``containerd`` access, Kata runtime
selection, image/runtime validation, quotas, TTL cleanup, logs, process
lifecycle, and orphan reaping. The control plane host must have Linux
virtualization support, ``containerd``, the Kata runtime (for example
``io.containerd.kata.v2``), and permission to run the configured images.

The optional ``NerdctlKataDevProvider`` is intended for local development and
opt-in integration tests only. It shells out to ``nerdctl`` and should not be
used as the production isolation boundary.

On hosts without ``/dev/kvm``, use ``bubblewrap_process`` or ``gvisor_container``
instead of ``microvm``. 

The Postgres VFS backend stores namespace-isolated file trees and revision
history in Postgres. It supports file tools through a driver-backed filesystem
tool factory and raises a compatibility error if shell is requested.

Background shell processes are tracked by the owning ``WorkspaceSession``.
``WorkspaceSession.cleanup()`` terminates dangling shell processes before
removing an owned workspace or destroying a provider. The shell wrapper fixes
``cwd`` to the workspace root, applies env allowlists, enforces timeouts/output
limits, disables raw shell unless explicitly enabled, and captures output. It is
not a filesystem jail by itself for local providers; production shell isolation
still depends on container or microVM boundaries, read-only root filesystems,
restricted mounts, non-root users, and network/IAM controls.

Upgrade notes for ``make_filesystem_tools``
-------------------------------------------

``make_filesystem_tools(allowed_dirs)`` remains supported and returns the same
v1 tool names. Internally, path handling is stricter:

- paths are resolved canonically with ``Path.resolve()`` and must remain under
  an allowed root
- relative paths are consistently rooted at the first allowed directory
- symlink crossings and symlink escapes are rejected by default
- read and write size limits are enforced by policy
- ``read_only=True`` blocks write, append, edit, patch, mkdir, and delete tools
- deleting an allowed workspace root is refused

Projects that assert exact error text, rely on symlinked workspace paths, or
depend on multi-root relative path quirks may need small updates. Importing
``FilesystemHelpers`` directly is discouraged; use ``make_filesystem_tools`` or
``WorkspaceSession`` instead.

Guidewire Tool Factory
----------------------

Andromeda also provides an optional Guidewire tool factory in
``andromeda.tools.guidewire.make_guidewire_tools``.

Example:

.. code-block:: python

   from andromeda.tools.guidewire import make_guidewire_tools

   gw_tools = make_guidewire_tools(
     base_url="https://example-guidewire.local",
     username="su",
     password="gw",
   )

   result = gw_tools["get_claim_details"].invoke({"claim_id": "cc:123"})
   print(result)

MCP Tools
---------

MCP server tools are registered into the global tool registry during config
loading, then resolved like any other ``tools: [...]`` entry.

See :doc:`../integrations/mcp-servers` for full setup.

Common Errors and Fixes
-----------------------

- ``Unknown tool '...'`` during config load:
  Register that tool (for example via ``register_tool`` or ``register_tools``)
  before loading YAML.
- Search tools fail at runtime:
  Ensure provider credentials are set (for example ``TAVILY_API_KEY``).
- ``crawl_url`` fails with dependency error:
  Install Crawl4AI and run setup (``pip install crawl4ai`` then
  ``crawl4ai-setup``).
- Filesystem path validation errors:
  Confirm the path is inside the configured allowed directory list.
