Using Configuration
===================

Define your agents, supervisor, planner, and report in a YAML file and load it with ``AndromedaConfig.load_from_file`` for clean separation of config and code. Tools can be referenced by name if they're registered in the global tool registry.

This page covers three practical patterns:

- Loading from a YAML file.
- Loading from inline YAML (string in Python).
- Handling common loading errors cleanly.

**config.yaml**

.. code-block:: yaml

   agents:
     researcher:
       name: researcher
       model:
         name: llama3.1:8b
         provider: litellm
         other_args:
           base_url: http://localhost:11434
       tools:
         - web_search          # Built-in tools referenced by name
         - news_search
       checkpointer:
         backend: in-memory
   supervisor:
     name: supervisor
     model:
       name: llama3.1:8b
       provider: litellm
     enable_planning: true
     allow_async_tasks: false
     checkpointer:
       backend: in-memory
   planner:
     model:
       name: llama3.1:8b
       provider: litellm
     task_type: research
   report:
     enabled: true
     model:
       name: llama3.1:8b
       provider: litellm
     format: markdown
     output_mode: state

**Loading and running**

When loading from YAML, built-in tools are automatically registered. String tool names in the config are resolved to actual tool instances:

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.core.team import Team

   cfg = AndromedaConfig.load_from_file("config.yaml")
   # Tools specified as strings in YAML are now resolved to BaseTool instances
   team = Team(cfg)
   result = team.begin("Summarize the competitive landscape for open-source LLM orchestration tools.")
   print(result.get("report_output") or result["messages"][-1].content)

How loading works internally
----------------------------

``AndromedaConfig.load_from_file(...)`` parses YAML/JSON, then normalizes and validates it.
During normalization, Andromeda can:

- Resolve tool names (for example, ``web_search``) to real tool instances.
- Interpolate environment variables in strings using ``${VAR_NAME}`` syntax.
- Register MCP servers and expose MCP tools (if ``mcp_servers`` is present).

If validation fails, you get a descriptive ``ValueError`` that includes context.

Checkpointer Configuration
--------------------------

Agents and supervisors can declare LangGraph persistence settings with
``checkpointer``. The default is ``in-memory`` for backward compatibility.
Use ``none`` to disable persistence, or ``postgres`` with a connection string
when persistent checkpoints are required.

.. code-block:: yaml

   checkpointer: in-memory

   # Equivalent object form:
   checkpointer:
     backend: postgres
     connection_string: ${DATABASE_URL}
     setup: false

Postgres support is optional; install the ``checkpointer-postgres`` extra before
using ``backend: postgres``. This is separate from Andromeda workflow
``.checkpoint(...)`` nodes, which are human-interrupt points in a workflow.

Inline YAML (In-Memory Config)
------------------------------

Use inline YAML when you want dynamic configuration without creating a file first,
for example in tests, notebooks, or generated workflows.

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.config.yaml_utils import yaml_load

   inline_yaml = """
   agents:
     researcher:
       name: researcher
       model:
         name: qwen3:8b
         provider: litellm
       tools:
         - web_search
   supervisor:
     name: supervisor
     model:
       name: qwen3:8b
       provider: litellm
   planner:
     model:
       name: qwen3:8b
       provider: litellm
   report:
     enabled: true
     model:
       name: qwen3:8b
       provider: litellm
   """

   data = yaml_load(inline_yaml)
   cfg = AndromedaConfig.load_from_config(data)
   print(type(cfg).__name__)  # AndromedaConfig

Advanced inline mode (normalize only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want normalized data without full top-level validation, use ``strict=False``:

.. code-block:: python

   from andromeda.config import AndromedaConfig
   from andromeda.config.yaml_utils import yaml_load

   partial_yaml = """
   supervisor:
     name: supervisor
     model:
       name: qwen3:8b
       provider: litellm
   """

   data = yaml_load(partial_yaml)
   normalized = AndromedaConfig.load_from_config(data, strict=False)
   print(type(normalized))  # dict
   print(normalized.keys())

Environment Variable Interpolation
----------------------------------

Andromeda supports ``${VAR_NAME}`` interpolation while loading config.

.. code-block:: yaml

   agents:
     researcher:
       name: researcher
       model:
         name: qwen3:8b
         provider: openai
         other_args:
           api_key: ${OPENAI_API_KEY}
           base_url: ${OPENAI_BASE_URL}

If a referenced variable is missing, loading raises a clear error with:

- The source file or config origin.
- The missing variable name.
- The location in the config tree.
- The exact expression (for example, ``${OPENAI_API_KEY}``).

Error Handling Patterns
-----------------------

Production code should catch and surface config errors early at startup.

Common failure cases:

- File not found (``FileNotFoundError``).
- Unsupported extension (must be ``.json``, ``.yml``, or ``.yaml``).
- Unknown tool names in ``tools`` list.
- Missing required sections/fields in strict mode.
- Missing environment variables used in ``${VAR}`` placeholders.

Recommended startup loader
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from andromeda.config import AndromedaConfig

   def load_config_or_exit(path: str) -> AndromedaConfig:
       try:
           return AndromedaConfig.load_from_file(path)
       except FileNotFoundError as exc:
           raise SystemExit(f"Config file missing: {exc}")
       except ValueError as exc:
           # Covers schema validation, unknown tools, bad extensions,
           # and missing env vars during interpolation.
           raise SystemExit(f"Invalid config: {exc}")

Unknown tool example
~~~~~~~~~~~~~~~~~~~~

If YAML references a tool that is not registered, loading fails with a clear message
that includes the tool name and owning agent.

.. code-block:: yaml

   agents:
     researcher:
       name: researcher
       model:
         name: fake-model
         provider: litellm
       tools:
         - this_tool_does_not_exist_123

.. code-block:: python

   from andromeda.config import AndromedaConfig

   try:
       AndromedaConfig.load_from_file("config_invalid_tools.yaml")
   except ValueError as exc:
       print(exc)
       # Message includes unknown tool id and agent context.

Best Practices
--------------

- Keep one canonical ``config.yaml`` for production.
- Use inline YAML for tests and generated configs.
- Fail fast on startup by loading and validating once.
- Prefer explicit tool registration for custom tools before loading.
- Use ``strict=False`` only when intentionally working with partial configs.

**Using custom tools in YAML:**

If you have custom tools, register them before loading the config:

.. code-block:: python

   from andromeda.tools import tool
   from andromeda.tools.toolkit import register_tool
   from andromeda.config import AndromedaConfig

   @tool
   def my_custom_tool(query: str) -> str:
       """Custom tool for specific processing."""
       return f"Custom processing: {query}"

   # Register before loading config
   register_tool(my_custom_tool)

   # Now use it in config.yaml:
   # tools:
   #   - my_custom_tool

   cfg = AndromedaConfig.load_from_file("config.yaml")

   # If not using config.yaml, this step is not needed
