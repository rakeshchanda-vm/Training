Command Line Interface (CLI)
============================

The Andromeda CLI is a comprehensive development and setup tool that helps developers get started with the Andromeda multi-agent framework. It provides configuration management, setup utilities, system diagnostics, and development tools.
Note: This is a work in progress and is in pre-alpha.

Installation
------------

The CLI's dependencies (``click``, ``rich``, ``questionary``) are optional and not installed
by a plain ``pip install andromeda``. Install the ``cli`` extra to use it:

.. code-block:: bash

   pip install "andromeda[cli]"

Without the extra, ``andromeda`` and ``python -m andromeda.cli`` print a reminder to install
it rather than failing with an import error.

Once installed, access the CLI using:

.. code-block:: bash

   python -m andromeda.cli --help

Or if the package is installed:

.. code-block:: bash

   andromeda --help

Getting Help
------------

To see all available commands and options:

.. code-block:: bash

   andromeda --help

For help with a specific command:

.. code-block:: bash

   andromeda <command> --help

Commands Overview
-----------------

Config Runtime Commands
^^^^^^^^^^^^^^^^^^^^^^^

Andromeda can execute agents and workflows defined in ``.andromeda`` files.
Definitions can live in the current project at ``./.andromeda`` or globally at
``~/.andromeda``.

Runtime discovery looks for:

* ``.andromeda/agents/*.yaml|yml|json``
* ``.andromeda/workflows/*.yaml|yml|json``
* bundled workflows at ``.andromeda/workflows/<name>/workflow.yaml|yml``

Project definitions and global definitions are both loaded by default. If a
project and global definition have the same name, Andromeda exposes suffixed
names such as ``review--project`` and ``review--global``. Otherwise the plain
name is used.

run
"""

Execute an agent or workflow from ``.andromeda`` definitions.

.. code-block:: bash

   andromeda run [OPTIONS] NAME [PROMPT]

**Options:**

* ``--kind``: Force ``agent`` or ``workflow`` when names collide
* ``--input``: JSON object of runtime inputs
* ``--input-file``: Path to a JSON file with runtime inputs
* ``--state-file``: Optional workflow state file to read before execution and write after execution
* ``--root``: Project root for ``.andromeda`` discovery
* ``--global-root``: Override the global config root
* ``--no-global``: Exclude ``~/.andromeda`` definitions
* ``--dry-run``: Run validation-only mode
* ``--json``: Emit result as JSON
* ``--raw``: Include verbose raw fields in JSON output
* ``--stream``: Stream execution progress as events

**Examples:**

.. code-block:: bash

   # Run an agent with a positional prompt
   andromeda run code-review "Review the latest changes"

   # Run a workflow with structured inputs
   andromeda run analyze --kind workflow --input '{"prompt":"Quick scan"}'

   # Stream workflow progress as JSON events
   andromeda run analyze --kind workflow --input '{"prompt":"Quick scan"}' --json --stream

   # Persist workflow state between runs
   andromeda run analyze --kind workflow --state-file .andromeda/state/analyze.json --input '{"prompt":"Scan auth changes"}'

   # Use only project-local definitions
   andromeda run analyze --no-global --input '{"prompt":"Quick scan"}'

``--stream`` currently does not support ``--state-file`` or ``--dry-run``.

list
""""

List discovered runtime agents and workflows.

.. code-block:: bash

   andromeda list [OPTIONS]

**Options:**

* ``--kind``: Filter by ``agent`` or ``workflow``
* ``--json``: Emit result as JSON
* ``--root``: Project root for ``.andromeda`` discovery
* ``--global-root``: Override the global config root
* ``--no-global``: Exclude ``~/.andromeda`` definitions

**Examples:**

.. code-block:: bash

   andromeda list
   andromeda list --kind agent
   andromeda list --json --no-global

inspect
"""""""

Show resolved metadata for a runtime definition.

.. code-block:: bash

   andromeda inspect [OPTIONS] NAME

**Options:**

* ``--kind``: Filter by ``agent`` or ``workflow`` when names collide
* ``--json``: Emit result as JSON
* ``--root``: Project root for ``.andromeda`` discovery
* ``--global-root``: Override the global config root
* ``--no-global``: Exclude ``~/.andromeda`` definitions

**Examples:**

.. code-block:: bash

   andromeda inspect code-review
   andromeda inspect analyze --kind workflow --json
   andromeda inspect review--global --kind agent

validate
""""""""

Validate runtime agent and workflow definitions.

.. code-block:: bash

   andromeda validate [OPTIONS] [NAME]

**Options:**

* ``--kind``: Validate a specific kind, ``agent`` or ``workflow``
* ``--json``: Emit result as JSON
* ``--root``: Project root for ``.andromeda`` discovery
* ``--global-root``: Override the global config root
* ``--no-global``: Exclude ``~/.andromeda`` definitions

**Examples:**

.. code-block:: bash

   # Validate all project and global definitions
   andromeda validate

   # Validate only project-local definitions
   andromeda validate --no-global

   # Validate a single workflow and emit machine-readable output
   andromeda validate analyze --kind workflow --json

Workflow Python Trust Boundary
""""""""""""""""""""""""""""""

Workflow ``function`` nodes load Python modules from the workflow directory.
Module paths are resolved relative to the workflow bundle and cannot use
absolute paths or escape that directory, but import-time Python code still runs.
Only run workflows from trusted project or global ``.andromeda`` roots.

generate-config
^^^^^^^^^^^^^^^

Generate example configuration files (config.yaml and/or config.json).

.. code-block:: bash

   andromeda generate-config [OPTIONS]

**Options:**

* ``--format, -f``: Output format for config files (yaml, json, or both). Default: yaml
* ``--output-dir, -o``: Output directory for generated files. Default: current directory
* ``--interactive, -i``: Use interactive prompts to customize configuration

**Examples:**

.. code-block:: bash

   # Generate YAML configuration
   andromeda generate-config

   # Generate both YAML and JSON configurations
   andromeda generate-config --format both

   # Generate in specific directory
   andromeda generate-config --output-dir ./config

   # Interactive configuration generation
   andromeda generate-config --interactive

generate-env
^^^^^^^^^^^^

Generate example .env file with all environment variables.

.. code-block:: bash

   andromeda generate-env [OPTIONS]

**Options:**

* ``--output-dir, -o``: Output directory for generated .env file. Default: current directory
* ``--interactive, -i``: Use interactive prompts to customize environment variables
* ``--include-pyeztrace``: Include PyEzTrace environment variables. Default: true
* ``--include-optional``: Include optional environment variables. Default: true

**Examples:**

.. code-block:: bash

   # Generate basic .env file
   andromeda generate-env

   # Generate without PyEzTrace variables
   andromeda generate-env --include-pyeztrace=false

   # Generate minimal configuration
   andromeda generate-env --include-optional=false

validate-config
^^^^^^^^^^^^^^^

Validate configuration files for correctness and completeness.

.. code-block:: bash

   andromeda validate-config [OPTIONS] [CONFIG_FILE]

**Options:**

* ``--format, -f``: Format of the config file to validate (yaml or json)

**Arguments:**

* ``CONFIG_FILE``: Path to configuration file to validate (optional)

If no config file is specified, the command will look for configuration files in the current directory.

**Examples:**

.. code-block:: bash

   # Validate specific config file
   andromeda validate-config ./config.yaml

   # Auto-detect config file in current directory
   andromeda validate-config

   # Validate JSON format explicitly
   andromeda validate-config --format json config.json

setup
^^^^^

Interactive setup wizard for new projects.

.. code-block:: bash

   andromeda setup

This command guides you through the initial setup process, helping you:

* Set up project structure and directories
* Generate configuration files
* Create .env.example file
* Set up requirements.txt
* Configure project scaffolding

**Examples:**

.. code-block:: bash

   # Run interactive setup wizard
   andromeda setup

diagnose
^^^^^^^^

Run system diagnostics to check setup and dependencies.

.. code-block:: bash

   andromeda diagnose [OPTIONS]

**Options:**

* ``--check-deps``: Check system dependencies
* ``--test-connections``: Test external service connections
* ``--check-env``: Check environment setup

If no options are specified, all diagnostic checks will be performed.

**Examples:**

.. code-block:: bash

   # Run all diagnostic checks
   andromeda diagnose

   # Check only dependencies
   andromeda diagnose --check-deps

   # Test only service connections
   andromeda diagnose --test-connections

   # Check only environment setup
   andromeda diagnose --check-env

show-config-options
^^^^^^^^^^^^^^^^^^^

Display all available configuration options with descriptions.

.. code-block:: bash

   andromeda show-config-options

This command shows a comprehensive table of all configuration options available in the Andromeda framework, including:

* Agent configuration options
* Model configuration parameters
* Validation settings
* Citation requirements
* Supervisor, planner, report, and event configurations

show-env-vars
^^^^^^^^^^^^^

Display all environment variables with descriptions.

.. code-block:: bash

   andromeda show-env-vars

This command shows a comprehensive table of all environment variables used by the Andromeda framework and its dependencies, including:

* Core Andromeda variables (required)
* PyEzTrace configuration variables
* Optional development and testing variables
* Model provider configurations
* LangSmith tracing variables

Configuration File Structure
----------------------------

The Andromeda configuration is organized into several main sections:

agents
^^^^^^

List of agent configurations. Each agent has:

* ``name``: Unique identifier for the agent
* ``model``: Model configuration for the agent
* ``tools``: List of tools available to the agent
* ``prompt``: Custom prompt for the agent
* ``validation``: Validation settings
* ``citations``: Citation requirements
* ``return_direct``: Whether to return results directly
* ``next``: Next agent in the chain
* ``type``: Agent type (react or codeact)

Model Configuration
^^^^^^^^^^^^^^^^^^^

Model settings include:

* ``name``: Model name/identifier
* ``provider``: Model provider (ollama, openai, anthropic, bedrock, azure_openai, litellm, github_copilot, openai_codex)
* ``base_url``: Base URL for API access
* ``temperature``: Sampling temperature (0.0-1.0)
* ``context_window``: Maximum context window size
* ``other_args``: Additional provider-specific arguments

Validation Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Validation settings control quality checks:

* ``enabled``: Enable/disable validation
* ``model``: Model used for validation
* ``skip_after_attempts``: Number of failures before skipping validation
* ``min_sufficiency_score``: Minimum score required for validation

Citation Configuration
^^^^^^^^^^^^^^^^^^^^^^

Citation requirements define reference standards:

* ``required``: Whether citations are mandatory
* ``min_density``: Minimum citation density as a fraction
* ``require_reference_section``: Whether to require a references section

Environment Variables
---------------------

Core Variables (Required)
^^^^^^^^^^^^^^^^^^^^^^^^^

* ``TAVILY_API_KEY``: API key for Tavily search service

PyEzTrace Variables (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Logging Configuration:

* ``EZTRACE_LOG_FORMAT``: Logging format (json, color, plain, csv, logfmt)
* ``EZTRACE_LOG_LEVEL``: Logging level (DEBUG, INFO, WARNING, ERROR)
* ``EZTRACE_LOG_FILE``: Path to log file
* ``EZTRACE_MAX_SIZE``: Maximum log file size in bytes
* ``EZTRACE_BACKUP_COUNT``: Number of backup log files to keep

OpenTelemetry Configuration:

* ``EZTRACE_OTEL_ENABLED``: Enable OpenTelemetry tracing (true/false)
* ``EZTRACE_OTEL_EXPORTER``: Tracing exporter (otlp, console, s3, azure)
* ``EZTRACE_OTLP_ENDPOINT``: OTLP endpoint for tracing
* ``EZTRACE_OTLP_HEADERS``: Headers for OTLP (key=value,key=value format)
* ``EZTRACE_SERVICE_NAME``: Service name for tracing

Export Configuration:

* ``EZTRACE_S3_BUCKET``: S3 bucket for trace exports
* ``EZTRACE_S3_PREFIX``: S3 key prefix for traces
* ``EZTRACE_S3_REGION``: AWS region for S3
* ``EZTRACE_COMPRESS``: Compress trace files (true/false)
* ``EZTRACE_AZURE_CONTAINER``: Azure container for trace exports
* ``EZTRACE_AZURE_PREFIX``: Azure blob prefix for traces
* ``EZTRACE_AZURE_CONNECTION_STRING``: Azure storage connection string
* ``EZTRACE_AZURE_ACCOUNT_URL``: Azure storage account URL

Optional Variables
^^^^^^^^^^^^^^^^^^

Development Variables:

* ``PYTHONPATH``: Python path for development
* ``DEBUG``: Enable debug mode (true/false)

Model Provider Configurations:

* ``OPENAI_API_KEY``: OpenAI API key
* ``ANTHROPIC_API_KEY``: Anthropic API key
* ``AWS_ACCESS_KEY_ID``: AWS access key for Bedrock
* ``AWS_SECRET_ACCESS_KEY``: AWS secret key for Bedrock
* ``AWS_REGION``: AWS region
* ``AZURE_OPENAI_ENDPOINT``: Azure OpenAI endpoint
* ``AZURE_OPENAI_API_KEY``: Azure OpenAI API key
* ``GITHUB_COPILOT_TOKEN``: GitHub Copilot token OR ``GITHUB_TOKEN``: GitHub OAuth/PAT token

LangSmith Configuration:

* ``LANGSMITH_API_KEY``: LangSmith API key for tracing
* ``LANGSMITH_PROJECT``: LangSmith project name
* ``LANGSMITH_ENDPOINT``: LangSmith endpoint

Development Server:

* ``OLLAMA_HOST``: Ollama server URL

Examples
--------

Basic Configuration Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate default configuration
   andromeda generate-config

   # Generate with interactive prompts
   andromeda generate-config --interactive

   # Generate in specific directory
   mkdir my-project
   cd my-project
   andromeda generate-config --output-dir .

Environment Setup
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate environment template
   andromeda generate-env

   # Generate minimal environment (no optional variables)
   andromeda generate-env --include-optional=false

   # Generate without PyEzTrace variables
   andromeda generate-env --include-pyeztrace=false

Project Setup
^^^^^^^^^^^^^

.. code-block:: bash

   # Create complete project structure
   andromeda setup

   # This will:
   # 1. Ask for project name
   # 2. Create directory structure
   # 3. Generate configuration files
   # 4. Create .env.example
   # 5. Set up requirements.txt

System Diagnostics
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run all diagnostic checks
   andromeda diagnose

   # Check only dependencies
   andromeda diagnose --check-deps

   # Test only service connections
   andromeda diagnose --test-connections

Configuration Validation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Validate existing configuration
   andromeda validate-config

   # Validate specific file
   andromeda validate-config path/to/config.yaml

   # Validate JSON format
   andromeda validate-config --format json config.json

Getting Help
^^^^^^^^^^^^

.. code-block:: bash

   # Show all available commands
   andromeda --help

   # Get help for specific command
   andromeda generate-config --help
   andromeda setup --help
   andromeda diagnose --help

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**ModuleNotFoundError when running CLI**

Make sure you have all required dependencies installed:

.. code-block:: bash

   pip install rich click questionary pyyaml

**Configuration validation fails**

Check that your configuration file follows the correct structure:

.. code-block:: bash

   andromeda validate-config --help

**Interactive prompts not working**

The CLI will fall back to basic input prompts if the ``questionary`` library is not installed. For enhanced interactive experience:

.. code-block:: bash

   pip install questionary

**Environment variables not being recognized**

Ensure your .env file is in the correct location and properly formatted:

.. code-block:: bash

   # Check environment setup
   andromeda diagnose --check-env

Best Practices
--------------

Project Organization
^^^^^^^^^^^^^^^^^^^^

When setting up a new project:

1. Create a dedicated project directory
2. Run the setup wizard: ``andromeda setup``
3. Configure your .env file with actual API keys
4. Customize the generated configuration files
5. Test your setup: ``andromeda diagnose``

Configuration Management
^^^^^^^^^^^^^^^^^^^^^^^^

1. Use ``generate-config --interactive`` for customized configurations
2. Always validate configurations: ``andromeda validate-config``
3. Keep .env.example files in version control
4. Use .env files for sensitive configuration (keep out of version control)
5. Document custom configurations in your project README

Environment Setup
^^^^^^^^^^^^^^^^^

1. Start with ``generate-env`` to create .env.example
2. Copy .env.example to .env and fill in actual values
3. Use ``diagnose --check-env`` to verify setup
4. Test service connections: ``diagnose --test-connections``

Development Workflow
^^^^^^^^^^^^^^^^^^^^

1. Use ``show-config-options`` and ``show-env-vars`` for reference
2. Generate configurations with ``generate-config``
3. Test and validate with ``validate-config`` and ``diagnose``
4. Use ``setup`` for scaffolding new projects

Extending the CLI
-----------------

Adding Custom Commands
^^^^^^^^^^^^^^^^^^^^^^

The CLI is built with Click and can be extended with custom commands. See the contributing guide for details on adding new CLI functionality.

Plugin System
^^^^^^^^^^^^^

The CLI supports a plugin architecture. Custom tools and commands can be added through the plugin system.

Integration with Other Tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CLI integrates with:

* **Rich**: For beautiful terminal formatting
* **Click**: For command-line argument parsing
* **Questionary**: For interactive prompts (optional)
* **PyYAML**: For YAML configuration handling
* **PyEzTrace**: For logging and tracing integration

Advanced Usage
--------------

Scripting with the CLI
^^^^^^^^^^^^^^^^^^^^^^

The CLI can be used in scripts and automation:

.. code-block:: python

   import subprocess
   import sys

   def generate_config(output_dir):
       """Generate configuration using CLI"""
       result = subprocess.run([
           sys.executable, "-m", "andromeda.cli", "generate-config",
           "--output-dir", output_dir, "--format", "yaml"
       ], capture_output=True, text=True)

       return result.returncode == 0, result.stdout, result.stderr

   # Usage
   success, stdout, stderr = generate_config("./config")
   if success:
       print("Configuration generated successfully")
   else:
       print(f"Error: {stderr}")

Batch Operations
^^^^^^^^^^^^^^^^

Multiple CLI commands can be chained for batch operations:

.. code-block:: bash

   # Setup complete project
   andromeda setup && \
   andromeda diagnose && \
   andromeda validate-config

CI/CD Integration
^^^^^^^^^^^^^^^^^

The CLI can be integrated into CI/CD pipelines:

.. code-block:: yaml

   # GitHub Actions example
   - name: Validate Configuration
     run: |
       python -m andromeda.cli validate-config --format yaml config.yaml

   - name: Run Diagnostics
     run: |
       python -m andromeda.cli diagnose --check-deps

   - name: Generate Environment Template
     run: |
       python -m andromeda.cli generate-env --include-optional=false

Contributing
------------

To contribute to the CLI:

1. Fork the repository
2. Create a feature branch
3. Add your CLI commands or features
4. Test thoroughly: ``python -m andromeda.cli --help``
5. Submit a pull request

For detailed contribution guidelines, see the contributing documentation.
