Changelog
=========

v1.1.10 (2026-07-13)
--------------------
- **Runtime workspace snapshotting**

  - ``AndromedaRuntime.run``/``arun`` now snapshot every workspace-backed agent's files before and after a (non-dry-run) invocation and diff them, so ``RunResult.artifacts`` reflects the files a run actually touched instead of staying permanently empty (**#123**).
  - Added ``RunResult.artifact_changes``: per-file diff metadata alongside ``artifacts`` -- ``path``, ``status`` (``added``/``modified``/``removed``), and ``lines_added``/``lines_removed`` computed git-stat style. Line counts are ``0`` when a file's content couldn't be captured (binary or over the 1MB diff cap); the file is still detected as changed via mtime.
  - The snapshot walk honors each workspace root's ``.gitignore``/``.andromedaignore`` via ``IgnoreMatcher``, pruning ignored directories rather than descending into them, and always skips ``.git``.
  - Added ``AndromedaRuntime.workspace_roots()`` and ``workspace_root_for(name)`` to resolve the local filesystem root(s) of workspace-backed agents built so far.

- **CLI optional dependencies**

  - ``click``, ``rich``, and ``questionary`` moved out of core dependencies into an optional ``andromeda[cli]`` extra (**#117**). ``pip install andromeda`` no longer pulls in CLI-only packages; running ``andromeda <command>`` without them now prints a friendly ``pip install 'andromeda[cli]'`` hint instead of failing on import.
  - Fixed ``andromeda.cli`` being unimportable (and ``from andromeda.cli import cli`` breaking) when CLI deps are missing, by moving ``cli``/``commands`` behind a module-level ``__getattr__`` that builds and caches them lazily on first access (**#119**).
  - CI now installs the ``cli`` extra alongside ``retrievers``/``retrievers-faiss`` so the full CLI test suite runs.

- **Coworker names**

  - Reworked ``COWORKER_NAMES`` in ``andromeda.core.workspace`` from human names to an Andromeda-themed roster of 50 constellation, star, and deep-sky-object names from the Andromeda (M31) neighborhood (**#120**).
  - Auto-spawned coworkers are now named from a shuffled copy of the roster (``_coworker_name``), so naming order varies between runs instead of always starting from the same name (**#122**).

v1.1.9 (2026-07-07)
-------------------
- **CLI Runtime**

  - Fixed a positional prompt being silently dropped when streaming a workflow (as opposed to running it); the prompt is now folded into ``inputs['prompt']`` on both the ``run``/``stream`` paths.
  - Added ``AndromedaRuntime.arun`` and ``AndromedaRuntime.astream`` async variants mirroring ``run``/``stream``, so workflows and agents can be driven from async callers. ``RuntimeWorkflow.stream()`` still executes synchronously under the hood, so workflow calls are bridged onto a background thread rather than blocking the event loop.
  - Fixed a race condition in ``AndromedaRuntime``'s agent/workflow build caches by guarding ``build_agent``, ``build_workflow``, and ``close`` with a shared lock.
  - Deduplicated prompt-folding and agent-result-mapping logic that was repeated across the sync and async entry points into shared helpers (``_fold_prompt_into_inputs``, ``_agent_result_from_output``).

v1.1.8 (2026-07-01)
-------------------
- **Workspace Agent and Supervisor**

  - Fixed conflicting prompts between ``WorkspaceAgent`` and ``Supervisor`` (**#105**).

v1.1.7 (2026-06-29)
-------------------
- **CLI**

  - New runtime implementation for yaml based agents and workflows.

v1.1.6 (2026-06-29)

  - Added a `WorkspaceAgent` class based on `Supervisor` for running workspace session driven agents.
  - Added a `WorkspaceSession` class for managing workspace sessions. This includes filesystem and shell tools along with policies, seeding and provider settings.
  - Introduced providers: `local_fs`, `ephemeral_fs`, `postgres_vfs`, `s3_snapshot`, `bubblewrap_process`, `gvisor_container`, `microvm`. MicroVM is currently untested due to OS limitations and should be considered experimental.

v1.1.5 (2026-06-17)
-------------------
- **GitHub Copilot provider**

  - Added a ``github_copilot`` chat provider (``ChatGithubCopilot``, a ``ChatOpenAI`` subclass over Copilot's OpenAI-compatible ``/chat/completions`` API) that works with existing orchestration, tools, memory, callbacks and streaming with no caller changes.
  - Added a ``github_copilot`` embeddings provider (``GithubCopilotEmbeddings`` over Copilot's ``/embeddings`` API, e.g. ``text-embedding-3-small``) sharing the same auth and token refresh.
  - Automatic authentication from the environment (env vars, the editor's ``~/.config/github-copilot`` auth files, or a cached token); short-lived Copilot tokens are refreshed proactively and reactively. Overrides flow through ``ModelConfig.other_args`` (``github_token``, ``copilot_token``, ``base_url``, ``editor_version``, ``default_headers``, ``auto_login``).
  - Optional one-time device-flow login when no auth is discovered (on by default in an interactive terminal; controlled via ``other_args={"auto_login": ...}`` or the ``ANDROMEDA_COPILOT_AUTO_LOGIN`` env var).
  - Helpers ``device_login()``, ``list_models()`` / ``print_models()``, and ``get_copilot_limits()``.
  - Requires the optional extra: ``pip install 'andromeda[github-copilot]'``.

- **Streaming & reasoning**

  - Fixed spurious newlines being injected between streamed reasoning fragments. Reasoning models emit chain-of-thought as many delta fragments; ``langtils`` was re-joining them with ``\n`` (in both the message-normalization and observability paths). Fragments are now concatenated verbatim, matching how the litellm/OpenAI client accumulates them.

- **Removed langchain_community Dependency**

  - Removed the ``langchain_community`` dependency and replaced all usage of its components with direct implementations (FAISS/BM25/OpenSearch) or alternatives from the main LangChain library (Azure, Mongo) following the announcement of the deprecation of the ``langchain_community`` package.

v1.1.4 (2026-05-15)
-------------------

- **Skills middleware**

  - Stabilized concurrent skill state and regressions surfaced during repeated skill-loading and middleware testing (**#66**).
  - Fixed discovery being skipped when the graph or checkpoint seeded ``skills_metadata`` to an empty list; ``before_agent`` / ``abefore_agent`` rescan filesystem sources whenever metadata is missing or empty until a non-empty list is populated.
  - Unified synchronous and asynchronous skill directory scanning (``_list_skills`` / ``_alist_skills``) via ``_collect_skill_dirs``, including sources whose root contains ``SKILL.md`` directly.
  - Introduced skill-scoped ``read_skill_file`` (with built-in tooling next to ``load_skill`` and prompt updates) so agents can read supporting markdown under an activated skill without widening generic filesystem sandbox paths.

v1.1.3 (2026-05-14)
-------------------
- **Retrievers & streaming**

  - Retriever enhancements (including async retriever interfaces), multi-hop retrieval tests, and broader LangGraph graph compatibility (**#63**).
  - Streaming and inference fixes: LiteLLM reasoning parsing, flushing of streamed chunks, removal of non-standard reasoning blocks that broke downstream inference, and corrected skills usage with reasoning-capable models and tools (**#63**).

- **Skills & tools**

  - Iterated skills middleware (loading reliability, concurrent updates) and landed initial ``apply_patch`` implementation; coordinated tracing plus filesystem tooling fixes.


v1.1.2 (2026-05-01)
-------------------
- **Supervisor & async**

  - Supervisor **``async_tasks``** support and related async retriever/tooling surface fixes.

- **Streaming & resilience**

  - Langfuse/streaming observability corrections, chunked flush behavior adjustments, LiteLLM reasoning-stream handling, skills tool invocation fixes, and tool-call error-handling improvements.


v1.1.1 (2026-04-07)
-------------------
- **Retrievers, tools & platform**

  - Retrievers implementation, crawler tool, MCP isolation primitives for low-level usage, Guidewire-focused tool additions (**#55**), and LiteLLM support with AWS CodeArtifact release workflow (**#61**).

- **Documentation**

  - Sphinx documentation reorganized into logical groups (**#54**), retriever docs, and output version support metadata for docs.


v1.1.0 (2026-02-17)
-------------------
- **LangChain/LangGraph v1 Migration**

  - Completed a comprehensive migration of the codebase to fully utilize LangChain and LangGraph v1.x APIs.
  - Integrated enhanced middleware architecture: introduced a flexible middleware pipeline to support pre/post-processing, error handling, input/output masking, guardrails, and tracing capabilities across agent workflows.
  - Implemented custom middleware support, allowing users to register and chain their own middleware components.


v1.0.5 (2026-01-27)
-------------------
- Expanded support for rich streaming metadata.
- Integrated Langfuse evaluators for automated agent output evaluation and scoring.
- Basic setup for prebuilt agents and connectors.


v1.0.4 (2025-11-26)
-------------------
- Added MCP adapter for remote MCP server


v1.0.3 (2025-11-24)
-------------------
- Deleted the monolithic `cli.py` file and refactored the CLI into a modular structure: added new files for commands, configuration generation, environment variable handling, diagnostics, and output display functions.
- Introduced a more interactive setup wizard and new CLI features for diagnostics and workflow visualization.
- Updated documentation to reflect changes in tool imports: replaced prior `internet_search` and `news_articles_search` with `web_search` and `news_search`; added details on new tool registry implementation.
- Introduced a global tool registry for better management and referencing of built-in tools by name in configuration files.

v1.0.2 (2025-11-21)
-------------------
- Removed the legacy events subsystem 
- Fixed parallel node execution
- Decoupled writer/reporter responsibilities from being mandatory in a Team
- Reworked configuration to require explicit model definitions, optional validation/citation knobs, planner metadata, and CLI support for the new schema

v1.0.1 (2025-11-04)
-------------------
- Removed the bundled Tavily developer key, made the client optional when the TAVILY_API_KEY env var is missing, and fail fast if search is invoked without credentials

v1.0.0 (2025-11-04)
-------------------
- Initial release of Andromeda with the CLI entry point, workflow builder/orchestration primitives, and packaging metadata 
- Extracted the Agentify multi-agent runtime into the andromeda package, providing supervisor, team, planner, writer, reporting, tools, prompts, schemas, and utilities plus an installation script
- Added streaming and structured-output improvements, Langfuse integration, langtils helpers, logging updates, docs, sample workflows, GitHub Actions tests
