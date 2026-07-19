Workspace Agents
================

Use ``WorkspaceAgent`` when you need a single, long-horizon agent that carries a
task end to end inside an **isolated workspace session** -- reading and editing
files, running shell commands, and tracking a plan across many iterations.

A ``WorkspaceAgent`` extends :doc:`Supervisor <supervisors>`, so on top of being
able to act directly inside its workspace it can also route sub tasks to
specialist agents. On top of a supervisor it adds:

- **A workspace session** that provisions the concrete tools it works with
  (filesystem, shell, ...). The session is configurable and truly isolated; its
  backend can be a sandbox (``bubblewrap_process``, ``gvisor_container``), a
  managed local filesystem (``ephemeral_fs`` / ``local_fs``), ``postgres_vfs``,
  a microVM, and more, and it may be read-only. See
  :ref:`Workspace sessions <workspace-sessions>` for how sessions are built.
- **Long-horizon middleware** (todo/plan tracking plus context editing) so it
  can stay on task across many iterations.
- **Optional skills middleware** and MCP-provided tools, resolved through the
  standard config tool resolution.

When to Use a Workspace Agent
-----------------------------

Use a workspace agent when you want:

- a coding-agent-style loop: inspect a project, edit files, run tests, iterate
- work to happen inside an isolated, configurable sandbox rather than on the host
- long-horizon planning that survives many tool calls
- the option to delegate concrete steps to a team of coworkers
- a zero-config path that auto-creates and cleans up its own workspace

If you only need filesystem/shell tools bound to a directory (without the
long-horizon agent loop), use a :ref:`WorkspaceSession <workspace-sessions>`
directly. If you need a coordinator over several specialist agents without a
workspace, use a :doc:`Supervisor <supervisors>` or :doc:`Team <teams>`.

Quick Start
-----------

The simplest path is zero-config: omit the ``session`` and let the agent
auto-create (and clean up) its own workspace. Using it as a context manager
guarantees the session is released when you are done.

.. code-block:: python

   from andromeda.core import WorkspaceAgent
   from andromeda.config.config import WorkspaceAgentConfig
   from andromeda.config import ModelConfig

   config = WorkspaceAgentConfig(
       name="builder_agent",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   # The agent auto-creates a workspace session (a bubblewrap sandbox when the
   # host supports it, otherwise ephemeral_fs) and tears it down on exit.
   with WorkspaceAgent(config) as agent:
       report = agent.run(
           "The workspace is empty. Create a CLI tool `wc_tool.py` that prints "
           "the line, word, and character counts of a file, add a unittest suite "
           "under `tests/`, run `python3 -m unittest discover -v`, and make sure "
           "it passes. Report the files you created and the final test output."
       )
       print(report)

``run`` drives the task to completion and returns the agent's final report
(usually a string). An async variant, ``arun``, is also available.

Bringing Your Own Session
-------------------------

When you need control over the backend, seeding, or policy, build a
:ref:`WorkspaceSession <workspace-sessions>` yourself and pass it in. The caller
then owns the session lifecycle (the agent will not clean up a session it did not
create).

.. code-block:: python

   from andromeda.core import WorkspaceAgent
   from andromeda.config.config import WorkspaceAgentConfig
   from andromeda.config import ModelConfig
   from andromeda.workspace import (
       DirectorySeed,
       ShellPolicy,
       WorkspacePolicy,
       WorkspaceSession,
   )

   # Seed an existing project into an isolated sandbox.
   session = WorkspaceSession.create(
       backend="bubblewrap_process",
       seed=DirectorySeed(source_dir="./my_project"),
       policy=WorkspacePolicy(
           read_only=False,
           enable_shell=True,
           shell=ShellPolicy(
               network_enabled=False,
               timeout_seconds=60,
               allowed_commands=("python3", "pytest", "rg", "cat", "ls"),
           ),
       ),
   )

   config = WorkspaceAgentConfig(
       name="fixer_agent",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
   )

   agent = WorkspaceAgent(config, session=session)
   try:
       report = agent.run(
           "Running `python3 -m unittest discover -v` fails. Investigate, fix the "
           "bugs in the source (do not weaken the tests), and iterate until the "
           "whole suite passes. Report each bug and paste the final test output."
       )
       print(report)
   finally:
       session.cleanup()  # you own the session you created

A **read-only** session (``WorkspacePolicy(read_only=True)``) withholds the
write/edit/shell tools, so the same agent can be used for analysis tasks where
it may inspect but not modify the project.

.. _workspace-team:

The Team and Coworkers
----------------------

A workspace agent always runs with a team. If you supply fewer than the minimum
team size (``WorkspaceAgent.MIN_AGENTS``, 4 by default) specialist agents, it
spins up *coworker* agents to fill the team out to that minimum; if you supply
that many or more, it adds one extra coworker to assist in parallel. Coworkers
share the workspace session's tools, so the supervisor can fan concrete work out
across them.

.. code-block:: python

   from andromeda.config import AgentConfig, ModelConfig

   reviewer = AgentConfig(
       name="reviewer",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
       prompt="Review diffs for correctness and edge cases.",
   )

   # Routes to `reviewer` plus auto-created coworkers (padded to the minimum team).
   agent = WorkspaceAgent(config, agents=[reviewer])

The minimum defaults to 4. Pass ``min_agents`` to the constructor to tune it for
a single instance (it must be a positive integer and overrides
``WorkspaceAgent.MIN_AGENTS`` for that instance only, leaving the class default
untouched):

.. code-block:: python

   # Smaller team: lead plus coworkers padded up to 2 agents total.
   agent = WorkspaceAgent(config, min_agents=2)

Because a workspace agent is a supervisor, the supervisor controls apply:
enable ``allow_parallel_agents`` to fan work out concurrently, restrict
``allowed_route_types``, and so on (see :doc:`supervisors`).

Configuration
-------------

``WorkspaceAgentConfig`` extends ``SupervisorConfig``, so it reuses every
core agent and supervisor setting (model, tools, routing, ``enable_planning``,
``allow_parallel_agents``, debug, validation, ...) and adds workspace-specific
fields:

- ``workspace_backend`` (str, default ``"auto"``): backend used when the agent
  auto-creates a session. ``"auto"`` prefers the ``bubblewrap_process`` sandbox
  and falls back to ``"ephemeral_fs"`` (no isolation) when bubblewrap is
  unavailable on the host (see :ref:`Isolation <workspace-isolation>`).
- ``workspace_root`` (Optional[str]): filesystem root for an auto-created
  session. When ``None``, the session manager picks a managed location under the
  agent home.
- ``coworker_tools`` (List[BaseTool], default empty): tools handed to the
  auto-spawned coworker agents. These are merged with the coworker's workspace
  session tools. When empty, coworkers receive only the session tools selected by
  ``coworker_tool_profile``.
- ``coworker_tool_profile`` (``"default"`` | ``"read_only"``, default
  ``"default"``): workspace tool profile for auto-spawned coworkers.
  ``"default"`` gives coworkers the same session tools as the supervisor;
  ``"read_only"`` gives coworkers read-only file tools for the same session and
  disables their shell tool.
- ``skill_sources`` (Optional[List[str]]): skill source paths (e.g.
  ``['/skills']``). When provided, the :doc:`skills middleware
  <../integrations/skills>` is attached so the agent can discover and load skills
  from these paths; when ``None`` it is not added.
- ``skills_backend`` (``"filesystem"`` | ``"in-memory"``, default
  ``"filesystem"``): backend for the skills middleware.

The constructor also accepts keyword-only arguments that apply when the agent
**auto-creates** a session (they are ignored when you pass your own ``session``):

- ``min_agents`` (Optional[int]): minimum team size for this instance only (see
  :ref:`The Team and Coworkers <workspace-team>`).
- ``policy``: a ``WorkspacePolicy`` for the auto-created session. When ``None``, a
  shell-enabled default policy is used. ``ShellPolicy.allowed_commands`` can be
  set to a tuple of executable names to enforce an explicit shell command
  allowlist; ``ShellPolicy.denied_commands`` defaults to blocking privileged
  sandbox escape tools such as ``sudo``, ``docker``, and ``mount``.
- ``settings``: backend-specific settings (e.g. ``BubblewrapProcessSettings``).
  When ``None``, the sandbox backends with a fully defaulted settings object --
  ``bubblewrap_process`` and ``gvisor_container`` -- have their defaults filled
  in, and the filesystem backends need none; others
  must be given settings explicitly.

.. _workspace-isolation:

Isolation and the ``auto`` Backend
----------------------------------

When auto-creating a session, the default ``workspace_backend="auto"`` prefers an
isolated sandbox: it uses ``bubblewrap_process`` when the host supports it and
otherwise falls back to ``ephemeral_fs`` -- a managed local workspace with **no
sandbox isolation**, where shell and filesystem tools run directly on the host.

That fallback is a real loss of containment. If your deployment requires
isolation, name the backend explicitly instead of relying on ``"auto"``.

.. code-block:: python

   config = WorkspaceAgentConfig(
       name="builder_agent",
       model=ModelConfig(name="gpt-oss:20b", provider="litellm"),
       workspace_backend="bubblewrap_process",
   )
   agent = WorkspaceAgent(config)

Lifecycle
---------

The agent only owns -- and therefore only cleans up -- a session it
auto-created. A session you pass in stays yours to release.

Prefer the context manager (or call ``close()``) so an auto-created session is
released deterministically:

.. code-block:: python

   with WorkspaceAgent(config) as agent:   # auto-created session
       agent.run("...")
   # session released here

   # Equivalent explicit form:
   agent = WorkspaceAgent(config)
   try:
       agent.run("...")
   finally:
       agent.close()

If an owning agent is dropped without ``close()``, the session is released on
garbage collection as a safety net and a ``ResourceWarning`` is emitted, so a
live sandbox/process is never orphaned.

One Conversation per Agent
--------------------------

A single ``WorkspaceAgent`` instance models one conversation. Its ``memory`` and
``plan`` are instance-level and are **not** partitioned by ``thread_id``, so
passing different ``thread_id`` values to ``run``/``arun`` does not isolate
separate conversations -- they share and accumulate the same history (the
``thread_id`` and ``metadata`` are still forwarded to the model invocation for
checkpointing/tracing). Concurrent ``run``/``arun`` calls on the same instance
are serialized by an internal lock rather than allowed to race. For independent conversations, use a separate ``WorkspaceAgent`` per conversation.

See Also
--------

- :ref:`Workspace sessions <workspace-sessions>` -- the isolated session that
  provisions the agent's filesystem/shell tools.
- :doc:`supervisors` -- the routing/orchestration behavior a workspace agent
  inherits.
- :doc:`../integrations/skills` -- discovering and loading skills.
