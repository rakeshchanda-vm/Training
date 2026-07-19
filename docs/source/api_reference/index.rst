API Reference
=============

.. toctree::
   :maxdepth: 2

   andromeda

Module Overview
---------------

**andromeda.config**
    Configuration models for agents, supervisor, planner, and reports.

**andromeda.core**
    Core logic for agents, teams, planning, and workflow execution.

**andromeda.reporting**
    Report generation, citation extraction, and diagram support.

**andromeda.runtime**
    Runtime for YAML-defined agents and workflows: discovery/registry, agent/workflow
    building, and workspace file snapshotting (``AndromedaRuntime.run``/``arun``
    diff a workspace-backed agent's files before/after a run into
    ``RunResult.artifacts``/``artifact_changes``).

**andromeda.tools**
    Tools, MCP adapter and context management.

**andromeda.utils**
    Utility functions, prompts, schemas, and logging.


For detailed documentation, see the respective module pages above.