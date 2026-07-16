from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from andromeda.config.config import WorkspaceAgentConfig, _normalize_config_object
from andromeda.core import WorkspaceAgent

from .context import RuntimeContext, _deep_merge


@dataclass
class RuntimeAgentSpec:
    name: str
    path: Path
    payload: Dict[str, Any]


def _as_mapping(payload: Any) -> Dict[str, Any]:
    return dict(payload) if isinstance(payload, dict) else {}


def _apply_aliases(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Apply minimal compatibility aliases for runtime agent files."""

    data = dict(raw)

    if "workspace" in data and isinstance(data["workspace"], dict):
        workspace = dict(data.pop("workspace"))
        if "backend" in workspace and "workspace_backend" not in data:
            data["workspace_backend"] = workspace["backend"]
        if "root" in workspace and "workspace_root" not in data:
            data["workspace_root"] = workspace["root"]
        if "read_only" in workspace and "read_only" not in data:
            data["read_only"] = workspace["read_only"]

    if "skills" in data and isinstance(data["skills"], dict):
        skills = dict(data.pop("skills"))
        if "sources" in skills and "skill_sources" not in data:
            data["skill_sources"] = skills["sources"]
        if "backend" in skills and "skills_backend" not in data:
            data["skills_backend"] = skills["backend"]

    return data


def _ensure_name(payload: Dict[str, Any], default_name: str) -> str:
    name = payload.get("name")
    if isinstance(name, str) and name:
        return name.strip()
    return default_name


def _build_agent_config(
    payload: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    context: RuntimeContext,
    source: Path,
) -> WorkspaceAgentConfig:
    merged = _deep_merge(dict(defaults), dict(payload))
    merged = _apply_aliases(merged)
    merged["kind"] = "agent"
    merged["name"] = _ensure_name(merged, source.stem)
    if "workspace_root" not in merged:
        merged["workspace_root"] = str(context.project_root)

    runtime_cfg = context.merged_runtime_config
    mcp_servers = merged.get("mcp_servers")
    if mcp_servers is None:
        mcp_servers = runtime_cfg.get("mcp_servers")
        if mcp_servers is not None:
            merged["mcp_servers"] = mcp_servers

    resolved = _normalize_config_object(
        dict(merged),
        source=str(source),
        register_mcp=True,
        resolve_tools=True,
        env=context.env,
        toolkit=context.toolkit,
        execution_context=context.execution_context,
        mcp_runtime=context.mcp_runtime,
    )

    if not isinstance(resolved, dict):
        raise ValueError(f"Failed to normalize agent config in {source}: expected a mapping.")

    # Keep only agent-relevant fields; WorkspaceAgentConfig ignores irrelevant values.
    # ``kind`` is not part of the model and must not be passed through.
    resolved.pop("kind", None)
    return WorkspaceAgentConfig(**resolved)


def build_workspace_agent(
    path: Path,
    payload: Mapping[str, Any],
    context: RuntimeContext,
    *,
    defaults: Mapping[str, Any],
) -> WorkspaceAgent:
    config = _build_agent_config(payload, defaults, context=context, source=path)
    return WorkspaceAgent(config)


def normalize_agent_payload(
    payload: Mapping[str, Any],
    context: RuntimeContext,
    *,
    defaults: Mapping[str, Any],
    source: Path,
) -> WorkspaceAgentConfig:
    return _build_agent_config(payload, defaults, context=context, source=source)
