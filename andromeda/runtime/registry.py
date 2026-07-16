from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from andromeda.config.yaml_utils import yaml_load

from .context import RuntimeContext
from .validation import error, ValidationIssue


RuntimeKind = Literal["agent", "workflow"]


def _workflow_candidates(config_root: Path) -> Iterable[Path]:
    workflows_dir = config_root / "workflows"
    if not workflows_dir.is_dir():
        return []

    entries: list[Path] = []
    for item in sorted(workflows_dir.iterdir()):
        if item.is_file() and item.suffix in {".yml", ".yaml", ".json"}:
            entries.append(item)
            continue

        if item.is_dir():
            workflow_file = item / "workflow.yaml"
            if not workflow_file.exists():
                workflow_file = item / "workflow.yml"
            if workflow_file.exists():
                entries.append(workflow_file)
    return entries


def _agent_candidates(config_root: Path) -> Iterable[Path]:
    agents_dir = config_root / "agents"
    if not agents_dir.is_dir():
        return []
    return sorted(
        item
        for item in agents_dir.iterdir()
        if item.is_file() and item.suffix in {".yml", ".yaml", ".json"}
    )


def _load_yaml_like(path: Path) -> Any:
    if path.suffix == ".json":
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return yaml_load(f)


def _extract_name(path: Path, payload: Mapping[str, Any] | None) -> str:
    if isinstance(payload, Mapping):
        declared = payload.get("name")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    if path.stem == "workflow" and path.parent.name not in {"workflows", ""}:
        return path.parent.name
    return path.stem


def _extract_kind(payload: Mapping[str, Any] | None, default: RuntimeKind) -> RuntimeKind:
    if not isinstance(payload, Mapping):
        return default
    raw = payload.get("kind")
    if raw == "workflow":
        return "workflow"
    if raw in ("agent", "workspace_agent", "workspace-agent"):
        return "agent"
    return default


@dataclass
class RuntimeEntry:
    """Discoverable executable definition from .andromeda."""

    kind: RuntimeKind
    name: str
    source: str
    path: Path
    definition_root: Path
    scope: str
    payload: Mapping[str, Any] | None = None
    effective_name: str | None = None

    def to_info(self) -> Dict[str, Any]:
        return {
            "name": self.effective_name or self.name,
            "kind": self.kind,
            "scope": self.scope,
            "source": str(self.path),
        }


@dataclass
class RunnableRegistry:
    """Runtime registry containing all discoverable agents and workflows."""

    entries: List[RuntimeEntry] = field(default_factory=list)

    @classmethod
    def from_context(cls, context: RuntimeContext) -> "RunnableRegistry":
        entries: list[RuntimeEntry] = []

        def _load_entries(scope_name: str, root: Optional[Path]) -> None:
            if root is None:
                return

            def _append(kind: RuntimeKind, path: Path, payload: Mapping[str, Any] | None) -> None:
                name = _extract_name(path, payload)
                entries.append(
                    RuntimeEntry(
                        kind=kind,
                        name=name,
                        source=f"{scope_name}",
                        path=path,
                        definition_root=(path.parent if kind == "workflow" else root),
                        scope=scope_name,
                        payload=payload,
                    )
                )

            for candidate in _agent_candidates(root):
                payload = _safe_load_candidate(candidate)
                _append("agent", candidate, payload)

            for candidate in _workflow_candidates(root):
                payload = _safe_load_candidate(candidate)
                inferred_kind = _extract_kind(payload, default="workflow")
                _append(inferred_kind, candidate, payload)

        _load_entries("project", context.project_config_root)

        if context.global_enabled:
            _load_entries("global", context.global_config_root)

        registry = cls(entries=entries)
        registry._apply_display_names()
        return registry

    @classmethod
    def _from_payloads(cls, entries: List[RuntimeEntry]) -> "RunnableRegistry":
        registry = cls(entries=entries)
        registry._apply_display_names()
        return registry

    def _apply_display_names(self) -> None:
        grouped: Dict[str, List[RuntimeEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.name, []).append(entry)

        for _, same_name_entries in grouped.items():
            scopes = {entry.scope for entry in same_name_entries}
            if len(scopes) == 1:
                for entry in same_name_entries:
                    entry.effective_name = entry.name
                continue

            for conflict in same_name_entries:
                conflict.effective_name = f"{conflict.name}--{conflict.scope}"

    def list(self, kind: RuntimeKind | None = None) -> List[RuntimeEntry]:
        if kind is None:
            return list(self.entries)
        return [entry for entry in self.entries if entry.kind == kind]

    def _match_suffix(self, name: str) -> Tuple[str, str | None]:
        for suffix in ("--project", "--global"):
            if name.endswith(suffix):
                return name[: -len(suffix)], suffix[2:]
        return name, None

    def resolve(
        self,
        name: str,
        kind: RuntimeKind | None = None,
    ) -> RuntimeEntry:
        base_name, requested_scope = self._match_suffix(name)

        scoped_candidates = [
            entry
            for entry in self.entries
            if entry.name == base_name or entry.effective_name == name
        ]

        if requested_scope is not None:
            scoped_candidates = [
                entry
                for entry in scoped_candidates
                if entry.scope == requested_scope
            ]

        if kind is not None:
            scoped_candidates = [entry for entry in scoped_candidates if entry.kind == kind]

        if not scoped_candidates:
            raise KeyError(f"Runnable not found: {name}")

        if len(scoped_candidates) == 1:
            return scoped_candidates[0]

        # If name is explicitly suffixed, the user should have one match at this point.
        if requested_scope is not None:
            candidates = [
                f"{entry.effective_name} ({entry.kind}, {entry.scope})"
                for entry in scoped_candidates
            ]
            raise KeyError(
                f"Ambiguous runnable '{name}'. Matches: {', '.join(candidates)}"
            )

        if kind is None:
            candidates = [
                f"{entry.effective_name} ({entry.kind}, {entry.scope})"
                for entry in scoped_candidates
            ]
            raise KeyError(
                f"Ambiguous runnable '{name}'. Use --kind or the suffixed name;"
                f" matches: {', '.join(candidates)}"
            )

        candidates = [
            f"{entry.effective_name} ({entry.scope})"
            for entry in scoped_candidates
        ]
        raise KeyError(
            f"Ambiguous runnable '{name}' for kind '{kind}'. Use suffixed name if needed;"
            f" matches: {', '.join(candidates)}"
        )

    def validate(self) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        seen: set[tuple[RuntimeKind, str, str]] = set()

        for entry in self.entries:
            key = (entry.kind, entry.scope, entry.name)
            if key in seen:
                issues.append(error(
                    f"Duplicate definition detected for '{entry.name}' in {entry.scope} {entry.kind}s.",
                    path=str(entry.path),
                ))
            seen.add(key)

            if entry.payload is None:
                issues.append(error(f"Invalid config file: empty payload", path=str(entry.path)))

        return issues


def _safe_load_candidate(path: Path) -> Mapping[str, Any] | None:
    try:
        loaded = _load_yaml_like(path)
    except Exception:
        return None
    return loaded if isinstance(loaded, Mapping) else None
