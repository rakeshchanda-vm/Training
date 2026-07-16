from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from andromeda.config.yaml_utils import yaml_load


RUNTIME_CONFIG_FILENAMES = ("andromeda.yaml", "andromeda.yml")


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML file for a runtime config root.

    Returns an empty dictionary when the file does not exist or is empty.
    Raises a ``RuntimeError`` when parsing fails, preserving the source path
    for easier debugging.
    """

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml_load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse runtime config file: {path}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid runtime config at {path}: expected a YAML mapping.")
    return data


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Return a recursive merge of two dictionaries.

    Values from ``extra`` override values from ``base``.
    """

    merged: Dict[str, Any] = dict(base)
    for key, value in extra.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_root(base: Path, config_dir_name: str | Path = ".andromeda") -> Optional[Path]:
    config_dir = Path(config_dir_name).expanduser()
    if config_dir.is_absolute():
        candidate = config_dir
    else:
        candidate = base / config_dir
    if candidate.is_dir():
        return candidate.resolve()
    return None


def _resolve_andromeda_yaml(config_root: Path) -> Optional[Path]:
    for filename in RUNTIME_CONFIG_FILENAMES:
        candidate = config_root / filename
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved runtime discovery context.

    ``project_root`` is always the explicit root supplied by caller (or cwd).
    ``project_config_root`` and ``global_config_root`` point at discovered
    ``.andromeda`` directories when they exist.
    """

    project_root: Path
    include_global: bool
    project_config_root: Optional[Path] = None
    global_config_root: Optional[Path] = None
    env: Optional[Mapping[str, str]] = None
    toolkit: Any = None
    execution_context: Any = None
    mcp_runtime: Any = None
    # Per-instance memo of parsed andromeda.yaml files. Mutating the dict's
    # contents is allowed on a frozen instance (we never rebind the attribute),
    # so config files are read and parsed at most once per context.
    _config_cache: Dict[Path, Dict[str, Any]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def _read_config(self, path: Path) -> Dict[str, Any]:
        cached = self._config_cache.get(path)
        if cached is not None:
            return cached
        data = _load_yaml_config(path)
        self._config_cache[path] = data
        return data

    @property
    def project_enabled(self) -> bool:
        return self.project_config_root is not None

    @property
    def global_enabled(self) -> bool:
        return self.include_global and self.global_config_root is not None

    @property
    def project_agent_defaults(self) -> Dict[str, Any]:
        if not self.project_config_root:
            return {}
        cfg_path = _resolve_andromeda_yaml(self.project_config_root)
        if not cfg_path:
            return {}
        payload = self._read_config(cfg_path)
        defaults = payload.get("agent_defaults")
        return dict(defaults) if isinstance(defaults, dict) else {}

    @property
    def project_workflow_defaults(self) -> Dict[str, Any]:
        if not self.project_config_root:
            return {}
        cfg_path = _resolve_andromeda_yaml(self.project_config_root)
        if not cfg_path:
            return {}
        payload = self._read_config(cfg_path)
        defaults = payload.get("workflow_defaults")
        return dict(defaults) if isinstance(defaults, dict) else {}

    @property
    def global_agent_defaults(self) -> Dict[str, Any]:
        if not self.global_config_root:
            return {}
        cfg_path = _resolve_andromeda_yaml(self.global_config_root)
        if not cfg_path:
            return {}
        payload = self._read_config(cfg_path)
        defaults = payload.get("agent_defaults")
        return dict(defaults) if isinstance(defaults, dict) else {}

    @property
    def global_workflow_defaults(self) -> Dict[str, Any]:
        if not self.global_config_root:
            return {}
        cfg_path = _resolve_andromeda_yaml(self.global_config_root)
        if not cfg_path:
            return {}
        payload = self._read_config(cfg_path)
        defaults = payload.get("workflow_defaults")
        return dict(defaults) if isinstance(defaults, dict) else {}

    @property
    def merged_agent_defaults(self) -> Dict[str, Any]:
        merged = {}
        merged = _deep_merge(merged, self.global_agent_defaults)
        merged = _deep_merge(merged, self.project_agent_defaults)
        return merged

    @property
    def merged_workflow_defaults(self) -> Dict[str, Any]:
        merged = {}
        merged = _deep_merge(merged, self.global_workflow_defaults)
        merged = _deep_merge(merged, self.project_workflow_defaults)
        return merged

    @property
    def merged_runtime_config(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}

        global_cfg: Dict[str, Any] = {}
        if self.global_config_root:
            global_path = _resolve_andromeda_yaml(self.global_config_root)
            if global_path:
                global_cfg = self._read_config(global_path)

        project_cfg: Dict[str, Any] = {}
        if self.project_config_root:
            project_path = _resolve_andromeda_yaml(self.project_config_root)
            if project_path:
                project_cfg = self._read_config(project_path)

        merged = _deep_merge(merged, global_cfg)
        merged = _deep_merge(merged, project_cfg)
        return merged

    def resolved_mcp_servers(self) -> Dict[str, Any]:
        merged = self.merged_runtime_config.get("mcp_servers")
        if merged is None:
            return {}
        if isinstance(merged, dict):
            return merged
        if isinstance(merged, list):
            mcp: Dict[str, Any] = {}
            for entry in merged:
                if not isinstance(entry, dict):
                    continue
                mcp.update(entry)
            return mcp
        raise RuntimeError("Invalid 'mcp_servers' setting in runtime config: expected mapping or list")

    @classmethod
    def discover(
        cls,
        root: str | Path | None = None,
        *,
        include_global: bool = True,
        global_root: str | Path | None = None,
        project_config_root: str | Path | None = None,
        config_dir_name: str | Path = ".andromeda",
        env: Mapping[str, str] | None = None,
        toolkit: Any = None,
        execution_context: Any = None,
        mcp_runtime: Any = None,
    ) -> "RuntimeContext":
        """Resolve project/global registry roots from filesystem locations.

        By default, project lookup is the **exact current working directory**,
        not an upward walk.
        """

        project_root = Path(root or Path.cwd()).resolve()
        if project_root.is_file():
            project_root = project_root.parent

        resolved_project_config_root = (
            Path(project_config_root).expanduser().resolve()
            if project_config_root is not None
            else _resolve_config_root(project_root, config_dir_name=config_dir_name)
        )
        if resolved_project_config_root is not None and not resolved_project_config_root.is_dir():
            resolved_project_config_root = None

        global_config_root = None
        if include_global:
            explicit = Path(global_root).expanduser().resolve() if global_root else Path.home() / ".andromeda"
            if explicit.is_dir():
                global_config_root = explicit

        return cls(
            project_root=project_root,
            include_global=include_global,
            project_config_root=resolved_project_config_root,
            global_config_root=global_config_root,
            env=env,
            toolkit=toolkit,
            execution_context=execution_context,
            mcp_runtime=mcp_runtime,
        )
