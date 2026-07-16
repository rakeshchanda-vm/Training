from __future__ import annotations

from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

from andromeda.tools.filesystem import make_filesystem_tools
from andromeda.tools.shell import (
    WorkspaceShellProcessManager,
    make_provider_shell_tools,
    make_shell_tools,
)
from andromeda.tools.vfs_filesystem import (
    ReadOnlyFilesystemDriver,
    make_vfs_filesystem_tools,
)
from andromeda.workspace.backends import (
    BackendCapabilities,
    WorkspaceBackendName,
    get_backend_capabilities,
    validate_backend_tool_profile,
    validate_backend_compatibility,
    validate_workspace_policy,
)
from andromeda.workspace.policy import ToolProfile, WorkspacePolicy
from andromeda.workspace.provider_settings import ProviderSettings
from andromeda.workspace.providers import (
    WorkspaceBackendProvider,
    WorkspaceProviderState,
    build_workspace_provider,
    _workspace_from_home,
)
from andromeda.workspace.seeds import WorkspaceSeed


MINIMAL_TOOL_KEYS = {"read_file", "apply_patch"}
READ_ONLY_TOOL_KEYS = {
    "read_file",
    "list_directory",
    "directory_tree",
    "grep_file",
    "search_files",
    "list_allowed_directories",
}
FULL_COMPATIBILITY_TOOL_KEYS = {
    "read_file",
    "write_file",
    "search_and_replace_file_edit",
    "edit_file",
    "append_to_file",
    "apply_patch",
    "list_directory",
    "list_allowed_directories",
    "directory_tree",
    "grep_file",
    "search_files",
    "create_directory",
    "delete_file_or_directory",
}
SHELL_DISABLED_TOOL_KEYS = READ_ONLY_TOOL_KEYS | MINIMAL_TOOL_KEYS
PROFILE_TOOL_KEYS: dict[ToolProfile, set[str]] = {
    "minimal": MINIMAL_TOOL_KEYS,
    "read_only": READ_ONLY_TOOL_KEYS,
    "full_compatibility": FULL_COMPATIBILITY_TOOL_KEYS,
    "shell_enabled": MINIMAL_TOOL_KEYS,
    "shell_disabled": SHELL_DISABLED_TOOL_KEYS,
}


@dataclass(frozen=True)
class WorkspaceToolset:
    files: dict[str, object]
    shell: dict[str, object] = field(default_factory=dict)
    artifacts: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkspaceHomeConfig:
    base_dir: Path = field(default_factory=lambda: Path("~/.andromeda/agent-home").expanduser())
    session_id: str | None = None
    workspace_dir_name: str = "workspace"

    def create_workspace_root(self) -> tuple[Path, str]:
        workspace_root, session_id, _ = _workspace_from_home(
            base_dir=self.base_dir,
            session_id=self.session_id,
            workspace_dir_name=self.workspace_dir_name,
        )
        return workspace_root, session_id


@dataclass
class WorkspaceSession:
    backend: WorkspaceBackendName | str
    root: Path
    policy: WorkspacePolicy = field(default_factory=WorkspacePolicy)
    capabilities: BackendCapabilities = field(init=False)
    owns_root: bool = False
    session_id: str | None = None
    metadata_path: Path | None = None
    provider: WorkspaceBackendProvider | None = None
    provider_state: WorkspaceProviderState | None = None
    process_manager: WorkspaceShellProcessManager | None = None

    def __post_init__(self) -> None:
        resolved_root = self.root.expanduser().resolve()
        object.__setattr__(self, "root", resolved_root)
        object.__setattr__(
            self,
            "capabilities",
            get_backend_capabilities(self.backend, root=resolved_root),
        )
        validate_backend_compatibility(self.backend, self.capabilities, self.policy)

    @classmethod
    def create(
        cls,
        *,
        backend: WorkspaceBackendName | str = "local_fs",
        root: str | Path | None = None,
        seed: WorkspaceSeed | Sequence[WorkspaceSeed] | None = None,
        policy: WorkspacePolicy | None = None,
        home: WorkspaceHomeConfig | None = None,
        provider: WorkspaceBackendProvider | None = None,
        settings: ProviderSettings | None = None,
    ) -> "WorkspaceSession":
        workspace_policy = policy or WorkspacePolicy()
        home_config = home or WorkspaceHomeConfig()
        if workspace_policy.enable_shell and root is not None and str(backend) in {
            "local_fs",
            "ephemeral_fs",
            "s3_snapshot",
            "bubblewrap_process",
            "gvisor_container",
        }:
            Path(root).expanduser().mkdir(parents=True, exist_ok=True)
        validate_backend_compatibility(
            str(backend),
            get_backend_capabilities(
                str(backend),
                root=Path(root).expanduser() if root else None,
            ),
            workspace_policy,
        )
        workspace_provider = build_workspace_provider(
            str(backend),
            settings,
            provider=provider,
        )
        provider_state: WorkspaceProviderState | None = None
        session: WorkspaceSession | None = None
        try:
            provider_state = workspace_provider.create(
                {
                    "backend": backend,
                    "root": root,
                    "policy": workspace_policy,
                    "home_base_dir": home_config.base_dir,
                    "session_id": home_config.session_id,
                    "workspace_dir_name": home_config.workspace_dir_name,
                }
            )

            session = cls(
                backend=backend,
                root=provider_state.root,
                policy=workspace_policy,
                owns_root=provider_state.owns_root,
                session_id=provider_state.session_id,
                metadata_path=provider_state.metadata_path,
                provider=workspace_provider,
                provider_state=provider_state,
            )
            session.write_metadata(seed)
            session.apply_seeds(seed)
            workspace_provider.after_seeds()
            return session
        except Exception:
            if session is not None:
                _cleanup_failed_session(session)
            else:
                _cleanup_failed_provider_state(workspace_provider, provider_state)
            raise

    def apply_seeds(self, seed: WorkspaceSeed | Sequence[WorkspaceSeed] | None) -> None:
        if seed is None:
            return
        seeds: Iterable[WorkspaceSeed]
        if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
            seeds = seed
        else:
            seeds = [seed]  # type: ignore[list-item]
        for item in seeds:
            item.apply(self.root, self.policy.file)

    def write_metadata(self, seed: WorkspaceSeed | Sequence[WorkspaceSeed] | None = None) -> None:
        if self.metadata_path is None:
            return
        seeds: list[str] = []
        if seed is not None:
            if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
                seeds = [type(item).__name__ for item in seed]
            else:
                seeds = [type(seed).__name__]
        metadata = {
            "backend": self.backend,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owns_root": self.owns_root,
            "root": str(self.root),
            "session_id": self.session_id,
            "seed_types": seeds,
            "policy": _redacted_policy_metadata(self.policy),
            "provider": self.provider_state.metadata if self.provider_state else {},
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def _default_tool_profile(self, enable_shell: bool) -> ToolProfile:
        if self.policy.tool_profile is not None:
            return self.policy.tool_profile
        return "shell_enabled" if enable_shell else "shell_disabled"

    def _filter_file_tools(
        self,
        tools: dict[str, object],
        profile: ToolProfile,
    ) -> dict[str, object]:
        allowed = PROFILE_TOOL_KEYS[profile]
        return {name: tool for name, tool in tools.items() if name in allowed}

    def _filesystem_path_aliases(self) -> list[tuple[str, str]]:
        if self.provider is None or self.provider_state is None:
            return []
        if not getattr(self.provider_state, "provider_backed_shell", False):
            return []
        settings = getattr(self.provider, "settings", None)
        alias = getattr(settings, "workspace_mount", None) or getattr(settings, "workspace_path", None)
        metadata = self.provider_state.metadata or {}
        if not alias and isinstance(metadata, dict):
            sandbox = metadata.get("sandbox")
            if isinstance(sandbox, dict):
                alias = sandbox.get("workspace_path")
            alias = alias or metadata.get("workspace_path")
        if not alias:
            return []
        return [(str(alias), str(self.root))]

    def toolset(
        self,
        *,
        read_only: bool | None = None,
        enable_shell: bool | None = None,
        tool_profile: ToolProfile | None = None,
    ) -> WorkspaceToolset:
        effective_read_only = self.policy.read_only if read_only is None else read_only
        effective_shell = self.policy.enable_shell if enable_shell is None else enable_shell
        validate_workspace_policy(
            self.policy,
            read_only=read_only,
            enable_shell=enable_shell,
            tool_profile=tool_profile,
        )
        validate_backend_tool_profile(
            self.backend,
            self.policy,
            enable_shell=effective_shell,
            tool_profile=tool_profile,
        )
        if effective_read_only:
            profile: ToolProfile = "read_only"
        else:
            profile = tool_profile or self._default_tool_profile(effective_shell)
        if self.provider_state and self.provider_state.filesystem_driver is not None:
            driver = self.provider_state.filesystem_driver
            if effective_read_only:
                driver = ReadOnlyFilesystemDriver(driver)
            files = make_vfs_filesystem_tools(driver)
        else:
            files = make_filesystem_tools(
                [str(self.root)],
                read_only=effective_read_only,
                file_policy=self.policy.file,
                path_aliases=self._filesystem_path_aliases(),
            )
        files = self._filter_file_tools(files, profile)

        shell_tools: dict[str, object] = {}
        if effective_shell:
            shell_capabilities = get_backend_capabilities(self.backend, root=self.root)
            shell_policy = WorkspacePolicy(
                read_only=effective_read_only,
                enable_shell=True,
                file=self.policy.file,
                shell=self.policy.shell,
                tool_profile=profile,
            )
            validate_backend_compatibility(self.backend, shell_capabilities, shell_policy)
            if self.provider_state and self.provider_state.provider_backed_shell:
                shell_tools = make_provider_shell_tools(self.provider, self.policy.shell)
            elif self.process_manager is None:
                self.process_manager = WorkspaceShellProcessManager(self.root, self.policy.shell)
                shell_tools = make_shell_tools(
                    self.root,
                    self.policy.shell,
                    process_manager=self.process_manager,
                )
            else:
                shell_tools = make_shell_tools(
                    self.root,
                    self.policy.shell,
                    process_manager=self.process_manager,
                )

        return WorkspaceToolset(files=files, shell=shell_tools)

    def tools(
        self,
        *,
        read_only: bool | None = None,
        enable_shell: bool | None = None,
        tool_profile: ToolProfile | None = None,
    ) -> dict[str, object]:
        toolset = self.toolset(
            read_only=read_only,
            enable_shell=enable_shell,
            tool_profile=tool_profile,
        )
        return {**toolset.files, **toolset.shell, **toolset.artifacts}

    def cleanup(self) -> None:
        if self.process_manager is not None:
            self.process_manager.kill_all()
        if self.provider is not None:
            self.provider.destroy()
        if self.owns_root:
            target = self.root.parent if self.metadata_path is not None else self.root
            shutil.rmtree(target, ignore_errors=True)

    def __enter__(self) -> "WorkspaceSession":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.cleanup()


def _redacted_policy_metadata(policy: WorkspacePolicy) -> dict[str, Any]:
    metadata = asdict(policy)
    shell = metadata.get("shell")
    if isinstance(shell, dict) and shell.get("extra_env"):
        shell["extra_env"] = "<redacted>"
    return metadata


def _cleanup_failed_session(session: WorkspaceSession) -> None:
    if session.process_manager is not None:
        with suppress(Exception):
            session.process_manager.kill_all()
    if session.provider is not None:
        with suppress(Exception):
            session.provider.destroy()
    if session.owns_root:
        target = session.root.parent if session.metadata_path is not None else session.root
        shutil.rmtree(target, ignore_errors=True)


def _cleanup_failed_provider_state(
    provider: WorkspaceBackendProvider,
    provider_state: WorkspaceProviderState | None,
) -> None:
    with suppress(Exception):
        provider.destroy()
    if provider_state is not None and provider_state.owns_root:
        target = (
            provider_state.root.parent
            if provider_state.metadata_path is not None
            else provider_state.root
        )
        shutil.rmtree(target, ignore_errors=True)
