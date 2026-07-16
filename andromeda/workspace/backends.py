from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from andromeda.workspace.policy import ToolProfile, WorkspacePolicy


WorkspaceBackendName = Literal[
    "local_fs",
    "ephemeral_fs",
    "postgres_vfs",
    "s3_snapshot",
    "bubblewrap_process",
    "gvisor_container",
    "microvm",
]


class WorkspaceCompatibilityError(ValueError):
    """Raised when requested workspace tools cannot be backed safely."""


@dataclass(frozen=True)
class BackendCapabilities:
    supports_file_tools: bool
    supports_shell: bool
    materialized_path: Path | None = None


SANDBOXED_SHELL_BACKENDS = {
    "microvm",
    "bubblewrap_process",
    "gvisor_container",
}
SHELL_SAFE_FILE_PROFILES = {None, "minimal", "shell_enabled"}


def get_backend_capabilities(
    backend: WorkspaceBackendName | str,
    *,
    root: Path | None = None,
) -> BackendCapabilities:
    if backend in {
        "local_fs",
        "ephemeral_fs",
        "s3_snapshot",
        "microvm",
        "bubblewrap_process",
        "gvisor_container",
    }:
        return BackendCapabilities(
            supports_file_tools=True,
            supports_shell=True,
            materialized_path=root,
        )
    if backend == "postgres_vfs":
        return BackendCapabilities(
            supports_file_tools=True,
            supports_shell=False,
            materialized_path=None,
        )
    raise WorkspaceCompatibilityError(f"Unsupported workspace backend: {backend}")


def validate_workspace_policy(
    policy: WorkspacePolicy,
    *,
    read_only: bool | None = None,
    enable_shell: bool | None = None,
    tool_profile: ToolProfile | None = None,
) -> None:
    """Validate workspace policy combinations used to assemble tools."""

    effective_read_only = policy.read_only if read_only is None else read_only
    effective_shell = policy.enable_shell if enable_shell is None else enable_shell
    effective_profile = tool_profile if tool_profile is not None else policy.tool_profile

    if effective_shell and policy.shell.allow_raw_shell and policy.shell.allowed_commands is not None:
        raise WorkspaceCompatibilityError(
            "ShellPolicy cannot combine allow_raw_shell=True with allowed_commands; "
            "raw shell execution would bypass command allowlist enforcement."
        )
    if effective_read_only and effective_shell:
        raise WorkspaceCompatibilityError(
            "WorkspacePolicy cannot combine read_only=True with enable_shell=True."
        )
    if (
        effective_read_only
        and effective_profile is not None
        and effective_profile != "read_only"
    ):
        raise WorkspaceCompatibilityError(
            "When read_only=True, tool_profile must be 'read_only' or omitted."
        )


def validate_backend_tool_profile(
    backend: WorkspaceBackendName | str,
    policy: WorkspacePolicy,
    *,
    enable_shell: bool | None = None,
    tool_profile: ToolProfile | None = None,
) -> None:
    effective_shell = policy.enable_shell if enable_shell is None else enable_shell
    effective_profile = tool_profile if tool_profile is not None else policy.tool_profile
    if (
        str(backend) in SANDBOXED_SHELL_BACKENDS
        and effective_shell
        and effective_profile not in SHELL_SAFE_FILE_PROFILES
    ):
        raise WorkspaceCompatibilityError(
            f"Workspace backend {backend!r} can only combine shell tools with "
            "the minimal file tool profile."
        )


def validate_backend_compatibility(
    backend: WorkspaceBackendName | str,
    capabilities: BackendCapabilities,
    policy: WorkspacePolicy,
) -> None:
    validate_workspace_policy(policy)
    validate_backend_tool_profile(backend, policy)
    if policy.enable_shell and not capabilities.supports_shell:
        raise WorkspaceCompatibilityError(
            f"Workspace backend {backend!r} does not support shell tools."
        )
    if not capabilities.supports_file_tools:
        raise NotImplementedError(
            f"Workspace backend {backend!r} does not provide ET-Agentify file tools yet."
        )
    if policy.enable_shell and capabilities.materialized_path is not None:
        materialized_path = capabilities.materialized_path
        if not materialized_path.is_dir():
            raise WorkspaceCompatibilityError(
                f"Workspace backend {backend!r} must provide a materialized directory for shell tools."
            )
