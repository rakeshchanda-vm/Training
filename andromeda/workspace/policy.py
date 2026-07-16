from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence


ToolProfile = Literal[
    "minimal",
    "read_only",
    "full_compatibility",
    "shell_enabled",
    "shell_disabled",
]

DEFAULT_DENIED_COMMANDS: tuple[str, ...] = (
    "sudo",
    "su",
    "mount",
    "umount",
    "nsenter",
    "unshare",
    "iptables",
    "modprobe",
    "systemctl",
    "service",
    "dockerd",
    "docker",
    "podman",
    "nerdctl",
    "ctr",
    "ssh-agent",
    "sshd",
)


@dataclass(frozen=True)
class FilePolicy:
    """Filesystem limits and path behavior for a workspace."""

    max_file_size_mb: int = 10
    allow_symlinks: bool = False
    protect_root: bool = True

    @property
    def max_file_size_bytes(self) -> int:
        return max(1, self.max_file_size_mb) * 1024 * 1024


@dataclass(frozen=True)
class ShellPolicy:
    """Execution limits for workspace-bound shell commands."""

    timeout_seconds: int = 30
    max_output_chars: int = 20_000
    allow_raw_shell: bool = False
    enable_background_shell: bool = False
    max_background_processes: int = 4
    background_timeout_seconds: int = 300
    background_output_storage: Literal["memory", "file"] = "file"
    env_allowlist: Sequence[str] = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )
    network_enabled: bool = False
    extra_env: Mapping[str, str] = field(default_factory=dict)
    allowed_commands: Sequence[str] | None = None
    denied_commands: Sequence[str] = DEFAULT_DENIED_COMMANDS


@dataclass(frozen=True)
class WorkspacePolicy:
    """Top-level policy used to assemble workspace file and shell tools."""

    read_only: bool = False
    enable_shell: bool = False
    tool_profile: ToolProfile | None = None
    file: FilePolicy = field(default_factory=FilePolicy)
    shell: ShellPolicy = field(default_factory=ShellPolicy)
