from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, Union

from andromeda.tools.vfs_filesystem import FilesystemDriver


@dataclass(frozen=True)
class NerdctlDevSettings:
    """Local dev/test microVM settings (nerdctl + Kata)."""

    image: str
    runtime: str = "io.containerd.kata.v2"
    namespace: str = "andromeda"
    workspace_path: str = "/workspace"
    container_name: str | None = None
    nerdctl_path: str = "nerdctl"
    create_timeout_seconds: int = 60


@dataclass(frozen=True)
class ContainerdKataSettings:
    """Production microVM settings (sandbox control plane + Kata)."""

    image: str
    control_plane_url: str | None = None
    runtime: str = "io.containerd.kata.v2"
    namespace: str = "andromeda"
    workspace_path: str = "/workspace"
    sandbox_id: str | None = None
    ttl_seconds: int = 3600
    token: str | None = field(default=None, repr=False)
    control_plane_timeout_seconds: int = 30
    quotas: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PostgresVFSSettings:
    """Postgres-backed virtual filesystem settings."""

    connection_string: str | None = field(default=None, repr=False)
    namespace_key: str | None = None
    namespace_kind: str = "workspace"
    namespace_metadata: Mapping[str, Any] = field(default_factory=dict)
    ensure_schema: bool = False
    root_path: str | None = None
    driver: FilesystemDriver | None = None
    connection_factory: Callable[[], Any] | None = None


@dataclass(frozen=True)
class BubblewrapProcessSettings:
    """Fast local Linux process sandbox settings (bubblewrap)."""

    bwrap_path: str = "bwrap"
    workspace_mount: str = "/workspace"
    network: bool = False
    timeout_seconds: int = 60
    max_output_bytes: int = 5_000_000
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
    extra_ro_binds: Sequence[tuple[str, str]] = ()


@dataclass(frozen=True)
class GVisorContainerSettings:
    """gVisor container sandbox settings (Docker + runsc)."""

    image: str = "python:3.12-slim"
    runtime: str = "runsc"
    docker_path: str = "docker"
    workspace_path: str = "/workspace"
    network: bool = False
    memory: str = "2g"
    cpus: str = "1.0"
    pids_limit: int = 256
    read_only_rootfs: bool = False
    container_name: str | None = None
    create_timeout_seconds: int = 60
    labels: Mapping[str, str] = field(
        default_factory=lambda: {"andromeda.workspace_session": "true"}
    )


ProviderSettings = Union[
    NerdctlDevSettings,
    ContainerdKataSettings,
    PostgresVFSSettings,
    BubblewrapProcessSettings,
    GVisorContainerSettings,
]
