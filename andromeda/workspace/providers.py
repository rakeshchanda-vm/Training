from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import io
import json
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from typing import Any, Callable, Collection, Protocol, Sequence
from urllib import error, request

from andromeda.tools.vfs_filesystem import (
    FilesystemDriver,
    InMemoryFilesystemDriver,
    PostgresFilesystemDriver,
    ReadOnlyFilesystemDriver,
    ScopedFilesystemDriver,
)
from andromeda.workspace.policy import WorkspacePolicy
from andromeda.workspace.provider_settings import (
    BubblewrapProcessSettings,
    ContainerdKataSettings,
    GVisorContainerSettings,
    NerdctlDevSettings,
    PostgresVFSSettings,
    ProviderSettings,
)
from andromeda.workspace.sandbox import SANDBOX_DENIED_ARGV, validate_shell_argv

_VALID_ENV_KEY_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_VALID_PROCESS_ID_RE = re.compile(r"[0-9a-f]{32}")
_VALID_PROVIDER_PROCESS_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")


class WorkspaceProviderError(RuntimeError):
    """Raised when a workspace provider cannot create or operate a backend."""


def _validate_process_id(process_id: str) -> None:
    if not _VALID_PROCESS_ID_RE.fullmatch(process_id):
        raise WorkspaceProviderError(f"Invalid process_id format: {process_id!r}")


def _validate_provider_process_id(process_id: str) -> None:
    if not _VALID_PROVIDER_PROCESS_ID_RE.fullmatch(process_id):
        raise WorkspaceProviderError(f"Invalid process_id format: {process_id!r}")


def _validate_env_keys(env: dict[str, str]) -> None:
    for key in env:
        if not isinstance(key, str) or "\x00" in key or not _VALID_ENV_KEY_RE.fullmatch(key):
            raise WorkspaceProviderError(f"Invalid environment variable key: {key!r}")


@dataclass
class WorkspaceProviderState:
    backend: str
    root: Path
    owns_root: bool = False
    session_id: str | None = None
    metadata_path: Path | None = None
    materialized_path: Path | None = None
    supports_shell: bool = False
    provider_backed_shell: bool = False
    filesystem_driver: FilesystemDriver | None = None
    sandbox_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShellExecutionResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: int = 0
    command: tuple[str, ...] = ()
    cwd: str = ""


ExecResult = ShellExecutionResult


class WorkspaceBackendProvider(Protocol):
    state: WorkspaceProviderState | None

    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        raise NotImplementedError

    def after_seeds(self) -> None:
        raise NotImplementedError

    def copy_in(self, local_path: str, workspace_path: str) -> None:
        raise NotImplementedError

    def copy_out(self, workspace_path: str, local_path: str) -> None:
        raise NotImplementedError

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
        network: bool = False,
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> ShellExecutionResult:
        raise NotImplementedError

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> str:
        raise NotImplementedError

    def status(self, process_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def output(self, process_id: str, max_chars: int | None = None) -> str:
        raise NotImplementedError

    def kill(self, process_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def list(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def destroy(self) -> None:
        raise NotImplementedError


def _workspace_from_home(
    *,
    base_dir: Path,
    session_id: str | None,
    workspace_dir_name: str,
) -> tuple[Path, str, Path]:
    raw_session_id = session_id if session_id is not None else uuid.uuid4().hex
    resolved_session_id = _safe_home_component(
        "session_id",
        raw_session_id,
    )
    resolved_workspace_dir_name = _safe_home_component(
        "workspace_dir_name",
        workspace_dir_name,
    )
    resolved_base_dir = base_dir.expanduser().resolve(strict=False)
    session_home = (resolved_base_dir / resolved_session_id).resolve(strict=False)
    workspace_root = (session_home / resolved_workspace_dir_name).resolve(strict=False)
    try:
        session_home.relative_to(resolved_base_dir)
        workspace_root.relative_to(resolved_base_dir)
    except ValueError as exc:
        raise WorkspaceProviderError(
            "Generated workspace path must stay inside base_dir."
        ) from exc
    workspace_root.mkdir(parents=True, exist_ok=True)
    return workspace_root, resolved_session_id, session_home / "metadata.json"


def _safe_home_component(name: str, value: str) -> str:
    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or "\x00" in component
        or "/" in component
        or "\\" in component
        or Path(component).is_absolute()
    ):
        raise WorkspaceProviderError(f"Invalid {name}: {value!r}.")
    return component


def _tar_directory_base64(path: Path) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(path, arcname=".")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _safe_tar_member_path(root: Path, member: tarfile.TarInfo) -> Path:
    raw_name = member.name.replace("\\", "/")
    if not raw_name or raw_name.startswith("/"):
        raise WorkspaceProviderError(
            f"Refusing to extract unsafe archive member: {member.name!r}"
        )
    parts = [part for part in raw_name.split("/") if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise WorkspaceProviderError(
            f"Refusing to extract unsafe archive member: {member.name!r}"
        )
    target = root.joinpath(*parts) if parts else root
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise WorkspaceProviderError(
            f"Refusing to extract unsafe archive member: {member.name!r}"
        ) from exc
    return target


def _safe_extract_tar(archive: tarfile.TarFile, target: Path) -> None:
    for member in archive.getmembers():
        if member.isdir():
            member_path = _safe_tar_member_path(target, member)
            member_path.mkdir(parents=True, exist_ok=True)
            if member.mode:
                member_path.chmod(member.mode & 0o777)
            continue
        if not member.isfile():
            raise WorkspaceProviderError(
                f"Refusing to extract unsupported archive member: {member.name!r}"
            )
        member_path = _safe_tar_member_path(target, member)
        member_path.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise WorkspaceProviderError(
                f"Archive member has no file data: {member.name!r}"
            )
        with source, member_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if member.mode:
            member_path.chmod(member.mode & 0o777)


def _replace_directory_contents(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in target.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in source.iterdir():
        shutil.move(str(item), str(target / item.name))


def _extract_tar_base64(data: str, target: Path) -> None:
    raw = base64.b64decode(data.encode("ascii"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.copy-out-",
        dir=target.parent,
    ) as tmp:
        staging = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            _safe_extract_tar(archive, staging)
        _replace_directory_contents(staging, target)


class BaseWorkspaceProvider:
    state: WorkspaceProviderState | None = None

    def __init__(self, settings: ProviderSettings | None = None) -> None:
        self.settings = settings

    def after_seeds(self) -> None:
        return None

    def copy_in(self, local_path: str, workspace_path: str) -> None:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support copy_in.")

    def copy_out(self, workspace_path: str, local_path: str) -> None:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support copy_out.")

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
        network: bool = False,
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> ShellExecutionResult:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support shell exec.")

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> str:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support background shell.")

    def status(self, process_id: str) -> dict[str, Any]:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support background shell.")

    def output(self, process_id: str, max_chars: int | None = None) -> str:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support background shell.")

    def kill(self, process_id: str) -> dict[str, Any]:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support background shell.")

    def list(self) -> list[dict[str, Any]]:
        raise WorkspaceProviderError(f"{type(self).__name__} does not support background shell.")

    def destroy(self) -> None:
        return None


class LocalFilesystemProvider(BaseWorkspaceProvider):
    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        root = spec.get("root")
        if root is None:
            raise ValueError("WorkspaceSession.create(root=...) is required for local_fs.")
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        self.state = WorkspaceProviderState(
            backend="local_fs",
            root=root_path,
            materialized_path=root_path,
            supports_shell=True,
        )
        return self.state


class EphemeralFilesystemProvider(BaseWorkspaceProvider):
    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        if spec.get("root") is not None:
            root_path = Path(spec["root"]).expanduser()
            root_path.mkdir(parents=True, exist_ok=True)
            owns_root = False
            session_id = spec.get("session_id")
            metadata_path = None
        else:
            root_path, session_id, metadata_path = _workspace_from_home(
                base_dir=Path(spec["home_base_dir"]),
                session_id=spec.get("session_id"),
                workspace_dir_name=spec.get("workspace_dir_name", "workspace"),
            )
            owns_root = True
        self.state = WorkspaceProviderState(
            backend="ephemeral_fs",
            root=root_path,
            owns_root=owns_root,
            session_id=session_id,
            metadata_path=metadata_path,
            materialized_path=root_path,
            supports_shell=True,
        )
        return self.state


class SandboxControlPlaneClient:
    """HTTP client for a sandbox control plane that owns containerd/Kata access."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_seconds: int = 30,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.opener = opener or request.urlopen

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self.opener(req, timeout=self.timeout_seconds) as response:
                data = response.read()
        except error.URLError as exc:
            raise WorkspaceProviderError(f"Sandbox control plane request failed: {exc}") from exc
        if not data:
            return None
        return json.loads(data.decode("utf-8"))

    def preflight(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/preflight", spec) or {}

    def create_sandbox(self, spec: dict[str, Any]) -> str:
        result = self._request("POST", "/sandboxes", spec) or {}
        sandbox_id = result.get("sandbox_id")
        if not sandbox_id:
            raise WorkspaceProviderError("Sandbox control plane did not return sandbox_id.")
        return str(sandbox_id)

    def copy_in(self, sandbox_id: str, local_path: str, sandbox_path: str) -> None:
        payload = {
            "sandbox_path": sandbox_path,
            "archive_base64": _tar_directory_base64(Path(local_path)),
        }
        self._request("POST", f"/sandboxes/{sandbox_id}/copy-in", payload)

    def copy_out(self, sandbox_id: str, sandbox_path: str, local_path: str) -> None:
        result = self._request(
            "POST",
            f"/sandboxes/{sandbox_id}/copy-out",
            {"sandbox_path": sandbox_path},
        ) or {}
        archive = result.get("archive_base64")
        if not archive:
            raise WorkspaceProviderError("Sandbox control plane did not return archive_base64.")
        _extract_tar_base64(str(archive), Path(local_path))

    def exec(self, sandbox_id: str, argv: Sequence[str], *, cwd: str, env: dict[str, str], timeout: int) -> ShellExecutionResult:
        result = self._request(
            "POST",
            f"/sandboxes/{sandbox_id}/exec",
            {"argv": list(argv), "cwd": cwd, "env": env, "timeout": timeout},
        ) or {}
        return ShellExecutionResult(
            exit_code=int(result.get("exit_code", 1)),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
        )

    def start(self, sandbox_id: str, argv: Sequence[str], *, cwd: str, env: dict[str, str]) -> str:
        result = self._request(
            "POST",
            f"/sandboxes/{sandbox_id}/processes",
            {"argv": list(argv), "cwd": cwd, "env": env},
        ) or {}
        process_id = result.get("process_id")
        if not process_id:
            raise WorkspaceProviderError("Sandbox control plane did not return process_id.")
        return str(process_id)

    def status(self, sandbox_id: str, process_id: str) -> dict[str, Any]:
        return self._request("GET", f"/sandboxes/{sandbox_id}/processes/{process_id}") or {}

    def output(self, sandbox_id: str, process_id: str, max_chars: int | None = None) -> str:
        suffix = "" if max_chars is None else f"?max_chars={max_chars}"
        result = self._request("GET", f"/sandboxes/{sandbox_id}/processes/{process_id}/output{suffix}") or {}
        return str(result.get("output", ""))

    def kill(self, sandbox_id: str, process_id: str) -> dict[str, Any]:
        return self._request("POST", f"/sandboxes/{sandbox_id}/processes/{process_id}/kill") or {}

    def list(self, sandbox_id: str) -> list[dict[str, Any]]:
        result = self._request("GET", f"/sandboxes/{sandbox_id}/processes") or {}
        processes = result.get("processes", [])
        return processes if isinstance(processes, list) else []

    def destroy_sandbox(self, sandbox_id: str) -> None:
        self._request("DELETE", f"/sandboxes/{sandbox_id}")


class ContainerdKataSandboxProvider(BaseWorkspaceProvider):
    def __init__(
        self,
        settings: ContainerdKataSettings | None = None,
        *,
        client: SandboxControlPlaneClient | None = None,
    ) -> None:
        super().__init__(settings)
        self.client = client
        self.sandbox_id: str | None = None

    def _settings(self) -> ContainerdKataSettings:
        if self.settings is None:
            raise WorkspaceProviderError(
                "microvm requires settings=ContainerdKataSettings(...) or an injected provider."
            )
        if not isinstance(self.settings, ContainerdKataSettings):
            raise WorkspaceProviderError(
                f"Expected ContainerdKataSettings, got {type(self.settings).__name__}."
            )
        return self.settings

    def _client(self) -> SandboxControlPlaneClient:
        if self.client is not None:
            return self.client
        settings = self._settings()
        if not settings.control_plane_url:
            raise WorkspaceProviderError(
                "ContainerdKataSettings.control_plane_url is required "
                "or inject a SandboxControlPlaneClient."
            )
        self.client = SandboxControlPlaneClient(
            settings.control_plane_url,
            token=settings.token,
            timeout_seconds=settings.control_plane_timeout_seconds,
        )
        return self.client

    @staticmethod
    def _validate_process_id(process_id: str) -> None:
        _validate_process_id(process_id)

    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        settings = self._settings()
        root_path, session_id, metadata_path = _workspace_from_home(
            base_dir=Path(spec["home_base_dir"]),
            session_id=spec.get("session_id"),
            workspace_dir_name=spec.get("workspace_dir_name", "workspace"),
        )
        sandbox_spec = {
            "image": settings.image,
            "runtime": settings.runtime,
            "namespace": settings.namespace,
            "sandbox_id": settings.sandbox_id or f"andromeda-{session_id}",
            "workspace_path": settings.workspace_path,
            "ttl_seconds": settings.ttl_seconds,
            "quotas": dict(settings.quotas),
            "labels": {
                "workspace_session_id": session_id,
                "backend": "microvm",
                "created_at": datetime.now(timezone.utc).isoformat(),
                **dict(settings.labels),
            },
        }
        client = self._client()
        client.preflight(sandbox_spec)
        self.sandbox_id = client.create_sandbox(sandbox_spec)
        self.state = WorkspaceProviderState(
            backend="microvm",
            root=root_path,
            owns_root=True,
            session_id=session_id,
            metadata_path=metadata_path,
            materialized_path=root_path,
            supports_shell=True,
            provider_backed_shell=True,
            sandbox_id=self.sandbox_id,
            metadata={"sandbox": sandbox_spec},
        )
        return self.state

    def after_seeds(self) -> None:
        if self.state is not None:
            self.copy_in(str(self.state.root), self._settings().workspace_path)

    def copy_in(self, local_path: str, workspace_path: str) -> None:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        self._client().copy_in(self.sandbox_id, local_path, workspace_path)

    def copy_out(self, workspace_path: str, local_path: str) -> None:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        self._client().copy_out(self.sandbox_id, workspace_path, local_path)

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
        network: bool = False,
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> ShellExecutionResult:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        validate_shell_argv(
            argv,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        _validate_env_keys(env)
        workspace_path = self._settings().workspace_path
        self.copy_in(str(self.state.root), workspace_path)  # type: ignore[union-attr]
        result = self._client().exec(self.sandbox_id, argv, cwd=cwd, env=env, timeout=timeout)
        self.copy_out(workspace_path, str(self.state.root))  # type: ignore[union-attr]
        return result

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> str:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        validate_shell_argv(
            argv,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        _validate_env_keys(env)
        workspace_path = self._settings().workspace_path
        self.copy_in(str(self.state.root), workspace_path)  # type: ignore[union-attr]
        return self._client().start(self.sandbox_id, argv, cwd=cwd, env=env)

    def status(self, process_id: str) -> dict[str, Any]:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        _validate_provider_process_id(process_id)
        status = self._client().status(self.sandbox_id, process_id)
        if status.get("running") is False and self.state is not None:
            self.copy_out(self._settings().workspace_path, str(self.state.root))
        return status

    def output(self, process_id: str, max_chars: int | None = None) -> str:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        _validate_provider_process_id(process_id)
        return self._client().output(self.sandbox_id, process_id, max_chars)

    def kill(self, process_id: str) -> dict[str, Any]:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        _validate_provider_process_id(process_id)
        return self._client().kill(self.sandbox_id, process_id)

    def list(self) -> list[dict[str, Any]]:
        if self.sandbox_id is None:
            raise WorkspaceProviderError("Sandbox has not been created.")
        return self._client().list(self.sandbox_id)

    def destroy(self) -> None:
        if self.sandbox_id is not None:
            self._client().destroy_sandbox(self.sandbox_id)
            self.sandbox_id = None


class NerdctlKataDevProvider(ContainerdKataSandboxProvider):
    def __init__(
        self,
        settings: NerdctlDevSettings | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        BaseWorkspaceProvider.__init__(self, settings)
        self.runner = runner or subprocess.run
        self.container_name: str | None = None
        self.sandbox_id = None

    def _settings(self) -> NerdctlDevSettings:
        if self.settings is None:
            raise WorkspaceProviderError(
                "microvm requires settings=NerdctlDevSettings(...) or an injected provider."
            )
        if not isinstance(self.settings, NerdctlDevSettings):
            raise WorkspaceProviderError(
                f"Expected NerdctlDevSettings, got {type(self.settings).__name__}."
            )
        return self.settings

    def _run(self, args: Sequence[str], *, timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.runner(
            list(args),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=check,
        )

    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        settings = self._settings()
        root_path, session_id, metadata_path = _workspace_from_home(
            base_dir=Path(spec["home_base_dir"]),
            session_id=spec.get("session_id"),
            workspace_dir_name=spec.get("workspace_dir_name", "workspace"),
        )
        nerdctl = settings.nerdctl_path
        runtime = settings.runtime
        namespace = settings.namespace
        self.container_name = settings.container_name or f"andromeda-{session_id}"
        try:
            self._run([nerdctl, "--version"], check=True)
            self._run([nerdctl, "--namespace", namespace, "image", "inspect", settings.image], check=True)
            self._run(
                [
                    nerdctl,
                    "--namespace",
                    namespace,
                    "run",
                    "-d",
                    "--runtime",
                    runtime,
                    "--name",
                    self.container_name,
                    "--workdir",
                    settings.workspace_path,
                    settings.image,
                    "sleep",
                    "infinity",
                ],
                timeout=settings.create_timeout_seconds,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise WorkspaceProviderError(f"Failed to create nerdctl Kata sandbox: {exc}") from exc
        self.sandbox_id = self.container_name
        self.state = WorkspaceProviderState(
            backend="microvm",
            root=root_path,
            owns_root=True,
            session_id=session_id,
            metadata_path=metadata_path,
            materialized_path=root_path,
            supports_shell=True,
            provider_backed_shell=True,
            sandbox_id=self.sandbox_id,
            metadata={"provider": "nerdctl", "container_name": self.container_name},
        )
        return self.state

    def _nerdctl_prefix(self) -> list[str]:
        settings = self._settings()
        return [
            settings.nerdctl_path,
            "--namespace",
            settings.namespace,
        ]

    def copy_in(self, local_path: str, workspace_path: str) -> None:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        self._run([*self._nerdctl_prefix(), "exec", self.container_name, "mkdir", "-p", workspace_path])
        self._run([*self._nerdctl_prefix(), "cp", f"{local_path}/.", f"{self.container_name}:{workspace_path}"])

    def copy_out(self, workspace_path: str, local_path: str) -> None:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.copy-out-",
            dir=target.parent,
        ) as tmp:
            staging = Path(tmp)
            self._run(
                [
                    *self._nerdctl_prefix(),
                    "cp",
                    f"{self.container_name}:{workspace_path}/.",
                    str(staging),
                ]
            )
            _replace_directory_contents(staging, target)

    def exec(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
        network: bool = False,
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> ShellExecutionResult:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        validate_shell_argv(
            argv,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        _validate_env_keys(env)
        self.copy_in(str(self.state.root), self._settings().workspace_path)  # type: ignore[union-attr]
        env_args = [item for key, value in env.items() for item in ("--env", f"{key}={value}")]
        completed = self._run(
            [
                *self._nerdctl_prefix(),
                "exec",
                "--workdir",
                cwd,
                *env_args,
                self.container_name,
                *list(argv),
            ],
            timeout=timeout,
            check=False,
        )
        self.copy_out(self._settings().workspace_path, str(self.state.root))  # type: ignore[union-attr]
        return ShellExecutionResult(completed.returncode, completed.stdout, completed.stderr)

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> str:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        validate_shell_argv(
            argv,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        _validate_env_keys(env)
        self.copy_in(str(self.state.root), self._settings().workspace_path)  # type: ignore[union-attr]
        process_id = uuid.uuid4().hex
        command = " ".join(shlex.quote(arg) for arg in argv)
        env_args = [item for key, value in env.items() for item in ("--env", f"{key}={value}")]
        shell_script = (
            "mkdir -p /tmp/andromeda-shell; "
            f"(cd {shlex.quote(cwd)} && ({command}) "
            f"> /tmp/andromeda-shell/{process_id}.log 2>&1; "
            f"echo $? > /tmp/andromeda-shell/{process_id}.exit) & "
            f"echo $! > /tmp/andromeda-shell/{process_id}.pid"
        )
        self._run([*self._nerdctl_prefix(), "exec", *env_args, self.container_name, "sh", "-lc", shell_script])
        return process_id

    def status(self, process_id: str) -> dict[str, Any]:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        _validate_provider_process_id(process_id)
        script = (
            f"if test -f /tmp/andromeda-shell/{process_id}.exit; then "
            f"echo exit_code=$(cat /tmp/andromeda-shell/{process_id}.exit); "
            f"else echo running=true; fi"
        )
        completed = self._run([*self._nerdctl_prefix(), "exec", self.container_name, "sh", "-lc", script], check=False)
        running = "running=true" in completed.stdout
        exit_code = None
        if "exit_code=" in completed.stdout:
            exit_code = int(completed.stdout.split("exit_code=", 1)[1].strip().splitlines()[0])
            self.copy_out(self._settings().workspace_path, str(self.state.root))  # type: ignore[union-attr]
        return {"process_id": process_id, "running": running, "exit_code": exit_code}

    def output(self, process_id: str, max_chars: int | None = None) -> str:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        _validate_provider_process_id(process_id)
        completed = self._run(
            [*self._nerdctl_prefix(), "exec", self.container_name, "sh", "-lc", f"cat /tmp/andromeda-shell/{process_id}.log 2>/dev/null || true"],
            check=False,
        )
        output = completed.stdout
        return output if max_chars is None or len(output) <= max_chars else output[:max_chars]

    def kill(self, process_id: str) -> dict[str, Any]:
        if self.container_name is None:
            raise WorkspaceProviderError("Container has not been created.")
        _validate_provider_process_id(process_id)
        script = f"kill $(cat /tmp/andromeda-shell/{process_id}.pid) 2>/dev/null || true; echo killed"
        self._run([*self._nerdctl_prefix(), "exec", self.container_name, "sh", "-lc", script], check=False)
        return {"process_id": process_id, "running": False, "killed": True}

    def list(self) -> list[dict[str, Any]]:
        return []

    def destroy(self) -> None:
        if self.container_name is not None:
            self._run([*self._nerdctl_prefix(), "rm", "-f", self.container_name], check=False)
            self.container_name = None
            self.sandbox_id = None


class PostgresVFSProvider(BaseWorkspaceProvider):
    def _settings(self) -> PostgresVFSSettings:
        if self.settings is None:
            raise WorkspaceProviderError(
                "postgres_vfs requires settings=PostgresVFSSettings(...)."
            )
        if not isinstance(self.settings, PostgresVFSSettings):
            raise WorkspaceProviderError(
                f"Expected PostgresVFSSettings, got {type(self.settings).__name__}."
            )
        return self.settings

    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        settings = self._settings()
        driver = settings.driver
        if driver is None:
            connection_factory = settings.connection_factory
            connection_string = settings.connection_string
            if connection_factory is None:
                if not connection_string:
                    raise WorkspaceProviderError(
                        "PostgresVFSSettings requires connection_string "
                        "or connection_factory."
                    )

                def connection_factory() -> Any:
                    try:
                        import psycopg
                    except ImportError as exc:
                        raise WorkspaceProviderError(
                            "postgres_vfs requires psycopg. Install the retrievers-postgres extra."
                        ) from exc
                    return psycopg.connect(connection_string)

            driver = PostgresFilesystemDriver(
                connection_factory=connection_factory,
                namespace_key=settings.namespace_key or spec.get("session_id") or uuid.uuid4().hex,
                namespace_kind=settings.namespace_kind,
                namespace_metadata=dict(settings.namespace_metadata),
                ensure_schema=settings.ensure_schema,
            )
        if settings.root_path:
            driver = ScopedFilesystemDriver(driver, settings.root_path)
        policy: WorkspacePolicy = spec["policy"]
        if policy.read_only:
            driver = ReadOnlyFilesystemDriver(driver)
        root_path, session_id, metadata_path = _workspace_from_home(
            base_dir=Path(spec["home_base_dir"]),
            session_id=spec.get("session_id"),
            workspace_dir_name=spec.get("workspace_dir_name", "workspace"),
        )
        self.state = WorkspaceProviderState(
            backend="postgres_vfs",
            root=root_path,
            owns_root=True,
            session_id=session_id,
            metadata_path=metadata_path,
            supports_shell=False,
            filesystem_driver=driver,
            metadata={"namespace_key": settings.namespace_key},
        )
        return self.state

    def after_seeds(self) -> None:
        if self.state is None or self.state.filesystem_driver is None:
            return
        import logging
        root = self.state.root
        max_size = 10 * 1024 * 1024  # 10MB limit
        for item in root.rglob("*"):
            if item.is_dir():
                self.state.filesystem_driver.mkdir(str(item.relative_to(root)))
            elif item.is_file():
                try:
                    if item.stat().st_size > max_size:
                        logging.warning(
                            "Skipping large file during seed upload: %s (%d bytes)",
                            item, item.stat().st_size,
                        )
                        continue
                    content = item.read_bytes().decode("utf-8")
                    self.state.filesystem_driver.write(
                        str(item.relative_to(root)),
                        content,
                    )
                except (UnicodeDecodeError, PermissionError, OSError) as exc:
                    logging.warning(
                        "Skipping file during seed upload (%s): %s", item, exc,
                    )


def _validate_settings_for_backend(backend: str, settings: ProviderSettings | None) -> None:
    if backend in {"local_fs", "ephemeral_fs", "s3_snapshot"}:
        if settings is not None:
            raise WorkspaceProviderError(
                f"backend={backend!r} does not accept settings; use root= or home= instead."
            )
        return
    if backend == "postgres_vfs":
        if settings is None:
            raise WorkspaceProviderError(
                "postgres_vfs requires settings=PostgresVFSSettings(...)."
            )
        if not isinstance(settings, PostgresVFSSettings):
            raise WorkspaceProviderError(
                f"postgres_vfs requires PostgresVFSSettings, got {type(settings).__name__}."
            )
        return
    if backend == "microvm":
        if settings is None:
            raise WorkspaceProviderError(
                "microvm requires settings=NerdctlDevSettings(...) "
                "or ContainerdKataSettings(...)."
            )
        if not isinstance(settings, (NerdctlDevSettings, ContainerdKataSettings)):
            raise WorkspaceProviderError(
                f"microvm requires NerdctlDevSettings or ContainerdKataSettings, "
                f"got {type(settings).__name__}."
            )
        return
    if backend == "bubblewrap_process":
        if settings is None:
            raise WorkspaceProviderError(
                "bubblewrap_process requires settings=BubblewrapProcessSettings(...)."
            )
        if not isinstance(settings, BubblewrapProcessSettings):
            raise WorkspaceProviderError(
                f"bubblewrap_process requires BubblewrapProcessSettings, "
                f"got {type(settings).__name__}."
            )
        return
    if backend == "gvisor_container":
        if settings is None:
            raise WorkspaceProviderError(
                "gvisor_container requires settings=GVisorContainerSettings(...)."
            )
        if not isinstance(settings, GVisorContainerSettings):
            raise WorkspaceProviderError(
                f"gvisor_container requires GVisorContainerSettings, "
                f"got {type(settings).__name__}."
            )
        return
    raise WorkspaceProviderError(f"Unsupported workspace backend: {backend}")


def _validate_injected_provider(
    backend: str,
    provider: WorkspaceBackendProvider,
    settings: ProviderSettings | None,
) -> None:
    if backend == "microvm":
        if settings is None:
            return
        if isinstance(settings, NerdctlDevSettings) and not isinstance(provider, NerdctlKataDevProvider):
            raise WorkspaceProviderError(
                "settings=NerdctlDevSettings(...) requires NerdctlKataDevProvider; "
                "remove provider=... or pass NerdctlKataDevProvider."
            )
        if isinstance(settings, ContainerdKataSettings) and not isinstance(
            provider, ContainerdKataSandboxProvider
        ):
            raise WorkspaceProviderError(
                "settings=ContainerdKataSettings(...) requires ContainerdKataSandboxProvider; "
                "remove provider=... or pass ContainerdKataSandboxProvider."
            )
        if isinstance(provider, NerdctlKataDevProvider) and isinstance(
            settings, ContainerdKataSettings
        ):
            raise WorkspaceProviderError(
                "NerdctlKataDevProvider cannot be used with ContainerdKataSettings."
            )
        if isinstance(provider, ContainerdKataSandboxProvider) and isinstance(
            settings, NerdctlDevSettings
        ):
            raise WorkspaceProviderError(
                "ContainerdKataSandboxProvider cannot be used with NerdctlDevSettings."
            )
    if backend == "postgres_vfs" and settings is not None:
        if not isinstance(provider, PostgresVFSProvider):
            raise WorkspaceProviderError(
                "settings=PostgresVFSSettings(...) requires PostgresVFSProvider."
            )
    if backend == "bubblewrap_process" and settings is not None:
        from andromeda.workspace.sandbox_providers import BubblewrapProcessProvider

        if not isinstance(provider, BubblewrapProcessProvider):
            raise WorkspaceProviderError(
                "settings=BubblewrapProcessSettings(...) requires BubblewrapProcessProvider."
            )
    if backend == "gvisor_container" and settings is not None:
        from andromeda.workspace.sandbox_providers import GVisorContainerProvider

        if not isinstance(provider, GVisorContainerProvider):
            raise WorkspaceProviderError(
                "settings=GVisorContainerSettings(...) requires GVisorContainerProvider."
            )


def build_workspace_provider(
    backend: str,
    settings: ProviderSettings | None = None,
    provider: WorkspaceBackendProvider | None = None,
) -> WorkspaceBackendProvider:
    effective_settings = settings
    if provider is not None and effective_settings is None:
        effective_settings = getattr(provider, "settings", None)
    _validate_settings_for_backend(backend, effective_settings)
    if provider is not None:
        if settings is not None and provider.settings is None:
            provider.settings = settings
        elif settings is not None and provider.settings is not None:
            if provider.settings != settings:
                raise WorkspaceProviderError(
                    "Conflicting settings: both an injected provider and settings=... were passed."
                )
        _validate_injected_provider(backend, provider, effective_settings)
        return provider
    if backend == "local_fs":
        return LocalFilesystemProvider()
    if backend in {"ephemeral_fs", "s3_snapshot"}:
        return EphemeralFilesystemProvider()
    if backend == "postgres_vfs":
        return PostgresVFSProvider(settings)  # type: ignore[arg-type]
    if backend == "bubblewrap_process":
        from andromeda.workspace.sandbox_providers import BubblewrapProcessProvider

        return BubblewrapProcessProvider(settings)  # type: ignore[arg-type]
    if backend == "gvisor_container":
        from andromeda.workspace.sandbox_providers import GVisorContainerProvider

        return GVisorContainerProvider(settings)  # type: ignore[arg-type]
    if backend == "microvm":
        if isinstance(settings, NerdctlDevSettings):
            return NerdctlKataDevProvider(settings)
        if isinstance(settings, ContainerdKataSettings):
            return ContainerdKataSandboxProvider(settings)
        raise WorkspaceProviderError("microvm settings type could not be resolved.")
    raise WorkspaceProviderError(f"Unsupported workspace backend: {backend}")


class NotImplementedMicroVMProvider(ContainerdKataSandboxProvider):
    """Backward-compatible alias that now fails with a configuration error."""

    def __init__(self) -> None:
        super().__init__(None)
