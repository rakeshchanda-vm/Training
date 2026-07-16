from __future__ import annotations

from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Callable, Collection, Sequence

_VALID_ENV_KEY_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

from andromeda.workspace.availability import ProviderAvailability
from andromeda.workspace.paths import workspace_relative_cwd
from andromeda.workspace.provider_settings import BubblewrapProcessSettings, GVisorContainerSettings
from andromeda.workspace.providers import (
    BaseWorkspaceProvider,
    ShellExecutionResult,
    WorkspaceProviderError,
    WorkspaceProviderState,
    _workspace_from_home,
)
from andromeda.workspace.sandbox import (
    SANDBOX_DENIED_ARGV,
    build_sandbox_env,
    log_shell_audit,
    make_shell_result,
    run_captured_command,
    validate_shell_argv,
)


def _validate_env_keys(env: dict[str, str]) -> None:
    for key in env:
        if not isinstance(key, str) or "\x00" in key or not _VALID_ENV_KEY_RE.fullmatch(key):
            raise WorkspaceProviderError(f"Invalid environment variable key: {key!r}")


def _default_ro_binds() -> list[tuple[str, str]]:
    binds: list[tuple[str, str]] = []
    for host_path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(host_path).exists():
            binds.append((host_path, host_path))
    return binds


def build_bwrap_command(
    settings: BubblewrapProcessSettings,
    host_workspace: Path,
    argv: Sequence[str],
    cwd: str,
    env: dict[str, str],
    *,
    network: bool,
    allowed_commands: Collection[str] | None = None,
    denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
) -> list[str]:
    validate_shell_argv(
        argv,
        allowed_commands=allowed_commands,
        denied_commands=denied_commands,
    )
    _validate_env_keys(env)
    sandbox_cwd = workspace_relative_cwd(
        host_workspace,
        cwd,
        workspace_mount=settings.workspace_mount,
    )
    command = [
        settings.bwrap_path,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    if network:
        command.append("--share-net")
    for key, value in env.items():
        command.extend(["--setenv", key, value])
    for host_path, sandbox_path in (*_default_ro_binds(), *settings.extra_ro_binds):
        command.extend(["--ro-bind", host_path, sandbox_path])
    if network:
        for host_path, sandbox_path in (("/etc/ssl", "/etc/ssl"), ("/etc/resolv.conf", "/etc/resolv.conf")):
            if Path(host_path).exists():
                command.extend(["--ro-bind", host_path, sandbox_path])
    command.extend(
        [
            "--bind",
            str(host_workspace),
            settings.workspace_mount,
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            sandbox_cwd,
            "--",
            *argv,
        ]
    )
    return command


class BubblewrapProcessProvider(BaseWorkspaceProvider):
    def __init__(
        self,
        settings: BubblewrapProcessSettings | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        super().__init__(settings or BubblewrapProcessSettings())
        self.runner = runner or subprocess.run

    def _settings(self) -> BubblewrapProcessSettings:
        if not isinstance(self.settings, BubblewrapProcessSettings):
            raise WorkspaceProviderError("Expected BubblewrapProcessSettings.")
        return self.settings

    @classmethod
    def check_available(cls, settings: BubblewrapProcessSettings | None = None) -> ProviderAvailability:
        cfg = settings or BubblewrapProcessSettings()
        details: dict[str, Any] = {}
        try:
            completed = subprocess.run(
                [cfg.bwrap_path, "--version"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return ProviderAvailability(
                available=False,
                provider="bubblewrap_process",
                reason=str(exc),
            )
        if completed.returncode != 0:
            return ProviderAvailability(
                available=False,
                provider="bubblewrap_process",
                reason=(completed.stderr or completed.stdout or "bwrap --version failed").strip(),
            )
        details["bwrap_version"] = (completed.stdout or completed.stderr).strip()
        max_user_ns = Path("/proc/sys/user/max_user_namespaces")
        if max_user_ns.exists():
            value = max_user_ns.read_text(encoding="utf-8").strip()
            details["max_user_namespaces"] = value
            if value in {"", "0"}:
                return ProviderAvailability(
                    available=False,
                    provider="bubblewrap_process",
                    reason="User namespaces are disabled (max_user_namespaces=0).",
                    details=details,
                )
        missing = [path for path in ("/usr", "/bin") if not Path(path).exists()]
        if missing:
            return ProviderAvailability(
                available=False,
                provider="bubblewrap_process",
                reason=f"Required host paths missing: {', '.join(missing)}",
                details=details,
            )
        from andromeda.workspace.policy import ShellPolicy

        smoke_argv = build_bwrap_command(
            cfg,
            Path.cwd(),
            ["/bin/sh", "-c", "echo ok"],
            ".",
            build_sandbox_env(ShellPolicy()),
            network=False,
        )
        try:
            smoke = subprocess.run(
                smoke_argv,
                text=True,
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return ProviderAvailability(
                available=False,
                provider="bubblewrap_process",
                reason=f"Bubblewrap smoke test failed: {exc}",
                details=details,
            )
        if smoke.returncode != 0 or "ok" not in smoke.stdout:
            return ProviderAvailability(
                available=False,
                provider="bubblewrap_process",
                reason=(smoke.stderr or smoke.stdout or "Bubblewrap smoke test failed").strip(),
                details=details,
            )
        return ProviderAvailability(available=True, provider="bubblewrap_process", details=details)

    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        settings = self._settings()
        if spec.get("root") is not None:
            root_path = Path(spec["root"]).expanduser().resolve()
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
            backend="bubblewrap_process",
            root=root_path,
            owns_root=owns_root,
            session_id=session_id,
            metadata_path=metadata_path,
            materialized_path=root_path,
            supports_shell=True,
            provider_backed_shell=True,
            metadata={"provider": "bubblewrap"},
        )
        return self.state

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
        if self.state is None:
            raise WorkspaceProviderError("Bubblewrap provider has not been created.")
        settings = self._settings()
        effective_network = network or settings.network
        command = build_bwrap_command(
            settings,
            self.state.root,
            argv,
            cwd,
            env,
            network=effective_network,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        exit_code, stdout, stderr, timed_out, duration_ms = run_captured_command(
            command,
            timeout=timeout,
            max_output_bytes=settings.max_output_bytes,
            runner=self.runner,
        )
        sandbox_cwd = workspace_relative_cwd(
            self.state.root,
            cwd,
            workspace_mount=settings.workspace_mount,
        )
        log_shell_audit(
            backend="bubblewrap_process",
            session_id=self.state.session_id,
            argv=argv,
            cwd=sandbox_cwd,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            network_enabled=effective_network,
        )
        return make_shell_result(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
            command=command,
            cwd=sandbox_cwd,
        )

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> str:
        raise WorkspaceProviderError(
            "bubblewrap_process does not support background shell in v1."
        )


class GVisorContainerProvider(BaseWorkspaceProvider):
    def __init__(
        self,
        settings: GVisorContainerSettings | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        super().__init__(settings or GVisorContainerSettings())
        self.runner = runner or subprocess.run
        self.container_name: str | None = None

    def _settings(self) -> GVisorContainerSettings:
        if not isinstance(self.settings, GVisorContainerSettings):
            raise WorkspaceProviderError("Expected GVisorContainerSettings.")
        return self.settings

    def _docker_prefix(self) -> list[str]:
        return [self._settings().docker_path]

    @classmethod
    def check_available(cls, settings: GVisorContainerSettings | None = None) -> ProviderAvailability:
        cfg = settings or GVisorContainerSettings()
        details: dict[str, Any] = {}
        try:
            completed = subprocess.run(
                [cfg.docker_path, "--version"],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return ProviderAvailability(
                available=False,
                provider="gvisor_container",
                reason=str(exc),
            )
        if completed.returncode != 0:
            return ProviderAvailability(
                available=False,
                provider="gvisor_container",
                reason=(completed.stderr or completed.stdout or "docker --version failed").strip(),
            )
        details["docker_version"] = (completed.stdout or completed.stderr).strip()
        try:
            smoke = subprocess.run(
                [
                    cfg.docker_path,
                    "run",
                    "--rm",
                    "--runtime",
                    cfg.runtime,
                    "--network",
                    "none",
                    cfg.image,
                    "echo",
                    "ok",
                ],
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return ProviderAvailability(
                available=False,
                provider="gvisor_container",
                reason=f"gVisor smoke test failed: {exc}",
                details=details,
            )
        if smoke.returncode != 0 or "ok" not in smoke.stdout:
            return ProviderAvailability(
                available=False,
                provider="gvisor_container",
                reason=(
                    smoke.stderr
                    or smoke.stdout
                    or f"Docker runtime {cfg.runtime!r} is not configured."
                ).strip(),
                details=details,
            )
        return ProviderAvailability(available=True, provider="gvisor_container", details=details)

    def _run(
        self,
        args: Sequence[str],
        *,
        timeout: int = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self.runner(
            list(args),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=check,
        )

    def create(self, spec: dict[str, Any]) -> WorkspaceProviderState:
        settings = self._settings()
        if spec.get("root") is not None:
            root_path = Path(spec["root"]).expanduser().resolve()
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
        self.container_name = settings.container_name or f"andromeda-{session_id}"
        network_mode = "bridge" if settings.network else "none"
        run_args = [
            *self._docker_prefix(),
            "run",
            "-d",
            "--runtime",
            settings.runtime,
            "--name",
            self.container_name,
            "--network",
            network_mode,
            "--memory",
            settings.memory,
            "--cpus",
            settings.cpus,
            "--pids-limit",
            str(settings.pids_limit),
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{root_path}:{settings.workspace_path}",
            "-w",
            settings.workspace_path,
        ]
        for key, value in settings.labels.items():
            run_args.extend(["--label", f"{key}={value}"])
        run_args.extend(["--label", f"andromeda.session_id={session_id}"])
        if settings.read_only_rootfs:
            run_args.append("--read-only")
        run_args.extend([settings.image, "sleep", "infinity"])
        try:
            self._run(run_args, timeout=settings.create_timeout_seconds, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise WorkspaceProviderError(f"Failed to create gVisor container: {exc}") from exc
        self.state = WorkspaceProviderState(
            backend="gvisor_container",
            root=root_path,
            owns_root=owns_root,
            session_id=session_id,
            metadata_path=metadata_path,
            materialized_path=root_path,
            supports_shell=True,
            provider_backed_shell=True,
            sandbox_id=self.container_name,
            metadata={"provider": "gvisor", "container_name": self.container_name},
        )
        return self.state

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
        if self.container_name is None or self.state is None:
            raise WorkspaceProviderError("gVisor container has not been created.")
        settings = self._settings()
        validate_shell_argv(
            argv,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        _validate_env_keys(env)
        sandbox_cwd = workspace_relative_cwd(
            self.state.root,
            cwd,
            workspace_mount=settings.workspace_path,
        )
        env_args = [item for key, value in env.items() for item in ("-e", f"{key}={value}")]
        command = [
            *self._docker_prefix(),
            "exec",
            "-w",
            sandbox_cwd,
            *env_args,
            self.container_name,
            *argv,
        ]
        exit_code, stdout, stderr, timed_out, duration_ms = run_captured_command(
            command,
            timeout=timeout,
            max_output_bytes=5_000_000,
            runner=self.runner,
        )
        log_shell_audit(
            backend="gvisor_container",
            session_id=self.state.session_id,
            argv=argv,
            cwd=sandbox_cwd,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            network_enabled=network or settings.network,
        )
        return make_shell_result(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_ms=duration_ms,
            command=command,
            cwd=sandbox_cwd,
        )

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: dict[str, str],
        allowed_commands: Collection[str] | None = None,
        denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
    ) -> str:
        if self.container_name is None or self.state is None:
            raise WorkspaceProviderError("gVisor container has not been created.")
        import uuid

        settings = self._settings()
        validate_shell_argv(
            argv,
            allowed_commands=allowed_commands,
            denied_commands=denied_commands,
        )
        _validate_env_keys(env)
        sandbox_cwd = workspace_relative_cwd(
            self.state.root,
            cwd,
            workspace_mount=settings.workspace_path,
        )
        process_id = uuid.uuid4().hex
        command = " ".join(shlex.quote(arg) for arg in argv)
        shell_script = (
            "mkdir -p /tmp/andromeda-shell; "
            f"(cd {shlex.quote(sandbox_cwd)} && ({command}) "
            f"> /tmp/andromeda-shell/{process_id}.log 2>&1; "
            f"echo $? > /tmp/andromeda-shell/{process_id}.exit) & "
            f"echo $! > /tmp/andromeda-shell/{process_id}.pid"
        )
        env_args = [item for key, value in env.items() for item in ("-e", f"{key}={value}")]
        self._run(
            [
                *self._docker_prefix(),
                "exec",
                "-w",
                sandbox_cwd,
                *env_args,
                self.container_name,
                "sh",
                "-lc",
                shell_script,
            ]
        )
        return process_id

    def status(self, process_id: str) -> dict[str, Any]:
        if self.container_name is None:
            raise WorkspaceProviderError("gVisor container has not been created.")
        self._validate_process_id(process_id)
        script = (
            f"if test -f /tmp/andromeda-shell/{process_id}.exit; then "
            f"echo exit_code=$(cat /tmp/andromeda-shell/{process_id}.exit); "
            f"else echo running=true; fi"
        )
        completed = self._run(
            [*self._docker_prefix(), "exec", self.container_name, "sh", "-lc", script],
            check=False,
        )
        running = "running=true" in completed.stdout
        exit_code = None
        if "exit_code=" in completed.stdout:
            exit_code = int(completed.stdout.split("exit_code=", 1)[1].strip().splitlines()[0])
        return {"process_id": process_id, "running": running, "exit_code": exit_code}

    def output(self, process_id: str, max_chars: int | None = None) -> str:
        if self.container_name is None:
            raise WorkspaceProviderError("gVisor container has not been created.")
        self._validate_process_id(process_id)
        completed = self._run(
            [
                *self._docker_prefix(),
                "exec",
                self.container_name,
                "sh",
                "-lc",
                f"cat /tmp/andromeda-shell/{process_id}.log 2>/dev/null || true",
            ],
            check=False,
        )
        output = completed.stdout
        return output if max_chars is None or len(output) <= max_chars else output[:max_chars]

    def kill(self, process_id: str) -> dict[str, Any]:
        if self.container_name is None:
            raise WorkspaceProviderError("gVisor container has not been created.")
        self._validate_process_id(process_id)
        script = f"kill $(cat /tmp/andromeda-shell/{process_id}.pid) 2>/dev/null || true; echo killed"
        self._run(
            [*self._docker_prefix(), "exec", self.container_name, "sh", "-lc", script],
            check=False,
        )
        return {"process_id": process_id, "running": False, "killed": True}

    def list(self) -> list[dict[str, Any]]:
        return []

    @staticmethod
    def _validate_process_id(process_id: str) -> None:
        import re
        if not re.fullmatch(r'[0-9a-f]{32}', process_id):
            raise WorkspaceProviderError(f"Invalid process_id format: {process_id!r}")

    def destroy(self) -> None:
        if self.container_name is not None:
            self._run([*self._docker_prefix(), "rm", "-f", self.container_name], check=False)
            self.container_name = None
