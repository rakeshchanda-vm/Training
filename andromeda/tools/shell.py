from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import shlex
import signal
import subprocess
import threading
import uuid
from typing import Any, List, Optional, Sequence

from langchain_core.tools import tool
from andromeda.utils.ignore_rules import (
    IgnoreMatcher,
    ensure_ripgrep_ignore_config,
    ripgrep_config_workspace_path,
)


logger = logging.getLogger(__name__)


SHELL_READ_COMMANDS: frozenset[str] = frozenset(
    {
        "awk",
        "cat",
        "du",
        "find",
        "grep",
        "head",
        "less",
        "ls",
        "more",
        "rg",
        "ripgrep",
        "sed",
        "tail",
        "tree",
    }
)


def _build_env(policy: Any) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in set(policy.env_allowlist)
    }
    env.update({str(key): str(value) for key, value in policy.extra_env.items()})
    return env


def _with_shell_read_ignore_env(
    env: dict[str, str],
    workspace_root: str | Path,
    *,
    sandbox_home: str | None = None,
) -> dict[str, str]:
    if "RIPGREP_CONFIG_PATH" in env:
        return env
    try:
        config_path = ensure_ripgrep_ignore_config(workspace_root)
    except (OSError, UnicodeDecodeError):
        return env
    if config_path is None:
        return env
    env["RIPGREP_CONFIG_PATH"] = (
        ripgrep_config_workspace_path(sandbox_home)
        if sandbox_home is not None
        else str(config_path)
    )
    return env


def _build_workspace_env(policy: Any, workspace_root: str | Path) -> dict[str, str]:
    return _with_shell_read_ignore_env(_build_env(policy), workspace_root)


def _ignored_shell_read_target(
    argv: Sequence[str],
    workspace_root: str | Path,
    *,
    sandbox_home: str | None = None,
) -> str | None:
    if not argv or Path(argv[0]).name not in SHELL_READ_COMMANDS:
        return None

    root = Path(workspace_root).expanduser().resolve()
    matcher = IgnoreMatcher.for_filesystem(root)
    for arg in argv[1:]:
        if not arg or arg.startswith("-"):
            continue
        candidate = _shell_arg_to_host_path(arg, root, sandbox_home=sandbox_home)
        if candidate is None or not candidate.exists():
            continue
        if matcher.is_ignored(candidate, is_dir=candidate.is_dir()):
            return arg
    return None


def _shell_arg_to_host_path(
    arg: str,
    root: Path,
    *,
    sandbox_home: str | None = None,
) -> Path | None:
    raw_path = Path(arg).expanduser()
    if raw_path.is_absolute():
        if sandbox_home is not None:
            normalized_home = sandbox_home.rstrip("/")
            raw_text = raw_path.as_posix()
            if raw_text == normalized_home:
                return root
            if raw_text.startswith(f"{normalized_home}/"):
                raw_path = Path(raw_text[len(normalized_home) + 1 :])
            else:
                return None
        else:
            candidate = raw_path.resolve(strict=False)
            try:
                candidate.relative_to(root)
            except ValueError:
                return None
            return candidate

    candidate = (root / raw_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _truncate_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n... output truncated, {omitted} characters omitted ..."


def _command_policy_kwargs(policy: Any) -> dict[str, Any]:
    from andromeda.workspace.sandbox import SANDBOX_DENIED_ARGV

    return {
        "allowed_commands": getattr(policy, "allowed_commands", None),
        "denied_commands": getattr(policy, "denied_commands", SANDBOX_DENIED_ARGV),
    }


def _validate_command_policy(argv: Sequence[str], policy: Any) -> None:
    from andromeda.workspace.sandbox import validate_shell_argv

    validate_shell_argv(argv, **_command_policy_kwargs(policy))


def _prepare_command(command: str, argv: Optional[List[str]], policy: Any) -> tuple[str | list[str], bool]:
    if argv:
        command_argv = [command, *argv]
        _validate_command_policy(command_argv, policy)
        return command_argv, False
    if policy.allow_raw_shell:
        if getattr(policy, "allowed_commands", None) is not None:
            raise ValueError(
                "ShellPolicy cannot use allow_raw_shell=True when allowed_commands is set."
            )
        _validate_command_policy(["sh"], policy)
        return command, True
    command_argv = shlex.split(command)
    if not command_argv:
        raise ValueError("command is empty.")
    _validate_command_policy(command_argv, policy)
    return command_argv, False


def _prepare_provider_argv(command: str, argv: Optional[List[str]], policy: Any) -> list[str]:
    run_command, use_shell = _prepare_command(command, argv, policy)
    if use_shell:
        return ["sh", "-lc", str(run_command)]
    if isinstance(run_command, str):
        return shlex.split(run_command)
    return list(run_command)


@dataclass
class ShellProcessRecord:
    process_id: str
    command: str
    started_at: datetime
    output_path: Path
    process: subprocess.Popen[str]
    output_handle: Any
    timeout_seconds: int
    completed_at: datetime | None = None
    exit_code: int | None = None
    killed: bool = False

    @property
    def running(self) -> bool:
        return self.exit_code is None


class WorkspaceShellProcessManager:
    """Thread-safe local background process registry for a workspace."""

    def __init__(self, root: str | Path, policy: Any):
        self.root = Path(root).expanduser().resolve()
        self.policy = policy
        self.output_dir = self.root / ".andromeda-shell"
        self._records: dict[str, ShellProcessRecord] = {}
        self._lock = threading.RLock()

    def _refresh_locked(self, record: ShellProcessRecord) -> None:
        if record.exit_code is not None:
            return
        exit_code = record.process.poll()
        elapsed = (datetime.now(timezone.utc) - record.started_at).total_seconds()
        if exit_code is None and elapsed > record.timeout_seconds:
            self._kill_locked(record)
            return
        if exit_code is not None:
            record.exit_code = exit_code
            record.completed_at = datetime.now(timezone.utc)
            try:
                record.output_handle.close()
            except Exception:
                pass

    def _kill_locked(self, record: ShellProcessRecord) -> None:
        if record.exit_code is not None:
            return
        try:
            if hasattr(os, "killpg"):
                os.killpg(record.process.pid, signal.SIGTERM)
            else:
                record.process.terminate()
        except ProcessLookupError:
            pass
        except Exception:
            record.process.terminate()
        try:
            record.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(record.process.pid, signal.SIGKILL)
                else:
                    record.process.kill()
            except ProcessLookupError:
                pass
            record.process.wait(timeout=2)
        record.exit_code = record.process.returncode
        record.completed_at = datetime.now(timezone.utc)
        record.killed = True
        try:
            record.output_handle.close()
        except Exception:
            pass

    def _running_count_locked(self) -> int:
        for record in self._records.values():
            self._refresh_locked(record)
        return sum(1 for record in self._records.values() if record.running)

    def start(self, command: str, argv: Optional[List[str]] = None) -> str:
        run_command, use_shell = _prepare_command(command, argv, self.policy)
        if not use_shell and isinstance(run_command, list):
            ignored_target = _ignored_shell_read_target(run_command, self.root)
            if ignored_target is not None:
                raise PermissionError(f"Shell read target is ignored: {ignored_target}")
        with self._lock:
            running_count = self._running_count_locked()
            if running_count >= self.policy.max_background_processes:
                raise RuntimeError(
                    f"Maximum background process limit reached: {self.policy.max_background_processes}"
                )
            process_id = uuid.uuid4().hex
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{process_id}.log"
            output_handle = output_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                run_command,
                cwd=str(self.root),
                env=_build_workspace_env(self.policy, self.root),
                shell=use_shell,
                text=True,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._records[process_id] = ShellProcessRecord(
                process_id=process_id,
                command=command,
                started_at=datetime.now(timezone.utc),
                output_path=output_path,
                process=process,
                output_handle=output_handle,
                timeout_seconds=self.policy.background_timeout_seconds,
            )
            return process_id

    def status(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(process_id)
            if record is None:
                raise KeyError(f"Unknown shell process: {process_id}")
            self._refresh_locked(record)
            elapsed = (datetime.now(timezone.utc) - record.started_at).total_seconds()
            return {
                "process_id": record.process_id,
                "command": record.command,
                "running": record.running,
                "exit_code": record.exit_code,
                "killed": record.killed,
                "started_at": record.started_at.isoformat(),
                "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                "elapsed_seconds": round(elapsed, 3),
                "output_path": str(record.output_path),
            }

    def output(self, process_id: str, max_chars: int | None = None) -> str:
        with self._lock:
            record = self._records.get(process_id)
            if record is None:
                raise KeyError(f"Unknown shell process: {process_id}")
            self._refresh_locked(record)
            try:
                record.output_handle.flush()
            except Exception:
                pass
            output = record.output_path.read_text(encoding="utf-8", errors="replace")
            return _truncate_output(output, max_chars or self.policy.max_output_chars)

    def kill(self, process_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._records.get(process_id)
            if record is None:
                raise KeyError(f"Unknown shell process: {process_id}")
            self._kill_locked(record)
            return self.status(process_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.status(process_id) for process_id in list(self._records)]

    def kill_all(self) -> None:
        with self._lock:
            for record in list(self._records.values()):
                self._kill_locked(record)


def make_shell_tools(
    root: str | Path,
    policy: Any | None = None,
    *,
    process_manager: WorkspaceShellProcessManager | None = None,
) -> dict[str, object]:
    workspace_root = Path(root).expanduser().resolve()
    if not workspace_root.is_dir():
        raise ValueError(f"Shell workspace root does not exist: {workspace_root}")

    from andromeda.workspace.policy import ShellPolicy

    shell_policy = policy or ShellPolicy()
    manager = process_manager or WorkspaceShellProcessManager(workspace_root, shell_policy)

    @tool
    def shell(command: str, argv: Optional[List[str]] = None) -> str:
        """
        Execute a non-interactive command in the workspace root.

        The command always runs with cwd fixed to the workspace root. Pass argv
        for argv-style execution, or pass a command string that can be tokenized
        with shlex. Raw shell mode is disabled unless policy explicitly enables it.
        """
        try:
            run_command, use_shell = _prepare_command(command, argv, shell_policy)
        except ValueError as exc:
            return f"Error executing command: {exc}"
        if not use_shell and isinstance(run_command, list):
            ignored_target = _ignored_shell_read_target(run_command, workspace_root)
            if ignored_target is not None:
                return f"Error executing command: Shell read target is ignored: {ignored_target}"

        logger.info("Running workspace shell command", extra={"cwd": str(workspace_root)})
        try:
            completed = subprocess.run(
                run_command,
                cwd=str(workspace_root),
                env=_build_workspace_env(shell_policy, workspace_root),
                shell=use_shell,
                text=True,
                capture_output=True,
                timeout=shell_policy.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return (
                f"Command timed out after {shell_policy.timeout_seconds} seconds.\n"
                + _truncate_output(output, shell_policy.max_output_chars)
            ).strip()
        except Exception as exc:
            return f"Error executing command: {exc}"

        logger.info(
            "Workspace shell command finished",
            extra={"cwd": str(workspace_root), "exit_code": completed.returncode},
        )
        output = completed.stdout
        if completed.stderr:
            output += ("\n" if output else "") + completed.stderr
        output = _truncate_output(output, shell_policy.max_output_chars).strip()
        prefix = f"exit_code={completed.returncode}"
        return f"{prefix}\n{output}" if output else prefix

    tools: dict[str, object] = {"shell": shell}

    if shell_policy.enable_background_shell:
        @tool
        def shell_start(command: str, argv: Optional[List[str]] = None) -> str:
            """Start a non-interactive background command in the workspace root."""
            try:
                process_id = manager.start(command, argv)
                return f"Started shell process {process_id}"
            except Exception as exc:
                return f"Error starting shell process: {exc}"

        @tool
        def shell_status(process_id: str) -> str:
            """Get status for a background shell process."""
            try:
                import json

                return json.dumps(manager.status(process_id), indent=2, sort_keys=True)
            except Exception as exc:
                return f"Error reading shell process status: {exc}"

        @tool
        def shell_output(process_id: str, max_chars: Optional[int] = None) -> str:
            """Read captured output for a background shell process."""
            try:
                return manager.output(process_id, max_chars)
            except Exception as exc:
                return f"Error reading shell process output: {exc}"

        @tool
        def shell_kill(process_id: str) -> str:
            """Terminate a background shell process."""
            try:
                import json

                return json.dumps(manager.kill(process_id), indent=2, sort_keys=True)
            except Exception as exc:
                return f"Error killing shell process: {exc}"

        @tool
        def shell_list() -> str:
            """List background shell processes for this workspace."""
            try:
                import json

                return json.dumps(manager.list(), indent=2, sort_keys=True)
            except Exception as exc:
                return f"Error listing shell processes: {exc}"

        tools.update(
            {
                "shell_start": shell_start,
                "shell_status": shell_status,
                "shell_output": shell_output,
                "shell_kill": shell_kill,
                "shell_list": shell_list,
            }
        )

    return tools


def _provider_sandbox_home(provider: Any) -> str:
    """Sandbox-internal HOME path for provider-backed shells."""
    settings = getattr(provider, "settings", None)
    if settings is not None:
        if hasattr(settings, "workspace_path"):
            return str(settings.workspace_path)
        if hasattr(settings, "workspace_mount"):
            return str(settings.workspace_mount)
    return "/workspace"


def _provider_exec_cwd(provider: Any) -> str:
    """Host-relative cwd for provider.exec (mapped inside the sandbox by the provider)."""
    backend = getattr(getattr(provider, "state", None), "backend", None)
    if backend in {"bubblewrap_process", "gvisor_container"}:
        return "."
    return _provider_sandbox_home(provider)


def _provider_uses_sandbox_env(provider: Any) -> bool:
    backend = getattr(getattr(provider, "state", None), "backend", None)
    return backend in {"bubblewrap_process", "gvisor_container", "microvm"}


def _provider_shell_env(provider: Any, shell_policy: Any) -> dict[str, str]:
    state = getattr(provider, "state", None)
    workspace_root = getattr(state, "root", None)
    if _provider_uses_sandbox_env(provider):
        from andromeda.workspace.sandbox import build_sandbox_env

        sandbox_home = _provider_sandbox_home(provider)
        env = build_sandbox_env(shell_policy, home=sandbox_home)
        if workspace_root is not None:
            return _with_shell_read_ignore_env(
                env,
                workspace_root,
                sandbox_home=sandbox_home,
            )
        return env
    env = _build_env(shell_policy)
    if workspace_root is not None:
        return _with_shell_read_ignore_env(env, workspace_root)
    return env


def _provider_network_enabled(provider: Any, shell_policy: Any) -> bool:
    if shell_policy.network_enabled:
        return True
    settings = getattr(provider, "settings", None)
    if settings is not None and getattr(settings, "network", False):
        return True
    return False


def make_provider_shell_tools(provider: Any, policy: Any | None = None) -> dict[str, object]:
    from andromeda.workspace.policy import ShellPolicy

    shell_policy = policy or ShellPolicy()
    workspace_cwd = _provider_exec_cwd(provider)

    @tool
    def shell(command: str, argv: Optional[List[str]] = None) -> str:
        """Execute a non-interactive command through the workspace provider."""
        try:
            command_argv = _prepare_provider_argv(command, argv, shell_policy)
            state = getattr(provider, "state", None)
            workspace_root = getattr(state, "root", None)
            if workspace_root is not None:
                ignored_target = _ignored_shell_read_target(
                    command_argv,
                    workspace_root,
                    sandbox_home=_provider_sandbox_home(provider),
                )
                if ignored_target is not None:
                    return (
                        "Error executing command: "
                        f"Shell read target is ignored: {ignored_target}"
                    )
            completed = provider.exec(
                command_argv,
                cwd=workspace_cwd,
                env=_provider_shell_env(provider, shell_policy),
                timeout=shell_policy.timeout_seconds,
                network=_provider_network_enabled(provider, shell_policy),
                **_command_policy_kwargs(shell_policy),
            )
            output = completed.stdout
            if completed.stderr:
                output += ("\n" if output else "") + completed.stderr
            output = _truncate_output(output, shell_policy.max_output_chars).strip()
            prefix = f"exit_code={completed.exit_code}"
            if completed.timed_out:
                prefix += " timed_out=true"
            return f"{prefix}\n{output}" if output else prefix
        except Exception as exc:
            return f"Error executing command: {exc}"

    tools: dict[str, object] = {"shell": shell}

    if shell_policy.enable_background_shell:
        backend = getattr(getattr(provider, "state", None), "backend", None)
        if backend == "bubblewrap_process":
            return tools

        @tool
        def shell_start(command: str, argv: Optional[List[str]] = None) -> str:
            """Start a background command through the workspace provider."""
            try:
                command_argv = _prepare_provider_argv(command, argv, shell_policy)
                state = getattr(provider, "state", None)
                workspace_root = getattr(state, "root", None)
                if workspace_root is not None:
                    ignored_target = _ignored_shell_read_target(
                        command_argv,
                        workspace_root,
                        sandbox_home=_provider_sandbox_home(provider),
                    )
                    if ignored_target is not None:
                        return (
                            "Error starting shell process: "
                            f"Shell read target is ignored: {ignored_target}"
                        )
                process_id = provider.start(
                    command_argv,
                    cwd=workspace_cwd,
                    env=_provider_shell_env(provider, shell_policy),
                    **_command_policy_kwargs(shell_policy),
                )
                return f"Started shell process {process_id}"
            except Exception as exc:
                return f"Error starting shell process: {exc}"

        @tool
        def shell_status(process_id: str) -> str:
            """Read provider-backed background process status."""
            try:
                import json

                return json.dumps(provider.status(process_id), indent=2, sort_keys=True)
            except Exception as exc:
                return f"Error reading shell process status: {exc}"

        @tool
        def shell_output(process_id: str, max_chars: Optional[int] = None) -> str:
            """Read provider-backed background process output."""
            try:
                return provider.output(process_id, max_chars)
            except Exception as exc:
                return f"Error reading shell process output: {exc}"

        @tool
        def shell_kill(process_id: str) -> str:
            """Kill a provider-backed background process."""
            try:
                import json

                return json.dumps(provider.kill(process_id), indent=2, sort_keys=True)
            except Exception as exc:
                return f"Error killing shell process: {exc}"

        @tool
        def shell_list() -> str:
            """List provider-backed background processes."""
            try:
                import json

                return json.dumps(provider.list(), indent=2, sort_keys=True)
            except Exception as exc:
                return f"Error listing shell processes: {exc}"

        tools.update(
            {
                "shell_start": shell_start,
                "shell_status": shell_status,
                "shell_output": shell_output,
                "shell_kill": shell_kill,
                "shell_list": shell_list,
            }
        )

    return tools
