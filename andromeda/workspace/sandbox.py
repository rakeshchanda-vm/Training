from __future__ import annotations

from pathlib import Path
import logging
import subprocess
import time
from typing import Any, Callable, Collection, Mapping, Sequence

from andromeda.workspace.policy import DEFAULT_DENIED_COMMANDS, ShellPolicy

logger = logging.getLogger(__name__)

SANDBOX_DEFAULT_ENV: dict[str, str] = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/workspace",
    "LANG": "C.UTF-8",
}

SANDBOX_DENIED_ARGV: frozenset[str] = frozenset(DEFAULT_DENIED_COMMANDS)


def build_sandbox_env(
    policy: ShellPolicy | Any,
    *,
    extra: Mapping[str, str] | None = None,
    home: str = "/workspace",
) -> dict[str, str]:
    """Build a sandbox environment without inheriting the host process env."""
    env = dict(SANDBOX_DEFAULT_ENV)
    env["HOME"] = home
    allowlist = set(policy.env_allowlist)
    for key in allowlist:
        if key in SANDBOX_DEFAULT_ENV:
            env[key] = SANDBOX_DEFAULT_ENV[key]
    env.update({str(key): str(value) for key, value in policy.extra_env.items()})
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def _command_basename(command: str) -> str:
    return Path(str(command)).name


def _normalize_command_set(commands: Collection[str] | None) -> set[str] | None:
    if commands is None:
        return None
    return {_command_basename(command) for command in commands}


def validate_shell_argv(
    argv: Sequence[str],
    *,
    allowed_commands: Collection[str] | None = None,
    denied_commands: Collection[str] | None = SANDBOX_DENIED_ARGV,
) -> None:
    """Validate a shell argv against optional command allow/deny lists."""
    if not argv:
        raise ValueError("argv is empty.")
    executable = _command_basename(argv[0])
    denied = _normalize_command_set(denied_commands) or set()
    if executable in denied:
        raise ValueError(f"Command {executable!r} is not allowed in sandbox execution.")
    allowed = _normalize_command_set(allowed_commands)
    if allowed is not None and executable not in allowed:
        raise ValueError(
            f"Command {executable!r} is not in the allowed shell command list."
        )


def reject_denied_argv(argv: Sequence[str]) -> None:
    """Reject obviously dangerous argv entries before sandbox execution."""
    validate_shell_argv(argv)


def run_captured_command(
    command: Sequence[str],
    *,
    timeout: int,
    max_output_bytes: int,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> tuple[int, str, str, bool, int]:
    """Run a command and return exit_code, stdout, stderr, timed_out, duration_ms."""
    from andromeda.workspace.providers import ShellExecutionResult

    run = runner or subprocess.run
    started = time.monotonic()
    timed_out = False
    try:
        completed = run(
            list(command),
            cwd=cwd,
            env=None if env is None else dict(env),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        exit_code = completed.returncode
        stdout = _truncate_bytes(completed.stdout or "", max_output_bytes)
        stderr = _truncate_bytes(completed.stderr or "", max_output_bytes)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = _truncate_bytes((exc.stdout or "") if isinstance(exc.stdout, str) else "", max_output_bytes)
        stderr = _truncate_bytes((exc.stderr or "") if isinstance(exc.stderr, str) else "", max_output_bytes)
        if not stderr:
            stderr = f"Command timed out after {timeout} seconds."
    duration_ms = int((time.monotonic() - started) * 1000)
    return exit_code, stdout, stderr, timed_out, duration_ms


def _truncate_bytes(text: str, limit: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text
    truncated = encoded[:limit].decode("utf-8", errors="replace")
    omitted = len(encoded) - limit
    return truncated + f"\n... output truncated, {omitted} bytes omitted ..."


def log_shell_audit(
    *,
    backend: str,
    session_id: str | None,
    argv: Sequence[str],
    cwd: str,
    exit_code: int,
    timed_out: bool,
    duration_ms: int,
    stdout: str,
    stderr: str,
    network_enabled: bool,
) -> None:
    logger.info(
        "Workspace sandbox command finished",
        extra={
            "backend": backend,
            "session_id": session_id,
            "argv": list(argv),
            "cwd": cwd,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "network_enabled": network_enabled,
        },
    )


def make_shell_result(
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    timed_out: bool,
    duration_ms: int,
    command: Sequence[str],
    cwd: str,
) -> "ShellExecutionResult":
    from andromeda.workspace.providers import ShellExecutionResult

    return ShellExecutionResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        duration_ms=duration_ms,
        command=tuple(command),
        cwd=cwd,
    )
