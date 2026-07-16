from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Protocol

from andromeda.workspace.policy import FilePolicy


class WorkspaceSeed(Protocol):
    def apply(self, root: Path, policy: FilePolicy) -> None:
        """Apply the seed into a materialized workspace root."""


def _resolve_seed_path(root: Path, path: str) -> Path:
    workspace_root = root.expanduser().resolve()
    raw_path = str(path or "/")
    if "\x00" in raw_path:
        raise ValueError("Seed path contains a null byte.")

    if raw_path in {"", ".", "/"}:
        candidate = workspace_root
    else:
        candidate_path = Path(raw_path).expanduser()
        if candidate_path.is_absolute():
            candidate = candidate_path.resolve(strict=False)
        else:
            candidate = (workspace_root / candidate_path).resolve(strict=False)

    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Seed path {path!r} resolves outside the workspace root.") from exc
    return candidate


def _reject_symlink_crossing(root: Path, candidate: Path) -> None:
    workspace_root = root.expanduser().resolve()
    probe = candidate
    existing: list[Path] = []
    while True:
        existing.append(probe)
        if probe == workspace_root or probe.parent == probe:
            break
        probe = probe.parent
    for path in reversed(existing):
        if path.is_symlink():
            raise ValueError(f"Seed path {candidate} crosses a symlink.")


def _enforce_size(policy: FilePolicy, content: str | bytes) -> None:
    size = len(content if isinstance(content, bytes) else content.encode("utf-8"))
    if size > policy.max_file_size_bytes:
        raise ValueError(
            f"Seed content is larger than the configured limit of {policy.max_file_size_mb} MB."
        )


@dataclass(frozen=True)
class FileSeed:
    path: str
    content: str | bytes

    def apply(self, root: Path, policy: FilePolicy) -> None:
        target = _resolve_seed_path(root, self.path)
        if not policy.allow_symlinks:
            _reject_symlink_crossing(root, target)
        _enforce_size(policy, self.content)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(self.content, bytes):
            target.write_bytes(self.content)
        else:
            target.write_text(self.content, encoding="utf-8")


@dataclass(frozen=True)
class DirectorySeed:
    source_dir: str | Path
    target_path: str = "/"

    def apply(self, root: Path, policy: FilePolicy) -> None:
        source = Path(self.source_dir).expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Directory seed source does not exist: {source}")

        target = _resolve_seed_path(root, self.target_path)
        if not policy.allow_symlinks:
            _reject_symlink_crossing(root, target)
        target.mkdir(parents=True, exist_ok=True)

        for item in source.rglob("*"):
            relative = item.relative_to(source)
            destination = target / relative
            if item.is_symlink() and not policy.allow_symlinks:
                raise ValueError(f"Directory seed source crosses a symlink: {item}")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                if item.stat().st_size > policy.max_file_size_bytes:
                    raise ValueError(f"Seed file exceeds size limit: {item}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination, follow_symlinks=policy.allow_symlinks)


@dataclass(frozen=True)
class GitSeed:
    repo_url: str
    ref: str | None = None
    target_path: str = "/"
    depth: int | None = None
    timeout_seconds: int = 120

    def apply(self, root: Path, policy: FilePolicy) -> None:
        target = _resolve_seed_path(root, self.target_path)
        if not policy.allow_symlinks:
            _reject_symlink_crossing(root, target)
        target.parent.mkdir(parents=True, exist_ok=True)

        clone_cmd = ["git", "clone"]
        if self.depth is not None:
            clone_cmd.extend(["--depth", str(self.depth)])
        if self.ref and self.depth is not None:
            clone_cmd.extend(["--branch", self.ref])
        clone_cmd.extend([self.repo_url, str(target)])
        subprocess.run(
            clone_cmd,
            cwd=str(root),
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )

        if self.ref and self.depth is None:
            subprocess.run(
                ["git", "checkout", self.ref],
                cwd=str(target),
                check=True,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )


@dataclass(frozen=True)
class S3SnapshotSeed:
    uri: str

    def apply(self, root: Path, policy: FilePolicy) -> None:
        raise NotImplementedError("S3 snapshot seeding is not implemented in ET-Agentify phase 1.")


@dataclass(frozen=True)
class PostgresSnapshotSeed:
    namespace_key: str

    def apply(self, root: Path, policy: FilePolicy) -> None:
        raise NotImplementedError(
            "Postgres snapshot seeding is not implemented in ET-Agentify phase 1."
        )
