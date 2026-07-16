from __future__ import annotations

from pathlib import Path


def resolve_workspace_path(
    root: Path,
    user_path: str,
    *,
    reject_symlinks: bool = True,
) -> Path:
    """Resolve a user path relative to a workspace root, rejecting escapes."""
    workspace_root = root.expanduser().resolve()
    raw_path = str(user_path or ".")
    if "\x00" in raw_path:
        raise ValueError("Path contains a null byte.")

    candidate_path = Path(raw_path).expanduser()
    if candidate_path.is_absolute():
        candidate = candidate_path.resolve(strict=False)
    else:
        candidate = (workspace_root / candidate_path).resolve(strict=False)

    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"Path {user_path!r} resolves outside the workspace root."
        ) from exc

    if reject_symlinks:
        _reject_symlink_escape(workspace_root, candidate)

    return candidate


def workspace_relative_cwd(
    root: Path,
    cwd: str,
    *,
    workspace_mount: str = "/workspace",
    reject_symlinks: bool = True,
) -> str:
    """Map a workspace-relative cwd to a sandbox-internal path."""
    resolved = resolve_workspace_path(root, cwd, reject_symlinks=reject_symlinks)
    workspace_root = root.expanduser().resolve()
    relative = resolved.relative_to(workspace_root)
    if str(relative) == ".":
        return workspace_mount
    return f"{workspace_mount.rstrip('/')}/{relative.as_posix()}"


def _reject_symlink_escape(workspace_root: Path, candidate: Path) -> None:
    probe = candidate
    existing: list[Path] = []
    while True:
        existing.append(probe)
        if probe == workspace_root or probe.parent == probe:
            break
        probe = probe.parent
    for path in reversed(existing):
        if path.is_symlink():
            raise ValueError(f"Path {candidate} crosses a symlink.")
