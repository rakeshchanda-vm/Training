from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


IGNORE_FILE_NAMES: tuple[str, str] = (".gitignore", ".andromedaignore")
RIPGREP_CONFIG_RELATIVE_PATH = ".andromeda/rg-ignore-config"
RIPGREP_CONFIG_CONTENT = "--ignore-file\n.andromedaignore\n"


@dataclass(frozen=True)
class IgnoreRule:
    base_path: str
    pattern: str
    negated: bool = False
    directory_only: bool = False
    anchored: bool = False
    has_slash: bool = False


def parse_ignore_rules(lines: Iterable[str], *, base_path: str = "") -> list[IgnoreRule]:
    rules: list[IgnoreRule] = []
    normalized_base = _normalize_relative_path(base_path)
    for raw_line in lines:
        line = raw_line.rstrip("\n\r")
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("\\#") or line.startswith("\\!"):
            line = line[1:]

        negated = line.startswith("!")
        if negated:
            line = line[1:].lstrip()
        if not line:
            continue

        directory_only = line.endswith("/")
        line = line.rstrip("/")
        anchored = line.startswith("/")
        if anchored:
            line = line.lstrip("/")
        pattern = _normalize_relative_path(line)
        if not pattern:
            continue
        rules.append(
            IgnoreRule(
                base_path=normalized_base,
                pattern=pattern,
                negated=negated,
                directory_only=directory_only,
                anchored=anchored,
                has_slash="/" in pattern,
            )
        )
    return rules


class IgnoreMatcher:
    def __init__(self, root: Path, rules_by_dir: dict[str, list[IgnoreRule]]):
        self.root = root.expanduser().resolve()
        self._rules_by_dir = rules_by_dir

    @classmethod
    def for_filesystem(cls, root: str | Path) -> "IgnoreMatcher":
        return cls(Path(root), {})

    def is_ignored(self, path: str | Path, *, is_dir: bool) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            relative_path = candidate.relative_to(self.root).as_posix()
        except ValueError:
            return False
        if relative_path in {"", "."}:
            return False
        return self._is_relative_ignored(relative_path, is_dir=is_dir)

    def _is_relative_ignored(self, relative_path: str, *, is_dir: bool) -> bool:
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            return False

        parts = normalized.split("/")
        parent_count = len(parts) if is_dir else len(parts) - 1
        for index in range(1, parent_count + 1):
            ancestor = "/".join(parts[:index])
            if self._match_without_parent_check(ancestor, is_dir=True):
                return True
        return self._match_without_parent_check(normalized, is_dir=is_dir)

    def _match_without_parent_check(self, relative_path: str, *, is_dir: bool) -> bool:
        parent = _parent_relative_path(relative_path)
        ignored = False
        for rule in self._rules_for_dir(parent):
            if rule_matches(rule, relative_path, is_dir=is_dir):
                ignored = not rule.negated
        return ignored

    def _rules_for_dir(self, relative_dir: str) -> list[IgnoreRule]:
        normalized_dir = _normalize_relative_path(relative_dir)
        cached = self._rules_by_dir.get(normalized_dir)
        if cached is not None:
            return cached

        parent = _parent_relative_path(normalized_dir)
        rules = [] if normalized_dir == "" else list(self._rules_for_dir(parent))
        ignore_dir = self.root / normalized_dir if normalized_dir else self.root
        for name in IGNORE_FILE_NAMES:
            ignore_file = ignore_dir / name
            try:
                if ignore_file.is_file():
                    rules.extend(
                        parse_ignore_rules(
                            ignore_file.read_text(encoding="utf-8").splitlines(),
                            base_path=normalized_dir,
                        )
                    )
            except (OSError, UnicodeDecodeError):
                continue
        self._rules_by_dir[normalized_dir] = rules
        return rules


class VFSIgnoreMatcher:
    def __init__(self, driver: Any):
        self.driver = driver
        self._rules_by_dir: dict[str, list[IgnoreRule]] = {}

    def is_ignored(self, path: str, *, is_dir: bool) -> bool:
        normalized = _normalize_vfs_path(path)
        if normalized == "/":
            return False
        relative = normalized.lstrip("/")
        return self._is_relative_ignored(relative, is_dir=is_dir)

    def _is_relative_ignored(self, relative_path: str, *, is_dir: bool) -> bool:
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            return False

        parts = normalized.split("/")
        parent_count = len(parts) if is_dir else len(parts) - 1
        for index in range(1, parent_count + 1):
            ancestor = "/".join(parts[:index])
            if self._match_without_parent_check(ancestor, is_dir=True):
                return True
        return self._match_without_parent_check(normalized, is_dir=is_dir)

    def _match_without_parent_check(self, relative_path: str, *, is_dir: bool) -> bool:
        parent = _parent_relative_path(relative_path)
        ignored = False
        for rule in self._rules_for_dir(parent):
            if rule_matches(rule, relative_path, is_dir=is_dir):
                ignored = not rule.negated
        return ignored

    def _rules_for_dir(self, relative_dir: str) -> list[IgnoreRule]:
        normalized_dir = _normalize_relative_path(relative_dir)
        cached = self._rules_by_dir.get(normalized_dir)
        if cached is not None:
            return cached

        parent = _parent_relative_path(normalized_dir)
        rules = [] if normalized_dir == "" else list(self._rules_for_dir(parent))
        for name in IGNORE_FILE_NAMES:
            ignore_path = _join_vfs_path(normalized_dir, name)
            try:
                rules.extend(
                    parse_ignore_rules(
                        self.driver.read(ignore_path).content.splitlines(),
                        base_path=normalized_dir,
                    )
                )
            except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, ValueError):
                continue
        self._rules_by_dir[normalized_dir] = rules
        return rules


def rule_matches(rule: IgnoreRule, relative_path: str, *, is_dir: bool) -> bool:
    relative = _normalize_relative_path(relative_path)
    if not relative or not _is_under_base(relative, rule.base_path):
        return False
    if rule.directory_only and not is_dir:
        return False

    candidate = _relative_to_base(relative, rule.base_path)
    if not candidate:
        return False
    if rule.anchored or rule.has_slash:
        return _wildmatch(rule.pattern, candidate)
    return any(fnmatch.fnmatchcase(part, rule.pattern) for part in candidate.split("/"))


def manual_ignore_matches(
    name: str,
    relative_path: str,
    patterns: Iterable[str] | None,
) -> bool:
    if not patterns:
        return False
    normalized_path = _normalize_relative_path(relative_path)
    lower_name = name.lower()
    lower_path = normalized_path.lower()
    lower_absolute_path = f"/{lower_path}" if lower_path else "/"
    for pattern in patterns:
        lower_pattern = str(pattern).lower()
        if fnmatch.fnmatch(lower_name, lower_pattern):
            return True
        if fnmatch.fnmatch(lower_path, lower_pattern):
            return True
        if fnmatch.fnmatch(lower_absolute_path, lower_pattern):
            return True
    return False


def ensure_ripgrep_ignore_config(root: str | Path) -> Path | None:
    workspace_root = Path(root).expanduser().resolve()
    andromeda_ignore_path = workspace_root / ".andromedaignore"
    if not andromeda_ignore_path.is_file():
        return None
    config_path = workspace_root / RIPGREP_CONFIG_RELATIVE_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        not config_path.exists()
        or config_path.read_text(encoding="utf-8") != RIPGREP_CONFIG_CONTENT
    ):
        config_path.write_text(RIPGREP_CONFIG_CONTENT, encoding="utf-8")
    return config_path


def ripgrep_config_workspace_path(workspace_home: str = "/workspace") -> str:
    return f"{workspace_home.rstrip('/')}/{RIPGREP_CONFIG_RELATIVE_PATH}"


def _wildmatch(pattern: str, path: str) -> bool:
    pattern_parts = _normalize_relative_path(pattern).split("/")
    path_parts = _normalize_relative_path(path).split("/")
    return _match_segments(pattern_parts, path_parts)


def _match_segments(pattern_parts: list[str], path_parts: list[str]) -> bool:
    if not pattern_parts:
        return not path_parts
    head = pattern_parts[0]
    if head == "**":
        if len(pattern_parts) == 1:
            return True
        return any(
            _match_segments(pattern_parts[1:], path_parts[index:])
            for index in range(len(path_parts) + 1)
        )
    if not path_parts:
        return False
    if not fnmatch.fnmatchcase(path_parts[0], head):
        return False
    return _match_segments(pattern_parts[1:], path_parts[1:])


def _normalize_relative_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/")
    if raw.startswith("/"):
        raw = raw.lstrip("/")
    normalized = PurePosixPath(raw).as_posix()
    if normalized in {"", "."}:
        return ""
    parts = [part for part in normalized.split("/") if part and part != "."]
    return "/".join(parts)


def _normalize_vfs_path(path: str) -> str:
    raw = str(path or "/").replace("\\", "/")
    pure = PurePosixPath(raw if raw.startswith("/") else f"/{raw}")
    parts = [part for part in pure.as_posix().split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise ValueError("Path traversal is not allowed.")
    return "/" + "/".join(parts) if parts else "/"


def _parent_relative_path(path: str) -> str:
    normalized = _normalize_relative_path(path)
    if not normalized or "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def _is_under_base(relative_path: str, base_path: str) -> bool:
    if not base_path:
        return True
    return relative_path == base_path or relative_path.startswith(f"{base_path}/")


def _relative_to_base(relative_path: str, base_path: str) -> str:
    if not base_path:
        return relative_path
    if relative_path == base_path:
        return ""
    return relative_path[len(base_path) + 1 :]


def _join_vfs_path(relative_dir: str, name: str) -> str:
    normalized_dir = _normalize_relative_path(relative_dir)
    if not normalized_dir:
        return f"/{name}"
    return f"/{normalized_dir}/{name}"
