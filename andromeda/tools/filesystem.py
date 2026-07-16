import re
from langchain_core.tools import tool
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional, Union
import glob
from datetime import datetime
from charset_normalizer import from_path
from pydantic import BaseModel
from andromeda.utils.ignore_rules import IgnoreMatcher, manual_ignore_matches


_PATCH_FILE_HEADER_RE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$")

class FilesystemHelpers:
    allowed_directories: List[str] = []

    def __init__(
        self,
        allowed_directories: List[str],
        *,
        read_only: bool = False,
        max_file_size_mb: int = 10,
        allow_symlinks: bool = False,
        protect_root: bool = True,
        path_aliases: list[tuple[str, str]] | None = None,
    ):
        """
        Initialize filesystem tools with allowed directories.
        
        Args:
            allowed_directories: List of directories that are allowed to be accessed
        """
        self.read_only = read_only
        self.max_file_size_mb = max(1, max_file_size_mb)
        self.max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
        self.allow_symlinks = allow_symlinks
        self.protect_root = protect_root
        self.allowed_roots = [Path(d).expanduser().resolve() for d in allowed_directories]
        self.allowed_directories = [str(path) for path in self.allowed_roots]
        # Validate directories
        for directory in self.allowed_roots:
            if not directory.is_dir():
                raise ValueError(f"Directory {directory} does not exist or is not accessible. Allowed directories: {self.allowed_directories}")
        self.path_aliases = self._normalize_path_aliases(path_aliases or [])
        self.allowed_directories_display = [
            self.display_path(path) for path in self.allowed_roots
        ]
    
        
    # Security utilities
    def normalize_path(self, p: str) -> str:
        """Normalize a path consistently."""
        path = Path(p)
        anchor = path.anchor
        parts: list[str] = []

        for part in path.parts:
            if part in {"", ".", anchor}:
                continue
            if part == "..":
                if parts and parts[-1] != "..":
                    parts.pop()
                elif not anchor:
                    parts.append(part)
                continue
            parts.append(part)

        if anchor:
            return str(Path(anchor, *parts)) if parts else anchor
        return str(Path(*parts)) if parts else "."

    def expand_home(self, filepath: str) -> str:
        """Expand ~ to user's home directory."""
        if filepath.startswith("~/") or filepath == "~":
            return str(Path(filepath).expanduser())
        return filepath

    def _normalize_path_aliases(self, aliases: list[tuple[str, str]]) -> list[tuple[str, Path]]:
        normalized_aliases: list[tuple[str, Path]] = []
        for alias, target in aliases:
            alias_text = str(alias).replace("\\", "/").rstrip("/")
            if not alias_text.startswith("/"):
                alias_text = f"/{alias_text}"
            if not alias_text or alias_text == "/":
                raise ValueError("Path alias must be a non-root absolute path.")
            target_path = Path(target).expanduser().resolve(strict=False)
            if not any(self._is_within_root(target_path, root) for root in self.allowed_roots):
                raise ValueError(f"Path alias target {target_path} is not within allowed directories.")
            normalized_aliases.append((alias_text, target_path))
        return sorted(normalized_aliases, key=lambda item: len(item[0]), reverse=True)

    def alias_to_real_path(self, requested_path: str) -> str:
        raw = str(requested_path)
        raw_posix = raw.replace("\\", "/")
        for alias, target in self.path_aliases:
            if raw_posix == alias:
                return str(target)
            if raw_posix.startswith(f"{alias}/"):
                return str(target / raw_posix[len(alias) + 1 :])
        return raw

    def display_path(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve(strict=False)
        for alias, target in self.path_aliases:
            try:
                relative = resolved.relative_to(target)
            except ValueError:
                continue
            if str(relative) == ".":
                return alias
            return f"{alias}/{relative.as_posix()}"
        return str(path)

    def response_path(self, requested_path: str, resolved_path: str | Path) -> str:
        if self.path_aliases:
            return self.display_path(resolved_path)
        return str(requested_path)

    def sanitize_text(self, text: str) -> str:
        sanitized = str(text)
        for alias, target in self.path_aliases:
            sanitized = sanitized.replace(str(target), alias)
        for root in self.allowed_roots:
            sanitized = sanitized.replace(str(root), self.display_path(root))
        return sanitized
    
    def _is_within_root(self, candidate: Path, root: Path) -> bool:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            return False

    def _matching_root(self, candidate: Path) -> Path | None:
        for root in self.allowed_roots:
            if self._is_within_root(candidate, root):
                return root
        return None

    def ignore_matcher_for(self, candidate: str | Path) -> IgnoreMatcher:
        path = Path(candidate).expanduser().resolve(strict=False)
        root = self._matching_root(path)
        if root is None:
            root = self.allowed_roots[0]
        return IgnoreMatcher.for_filesystem(root)

    def relative_to_ignore_root(self, candidate: str | Path, matcher: IgnoreMatcher) -> str:
        path = Path(candidate).expanduser().resolve(strict=False)
        try:
            return path.relative_to(matcher.root).as_posix()
        except ValueError:
            return path.name

    def _reject_symlink_crossing(self, raw_candidate: Path, root: Path, display_path: str) -> None:
        if self.allow_symlinks:
            return

        probe = raw_candidate if raw_candidate.is_absolute() else root / raw_candidate
        probe = probe.absolute()
        checked: list[Path] = []
        while True:
            checked.append(probe)
            if probe == root or probe.parent == probe:
                break
            probe = probe.parent

        for path in reversed(checked):
            if path.is_symlink():
                raise ValueError(f"Path {display_path} crosses a symlink.")

    def validate_path(self, requested_path: str, allowed_directories: List[str] | None = None, must_exist: bool = False) -> str:
        """
        Validate that a path is within allowed directories.
        Accepts absolute paths or paths relative to the first allowed directory.
        Returns the real path if valid, otherwise raises an error.
        """
        if allowed_directories is None:
            allowed_roots = self.allowed_roots
        else:
            allowed_roots = [
                Path(path).expanduser().resolve()
                for path in allowed_directories
                if Path(path).expanduser().is_dir()
            ]
        allowed_abs = [str(root) for root in allowed_roots if root.is_dir()]
        if not allowed_abs:
            raise ValueError("No allowed directories are configured or accessible.")

        display_path = str(requested_path)
        expanded = self.alias_to_real_path(self.expand_home(display_path))
        normalized = self.normalize_path(expanded)
        primary_root = allowed_roots[0]

        if normalized in {"", ".", "/"}:
            raw_candidate = primary_root
        elif Path(normalized).is_absolute():
            raw_candidate = Path(normalized)
        else:
            raw_candidate = primary_root / normalized

        resolved_path = raw_candidate.expanduser().resolve(strict=False)
        matching_root = None
        for root in allowed_roots:
            if self._is_within_root(resolved_path, root):
                matching_root = root
                break
        if matching_root is None:
            allowed_display = [self.display_path(root) for root in allowed_roots if root.is_dir()]
            raise ValueError(
                f"Path {self.display_path(resolved_path)} is not within allowed directories: {allowed_display}"
            )

        self._reject_symlink_crossing(raw_candidate, matching_root, display_path)

        if must_exist and not resolved_path.exists():
            suggested_paths = []
            stem = resolved_path.stem
            for base in allowed_roots:
                for p in base.rglob(f"{stem}*"):
                    try:
                        if self.path_aliases:
                            suggested_paths.append(self.display_path(p))
                        else:
                            suggested_paths.append(str(p.relative_to(base)))
                    except ValueError:
                        continue
            if len(suggested_paths) > 0:
                raise ValueError(f"Path {self.display_path(resolved_path)} does not exist. Did you mean one of these: {suggested_paths}?")
            else:
                raise ValueError(f"Path {self.display_path(resolved_path)} does not exist.")
        return str(resolved_path)

    def ensure_writable(self, operation: str = "write") -> None:
        if self.read_only:
            raise PermissionError(f"Filesystem is read-only; cannot {operation}.")

    def ensure_read_size(self, file_path: str) -> None:
        path = Path(file_path)
        if path.is_file() and path.stat().st_size > self.max_file_size_bytes:
            raise ValueError(f"File exceeds configured read limit of {self.max_file_size_mb} MB: {self.display_path(file_path)}")

    def ensure_write_size(self, content: str | bytes) -> None:
        size = len(content if isinstance(content, bytes) else str(content).encode("utf-8"))
        if size > self.max_file_size_bytes:
            raise ValueError(f"Content exceeds configured write limit of {self.max_file_size_mb} MB.")

    def is_protected_root(self, path: str) -> bool:
        if not self.protect_root:
            return False
        normalized = Path(path).resolve(strict=False)
        return any(normalized == root for root in self.allowed_roots)

    # File info utilities
    def get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get detailed statistics about a file or directory."""
        stats = os.stat(file_path)
        return {
            "size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stats.st_atime).isoformat(),
            "isDirectory": Path(file_path).is_dir(),
            "isFile": Path(file_path).is_file(),
            "permissions": oct(stats.st_mode)[-3:],
        }

def make_filesystem_tools(
    allowed_dirs: list[str],
    *,
    read_only: bool = False,
    file_policy: Any | None = None,
    path_aliases: list[tuple[str, str]] | None = None,
):
    fs = FilesystemHelpers(
        allowed_dirs,
        read_only=read_only,
        max_file_size_mb=getattr(file_policy, "max_file_size_mb", 10),
        allow_symlinks=getattr(file_policy, "allow_symlinks", False),
        protect_root=getattr(file_policy, "protect_root", True),
        path_aliases=path_aliases,
    )

    def _strip_patch_fence(patch: str) -> str:
        text = (patch or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _resolve_patch_path(path_text: str) -> str:
        raw_path = (path_text or "").strip()
        if not raw_path:
            raise ValueError("Patch file path is required.")
        if "\x00" in raw_path:
            raise ValueError("Patch file path contains a null byte.")
        if raw_path.startswith("-") or raw_path == "--":
            raise ValueError(f"Patch file path must not look like an option: {raw_path}")

        return fs.validate_path(raw_path, must_exist=False)

    def _read_text_for_patch(path: str) -> str:
        if Path(path).is_symlink():
            raise ValueError(f"Refusing to patch symlink: {path}")
        with open(path, "rb") as f:
            data = f.read()
        if b"\x00" in data[:8192]:
            raise ValueError(f"Refusing to patch binary file: {path}")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Refusing to patch non-UTF-8 file: {path}") from exc

    def _find_unique_match(lines: list[str], needle: list[str]) -> int:
        if not needle:
            if not lines:
                return 0
            raise ValueError("Update hunk has no removable/context lines and cannot be placed safely.")

        matches = [
            i for i in range(0, len(lines) - len(needle) + 1)
            if lines[i:i + len(needle)] == needle
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Update hunk matched multiple locations exactly.")

        normalized_needle = [line.strip() for line in needle]
        fuzzy_matches = [
            i for i in range(0, len(lines) - len(needle) + 1)
            if [line.strip() for line in lines[i:i + len(needle)]] == normalized_needle
        ]
        if len(fuzzy_matches) == 1:
            return fuzzy_matches[0]
        if len(fuzzy_matches) > 1:
            raise ValueError("Update hunk matched multiple locations after whitespace normalization.")
        raise ValueError("Update hunk did not match file contents.")

    def _apply_update_hunks(original: str, hunks: list[list[str]]) -> str:
        lines = original.splitlines()
        had_trailing_newline = original.endswith("\n")

        for hunk in hunks:
            old_lines: list[str] = []
            new_lines: list[str] = []
            for line in hunk:
                if not line:
                    raise ValueError("Patch hunk line is missing a prefix.")
                prefix = line[0]
                text = line[1:]
                if prefix == " ":
                    old_lines.append(text)
                    new_lines.append(text)
                elif prefix == "-":
                    old_lines.append(text)
                elif prefix == "+":
                    new_lines.append(text)
                else:
                    raise ValueError(f"Invalid patch hunk line prefix: {prefix!r}")

            pos = _find_unique_match(lines, old_lines)
            lines[pos:pos + len(old_lines)] = new_lines

        output = "\n".join(lines)
        if had_trailing_newline or (lines and any(hunk[-1].startswith("+") for hunk in hunks if hunk)):
            output += "\n"
        return output

    def _parse_patch(patch: str) -> list[dict[str, Any]]:
        text = _strip_patch_fence(patch)
        lines = text.splitlines()
        if not lines or lines[0] != "*** Begin Patch":
            raise ValueError("Patch must start with '*** Begin Patch'.")
        if lines[-1] != "*** End Patch":
            raise ValueError("Patch must end with '*** End Patch'.")

        ops: list[dict[str, Any]] = []
        i = 1
        while i < len(lines) - 1:
            if not lines[i].strip():
                i += 1
                continue

            match = _PATCH_FILE_HEADER_RE.match(lines[i])
            if not match:
                raise ValueError(f"Expected file operation header, got: {lines[i]}")
            action, path_text = match.groups()
            i += 1

            if action == "Add":
                body: list[str] = []
                while i < len(lines) - 1 and not _PATCH_FILE_HEADER_RE.match(lines[i]):
                    if not lines[i].startswith("+"):
                        raise ValueError("Add File lines must start with '+'.")
                    body.append(lines[i][1:])
                    i += 1
                ops.append({"action": action, "path": path_text, "content": "\n".join(body) + ("\n" if body else "")})
                continue

            if action == "Delete":
                while i < len(lines) - 1 and not _PATCH_FILE_HEADER_RE.match(lines[i]):
                    if lines[i].strip():
                        raise ValueError("Delete File operation must not include hunk content.")
                    i += 1
                ops.append({"action": action, "path": path_text})
                continue

            hunks: list[list[str]] = []
            while i < len(lines) - 1 and not _PATCH_FILE_HEADER_RE.match(lines[i]):
                if not lines[i].startswith("@@"):
                    raise ValueError("Update File hunks must start with '@@'.")
                i += 1
                hunk: list[str] = []
                while (
                    i < len(lines) - 1
                    and not lines[i].startswith("@@")
                    and not _PATCH_FILE_HEADER_RE.match(lines[i])
                ):
                    hunk.append(lines[i])
                    i += 1
                if not hunk:
                    raise ValueError("Update File hunk is empty.")
                hunks.append(hunk)
            if not hunks:
                raise ValueError("Update File operation requires at least one hunk.")
            ops.append({"action": action, "path": path_text, "hunks": hunks})

        if not ops:
            raise ValueError("Patch contains no file operations.")
        return ops

    @tool
    def read_file(path: str, start_line: int = 0, end_line: int = 100) -> str:
        """
        Read the complete contents of a file from the file system.
        
        Use this tool when you need to examine the contents of a single file.
        Do not use this repeatedly for large files, as it may consume a lot of memory and possibly overflow your context.
        Only works within allowed directories.

        Be intelligent and proactive in making use of start_line and end_line to minimize memory and token usage.
        
        Args:
            path: Path to the file to read
            start_line: Line number to start reading from (0-based, optional, default is 0)
            end_line: Line number to end reading at (0-based, optional, default is 100)
        Returns:
            str: Content of the file
        """
        if path is None or str(path).strip() == "":
            return (
                "Error: read_file requires a file path. Use the given tools to "
                "inspect available directories and find a specific file before "
                "calling read_file again."
            )
        try:
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
            fs.ensure_read_size(valid_path)
        except Exception as e:
            return f"Error validating path: {fs.sanitize_text(str(e))}"

        resolved_path = Path(valid_path)
        if not resolved_path.exists():
            return f"File {fs.display_path(valid_path)} does not exist. Please check the directory."
        if not resolved_path.is_file():
            return f"Path {fs.response_path(path, valid_path)} is not a file. Please check the directory."
        if resolved_path.stat().st_size == 0:
            return f"File {fs.display_path(valid_path)} is empty."
        
        encoding_result = from_path(valid_path).best()
        encoding = encoding_result.encoding if encoding_result else 'utf-8'
        try:
            with open(valid_path, 'r', encoding=encoding) as f:
                contents = f.read()
                lines = contents.splitlines()
                
                # Validate and adjust line indices
                if start_line < 0:
                    start_line = 0
                if end_line < 0:
                    end_line = 0
                if start_line >= len(lines):
                    start_line = max(0, len(lines) - 1)
                if end_line > len(lines):
                    end_line = len(lines)
                if start_line > end_line:
                    return f"Error: Start line {start_line} is greater than end line {end_line}."
                
                # Extract the requested line range
                selected_lines = lines[start_line:end_line]
                if not selected_lines:
                    return f"No content found between lines {start_line} and {end_line}."
                
                return '\n'.join(selected_lines)
                
        except UnicodeDecodeError:
            pass

        return "Not a text file or encoding not supported."
    
    @tool
    def grep_file(pattern: str, path: str, ignore_patterns: Optional[List[str]] = None) -> str:
        """
        Search for a pattern return matching lines from all files in the specified path.
        
        Use ignore_patterns to ignore unneccesary files and directories to minimize memory and token usage.
        
        Args:
            pattern: The pattern to search for in the file
            path: Path to a specific file or directory to search in
            ignore_patterns: Optional list of glob-like syntax patterns to ignore
        Returns:
            str: The first 20 matching lines with 1 line above and 1 line below the matching line if found.
        """
        try:
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
            if Path(valid_path).is_file():
                fs.ensure_read_size(valid_path)
        except Exception as e:
            return f"Error validating path: {fs.sanitize_text(str(e))}"
        
        resolved_path = Path(valid_path)
        if not resolved_path.is_dir() and resolved_path.stat().st_size == 0:
            return f"File {fs.display_path(valid_path)} is empty."

        if resolved_path.is_dir():
            matching_lines = []
            ignore_matcher = fs.ignore_matcher_for(valid_path)

            def search_directory(directory_path, pattern, ignore_patterns, matching_lines):
                for root, dirs, files in os.walk(directory_path):
                    pruned_dirs = []
                    for directory in dirs:
                        child_directory_path = Path(root) / directory
                        relative_dir = fs.relative_to_ignore_root(
                            child_directory_path,
                            ignore_matcher,
                        )
                        if not fs.allow_symlinks and child_directory_path.is_symlink():
                            continue
                        if ignore_matcher.is_ignored(child_directory_path, is_dir=True):
                            continue
                        if manual_ignore_matches(directory, relative_dir, ignore_patterns):
                            continue
                        pruned_dirs.append(directory)
                    dirs[:] = pruned_dirs
                    for file in files:
                        try:
                            file_path = Path(root) / file
                            relative_file = fs.relative_to_ignore_root(file_path, ignore_matcher)
                            if ignore_matcher.is_ignored(file_path, is_dir=False):
                                continue
                            if manual_ignore_matches(file, relative_file, ignore_patterns):
                                continue
                            if not fs.allow_symlinks and file_path.is_symlink():
                                continue
                            fs.ensure_read_size(file_path)
                            encoding_result = from_path(file_path).best()
                            encoding = encoding_result.encoding if encoding_result else 'utf-8'
                            with open(file_path, 'r', encoding=encoding) as f:
                                contents = f.read()
                                lines = contents.splitlines()
                                file_matches = [
                                    f"{fs.display_path(file_path)},\n" +
                                    (f"Line {i}: {lines[i-1]}\n" if i > 0 else "") +
                                    f"Line {i+1}: {line}\n" +
                                    (f"Line {i+2}: {lines[i+2]}\n" if i+2 < len(lines) else "") 
                                    for i, line in enumerate(lines) if re.search(pattern, line, re.IGNORECASE)
                                ]
                                if file_matches:
                                    matching_lines.extend(file_matches)
                                    if len(matching_lines) >= 20:
                                        return True  # Signal to stop searching
                        except (UnicodeDecodeError, PermissionError, OSError):
                            pass
                return False
            
            matching_lines = []
            search_directory(valid_path, pattern, ignore_patterns, matching_lines)
                    
            if matching_lines:
                return "\n".join(matching_lines)
            else:
                return f"No matches found for pattern '{pattern}' in any file in directory '{fs.display_path(valid_path)}'."
        else:
            encoding_result = from_path(valid_path).best()
            encoding = encoding_result.encoding if encoding_result else 'utf-8'
            try:
                with open(valid_path, 'r', encoding=encoding) as f:
                    contents = f.read()
                    lines = contents.splitlines()
                    matching_lines = [f"Line {i+1}: {line}" for i, line in enumerate(lines) if re.search(pattern, line)]
                    if len(matching_lines) > 20:
                        matching_lines = matching_lines[:20] + [f"... {len(matching_lines) - 20} more lines truncated..."]
                    if matching_lines:
                        return "\n".join(matching_lines)
                    else:
                        return f"No matches found for pattern '{pattern}' in file '{fs.display_path(valid_path)}'."
            except UnicodeDecodeError:
                pass
        return "Not a valid directory, text file or encoding not supported."
    
    @tool
    def delete_file_content(path: str, line: int = None, lines: list = None, substring: str = None) -> str:
        """
        Delete content at specific line(s) from a file.
        Only works within allowed directories.
        Use with caution as it will delete content from a file without warning.

        Args:
            path: Path to the file
            line: line number to delete (0-based, optional)
            lines: List of line numbers to delete (0-based, optional)
            substring: If provided, only delete this substring within the specified line(s), not the entire line (optional)
        
        Returns:
            Operation result information
        """
        try:
            fs.ensure_writable("delete content")
            valid_path = fs.validate_path(path, fs.allowed_directories)
            resolved_path = Path(valid_path)
            if not resolved_path.exists():
                return f"Error: File '{fs.display_path(valid_path)}' does not exist."
                
            if not resolved_path.is_file():
                return f"Error: '{fs.display_path(valid_path)}' is not a file."
                
            with open(valid_path, 'r', encoding='utf-8', errors='replace') as file:
                file_lines = file.readlines()
            
            total_lines = len(file_lines)
            deleted_lines = []
            modified_lines = []
            
            # Handle substring deletion (doesn't delete entire lines)
            if substring is not None:
                # For multiple lines
                if lines is not None:
                    if not isinstance(lines, list):
                        return "Error: 'lines' parameter must be a list of integers."
                    
                    for r in lines:
                        if not isinstance(r, int) or r < 0:
                            return "Error: line numbers must be non-negative integers."
                            
                        if r < total_lines and substring in file_lines[r]:
                            original_line = file_lines[r]
                            file_lines[r] = file_lines[r].replace(substring, '')
                            # Ensure line ends with newline if original did
                            if original_line.endswith('\n') and not file_lines[r].endswith('\n'):
                                file_lines[r] += '\n'
                            modified_lines.append(r)
                
                # For single line
                elif line is not None:
                    if not isinstance(line, int) or line < 0:
                        return "Error: line number must be a non-negative integer."
                        
                    if line >= total_lines:
                        return f"Error: line {line} is out of range (file has {total_lines} lines)."
                        
                    if substring in file_lines[line]:
                        original_line = file_lines[line]
                        file_lines[line] = file_lines[line].replace(substring, '')
                        # Ensure line ends with newline if original did
                        if original_line.endswith('\n') and not file_lines[line].endswith('\n'):
                            file_lines[line] += '\n'
                        modified_lines.append(line)
                
                # For entire file
                else:
                    for i in range(len(file_lines)):
                        if substring in file_lines[i]:
                            original_line = file_lines[i]
                            file_lines[i] = file_lines[i].replace(substring, '')
                            # Ensure line ends with newline if original did
                            if original_line.endswith('\n') and not file_lines[i].endswith('\n'):
                                file_lines[i] += '\n'
                            modified_lines.append(i)
                
                # Write back to the file
                with open(valid_path, 'w', encoding='utf-8') as file:
                    file.writelines(file_lines)
                    
                if not modified_lines:
                    return f"No occurrences of '{substring}' found in the specified lines."
                return f"Successfully removed '{substring}' from {len(modified_lines)} lines ({modified_lines}) in '{fs.response_path(path, valid_path)}'."
            
            # Handle deleting multiple lines
            elif lines is not None:
                if not isinstance(lines, list):
                    return "Error: 'lines' parameter must be a list of integers."
                    
                # Sort lines in descending order to avoid changing indices during deletion
                lines = sorted(lines, reverse=True)
                
                for r in lines:
                    if not isinstance(r, int) or r < 0:
                        return "Error: line numbers must be non-negative integers."
                        
                    if r < total_lines:
                        file_lines.pop(r)
                        deleted_lines.append(r)
                
                # Write back to the file
                with open(valid_path, 'w', encoding='utf-8') as file:
                    file.writelines(file_lines)
                    
                if not deleted_lines:
                    return f"No lines were within range to delete (file has {total_lines} lines)."
                return f"Successfully deleted {len(deleted_lines)} lines ({deleted_lines}) from '{fs.response_path(path, valid_path)}'."
                
            # Handle deleting a single line
            elif line is not None:
                if not isinstance(line, int) or line < 0:
                    return "Error: line number must be a non-negative integer."
                    
                if line >= total_lines:
                    return f"Error: line {line} is out of range (file has {total_lines} lines)."
                    
                # Delete the specified line
                file_lines.pop(line)
                
                # Write back to the file
                with open(valid_path, 'w', encoding='utf-8') as file:
                    file.writelines(file_lines)
                    
                return f"Successfully deleted line {line} from '{fs.response_path(path, valid_path)}'."
            
            # If neither line nor lines specified, clear the file
            else:
                with open(valid_path, 'w', encoding='utf-8') as file:
                    pass
                return f"Successfully cleared all content from '{fs.response_path(path, valid_path)}'."
                
        except PermissionError:
            return f"Error: No permission to modify file '{fs.sanitize_text(str(path))}'."
        except Exception as e:
            return f"Error deleting content: {fs.sanitize_text(str(e))}"
    @tool
    def update_file_content(path: str, content: str, line: int = None, lines: list = None, substring: str = None) -> str:
        """
        Update content at specific line(s) in a file
        
        Args:
            path: Path to the file
            content: New content to place at the specified line(s)
            line: line number to update (0-based, optional)
            lines: List of line numbers to update (0-based, optional)
            substring: If provided, only replace this substring within the specified line(s), not the entire line
        
        Returns:
            Operation result information
        """
        try:
            fs.ensure_writable("update content")
            valid_path = fs.validate_path(path, fs.allowed_directories)
            fs.ensure_write_size(content)
            # Handle different content types
            if not isinstance(content, str):
                try:
                    import json
                    content = json.dumps(content, indent=4, sort_keys=False, ensure_ascii=False, default=str)
                except Exception as e:
                    return f"Error: Unable to convert content to JSON string: {str(e)}"
                fs.ensure_write_size(content)
            
            resolved_path = Path(valid_path)
            if not resolved_path.exists():
                return f"Error: File '{fs.display_path(valid_path)}' does not exist."
                
            if not resolved_path.is_file():
                return f"Error: '{fs.display_path(valid_path)}' is not a file."
                
            with open(valid_path, 'r', encoding='utf-8', errors='replace') as file:
                file_lines = file.readlines()
            
            total_lines = len(file_lines)
            updated_lines = []
            
            # Ensure content ends with a newline if replacing a full line and doesn't already have one
            if substring is None and content and not content.endswith('\n'):
                content += '\n'
            
            # Prepare lines for update
            content_lines = content.splitlines(True) if substring is None else [content]
            
            # Handle updating multiple lines
            if lines is not None:
                if not isinstance(lines, list):
                    return "Error: 'lines' parameter must be a list of integers."
                    
                for r in lines:
                    if not isinstance(r, int) or r < 0:
                        return "Error: line numbers must be non-negative integers."
                        
                    if r < total_lines:
                        # If substring is provided, only replace that part
                        if substring is not None:
                            # Only update if substring exists in the line
                            if substring in file_lines[r]:
                                original_line = file_lines[r]
                                file_lines[r] = file_lines[r].replace(substring, content)
                                # Ensure line ends with newline if original did
                                if original_line.endswith('\n') and not file_lines[r].endswith('\n'):
                                    file_lines[r] += '\n'
                                updated_lines.append(r)
                        else:
                            # Otherwise, replace the entire line
                            # If we have multiple content lines, use them in sequence
                            if len(content_lines) > 1:
                                content_index = r % len(content_lines)
                                file_lines[r] = content_lines[content_index]
                            else:
                                # If we have only one content line, use it for all lines
                                file_lines[r] = content_lines[0] if content_lines else "\n"
                            updated_lines.append(r)
                
                # Write back to the file
                with open(valid_path, 'w', encoding='utf-8') as file:
                    file.writelines(file_lines)
                    
                if not updated_lines:
                    if substring is not None:
                        return f"No occurrences of substring '{substring}' found in the specified lines (file has {total_lines} lines)."
                    else:
                        return f"No lines were within range to update (file has {total_lines} lines)."
                
                if substring is not None:
                    return f"Successfully updated substring in {len(updated_lines)} lines ({updated_lines}) in '{fs.response_path(path, valid_path)}'."
                else:
                    return f"Successfully updated {len(updated_lines)} lines ({updated_lines}) in '{fs.response_path(path, valid_path)}'."
                
            # Handle updating a single line
            elif line is not None:
                if not isinstance(line, int) or line < 0:
                    return "Error: line number must be a non-negative integer."
                    
                if line >= total_lines:
                    return f"Error: line {line} is out of range (file has {total_lines} lines)."
                    
                # If substring is provided, only replace that part
                if substring is not None:
                    # Only update if substring exists in the line
                    if substring in file_lines[line]:
                        original_line = file_lines[line]
                        file_lines[line] = file_lines[line].replace(substring, content)
                        # Ensure line ends with newline if original did
                        if original_line.endswith('\n') and not file_lines[line].endswith('\n'):
                            file_lines[line] += '\n'
                    else:
                        return f"Substring '{substring}' not found in line {line}."
                else:
                    # Otherwise, replace the entire line
                    file_lines[line] = content_lines[0] if content_lines else "\n"
                
                # Write back to the file
                with open(valid_path, 'w', encoding='utf-8') as file:
                    file.writelines(file_lines)
                    
                if substring is not None:
                    return f"Successfully updated substring in line {line} in '{fs.response_path(path, valid_path)}'."
                else:
                    return f"Successfully updated line {line} in '{fs.response_path(path, valid_path)}'."
            
            # If neither line nor lines specified, update the entire file
            else:
                if substring is not None:
                    # Replace substring throughout the file
                    updated_count = 0
                    for i in range(len(file_lines)):
                        if substring in file_lines[i]:
                            original_line = file_lines[i]
                            file_lines[i] = file_lines[i].replace(substring, content)
                            # Ensure line ends with newline if original did
                            if original_line.endswith('\n') and not file_lines[i].endswith('\n'):
                                file_lines[i] += '\n'
                            updated_count += 1
                    
                    with open(valid_path, 'w', encoding='utf-8') as file:
                        file.writelines(file_lines)
                    
                    if updated_count == 0:
                        return f"Substring '{substring}' not found in any line of '{fs.response_path(path, valid_path)}'."
                    return f"Successfully updated substring in {updated_count} lines in '{fs.response_path(path, valid_path)}'."
                else:
                    # Replace entire file content
                    with open(valid_path, 'w', encoding='utf-8') as file:
                        file.write(content)
                    return f"Successfully updated all content in '{fs.response_path(path, valid_path)}'."
                
        except PermissionError:
            return f"Error: No permission to modify file '{fs.sanitize_text(str(path))}'."
        except Exception as e:
            return f"Error updating content: {fs.sanitize_text(str(e))}"
    
    @tool
    def write_file(path: str, content: str) -> str:
        """
        Create a new file or completely overwrite an existing file with new content.
        
        Use with caution as it will overwrite existing files without warning.
        Handles text content with proper encoding.
        
        Args:
            path: Path where the file should be written
            content: Content to write to the file
            
        Returns:
            str: Success message
        """
        try:
            fs.ensure_writable("write file")
            fs.ensure_write_size(content)
            valid_path = fs.validate_path(path, fs.allowed_directories)
        except Exception as e:
            return f"Error writing file: {fs.sanitize_text(str(e))}"

        Path(valid_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            encoding = from_path(valid_path).best()
            encoding = encoding.encoding if encoding else 'utf-8'
            if encoding.lower() == 'ascii':
                encoding = 'utf-8'
        except:
            encoding = 'utf-8'
        with open(valid_path, 'w', encoding=encoding) as f:
            f.write(content)
        return f"Successfully wrote to {fs.response_path(path, valid_path)}"
    
    @tool
    def search_and_replace_file_edit(path: str, search: str, replace: str) -> str:
        """
        Search and replace a string in a file.

        Args:
            path: Path to the file to edit
            search: String to search for
            replace: String to replace with
            
        Returns:
            str: Success message
            
        Use this tool when you need to search and replace a string in a file.
        Do not use this tool for large files, as it may consume a lot of memory and possibly overflow your context.
        Ensure search string is an exact match and unique in the file.
        Only works within allowed directories.
        """
        try:
            fs.ensure_writable("search and replace file")
            valid_path = fs.validate_path(path, fs.allowed_directories)
            encoding = from_path(valid_path).best()
            encoding = encoding.encoding if encoding else 'utf-8'
            if encoding.lower() == 'ascii':
                encoding = 'utf-8'
            fs.ensure_read_size(valid_path)
            with open(valid_path, 'r', encoding=encoding) as f:
                content = f.read()
            if search not in content:
                return f"'{search}' not found in {fs.response_path(path, valid_path)}"
            content = content.replace(search, replace)
            fs.ensure_write_size(content)
            with open(valid_path, 'w', encoding=encoding) as f:
                f.write(content)
            return f"Successfully searched and replaced '{search}' with '{replace}' in {fs.response_path(path, valid_path)}"
        except Exception as e:
            return f"Error editing file: {fs.sanitize_text(str(e))}"
    
    class EditFileArgs(BaseModel):
        path: str
        edits: List[Dict[str, str]]

    @tool(args_schema=EditFileArgs)
    def edit_file(path: str, edits: List[Dict[str, str]], dry_run: bool = False) -> str:
        """
        Make line-based edits to a text file.

        Args:
            path: Path to the file to edit/write
            edits: List of edits, where each edit is a dict with 'oldText' and 'newText' keys. 'oldText' can be empty if writing to a new file.
        Returns:
            str: Detailed diff of the changes made
        """
        import difflib
        import re
        import hashlib
        from collections import defaultdict
        
        def hash_line(line):
            """Create a fast hash for a line to use in indexing"""
            return hashlib.md5(line.encode('utf-8')).hexdigest()[:8]
        
        def tokenize(text):
            """Extract key tokens from text for faster matching"""
            # Remove common symbols and split by whitespace
            return re.sub(r'[^\w\s]', '', text.lower()).split()
        
        def create_line_index(lines):
            """Create index of distinctive lines for faster matching"""
            index = defaultdict(list)
            
            # Index only distinctive lines (skipping blank or common lines)
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('//') or line.startswith('#'):
                    continue
                    
                # Get distinctive tokens
                tokens = tokenize(line)
                if len(tokens) > 1:  # Skip lines with just one token as they're too common
                    # Use first and last token as key for more uniqueness
                    if len(tokens) > 2:
                        key = (tokens[0], tokens[-1])
                        index[key].append(i)
                    
                    # Also index by hash for exact matches
                    line_hash = hash_line(line)
                    index[line_hash].append(i)
            
            return index
        
        def find_potential_matches(text_lines, pattern_lines, line_index):
            """Find potential match positions using the index"""
            if not pattern_lines:
                return []
                
            # Get distinctive pattern line to search for
            pattern_idx = len(pattern_lines) // 2  # Use middle line as anchor
            pattern_line = pattern_lines[pattern_idx].strip()
            
            # Try to find matches by line hash
            line_hash = hash_line(pattern_line)
            positions = line_index.get(line_hash, [])
            
            # If no exact hash matches, try to find by tokens
            if not positions and pattern_line:
                tokens = tokenize(pattern_line)
                if len(tokens) > 2:
                    key = (tokens[0], tokens[-1])
                    positions = line_index.get(key, [])
            
            # Adjust positions to account for the anchor line's position in pattern
            return [pos - pattern_idx for pos in positions if 0 <= (pos - pattern_idx)]
        
        def score_match(pattern_lines, text_lines, start_pos):
            """Score how well a pattern matches text at a given position"""
            end_pos = min(start_pos + len(pattern_lines), len(text_lines))
            if end_pos - start_pos != len(pattern_lines):
                return 0.0
                
            matches = 0
            for i, (p_line, t_line) in enumerate(zip(pattern_lines, text_lines[start_pos:end_pos])):
                # Quickly check if lines are exactly the same
                if p_line.strip() == t_line.strip():
                    matches += 1
                elif i == 0 or i == len(pattern_lines) - 1:
                    # First and last lines are important - use sequence matcher for them
                    ratio = difflib.SequenceMatcher(None, p_line.strip(), t_line.strip()).ratio()
                    if ratio > 0.8:  # High threshold for anchor lines
                        matches += ratio
            
            return matches / len(pattern_lines)
        
        def get_indentation(line):
            """Get leading whitespace of a line"""
            return line[:len(line) - len(line.lstrip())]
        
        def apply_indentation(text, indent):
            """Apply indentation to all lines in text"""
            lines = text.splitlines()
            if not lines:
                return text
            return '\n'.join(indent + line if line.strip() else line for line in lines)
        
        try:
            fs.ensure_writable("edit file")
            # Validate path but don't require it to exist
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=False)
            resolved_path = Path(valid_path)
            
            # Create parent directories if they don't exist
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            if not resolved_path.exists():
                fs.ensure_write_size('\n'.join([edit['newText'] for edit in edits]))
                with open(valid_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join([edit['newText'] for edit in edits]))
                return f"Successfully created and wrote to {fs.response_path(path, valid_path)}"

            if len(edits) > 0 and not edits[0]['oldText']:
                return f"Error: Existing files must get oldText to identify where the changes should be made."
            # Read file efficiently
            encoding = from_path(valid_path).best()
            encoding = encoding.encoding if encoding else 'utf-8'
            if encoding.lower() == 'ascii':
                encoding = 'utf-8'
            with open(valid_path, 'r', encoding=encoding) as f:
                content_lines = f.read().splitlines()
            
            # Create line index for faster matching
            line_index = create_line_index(content_lines)
            
            # Process each edit, creating a set of operations to apply
            modifications = []
            for edit in edits:
                old_text = edit['oldText'].strip()
                new_text = edit['newText'].strip()
                old_lines = old_text.splitlines()
                
                if not old_lines:
                    continue
                
                # Find potential match positions using the index
                candidates = find_potential_matches(content_lines, old_lines, line_index)
                
                # Score candidates and find best match
                best_score = 0
                best_pos = -1
                
                for pos in candidates:
                    score = score_match(old_lines, content_lines, pos)
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                
                # Fall back to sequential search if indexing didn't find good matches
                if best_score < 0.7:
                    # Try a sequential search for the first line
                    first_line = old_lines[0].strip()
                    for i in range(len(content_lines) - len(old_lines) + 1):
                        if content_lines[i].strip() == first_line:
                            score = score_match(old_lines, content_lines, i)
                            if score > best_score:
                                best_score = score
                                best_pos = i
                
                if best_pos != -1 and best_score >= 0.7:  # Only if we found a good match
                    matched_length = len(old_lines)
                    base_indent = get_indentation(content_lines[best_pos])
                    new_text_lines = apply_indentation(new_text, base_indent).splitlines()
                    
                    # Store this modification
                    modifications.append({
                        'position': best_pos,
                        'old_lines': content_lines[best_pos:best_pos + matched_length],
                        'new_lines': new_text_lines,
                        'score': best_score,
                        'length': matched_length
                    })
            
            # Sort modifications in reverse order to prevent offset changes
            modifications.sort(key=lambda x: x['position'], reverse=True)
            
            # Create a copy of content lines to modify
            modified_lines = content_lines.copy()
            
            # Apply all modifications at once
            for mod in modifications:
                pos = mod['position']
                length = mod['length']
                modified_lines[pos:pos + length] = mod['new_lines']
            
            if not dry_run:
                fs.ensure_write_size('\n'.join(modified_lines))
                with open(valid_path, 'w', encoding=encoding) as f:
                    f.write('\n'.join(modified_lines))
            
            # Generate a summary of changes
            summary = ["Changes found:"]
            for mod in sorted(modifications, key=lambda x: x['position']):
                summary.append(f"- Match found at line {mod['position'] + 1} (similarity: {mod['score']:.2%})")
            

            
            return '\n'.join(summary)
            
        except Exception as e:
            return f"Error during file edit: {fs.sanitize_text(str(e))}"
    
    @tool
    def append_to_file(path: str, content: str) -> str:
        """
        Append content to a file.
        If the file does not exist, it will be created.
        If the file exists, the content will be appended to the end of the file.
        If the file exists and the content is the same as the existing content, the operation will succeed silently.

        Args:
            path: Full path to the file to append to
            content: Content to append to the file

        Returns:
            str: Success message
        """
        try:
            fs.ensure_writable("append to file")
            fs.ensure_write_size(content)
            valid_path = fs.validate_path(path, fs.allowed_directories)
        except Exception as e:
            return f"Error appending to file: {fs.sanitize_text(str(e))}"

        if not Path(valid_path).exists():
            with open(valid_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully created and wrote to {fs.response_path(path, valid_path)}"

        encoding = from_path(valid_path).best()
        encoding = encoding.encoding if encoding else 'utf-8'
        if encoding.lower() == 'ascii':
            encoding = 'utf-8'
        with open(valid_path, 'a', encoding=encoding) as f:
            f.write(content)

        return f"Successfully appended to {fs.response_path(path, valid_path)}"

    @tool
    def apply_patch(patch: str, dry_run: bool = False) -> str:
        """
        Apply a structured patch to create, update, or delete files.

        Use this tool for atomic, single or multi-file edits when you can identify the
        exact text to change. Prefer one patch containing all related edits over
        a sequence of separate write operations, unless you need iterative validation
        and changes.

        Usage guide:
        - Start with `*** Begin Patch` and end with `*** End Patch`.
        - Use `*** Add File: path` for new files. Every content line must start
          with `+`.
        - Use `*** Update File: path` for existing files. Each hunk starts with
          `@@`, then uses space-prefixed context lines, `-` lines to remove, and
          `+` lines to add.
        - Include enough unchanged context in update hunks to match exactly one
          location. If a hunk matches zero or multiple locations, the whole patch
          is rejected.
        - Use `*** Delete File: path` for text files that should be removed. Do
          not include hunk content for deletes.
        - Use `dry_run=True` only for broad or risky edits; it validates the
          patch and reports planned operations without writing files.

        Example patch:
        *** Begin Patch
        *** Add File: relative/or/absolute/path
        +new file line
        *** Update File: relative/or/absolute/path
        @@
         context line
        -old line
        +new line
        *** Delete File: relative/or/absolute/path
        *** End Patch

        Paths may be relative to the workspace root or absolute within allowed directories.
        The patch is atomic: if any operation fails, no files are changed.
        Symlinks, binary files, non-UTF-8 files, and paths outside allowed
        directories are rejected.
        """
        try:
            fs.ensure_writable("apply patch")
            ops = _parse_patch(patch)
            planned: dict[str, Optional[str]] = {}
            summaries: list[str] = []

            for op in ops:
                path = _resolve_patch_path(op["path"])
                action = op["action"]

                if action == "Add":
                    if Path(path).exists() and path not in planned:
                        raise ValueError(f"Add File target already exists: {op['path']}")
                    fs.ensure_write_size(op["content"])
                    planned[path] = op["content"]
                    summaries.append(f"add {fs.display_path(path)}")
                    continue

                if action == "Delete":
                    if path in planned:
                        current_content = planned[path]
                        if current_content is None:
                            raise ValueError(f"Delete File target already planned for deletion: {op['path']}")
                    else:
                        if not Path(path).is_file():
                            raise ValueError(f"Delete File target does not exist or is not a file: {op['path']}")
                        _read_text_for_patch(path)
                    planned[path] = None
                    summaries.append(f"delete {fs.display_path(path)}")
                    continue

                if path in planned:
                    base_content = planned[path]
                    if base_content is None:
                        raise ValueError(f"Update File target is already planned for deletion: {op['path']}")
                else:
                    if not Path(path).is_file():
                        raise ValueError(f"Update File target does not exist or is not a file: {op['path']}")
                    base_content = _read_text_for_patch(path)

                updated_content = _apply_update_hunks(base_content, op["hunks"])
                fs.ensure_write_size(updated_content)
                planned[path] = updated_content
                summaries.append(f"update {fs.display_path(path)}")

            if dry_run:
                return "Patch dry run succeeded:\n" + "\n".join(f"- {item}" for item in summaries)

            backups: dict[str, tuple[bool, Optional[bytes]]] = {}
            try:
                for path in planned:
                    if Path(path).exists():
                        with open(path, "rb") as f:
                            backups[path] = (True, f.read())
                    else:
                        backups[path] = (False, None)

                for path, content in planned.items():
                    if content is None:
                        os.remove(path)
                    else:
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "w", encoding="utf-8", newline="\n") as f:
                            f.write(content)
            except Exception:
                for path, (existed, data) in backups.items():
                    if existed and data is not None:
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(data)
                    elif not existed and Path(path).exists():
                        os.remove(path)
                raise

            return "Patch applied successfully:\n" + "\n".join(f"- {item}" for item in summaries)
        except Exception as e:
            return f"Error applying patch: {fs.sanitize_text(str(e))}"
    
    @tool
    def create_directory(path: str) -> str:
        """
        Create a new directory or ensure a directory exists.
        
        Can create multiple nested directories in one operation.
        If the directory already exists, this operation will succeed silently.
        
        Args:
            path: Relative Path of the directory to create
            
        Returns:
            str: Success message
        """
        try:
            fs.ensure_writable("create directory")
            valid_path = fs.validate_path(path, fs.allowed_directories)
            os.makedirs(valid_path, exist_ok=True)
            return f"Successfully created directory {fs.response_path(path, valid_path)}"
        except Exception as e:
            return f"Error creating directory: {fs.sanitize_text(str(e))}"
    
    @tool
    def list_directory(path: str, ignore: Optional[List[str]] = None) -> str:
        """
        Get a detailed listing of all files and directories in a specified path.
        
        Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes.
        
        Args:
            path: Relative Path to the directory to list
            ignore: Optional list of glob-like syntax patterns to ignore during the tree generation. Use this to ignore unneccesary files and directories.
        Returns:
            str: Formatted list of directory contents
        """
        try:
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
        except Exception as e:
            return f"Error validating path: {fs.sanitize_text(str(e))}"
        entries = os.listdir(valid_path)
        formatted = []
        ignore_matcher = fs.ignore_matcher_for(valid_path)
        
        for entry in entries:
            full_path = Path(valid_path) / entry
            relative_entry = fs.relative_to_ignore_root(full_path, ignore_matcher)
            is_dir = full_path.is_dir()
            if ignore_matcher.is_ignored(full_path, is_dir=is_dir):
                continue
            if manual_ignore_matches(entry, relative_entry, ignore):
                continue
            if not fs.allow_symlinks and full_path.is_symlink():
                formatted.append(f"[SYMLINK_REJECTED] {entry}")
                continue
            if is_dir:
                formatted.append(f"[DIR] {entry}")
            else:
                formatted.append(f"[FILE] {entry}")
        
        return "\n".join(formatted)
    
    @tool
    def directory_tree(path: str, ignore: Optional[List[str]] = None, max_depth: int = 3) -> str:
        """
        Get a recursive tree view of files and directories as a JSON structure.
        
        Each entry includes 'name', 'type' (file/directory), and 'children' for directories.
        Files have no children array, while directories always have a children array (which may be empty).
        
        Args:
            path: Path to get tree structure for. Use '/' to get the root directory.
            ignore: Optional list of glob-like syntax patterns to ignore during the tree generation. Use this to ignore unneccesary files and directories.
            max_depth: Optional maximum depth of the tree to generate. Use this to limit the depth of the tree to prevent context window issues.
        Returns:
            str: JSON formatted tree structure
        """
        try:
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
        except Exception as e:
            return f"Error validating path: {fs.sanitize_text(str(e))}"
        ignore_matcher = fs.ignore_matcher_for(valid_path)
        
        def build_tree(current_path, depth=0):
            if depth >= max_depth:
                return [f'[MAX_DEPTH_REACHED] Make another tool call with {fs.display_path(current_path)} as the path if needed.']
            result = []
            entries = os.listdir(current_path)
            
            for entry in entries:
                entry_path = Path(current_path) / entry
                relative_entry = fs.relative_to_ignore_root(entry_path, ignore_matcher)
                is_dir = entry_path.is_dir()
                if ignore_matcher.is_ignored(entry_path, is_dir=is_dir):
                    continue
                if manual_ignore_matches(entry, relative_entry, ignore):
                    continue
                if not fs.allow_symlinks and entry_path.is_symlink():
                    result.append({
                        "name": entry,
                        "type": "symlink_rejected",
                    })
                    continue
                entry_data = {
                    "name": entry,
                    "type": "directory" if is_dir else "file"
                }
                
                if is_dir:
                    entry_data["children"] = build_tree(entry_path, depth + 1)
                
                result.append(entry_data)
            
            return result
        
        tree_data = build_tree(valid_path)
        return json.dumps(tree_data, indent=2)
    
    @tool
    def search_files(path: str, pattern: str, exclude_patterns: Optional[List[str]] = None) -> str:
        """
        Recursively search for files and directories matching a pattern.
        
        Searches through all subdirectories from the starting path.
        The search is case-insensitive and matches partial names.
        
        Args:
            path: Root path to start searching from
            pattern: Pattern to search for in filenames with glob-like syntax (e.g., "*.txt" for all text files)
            exclude_patterns: Optional list of patterns to exclude
            
        Returns:
            str: Newline-separated list of matching files and directories
        """
        if exclude_patterns is None:
            exclude_patterns = []
            
        try:
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
        except Exception as e:
            return f"Error validating path: {fs.sanitize_text(str(e))}"
        results = []
        
        pattern = pattern.lower()
        ignore_matcher = fs.ignore_matcher_for(valid_path)
        
        for root, dirs, files in os.walk(valid_path):
            pruned_dirs = []
            for directory in dirs:
                directory_path = Path(root) / directory
                relative_dir = fs.relative_to_ignore_root(directory_path, ignore_matcher)
                if not fs.allow_symlinks and directory_path.is_symlink():
                    continue
                if ignore_matcher.is_ignored(directory_path, is_dir=True):
                    continue
                if manual_ignore_matches(directory, relative_dir, exclude_patterns):
                    continue
                pruned_dirs.append(directory)
            dirs[:] = pruned_dirs
            # Check if path should be excluded
            rel_path = Path(root).relative_to(valid_path)
            should_exclude = any(
                glob.fnmatch.fnmatch(str(rel_path), 
                                     ex_pattern if '*' in ex_pattern else f"**/{ex_pattern}/**")
                for ex_pattern in exclude_patterns
            )
            
            if should_exclude:
                continue
                
            # Check directories
            if ',' in pattern:
                patterns = pattern.split(',')
                for dir_name in dirs:
                    if any(glob.fnmatch.fnmatch(dir_name.lower(), pat) for pat in patterns):
                        results.append(fs.display_path(Path(root) / dir_name))
                for file_name in files:
                    file_path = Path(root) / file_name
                    relative_file = fs.relative_to_ignore_root(file_path, ignore_matcher)
                    if ignore_matcher.is_ignored(file_path, is_dir=False):
                        continue
                    if manual_ignore_matches(file_name, relative_file, exclude_patterns):
                        continue
                    if not fs.allow_symlinks and file_path.is_symlink():
                        continue
                    if any(glob.fnmatch.fnmatch(file_name.lower(), pat) for pat in patterns):
                        results.append(fs.display_path(file_path))
            else:
                for dir_name in dirs:
                    if glob.fnmatch.fnmatch(dir_name.lower(), pattern):
                        results.append(fs.display_path(Path(root) / dir_name))
                for file_name in files:
                    file_path = Path(root) / file_name
                    relative_file = fs.relative_to_ignore_root(file_path, ignore_matcher)
                    if ignore_matcher.is_ignored(file_path, is_dir=False):
                        continue
                    if manual_ignore_matches(file_name, relative_file, exclude_patterns):
                        continue
                    if not fs.allow_symlinks and file_path.is_symlink():
                        continue
                    if glob.fnmatch.fnmatch(file_name.lower(), pattern):
                        results.append(fs.display_path(file_path))
            
        return ("\n".join(results) if results else "No matches found")
    
    @tool
    def get_file_info(path: str) -> str:
        """
        Retrieve detailed metadata about a file or directory.
        
        Returns comprehensive information including size, creation time,
        last modified time, permissions, and type.
        
        Args:
            path: Path to the file or directory
            
        Returns:
            str: Formatted file information
        """
        try:
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
        except Exception as e:
            return f"Error validating path: {fs.sanitize_text(str(e))}"
        info = fs.get_file_stats(valid_path)
        
        return "\n".join(f"{key}: {value}" for key, value in info.items())
    
    @tool
    @staticmethod
    def list_allowed_directories() -> str:
        """
        Returns the list of directories that these tools are allowed to access.
        
        Use this to understand which directories are available before trying to access files.
        
        Returns:
            str: List of allowed directories
        """
        return "Allowed directories:\n" + "\n".join(fs.allowed_directories_display)
    
    @tool
    def delete_file_or_directory(path: str) -> str:
        """
        Delete a file or directory.
        Only works within allowed directories.
        Use with caution as it will delete a file or directory without warning.

        Args:
            path: Path to the file or directory to delete

        Returns:
            str: Success message
        """
        try:
            fs.ensure_writable("delete file or directory")
            valid_path = fs.validate_path(path, fs.allowed_directories, must_exist=True)
            if fs.is_protected_root(valid_path):
                return f"Error: Cannot delete allowed directory {fs.response_path(path, valid_path)}"
            resolved_path = Path(valid_path)
            if resolved_path.is_file():
                os.remove(valid_path)
            elif resolved_path.is_dir():
                if any(resolved_path == Path(allowed_dir) for allowed_dir in fs.allowed_directories):
                    return f"Error: Cannot delete allowed directory {fs.response_path(path, valid_path)}"
                else:
                    shutil.rmtree(valid_path)
            return f"Successfully deleted {fs.response_path(path, valid_path)}"
        except Exception as e:
            return f"Error deleting file or directory: {fs.sanitize_text(str(e))}"
    return {
        'read_file': read_file,
        'write_file': write_file,
        'search_and_replace_file_edit': search_and_replace_file_edit,
        'edit_file': edit_file,
        'append_to_file': append_to_file,
        'apply_patch': apply_patch,
        'list_directory': list_directory,
        'list_allowed_directories': list_allowed_directories,
        'directory_tree': directory_tree,
        'grep_file': grep_file,
        'search_files': search_files,
        'create_directory': create_directory,
        'delete_file_or_directory': delete_file_or_directory
    }
