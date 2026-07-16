from __future__ import annotations

import contextvars
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

import yaml
from langchain.tools import BaseTool, tool
from langchain.messages import SystemMessage, ToolMessage
from langchain.agents.middleware.types import PrivateStateAttr
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langgraph.prebuilt import ToolRuntime
from langgraph.prebuilt.tool_node import ToolCallRequest

from andromeda.core.middleware.tooling import _coerce_tool_call_id
from andromeda.utils.logger import log_error, log_warning

# Security: Maximum size for SKILL.md files to prevent DoS attacks (10MB)
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills specification constraints (https://agentskills.io/specification)
MAX_SKILL_NAME_LENGTH = 64
MAX_SKILL_DESCRIPTION_LENGTH = 1024
MAX_SKILL_COMPATIBILITY_LENGTH = 500


class BackendDownloadResponse(Protocol):
    error: Any
    content: bytes | None


class BackendProtocol(Protocol):
    def ls_info(self, path: str) -> list[dict[str, Any]]: ...

    async def als_info(self, path: str) -> list[dict[str, Any]]: ...

    def download_files(self, paths: list[str]) -> list[BackendDownloadResponse]: ...

    async def adownload_files(self, paths: list[str]) -> list[BackendDownloadResponse]: ...


_skills_backend_ctx: contextvars.ContextVar[BackendProtocol | None] = contextvars.ContextVar(
    "_skills_backend_ctx",
    default=None,
)


ToolDef = BaseTool | dict[str, Any]


def _merge_unique_strings(
    left: list[str] | None,
    right: list[str] | None,
) -> list[str]:
    """Merge string lists for LangGraph state while preserving first-seen order."""

    merged: list[str] = []
    for items in (left or [], right or []):
        if isinstance(items, str):
            if items:
                merged.append(items)
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and item:
                merged.append(item)
    return list(dict.fromkeys(merged))


def _merge_skill_metadata_lists(
    left: list["SkillMetadata"] | None,
    right: list["SkillMetadata"] | None,
) -> list["SkillMetadata"]:
    """Merge skill metadata lists for concurrent LangGraph state updates."""

    def _side(label: str, value: list["SkillMetadata"] | None) -> list["SkillMetadata"]:
        if value is None:
            return []
        if not isinstance(value, list):
            log_warning(
                f"[skills] merge skills_metadata: {label} is not a list "
                f"(type={type(value).__name__}); treating as empty",
            )
            return []
        return value

    merged: dict[str, SkillMetadata] = {}
    for label, items in (("left", _side("left", left)), ("right", _side("right", right))):
        for item in items:
            if not isinstance(item, dict):
                log_warning(
                    f"[skills] merge skills_metadata: skipping non-dict item from {label} "
                    f"(type={type(item).__name__})",
                )
                continue
            key = str(item.get("name") or item.get("path") or "")
            if not key:
                log_warning(
                    "[skills] merge skills_metadata: skipping item with empty name and path "
                    f"(from {label})",
                )
                continue
            merged[key] = item
    return list(merged.values())


def _merge_dynamic_tool_dicts(
    left: dict[str, BaseTool] | None,
    right: dict[str, BaseTool] | None,
) -> dict[str, BaseTool]:
    """Merge dynamic tool registries for concurrent LangGraph state updates."""

    return operator.or_(left or {}, right or {})


class ToolDiscoveryFunc(Protocol):
    def __call__(
        self,
        *,
        skills: list["SkillMetadata"],
        state: "SkillsState",
        runtime: "Runtime",
        current_tools: list[ToolDef],
    ) -> list[ToolDef]: ...


class AsyncToolDiscoveryFunc(Protocol):
    async def __call__(
        self,
        *,
        skills: list["SkillMetadata"],
        state: "SkillsState",
        runtime: "Runtime",
        current_tools: list[ToolDef],
    ) -> list[ToolDef]: ...


@dataclass
class DownloadResponse:
    error: str | None = None
    content: bytes | None = None


class InMemoryBackend:
    """In-memory backend for skills content."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}

    def ls_info(self, path: str) -> list[dict[str, Any]]:
        path = path.rstrip("/")
        entries: dict[str, bool] = {}
        for file_path in self.files:
            if not file_path.startswith(path + "/"):
                continue
            rel = file_path[len(path) + 1 :]
            parts = rel.split("/")
            if not parts or not parts[0]:
                continue
            child_path = f"{path}/{parts[0]}"
            if len(parts) >= 2:
                entries[child_path] = True
            else:
                entries.setdefault(child_path, False)
        return [{"path": entry_path, "is_dir": is_dir} for entry_path, is_dir in sorted(entries.items())]

    async def als_info(self, path: str) -> list[dict[str, Any]]:
        return self.ls_info(path)

    def download_files(self, paths: list[str]) -> list[DownloadResponse]:
        out: list[DownloadResponse] = []
        for p in paths:
            if p not in self.files:
                out.append(DownloadResponse(error="not_found", content=None))
            else:
                out.append(DownloadResponse(error=None, content=self.files[p]))
        return out

    async def adownload_files(self, paths: list[str]) -> list[DownloadResponse]:
        return self.download_files(paths)


class FileSystemBackend:
    """Filesystem backend rooted at repo/workspace directory."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()

    def _to_local(self, virtual_path: str) -> Path:
        rel = virtual_path.lstrip("/")
        local = (self.root_dir / rel).resolve()
        if not str(local).startswith(str(self.root_dir)):
            raise ValueError(f"Path escapes backend root: {virtual_path}")
        return local

    def ls_info(self, path: str) -> list[dict[str, Any]]:
        local = self._to_local(path)
        if not local.exists() or not local.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for child in sorted(local.iterdir()):
            out.append(
                {
                    "path": f"{path.rstrip('/')}/{child.name}",
                    "is_dir": child.is_dir(),
                }
            )
        return out

    async def als_info(self, path: str) -> list[dict[str, Any]]:
        return self.ls_info(path)

    def download_files(self, paths: list[str]) -> list[DownloadResponse]:
        out: list[DownloadResponse] = []
        for p in paths:
            local = self._to_local(p)
            if not local.exists() or not local.is_file():
                out.append(DownloadResponse(error="not_found", content=None))
                continue
            out.append(DownloadResponse(error=None, content=local.read_bytes()))
        return out

    async def adownload_files(self, paths: list[str]) -> list[DownloadResponse]:
        return self.download_files(paths)


def append_to_system_message(system_message: Any, content: str) -> SystemMessage:
    base_content = getattr(system_message, "content", "")
    if isinstance(base_content, str):
        merged_content = f"{base_content}\n\n{content}" if base_content else content
    elif isinstance(base_content, list):
        merged_content = list(base_content) + [{"type": "text", "text": content}]
    else:
        merged_content = f"{base_content}\n\n{content}"
    return SystemMessage(content=merged_content)


class SkillMetadata(TypedDict):
    """Metadata for a skill per Agent Skills specification (https://agentskills.io/specification)."""

    path: str
    """Path to the SKILL.md file."""

    name: str
    """Skill identifier.

    Constraints per Agent Skills specification:

    - 1-64 characters
    - Unicode lowercase alphanumeric and hyphens only (`a-z` and `-`).
    - Must not start or end with `-`
    - Must not contain consecutive `--`
    - Must match the parent directory name containing the `SKILL.md` file
    """

    description: str
    """What the skill does.

    Constraints per Agent Skills specification:

    - 1-1024 characters
    - Should describe both what the skill does and when to use it
    - Should include specific keywords that help agents identify relevant tasks
    """

    license: str | None
    """License name or reference to bundled license file."""

    compatibility: str | None
    """Environment requirements.

    Constraints per Agent Skills specification:

    - 1-500 characters if provided
    - Should only be included if there are specific compatibility requirements
    - Can indicate intended product, required packages, etc.
    """

    metadata: dict[str, str]
    """Arbitrary key-value mapping for additional metadata.

    Clients can use this to store additional properties not defined by the spec.

    It is recommended to keep key names unique to avoid conflicts.
    """

    allowed_tools: list[str]
    """Tool names the skill recommends using.

    Warning: this is experimental.

    Constraints per Agent Skills specification:

    - Space-delimited list of tool names
    """


class SkillsState(AgentState):
    """State for the skills middleware."""

    skills_metadata: NotRequired[
        Annotated[list[SkillMetadata], PrivateStateAttr, _merge_skill_metadata_lists]
    ]
    """List of loaded skill metadata from configured sources. Not propagated to parent agents."""

    active_skills: NotRequired[Annotated[list[str], PrivateStateAttr, _merge_unique_strings]]
    """Loaded/active skills for this thread."""

    dynamic_skill_tools: NotRequired[
        Annotated[dict[str, BaseTool], PrivateStateAttr, _merge_dynamic_tool_dicts]
    ]
    """Runtime-discovered tool registry keyed by tool name."""


class SkillsStateUpdate(TypedDict):
    """State update for the skills middleware."""

    skills_metadata: list[SkillMetadata]
    """List of loaded skill metadata to merge into state."""
    active_skills: NotRequired[list[str]]
    dynamic_skill_tools: NotRequired[dict[str, BaseTool]]


def _skill_instruction_markdown_body(skill_md_text: str) -> str:
    """Return markdown body of SKILL.md without YAML frontmatter.

    Uses the same opening ``---`` / closing ``---`` rule as metadata discovery
    (:func:`_parse_skill_metadata`). When no valid frontmatter block is found,
    returns the file text as-is (aside from a leading BOM strip).
    """
    text = skill_md_text.lstrip("\ufeff")
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, text, re.DOTALL)
    if match:
        return text[match.end() :].lstrip("\n\r\t ")
    return text


def _download_backend_text(path: str, backend: BackendProtocol | None) -> tuple[str | None, str]:
    """Download and decode a UTF-8 text file via the skills backend."""
    if backend is None:
        return None, "skills backend unavailable"
    try:
        responses = backend.download_files([path])
    except ValueError as e:
        log_warning(f"download_files rejected path {path}: {e}")
        return None, "path rejected by backend"
    except Exception:
        log_error(f"backend.download_files failed for {path}")
        return None, "failed to read file via skills backend"

    if not responses:
        return None, "empty response from skills backend"
    resp = responses[0]
    err = getattr(resp, "error", None)
    if err:
        return None, str(err)
    raw = getattr(resp, "content", None)
    if raw is None:
        return None, "file had no readable content"
    if len(raw) > MAX_SKILL_FILE_SIZE:
        return None, (
            f"file exceeds configured maximum ({len(raw)} bytes; "
            f"limit {MAX_SKILL_FILE_SIZE})."
        )
    try:
        return raw.decode("utf-8"), ""
    except UnicodeDecodeError:
        return None, "file must be UTF-8"


def _download_skill_body_text(skill_md_path: str, backend: BackendProtocol | None) -> tuple[str | None, str]:
    """Download and decode SKILL.md for ``load_skill`` tool output."""
    text, err = _download_backend_text(skill_md_path, backend)
    if err == "skills backend unavailable":
        return None, "skills backend unavailable (cannot attach SKILL.md to this invocation)"
    if err == "failed to read file via skills backend":
        return None, "failed to read SKILL.md via skills backend"
    if err == "file had no readable content":
        return None, "SKILL.md had no readable content"
    if err == "file must be UTF-8":
        return None, "SKILL.md must be UTF-8"
    return text, err


def _active_skill_names_from_state(state: SkillsState) -> list[str]:
    """Return active skill names from middleware state."""
    names: list[str] = []
    for key in ("active_skills", "skills_loaded"):
        raw = state.get(key, []) if isinstance(state, dict) else getattr(state, key, [])
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item:
                    names.append(item)
    return list(dict.fromkeys(names))


def _get_skill_metadata_by_name(skills: list[SkillMetadata], skill_name: str) -> SkillMetadata | None:
    """Resolve skill metadata by skill name."""
    for skill in skills:
        if skill.get("name") == skill_name:
            return skill
    return None


def _normalize_skill_relative_path(relative_path: str) -> tuple[str | None, str]:
    """Validate a path relative to a skill directory."""
    raw = str(relative_path or "").strip()
    if not raw:
        return None, "relative_path is required"

    pure = PurePosixPath(raw)
    if pure.is_absolute():
        return None, "relative_path must be relative to the skill directory"

    parts = [part for part in pure.parts if part not in ("", ".")]
    if not parts:
        return None, "relative_path is required"
    if any(part == ".." for part in parts):
        return None, "relative_path must not escape the skill directory"

    normalized = PurePosixPath(*parts)
    if normalized.suffix.lower() != ".md":
        return None, "read_skill_file only supports markdown files"
    return str(normalized), ""


def _list_skill_markdown_files(skill_root: str, backend: BackendProtocol | None) -> list[str]:
    """Recursively list markdown files under a skill directory, excluding SKILL.md."""
    if backend is None:
        return []

    root = PurePosixPath(skill_root)
    pending = [root]
    seen: set[str] = set()
    markdown_files: list[str] = []

    while pending:
        current = pending.pop()
        current_str = str(current)
        if current_str in seen:
            continue
        seen.add(current_str)

        try:
            items = backend.ls_info(current_str)
        except Exception as exc:  # noqa: BLE001
            log_warning(f"Failed to list skill directory {current_str}: {exc}")
            continue

        for item in items:
            item_path = item.get("path")
            if not isinstance(item_path, str) or not item_path:
                continue
            item_pure = PurePosixPath(item_path)
            if item.get("is_dir"):
                pending.append(item_pure)
                continue
            if item_pure.suffix.lower() != ".md":
                continue
            try:
                rel = item_pure.relative_to(root)
            except ValueError:
                continue
            if str(rel) == "SKILL.md":
                continue
            markdown_files.append(str(rel))

    return list(dict.fromkeys(sorted(markdown_files)))


def _slice_text_by_lines(text: str, start_line: int = 0, end_line: int = 100) -> str:
    """Return a 0-based, end-exclusive line slice matching filesystem tool semantics."""
    if not text:
        return "File is empty."

    lines = text.splitlines()
    if not lines:
        return "File is empty."

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

    selected_lines = lines[start_line:end_line]
    if not selected_lines:
        return f"No content found between lines {start_line} and {end_line}."
    return "\n".join(selected_lines)


def _read_skill_file_text(
    *,
    skill_name: str,
    relative_path: str,
    runtime: ToolRuntime,
    start_line: int = 0,
    end_line: int = 100,
) -> str:
    """Read a markdown file from an active skill directory."""
    active_skills = set(_active_skill_names_from_state(runtime.state))
    if skill_name not in active_skills:
        return (
            f"Skill '{skill_name}' is not active. "
            f"Call load_skill(skill_name=\"{skill_name}\") first."
        )

    skills: list[SkillMetadata] = runtime.state.get("skills_metadata", [])
    skill = _get_skill_metadata_by_name(skills, skill_name)
    if skill is None:
        return f"Skill '{skill_name}' was not found in skills_metadata."

    normalized_path, err = _normalize_skill_relative_path(relative_path)
    if normalized_path is None:
        return f"Error: {err}"

    backend = _skills_backend_ctx.get()
    skill_root = PurePosixPath(skill["path"]).parent
    target_path = str(skill_root / normalized_path)
    text, read_err = _download_backend_text(target_path, backend)
    if text is None:
        available = _list_skill_markdown_files(str(skill_root), backend)
        available_msg = ""
        if available:
            available_msg = "\nAvailable markdown files:\n" + "\n".join(f"- `{path}`" for path in available)
        return (
            f"Error reading `{normalized_path}` from skill `{skill_name}`: {read_err}."
            f"{available_msg}"
        )

    return _slice_text_by_lines(text, start_line=start_line, end_line=end_line)


def _build_load_skill_command(skill_name: str, runtime: ToolRuntime) -> Command:
    """Build state update command for loading a skill."""

    backend = _skills_backend_ctx.get()
    tcid = runtime.tool_call_id or ""

    skills: list[SkillMetadata] = runtime.state.get("skills_metadata", [])
    available = [s["name"] for s in skills]
    if skill_name not in available:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"Skill '{skill_name}' not found. "
                            f"Available skills: {', '.join(sorted(available)) or '(none)'}"
                        ),
                        tool_call_id=tcid,
                        status="error",
                    )
                ]
            }
        )

    md_path = next(s["path"] for s in skills if s["name"] == skill_name)
    skill_body, dl_err = _download_skill_body_text(md_path, backend)
    supporting_markdown = _list_skill_markdown_files(str(PurePosixPath(md_path).parent), backend)

    lines = [f"Skill '{skill_name}' is activated for this thread.", ""]
    if skill_body is not None:
        instruction_md = _skill_instruction_markdown_body(skill_body)
        lines.extend(["<skill_instructions>", instruction_md, "</skill_instructions>"])
    else:
        lines.extend(
            [
                dl_err,
                "",
                f"Activation succeeded; SKILL.md lives at `{md_path}` (read manually if needed).",
            ]
        )
    if supporting_markdown:
        lines.extend(
            [
                "",
                "Supporting markdown files are available via `read_skill_file` if needed:",
                *[f"- `{path}`" for path in supporting_markdown],
            ]
        )

    msg = ToolMessage(
        content="\n".join(lines),
        tool_call_id=tcid,
        name="load_skill",
    )

    return Command(
        update={
            "messages": [msg],
            # Emit only the new skill. LangGraph applies the reducer on the
            # state field, which safely merges parallel load_skill updates.
            "active_skills": [skill_name],
        }
    )


@tool
def load_skill(skill_name: str, runtime: ToolRuntime) -> Command:
    """Load a skill: activate it and return the markdown instruction body (no YAML frontmatter).

    Prefer this over generic file-read tools so the skill is tracked as active
    and any ``allowed-tools`` gating applies. This tool is injected by
    ``SkillsMiddleware`` when skills are available.
    """
    return _build_load_skill_command(skill_name, runtime)


@tool
def read_skill_file(
    skill_name: str,
    relative_path: str,
    runtime: ToolRuntime,
    start_line: int = 0,
    end_line: int = 100,
) -> str:
    """Read a markdown file from an active skill directory.

    Use this for supporting docs such as `references/*.md` after activating the
    skill with `load_skill`. This keeps skills accessible even when generic
    filesystem tools are sandboxed elsewhere. DO NOT USE THIS TOOL TO LOAD A SKILL.
    """
    return _read_skill_file_text(
        skill_name=skill_name,
        relative_path=relative_path,
        runtime=runtime,
        start_line=start_line,
        end_line=end_line,
    )


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """Validate skill name per Agent Skills specification.

    Constraints per Agent Skills specification:

    - 1-64 characters
    - Unicode lowercase alphanumeric and hyphens only (`a-z` and `-`).
    - Must not start or end with `-`
    - Must not contain consecutive `--`
    - Must match the parent directory name containing the `SKILL.md` file

    Unicode lowercase alphanumeric means any character where `c.isalpha() and
    c.islower()` or `c.isdigit()` returns `True`, which covers accented Latin
    characters (e.g., `'café'`, `'über-tool'`) and other scripts.

    Args:
        name: Skill name from YAML frontmatter
        directory_name: Parent directory name

    Returns:
        `(is_valid, error_message)` tuple.

            Error message is empty if valid.
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, "name must be lowercase alphanumeric with single hyphens only"
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return False, "name must be lowercase alphanumeric with single hyphens only"
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(  # noqa: C901
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """Parse YAML frontmatter from `SKILL.md` content.

    Extracts metadata per Agent Skills specification from YAML frontmatter
    delimited by `---` markers at the start of the content.

    Args:
        content: Content of the `SKILL.md` file
        skill_path: Path to the `SKILL.md` file (for error messages and metadata)
        directory_name: Name of the parent directory containing the skill

    Returns:
        `SkillMetadata` if parsing succeeds, `None` if parsing fails or
            validation errors occur
    """
    if len(content) > MAX_SKILL_FILE_SIZE:
        log_warning(f"Skipping {skill_path}: content too large ({len(content)} bytes)")
        return None

    # Match YAML frontmatter between --- delimiters
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        log_warning(f"Skipping {skill_path}: no valid YAML frontmatter found")
        return None

    frontmatter_str = match.group(1)

    # Parse YAML using safe_load for proper nested structure support
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        log_warning(f"Invalid YAML in {skill_path}: {e}")
        return None

    if not isinstance(frontmatter_data, dict):
        log_warning(f"Skipping {skill_path}: frontmatter is not a mapping")
        return None

    name = str(frontmatter_data.get("name", "")).strip()
    description = str(frontmatter_data.get("description", "")).strip()
    if not name or not description:
        log_warning(f"Skipping {skill_path}: missing required 'name' or 'description'")
        return None

    # Validate name format per spec (warn but continue loading for backwards compatibility)
    is_valid, error = _validate_skill_name(str(name), directory_name)
    if not is_valid:
        log_warning(
            f"Skill '{name}' in {skill_path} does not follow Agent Skills specification: {error}. Consider renaming for spec compliance.",
        )

    description_str = description
    if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
        log_warning(
            f"Description exceeds {MAX_SKILL_DESCRIPTION_LENGTH} characters in {skill_path}, truncating",
        )
        description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

    raw_tools = frontmatter_data.get("allowed-tools")
    if isinstance(raw_tools, str):
        allowed_tools = [
            t.strip(",")  # Support commas for compatibility with skills created for Claude Code.
            for t in raw_tools.split()
            if t.strip(",")
        ]
    else:
        if raw_tools is not None:
            log_warning(
                f"Ignoring non-string 'allowed-tools' in {skill_path} (got {type(raw_tools).__name__})",
            )
        allowed_tools = []

    compatibility_str = str(frontmatter_data.get("compatibility", "")).strip() or None
    if compatibility_str and len(compatibility_str) > MAX_SKILL_COMPATIBILITY_LENGTH:
        log_warning(
            f"Compatibility exceeds {MAX_SKILL_COMPATIBILITY_LENGTH} characters in {skill_path}, truncating",
        )
        compatibility_str = compatibility_str[:MAX_SKILL_COMPATIBILITY_LENGTH]

    return SkillMetadata(
        name=str(name),
        description=description_str,
        path=skill_path,
        metadata=_validate_metadata(frontmatter_data.get("metadata", {}), skill_path),
        license=str(frontmatter_data.get("license", "")).strip() or None,
        compatibility=compatibility_str,
        allowed_tools=allowed_tools,
    )


def _validate_metadata(
    raw: object,
    skill_path: str,
) -> dict[str, str]:
    """Validate and normalize the metadata field from YAML frontmatter.

    YAML `safe_load` can return any type for the `metadata` key. This
    ensures the values in `SkillMetadata` are always a `dict[str, str]` by
    coercing via `str()` and rejecting non-dict inputs.

    Args:
        raw: Raw value from `frontmatter_data.get("metadata", {})`.
        skill_path: Path to the `SKILL.md` file (for warning messages).

    Returns:
        A validated `dict[str, str]`.
    """
    if not isinstance(raw, dict):
        if raw:
            log_warning(
                f"Ignoring non-dict metadata in {skill_path} (got {type(raw).__name__})",
            )
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _format_skill_annotations(skill: SkillMetadata) -> str:
    """Build a parenthetical annotation string from optional skill fields.

    Combines license and compatibility into a comma-separated string for
    display in the system prompt skill listing.

    Args:
        skill: Skill metadata to extract annotations from.

    Returns:
        Annotation string like `'License: MIT, Compatibility: Python 3.10+'`,
            or empty string if neither field is set.
    """
    parts: list[str] = []
    if skill.get("license"):
        parts.append(f"License: {skill['license']}")
    if skill.get("compatibility"):
        parts.append(f"Compatibility: {skill['compatibility']}")
    return ", ".join(parts)


def _collect_skill_dirs(items: list[dict[str, Any]], base_path: str) -> list[str]:
    """Collect skill directories from a backend listing.

    Includes child directories plus the source directory itself when it contains
    a top-level `SKILL.md`.
    """
    skill_dirs: list[str] = []
    for item in items:
        if not item.get("is_dir"):
            if item["path"].endswith("SKILL.md"):
                skill_dirs.append(base_path)
            continue
        skill_dirs.append(item["path"])
    return skill_dirs


def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """List all skills from a backend source.

    Scans backend for subdirectories containing `SKILL.md` files, downloads
    their content, parses YAML frontmatter, and returns skill metadata.

    Expected structure:

    ```txt
    source_path/
    └── skill-name/
        ├── SKILL.md   # Required
        └── helper.py  # Optional
    ```

    Args:
        backend: Backend instance to use for file operations
        source_path: Path to the skills directory in the backend

    Returns:
        List of skill metadata from successfully parsed `SKILL.md` files
    """
    base_path = source_path

    skills: list[SkillMetadata] = []
    items = backend.ls_info(base_path)

    skill_md_paths = []
    skill_dirs = _collect_skill_dirs(items, base_path)

    if not skill_dirs:
        log_warning(
            f"[skills] {base_path!r}: no skill dirs after scan "
            f"({len(items)} entries)",
        )
        return []

    # For each skill directory, check if SKILL.md exists and download it
    for skill_dir_path in skill_dirs:
        # Construct SKILL.md path using PurePosixPath for safe, standardized path operations
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)

    # Parse each downloaded SKILL.md
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            log_warning(
                f"[skills] skip {skill_md_path} (download error={response.error!r})",
            )
            continue

        if response.content is None:
            log_warning(f"Downloaded skill file {skill_md_path} has no content")
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            log_warning(f"Error decoding {skill_md_path}: {e}")
            continue

        # Extract directory name from path using PurePosixPath
        directory_name = PurePosixPath(skill_dir_path).name

        # Parse metadata
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


async def _alist_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """List all skills from a backend source (async version).

    Scans backend for subdirectories containing `SKILL.md` files, downloads
    their content, parses YAML frontmatter, and returns skill metadata.

    Expected structure:

    ```txt
    source_path/
    └── skill-name/
        ├── SKILL.md   # Required
        └── helper.py  # Optional
    ```

    Args:
        backend: Backend instance to use for file operations
        source_path: Path to the skills directory in the backend

    Returns:
        List of skill metadata from successfully parsed `SKILL.md` files
    """
    base_path = source_path

    skills: list[SkillMetadata] = []
    items = await backend.als_info(base_path)

    skill_dirs = _collect_skill_dirs(items, base_path)

    if not skill_dirs:
        log_warning(
            f"[skills] {base_path!r}: no skill dirs after scan "
            f"({len(items)} entries)",
        )
        return []

    # For each skill directory, check if SKILL.md exists and download it
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # Construct SKILL.md path using PurePosixPath for safe, standardized path operations
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = await backend.adownload_files(paths_to_download)

    # Parse each downloaded SKILL.md
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            log_warning(
                f"[skills] skip {skill_md_path} (download error={response.error!r})",
            )
            continue

        if response.content is None:
            log_warning(f"Downloaded skill file {skill_md_path} has no content")
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            log_warning(f"Error decoding {skill_md_path}: {e}")
            continue

        # Extract directory name from path using PurePosixPath
        directory_name = PurePosixPath(skill_dir_path).name

        # Parse metadata
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only load the full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Load the skill's full instructions**: Call `load_skill` with the skill's **name** from the list above (for example `load_skill(skill_name="api-testing")`).
3. **Follow the skill's instructions**: Loaded content is markdown workflows, best practices, and examples
4. **Access supporting markdown docs when needed**: After loading a skill, use `read_skill_file` for reference markdown such as `references/*.md`

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Use filesystem paths rooted at the workspace / skills directories when needed.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill listed by name
2. Load it with `load_skill(skill_name="web-research")`
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use `read_skill_file` to inspect supporting markdown files if the skill references them

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware[SkillsState, ContextT, ResponseT]):
    """Middleware for loading and exposing agent skills to the system prompt.

    Loads skills from backend sources and injects them into the system prompt
    using progressive disclosure (metadata first, full content on demand).

    Skills are loaded in source order with later sources overriding
    earlier ones.

    Example:
        ```python
        middleware = SkillsMiddleware(
            backend="filesystem",
            repo_root=".",
            sources=[
                "/path/to/skills/user/",
                "/path/to/skills/project/",
            ],
        )
        ```

    Args:
        backend: Backend mode ("filesystem" or "in-memory").
        sources: List of skill source paths.

            Source names are derived from the last path component.
    """

    state_schema = SkillsState

    def __init__(
        self,
        *,
        backend: Literal["filesystem", "in-memory"] = "filesystem",
        sources: list[str] | None = None,
        repo_root: str | Path | None = None,
        in_memory_files: dict[str, bytes] | None = None,
    ) -> None:
        """Initialize the skills middleware.

        Args:
            backend: Backend mode ("filesystem" or "in-memory").
            sources: List of skill source paths (e.g.,
                `['/skills/user/', '/skills/project/']`).
            repo_root: Filesystem backend root. Defaults to current working directory.
            in_memory_files: Virtual path -> file content map for in-memory backend.
        """
        self.backend_mode = backend
        self.repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self.sources = sources or ["/skills"]
        if backend == "filesystem":
            self._backend: BackendProtocol = FileSystemBackend(root_dir=self.repo_root)
        elif backend == "in-memory":
            self._backend = InMemoryBackend(files=in_memory_files)
        else:
            raise ValueError("backend must be one of: 'filesystem', 'in-memory'")
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT
        self.tool_discovery: ToolDiscoveryFunc | None = None
        self.async_tool_discovery: AsyncToolDiscoveryFunc | None = None
     
    def configure_tool_discovery(
        self,
        *,
        discover: ToolDiscoveryFunc | None = None,
        adiscover: AsyncToolDiscoveryFunc | None = None,
    ) -> None:
        """Configure dynamic runtime tool discovery callbacks."""
        self.tool_discovery = discover
        self.async_tool_discovery = adiscover

    @staticmethod
    def _state_get(state: SkillsState, key: str, default: Any = None) -> Any:
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    @staticmethod
    def _state_set(state: SkillsState, key: str, value: Any) -> None:
        if isinstance(state, dict):
            state[key] = value
        else:
            setattr(state, key, value)

    @staticmethod
    def _tool_name(tool: ToolDef) -> str | None:
        if isinstance(tool, BaseTool):
            return tool.name
        if isinstance(tool, dict):
            if isinstance(tool.get("name"), str):
                return str(tool["name"])
            fn = tool.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                return str(fn["name"])
        return None

    def _inject_builtin_skill_tools(self, tools: list[ToolDef], skills: list[SkillMetadata]) -> list[ToolDef]:
        """Expose built-in skill tools only when at least one skill exists."""
        if not skills:
            return tools
        merged = list(tools)
        existing_names = {self._tool_name(t) for t in merged}
        for builtin in (load_skill, read_skill_file):
            if builtin.name not in existing_names:
                merged.append(builtin)
                existing_names.add(builtin.name)
        return merged

    def _active_skills(self, state: SkillsState) -> list[str]:
        return _active_skill_names_from_state(state)

    @staticmethod
    def _extract_tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
        args = tool_call.get("args")
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            raw = args.strip()
            if not raw:
                return {}
            try:
                decoded = json.loads(raw)
            except Exception:  # noqa: BLE001
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    @staticmethod
    def _coerce_optional_int(value: Any, default: int) -> int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return default
        return default

    def _discover_tools_from_registry(
        self,
        *,
        skills: list[SkillMetadata],
        state: SkillsState,
    ) -> list[ToolDef]:
        """Resolve allowed-tools for active skills from Andromeda's global Toolkit."""
        active = set(self._active_skills(state))
        if not active:
            return []

        allowed_tool_names: list[str] = []
        for skill in skills:
            if skill["name"] not in active:
                continue
            for tool_name in skill.get("allowed_tools", []):
                if isinstance(tool_name, str) and tool_name:
                    allowed_tool_names.append(tool_name)

        if not allowed_tool_names:
            return []

        try:
            from andromeda.tools.toolkit import get_default_toolkit
        except Exception:  # noqa: BLE001
            log_error("Failed to import andromeda.tools.toolkit")
            return []

        toolkit = get_default_toolkit()
        out: list[ToolDef] = []
        for name in dict.fromkeys(allowed_tool_names):
            if toolkit.has(name):
                out.append(toolkit.get(name))
            else:
                log_warning(
                    f"Skill requested allowed tool '{name}' but it is not registered in Toolkit",
                )
        return out

    def _resolve_registry_tool_for_call(
        self,
        *,
        tool_name: str,
        state: SkillsState,
    ) -> BaseTool | None:
        """Resolve a specific tool name from Toolkit if allowed by active skills."""
        active = set(self._active_skills(state))
        if not active:
            return None

        skills_metadata = self._state_get(state, "skills_metadata", []) or []
        allowed = {
            name
            for skill in skills_metadata
            if skill.get("name") in active
            for name in skill.get("allowed_tools", [])
            if isinstance(name, str) and name
        }
        if allowed and tool_name not in allowed:
            return None

        try:
            from andromeda.tools.toolkit import get_default_toolkit
        except Exception:  # noqa: BLE001
            log_error("Failed to import andromeda.tools.toolkit")
            return None

        toolkit = get_default_toolkit()
        if toolkit.has(tool_name):
            return toolkit.get(tool_name)
        return None

    def _merge_tools(
        self,
        existing: list[ToolDef],
        discovered: list[ToolDef],
        skills: list[SkillMetadata],
    ) -> tuple[list[ToolDef], dict[str, BaseTool]]:
        allowed_names = {
            name
            for skill in skills
            for name in skill.get("allowed_tools", [])
            if isinstance(name, str) and name
        }

        merged_by_name: dict[str, ToolDef] = {}
        anonymous: list[ToolDef] = []
        dynamic_registry: dict[str, BaseTool] = {}

        # Keep existing tools first.
        for tool in existing:
            name = self._tool_name(tool)
            if name:
                merged_by_name[name] = tool
            else:
                anonymous.append(tool)

        # Add discovered tools, optionally constrained by allowed-tools.
        for tool in discovered:
            name = self._tool_name(tool)
            if name and allowed_names and name not in allowed_names:
                continue
            if name:
                merged_by_name[name] = tool
                if isinstance(tool, BaseTool):
                    dynamic_registry[name] = tool
            else:
                anonymous.append(tool)

        return anonymous + list(merged_by_name.values()), dynamic_registry

    def _discover_tools_sync(self, request: ModelRequest[ContextT]) -> list[ToolDef]:
        skills_metadata = self._state_get(request.state, "skills_metadata", []) or []
        discovered: list[ToolDef] = self._discover_tools_from_registry(
            skills=skills_metadata,
            state=request.state,
        )

        if self.tool_discovery is None:
            return discovered

        current_tools = list(request.tools or [])
        try:
            custom = self.tool_discovery(
                skills=skills_metadata,
                state=request.state,
                runtime=request.runtime,
                current_tools=current_tools,
            )
            return [*discovered, *custom]
        except Exception:
            log_error("Dynamic tool discovery failed")
            return discovered

    async def _discover_tools_async(self, request: ModelRequest[ContextT]) -> list[ToolDef]:
        skills_metadata = self._state_get(request.state, "skills_metadata", []) or []
        discovered: list[ToolDef] = self._discover_tools_from_registry(
            skills=skills_metadata,
            state=request.state,
        )

        if self.async_tool_discovery is not None:
            current_tools = list(request.tools or [])
            try:
                custom = await self.async_tool_discovery(
                    skills=skills_metadata,
                    state=request.state,
                    runtime=request.runtime,
                    current_tools=current_tools,
                )
                return [*discovered, *custom]
            except Exception:
                log_error("Async dynamic tool discovery failed")
                return discovered
        return self._discover_tools_sync(request)

    @staticmethod
    def _result_to_tool_message(result: Any, tool_call_id: str | None) -> Any:
        if isinstance(result, (ToolMessage, Command)):
            return result
        if isinstance(result, str):
            content = result
        else:
            content = str(result)
        return ToolMessage(content=content, tool_call_id=_coerce_tool_call_id(tool_call_id))

    @staticmethod
    def _extract_skill_name_from_tool_call(tool_call: dict[str, Any]) -> str | None:
        args = tool_call.get("args")
        if isinstance(args, dict):
            value = args.get("skill_name")
            return value if isinstance(value, str) and value.strip() else None
        if isinstance(args, str):
            raw = args.strip()
            if not raw:
                return None
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    value = decoded.get("skill_name")
                    return value if isinstance(value, str) and value.strip() else None
                if isinstance(decoded, str) and decoded.strip():
                    return decoded
            except Exception:  # noqa: BLE001
                # Some models emit plain string args instead of JSON.
                if raw:
                    return raw
        return None

    def _get_backend(self, state: SkillsState, runtime: Runtime, config: Any) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance
        """
        return self._backend


    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """Format skills metadata for display in system prompt."""
        if not skills:
            paths = [f"{source_path}" for source_path in self.sources]
            return f"(No skills available yet. You can create skills in {' or '.join(paths)})"

        lines = []
        for skill in skills:
            annotations = _format_skill_annotations(skill)
            desc_line = f"- **{skill['name']}**: {skill['description']}"
            if annotations:
                desc_line += f" ({annotations})"
            lines.append(desc_line)
            if skill["allowed_tools"]:
                lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
            lines.append(f"  -> Load `{skill['name']}` skill for full instructions")

        return "\n".join(lines)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """Inject skills documentation into a model request's system message.

        Args:
            request: Model request to modify

        Returns:
            New model request with skills documentation injected into system message
        """
        skills_metadata = request.state.get("skills_metadata", [])
        if not skills_metadata:
            log_warning(
                "[skills] skills_metadata is empty in request.state "
                f"(configured sources={self.sources!r})",
            )
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_list=skills_list,
        )

        new_system_message = append_to_system_message(request.system_message, skills_section)

        return request.override(system_message=new_system_message)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """Load skills metadata before agent execution (synchronous).

        When ``skills_metadata`` is missing or empty (including graphs that default
        it to ``[]``), scans all configured sources. Skips filesystem I/O once a
        non-empty metadata list is present for the thread.

        Skills merge in source order; later sources override earlier entries with the
        same skill name.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with `skills_metadata` when it was missing or empty.
            Otherwise may only initialize `active_skills` / `dynamic_skill_tools`.
        """
        update: SkillsStateUpdate = {}
        # Initial graph/checkpoint state often sets `skills_metadata` to `[]`. Treat empty the
        # same as missing so discovery runs once real sources are available.
        if not state.get("skills_metadata"):
            backend = self._get_backend(state, runtime, config)
            all_skills: dict[str, SkillMetadata] = {}
            for source_path in self.sources:
                source_skills = _list_skills(backend, source_path)
                for skill in source_skills:
                    all_skills[skill["name"]] = skill
            update["skills_metadata"] = list(all_skills.values())

        if "active_skills" not in state:
            update["active_skills"] = []
        if "dynamic_skill_tools" not in state:
            update["dynamic_skill_tools"] = {}
        return update or None

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """Load skills metadata before agent execution (async).

        When ``skills_metadata`` is missing or empty (including graphs that default
        it to ``[]``), scans all configured sources. Skips filesystem I/O once a
        non-empty metadata list is present for the thread.

        Skills merge in source order; later sources override earlier entries with the
        same skill name.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with `skills_metadata` when it was missing or empty.
            Otherwise may only initialize `active_skills` / `dynamic_skill_tools`.
        """
        update: SkillsStateUpdate = {}
        if not state.get("skills_metadata"):
            backend = self._get_backend(state, runtime, config)
            all_skills: dict[str, SkillMetadata] = {}
            for source_path in self.sources:
                source_skills = await _alist_skills(backend, source_path)
                for skill in source_skills:
                    all_skills[skill["name"]] = skill
            update["skills_metadata"] = list(all_skills.values())

        if "active_skills" not in state:
            update["active_skills"] = []
        if "dynamic_skill_tools" not in state:
            update["dynamic_skill_tools"] = {}
        return update or None

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject skills documentation into the system prompt.

        Args:
            request: Model request being processed
            handler: Handler function to call with modified request

        Returns:
            Model response from handler
        """
        modified_request = self.modify_request(request)

        skills_metadata = self._state_get(modified_request.state, "skills_metadata", []) or []
        discovered = self._discover_tools_sync(modified_request)
        merged_tools, dynamic_registry = self._merge_tools(
            list(modified_request.tools or []), discovered, skills_metadata
        )
        merged_tools = self._inject_builtin_skill_tools(merged_tools, skills_metadata)
        self._state_set(modified_request.state, "dynamic_skill_tools", dynamic_registry)
        modified_request = modified_request.override(tools=merged_tools)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Inject skills documentation into the system prompt (async version).

        Args:
            request: Model request being processed
            handler: Async handler function to call with modified request

        Returns:
            Model response from handler
        """
        modified_request = self.modify_request(request)

        skills_metadata = self._state_get(modified_request.state, "skills_metadata", []) or []
        discovered = await self._discover_tools_async(modified_request)
        merged_tools, dynamic_registry = self._merge_tools(
            list(modified_request.tools or []), discovered, skills_metadata
        )
        merged_tools = self._inject_builtin_skill_tools(merged_tools, skills_metadata)
        self._state_set(modified_request.state, "dynamic_skill_tools", dynamic_registry)

        modified_request = modified_request.override(tools=merged_tools)
        return await handler(modified_request)

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        token = _skills_backend_ctx.set(self._backend)
        try:
            # If this tool is already registered by the tool node, let normal execution continue.
            if getattr(request, "tool", None) is not None:
                return handler(request)

            tool_call = getattr(request, "tool_call", {}) or {}
            tool_name = tool_call.get("name")
            tool_call_id = str(tool_call.get("id") or "")

            dynamic_registry = self._state_get(request.state, "dynamic_skill_tools", {}) or {}
            dynamic_tool = dynamic_registry.get(tool_name)
            if dynamic_tool is None and tool_name == load_skill.name:
                skill_name = self._extract_skill_name_from_tool_call(tool_call)
                if not skill_name:
                    return ToolMessage(
                        content="Tool error: load_skill requires argument 'skill_name'.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                request.runtime.tool_call_id = tool_call_id
                if isinstance(request, ToolCallRequest):
                    return handler(request.override(tool=load_skill))
                return _build_load_skill_command(skill_name, request.runtime)
            if dynamic_tool is None and tool_name == read_skill_file.name:
                args = self._extract_tool_args(tool_call)
                skill_name = args.get("skill_name")
                relative_path = args.get("relative_path")
                if not isinstance(skill_name, str) or not skill_name.strip():
                    return ToolMessage(
                        content="Tool error: read_skill_file requires argument 'skill_name'.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                if not isinstance(relative_path, str) or not relative_path.strip():
                    return ToolMessage(
                        content="Tool error: read_skill_file requires argument 'relative_path'.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                request.runtime.tool_call_id = tool_call_id
                if isinstance(request, ToolCallRequest):
                    return handler(request.override(tool=read_skill_file))
                return ToolMessage(
                    content=_read_skill_file_text(
                        skill_name=skill_name,
                        relative_path=relative_path,
                        runtime=request.runtime,
                        start_line=self._coerce_optional_int(args.get("start_line"), 0),
                        end_line=self._coerce_optional_int(args.get("end_line"), 100),
                    ),
                    tool_call_id=tool_call_id,
                    name=read_skill_file.name,
                )
            if dynamic_tool is None and isinstance(tool_name, str):
                dynamic_tool = self._resolve_registry_tool_for_call(
                    tool_name=tool_name,
                    state=request.state,
                )
            if dynamic_tool is None:
                return handler(request)

            try:
                result = dynamic_tool.invoke(tool_call, config=request.runtime.config)
                return self._result_to_tool_message(result, tool_call_id=tool_call_id)
            except Exception as exc:  # noqa: BLE001
                log_error(f"Dynamic Tool invocation failed for '{tool_name}'")
                return ToolMessage(
                    content=f"Tool error: tool '{tool_name}' failed. ({exc})",
                    tool_call_id=tool_call_id,
                    status="error",
                )
        finally:
            _skills_backend_ctx.reset(token)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        token = _skills_backend_ctx.set(self._backend)
        try:
            # If this tool is already registered by the tool node, let normal execution continue.
            if getattr(request, "tool", None) is not None:
                return await handler(request)

            tool_call = getattr(request, "tool_call", {}) or {}
            tool_name = tool_call.get("name")
            tool_call_id = str(tool_call.get("id") or "")

            dynamic_registry = self._state_get(request.state, "dynamic_skill_tools", {}) or {}
            dynamic_tool = dynamic_registry.get(tool_name)
            if dynamic_tool is None and tool_name == load_skill.name:
                skill_name = self._extract_skill_name_from_tool_call(tool_call)
                if not skill_name:
                    return ToolMessage(
                        content="Tool error: load_skill requires argument 'skill_name'.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                request.runtime.tool_call_id = tool_call_id
                if isinstance(request, ToolCallRequest):
                    return await handler(request.override(tool=load_skill))
                return _build_load_skill_command(skill_name, request.runtime)
            if dynamic_tool is None and tool_name == read_skill_file.name:
                args = self._extract_tool_args(tool_call)
                skill_name = args.get("skill_name")
                relative_path = args.get("relative_path")
                if not isinstance(skill_name, str) or not skill_name.strip():
                    return ToolMessage(
                        content="Tool error: read_skill_file requires argument 'skill_name'.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                if not isinstance(relative_path, str) or not relative_path.strip():
                    return ToolMessage(
                        content="Tool error: read_skill_file requires argument 'relative_path'.",
                        tool_call_id=tool_call_id,
                        status="error",
                    )
                request.runtime.tool_call_id = tool_call_id
                if isinstance(request, ToolCallRequest):
                    return await handler(request.override(tool=read_skill_file))
                return ToolMessage(
                    content=_read_skill_file_text(
                        skill_name=skill_name,
                        relative_path=relative_path,
                        runtime=request.runtime,
                        start_line=self._coerce_optional_int(args.get("start_line"), 0),
                        end_line=self._coerce_optional_int(args.get("end_line"), 100),
                    ),
                    tool_call_id=tool_call_id,
                    name=read_skill_file.name,
                )
            if dynamic_tool is None and isinstance(tool_name, str):
                dynamic_tool = self._resolve_registry_tool_for_call(
                    tool_name=tool_name,
                    state=request.state,
                )
            if dynamic_tool is None:
                return await handler(request)

            try:
                result = await dynamic_tool.ainvoke(tool_call, config=request.runtime.config)
                return self._result_to_tool_message(result, tool_call_id=tool_call_id)
            except Exception as exc:  # noqa: BLE001
                log_error(f"Async dynamic tool invocation failed for '{tool_name}'")
                return ToolMessage(
                    content=f"Tool error: tool '{tool_name}' failed. ({exc})",
                    tool_call_id=tool_call_id,
                    status="error",
                )
        finally:
            _skills_backend_ctx.reset(token)


__all__ = [
    "SkillMetadata",
    "InMemoryBackend",
    "FileSystemBackend",
    "SkillsMiddleware",
    "load_skill",
    "read_skill_file",
]
