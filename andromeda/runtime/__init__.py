from __future__ import annotations

import asyncio
import difflib
import os
import queue as _queue_mod
import threading
import warnings
from contextlib import suppress
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Mapping, Optional

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except Exception:  # pragma: no cover - fallback for environments without this import path
    LangChainPendingDeprecationWarning = DeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
)

from andromeda.config.yaml_utils import yaml_load
from andromeda.utils.ignore_rules import IgnoreMatcher

from .agents import build_workspace_agent, normalize_agent_payload
from .context import RuntimeContext
from .json_utils import to_json_compatible as _to_json_compatible
from .registry import RunnableRegistry, RuntimeEntry, RuntimeKind
from .validation import (
    AgentBuildError,
    RunnableAmbiguousError,
    RunnableNotFoundError,
    ValidationIssue,
    ValidationResult,
    WorkflowValidationError,
    result_from_issues,
    error,
)
from .workflows import RuntimeWorkflow, WorkflowRunResult, parse_workflow_definition
from .workflows import WorkflowValidationError as _WFV
from andromeda import HumanMessage


@dataclass
class RunResult:
    """Common result object returned by runtime execution."""

    kind: RuntimeKind
    text: str | None = None
    messages: list[Any] = field(default_factory=list)
    state: Mapping[str, Any] | None = None
    structured_response: Any = None
    artifacts: list[str] = field(default_factory=list)
    #: Per-file diff metadata paralleling ``artifacts`` (same paths, plus
    #: status + line counts). See :func:`diff_workspace_files` for the shape.
    #: Empty unless a workspace diff ran (blocking ``run()``/``arun()``, non-dry).
    artifact_changes: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None

    def to_dict(self, *, verbose: bool = False) -> dict[str, Any]:
        # Default view is the final output only: ``text`` plus the declared
        # ``structured_response`` (a workflow's declared ``outputs``). The full
        # execution ``state``/``messages`` and the duplicative ``raw`` are bulky
        # and only emitted with ``verbose`` (CLI ``--raw``). When nothing
        # summarizes the run, ``state`` is shown as a fallback so output is never
        # empty. Internal ``__``-prefixed state keys are always stripped.
        out: dict[str, Any] = {"kind": self.kind}
        if self.text is not None:
            out["text"] = self.text
        if self.structured_response is not None:
            out["structured_response"] = _to_json_compatible(self.structured_response)
        if self.artifacts:
            out["artifacts"] = _to_json_compatible(self.artifacts)
        if self.artifact_changes:
            out["artifact_changes"] = _to_json_compatible(self.artifact_changes)

        has_summary = self.text is not None or self.structured_response is not None
        if self.state is not None and (verbose or not has_summary):
            out["state"] = _to_json_compatible(_public_state(self.state))
        if self.messages and verbose:
            out["messages"] = _to_json_compatible(self.messages)
        if verbose and self.raw is not None:
            out["raw"] = _to_json_compatible(self.raw)
        return out


def _public_state(state: Any) -> Any:
    """Drop internal ``__``-prefixed bookkeeping keys (``__inputs``/``__result``/``__node``)."""
    if isinstance(state, Mapping):
        return {k: v for k, v in state.items() if not str(k).startswith("__")}
    return state


#: Above this size a file's text isn't captured for line-level diffing (it's
#: still detected as changed via mtime, just reported with 0 line counts). Keeps
#: the before-snapshot from slurping a huge generated blob into memory.
_MAX_DIFF_BYTES = 1_000_000


@dataclass
class _FileState:
    """One file's before/after fingerprint: mtime plus captured text lines.

    ``lines`` is None for a binary or oversized file — its change is still
    detectable by mtime, but a line-level delta can't be computed for it.
    """

    mtime: float
    lines: Optional[list[str]]


def snapshot_workspace_files(
    roots: set[Path], *, respect_ignores: bool = True
) -> dict[str, _FileState]:
    """Opaque before/after snapshot of every file under each root.

    Maps absolute path -> :class:`_FileState` (mtime + captured text lines).
    Pass the *before* and *after* snapshots to :func:`diff_workspace_files`;
    treat the value as opaque. Text is captured so line counts can be computed
    later — the old content is gone once the run has written over it, so it has
    to be grabbed up front, not reconstructed after.

    With ``respect_ignores`` (the default), the walk honors each root's
    ``.gitignore``/``.andromedaignore`` files via :class:`IgnoreMatcher` and
    always skips ``.git`` — so ignored build output, caches, and the git object
    store are never read and never surface as artifacts. Whole ignored
    directories are *pruned* rather than descended into, which is where the
    speed comes from (a raw ``rglob`` can't prune). Only in-tree ignore files
    are consulted, not a global gitignore or ``.git/info/exclude``. Pass
    ``respect_ignores=False`` to walk everything (legacy behavior).
    """
    snapshot: dict[str, _FileState] = {}
    for root in roots:
        matcher = IgnoreMatcher.for_filesystem(root) if respect_ignores else None
        for dirpath_str, dirnames, filenames in os.walk(root):
            dirpath = Path(dirpath_str)
            if matcher is not None:
                # Prune in place so we never descend into an ignored subtree —
                # ``.git`` unconditionally (no ``.gitignore`` lists it; git
                # ignores it implicitly), the rest by the ignore rules.
                dirnames[:] = [
                    name
                    for name in dirnames
                    if name != ".git"
                    and not matcher.is_ignored(dirpath / name, is_dir=True)
                ]
            for name in filenames:
                path = dirpath / name
                if matcher is not None and matcher.is_ignored(path, is_dir=False):
                    continue
                if not path.is_file():
                    continue
                with suppress(OSError):
                    stat = path.stat()
                    lines: Optional[list[str]] = None
                    if stat.st_size <= _MAX_DIFF_BYTES:
                        try:
                            lines = path.read_text(encoding="utf-8").splitlines()
                        except (OSError, UnicodeDecodeError):
                            lines = None  # binary / unreadable -> mtime-only tracking
                    snapshot[str(path)] = _FileState(mtime=stat.st_mtime, lines=lines)
    return snapshot


def _line_delta(before_lines: list[str], after_lines: list[str]) -> tuple[int, int]:
    """(lines_added, lines_removed) between two line lists, git-stat style.

    A changed line counts as one removed + one added, matching how ``git diff``
    tallies ``--numstat``.
    """
    added = removed = 0
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            removed += i2 - i1
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "insert":
            added += j2 - j1
    return added, removed


def diff_workspace_files(
    before: dict[str, _FileState], after: dict[str, _FileState]
) -> list[dict[str, Any]]:
    """Per-file changes between two snapshots, sorted by path.

    Each entry is a plain (JSON-able) dict::

        {"path": str, "status": "added"|"modified"|"removed",
         "lines_added": int, "lines_removed": int}

    Line counts are 0 when a file's content couldn't be captured on either side
    (binary or oversized — see :data:`_MAX_DIFF_BYTES`); the ``status`` is still
    accurate. A file rewritten with identical content shows as ``"modified"``
    with 0/0.
    """
    changes: list[dict[str, Any]] = []
    for path, state in after.items():
        prior = before.get(path)
        if prior is None:
            added = len(state.lines) if state.lines is not None else 0
            changes.append(
                {"path": path, "status": "added", "lines_added": added, "lines_removed": 0}
            )
        elif prior.mtime != state.mtime:
            if prior.lines is not None and state.lines is not None:
                added, removed = _line_delta(prior.lines, state.lines)
            else:
                added = removed = 0
            changes.append(
                {"path": path, "status": "modified",
                 "lines_added": added, "lines_removed": removed}
            )
    for path, prior in before.items():
        if path not in after:
            removed = len(prior.lines) if prior.lines is not None else 0
            changes.append(
                {"path": path, "status": "removed", "lines_added": 0, "lines_removed": removed}
            )
    changes.sort(key=lambda change: change["path"])
    return changes


def _resolve_state_path(state_file: str | Path) -> Path:
    state_path = Path(state_file).expanduser()
    if state_path.exists() and state_path.is_dir():
        raise AgentBuildError(f"Workflow state file points to a directory: {state_path}")
    return state_path


def _fold_prompt_into_inputs(
    prompt: str | None, inputs: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """Route a positional prompt into ``inputs['prompt']`` for workflow entries.

    Workflows are driven by ``inputs``. When a positional prompt is supplied
    (e.g. ``andromeda run <workflow> "..."`` without --kind), expose it as
    ``inputs['prompt']`` so it is not silently dropped. Shared by the sync and
    async run/stream entrypoints so their behavior cannot drift apart.
    """
    if prompt is None:
        return inputs
    merged = dict(inputs or {})
    merged.setdefault("prompt", prompt)
    return merged


def _agent_result_from_output(resolved_prompt: str, output: Any) -> RunResult:
    # ``WorkspaceAgent.run``/``arun`` return the final message *content*: a string
    # for normal replies, a list of content blocks for multimodal/structured
    # output, or ``None`` when no messages were produced. Neither returns a
    # message list, so we map content to ``text``/``structured_response`` accordingly.
    text: str | None
    structured: Any = None
    if output is None:
        text = None
    elif isinstance(output, str):
        text = output
    else:
        structured = output
        text = None

    return RunResult(
        kind="agent",
        text=text,
        messages=[],
        state=None,
        structured_response=structured,
        raw={"input": resolved_prompt, "output": output},
    )


async def _async_iter_from_sync(iterator: Iterator[Any]) -> AsyncIterator[Any]:
    """Bridge a synchronous iterator onto an async iterator.

    ``RuntimeWorkflow.stream()`` drives workflow nodes synchronously (it calls
    ``WorkspaceAgent.stream``/``._run_workflow`` directly rather than through an
    async graph engine), so there is nothing to natively ``await`` here. The
    iterator is drained on a background thread and its items are handed back
    through a thread-safe queue so the event loop is not blocked while waiting
    on the next chunk.
    """
    q: _queue_mod.Queue = _queue_mod.Queue(maxsize=1)
    stop = threading.Event()

    def _drain() -> None:
        try:
            for item in iterator:
                if stop.is_set():
                    return
                q.put(("chunk", item))
        except BaseException as exc:  # noqa: BLE001
            if not stop.is_set():
                q.put(("error", exc))
        else:
            q.put(("done", None))

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = await asyncio.to_thread(q.get)
            if kind == "chunk":
                yield payload
            elif kind == "error":
                raise payload
            else:
                break
    finally:
        # If the consumer stops early (e.g. breaks out of the loop), the
        # producer thread may be blocked on a full queue. Signal it to stop and
        # drain one slot so it can exit instead of leaking.
        stop.set()
        with suppress(_queue_mod.Empty):
            q.get_nowait()


def _load_entry_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Workflow/agent definition not found: {path}")
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml_load(f)
    return dict(data) if isinstance(data, dict) else {}


class AndromedaRuntime:
    """Programmatic entrypoint for config-driven Andromeda execution."""

    def __init__(self, context: RuntimeContext, registry: RunnableRegistry):
        self.context = context
        self.registry = registry
        self._agent_cache: dict[str, Any] = {}
        self._workflow_cache: dict[str, RuntimeWorkflow] = {}
        self._build_lock = threading.RLock()

    def workspace_roots(self) -> set[Path]:
        """Local filesystem roots of every workspace-backed agent built so far.

        A workflow's steps are built lazily as agents via :meth:`build_agent`
        (the same cache a standalone agent run uses), so this covers both a
        single agent invocation and every agent node a workflow has touched --
        whatever's actually been built by the time this is called. Agents
        without a local workspace session (e.g. non-filesystem backends, or
        not yet built) contribute nothing.
        """
        roots: set[Path] = set()
        with self._build_lock:
            agents = list(self._agent_cache.values())
        for agent in agents:
            session = getattr(agent, "session", None)
            root = getattr(session, "root", None)
            if isinstance(root, Path) and root.is_dir():
                roots.add(root)
        return roots

    def workspace_root_for(self, name: str) -> Path | None:
        """The local workspace root of one specific agent, by name (builds it if needed).

        Unlike :meth:`workspace_roots` (the union across every agent built so
        far), this resolves/builds exactly the one agent asked for -- the
        caller to reach for when it already knows which agent produced a
        given step (e.g. a workflow node's ``ref``) and wants to diff just
        that agent's files, not everyone's. Returns ``None`` if the agent has
        no local workspace session (e.g. a non-filesystem backend).
        """
        agent = self.build_agent(name, kind="agent")
        session = getattr(agent, "session", None)
        root = getattr(session, "root", None)
        return root if isinstance(root, Path) and root.is_dir() else None

    @classmethod
    def discover(
        cls,
        root: str | Path | None = None,
        *,
        include_global: bool = True,
        global_root: str | Path | None = None,
        project_config_root: str | Path | None = None,
        config_dir_name: str | Path = ".andromeda",
        env=None,
        toolkit=None,
        execution_context=None,
        mcp_runtime=None,
    ) -> "AndromedaRuntime":
        context = RuntimeContext.discover(
            root=root,
            include_global=include_global,
            global_root=global_root,
            project_config_root=project_config_root,
            config_dir_name=config_dir_name,
            env=env,
            toolkit=toolkit,
            execution_context=execution_context,
            mcp_runtime=mcp_runtime,
        )
        registry = RunnableRegistry.from_context(context)
        return cls(context=context, registry=registry)

    @classmethod
    def from_context(cls, context: RuntimeContext) -> "AndromedaRuntime":
        return cls(context=context, registry=RunnableRegistry.from_context(context))

    def list(self, kind: RuntimeKind | None = None) -> list[dict[str, Any]]:
        return [entry.to_info() for entry in self.registry.list(kind=kind)]

    def inspect(self, name: str, kind: RuntimeKind | None = None) -> dict[str, Any]:
        try:
            entry = self._resolve(name, kind=kind)
        except KeyError as exc:
            raise RunnableNotFoundError(str(exc)) from exc

        payload = entry.payload
        if payload is None:
            try:
                payload = _load_entry_payload(entry.path)
            except Exception as exc:
                payload = {"error": str(exc)}

        return {
            "name": entry.effective_name or entry.name,
            "kind": entry.kind,
            "source": str(entry.path),
            "scope": entry.scope,
            "raw": payload,
        }

    def validate(
        self,
        name: str | None = None,
        kind: RuntimeKind | None = None,
    ) -> ValidationResult:
        if name is None:
            entries = self.registry.list(kind=kind)
            issues: list[ValidationIssue] = []
            issues.extend(self.registry.validate())

            for entry in entries:
                payload = entry.payload
                if payload is None:
                    try:
                        payload = _load_entry_payload(entry.path)
                    except Exception as exc:
                        issues.append(
                            error(f"Invalid definition at {entry.path}: {exc}")
                        )
                        continue

                if entry.kind == "workflow":
                    try:
                        from .workflows import WorkflowDefinition

                        definition = WorkflowDefinition.from_mapping(
                            entry.name,
                            entry.path,
                            payload,
                            entry.definition_root,
                        )
                        issues.extend(definition.validate())
                    except Exception as exc:
                        issues.append(
                            error(f"Invalid workflow '{entry.name}': {exc}")
                        )
                else:
                    try:
                        normalize_agent_payload(
                            payload,
                            context=self.context,
                            defaults=self._agent_defaults_for_scope(entry.scope),
                            source=entry.path,
                        )
                    except Exception as exc:
                        issues.append(
                            error(f"Invalid agent '{entry.name}': {exc}")
                        )

            return result_from_issues(issues)

        issues: list[ValidationIssue] = []
        try:
            entry = self._resolve(name, kind=kind)
        except KeyError as exc:
            issues.append(error(str(exc)))
            return result_from_issues(issues)

        payload = entry.payload
        if payload is None:
            try:
                payload = _load_entry_payload(entry.path)
            except Exception as exc:
                return result_from_issues([error(f"Invalid definition at {entry.path}: {exc}")])

        if entry.kind == "workflow":
            try:
                from .workflows import WorkflowDefinition

                definition = WorkflowDefinition.from_mapping(
                    entry.name,
                    entry.path,
                    payload,
                    entry.definition_root,
                )
                issues.extend(definition.validate())
            except Exception as exc:  # noqa: BLE001
                issues.append(error(f"Invalid workflow '{entry.name}': {exc}"))
        else:
            try:
                normalize_agent_payload(
                    payload,
                    context=self.context,
                    defaults=self._agent_defaults_for_scope(entry.scope),
                    source=entry.path,
                )
            except Exception as exc:  # noqa: BLE001
                issues.append(error(f"Invalid agent '{entry.name}': {exc}"))

        return result_from_issues(issues)

    def close(self) -> None:
        """Release runtime-owned agents and their transient resources."""
        with self._build_lock:
            agents = list(self._agent_cache.values())
            self._agent_cache.clear()
            self._workflow_cache.clear()

        for agent in agents:
            closer = getattr(agent, "close", None)
            if callable(closer):
                with suppress(Exception):
                    closer()

    def build_agent(self, name: str, kind: RuntimeKind | None = "agent") -> Any:
        if kind is not None and kind != "agent":
            raise AgentBuildError("build_agent() can only build agents.")

        entry = self._resolve(name, kind="agent")
        cache_key = f"{entry.scope}:{entry.effective_name}:{entry.path}"
        with self._build_lock:
            if cache_key in self._agent_cache:
                return self._agent_cache[cache_key]

            payload = _safe_entry_payload(entry)
            merged_defaults = _deep_merge_defaults_for_entry(
                self.context,
                entry,
                defaults_name="agent",
            )
            agent = build_workspace_agent(
                path=entry.path,
                payload=payload,
                context=self.context,
                defaults=merged_defaults,
            )
            self._agent_cache[cache_key] = agent
            return agent

    def build_workflow(self, name: str, kind: RuntimeKind | None = "workflow") -> RuntimeWorkflow:
        if kind is not None and kind != "workflow":
            raise AgentBuildError("build_workflow() can only build workflows.")

        entry = self._resolve(name, kind="workflow")
        cache_key = f"{entry.scope}:{entry.effective_name}:{entry.path}"
        with self._build_lock:
            if cache_key in self._workflow_cache:
                return self._workflow_cache[cache_key]

            payload = _safe_entry_payload(entry)
            defaults = _deep_merge_defaults_for_entry(
                self.context,
                entry,
                defaults_name="workflow",
            )
            try:
                workflow = parse_workflow_definition(
                    path=entry.path,
                    payload=payload,
                    context=self.context,
                    defaults=defaults,
                    agent_builder=lambda entry_name: self.build_agent(entry_name),
                )
            except Exception as exc:
                if isinstance(exc, _WFV):
                    raise WorkflowValidationError(str(exc)) from exc
                raise AgentBuildError(f"Failed to build workflow '{entry.name}': {exc}") from exc

            self._workflow_cache[cache_key] = workflow
            return workflow

    def run(
        self,
        name: str,
        *,
        prompt: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        kind: RuntimeKind | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        state_file: str | Path | None = None,
        dry_run: bool = False,
    ) -> RunResult:
        entry = self._resolve(name, kind=kind)
        if dry_run:
            if entry.kind == "agent":
                return self._run_agent(
                    entry, prompt=prompt, inputs=inputs, thread_id=thread_id,
                    metadata=metadata, dry_run=True,
                )
            return self._run_workflow(
                entry, inputs=_fold_prompt_into_inputs(prompt, inputs),
                state_file=state_file, thread_id=thread_id, metadata=metadata, dry_run=True,
            )

        # Snapshot every workspace-backed agent's files before dispatch, then again
        # after (re-collecting roots too -- a workflow builds its step agents lazily,
        # so a fresh one's workspace wouldn't be in the "before" set otherwise) and
        # diff, so ``result.artifacts`` reflects files this run actually touched
        # rather than staying permanently empty.
        before_roots = self.workspace_roots()
        before = snapshot_workspace_files(before_roots)

        if entry.kind == "agent":
            result = self._run_agent(
                entry,
                prompt=prompt,
                inputs=inputs,
                thread_id=thread_id,
                metadata=metadata,
                dry_run=dry_run,
            )
        else:
            result = self._run_workflow(
                entry,
                inputs=_fold_prompt_into_inputs(prompt, inputs),
                state_file=state_file,
                thread_id=thread_id,
                metadata=metadata,
                dry_run=dry_run,
            )

        after = snapshot_workspace_files(before_roots | self.workspace_roots())
        if not result.artifacts:
            changes = diff_workspace_files(before, after)
            result.artifacts = [change["path"] for change in changes]
            result.artifact_changes = changes
        return result

    async def arun(
        self,
        name: str,
        *,
        prompt: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        kind: RuntimeKind | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        state_file: str | Path | None = None,
        dry_run: bool = False,
    ) -> RunResult:
        """Async variant of :meth:`run`, mirroring its resolution/build/prompt-folding logic."""
        entry = self._resolve(name, kind=kind)
        if dry_run:
            if entry.kind == "agent":
                return await self._arun_agent(
                    entry, prompt=prompt, inputs=inputs, thread_id=thread_id,
                    metadata=metadata, dry_run=True,
                )
            return await asyncio.to_thread(
                self._run_workflow, entry, inputs=_fold_prompt_into_inputs(prompt, inputs),
                state_file=state_file, thread_id=thread_id, metadata=metadata, dry_run=True,
            )

        # See ``run()`` -- same before/after workspace diff, so the async path gets
        # real ``artifacts`` too rather than only the sync one.
        before_roots = self.workspace_roots()
        before = await asyncio.to_thread(snapshot_workspace_files, before_roots)

        if entry.kind == "agent":
            result = await self._arun_agent(
                entry,
                prompt=prompt,
                inputs=inputs,
                thread_id=thread_id,
                metadata=metadata,
                dry_run=dry_run,
            )
        else:
            # ``RuntimeWorkflow`` drives its nodes synchronously (it calls
            # ``WorkspaceAgent.stream``/``._run_workflow`` directly rather than through an
            # async graph engine), so run it off the event loop rather than blocking it.
            result = await asyncio.to_thread(
                self._run_workflow,
                entry,
                inputs=_fold_prompt_into_inputs(prompt, inputs),
                state_file=state_file,
                thread_id=thread_id,
                metadata=metadata,
                dry_run=dry_run,
            )

        after = await asyncio.to_thread(
            snapshot_workspace_files, before_roots | self.workspace_roots()
        )
        if not result.artifacts:
            # difflib is CPU-bound (worst-case quadratic on a near-cap file), so
            # keep it off the event loop, same as the snapshots.
            changes = await asyncio.to_thread(diff_workspace_files, before, after)
            result.artifacts = [change["path"] for change in changes]
            result.artifact_changes = changes
        return result

    def stream(
        self,
        name: str,
        *,
        prompt: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        kind: RuntimeKind | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream_mode: str = "values",
    ) -> Iterator[Any]:
        entry = self._resolve(name, kind=kind)
        if entry.kind == "agent":
            resolved_prompt = self._require_prompt(entry, prompt=prompt, inputs=inputs)
            agent = self.build_agent(entry.name, kind="agent")
            for chunk in agent.stream([
                HumanMessage(content=resolved_prompt),
            ], thread_id=thread_id, stream_mode=stream_mode, metadata=metadata):
                yield chunk
            return

        workflow = self.build_workflow(entry.name, kind="workflow")
        yield from workflow.stream(
            inputs=_fold_prompt_into_inputs(prompt, inputs) or {},
            thread_id=thread_id,
            metadata=metadata,
            stream_mode=stream_mode,
        )

    async def astream(
        self,
        name: str,
        *,
        prompt: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        kind: RuntimeKind | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream_mode: str = "values",
    ) -> AsyncIterator[Any]:
        """Async variant of :meth:`stream`, mirroring its resolution/build/prompt-folding logic."""
        entry = self._resolve(name, kind=kind)
        if entry.kind == "agent":
            resolved_prompt = self._require_prompt(entry, prompt=prompt, inputs=inputs)
            # Building can block on ``_build_lock`` (held across another thread's
            # slow build) and on workspace-session creation, so keep it off the loop.
            agent = await asyncio.to_thread(self.build_agent, entry.name, kind="agent")
            async for chunk in agent.astream([
                HumanMessage(content=resolved_prompt),
            ], thread_id=thread_id, stream_mode=stream_mode, metadata=metadata):
                yield chunk
            return

        workflow = await asyncio.to_thread(
            self.build_workflow, entry.name, kind="workflow"
        )
        # ``RuntimeWorkflow.stream()`` is a plain sync generator (no async graph
        # engine backs it), so drain it on a background thread and hand chunks
        # back through the event loop rather than blocking it.
        sync_iterator = workflow.stream(
            inputs=_fold_prompt_into_inputs(prompt, inputs) or {},
            thread_id=thread_id,
            metadata=metadata,
            stream_mode=stream_mode,
        )
        async for chunk in _async_iter_from_sync(sync_iterator):
            yield chunk

    def _run_agent(
        self,
        entry: RuntimeEntry,
        *,
        prompt: str | None,
        inputs: Mapping[str, Any] | None,
        thread_id: str | None,
        metadata: Mapping[str, Any] | None,
        dry_run: bool,
    ) -> RunResult:
        resolved_prompt = self._require_prompt(entry, prompt=prompt, inputs=inputs)
        if dry_run:
            return RunResult(kind="agent", text=resolved_prompt)

        agent = self.build_agent(entry.name, kind="agent")
        output = agent.run(resolved_prompt, thread_id=thread_id, metadata=metadata)
        return _agent_result_from_output(resolved_prompt, output)

    async def _arun_agent(
        self,
        entry: RuntimeEntry,
        *,
        prompt: str | None,
        inputs: Mapping[str, Any] | None,
        thread_id: str | None,
        metadata: Mapping[str, Any] | None,
        dry_run: bool,
    ) -> RunResult:
        resolved_prompt = self._require_prompt(entry, prompt=prompt, inputs=inputs)
        if dry_run:
            return RunResult(kind="agent", text=resolved_prompt)

        # Building can block on ``_build_lock`` (held across another thread's
        # slow build) and on workspace-session creation, so keep it off the loop.
        agent = await asyncio.to_thread(self.build_agent, entry.name, kind="agent")
        output = await agent.arun(resolved_prompt, thread_id=thread_id, metadata=metadata)
        return _agent_result_from_output(resolved_prompt, output)

    def _run_workflow(
        self,
        entry: RuntimeEntry,
        *,
        inputs: Mapping[str, Any] | None,
        state_file: str | Path | None,
        thread_id: str | None,
        metadata: Mapping[str, Any] | None,
        dry_run: bool,
    ) -> RunResult:
        if state_file:
            state_path = _resolve_state_path(state_file)
            merged_inputs: dict[str, Any] = {}
            if state_path.exists():
                try:
                    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    raise AgentBuildError(f"Unable to read workflow state file: {state_path}")
                if not isinstance(state_payload, Mapping):
                    raise AgentBuildError(
                        f"Workflow state file must contain a JSON object: {state_path}"
                    )
                merged_inputs.update(dict(state_payload))

            if inputs:
                merged_inputs.update(dict(inputs))
        else:
            merged_inputs = dict(inputs or {})

        if dry_run:
            return RunResult(
                kind="workflow",
                text=None,
                state=merged_inputs,
                raw={"name": entry.name, "inputs": merged_inputs},
            )

        workflow = self.build_workflow(entry.name, kind="workflow")
        result: WorkflowRunResult = workflow.run(
            inputs=merged_inputs,
            thread_id=thread_id,
            metadata=metadata,
        )

        if state_file:
            state_path = _resolve_state_path(state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(_to_json_compatible(_public_state(result.state)), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        return RunResult(
            kind="workflow",
            text=result.text,
            messages=result.messages,
            state=result.state,
            # The workflow's declared ``outputs`` are the author's explicit result
            # contract; surface them as the primary payload so the default view is
            # the final output rather than the full execution state.
            structured_response=result.outputs or None,
            raw=result.raw,
            artifacts=[],
        )

    def _resolve(self, name: str, kind: RuntimeKind | None = None) -> RuntimeEntry:
        try:
            return self.registry.resolve(name, kind=kind)
        except KeyError as exc:
            msg = str(exc)
            if "Ambiguous" in msg:
                raise RunnableAmbiguousError(msg) from exc
            raise RunnableNotFoundError(msg) from exc

    def _resolve_prompt(
        self,
        entry: RuntimeEntry,
        *,
        prompt: str | None,
        inputs: Mapping[str, Any] | None,
    ) -> str | None:
        if prompt is not None:
            return prompt
        if inputs and isinstance(inputs.get("prompt"), str):
            return inputs["prompt"]
        return None

    def _require_prompt(
        self,
        entry: RuntimeEntry,
        *,
        prompt: str | None,
        inputs: Mapping[str, Any] | None,
    ) -> str:
        resolved = self._resolve_prompt(entry, prompt=prompt, inputs=inputs)
        if resolved is None:
            raise AgentBuildError(
                f"Agent '{entry.name}' requires a prompt argument or workflow input key 'prompt'."
            )
        return resolved

    def _agent_defaults_for_scope(self, scope: str) -> dict[str, Any]:
        if scope == "global":
            return self.context.global_agent_defaults
        if scope == "project":
            return self.context.project_agent_defaults
        return self.context.merged_agent_defaults


def _safe_entry_payload(entry: RuntimeEntry) -> dict[str, Any]:
    if entry.payload is None:
        return _load_entry_payload(entry.path)
    return dict(entry.payload)


def _deep_merge_defaults_for_entry(context: RuntimeContext, entry: RuntimeEntry, *, defaults_name: str) -> dict[str, Any]:
    if defaults_name == "agent":
        return context.merged_agent_defaults
    return context.merged_workflow_defaults
