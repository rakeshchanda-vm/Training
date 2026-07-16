"""Optional workflow evaluation hooks.

Currently this module focuses on ingesting per-node evaluation scores into Langfuse.

Langfuse scoring docs (conceptually):
- Scores can be ingested via SDK/API and linked to traces/observations.
- Score data types: NUMERIC, CATEGORICAL, BOOLEAN.
"""

from __future__ import annotations

from collections.abc import Awaitable
import asyncio
import random
from dataclasses import dataclass
import inspect
import os
import threading
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from andromeda.utils.logger import log_warning
from .base import ExecutionContext
from .eval_scheduler import SchedulerConfig


LangfuseScoreDataType = Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"]
EvaluationPreset = Literal[
    "raw_payload",
    "first_message",
    "last_message",
    "all_messages",
    "tools_called",
    "tool_results",
    "trajectory",
]


@dataclass(frozen=True, slots=True)
class WorkflowEvaluationInput:
    """Input provided to a workflow evaluator."""

    step_name: str
    before: dict[str, Any]
    after: dict[str, Any]
    input: Any
    output: Any
    trajectory: dict[str, Any]
    context: ExecutionContext
    model: Any = None
    prompt: Optional[str] = None


@dataclass(frozen=True, slots=True)
class LangfuseScore:
    """A single score to ingest into Langfuse."""

    name: str
    value: Union[float, str, int, bool]
    data_type: Optional[LangfuseScoreDataType] = None
    comment: Optional[str] = None
    score_id: Optional[str] = None
    config_id: Optional[str] = None


EvaluatorReturn = Union[
    None,
    LangfuseScore,
    Sequence[LangfuseScore],
    Union[float, str, int, bool],
    Tuple[Union[float, str, int, bool], Optional[str]],
]


@dataclass(frozen=True, slots=True)
class LangfuseEvaluator:
    """Evaluator configuration attached to a WorkflowBuilder node.

    The evaluator function receives a :class:`WorkflowEvaluationInput` and returns either:
    - a value (numeric / categorical string / boolean),
    - a (value, comment) tuple,
    - a LangfuseScore (or list of LangfuseScore),
    - or None to skip scoring.
    """

    name: str
    eval_fn: Callable[[WorkflowEvaluationInput], Union[EvaluatorReturn, Awaitable[EvaluatorReturn]]]
    prompt: Optional[str] = None
    data_type: Optional[LangfuseScoreDataType] = None
    config_id: Optional[str] = None
    score_id: Optional[str] = None
    sample_rate: float = 1.0
    input_preset: EvaluationPreset = "trajectory"
    output_preset: EvaluationPreset = "trajectory"


@dataclass(frozen=True, slots=True)
class EvaluatorRuntimeConfig:
    """Runtime settings applied to a node's evaluator group."""

    scheduler: Optional[SchedulerConfig] = None
    model: Any = None


def _langfuse_env_configured() -> bool:
    return bool(
        os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_HOST")
    )


def _serialize_message(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": msg.__class__.__name__}
    try:
        content = getattr(msg, "content", None)
        if content is not None:
            out["content"] = content
    except Exception:
        pass

    # Tool calls (AIMessage.tool_calls or additional_kwargs["tool_calls"])
    tool_calls: Any = None
    try:
        tool_calls = getattr(msg, "tool_calls", None)
    except Exception:
        tool_calls = None
    if not tool_calls:
        try:
            additional_kwargs = getattr(msg, "additional_kwargs", None)
            if isinstance(additional_kwargs, dict):
                tool_calls = additional_kwargs.get("tool_calls")
        except Exception:
            tool_calls = None
    if tool_calls:
        out["tool_calls"] = tool_calls

    # Tool message linkage
    for attr in ("tool_call_id", "name", "id"):
        try:
            val = getattr(msg, attr, None)
            if val is not None:
                out[attr] = val
        except Exception:
            continue
    return out


def _extract_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("messages")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for msg in raw:
        if isinstance(msg, dict):
            out.append(msg)
        else:
            out.append(_serialize_message(msg))
    return out


def _first_message_text(messages: Sequence[dict[str, Any]]) -> Optional[str]:
    if not messages:
        return None
    first = messages[0]
    content = first.get("content")
    return content if isinstance(content, str) else None


def _last_message_text(messages: Sequence[dict[str, Any]]) -> Optional[str]:
    if not messages:
        return None
    last = messages[-1]
    content = last.get("content")
    return content if isinstance(content, str) else None


def _extract_tool_calls(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict):
                    calls.append(call)
                else:
                    calls.append({"raw": call})
    return calls


def _extract_tool_results(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for msg in messages:
        msg_type = str(msg.get("type", "")).lower()
        if msg_type in {"toolmessage", "tool"} or msg.get("tool_call_id") is not None:
            results.append(
                {
                    "tool_call_id": msg.get("tool_call_id"),
                    "name": msg.get("name"),
                    "content": msg.get("content"),
                }
            )
    return results


def _build_trajectory(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_msgs = _extract_messages(before)
    after_msgs = _extract_messages(after)
    tool_calls = _extract_tool_calls(after_msgs)
    tool_results = _extract_tool_results(after_msgs)
    return {
        "input": _first_message_text(before_msgs) or _first_message_text(after_msgs),
        "output": _last_message_text(after_msgs),
        "messages": after_msgs,
        "tools_called": tool_calls,
        "tool_results": tool_results,
    }


def _select_preset(
    preset: EvaluationPreset,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    trajectory: dict[str, Any],
    source: Literal["before", "after"],
) -> Any:
    payload = before if source == "before" else after
    messages = _extract_messages(payload)

    if preset == "raw_payload":
        return payload
    if preset == "first_message":
        return _first_message_text(messages)
    if preset == "last_message":
        return _last_message_text(messages)
    if preset == "all_messages":
        return messages
    if preset == "tools_called":
        # Tool calls are typically emitted in "after".
        return trajectory.get("tools_called", []) if source == "after" else _extract_tool_calls(messages)
    if preset == "tool_results":
        return trajectory.get("tool_results", []) if source == "after" else _extract_tool_results(messages)
    if preset == "trajectory":
        return trajectory
    return payload


def _normalize_score_value(
    value: Union[float, str, int, bool],
    data_type: Optional[LangfuseScoreDataType],
) -> tuple[Union[float, str], Optional[LangfuseScoreDataType]]:
    # If explicitly typed, validate/coerce.
    if data_type == "NUMERIC":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("NUMERIC scores require an int/float value.")
        return float(value), data_type
    if data_type == "CATEGORICAL":
        if not isinstance(value, str):
            raise TypeError("CATEGORICAL scores require a string value.")
        return value, data_type
    if data_type == "BOOLEAN":
        if isinstance(value, bool):
            return float(1 if value else 0), data_type
        if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
            return float(value), data_type
        raise TypeError("BOOLEAN scores require a bool or numeric 0/1 value.")

    # Infer type when not provided.
    if isinstance(value, bool):
        return float(1 if value else 0), "BOOLEAN"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), "NUMERIC"
    if isinstance(value, str):
        return value, "CATEGORICAL"
    raise TypeError("Unsupported score value type.")


def _coerce_evaluator_return(
    evaluator: LangfuseEvaluator,
    result: EvaluatorReturn,
) -> list[LangfuseScore]:
    if result is None:
        return []
    if isinstance(result, LangfuseScore):
        return [result]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        if not result:
            return []
        if isinstance(result[0], LangfuseScore):
            return list(result)  # type: ignore[return-value]
    if isinstance(result, tuple) and len(result) == 2:
        value, comment = result
        return [
            LangfuseScore(
                name=evaluator.name,
                value=value,
                data_type=evaluator.data_type,
                comment=comment,
                score_id=evaluator.score_id,
                config_id=evaluator.config_id,
            )
        ]
    # Plain value (numeric/categorical/bool)
    return [
        LangfuseScore(
            name=evaluator.name,
            value=result,  # type: ignore[arg-type]
            data_type=evaluator.data_type,
            score_id=evaluator.score_id,
            config_id=evaluator.config_id,
        )
    ]


def _scoped_score_name(*, step_name: str, evaluator_name: str, score_name: str) -> str:
    """Prefix score names with the node name to disambiguate per-step scoring."""

    if score_name.startswith(f"{step_name}:"):
        return score_name
    # Only auto-scope names that match the evaluator name (i.e. "correctness").
    if score_name == evaluator_name:
        return f"{step_name}:{score_name}"
    return score_name


class _LangfuseScorer:
    """Internal helper that ingests scores into Langfuse via the Python SDK."""

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._local = threading.local()

    def _ensure_client(self) -> Optional[Any]:
        if self._available is False:
            return None
        if self._available is True:
            existing = getattr(self._local, "client", None)
            if existing is not None:
                return existing

        if not _langfuse_env_configured():
            log_warning(
                "Langfuse scoring is enabled but Langfuse env vars are not configured; skipping score ingestion.\n"
                "- required: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST"
            )
            self._available = False
            return None

        try:
            from langfuse import get_client  # type: ignore
        except Exception:  # noqa: BLE001
            log_warning(
                "Langfuse scoring is enabled but the 'langfuse' package is not installed; skipping score ingestion.\n"
                "- install: pip install langfuse"
            )
            self._available = False
            return None

        self._available = True
        try:
            client = get_client()
            setattr(self._local, "client", client)
            return client
        except Exception:  # noqa: BLE001
            return None

    def create_scores(
        self,
        *,
        trace_id: str,
        step_name: str,
        scores: Iterable[LangfuseScore],
    ) -> None:
        client = self._ensure_client()
        if client is None:
            return

        for score in scores:
            try:
                normalized_value, inferred_type = _normalize_score_value(
                    score.value, score.data_type
                )
                client.create_score(
                    trace_id=trace_id,
                    name=score.name,
                    value=normalized_value,
                    data_type=inferred_type,
                    comment=score.comment,
                    score_id=score.score_id,
                    config_id=score.config_id,
                )
            except Exception as exc:  # noqa: BLE001
                log_warning(
                    "Langfuse scoring failed; continuing without blocking workflow.\n"
                    f"- step: {step_name}\n"
                    f"- score: {score.name}\n"
                    f"- error: {exc.__class__.__name__}: {exc}"
                )

        flush = getattr(client, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                # Best-effort flush; do not fail workflow.
                pass


_scorer_singleton = _LangfuseScorer()


def ingest_langfuse_scores(
    *,
    context: ExecutionContext,
    step_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    evaluators: Sequence[LangfuseEvaluator],
    runtime: Optional[EvaluatorRuntimeConfig] = None,
) -> None:
    trace_id = (
        str(context.metadata.get("langfuse_trace_id") or "")
        or context.thread_id
        or str(context.metadata.get("thread_id") or "")
    )
    if not trace_id:
        return

    trajectory = _build_trajectory(before, after)
    resolved_model = runtime.model if runtime is not None else None
    all_scores: list[LangfuseScore] = []
    for evaluator in evaluators:
        p = evaluator.sample_rate
        if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
            log_warning(
                "Invalid evaluator sample_rate; skipping evaluator.\n"
                f"- step: {step_name}\n"
                f"- evaluator: {evaluator.name}\n"
                f"- sample_rate: {p!r}"
            )
            continue
        if float(p) < 1.0 and random.random() >= float(p):
            continue

        try:
            result = evaluator.eval_fn(
                WorkflowEvaluationInput(
                    step_name=step_name,
                    before=before,
                    after=after,
                    input=_select_preset(
                        evaluator.input_preset,
                        before=before,
                        after=after,
                        trajectory=trajectory,
                        source="before",
                    ),
                    output=_select_preset(
                        evaluator.output_preset,
                        before=before,
                        after=after,
                        trajectory=trajectory,
                        source="after",
                    ),
                    trajectory=trajectory,
                    context=context,
                    model=resolved_model,
                    prompt=evaluator.prompt,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                "Workflow evaluator raised; skipping its Langfuse score.\n"
                f"- step: {step_name}\n"
                f"- evaluator: {evaluator.name}\n"
                f"- error: {exc.__class__.__name__}: {exc}"
            )
            continue

        if inspect.isawaitable(result):
            try:
                try:
                    asyncio.get_running_loop()
                    log_warning(
                        "Async evaluator returned an awaitable but ingest_langfuse_scores() is running on an event loop; skipping.\n"
                        f"- step: {step_name}\n"
                        f"- evaluator: {evaluator.name}"
                    )
                    continue
                except RuntimeError:
                    pass

                result = asyncio.run(result)  # type: ignore[arg-type]
            except Exception as exc:  # noqa: BLE001
                log_warning(
                    "Async evaluator failed; skipping its Langfuse score.\n"
                    f"- step: {step_name}\n"
                    f"- evaluator: {evaluator.name}\n"
                    f"- error: {exc.__class__.__name__}: {exc}"
                )
                continue

        coerced = _coerce_evaluator_return(evaluator, result)
        for score in coerced:
            all_scores.append(
                LangfuseScore(
                    name=_scoped_score_name(
                        step_name=step_name,
                        evaluator_name=evaluator.name,
                        score_name=score.name,
                    ),
                    value=score.value,
                    data_type=score.data_type,
                    comment=score.comment,
                    score_id=score.score_id,
                    config_id=score.config_id,
                )
            )

    if all_scores:
        _scorer_singleton.create_scores(
            trace_id=trace_id, step_name=step_name, scores=all_scores
        )


async def ingest_langfuse_scores_async(
    *,
    context: ExecutionContext,
    step_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    evaluators: Sequence[LangfuseEvaluator],
    runtime: Optional[EvaluatorRuntimeConfig] = None,
) -> None:
    trace_id = (
        str(context.metadata.get("langfuse_trace_id") or "")
        or context.thread_id
        or str(context.metadata.get("thread_id") or "")
    )
    if not trace_id:
        return

    trajectory = _build_trajectory(before, after)
    resolved_model = runtime.model if runtime is not None else None
    all_scores: list[LangfuseScore] = []
    for evaluator in evaluators:
        p = evaluator.sample_rate
        if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
            log_warning(
                "Invalid evaluator sample_rate; skipping evaluator.\n"
                f"- step: {step_name}\n"
                f"- evaluator: {evaluator.name}\n"
                f"- sample_rate: {p!r}"
            )
            continue
        if float(p) < 1.0 and random.random() >= float(p):
            continue

        try:
            result_or_awaitable = evaluator.eval_fn(
                WorkflowEvaluationInput(
                    step_name=step_name,
                    before=before,
                    after=after,
                    input=_select_preset(
                        evaluator.input_preset,
                        before=before,
                        after=after,
                        trajectory=trajectory,
                        source="before",
                    ),
                    output=_select_preset(
                        evaluator.output_preset,
                        before=before,
                        after=after,
                        trajectory=trajectory,
                        source="after",
                    ),
                    trajectory=trajectory,
                    context=context,
                    model=resolved_model,
                    prompt=evaluator.prompt,
                )
            )
            result = (
                await result_or_awaitable
                if inspect.isawaitable(result_or_awaitable)
                else result_or_awaitable
            )
        except Exception as exc:  # noqa: BLE001
            log_warning(
                "Workflow evaluator raised; skipping its Langfuse score.\n"
                f"- step: {step_name}\n"
                f"- evaluator: {evaluator.name}\n"
                f"- error: {exc.__class__.__name__}: {exc}"
            )
            continue

        coerced = _coerce_evaluator_return(evaluator, result)
        for score in coerced:
            all_scores.append(
                LangfuseScore(
                    name=_scoped_score_name(
                        step_name=step_name,
                        evaluator_name=evaluator.name,
                        score_name=score.name,
                    ),
                    value=score.value,
                    data_type=score.data_type,
                    comment=score.comment,
                    score_id=score.score_id,
                    config_id=score.config_id,
                )
            )

    if all_scores:
        await asyncio.to_thread(
            _scorer_singleton.create_scores,
            trace_id=trace_id,
            step_name=step_name,
            scores=all_scores,
        )
