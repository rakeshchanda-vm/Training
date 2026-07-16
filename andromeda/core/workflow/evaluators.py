"""Built-in workflow evaluators.

These evaluators are optional helpers that can be attached to WorkflowBuilder nodes via
``with_evaluators([...])``. They are designed to be best-effort and safe to run in the
background: failures should not block the workflow.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Optional, TypedDict

from andromeda.utils.logger import log_warning

from .evaluation import LangfuseEvaluator, WorkflowEvaluationInput

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def _model_key(model: Any) -> Optional[str]:
    if model is None:
        return None
    # If already a model instance, treat it as a singleton.
    if hasattr(model, "invoke") or hasattr(model, "ainvoke"):
        return f"instance:{id(model)}"
    # Common dict / pydantic-ish configs.
    try:
        if isinstance(model, dict):
            name = model.get("name")
            provider = model.get("provider")
            other_args = model.get("other_args") or {}
            temperature = model.get("temperature", 0.0)
            return f"cfg:{name}:{provider}:{temperature}:{json.dumps(other_args, sort_keys=True, default=str)}"
        name = getattr(model, "name", None)
        provider = getattr(model, "provider", None)
        other_args = getattr(model, "other_args", None) or {}
        temperature = getattr(model, "temperature", 0.0)
        if name and provider:
            return f"cfg:{name}:{provider}:{temperature}:{json.dumps(other_args, sort_keys=True, default=str)}"
    except Exception:
        return None
    return None


def _resolve_eval_model(model: Any) -> Any:
    """Resolve a model instance from either an instantiated model or a config-like object."""

    if model is None:
        return None
    if hasattr(model, "invoke") or hasattr(model, "ainvoke"):
        return model

    key = _model_key(model)
    if key is None:
        return None

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            from langchain.chat_models import init_chat_model
        except Exception:  # noqa: BLE001
            return None

        if isinstance(model, dict):
            name = model.get("name")
            provider = model.get("provider")
            other_args = model.get("other_args") or {}
            temperature = model.get("temperature", 0.0)
        else:
            name = getattr(model, "name", None)
            provider = getattr(model, "provider", None)
            other_args = getattr(model, "other_args", None) or {}
            temperature = getattr(model, "temperature", 0.0)

        if not name or not provider:
            return None

        try:
            instance = init_chat_model(
                name,
                model_provider=provider,
                temperature=float(temperature) if temperature is not None else 0.0,
                **(other_args if isinstance(other_args, dict) else {}),
            )
            _MODEL_CACHE[key] = instance
            return instance
        except Exception:  # noqa: BLE001
            return None


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


async def _llm_score(
    *,
    inp: WorkflowEvaluationInput,
    default_prompt: str,
    model: Any = None,
) -> Optional[tuple[float, Optional[str]]]:
    from langchain_core.language_models import BaseChatModel
    resolved_model: BaseChatModel = _resolve_eval_model(model if model is not None else inp.model)
    if resolved_model is None:
        log_warning(
            "Built-in evaluator requires an eval model; skipping.\n"
            "- pass a model config via WorkflowBuilder.with_evaluators(..., model=...)"
        )
        return None

    prompt = inp.prompt or default_prompt

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:  # noqa: BLE001
        return None

    payload = {
        "input": inp.input,
        "output": inp.output,
    }
    request = json.dumps(payload, ensure_ascii=False)
    messages = [SystemMessage(content=prompt), HumanMessage(content=request)]

    class ScoreSchema(TypedDict):
        score: float
        comment: str
    try:
        # Prefer async when available.
        if hasattr(resolved_model, "ainvoke"):
            response = await resolved_model.with_structured_output(ScoreSchema, include_raw=True).ainvoke(messages)
        else:
            response = resolved_model.with_structured_output(ScoreSchema, include_raw=True).invoke(messages)
        
        if isinstance(response, dict):
            text = response.get("raw", "")
            data = response.get("parsed")
            if not data:
                data = _extract_json(str(text))
        else:
            text = getattr(response, "content", None)
            if not isinstance(text, str):
                text = str(response)
                data = _extract_json(text)
        if not isinstance(data, dict):
            return None
    except Exception as exc:  # noqa: BLE001
        log_warning(
            "Built-in evaluator model invocation failed; skipping.\n"
            f"- error: {exc.__class__.__name__}: {exc}"
        )
        return None    

    score_raw = data.get("score")
    comment = data.get("comment")
    try:
        score = float(score_raw)
    except Exception:
        return None
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return score, comment if isinstance(comment, str) else None


_CORRECTNESS_PROMPT = (
    "You are an evaluator. Score the assistant output for correctness based on the given input.\n"
    "Return ONLY JSON: {\"score\": <float 0..1>, \"comment\": <string or null>}.\n"
    "Score 1 = fully correct, 0 = incorrect or unsupported."
)

_HALLUCINATION_PROMPT = (
    "You are an evaluator. Score the assistant output for hallucinations based on the given input.\n"
    "Return ONLY JSON: {\"score\": <float 0..1>, \"comment\": <string or null>}.\n"
    "Score 1 = no hallucinations, 0 = clearly hallucinated or fabricated."
)

_RELEVANCE_PROMPT = (
    "You are an evaluator. Score how relevant the assistant output is to the input.\n"
    "Return ONLY JSON: {\"score\": <float 0..1>, \"comment\": <string or null>}.\n"
    "Score 1 = directly relevant, 0 = irrelevant."
)


def correctness(
    *,
    sample_rate: float = 1.0,
    prompt: Optional[str] = None,
    model: Any = None,
) -> LangfuseEvaluator:
    async def _eval(inp: WorkflowEvaluationInput):
        res = await _llm_score(inp=inp, default_prompt=_CORRECTNESS_PROMPT, model=model)
        if res is None:
            return None
        score, comment = res
        return score, comment

    return LangfuseEvaluator(
        name="correctness",
        data_type="NUMERIC",
        eval_fn=_eval,
        prompt=prompt,
        sample_rate=sample_rate,
        input_preset="trajectory",
        output_preset="trajectory",
    )


def hallucination(
    *,
    sample_rate: float = 1.0,
    prompt: Optional[str] = None,
    model: Any = None,
) -> LangfuseEvaluator:
    async def _eval(inp: WorkflowEvaluationInput):
        res = await _llm_score(inp=inp, default_prompt=_HALLUCINATION_PROMPT, model=model)
        if res is None:
            return None
        score, comment = res
        return score, comment

    return LangfuseEvaluator(
        name="hallucination",
        data_type="NUMERIC",
        eval_fn=_eval,
        prompt=prompt,
        sample_rate=sample_rate,
        input_preset="trajectory",
        output_preset="trajectory",
    )


def relevance(
    *,
    sample_rate: float = 1.0,
    prompt: Optional[str] = None,
    model: Any = None,
) -> LangfuseEvaluator:
    async def _eval(inp: WorkflowEvaluationInput):
        res = await _llm_score(inp=inp, default_prompt=_RELEVANCE_PROMPT, model=model)
        if res is None:
            return None
        score, comment = res
        return score, comment

    return LangfuseEvaluator(
        name="relevance",
        data_type="NUMERIC",
        eval_fn=_eval,
        prompt=prompt,
        sample_rate=sample_rate,
        input_preset="trajectory",
        output_preset="trajectory",
    )


def tool_usage(*, sample_rate: float = 1.0) -> LangfuseEvaluator:
    def _eval(inp: WorkflowEvaluationInput):
        tools_called = inp.trajectory.get("tools_called") or []
        tool_results = inp.trajectory.get("tool_results") or []
        used = bool(tools_called)
        has_results = bool(tool_results)
        score = 1.0 if (used and has_results) else (0.5 if used else 0.0)
        comment = (
            "Used tools and captured results."
            if score == 1.0
            else ("Used tools but no tool results observed." if used else "No tool usage observed.")
        )
        return score, comment

    return LangfuseEvaluator(
        name="tool_usage",
        data_type="NUMERIC",
        eval_fn=_eval,
        sample_rate=sample_rate,
        input_preset="trajectory",
        output_preset="trajectory",
    )
