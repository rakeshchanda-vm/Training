from __future__ import annotations

import secrets
from dataclasses import replace
from typing import Any, Awaitable, Callable, List, Sequence

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.messages import ToolMessage
from langchain_core.messages import AIMessage, BaseMessage

from andromeda.utils.logger import log_error


def _coerce_tool_call_id(value: Any) -> str:
    """ToolMessage.tool_call_id must be str; never pass None (e.g. tool_call['id'] is null)."""
    if value is None:
        return ""
    return str(value)


def _synthetic_tool_call_id() -> str:
    return f"call_{secrets.token_hex(12)}"


def _normalize_tool_call_dict(tc: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (possibly updated dict, changed)."""
    tid = tc.get("id")
    if tid is None or (isinstance(tid, str) and not tid.strip()):
        return {**tc, "id": _synthetic_tool_call_id()}, True
    return tc, False


def _patch_ai_message_tool_ids(msg: AIMessage) -> AIMessage:
    tcs = msg.tool_calls
    if not tcs:
        return msg
    new_tcs: List[Any] = []
    changed = False
    for tc in tcs:
        if isinstance(tc, dict):
            d, c = _normalize_tool_call_dict(tc)
            new_tcs.append(d)
            changed = changed or c
        elif hasattr(tc, "model_copy"):
            tid = getattr(tc, "id", None)
            if tid is None or (isinstance(tid, str) and not str(tid).strip()):
                new_tcs.append(tc.model_copy(update={"id": _synthetic_tool_call_id()}))
                changed = True
            else:
                new_tcs.append(tc)
        else:
            new_tcs.append(tc)
    if not changed:
        return msg
    return msg.model_copy(update={"tool_calls": new_tcs})


def _patch_model_response_tool_ids(response: Any) -> Any:
    """Normalize AIMessage.tool_calls so LangGraph ToolNode never sees id=None."""
    if isinstance(response, ExtendedModelResponse):
        inner = _patch_model_response_tool_ids(response.model_response)
        if inner is response.model_response:
            return response
        return replace(response, model_response=inner)

    if isinstance(response, ModelResponse):
        if not response.result:
            return response
        new_result: List[BaseMessage] = []
        changed = False
        for msg in response.result:
            if isinstance(msg, AIMessage):
                patched = _patch_ai_message_tool_ids(msg)
                new_result.append(patched)
                changed = changed or patched is not msg
            else:
                new_result.append(msg)
        if not changed:
            return response
        return replace(response, result=new_result)

    if isinstance(response, AIMessage):
        return _patch_ai_message_tool_ids(response)

    return response


def _extract_request_tool_call_id(request: Any) -> Any:
    """Best-effort extraction of a tool call id from LangChain request variants."""
    try:
        tool_call = getattr(request, "tool_call", None)
        if tool_call is None and hasattr(request, "get"):
            try:
                tool_call = request.get("tool_call")
            except Exception:
                tool_call = None

        if isinstance(tool_call, dict):
            return tool_call.get("id")
        if hasattr(tool_call, "id"):
            try:
                return getattr(tool_call, "id")
            except Exception:
                return None
    except Exception as exc:
        log_error(f"Unable to extract tool_call id from request: {exc}")
    return None


def _tool_error_message(request: Any, exc: Exception) -> ToolMessage:
    return ToolMessage(
        content=f"Tool error: Please check your input and try again. ({exc})",
        tool_call_id=_coerce_tool_call_id(_extract_request_tool_call_id(request)),
    )


class ToolErrorHandlerMiddleware(AgentMiddleware[AgentState[Any], ContextT, Any]):
    """Convert tool exceptions into ToolMessages on sync and async execution paths."""

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        try:
            return handler(request)
        except Exception as exc:  # noqa: BLE001
            return _tool_error_message(request, exc)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        try:
            return await handler(request)
        except Exception as exc:  # noqa: BLE001
            return _tool_error_message(request, exc)


tool_error_handler = ToolErrorHandlerMiddleware()


class EnsureToolCallIdsMiddleware(AgentMiddleware[AgentState[Any], ContextT, Any]):
    """Patch AIMessage.tool_calls missing ids before ToolNode builds ToolMessages.

    Some OpenAI-compatible stacks (e.g. Gemma) emit ``tool_calls`` with ``id: null``.
    LangGraph's ``ToolNode`` uses ``tool_call_id=call["id"]``, which pydantic rejects.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools: Sequence[Any] = []

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Any],
    ) -> Any:
        return _patch_model_response_tool_ids(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[Any]],
    ) -> Any:
        return _patch_model_response_tool_ids(await handler(request))


__all__ = [
    "EnsureToolCallIdsMiddleware",
    "ToolErrorHandlerMiddleware",
    "tool_error_handler",
    "_coerce_tool_call_id",
]
