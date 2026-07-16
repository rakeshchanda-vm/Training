from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.messages import AIMessage, ToolMessage

from andromeda.core.middleware.common import message_text


class PromptInjectionMiddleware(AgentMiddleware):
    """Prompt injection and instruction-conflict guardrails with configurable patterns."""

    def __init__(
        self,
        *,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
        patterns: Optional[List[str]] = None,
        blocked_message: str = "Prompt injection detected. I can't comply with prompt-injection requests.",
    ) -> None:
        super().__init__()
        self.patterns = [re.compile(p, flags=re.IGNORECASE) for p in (patterns or [])]
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results
        self.blocked_message = blocked_message

    def _contains_injection(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> Dict[str, Any] | None:
        if not self.apply_to_input:
            return None

        for msg in reversed(state.get("messages", [])):
            if str(getattr(msg, "type", "")) != "human":
                continue
            text = message_text(getattr(msg, "content", ""))
            if self._contains_injection(text):
                return {
                    "messages": [AIMessage(content=self.blocked_message)],
                    "jump_to": "end",
                }
            break

        return None

    def after_model(self, state: Any, runtime: Any) -> Dict[str, Any] | None:
        if not self.apply_to_output:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if str(getattr(last, "type", "")) != "ai":
            return None

        text = message_text(getattr(last, "content", ""))
        if self._contains_injection(text):
            return {
                "messages": [AIMessage(content=self.blocked_message)],
                "jump_to": "end",
            }

        return None

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = handler(request)
        return self._process_tool_result(result)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        result = await handler(request)
        return self._process_tool_result(result)

    def _process_tool_result(self, result: Any) -> Any:
        if not self.apply_to_tool_results or not isinstance(result, ToolMessage):
            return result

        text = message_text(result.content)
        if self._contains_injection(text):
            return ToolMessage(
                content=self.blocked_message, tool_call_id=result.tool_call_id
            )
        return result


class ComplianceMiddleware(AgentMiddleware):
    """Broad insurance/compliance output guardrails with configurable patterns."""

    def __init__(
        self,
        *,
        apply_to_output: bool = True,
        apply_to_tool_results: bool = False,
        patterns: Optional[List[str]] = None,
        replacement_message: str = (
            "I can't provide guidance that could be deceptive, discriminatory, "
            "or non-compliant with insurance/security regulations."
        ),
    ) -> None:
        super().__init__()
        self.patterns = [re.compile(p, flags=re.IGNORECASE) for p in (patterns or [])]
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results
        self.replacement_message = replacement_message

    def _is_non_compliant(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)

    def after_model(self, state: Any, runtime: Any) -> Dict[str, Any] | None:
        if not self.apply_to_output:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if str(getattr(last, "type", "")) != "ai":
            return None

        if self._is_non_compliant(message_text(getattr(last, "content", ""))):
            last.content = self.replacement_message

        return None

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = handler(request)
        return self._process_tool_result(result)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        result = await handler(request)
        return self._process_tool_result(result)

    def _process_tool_result(self, result: Any) -> Any:
        if not self.apply_to_tool_results or not isinstance(result, ToolMessage):
            return result

        if self._is_non_compliant(message_text(result.content)):
            return ToolMessage(
                content=self.replacement_message, tool_call_id=result.tool_call_id
            )

        return result
