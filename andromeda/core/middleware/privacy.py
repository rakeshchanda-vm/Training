from __future__ import annotations

import hashlib
import re
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.messages import AIMessage, ToolMessage

from andromeda.core.middleware.common import (
    default_data_patterns,
    mask,
    message_text,
    rewrite_message_content,
)
from andromeda.utils.secure_store import InMemoryEncryptedTokenStore


class DataPrivacyMiddleware(AgentMiddleware):
    """PII/PHI-aware data privacy middleware with block/redact/mask/hash strategies."""

    def __init__(
        self,
        *,
        strategy: str = "redact",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        apply_to_tool_results: bool = False,
        patterns: Optional[Mapping[str, str]] = None,
        token_store: Optional[InMemoryEncryptedTokenStore] = None,
        block_input_message: str = "Request blocked by data privacy guardrail.",
        block_output_message: str = "Response blocked by data privacy guardrail.",
        block_tool_message: str = "Tool result blocked by data privacy guardrail.",
    ) -> None:
        super().__init__()
        if strategy not in {"redact", "mask", "hash", "block", "tokenize"}:
            raise ValueError(
                "strategy must be one of: redact, mask, hash, block, tokenize"
            )

        pattern_map = dict(patterns or default_data_patterns())
        self.patterns: Dict[str, re.Pattern[str]] = {
            name: re.compile(pattern, flags=re.IGNORECASE)
            for name, pattern in pattern_map.items()
            if isinstance(pattern, str) and pattern.strip()
        }
        self.strategy = strategy
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results
        self.block_input_message = block_input_message
        self.block_output_message = block_output_message
        self.block_tool_message = block_tool_message
        self.token_store = token_store

    def _transform(self, text: str) -> tuple[str, bool]:
        detected = False
        for pii_type, pattern in self.patterns.items():
            if not pattern.search(text):
                continue
            detected = True
            if self.strategy == "block":
                continue

            def _replacement(match: re.Match[str], *, _pii_type: str = pii_type) -> str:
                value = match.group(0)
                if self.strategy == "redact":
                    return f"[REDACTED_{_pii_type.upper()}]"
                if self.strategy == "mask":
                    return mask(value)
                if self.strategy == "tokenize":
                    if self.token_store is None:
                        raise ValueError(
                            "DataPrivacyMiddleware(strategy='tokenize') requires "
                            "a token_store."
                        )
                    return self.token_store.put(value)
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                return f"[HASH_{_pii_type.upper()}_{digest[:12]}]"

            text = pattern.sub(_replacement, text)

        return text, detected

    def _process_message(self, msg: Any) -> bool:
        content = message_text(getattr(msg, "content", ""))
        transformed, detected = self._transform(content)
        if not detected:
            return False
        if self.strategy == "block":
            return True
        msg.content = rewrite_message_content(getattr(msg, "content", ""), transformed)
        return True

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Any, runtime: Any) -> Dict[str, Any] | None:
        if not self.apply_to_input:
            return None

        for msg in state.get("messages", []):
            if str(getattr(msg, "type", "")) != "human":
                continue
            detected = self._process_message(msg)
            if detected and self.strategy == "block":
                return {
                    "messages": [AIMessage(content=self.block_input_message)],
                    "jump_to": "end",
                }

        return None

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: Any, runtime: Any) -> Dict[str, Any] | None:
        if not self.apply_to_output:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        last = messages[-1]
        if str(getattr(last, "type", "")) != "ai":
            return None

        detected = self._process_message(last)
        if detected and self.strategy == "block":
            return {
                "messages": [AIMessage(content=self.block_output_message)],
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
        transformed, detected = self._transform(text)
        if not detected:
            return result

        if self.strategy == "block":
            return ToolMessage(
                content=self.block_tool_message, tool_call_id=result.tool_call_id
            )

        result.content = transformed
        return result
