from __future__ import annotations

from typing import Any, List, Optional

from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware

from andromeda.config.config import MiddlewareConfig, ModelConfig
from andromeda.core.middleware.common import (
    resolve_compliance_patterns,
    resolve_data_patterns,
    resolve_prompt_injection_patterns,
)
from andromeda.core.middleware.guardrails import (
    ComplianceMiddleware,
    PromptInjectionMiddleware,
)
from andromeda.core.middleware.privacy import DataPrivacyMiddleware
from andromeda.core.middleware.tooling import tool_error_handler
from andromeda.utils.secure_store import InMemoryEncryptedTokenStore, get_secure_store


def resolve_summarization_model(
    summarization_cfg: MiddlewareConfig.SummarizationOptions,
    fallback_model: Any,
) -> Any:
    configured = summarization_cfg.model
    if configured is None:
        if isinstance(fallback_model, ModelConfig):
            from andromeda.utils.langtils import get_chat_model

            return get_chat_model(fallback_model)
        return fallback_model
    if isinstance(configured, ModelConfig):
        from andromeda.utils.langtils import get_chat_model

        return get_chat_model(configured)
    return configured


def is_config_enabled(cfg: MiddlewareConfig) -> bool:
    if cfg.enabled is False:
        return False

    has_guardrails = cfg.guardrails.input or cfg.guardrails.output or cfg.guardrails.tool
    has_masking = cfg.masking.input or cfg.masking.output or cfg.masking.tool
    inferred = (
        cfg.tool_error_handler
        or cfg.summarization is not None
        or cfg.hitl is not None
        or has_guardrails
        or has_masking
        or bool(cfg.custom)
    )
    return bool(cfg.enabled) if cfg.enabled is not None else inferred


def build_middleware(
    cfg: Optional[MiddlewareConfig],
    *,
    fallback_model: Any,
) -> List[Any]:
    if cfg is None or not is_config_enabled(cfg):
        return []

    middleware: List[Any] = []

    if cfg.tool_error_handler:
        middleware.append(tool_error_handler)

    if cfg.summarization is not None:
        middleware.append(
            SummarizationMiddleware(
                model=resolve_summarization_model(cfg.summarization, fallback_model),
                trigger=("tokens", cfg.summarization.trigger_tokens),
                keep=("messages", cfg.summarization.keep),
            )
        )

    if cfg.hitl is not None and cfg.hitl.interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=cfg.hitl.interrupt_on))

    if cfg.guardrails.input or cfg.guardrails.output or cfg.guardrails.tool:
        middleware.append(
            DataPrivacyMiddleware(
                strategy="block",
                apply_to_input=cfg.guardrails.input,
                apply_to_output=cfg.guardrails.output,
                apply_to_tool_results=cfg.guardrails.tool,
                patterns=resolve_data_patterns(cfg.guardrails.data_patterns),
                block_input_message=cfg.guardrails.blocked_message,
                block_output_message=cfg.guardrails.blocked_message,
                block_tool_message=cfg.guardrails.blocked_message,
            )
        )
        middleware.append(
            PromptInjectionMiddleware(
                apply_to_input=cfg.guardrails.input,
                apply_to_output=cfg.guardrails.output,
                apply_to_tool_results=cfg.guardrails.tool,
                patterns=resolve_prompt_injection_patterns(
                    cfg.guardrails.prompt_injection_patterns
                ),
                blocked_message=cfg.guardrails.blocked_message,
            )
        )
        middleware.append(
            ComplianceMiddleware(
                apply_to_output=cfg.guardrails.output,
                apply_to_tool_results=cfg.guardrails.tool,
                patterns=resolve_compliance_patterns(cfg.guardrails.compliance_patterns),
                replacement_message=cfg.guardrails.blocked_message,
            )
        )

    if cfg.masking.input or cfg.masking.output or cfg.masking.tool:
        token_store: Optional[InMemoryEncryptedTokenStore] = None
        if cfg.masking.strategy == "tokenize":
            token_store = get_secure_store(
                token_prefix=cfg.masking.token_prefix,
                ttl_seconds=cfg.masking.token_ttl_seconds,
            )

        middleware.append(
            DataPrivacyMiddleware(
                strategy=cfg.masking.strategy,
                apply_to_input=cfg.masking.input,
                apply_to_output=cfg.masking.output,
                apply_to_tool_results=cfg.masking.tool,
                patterns=resolve_data_patterns(cfg.masking.data_patterns),
                token_store=token_store,
            )
        )

    middleware.extend(cfg.custom)
    return middleware
