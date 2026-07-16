from __future__ import annotations

from typing import Any, Dict, List

from andromeda.config.config import (
    CompliancePatternsConfig,
    DataPatternsConfig,
    PromptInjectionPatternsConfig,
)


def default_data_patterns() -> Dict[str, str]:
    cfg = DataPatternsConfig()
    patterns: Dict[str, str] = {}
    for field_name in ("email", "ssn", "phone", "credit_card"):
        value = getattr(cfg, field_name, "")
        if isinstance(value, str) and value.strip():
            patterns[field_name] = value
    if cfg.extra_patterns:
        for name, pattern in cfg.extra_patterns.items():
            if isinstance(pattern, str) and pattern.strip():
                patterns[str(name)] = pattern
    return patterns


def resolve_data_patterns(cfg: DataPatternsConfig) -> Dict[str, str]:
    pattern_map: Dict[str, str] = {}
    for field_name in ("email", "ssn", "phone", "credit_card"):
        value = getattr(cfg, field_name, "")
        if isinstance(value, str) and value.strip():
            pattern_map[field_name] = value

    if cfg.extra_patterns:
        for name, pattern in cfg.extra_patterns.items():
            if isinstance(pattern, str) and pattern.strip():
                pattern_map[str(name)] = pattern

    return pattern_map


def resolve_prompt_injection_patterns(cfg: PromptInjectionPatternsConfig) -> List[str]:
    return [p for p in cfg.patterns if isinstance(p, str) and p.strip()]


def resolve_compliance_patterns(cfg: CompliancePatternsConfig) -> List[str]:
    return [p for p in cfg.patterns if isinstance(p, str) and p.strip()]


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def rewrite_message_content(content: Any, rewritten_text: str) -> Any:
    if isinstance(content, str):
        return rewritten_text
    if isinstance(content, list):
        out: List[Any] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                updated = dict(block)
                updated["text"] = rewritten_text
                out.append(updated)
            elif isinstance(block, str):
                out.append(rewritten_text)
            else:
                out.append(block)
        return out
    return rewritten_text


def mask(value: str, *, keep_tail: int = 4) -> str:
    if len(value) <= keep_tail:
        return "*" * len(value)
    return "*" * (len(value) - keep_tail) + value[-keep_tail:]

