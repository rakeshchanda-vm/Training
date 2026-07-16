from __future__ import annotations

from pathlib import Path
from typing import Any

from andromeda import BaseMessage


_BASE_MESSAGE_TYPE = BaseMessage if isinstance(BaseMessage, type) else ()


def serialize_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any]
    try:
        dumper = getattr(message, "model_dump", None) or getattr(message, "dict", None)
        payload = dumper() if dumper else {}
    except Exception:
        payload = {}

    payload["type"] = message.__class__.__name__
    content = payload.get("content")
    if content is None:
        content = getattr(message, "content", None)
    if content is not None:
        payload["content"] = content

    return {
        key: to_json_compatible(value)
        for key, value in payload.items()
        if value is not None
    }


def message_content(message: Any) -> Any:
    return getattr(message, "content", message)


def to_json_compatible(value: Any) -> Any:
    if _BASE_MESSAGE_TYPE and isinstance(value, _BASE_MESSAGE_TYPE):
        return serialize_message(value)

    if isinstance(value, dict):
        return {key: to_json_compatible(entry) for key, entry in value.items()}

    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, tuple):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, set):
        return [to_json_compatible(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return to_json_compatible(value.model_dump())
        except Exception:
            return str(value)

    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return to_json_compatible(value.dict())
        except Exception:
            return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return value
