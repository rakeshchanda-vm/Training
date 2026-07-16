
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
)
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.embeddings import init_embeddings, Embeddings

from andromeda.config import ModelConfig


def _uses_openai_responses_api(model_config: ModelConfig) -> bool:
    other_args = model_config.other_args or {}
    explicit = other_args.get("use_responses_api")
    if explicit is True:
        return True
    if explicit is False:
        return False

    if model_config.output_version == "responses/v1":
        return True

    responses_only_or_selecting_args = (
        "context_management",
        "include",
        "reasoning",
        "truncation",
        "use_previous_response_id",
    )
    return any(
        other_args.get(key) is not None for key in responses_only_or_selecting_args
    )


def get_chat_model(model_config: ModelConfig) -> BaseChatModel:
    if model_config.provider == "github_copilot":
        from andromeda.utils.github_copilot import ChatGithubCopilot
        gh_kwargs = {
            "temperature": model_config.temperature,
            **model_config.other_args,
        }
        gh_kwargs.setdefault("streaming", True)
        return ChatGithubCopilot(
            model=model_config.name,
            output_version=model_config.output_version,
            **gh_kwargs,
        )
    if model_config.provider == "openai_codex":
        from andromeda.utils.openai_codex import ChatOpenAICodex
        return ChatOpenAICodex(
            model=model_config.name,
            output_version=model_config.output_version,
            **model_config.other_args,
        )
    if model_config.provider == "litellm":
        from langchain_litellm import ChatLiteLLM
        litellm_kwargs = {
            "temperature": model_config.temperature,
            **model_config.other_args,
        }
        litellm_kwargs.setdefault("streaming", True)
        return ChatLiteLLM(
            model=model_config.name if '/' in model_config.name else f'openai/{model_config.name}',
            output_version=model_config.output_version,
            **litellm_kwargs,
        )
    if model_config.provider == "openai" and _uses_openai_responses_api(model_config):
        from andromeda.utils.openai_responses import AndromedaChatOpenAI

        openai_kwargs = {
            "temperature": model_config.temperature,
            **model_config.other_args,
        }
        return AndromedaChatOpenAI(
            model=model_config.name,
            output_version=model_config.output_version,
            **openai_kwargs,
        )
    return init_chat_model(
        model_config.name,
        model_provider=model_config.provider,
        output_version=model_config.output_version,
        **{
            "temperature": model_config.temperature,
            **model_config.other_args,
        },
    )


def get_embedding_model(model_config: ModelConfig) -> Embeddings:
    if model_config.provider == "github_copilot":
        from andromeda.utils.github_copilot import GithubCopilotEmbeddings
        return GithubCopilotEmbeddings(
            model=model_config.name,
            **model_config.other_args,
        )
    if model_config.provider == "litellm":
        from langchain_litellm import LiteLLMEmbeddings
        return LiteLLMEmbeddings(
            model=model_config.name if '/' in model_config.name else f'openai/{model_config.name}',
            **model_config.other_args,
        )

    return init_embeddings(
        model=model_config.name,
        provider=model_config.provider,
        **model_config.other_args,
    )

def _json_dumps_safe(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def _extract_reasoning_text(block: Dict[str, Any]) -> Optional[str]:
    """
    Supports LangChain's standardized blocks + provider-ish variants.
    OpenAI-style reasoning blocks sometimes carry:
      {"type":"reasoning","summary":[{"type":"summary_text","text":"..."}]}
    LangChain standardized is usually:
      {"type":"reasoning","reasoning":"..."}
    """
    if block.get("type") == "non_standard" and isinstance(block.get("value"), dict):
        inner = block["value"]
        if inner.get("type") in ("reasoning", "thinking"):
            block = inner

    if block.get("type") not in ("reasoning", "thinking"):
        return None

    if "reasoning" in block and isinstance(block["reasoning"], str):
        return block["reasoning"]

    def _extract_reasoning_content_items(items: Any) -> Optional[str]:
        if not isinstance(items, list):
            return None
        parts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in ("reasoning_text", "summary_text", "text"):
                continue
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        if parts:
            return "".join(parts)
        return None

    # vLLM/OpenAI-compatible Responses API blocks may carry reasoning text in
    # nested content items. LangChain currently preserves provider-specific
    # fields under extras for this shape.
    nested_reasoning = _extract_reasoning_content_items(block.get("content"))
    if nested_reasoning is not None:
        return nested_reasoning

    extras = block.get("extras")
    if isinstance(extras, dict):
        nested_reasoning = _extract_reasoning_content_items(extras.get("content"))
        if nested_reasoning is not None:
            return nested_reasoning

    summary = block.get("summary")
    if isinstance(summary, list):
        parts = []
        for item in summary:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)
        if parts:
            return "\n".join(parts)

    # Some providers use "thinking" field
    if "thinking" in block and isinstance(block["thinking"], str):
        return block["thinking"]

    # Metadata-only reasoning block (no textual reasoning/summary) should be
    # treated as a known non-user-facing block, not serialized as text.
    return ""


def _is_ignored_non_standard_block(block: Dict[str, Any]) -> bool:
    # Providers occasionally emit metadata/control payloads as non_standard
    # blocks alongside tool calls or tool results. Unless they unwrap into a
    # reasoning/thinking block handled above, they should not be surfaced.
    return block.get("type") == "non_standard"


def strip_non_standard_content_blocks(content: Any) -> Any:
    """
    Drop LangChain ``non_standard`` content blocks from list-shaped message content.

    Used before observability export (e.g. Langfuse) so provider escape-hatch
    payloads are not ingested. String and other content shapes are unchanged.
    """
    if not isinstance(content, list):
        return content
    filtered: List[Any] = [
        b
        for b in content
        if not (isinstance(b, dict) and b.get("type") == "non_standard")
    ]
    if not filtered:
        return ""
    return filtered


def _append_unique_text(parts: List[str], value: Any) -> None:
    if isinstance(value, str) and value and (not parts or parts[-1] != value):
        parts.append(value)


def _collapse_content_blocks_for_observability(
    content: List[Any],
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """Collapse block-shaped message content into trace-friendly text fields."""

    text_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        reasoning = _extract_reasoning_text(block)
        if reasoning is not None:
            _append_unique_text(reasoning_parts, reasoning)
            continue

        if _is_ignored_non_standard_block(block):
            continue

        if _is_tool_call(block):
            tc = _tool_call_from_block(block)
            if tc is not None:
                tool_calls.append(tc)
                continue

        if _is_tool_call_chunk(block):
            continue

        btype = block.get("type")

        if btype in ("text", "text-plain"):
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
            else:
                text_parts.append(_json_dumps_safe(block))
            continue

        if btype in ("image", "audio", "video", "file", "image_url"):
            text_parts.append(_stringify_multimodal_block(block))
            continue

        if _is_server_tool_result(block):
            text_parts.append(_json_dumps_safe(block.get("output")))
            continue

        text_parts.append(_json_dumps_safe(block))

    reasoning_text = "".join(reasoning_parts) if reasoning_parts else None
    return "".join(text_parts), reasoning_text, tool_calls


def collapse_message_for_observability(msg: BaseMessage) -> BaseMessage:
    """Collapse list-shaped content into a single text payload for tracing/export."""

    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return msg

    text, reasoning_text, tool_calls = _collapse_content_blocks_for_observability(content)

    additional_kwargs = dict(getattr(msg, "additional_kwargs", {}) or {})
    merged_reasoning: List[str] = []
    _append_unique_text(merged_reasoning, additional_kwargs.get("reasoning_content"))
    _append_unique_text(merged_reasoning, reasoning_text)
    if merged_reasoning:
        additional_kwargs["reasoning_content"] = "\n".join(merged_reasoning)

    updated_fields: Dict[str, Any] = {
        "content": text,
        "additional_kwargs": additional_kwargs,
    }

    existing_tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        updated_fields["tool_calls"] = tool_calls
    elif isinstance(existing_tool_calls, list):
        updated_fields["tool_calls"] = existing_tool_calls

    return msg.model_copy(update=updated_fields)


def collapse_message_dict_for_observability(data: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse model-dumped message dicts into a trace-friendly shape."""

    content = data.get("content")
    if not isinstance(content, list):
        return data

    text, reasoning_text, tool_calls = _collapse_content_blocks_for_observability(content)
    out = dict(data)
    out["content"] = text

    additional_kwargs = dict(out.get("additional_kwargs") or {})
    merged_reasoning: List[str] = []
    _append_unique_text(merged_reasoning, additional_kwargs.get("reasoning_content"))
    _append_unique_text(merged_reasoning, reasoning_text)
    if merged_reasoning:
        additional_kwargs["reasoning_content"] = "\n".join(merged_reasoning)
    if additional_kwargs or "additional_kwargs" in out:
        out["additional_kwargs"] = additional_kwargs

    if tool_calls:
        out["tool_calls"] = tool_calls

    return out

def _stringify_multimodal_block(block: Dict[str, Any]) -> str:
    """
    Converts image/audio/video/file blocks into a string.
    Preference order: data URI from base64, else URL, else file_id/id.
    """
    btype = block.get("type", "non_standard")
    mime = block.get("mime_type")

    if "base64" in block and isinstance(block["base64"], str) and block["base64"]:
        if isinstance(mime, str) and mime:
            return f"data:{mime};base64,{block['base64']}"
        # no mime: keep base64 as-is
        return block["base64"]

    if "url" in block and isinstance(block["url"], str) and block["url"]:
        return block["url"]

    # Some providers use nested image_url structure
    if btype == "image_url":
        image_url = block.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str) and url:
                return url

    fid = block.get("file_id") or block.get("id")
    if isinstance(fid, str) and fid:
        return f"[{btype} id={fid}]"

    return f"[{btype}]"

def _is_server_tool_result(block: Dict[str, Any]) -> bool:
    return block.get("type") == "server_tool_result"

def _is_tool_call(block: Dict[str, Any]) -> bool:
    return block.get("type") in ("tool_call", "server_tool_call")

def _is_tool_call_chunk(block: Dict[str, Any]) -> bool:
    return block.get("type") == "tool_call_chunk"

def _tool_call_from_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize into LangChain tool_calls shape: {"name": str, "args": dict, "id": str}
    """
    name = block.get("name")
    args = block.get("args")
    call_id = block.get("id")

    if isinstance(name, str) and isinstance(call_id, str):
        # args may be dict or a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {"raw": args}
        return {"name": name, "args": args, "id": call_id}
    return None

def _make_tool_message_from_server_result(block: Dict[str, Any]) -> ToolMessage:
    """
    Map:
      {
        "type":"server_tool_result",
        "tool_call_id":"...",
        "status":"success"|"error",
        "output": <any>,
        ...
      }
    to ToolMessage(content=str, tool_call_id=str, name=str)
    """
    tool_call_id = block.get("tool_call_id") or block.get("id") or "unknown"
    status = block.get("status", "success")
    output = block.get("output")

    # Keep output as a compact string; include status wrapper if needed
    if status != "success":
        content = f"[tool_error] { _json_dumps_safe(output) }"
    else:
        content = _json_dumps_safe(output)

    # ToolMessage requires a name; if you don't have it, pick a stable placeholder.
    # If your blocks include it elsewhere, swap in that field here.
    return ToolMessage(
        content=content,
        tool_call_id=str(tool_call_id),
        name="server_tool",
    )

def explode_message_to_base_messages(msg: BaseMessage) -> List[BaseMessage]:
    """
    - Ensures every returned message has content as a string.
    - Converts embedded blocks into multiple BaseMessages (preserving block order).
    - Reasoning blocks become additional_kwargs["reasoning_content"] on the nearest emitted
      non-Tool message segment (the one we’re currently building).
    - Server-side tool results become ToolMessage entries.
    - Tool call blocks become AIMessage.tool_calls (if original is AIMessage); otherwise
      they are stringified into content.

    If msg.content is already a string, returns [msg] with optional reasoning_content untouched.
    """
    content = msg.content
    if isinstance(content, str):
        return [msg]

    if not isinstance(content, list):
        # Defensive fallback: stringify unknown content
        return [msg.model_copy(update={"content": str(content)})]

    out: List[BaseMessage] = []

    # Segment builder state
    seg_text_parts: List[str] = []
    seg_reasoning_parts: List[str] = []
    seg_tool_calls: List[Dict[str, Any]] = []

    def flush_segment_if_any() -> None:
        nonlocal seg_text_parts, seg_reasoning_parts, seg_tool_calls, out

        has_any = bool(seg_text_parts) or bool(seg_reasoning_parts) or bool(seg_tool_calls)
        if not has_any:
            return

        # Merge reasoning into additional_kwargs
        ak = dict(getattr(msg, "additional_kwargs", {}) or {})
        if seg_reasoning_parts:
            # Reasoning arrives as streamed delta fragments; concatenate them as
            # the litellm/OpenAI client does.
            ak["reasoning_content"] = "".join(seg_reasoning_parts)

        # Content must be string
        text = "".join(seg_text_parts)

        # Preserve original message type for the segment
        updated_fields: Dict[str, Any] = {"content": text, "additional_kwargs": ak}
        # Clear inherited stale tool_calls for AI messages (model_copy preserves them otherwise)
        if isinstance(msg, AIMessage) or isinstance(msg, BaseMessageChunk):
            updated_fields["tool_calls"] = []

        seg_msg = msg.model_copy(update=updated_fields)

        # If this is an AIMessage, attach tool_calls (LangChain uses .tool_calls attr)
        if (isinstance(seg_msg, AIMessage) or isinstance(seg_msg, BaseMessageChunk)) and seg_tool_calls:
            seg_msg.tool_calls = list(seg_tool_calls)

        out.append(seg_msg)

        # Reset segment
        seg_text_parts = []
        seg_reasoning_parts = []
        seg_tool_calls = []

    # Walk blocks in order, emitting ToolMessages inline
    for block in content:
        if not isinstance(block, dict):
            seg_text_parts.append(str(block))
            continue

        r = _extract_reasoning_text(block)
        if r is not None:
            if r:
                if not seg_reasoning_parts or seg_reasoning_parts[-1] != r:
                    seg_reasoning_parts.append(r)
            continue

        btype = block.get("type")

        if _is_ignored_non_standard_block(block):
            continue

        if _is_server_tool_result(block):
            flush_segment_if_any()
            out.append(_make_tool_message_from_server_result(block))
            continue
        

        if _is_tool_call(block):
            tc = _tool_call_from_block(block)
            if tc and isinstance(msg, AIMessage):
                seg_tool_calls.append(tc)
            else:
                # Non-AI message or malformed call: serialize into text
                seg_text_parts.append(_json_dumps_safe(block))
            continue

        # Streaming tool call chunks are control metadata; avoid leaking them as user-facing text.
        if _is_tool_call_chunk(block):
            continue

        if btype in ("text", "text-plain"):
            t = block.get("text")
            if isinstance(t, str):
                seg_text_parts.append(t)
            else:
                seg_text_parts.append(_json_dumps_safe(block))
            continue

        # 5) Multimodal blocks
        if btype in ("image", "audio", "video", "file", "image_url"):
            seg_text_parts.append(_stringify_multimodal_block(block))
            continue

        # 6) Everything else (invalid tool chunks etc.) → stringify
        seg_text_parts.append(_json_dumps_safe(block))

    flush_segment_if_any()

    if not out:
        ak = dict(getattr(msg, "additional_kwargs", {}) or {})
        # If content was only reasoning, stash it
        reasonings = []
        for b in content:
            if isinstance(b, dict):
                r = _extract_reasoning_text(b)
                if r:
                    reasonings.append(r)
        if reasonings:
            ak["reasoning_content"] = "".join(reasonings)
        out = [msg.model_copy(update={"content": "", "additional_kwargs": ak})]

    return out


def normalize_message_list(messages: List[BaseMessage]) -> List[BaseMessage]:
    normalized: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            normalized.extend(explode_message_to_base_messages(m))
    return normalized
