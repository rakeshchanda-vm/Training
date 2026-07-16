from __future__ import annotations

from typing import Any, AsyncIterator, Iterator

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models import base as lc_openai_base


def _get_chunk_attr(chunk: Any, key: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(key, default)
    return getattr(chunk, key, default)


def _responses_reasoning_delta_to_generation_chunk(
    chunk: Any,
    current_index: int,
    current_output_index: int,
    current_sub_index: int,
    *,
    output_version: str | None,
) -> tuple[int, int, int, ChatGenerationChunk | None]:
    """Convert Responses reasoning text deltas omitted by langchain-openai.

    ``langchain-openai`` currently handles reasoning output items and reasoning
    summaries, but not ``response.reasoning_text.delta`` events. OpenAI-compatible
    backends that expose full reasoning text stream those events with the same
    output/content indexes used by text deltas.
    """

    output_index = _get_chunk_attr(chunk, "output_index")
    content_index = _get_chunk_attr(chunk, "content_index")
    delta = _get_chunk_attr(chunk, "delta", "")

    if output_index is not None:
        if content_index is None:
            if current_output_index != output_index:
                current_index += 1
        elif (
            current_output_index != output_index
            or current_sub_index != content_index
        ):
            current_index += 1
        current_output_index = output_index
        if content_index is not None:
            current_sub_index = content_index

    if not isinstance(delta, str) or not delta:
        return current_index, current_output_index, current_sub_index, None

    content_block: dict[str, Any] = {
        "type": "reasoning",
        "reasoning": delta,
        "index": current_index,
    }
    item_id = _get_chunk_attr(chunk, "item_id")
    if isinstance(item_id, str) and item_id:
        content_block["id"] = item_id

    response_metadata: dict[str, Any] = {"model_provider": "openai"}
    model_name = _get_chunk_attr(chunk, "model")
    if isinstance(model_name, str) and model_name:
        response_metadata["model_name"] = model_name

    message = AIMessageChunk(
        content=[content_block],
        response_metadata=response_metadata,
    )
    if output_version == "v0":
        message = lc_openai_base._convert_to_v03_ai_message(message)

    return (
        current_index,
        current_output_index,
        current_sub_index,
        ChatGenerationChunk(message=message),
    )


def _convert_responses_chunk_to_generation_chunk(
    chunk: Any,
    current_index: int,
    current_output_index: int,
    current_sub_index: int,
    *,
    schema: Any = None,
    metadata: dict | None = None,
    has_reasoning: bool = False,
    output_version: str | None = None,
) -> tuple[int, int, int, ChatGenerationChunk | None]:
    if _get_chunk_attr(chunk, "type") == "response.reasoning_text.delta":
        return _responses_reasoning_delta_to_generation_chunk(
            chunk,
            current_index,
            current_output_index,
            current_sub_index,
            output_version=output_version,
        )

    return lc_openai_base._convert_responses_chunk_to_generation_chunk(
        chunk,
        current_index,
        current_output_index,
        current_sub_index,
        schema=schema,
        metadata=metadata,
        has_reasoning=has_reasoning,
        output_version=output_version,
    )


def _generation_chunk_has_reasoning(generation_chunk: ChatGenerationChunk) -> bool:
    message = generation_chunk.message
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if any(key in additional_kwargs for key in ("reasoning", "reasoning_content")):
        return True

    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False

    return any(
        isinstance(block, dict) and block.get("type") in ("reasoning", "thinking")
        for block in content
    )


class AndromedaChatOpenAI(ChatOpenAI):
    """ChatOpenAI with Responses streaming reasoning-text delta support."""

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return self._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in self._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def _stream_responses(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._ensure_sync_client_available()
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            if self.include_response_headers:
                raw_context_manager = (
                    self.root_client.with_raw_response.responses.create(**payload)
                )
                context_manager = raw_context_manager.parse()
                headers = {"headers": dict(raw_context_manager.headers)}
            else:
                context_manager = self.root_client.responses.create(**payload)
                headers = {}
            original_schema_obj = kwargs.get("response_format")

            with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                for chunk in response:
                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                    )
                    if generation_chunk:
                        if run_manager:
                            run_manager.on_llm_new_token(
                                generation_chunk.text, chunk=generation_chunk
                            )
                        is_first_chunk = False
                        if _generation_chunk_has_reasoning(generation_chunk):
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as e:
            lc_openai_base._handle_openai_bad_request(e)
        except openai.APIError as e:
            lc_openai_base._handle_openai_api_error(e)

    async def _astream_responses(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        try:
            if self.include_response_headers:
                raw_context_manager = (
                    await self.root_async_client.with_raw_response.responses.create(
                        **payload
                    )
                )
                context_manager = raw_context_manager.parse()
                headers = {"headers": dict(raw_context_manager.headers)}
            else:
                context_manager = await self.root_async_client.responses.create(
                    **payload
                )
                headers = {}
            original_schema_obj = kwargs.get("response_format")

            async with context_manager as response:
                is_first_chunk = True
                current_index = -1
                current_output_index = -1
                current_sub_index = -1
                has_reasoning = False
                async for chunk in lc_openai_base._astream_with_chunk_timeout(
                    response,
                    self.stream_chunk_timeout,
                    model_name=self.model_name,
                ):
                    metadata = headers if is_first_chunk else {}
                    (
                        current_index,
                        current_output_index,
                        current_sub_index,
                        generation_chunk,
                    ) = _convert_responses_chunk_to_generation_chunk(
                        chunk,
                        current_index,
                        current_output_index,
                        current_sub_index,
                        schema=original_schema_obj,
                        metadata=metadata,
                        has_reasoning=has_reasoning,
                        output_version=self.output_version,
                    )
                    if generation_chunk:
                        if run_manager:
                            await run_manager.on_llm_new_token(
                                generation_chunk.text, chunk=generation_chunk
                            )
                        is_first_chunk = False
                        if _generation_chunk_has_reasoning(generation_chunk):
                            has_reasoning = True
                        yield generation_chunk
        except openai.BadRequestError as e:
            lc_openai_base._handle_openai_bad_request(e)
        except openai.APIError as e:
            lc_openai_base._handle_openai_api_error(e)
