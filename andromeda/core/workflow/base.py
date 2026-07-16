"""Shared workflow abstractions built on top of LangGraph."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import contextlib
import contextvars
import os
import sys
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
)
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RunnableConfig
from andromeda.core.workflow.checkpointing import CheckpointerProvider
from andromeda.utils.logger import log_output, log_warning
from andromeda.utils.langtils import (
    collapse_message_dict_for_observability,
    collapse_message_for_observability,
)

_LANGFUSE_ACTIVE_SPAN_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "langfuse_active_span_depth", default=0
)

_LANGFUSE_LANGCHAIN_HANDLER_TYPE: Optional[type] = None


def _langfuse_env_configured() -> bool:
    """Return True when Langfuse is explicitly configured via environment variables."""

    return bool(
        os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_HOST")
    )


if (
    os.getenv("LANGFUSE_SECRET_KEY")
    and os.getenv("LANGFUSE_PUBLIC_KEY")
    and os.getenv("LANGFUSE_HOST")
):
    try:
        from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler  # type: ignore

        _LANGFUSE_LANGCHAIN_HANDLER_TYPE = _LangfuseCallbackHandler

        class _AndromedaLangfuseCallbackHandler(_LangfuseCallbackHandler):  # type: ignore[misc]
            """Langfuse handler that omits ``non_standard`` blocks from recorded I/O."""

            def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
                message = collapse_message_for_observability(message)
                return super()._convert_message_to_dict(message)

            def on_llm_end(self, response: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> Any:
                inp = kwargs.get("inputs")
                if inp is not None:
                    kwargs = dict(kwargs)
                    kwargs["inputs"] = _deep_strip_non_standard_for_langfuse(inp)
                return super().on_llm_end(
                    response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
                )

            def on_llm_error(
                self,
                error: BaseException,
                *,
                run_id: Any,
                parent_run_id: Any = None,
                **kwargs: Any,
            ) -> Any:
                inp = kwargs.get("inputs")
                if inp is not None:
                    kwargs = dict(kwargs)
                    kwargs["inputs"] = _deep_strip_non_standard_for_langfuse(inp)
                return super().on_llm_error(
                    error, run_id=run_id, parent_run_id=parent_run_id, **kwargs
                )

        _LANGFUSE_CALLBACK_HANDLER_CLS = _AndromedaLangfuseCallbackHandler
    except ImportError:
        log_warning("Langfuse ENV variables found but Langfuse is not installed; proceeding without Langfuse integration. pip install langfuse to enable.")
        _LANGFUSE_CALLBACK_HANDLER_CLS = None
else:
    _LANGFUSE_CALLBACK_HANDLER_CLS = None


def _create_langfuse_handler(*, trace_id: Optional[str] = None) -> Any:
    """Create a per-execution Langfuse LangChain callback handler."""
    if _LANGFUSE_CALLBACK_HANDLER_CLS is None:
        return None
    return _LANGFUSE_CALLBACK_HANDLER_CLS()


def _iter_config_callbacks(callbacks: Any) -> Iterator[Any]:
    """Yield leaf callback handler objects from LangChain config ``callbacks``."""

    if callbacks is None:
        return
    if isinstance(callbacks, (list, tuple)):
        for item in callbacks:
            yield from _iter_config_callbacks(item)
        return
    nested = getattr(callbacks, "handlers", None)
    if nested:
        for item in nested:
            yield from _iter_config_callbacks(item)
        return
    inheritable = getattr(callbacks, "inheritable_handlers", None)
    if inheritable:
        for item in inheritable:
            yield from _iter_config_callbacks(item)
        return
    yield callbacks


def _graph_config_inherits_langfuse_handler(config: Mapping[str, Any]) -> bool:
    """Return True when inherited runnable config already includes a Langfuse handler."""

    base = _LANGFUSE_LANGCHAIN_HANDLER_TYPE
    if base is None:
        return False
    for cb in _iter_config_callbacks(config.get("callbacks")):
        try:
            if isinstance(cb, base):
                return True
        except TypeError:
            continue
    return False


def _create_langfuse_trace_id(seed: str) -> Optional[str]:
    """Create a deterministic Langfuse trace_id from an external seed."""

    # Avoid initializing the Langfuse client when it isn't configured. The
    # upstream SDK emits a warning on init when keys are missing.
    if not _langfuse_env_configured():
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        trace_id = Langfuse.create_trace_id(seed=seed)
        return trace_id
    except Exception as e:
        log_warning(str(e))
        pass

    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
        trace_id = client.create_trace_id(seed=seed)
        return trace_id
    except Exception as e:
        log_warning(str(e))
        return None


def _deep_strip_non_standard_for_langfuse(value: Any) -> Any:
    """Recursively collapse message blocks into trace-friendly Langfuse payloads."""

    if isinstance(value, BaseMessage):
        return collapse_message_for_observability(value)
    if isinstance(value, dict):
        normalized = {
            k: _deep_strip_non_standard_for_langfuse(v) for k, v in value.items()
        }
        if (
            "content" in normalized
            and any(
                key in normalized
                for key in ("additional_kwargs", "tool_calls", "response_metadata", "type", "id")
            )
        ):
            return collapse_message_dict_for_observability(normalized)
        return normalized
    if isinstance(value, list):
        return [_deep_strip_non_standard_for_langfuse(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_deep_strip_non_standard_for_langfuse(v) for v in value)
    return value


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable data for Langfuse."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseMessage):
        try:
            data = collapse_message_for_observability(value).model_dump()
        except Exception:
            collapsed = collapse_message_for_observability(value)
            data = {"type": value.__class__.__name__, "content": collapsed.content}
        return _to_jsonable(data)
    if isinstance(value, dict):
        normalized = {str(k): _to_jsonable(v) for k, v in value.items()}
        if (
            "content" in normalized
            and any(
                key in normalized
                for key in ("additional_kwargs", "tool_calls", "response_metadata", "type", "id")
            )
        ):
            return collapse_message_dict_for_observability(normalized)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    for attr in ("model_dump", "dict"):
        try:
            fn = getattr(value, attr, None)
            if callable(fn):
                return _to_jsonable(fn())
        except Exception:
            pass

    try:
        content = getattr(value, "content", None)
        if content is not None:
            out: dict[str, Any] = {"type": value.__class__.__name__, "content": content}
            tool_calls = getattr(value, "tool_calls", None)
            if tool_calls:
                out["tool_calls"] = _to_jsonable(tool_calls)
            return out
    except Exception:
        pass

    try:
        return str(value)
    except Exception:
        return "<unserializable>"


def _update_langfuse_trace(
    span: Any, *, input_state: Any = None, output_state: Any = None, attributes: Optional[Mapping[str, Any]] = None
) -> None:
    """Best-effort trace update using the span's update_trace API."""

    if span is None:
        return
    update_trace = getattr(span, "update_trace", None)
    if not callable(update_trace):
        return
    payload: dict[str, Any] = {}
    if input_state is not None:
        payload["input"] = _to_jsonable(input_state)
    if output_state is not None:
        payload["output"] = _to_jsonable(output_state)
    if attributes is not None:
        attributes_payload = _to_jsonable(attributes)
    else:
        attributes_payload = {}
    if not payload and not attributes_payload:
        return
    try:
        update_trace(**payload, **attributes_payload)
    except Exception:
        pass


@contextlib.contextmanager
def _langfuse_trace_context(context: ExecutionContext, *, name: str) -> Iterator[Any]:
    """Ensure Langfuse spans use a trace_id derived from the workflow thread_id."""

    if not _langfuse_env_configured():
        yield None
        return

    trace_id = context.metadata.get("langfuse_trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        seed = context.thread_id or str(context.metadata.get("thread_id") or "")
        if seed:
            trace_id = _create_langfuse_trace_id(seed)
            if trace_id:
                context.metadata["langfuse_trace_id"] = trace_id

    if not trace_id:
        yield None
        return

    try:
        from langfuse import get_client  # type: ignore

        langfuse = get_client()
        has_parent_observation = _LANGFUSE_ACTIVE_SPAN_DEPTH.get() > 0
        if has_parent_observation:
            span_cm = langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                metadata=context.metadata,
            )
        else:
            span_cm = langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                trace_context={"trace_id": trace_id},
                metadata=context.metadata,
            )
        span = span_cm.__enter__()
    except Exception as e:
        log_warning(str(e))
        yield None
        return

    exited = False
    depth_token = _LANGFUSE_ACTIVE_SPAN_DEPTH.set(_LANGFUSE_ACTIVE_SPAN_DEPTH.get() + 1)
    try:
        try:
            obs_id = (
                getattr(span, "id", None)
                or getattr(span, "observation_id", None)
                or getattr(span, "span_id", None)
            )
            if isinstance(obs_id, str) and obs_id and not has_parent_observation:
                context.metadata.setdefault("langfuse_root_observation_id", obs_id)
        except Exception:
            pass

        try:
            yield span
        except BaseException:
            exc_type, exc, tb = sys.exc_info()
            suppress = False
            try:
                suppress = bool(span_cm.__exit__(exc_type, exc, tb))
                exited = True
            except Exception as exit_exc:
                with contextlib.suppress(Exception):
                    log_warning(str(exit_exc))
            if suppress:
                return
            raise
        else:
            try:
                span_cm.__exit__(None, None, None)
                exited = True
            except Exception as exit_exc:
                with contextlib.suppress(Exception):
                    log_warning(str(exit_exc))
    finally:
        _LANGFUSE_ACTIVE_SPAN_DEPTH.reset(depth_token)
        if not exited:
            with contextlib.suppress(Exception):
                span_cm.__exit__(None, None, None)


@contextlib.asynccontextmanager
async def _alangfuse_trace_context(
    context: ExecutionContext, *, name: str
) -> AsyncIterator[Any]:
    if not _langfuse_env_configured():
        yield None
        return

    trace_id = context.metadata.get("langfuse_trace_id")
    if not isinstance(trace_id, str) or not trace_id:
        seed = context.thread_id or str(context.metadata.get("thread_id") or "")
        if seed:
            trace_id = _create_langfuse_trace_id(seed)
            if trace_id:
                context.metadata["langfuse_trace_id"] = trace_id

    if not trace_id:
        yield None
        return

    try:
        from langfuse import get_client  # type: ignore

        langfuse = get_client()
        has_parent_observation = _LANGFUSE_ACTIVE_SPAN_DEPTH.get() > 0
        if has_parent_observation:
            span_cm = langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                metadata=context.metadata,
            )
        else:
            span_cm = langfuse.start_as_current_observation(
                as_type="span",
                name=name,
                trace_context={"trace_id": trace_id},
                metadata=context.metadata,
            )
        span = span_cm.__enter__()
    except Exception as e:
        log_warning(str(e))
        yield None
        return

    exited = False
    depth_token = _LANGFUSE_ACTIVE_SPAN_DEPTH.set(_LANGFUSE_ACTIVE_SPAN_DEPTH.get() + 1)
    try:
        try:
            obs_id = (
                getattr(span, "id", None)
                or getattr(span, "observation_id", None)
                or getattr(span, "span_id", None)
            )
            if isinstance(obs_id, str) and obs_id and not has_parent_observation:
                context.metadata.setdefault("langfuse_root_observation_id", obs_id)
        except Exception:
            pass

        try:
            yield span
        except BaseException:
            exc_type, exc, tb = sys.exc_info()
            suppress = False
            try:
                suppress = bool(span_cm.__exit__(exc_type, exc, tb))
                exited = True
            except Exception as exit_exc:
                with contextlib.suppress(Exception):
                    log_warning(str(exit_exc))
            if suppress:
                return
            raise
        else:
            try:
                span_cm.__exit__(None, None, None)
                exited = True
            except Exception as exit_exc:
                with contextlib.suppress(Exception):
                    log_warning(str(exit_exc))
    finally:
        _LANGFUSE_ACTIVE_SPAN_DEPTH.reset(depth_token)
        if not exited:
            with contextlib.suppress(Exception):
                span_cm.__exit__(None, None, None)


@dataclass(slots=True)
class ExecutionContext:
    """Holds metadata and transient state for a workflow execution."""

    name: str
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    debug: bool = False
    monitor: bool = False
    metadata: MutableMapping[str, Any] = field(default_factory=dict)
    state: MutableMapping[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    thread_id: Optional[str] = None
    env: MutableMapping[str, str] = field(default_factory=dict)
    toolkit: Optional[Any] = None
    mcp_runtime: Optional[Any] = None

    def record(self, key: str, value: Any) -> None:
        """Store an arbitrary value in the execution state."""

        self.state[key] = value


@dataclass(slots=True)
class WorkflowResult:
    """Represents the outcome returned by a workflow run."""

    data: Any
    context: ExecutionContext
    interrupts: Optional[Any] = None


class WorkflowExecutionError(RuntimeError):
    """Raised when a workflow fails during execution."""


class WorkflowBase:
    """Abstract base class shared by all workflow authoring styles."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        checkpointer: Optional[Any] = None,
    ) -> None:
        self.name = name or self.__class__.__name__
        self._compiled_graph: Optional[CompiledStateGraph] = None
        self._compiled_graph_async: Optional[CompiledStateGraph] = None
        self._checkpointer_provider = CheckpointerProvider(checkpointer)

    def _prepare_execution_context(
        self,
        *,
        execution_context: Optional[ExecutionContext] = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        async_mode: bool = False,
    ) -> ExecutionContext:
        context = execution_context or ExecutionContext(name=self.name)
        context.name = self.name
        context.debug = debug
        context.monitor = monitor
        if metadata:
            context.metadata.update(dict(metadata))
        context.metadata["__async__"] = async_mode
        context.metadata.setdefault("execution_id", context.execution_id)
        if thread_id is not None:
            context.thread_id = thread_id
            context.metadata["thread_id"] = thread_id
        return context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        *args: Any,
        state: Optional[Any] = None,
        resume: Any = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute the workflow synchronously.

        Concrete subclasses are expected to implement ``_execute``.
        ``state`` is forwarded verbatim while ``args``/``kwargs``
        can be used to pass additional contextual information.
        """

        context = self._prepare_execution_context(
            execution_context=execution_context,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            async_mode=False,
        )

        try:
            graph = self._ensure_compiled(context, async_mode=False, **kwargs)
            graph_config = self._graph_config(context, thread_id=thread_id)

            with _langfuse_trace_context(context, name=f"workflow:{self.name}") as span:
                try:
                    if resume is not None:
                        if isinstance(resume, Command):
                            resume_command = resume
                        elif isinstance(resume, dict):
                            resume_command = Command(
                                goto=resume.get("goto"), update=resume.get("update")
                            )
                        else:
                            resume_command = Command(resume=resume)
                        _update_langfuse_trace(span, input_state={"resume": resume_command}, attributes=context.metadata.get("langfuse_attributes", {}))
                        raw_result = graph.invoke(
                            resume_command, config=graph_config, debug=debug, **kwargs
                        )
                    else:
                        initial_state = self._initial_state(context, state, *args, **kwargs)
                        _update_langfuse_trace(span, input_state=initial_state, attributes=context.metadata.get("langfuse_attributes", {}))
                        raw_result = graph.invoke(
                            initial_state, config=graph_config, debug=debug, **kwargs
                        )
                    _update_langfuse_trace(span, output_state=raw_result, attributes=context.metadata.get("langfuse_attributes", {}))
                except Exception as e:
                    _update_langfuse_trace(span, output_state={
                        "error": str(e)
                    }, attributes=context.metadata.get("langfuse_attributes", {}))
                    raise
            interrupts = None
            if isinstance(raw_result, dict) and "__interrupt__" in raw_result:
                interrupts = raw_result["__interrupt__"]
                context.record("interrupts", interrupts)
                data = raw_result
            else:
                data = self._post_process(raw_result, context)
                if isinstance(data, MutableMapping):
                    try:
                        state_snapshot = graph.get_state(graph_config)
                        if state_snapshot.tasks:
                            task_interrupts = state_snapshot.tasks[0].interrupts
                            if task_interrupts:
                                interrupts = task_interrupts[0].value
                                data = dict(data)
                                data.setdefault("__interrupt__", interrupts)
                                context.record("interrupts", interrupts)
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            self._handle_failure(exc, context)
            raise WorkflowExecutionError(str(exc)) from exc

        self._handle_success(context)
        return WorkflowResult(data=data, context=context, interrupts=interrupts)

    async def arun(
        self,
        *args: Any,
        state: Optional[Any] = None,
        resume: Any = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Asynchronous variant of :meth:`run` using LangGraph's ``ainvoke``."""

        context = self._prepare_execution_context(
            execution_context=execution_context,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            async_mode=True,
        )

        try:
            graph = await self._ensure_compiled_async(context, **kwargs)
            graph_config = self._graph_config(context, thread_id=thread_id)

            async with _alangfuse_trace_context(context, name=f"workflow:{self.name}") as span:
                try:
                    if resume is not None:
                        if isinstance(resume, Command):
                            resume_command = resume
                        elif isinstance(resume, dict):
                            resume_command = Command(
                                goto=resume.get("goto"), update=resume.get("update")
                            )
                        else:
                            resume_command = Command(resume=resume)
                        _update_langfuse_trace(span, input_state={"resume": resume_command})
                        raw_result = await graph.ainvoke(
                            resume_command, config=graph_config, debug=debug, **kwargs
                        )
                    else:
                        initial_state = self._initial_state(context, state, *args, **kwargs)
                        _update_langfuse_trace(span, input_state=initial_state)
                        raw_result = await graph.ainvoke(
                            initial_state, config=graph_config, debug=debug, **kwargs
                        )
                    _update_langfuse_trace(span, output_state=raw_result)  
                except Exception as e:
                    _update_langfuse_trace(span, output_state={
                        "error": str(e)
                    })
                    raise e

            interrupts = None
            if isinstance(raw_result, dict) and "__interrupt__" in raw_result:
                interrupts = raw_result["__interrupt__"]
                context.record("interrupts", interrupts)
                data = raw_result
            else:
                data = self._post_process(raw_result, context)
                if isinstance(data, MutableMapping):
                    try:
                        state_snapshot = await graph.aget_state(graph_config)
                        if state_snapshot.tasks:
                            task_interrupts = state_snapshot.tasks[0].interrupts
                            if task_interrupts:
                                interrupts = task_interrupts[0].value
                                data = dict(data)
                                data.setdefault("__interrupt__", interrupts)
                                context.record("interrupts", interrupts)
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            self._handle_failure(exc, context)
            raise WorkflowExecutionError(str(exc)) from exc

        self._handle_success(context)
        return WorkflowResult(data=data, context=context, interrupts=interrupts)

    # ------------------------------------------------------------------
    # Hooks for sub-classes
    # ------------------------------------------------------------------
    def _build_graph(
        self, context: ExecutionContext, *, async_mode: bool = False
    ) -> StateGraph:
        """Return the LangGraph ``StateGraph`` definition for the workflow."""

        raise NotImplementedError

    def _initial_state(
        self,
        context: ExecutionContext,
        state: Any,
        *args: Any,
        **kwargs: Any,
    ) -> MutableMapping[str, Any]:
        """Return the initial state for the LangGraph workflow."""

        return {"args": args, "kwargs": kwargs, **state}

    def _graph_config(
        self,
        context: ExecutionContext,
        *,
        thread_id: Optional[str] = None,
    ) -> RunnableConfig:
        """Return the configuration passed to LangGraph invoke."""

        config: RunnableConfig = {}
        try:
            from langchain_core.runnables.config import (  # type: ignore
                ensure_config as _lc_ensure_config,
            )

            inherited_config = _lc_ensure_config()
            if isinstance(inherited_config, dict):
                config = dict(inherited_config)
        except Exception:
            pass

        if context.debug:
            config.setdefault("tags", []).append("debug")
        if context.metadata:
            config.setdefault("metadata", {}).update(dict(context.metadata))

        resolved_thread_id = thread_id or context.metadata.get("thread_id")
        if resolved_thread_id is None:
            resolved_thread_id = str(uuid4())
        context.thread_id = resolved_thread_id
        config.setdefault("metadata", {}).setdefault("thread_id", resolved_thread_id)

        # Ensure a deterministic Langfuse trace id is available early so both
        # callbacks and scoring can reference the same trace.
        if "langfuse_trace_id" not in context.metadata:
            trace_id = _create_langfuse_trace_id(str(resolved_thread_id))
            if trace_id:
                context.metadata["langfuse_trace_id"] = trace_id
        if context.metadata.get("langfuse_trace_id"):
            config.setdefault("metadata", {}).setdefault(
                "langfuse_trace_id", context.metadata["langfuse_trace_id"]
            )

        config.setdefault("configurable", {})["thread_id"] = resolved_thread_id
        handler = _create_langfuse_handler(trace_id=context.metadata.get("langfuse_trace_id"))
        if handler is not None and not _graph_config_inherits_langfuse_handler(config):
            try:
                from langchain_core.runnables.config import (  # type: ignore
                    merge_configs as _lc_merge_configs,
                )

                config = _lc_merge_configs(config, {"callbacks": [handler]})
            except Exception:
                callbacks = config.get("callbacks")
                if callbacks is None:
                    config["callbacks"] = [handler]
                elif isinstance(callbacks, list):
                    callbacks.append(handler)
                else:
                    config["callbacks"] = [callbacks, handler]
        
        return config

    def _post_process(self, raw_result: Any, context: ExecutionContext) -> Any:
        """Transform the raw LangGraph result before returning to the caller."""
        return raw_result

    # ------------------------------------------------------------------
    # Lifecycle utilities
    # ------------------------------------------------------------------
    def _handle_success(self, context: ExecutionContext) -> None:
        if context.monitor:
            log_output(f"Workflow '{self.name}' completed successfully.")

    def _handle_failure(self, exc: Exception, context: ExecutionContext) -> None:
        if context.monitor:
            log_output(
                f"Workflow '{self.name}' failed with error: {exc.__class__.__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    async def _call_callable_async(
        fn: Callable[..., Any],
        value: Any,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Async-safe callable runner.

        LangGraph's async execution will await this helper. If ``fn`` is a sync
        callable (e.g. many tool wrappers or local workflow steps), calling it
        directly would run on the event loop thread and can block streaming
        callbacks/events until it completes.

        To avoid starving the loop, we offload sync callables to a short-lived
        worker thread while preserving contextvars for tracing/callbacks.
        """
        import inspect
        import contextvars
        import functools
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        signature = inspect.signature(fn)
        bound_kwargs: Dict[str, Any] = dict(kwargs)

        if "context" in signature.parameters:
            bound_kwargs.setdefault("context", context)

        parameters = signature.parameters

        def _invoke() -> Any:
            if parameters:
                first_param = next(iter(parameters.values()))
                if first_param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    if value is not None or first_param.default is inspect.Parameter.empty:
                        return fn(value, **bound_kwargs)
                    return fn(**bound_kwargs)
                return fn(**bound_kwargs)
            return fn(**bound_kwargs)

        # If it is already async (or returns an awaitable), keep it on-loop.
        if inspect.iscoroutinefunction(fn):
            result = _invoke()
            return await result

        # Offload sync callables to a thread to keep streaming responsive.
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        run_in_ctx = functools.partial(ctx.run, _invoke)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="andromeda") as executor:
            result = await loop.run_in_executor(executor, run_in_ctx)

        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _call_callable(
        fn: Callable[..., Any],
        value: Any,
        context: ExecutionContext,
        **kwargs: Any,
    ) -> Any:
        """Invoke a callable, attempting to match its signature.

        The helper supports the following call patterns (detected via
        argument inspection):

        * ``fn(value, context=context, **kwargs)``
        * ``fn(value, **kwargs)``
        * ``fn(context=context, **kwargs)``
        * ``fn(**kwargs)``
        * ``fn()``
        """

        import inspect

        signature = inspect.signature(fn)
        bound_kwargs: Dict[str, Any] = dict(kwargs)

        if "context" in signature.parameters:
            bound_kwargs.setdefault("context", context)

        parameters = signature.parameters

        if parameters:
            first_param = next(iter(parameters.values()))
            if first_param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if value is not None or first_param.default is inspect.Parameter.empty:
                    return fn(value, **bound_kwargs)

        return fn(**bound_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_compiled(
        self, context: ExecutionContext, *, async_mode: bool = False, **kwargs: Any
    ) -> CompiledStateGraph:
        cache = self._compiled_graph_async if async_mode else self._compiled_graph

        if context.debug or cache is None:
            graph_builder = self._build_graph(context, async_mode=async_mode)

            if isinstance(graph_builder, CompiledStateGraph):
                compiled = graph_builder
            else:
                checkpointer = self._checkpointer_provider.resolve_sync()
                compiled = graph_builder.compile(
                    checkpointer=checkpointer, **kwargs
                )

            if context.debug:
                return compiled

            if async_mode:
                self._compiled_graph_async = compiled
            else:
                self._compiled_graph = compiled
            cache = compiled

        return cache

    async def _ensure_compiled_async(
        self, context: ExecutionContext, **kwargs: Any
    ) -> CompiledStateGraph:
        cache = self._compiled_graph_async

        if context.debug or cache is None:
            graph_builder = self._build_graph(context, async_mode=True)

            if isinstance(graph_builder, CompiledStateGraph):
                compiled = graph_builder
            else:
                checkpointer = await self._checkpointer_provider.resolve_async()
                compiled = graph_builder.compile(
                    checkpointer=checkpointer, **kwargs
                )

            if context.debug:
                return compiled

            self._compiled_graph_async = compiled
            cache = compiled

        return cache

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------
    def stream(
        self,
        *args: Any,
        state: Optional[Any] = None,
        resume: Any = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        execution_context: Optional[ExecutionContext] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        context = self._prepare_execution_context(
            execution_context=execution_context,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            async_mode=False,
        )

        graph = self._ensure_compiled(context, async_mode=False, **kwargs)
        graph_config = self._graph_config(context, thread_id=thread_id)

        def _iterator() -> Iterator[Any]:
            completed = False
            errored = False
            last_chunk: Any = None
            with _langfuse_trace_context(context, name=f"workflow:{self.name}") as span:
                try:
                    if resume is not None:
                        if isinstance(resume, Command):
                            resume_command = resume
                        elif isinstance(resume, dict):
                            resume_command = Command(
                                goto=resume.get("goto"), update=resume.get("update")
                            )
                        else:
                            resume_command = Command(resume=resume)
                        _update_langfuse_trace(
                            span, input_state={"resume": resume_command}, attributes=context.metadata.get("langfuse_attributes", {})
                        )
                        iterator = graph.stream(
                            resume_command,
                            config=graph_config,
                            stream_mode=stream_mode,
                            **kwargs,
                        )
                    else:
                        initial_state = self._initial_state(context, state, *args, **kwargs)
                        _update_langfuse_trace(span, input_state=initial_state, attributes=context.metadata.get("langfuse_attributes", {}))
                        iterator = graph.stream(
                            initial_state,
                            config=graph_config,
                            stream_mode=stream_mode,
                            **kwargs,
                        )

                    for chunk in iterator:
                        last_chunk = chunk
                        self._inspect_stream_chunk(context, chunk)
                        yield chunk
                    completed = True
                except Exception as exc:  # noqa: BLE001
                    errored = True
                    _update_langfuse_trace(span, output_state={"error": str(exc)}, attributes=context.metadata.get("langfuse_attributes", {}))
                    self._handle_failure(exc, context)
                    raise
                finally:
                    if not errored and last_chunk is not None:
                        _update_langfuse_trace(span, output_state=last_chunk, attributes=context.metadata.get("langfuse_attributes", {}))

            if completed:
                self._handle_success(context)

        return _iterator()

    async def astream(
        self,
        *args: Any,
        state: Optional[Any] = None,
        resume: Any = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        execution_context: Optional[ExecutionContext] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        context = self._prepare_execution_context(
            execution_context=execution_context,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            async_mode=True,
        )

        graph = await self._ensure_compiled_async(context, **kwargs)
        graph_config = self._graph_config(context, thread_id=thread_id)
        import asyncio

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)

        async def _producer() -> None:
            errored = False
            last_chunk: Any = None
            async with _alangfuse_trace_context(
                context, name=f"workflow:{self.name}"
            ) as span:
                try:
                    if resume is not None:
                        if isinstance(resume, Command):
                            inp = resume
                        elif isinstance(resume, dict):
                            inp = Command(goto=resume.get("goto"), update=resume.get("update"))
                        else:
                            inp = Command(resume=resume)
                        _update_langfuse_trace(span, input_state={"resume": inp}, attributes=context.metadata.get("langfuse_attributes", {}))
                    else:
                        inp = self._initial_state(context, state, *args, **kwargs)
                        _update_langfuse_trace(span, input_state=inp, attributes=context.metadata.get("langfuse_attributes", {}))

                    ait = (
                        graph.astream_events(
                            inp,
                            config=graph_config,
                            stream_mode=stream_mode,
                            **kwargs,
                        )
                        if stream_mode == "events"
                        else graph.astream(
                            inp,
                            config=graph_config,
                            stream_mode=stream_mode,
                            **kwargs,
                        )
                    )

                    async for chunk in ait:
                        last_chunk = chunk
                        await queue.put(chunk)
                except (asyncio.CancelledError, GeneratorExit):
                    raise
                except Exception as exc:  # noqa: BLE001
                    errored = True
                    _update_langfuse_trace(span, output_state={"error": str(exc)}, attributes=context.metadata.get("langfuse_attributes", {}))
                    raise
                finally:
                    if not errored and last_chunk is not None:
                        _update_langfuse_trace(span, output_state=last_chunk, attributes=context.metadata.get("langfuse_attributes", {}))

                    # Best-effort flush to ship spans promptly (no-op if unsupported).
                    try:
                        from langfuse import get_client  # type: ignore

                        client = get_client()
                        flush = getattr(client, "flush", None)
                        if callable(flush):
                            flush()
                    except Exception:
                        pass

        producer_task = asyncio.create_task(_producer())
        completed = False

        try:
            while True:
                while True:
                    try:
                        chunk = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    self._inspect_stream_chunk(context, chunk)
                    yield chunk

                if producer_task.done():
                    exc = producer_task.exception()
                    if exc is not None:
                        self._handle_failure(exc, context)
                        raise exc
                    completed = True
                    break

                get_task = asyncio.create_task(queue.get())
                done, _pending = await asyncio.wait(
                    {get_task, producer_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_task in done:
                    chunk = get_task.result()
                    self._inspect_stream_chunk(context, chunk)
                    yield chunk
                else:
                    get_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await get_task
        finally:
            if not producer_task.done():
                producer_task.cancel()
            with contextlib.suppress(Exception):
                await producer_task

            if completed:
                self._handle_success(context)

    def _inspect_stream_chunk(self, context: ExecutionContext, chunk: Any) -> None:
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            context.record("interrupts", chunk["__interrupt__"])

    def _stream_wrapper(
        self, iterator: Iterator[Any], context: ExecutionContext
    ) -> Iterator[Any]:
        for chunk in iterator:
            self._inspect_stream_chunk(context, chunk)
            yield chunk

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def set_checkpointer(self, checkpointer: Any) -> None:
        """Update the checkpointer used when compiling LangGraph graphs."""

        self._checkpointer_provider.set(checkpointer)
        self._compiled_graph = None
        self._compiled_graph_async = None

    def resume(
        self,
        resume_value: Any,
        *,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Resume a workflow that previously paused via ``interrupt``."""

        return self.run(resume=resume_value, thread_id=thread_id, **kwargs)

    async def aresume(
        self,
        resume_value: Any,
        *,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Async counterpart to :meth:`resume`."""

        return await self.arun(resume=resume_value, thread_id=thread_id, **kwargs)

    def get_state(self, thread_id: str) -> Optional[Any]:
        """Fetch persisted state for a given thread from the underlying graph."""

        compiled = self._ensure_compiled(ExecutionContext(name=self.name))
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return compiled.get_state(config)
        except Exception:  # noqa: BLE001
            return None
