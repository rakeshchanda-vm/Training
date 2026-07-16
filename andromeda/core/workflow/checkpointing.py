"""LangGraph checkpointer resolution helpers."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver

from andromeda.config.config import CheckpointerConfig


class CheckpointerDependencyError(RuntimeError):
    """Raised when an optional checkpointer backend dependency is missing."""


def _coerce_config(spec: Any) -> Optional[CheckpointerConfig]:
    if isinstance(spec, CheckpointerConfig):
        return spec
    if isinstance(spec, (str, dict)):
        return CheckpointerConfig.model_validate(spec)
    return None


def _missing_postgres_dependency() -> CheckpointerDependencyError:
    return CheckpointerDependencyError(
        "Postgres checkpointer support requires the optional dependency extra "
        "'andromeda[checkpointer-postgres]'."
    )


def _maybe_setup(saver: Any, enabled: bool) -> None:
    if not enabled:
        return
    setup = getattr(saver, "setup", None)
    if callable(setup):
        setup()


async def _maybe_asetup(saver: Any, enabled: bool) -> None:
    if not enabled:
        return
    setup = getattr(saver, "setup", None)
    if callable(setup):
        result = setup()
        if inspect.isawaitable(result):
            await result


class CheckpointerProvider:
    """Lazily materialize sync and async checkpointers from a config spec."""

    def __init__(self, spec: Any = None) -> None:
        self._spec = CheckpointerConfig() if spec is None else spec
        self._lock = threading.RLock()
        self._sync_ready = False
        self._async_ready = False
        self._sync_saver: Any = None
        self._async_saver: Any = None
        self._sync_cm: Any = None
        self._async_cm: Any = None
        self._async_build_lock: Optional[asyncio.Lock] = None

    @property
    def spec(self) -> Any:
        return self._spec

    def set(self, spec: Any = None) -> None:
        self.close()
        with self._lock:
            self._spec = CheckpointerConfig() if spec is None else spec
            self._sync_ready = False
            self._async_ready = False
            self._sync_saver = None
            self._async_saver = None
            self._sync_cm = None
            self._async_cm = None
            self._async_build_lock = None

    def resolve_sync(self) -> Any:
        with self._lock:
            if not self._sync_ready:
                self._sync_saver = self._build_sync()
                self._sync_ready = True
            return self._sync_saver

    async def resolve_async(self) -> Any:
        with self._lock:
            if self._async_ready:
                return self._async_saver
            if self._async_build_lock is None:
                self._async_build_lock = asyncio.Lock()
            build_lock = self._async_build_lock

        async with build_lock:
            with self._lock:
                if self._async_ready:
                    return self._async_saver
            saver = await self._build_async()
            with self._lock:
                self._async_saver = saver
                self._async_ready = True
                return self._async_saver

    def close(self) -> None:
        with self._lock:
            sync_cm = self._sync_cm
            sync_saver = self._sync_saver
            async_cm = self._async_cm
            async_saver = self._async_saver
            self._sync_cm = None
            self._sync_saver = None
            self._sync_ready = False
            self._async_cm = None
            self._async_saver = None
            self._async_ready = False

        if sync_cm is not None:
            with contextlib.suppress(Exception):
                sync_cm.__exit__(None, None, None)
        elif sync_saver is not None and sync_saver is not async_saver:
            close = getattr(sync_saver, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()

        if async_cm is not None or async_saver is not None:
            async def _close_async_resource() -> None:
                if async_cm is not None:
                    await async_cm.__aexit__(None, None, None)
                    return
                close = getattr(async_saver, "aclose", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result

            with contextlib.suppress(Exception):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(_close_async_resource())
                else:
                    loop.create_task(_close_async_resource())

    async def aclose(self) -> None:
        with self._lock:
            async_cm = self._async_cm
            async_saver = self._async_saver
            self._async_cm = None
            self._async_saver = None
            self._async_ready = False

        if async_cm is not None:
            with contextlib.suppress(Exception):
                await async_cm.__aexit__(None, None, None)
        elif async_saver is not None:
            close = getattr(async_saver, "aclose", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    result = close()
                    if inspect.isawaitable(result):
                        await result

    def _build_sync(self) -> Any:
        config = _coerce_config(self._spec)
        if config is None:
            return self._spec

        if config.backend == "none":
            return None
        if config.backend == "in-memory":
            if self._async_ready and isinstance(self._async_saver, InMemorySaver):
                return self._async_saver
            return InMemorySaver()

        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise _missing_postgres_dependency() from exc

        candidate = PostgresSaver.from_conn_string(config.connection_string)
        if hasattr(candidate, "__enter__"):
            self._sync_cm = candidate
            saver = candidate.__enter__()
        else:
            saver = candidate
        _maybe_setup(saver, config.setup)
        return saver

    async def _build_async(self) -> Any:
        config = _coerce_config(self._spec)
        if config is None:
            return self._spec

        if config.backend == "none":
            return None
        if config.backend == "in-memory":
            with self._lock:
                if self._sync_ready and isinstance(self._sync_saver, InMemorySaver):
                    return self._sync_saver
            return InMemorySaver()

        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise _missing_postgres_dependency() from exc

        candidate = AsyncPostgresSaver.from_conn_string(config.connection_string)
        if hasattr(candidate, "__aenter__"):
            self._async_cm = candidate
            saver = await candidate.__aenter__()
        else:
            saver = candidate
        await _maybe_asetup(saver, config.setup)
        return saver
