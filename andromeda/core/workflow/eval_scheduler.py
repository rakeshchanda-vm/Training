"""Internal scheduling for workflow evaluators.

Goals:
- Never block the main workflow execution.
- Provide backpressure so evaluation does not spawn unbounded threads/work.
- Best-effort behavior: drop work when overloaded and log appropriately.

This module keeps configuration internal by default but allows explicit overrides
per node via ``WorkflowBuilder.with_evaluators(..., scheduler=SchedulerConfig(...))``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import multiprocessing
import threading
import time
from typing import Any, Callable, Optional

from andromeda.utils.logger import log_warning


def get_safe_process_count() -> int:
    """Return a safe number of processes to use.

    Algorithm:
    - Try os.sched_getaffinity(0) -> available_cores = len(...). If AttributeError,
      fallback to multiprocessing.cpu_count(). If that fails, fallback to os.cpu_count() or 1.
    - Compute safe_cores: if available_cores > 4: available_cores - 2; elif available_cores > 1: available_cores - 1; else 1.
    - Return safe_cores.
    """
    # Determine available cores with best-effort fallbacks.
    try:
        try:
            avail = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except AttributeError:
            # os.sched_getaffinity may not exist on some platforms (e.g., Windows).
            avail = multiprocessing.cpu_count()
        if not isinstance(avail, int) or avail < 1:
            raise ValueError("invalid cpu count")
        available_cores = avail
    except Exception:
        # Fallback to os.cpu_count() or 1
        available_cores = os.cpu_count() or 1
        try:
            available_cores = int(available_cores)
        except Exception:
            available_cores = 1

    # Compute safe cores
    if available_cores > 4:
        safe_cores = available_cores - 2
    elif available_cores > 1:
        safe_cores = available_cores - 1
    else:
        safe_cores = 1

    return safe_cores


def _default_max_workers() -> int:
    # Use a safe process count so we don't starve the main workflow threads.
    return max(1, get_safe_process_count())


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    max_workers: int
    max_pending: int


class EvaluationScheduler:
    """Bounded, best-effort scheduler for background evaluation work."""

    def __init__(self, config: SchedulerConfig) -> None:
        if config.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if config.max_pending < 0:
            raise ValueError("max_pending must be >= 0")

        self._config = config
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix="andromeda-eval"
        )
        # Capacity includes in-flight workers + queued work.
        self._capacity = threading.BoundedSemaphore(config.max_workers + config.max_pending)
        self._drop_lock = threading.Lock()
        self._last_drop_log = 0.0

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def submit(self, fn: Callable[[], Any], *, label: str) -> bool:
        """Submit work; returns False if dropped due to capacity."""

        acquired = self._capacity.acquire(blocking=False)
        if not acquired:
            self._log_drop(label)
            return False

        def _wrapped() -> None:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                try:
                    log_warning(
                        "Background evaluation task failed.\n"
                        f"- label: {label}\n"
                        f"- error: {exc.__class__.__name__}: {exc}"
                    )
                except Exception:
                    pass
            finally:
                try:
                    self._capacity.release()
                except Exception:
                    pass

        try:
            self._executor.submit(_wrapped)
            return True
        except Exception:
            try:
                self._capacity.release()
            except Exception:
                pass
            self._log_drop(label)
            return False

    def _log_drop(self, label: str) -> None:
        now = time.time()
        with self._drop_lock:
            # Rate-limit to avoid log spam under sustained overload.
            if now - self._last_drop_log < 5.0:
                return
            self._last_drop_log = now
        try:
            log_warning(
                "Dropping background evaluation task due to scheduler capacity.\n"
                f"- label: {label}\n"
                f"- max_workers: {self._config.max_workers}\n"
                f"- max_pending: {self._config.max_pending}"
            )
        except Exception:
            pass


_scheduler_by_config: dict[tuple[int, int], EvaluationScheduler] = {}
_scheduler_lock = threading.Lock()


def get_evaluation_scheduler(config: Optional[SchedulerConfig] = None) -> EvaluationScheduler:
    """Return a scheduler instance.

    If ``config`` is not provided, a default bounded scheduler is returned.
    If ``config`` is provided, schedulers are cached by (max_workers, max_pending).
    """

    cfg = config or SchedulerConfig(max_workers=_default_max_workers(), max_pending=64)
    key = (cfg.max_workers, cfg.max_pending)
    with _scheduler_lock:
        existing = _scheduler_by_config.get(key)
        if existing is not None:
            return existing
        created = EvaluationScheduler(cfg)
        _scheduler_by_config[key] = created
        return created
