"""Workflow abstractions for Andromeda.

This subpackage exposes a lightweight workflow interface.

Both authoring styles derive from ``WorkflowBase`` which
contains the shared execution engine and basic lifecycle hooks.
"""

from .base import ExecutionContext, WorkflowBase, WorkflowExecutionError, WorkflowResult, Command, RunnableConfig
from .builder import WorkflowBuilder, WorkflowExpression, conditional, parallel, task, interrupt
from .eval_scheduler import SchedulerConfig
from .evaluation import EvaluationPreset, LangfuseEvaluator, LangfuseScore, LangfuseScoreDataType
from .evaluation import EvaluatorRuntimeConfig
from .evaluators import correctness, hallucination, relevance, tool_usage

__all__ = [
    "ExecutionContext",
    "WorkflowBase",
    "WorkflowExecutionError",
    "WorkflowResult",
    "WorkflowBuilder",
    "WorkflowExpression",
    "conditional",
    "parallel",
    "task",
    "Command",
    "RunnableConfig",
    "interrupt",
    "SchedulerConfig",
    "EvaluationPreset",
    "LangfuseEvaluator",
    "LangfuseScore",
    "LangfuseScoreDataType",
    "EvaluatorRuntimeConfig",
    "correctness",
    "hallucination",
    "relevance",
    "tool_usage",
]
