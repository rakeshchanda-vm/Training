"""Declarative workflow builder implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel
from .base import ExecutionContext, WorkflowBase
from .eval_scheduler import SchedulerConfig, get_evaluation_scheduler
from .evaluation import (
    LangfuseEvaluator,
    EvaluatorRuntimeConfig,
    ingest_langfuse_scores,
)


def _expr_init_state(value: Any) -> MutableMapping[str, Any]:
    return {"__internal__": {"current": value}}


def _expr_get_current(state: MutableMapping[str, Any]) -> Any:
    return state.get("__internal__", {}).get("current")


def _expr_set_current(
    state: MutableMapping[str, Any], value: Any
) -> MutableMapping[str, Any]:
    new_state = dict(state)
    internal = dict(new_state.get("__internal__", {}))
    internal["current"] = value
    new_state["__internal__"] = internal
    return new_state


@dataclass(slots=True)
class StepDescriptor:
    name: str
    kind: str = "task"
    func: Optional[Callable[..., Any]] = None
    options: Dict[str, Any] = field(default_factory=dict)
    children: Sequence[Tuple[str, Callable[..., Any]]] = field(default_factory=tuple)


class _RouteBinding:
    """Helper used for chaining routing DSL (e.g. ``if_fails().goto("x")``)."""

    def __init__(
        self, builder: "WorkflowBuilder", step: StepDescriptor, route_key: str
    ):
        self._builder = builder
        self._step = step
        self._route_key = route_key

    def goto(self, destination: str) -> "WorkflowBuilder":
        """
        Go to the specified step.

        Args:
            destination (str): The name of the step to go to.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._step.options.setdefault("routes", {})[self._route_key] = destination
        return self._builder

    def continue_to(self, destination: str) -> "WorkflowBuilder":
        """
        Continue to the specified step.

        Args:
            destination (str): The name of the step to continue to.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        return self.goto(destination)


class _TimeoutBinding:
    """Helper used for ``on_timeout().send_reminder()`` style chaining."""

    def __init__(self, builder: "WorkflowBuilder", step: StepDescriptor):
        self._builder = builder
        self._step = step

    def send_reminder(self, message: str | None = None) -> "WorkflowBuilder":
        """
        Send a reminder to the user.

        Args:
            message (str, optional): The message to send.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._step.options.setdefault("timeout_actions", []).append(
            {"action": "send_reminder", "message": message}
        )
        return self._builder


@dataclass(slots=True)
class WorkflowExpression:
    parts: Tuple[
        Callable[
            [MutableMapping[str, Any], ExecutionContext], MutableMapping[str, Any]
        ],
        ...,
    ]

    def __rshift__(
        self, other: "WorkflowExpression | Callable[..., Any]"
    ) -> "WorkflowExpression":
        other_expr = (
            other
            if isinstance(other, WorkflowExpression)
            else WorkflowExpression.wrap(other)
        )
        return WorkflowExpression(parts=self.parts + other_expr.parts)

    def __rrshift__(
        self, other: "WorkflowExpression | Callable[..., Any]"
    ) -> "WorkflowExpression":
        """
        Support ``callable >> WorkflowExpression`` by letting the right-hand
        side handle the operator when the left-hand side is a plain function.
        """
        left_expr = (
            other
            if isinstance(other, WorkflowExpression)
            else WorkflowExpression.wrap(other)
        )
        return WorkflowExpression(parts=left_expr.parts + self.parts)

    @staticmethod
    def wrap(callable_obj: Callable[..., Any]) -> "WorkflowExpression":
        """
        Wrap a callable object in a workflow expression.

        Args:
            callable_obj (Callable[..., Any]): The callable object to wrap.
        Returns:
            WorkflowExpression: The workflow expression.
        """

        def node(
            state: MutableMapping[str, Any], context: ExecutionContext
        ) -> MutableMapping[str, Any]:
            current_value = _expr_get_current(state)
            result = WorkflowBase._call_callable(
                callable_obj, current_value, context=context
            )
            return _expr_set_current(state, result)

        return WorkflowExpression(parts=(node,))


def conditional(
    true_branch: "WorkflowExpression | Callable[..., Any]",
    false_branch: "WorkflowExpression | Callable[..., Any]",
    *,
    condition: Callable[[Any], bool],
) -> WorkflowExpression:
    """
    Create a conditional task in the workflow execution.

    Args:
        true_branch (WorkflowExpression | Callable[..., Any]): The true branch.
        false_branch (WorkflowExpression | Callable[..., Any]): The false branch.
        condition (Callable[[Any], bool]): The condition to check.
    Returns:
        WorkflowExpression: The conditional task.
    """
    true_callable = _as_value_callable(true_branch)
    false_callable = _as_value_callable(false_branch)

    def node(
        state: MutableMapping[str, Any], context: ExecutionContext
    ) -> MutableMapping[str, Any]:
        current_value = _expr_get_current(state)
        chosen_callable = true_callable if condition(current_value) else false_callable
        result = chosen_callable(current_value, context)
        return _expr_set_current(state, result)

    return WorkflowExpression(parts=(node,))


def parallel(
    *callables: "WorkflowExpression | Callable[..., Any]",
) -> WorkflowExpression:
    """
    Create a parallel task in the workflow execution.

    Args:
        callables (Sequence[WorkflowExpression | Callable[..., Any]]): The tasks to parallelize.
    Returns:
        WorkflowExpression: The workflow expression representing the parallel step.

    This function annotates the expression so that :meth:`WorkflowBuilder._build_expression_graph`
    can expand it into multiple LangGraph nodes with parallel edges. When the
    expression is executed directly (e.g. via ``_as_value_callable``), this node
    falls back to sequential execution of the branches.
    """
    value_callables = [_as_value_callable(fn) for fn in callables]

    def node(
        state: MutableMapping[str, Any], context: ExecutionContext
    ) -> MutableMapping[str, Any]:
        # Sequential fallback when this expression is executed outside of a
        # LangGraph compiled by WorkflowBuilder.
        current_value = _expr_get_current(state)
        results = [callable_fn(current_value, context) for callable_fn in value_callables]
        return _expr_set_current(state, results)

    # Mark this node so that _build_expression_graph can expand it into a
    # fan-out + join pattern using LangGraph edges for true parallelism.
    node.__expr_parallel_value_callables__ = tuple(value_callables)
    node.__expr_parallel_raw_callables__ = tuple(callables)

    return WorkflowExpression(parts=(node,))


def _as_value_callable(
    fn: "WorkflowExpression | Callable[..., Any]",
) -> Callable[[Any, ExecutionContext], Any]:
    if isinstance(fn, WorkflowExpression):

        def value_callable(value: Any, context: ExecutionContext) -> Any:
            state: MutableMapping[str, Any] = _expr_init_state(value)
            current_state = state
            for part in fn.parts:
                current_state = part(current_state, context)
            return _expr_get_current(current_state)

        return value_callable

    def value_callable(value: Any, context: ExecutionContext) -> Any:
        return WorkflowBase._call_callable(fn, value, context)

    return value_callable


def task(
    function: Optional[Callable[..., Any]] = None,
) -> "WorkflowExpression | Callable[[Callable[..., Any]], WorkflowExpression]":
    """
    Create a task in the workflow execution.

    Args:
        function (Callable[..., Any], optional): The function to run.
    Returns:
        WorkflowExpression | Callable[[Callable[..., Any]], WorkflowExpression]: The task.
    """

    def _wrap(fn: Callable[..., Any]) -> WorkflowExpression:
        return WorkflowExpression.wrap(fn)

    if function is None:
        return _wrap

    return _wrap(function)


class WorkflowBuilder(WorkflowBase):
    """Fluent builder for declarative workflows."""

    def __init__(
        self,
        name: Optional[str] = None,
        state_schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        *,
        expression: Optional["WorkflowExpression"] = None,
        checkpointer: Optional[Any] = None,
    ) -> None:
        super().__init__(name=name or "WorkflowBuilder", checkpointer=checkpointer)
        self._steps: List[StepDescriptor] = []
        self._current: Optional[StepDescriptor] = None
        self._state_schema = state_schema
        self._expression = expression
        self._mode = "expression" if expression is not None else "declarative"

    @classmethod
    def from_expression(
        cls,
        expression: "WorkflowExpression",
        name: Optional[str] = None,
        checkpointer: Optional[Any] = None,
    ) -> "WorkflowBuilder":
        """
        Create a workflow builder from a workflow expression.

        Args:
            expression (WorkflowExpression): The workflow expression.
            name (str, optional): The name of the workflow builder.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        return cls(name=name, expression=expression, checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # DSL helpers
    # ------------------------------------------------------------------
    def start(self, name: str) -> "WorkflowBuilder":
        """
        Create a start step in the workflow execution.

        Args:
            name (str): The name of the start step.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("start")
        return self._add_step(name, kind="task")

    def run(self, func: Callable[..., Any]) -> "WorkflowBuilder":
        """
        Create a run step in the workflow execution.

        Args:
            func (Callable[..., Any]): The function to run.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("run")
        self._ensure_current().func = func
        return self

    def with_retry(
        self, attempts: int = 1, backoff: str = "fixed"
    ) -> "WorkflowBuilder":
        """
        Create a retry step in the workflow execution.

        Args:
            attempts (int, default=1): The number of attempts to retry.
            backoff (str, default="fixed"): The backoff strategy to use.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("with_retry")
        self._ensure_current().options["retry"] = {
            "attempts": attempts,
            "backoff": backoff,
        }
        return self

    def timeout(self, duration: Any) -> "WorkflowBuilder":
        """
        Create a timeout step in the workflow execution.

        Args:
            duration (Any): The duration of the timeout.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("timeout")
        self._ensure_current().options["timeout"] = duration
        return self

    def on_timeout(self) -> _TimeoutBinding:
        """
        Create a timeout step in the workflow execution.

        Returns:
            _TimeoutBinding: The timeout binding.
        """
        self._assert_declarative_mode("on_timeout")
        return _TimeoutBinding(self, self._ensure_current())

    def then(self, name: str) -> "WorkflowBuilder":
        """
        Create a then step in the workflow execution.

        Args:
            name (str): The name of the then step.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("then")
        return self._add_step(name, kind="task")

    def branch(self, name: str) -> "WorkflowBuilder":
        """
        Create a branch in the workflow execution.

        Args:
            name (str): The name of the branch.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("branch")
        return self._add_step(name, kind="branch")

    def parallel(
        self,
        tasks: Sequence[
            Union[
                Tuple[str, Callable[..., Any]],
                Tuple[str, Callable[..., Any], Sequence[LangfuseEvaluator]],
                Tuple[
                    str,
                    Callable[..., Any],
                    Sequence[LangfuseEvaluator],
                    EvaluatorRuntimeConfig,
                ],
            ]
        ],
    ) -> "WorkflowBuilder":
        """
        Create a parallel task in the workflow execution.

        Args:
            tasks: The tasks to parallelize.
                Each entry is either:
                - (name, callable)
                - (name, callable, evaluators) to attach Langfuse evaluators to that child node.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("parallel")
        step = self._ensure_current()
        if step.kind != "branch":
            raise ValueError("parallel() can only be used after branch().")
        children: list[tuple[str, Callable[..., Any]]] = []
        child_evaluators: dict[str, Sequence[LangfuseEvaluator]] = {}
        child_runtime: dict[str, EvaluatorRuntimeConfig] = {}
        for item in tasks:
            if len(item) == 2:
                child_name, child_callable = item
                children.append((child_name, child_callable))
            elif len(item) == 3:
                child_name, child_callable, evaluators = item
                children.append((child_name, child_callable))
                child_evaluators[child_name] = evaluators
            elif len(item) == 4:
                child_name, child_callable, evaluators, runtime = item
                children.append((child_name, child_callable))
                child_evaluators[child_name] = evaluators
                child_runtime[child_name] = runtime
            else:
                raise ValueError(
                    "parallel() tasks must be (name, fn), (name, fn, evaluators), or (name, fn, evaluators, runtime)."
                )
        step.children = tuple(children)
        if child_evaluators:
            step.options.setdefault("child_evaluators", {}).update(child_evaluators)
        if child_runtime:
            step.options.setdefault("child_evaluator_runtime", {}).update(child_runtime)
        return self

    def with_evaluators(
        self,
        evaluators: Sequence[LangfuseEvaluator],
        *,
        scheduler: Optional[SchedulerConfig | Mapping[str, Any]] = None,
        model: Any = None,
    ) -> "WorkflowBuilder":
        """Attach Langfuse evaluators to the current node.

        Evaluators are executed in the background (best-effort) and will not block
        the workflow's control flow.
        """

        self._assert_declarative_mode("with_evaluators")
        self._ensure_current().options["evaluators"] = list(evaluators)
        scheduler_cfg: Optional[SchedulerConfig]
        if isinstance(scheduler, SchedulerConfig) or scheduler is None:
            scheduler_cfg = scheduler
        elif isinstance(scheduler, Mapping):
            defaults = get_evaluation_scheduler().config
            scheduler_cfg = SchedulerConfig(
                max_workers=int(scheduler.get("max_workers", defaults.max_workers)),
                max_pending=int(scheduler.get("max_pending", defaults.max_pending)),
            )
        else:
            raise TypeError("scheduler must be a SchedulerConfig, mapping, or None.")
        self._ensure_current().options["evaluator_runtime"] = EvaluatorRuntimeConfig(
            scheduler=scheduler_cfg, model=model
        )
        return self

    def merge_results(self) -> "WorkflowBuilder":
        """
        Merge the results of the parallel tasks.

        Note:
            With the LangGraph-native parallel implementation, branch children
            already merge any mapping results back into the shared state. This
            method is kept for backwards compatibility and is currently a
            no-op hint; callers should read child keys (e.g. ``sum``, ``count``)
            directly from the workflow state.

        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("merge_results")
        self._ensure_current().options["merge_results"] = True
        return self

    def if_fails(self) -> _RouteBinding:
        self._assert_declarative_mode("if_fails")
        """
        If the workflow execution fails, route to the specified step.

        Returns:
            _RouteBinding: The route binding.
        """
        return _RouteBinding(self, self._ensure_current(), "failure")

    def if_succeeds(self) -> _RouteBinding:
        """
        If the workflow execution succeeds, route to the specified step.

        Returns:
            _RouteBinding: The route binding.
        """
        self._assert_declarative_mode("if_succeeds")
        return _RouteBinding(self, self._ensure_current(), "success")

    def checkpoint(self, name: str) -> "WorkflowBuilder":
        """
        Create a checkpoint in the workflow execution.

        Args:
            name (str): The name of the checkpoint.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("checkpoint")
        return self._add_step(name, kind="checkpoint")

    def pause_for_approval(self, message: str | None = None) -> "WorkflowBuilder":
        """
        Pause the workflow execution for approval.

        Args:
            message (str, optional): The message to display to the user.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("pause_for_approval")
        step = self._ensure_current()
        step.options["approval_message"] = message
        return self

    def finish(self, name: str) -> "WorkflowBuilder":
        """
        Finish the workflow execution.

        Args:
            name (str): The name of the step to finish.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("finish")
        return self._add_step(name, kind="task")

    def save_to(self, destination: str) -> "WorkflowBuilder":
        """
        Save the workflow execution to the specified destination.

        Args:
            destination (str): The destination to save the workflow execution.
        Returns:
            WorkflowBuilder: The workflow builder.
        """
        self._assert_declarative_mode("save_to")
        self._ensure_current().options["save_to"] = destination
        return self

    def execute(
        self,
        *,
        state: Optional[Dict[str, Any]] = None,
        resume: Optional[Union[bool, Dict[str, Any], Command]] = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Synchronous execution of the workflow.

        Args:
            state (Any, optional): The initial input to the workflow.
            resume (Any, optional): Resume token or state for continuing a previous run.
            debug (bool, default=False): Enable debug mode for verbose output and extra checks.
            monitor (bool, default=False): Enable monitoring/logging of workflow execution.
            metadata (Mapping[str, Any], optional): Additional metadata to attach to the workflow context.
            thread_id (str, optional): Unique identifier for the workflow execution thread.
        Returns:
            WorkflowResult: The result of the workflow execution.
        """
        execution = WorkflowBase.run(
            self,
            state=state,
            resume=resume,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            execution_context=execution_context,
            **kwargs,
        )
        return execution.data

    async def aexecute(
        self,
        *,
        state: Optional[Dict[str, Any]] = None,
        resume: Optional[Union[bool, Dict[str, Any], Command]] = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        execution_context: Optional[ExecutionContext] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Asynchronous execution of the workflow.

        Args:
            state (Any, optional): The initial input to the workflow.
            resume (Any, optional): Resume token or state for continuing a previous run.
            debug (bool, default=False): Enable debug mode for verbose output and extra checks.
            monitor (bool, default=False): Enable monitoring/logging of workflow execution.
            metadata (Mapping[str, Any], optional): Additional metadata to attach to the workflow context.
            thread_id (str, optional): Unique identifier for the workflow execution thread.
        Returns:
            WorkflowResult: The result of the workflow execution.
        """
        execution = await WorkflowBase.arun(
            self,
            state=state,
            resume=resume,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            execution_context=execution_context,
            **kwargs,
        )
        return execution.data

    def stream(
        self,
        *,
        state: Optional[Dict[str, Any]] = None,
        resume: Optional[Union[bool, Dict[str, Any], Command]] = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        **kwargs: Any,
    ) -> Iterator[Any]:
        for chunk in WorkflowBase.stream(
            self,
            state=state,
            resume=resume,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            stream_mode=stream_mode,
            **kwargs,
        ):
            yield chunk

    async def astream(
        self,
        *,
        state: Optional[Dict[str, Any]] = None,
        resume: Optional[Union[bool, Dict[str, Any], Command]] = None,
        debug: bool = False,
        monitor: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async for chunk in WorkflowBase.astream(
            self,
            state=state,
            resume=resume,
            debug=debug,
            monitor=monitor,
            metadata=metadata,
            thread_id=thread_id,
            stream_mode=stream_mode,
            **kwargs,
        ):
            yield chunk

    # ------------------------------------------------------------------
    # WorkflowBase hooks
    # ------------------------------------------------------------------
    def _build_graph(
        self, context: ExecutionContext, *, async_mode: bool = False
    ) -> StateGraph:
        if self._expression is not None:
            return self._build_expression_graph(context, async_mode=async_mode)

        # When no explicit state schema is provided, default to a dict that can
        # safely accept concurrent updates from parallel branches. We use an
        # Annotated root with ``operator.or_`` so that multiple writes in a
        # single step are merged instead of raising INVALID_CONCURRENT_GRAPH_UPDATE.
        state_schema = self._state_schema or Annotated[dict, operator.or_]
        graph = StateGraph(state_schema)

        if not self._steps:
            graph.add_edge(START, END)
            return graph

        # Map step name -> descriptor for quick lookup (for routing).
        step_lookup: Dict[str, StepDescriptor] = {step.name: step for step in self._steps}

        # ------------------------------------------------------------------
        # Node definitions
        # ------------------------------------------------------------------
        for step in self._steps:
            if step.kind == "task":
                node_callable = self._build_task_node(
                    step, context, async_mode=async_mode
                )
                graph.add_node(step.name, node_callable)
            elif step.kind == "branch":
                # For branches we only create nodes for the child tasks; the
                # branch name itself is purely structural/dev-reference and not a LangGraph
                # node.
                for child_name, child_callable in step.children:
                    child_node = self._build_branch_child_node(
                        step,
                        child_name,
                        child_callable,
                        context,
                        async_mode=async_mode,
                    )
                    graph.add_node(child_name, child_node)
            elif step.kind == "checkpoint":
                node_callable = self._build_checkpoint_node(step, context)
                graph.add_node(step.name, node_callable)
            else:
                raise ValueError(f"Unsupported step kind: {step.kind}")

        # ------------------------------------------------------------------
        # Wiring between steps
        # ------------------------------------------------------------------
        def _entry_nodes(step: StepDescriptor) -> List[str]:
            """Graph nodes that represent the 'start' of this logical step."""
            if step.kind == "branch":
                return [child_name for child_name, _ in step.children]
            return [step.name]

        def _exit_nodes(step: StepDescriptor) -> List[str]:
            """Graph nodes that represent the 'end' of this logical step."""
            # For now, entry and exit sets are the same.
            return _entry_nodes(step)

        def _route_destination_node(dest_name: Optional[str]) -> Optional[str]:
            """Map a logical destination step name to a concrete graph node."""
            if dest_name is None:
                return None
            target_step = step_lookup.get(dest_name)
            if target_step is None:
                return dest_name
            entries = _entry_nodes(target_step)
            return entries[0] if entries else None

        first_step = self._steps[0]
        for entry in _entry_nodes(first_step):
            graph.add_edge(START, entry)

        # Wire each step's exit nodes to the next step (or END), taking routes into account.
        for index, step in enumerate(self._steps):
            exit_nodes = _exit_nodes(step)
            next_step: Optional[StepDescriptor] = (
                self._steps[index + 1] if index + 1 < len(self._steps) else None
            )
            default_dest_node = (
                _route_destination_node(next_step.name) if next_step else None
            )

            routes: Dict[str, str] = step.options.get("routes", {})

            if routes:
                normalized_routes: Dict[str, Optional[str]] = {
                    key: _route_destination_node(dest) for key, dest in routes.items()
                }

                possible_destinations = set(
                    value for value in normalized_routes.values() if value is not None
                )
                if default_dest_node:
                    possible_destinations.add(default_dest_node)
                else:
                    possible_destinations.add(END)

                def _route_fn(
                    state: MutableMapping[str, Any],
                    *,
                    routes=normalized_routes,
                    default=default_dest_node,
                ):
                    route_key = state.get("__route__", "success")
                    destination = routes.get(route_key)
                    if destination is None:
                        return default if default is not None else END
                    return destination

                for from_node in exit_nodes:
                    graph.add_conditional_edges(
                        from_node, _route_fn, list(possible_destinations)
                    )
            else:
                # Simple sequential wiring: each exit node goes to all entry
                # nodes of the next step, or END if there is no next step.
                if next_step is None:
                    for from_node in exit_nodes:
                        graph.add_edge(from_node, END)
                else:
                    next_entries = _entry_nodes(next_step)
                    for from_node in exit_nodes:
                        for to_node in next_entries:
                            graph.add_edge(from_node, to_node)


        return graph

    def _build_expression_graph(
        self,
        context: ExecutionContext,
        *,
        async_mode: bool = False,
    ) -> StateGraph:
        state_schema = self._state_schema or Annotated[dict, operator.or_]
        graph = StateGraph(state_schema)
        parts: Sequence[
            Callable[
                [MutableMapping[str, Any], ExecutionContext], MutableMapping[str, Any]
            ]
        ] = (self._expression.parts if self._expression is not None else ())

        if not parts:
            graph.add_edge(START, END)
            return graph

        previous_node: Optional[str] = None

        async def _eval_expr_callable_async(
            fn: Callable[..., Any] | "WorkflowExpression",
            value: Any,
        ) -> Any:
            """Evaluate a WorkflowExpression or plain callable in async mode."""
            if isinstance(fn, WorkflowExpression):
                state: MutableMapping[str, Any] = _expr_init_state(value)
                current_state = state
                for part in fn.parts:
                    result = part(current_state, context)
                    if inspect.isawaitable(result):
                        result = await result
                    if isinstance(result, MutableMapping):
                        current = _expr_get_current(result)
                        if inspect.isawaitable(current):
                            current = await current
                            result = _expr_set_current(result, current)
                        current_state = result
                    else:
                        current_state = _expr_set_current(current_state, result)
                final_value = _expr_get_current(current_state)
                if inspect.isawaitable(final_value):
                    final_value = await final_value
                return final_value

            return await self._call_callable_async(
                fn, value, context=context  # type: ignore[arg-type]
            )

        index = 0
        while index < len(parts):
            part = parts[index]
            parallel_value_callables = getattr(
                part, "__expr_parallel_value_callables__", None
            )
            parallel_raw_callables = getattr(
                part, "__expr_parallel_raw_callables__", None
            )

            # ------------------------------------------------------------------
            # Regular expression step
            # ------------------------------------------------------------------
            if not parallel_value_callables or not parallel_raw_callables:
                # Internal expression nodes avoid double-underscore in names.
                node_name = f"expression_step_{index}"

                if async_mode:

                    async def node(
                        state: MutableMapping[str, Any], step=part
                    ) -> MutableMapping[str, Any]:
                        result = step(state, context)
                        if inspect.isawaitable(result):
                            result = await result
                        if isinstance(result, MutableMapping):
                            current = _expr_get_current(result)
                            if inspect.isawaitable(current):
                                awaited = await current
                                result = _expr_set_current(result, awaited)
                            return result
                        if self._mode == "expression":
                            return _expr_set_current(state, result)
                        return result

                else:

                    def node(
                        state: MutableMapping[str, Any], step=part
                    ) -> MutableMapping[str, Any]:
                        result = step(state, context)
                        if inspect.isawaitable(result):
                            raise RuntimeError(
                                "Expression step returned awaitable during synchronous execution."
                            )
                        if isinstance(result, MutableMapping):
                            current = _expr_get_current(result)
                            if inspect.isawaitable(current):
                                raise RuntimeError(
                                    "Expression step produced awaitable result in synchronous execution."
                                )
                            return result
                        if self._mode == "expression":
                            return _expr_set_current(state, result)
                        return result

                graph.add_node(node_name, node)

                if previous_node is None:
                    graph.add_edge(START, node_name)
                else:
                    graph.add_edge(previous_node, node_name)

                previous_node = node_name
                index += 1
                continue

            # ------------------------------------------------------------------
            # Parallel expression step – expand into branch nodes + join node
            # ------------------------------------------------------------------
            num_branches = len(parallel_value_callables)
            branch_keys = [
                f"expr_parallel_{index}_{branch_idx}"
                for branch_idx in range(num_branches)
            ]

            # Create branch nodes
            for branch_idx, (value_fn, raw_fn, branch_key) in enumerate(
                zip(
                    parallel_value_callables,
                    parallel_raw_callables,
                    branch_keys,
                )
            ):
                branch_node_name = f"expression_step_{index}_branch_{branch_idx}"

                if async_mode:

                    async def branch_node(
                        state: MutableMapping[str, Any],
                        raw_fn=raw_fn,
                        branch_key=branch_key,
                    ) -> MutableMapping[str, Any]:
                        current_value = _expr_get_current(state)
                        result = await _eval_expr_callable_async(raw_fn, current_value)
                        new_state = dict(state)
                        new_state[branch_key] = result
                        return new_state

                else:

                    def branch_node(
                        state: MutableMapping[str, Any],
                        value_fn=value_fn,
                        branch_key=branch_key,
                    ) -> MutableMapping[str, Any]:
                        current_value = _expr_get_current(state)
                        result = value_fn(current_value, context)
                        new_state = dict(state)
                        new_state[branch_key] = result
                        return new_state

                graph.add_node(branch_node_name, branch_node)

                if previous_node is None:
                    graph.add_edge(START, branch_node_name)
                else:
                    graph.add_edge(previous_node, branch_node_name)

            # Join node that collects branch outputs into a list in __internal__.current
            join_node_name = f"expression_step_{index}"

            def _join_impl(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
                new_state = dict(state)
                results: List[Any] = []
                for branch_key in branch_keys:
                    if branch_key in new_state:
                        results.append(new_state.pop(branch_key))
                    else:
                        results.append(None)
                return _expr_set_current(new_state, results)

            if async_mode:

                async def join_node(
                    state: MutableMapping[str, Any],
                ) -> MutableMapping[str, Any]:
                    # Joining is purely synchronous over the accumulated state.
                    return _join_impl(state)

            else:

                def join_node(
                    state: MutableMapping[str, Any],
                ) -> MutableMapping[str, Any]:
                    return _join_impl(state)

            graph.add_node(join_node_name, join_node)

            for branch_idx in range(num_branches):
                branch_node_name = f"expression_step_{index}_branch_{branch_idx}"
                graph.add_edge(branch_node_name, join_node_name)

            previous_node = join_node_name
            index += 1

        graph.add_edge(previous_node, END)
        return graph

    def _initial_state(
        self,
        context: ExecutionContext,
        state: Any,
        *args: Any,
        **kwargs: Any,
    ) -> MutableMapping[str, Any]:
        if self._mode == "expression":
            return _expr_init_state(state)

        if state is None:
            base_state: MutableMapping[str, Any] = {}
        elif isinstance(state, MutableMapping):
            base_state = dict(state)
        else:
            base_state = {"value": state}

        base_state.setdefault("messages", [])
        return base_state

    def _post_process(self, raw_result: Any, context: ExecutionContext) -> Any:
        if self._mode == "expression" and isinstance(raw_result, dict):
            return _expr_get_current(raw_result)
        return raw_result

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------
    def _build_task_node(
        self,
        step: StepDescriptor,
        context: ExecutionContext,
        *,
        async_mode: bool = False,
    ) -> Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]]:
        if step.func is None:
            raise ValueError(
                f"Step '{step.name}' is missing an associated callable via run()."
            )

        routes = step.options.get("routes", {})
        evaluators: Sequence[LangfuseEvaluator] = step.options.get("evaluators", ())
        runtime: Optional[EvaluatorRuntimeConfig] = step.options.get("evaluator_runtime")

        def _schedule_scoring(before: dict[str, Any], after: dict[str, Any]) -> None:
            if not evaluators:
                return

            def _run() -> None:
                ingest_langfuse_scores(
                    context=context,
                    step_name=step.name,
                    before=before,
                    after=after,
                    evaluators=evaluators,
                    runtime=runtime,
                )

            scheduler_cfg = runtime.scheduler if runtime is not None else None
            get_evaluation_scheduler(scheduler_cfg).submit(
                _run, label=f"{self.name}:{step.name}"
            )

        async def async_node(
            state: MutableMapping[str, Any],
        ) -> MutableMapping[str, Any]:
            payload = dict(state)
            before_payload = dict(payload)
            if self._mode == "expression":
                result = await self._call_callable_async(
                    step.func, _expr_get_current(payload), context=context
                )
                return _expr_set_current(payload, result)

            try:
                result = await self._call_callable_async(
                    step.func, payload, context=context
                )
            except Exception as exc:  # noqa: BLE001
                if "failure" not in routes:
                    raise
                payload["__route__"] = "failure"
                payload["__error__"] = exc
                _schedule_scoring(before_payload, payload)
                return payload

            if isinstance(result, Command):
                return result

            if isinstance(result, MutableMapping):
                payload.update(result)
            elif result is not None:
                payload[step.name] = result

            payload.setdefault("__route__", "success")
            if "success" not in routes:
                payload.pop("__route__", None)
            _schedule_scoring(before_payload, payload)
            return payload

        def sync_node(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
            payload = dict(state)
            before_payload = dict(payload)
            if self._mode == "expression":
                result = self._call_callable(
                    step.func, _expr_get_current(payload), context=context
                )
                return _expr_set_current(payload, result)
            try:
                result = self._call_callable(step.func, payload, context=context)
            except Exception as exc:  # noqa: BLE001
                if "failure" not in routes:
                    raise
                payload["__route__"] = "failure"
                payload["__error__"] = exc
                _schedule_scoring(before_payload, payload)
                return payload

            if isinstance(result, Command):
                return result

            if isinstance(result, MutableMapping):
                payload.update(result)
            elif result is not None:
                payload[step.name] = result

            payload.setdefault("__route__", "success")
            if "success" not in routes:
                payload.pop("__route__", None)
            _schedule_scoring(before_payload, payload)
            return payload

        return async_node if async_mode else sync_node

    def _build_branch_child_node(
        self,
        step: StepDescriptor,
        child_name: str,
        child_callable: Callable[..., Any],
        context: ExecutionContext,
        *,
        async_mode: bool = False,
    ) -> Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]]:
        """Build a node for an individual branch child.

        The child behaves like a regular task node keyed by ``child_name``:
        it receives the current payload, runs ``child_callable``, and merges
        its result back into the payload. Multiple branch children can safely
        run in parallel because the overall state schema is configured to
        merge concurrent dict updates.
        """
        routes = step.options.get("routes", {})
        branch_evaluators: Sequence[LangfuseEvaluator] = step.options.get("evaluators", ())
        child_evaluators_map: Mapping[str, Sequence[LangfuseEvaluator]] = step.options.get("child_evaluators", {})
        child_runtime_map: Mapping[str, EvaluatorRuntimeConfig] = step.options.get("child_evaluator_runtime", {})
        evaluators: Sequence[LangfuseEvaluator] = child_evaluators_map.get(
            child_name, branch_evaluators
        )
        runtime: Optional[EvaluatorRuntimeConfig] = child_runtime_map.get(
            child_name, step.options.get("evaluator_runtime")
        )

        def _schedule_scoring(before: dict[str, Any], after: dict[str, Any]) -> None:
            if not evaluators:
                return

            def _run() -> None:
                ingest_langfuse_scores(
                    context=context,
                    step_name=child_name,
                    before=before,
                    after=after,
                    evaluators=evaluators,
                    runtime=runtime,
                )

            scheduler_cfg = runtime.scheduler if runtime is not None else None
            get_evaluation_scheduler(scheduler_cfg).submit(
                _run, label=f"{self.name}:{child_name}"
            )

        if async_mode:

            async def async_node(
                state: MutableMapping[str, Any],
            ) -> MutableMapping[str, Any]:
                payload = dict(state)
                before_payload = dict(payload)
                try:
                    result = await self._call_callable_async(
                        child_callable, payload, context=context
                    )
                except Exception as exc:  # noqa: BLE001
                    if "failure" not in routes:
                        raise
                    payload["__route__"] = "failure"
                    payload["__error__"] = exc
                    _schedule_scoring(before_payload, payload)
                    return payload

                if isinstance(result, Command):
                    return result

                if isinstance(result, MutableMapping):
                    payload.update(result)
                elif result is not None:
                    payload[child_name] = result

                payload.setdefault("__route__", "success")
                if "success" not in routes:
                    payload.pop("__route__", None)
                _schedule_scoring(before_payload, payload)
                return payload

            return async_node

        def sync_node(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
            payload = dict(state)
            before_payload = dict(payload)
            try:
                result = self._call_callable(child_callable, payload, context=context)
            except Exception as exc:  # noqa: BLE001
                if "failure" not in routes:
                    raise
                payload["__route__"] = "failure"
                payload["__error__"] = exc
                _schedule_scoring(before_payload, payload)
                return payload

            if isinstance(result, Command):
                return result

            if isinstance(result, MutableMapping):
                payload.update(result)
            elif result is not None:
                payload[child_name] = result

            payload.setdefault("__route__", "success")
            if "success" not in routes:
                payload.pop("__route__", None)
            _schedule_scoring(before_payload, payload)
            return payload

        return sync_node

    def _build_checkpoint_node(
        self,
        step: StepDescriptor,
        context: ExecutionContext,
    ) -> Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]]:
        message = step.options.get("approval_message")

        def node(state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
            # In LangGraph v1, the interrupt() payload must be JSON-serializable
            # State is already managed by checkpointing, so we don't need to include it
            payload = {
                "checkpoint": step.name,
                "message": message,
            }
            response = interrupt(payload)

            if isinstance(response, Command):
                new_state = dict(state)
                if response.update:
                    new_state.update(response.update)
                if response.goto:
                    new_state["__route__"] = response.goto
                new_state.pop("__interrupt__", None)
                return new_state

            if isinstance(response, dict):
                updated_state = dict(state)
                for key, value in response.items():
                    updated_state[key] = value
                updated_state.setdefault("__route__", "success")
                updated_state.setdefault("__interrupt__", None)
                return updated_state

            # No explicit update provided; continue with original state
            new_state = dict(state)
            new_state["__interrupt__"] = payload
            return new_state

        return node

    def _add_step(self, name: str, kind: str) -> "WorkflowBuilder":
        self._assert_declarative_mode("_add_step")
        descriptor = StepDescriptor(name=name, kind=kind)
        self._steps.append(descriptor)
        self._current = descriptor
        return self

    def _ensure_current(self) -> StepDescriptor:
        if self._current is None:
            raise RuntimeError("No active step. Call start() before configuring steps.")
        return self._current

    def _assert_declarative_mode(self, method_name: str) -> None:
        if self._expression is not None:
            raise RuntimeError(
                f"Method '{method_name}' is not available when WorkflowBuilder is initialized from a WorkflowExpression."
            )
