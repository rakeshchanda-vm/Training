from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Sequence

from andromeda import BaseMessage, HumanMessage

from .context import RuntimeContext, _deep_merge
from .json_utils import to_json_compatible as _to_json_compatible
from .validation import WorkflowValidationError, error


def _function_context(runtime_context: RuntimeContext, thread_id: str | None = None, metadata: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "runtime_context": runtime_context,
        **({"thread_id": thread_id} if thread_id is not None else {}),
        **({"metadata": metadata} if metadata is not None else {}),
    }


TOKEN_RE = re.compile(r"{{\s*([^}]+)\s*}}")


def _is_int_text(value: str) -> bool:
    return value.isascii() and value.isdigit()


def _get_by_path(root: Any, expr: str) -> Any:
    parts = expr.split(".")
    current = root
    for part in parts:
        if part == "":
            continue
        if isinstance(current, list) and _is_int_text(part):
            idx = int(part)
            if idx < 0 or idx >= len(current):
                raise KeyError(f"Index out of range: {part}")
            current = current[idx]
            continue
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Missing path part '{part}' in {expr}")
            current = current[part]
            continue
        raise KeyError(f"Cannot traverse path '{expr}' through type {type(current)!r}")
    return current


def _resolve_template_expr(expr: str, *, state: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
    expr = expr.strip()
    if expr.startswith("state."):
        return _get_by_path(state, expr[6:])
    if expr.startswith("inputs."):
        return _get_by_path(inputs, expr[7:])
    if expr == "state":
        return state
    if expr == "inputs":
        return inputs
    if expr.startswith("$."):
        return _get_by_path(state, expr[2:])

    # Literal fallback.
    return expr


def _render_value(value: Any, *, state: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        matches = list(TOKEN_RE.finditer(value))
        if not matches:
            return value

        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return _resolve_template_expr(matches[0].group(1), state=state, inputs=inputs)

        out = value
        for match in reversed(matches):
            expr = match.group(1)
            replacement = _resolve_template_expr(expr, state=state, inputs=inputs)
            if not isinstance(replacement, str):
                replacement = json.dumps(_to_json_compatible(replacement))
            out = out[: match.start()] + str(replacement) + out[match.end() :]
        return out

    if isinstance(value, list):
        return [_render_value(item, state=state, inputs=inputs) for item in value]

    if isinstance(value, dict):
        return {
            key: _render_value(item, state=state, inputs=inputs) for key, item in value.items()
        }

    return value


def _parse_messages(payload: Any, *, state: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[HumanMessage]:
    rendered = _render_value(payload, state=state, inputs=inputs)

    messages: list[HumanMessage] = []
    if isinstance(rendered, (str, int, float, bool)) or rendered is None:
        messages.append(HumanMessage(content="" if rendered is None else str(rendered)))
        return messages

    if isinstance(rendered, list):
        for entry in rendered:
            if isinstance(entry, str):
                messages.append(HumanMessage(content=entry))
                continue
            if isinstance(entry, dict):
                content = entry.get("content")
                if not isinstance(content, str):
                    raise WorkflowValidationError("Agent message entries must include string 'content'.")
                messages.append(HumanMessage(content=content))
                continue
            raise WorkflowValidationError(
                f"Unsupported message entry type {type(entry)!r} in workflow agent node."
            )
        return messages

    if isinstance(rendered, dict):
        content = rendered.get("content")
        if isinstance(content, str):
            messages.append(HumanMessage(content=content))
            return messages
        prompt = rendered.get("prompt")
        if isinstance(prompt, str):
            messages.append(HumanMessage(content=prompt))
            return messages
        raise WorkflowValidationError(
            "Agent node input must contain a 'content' or 'prompt' string."
        )

    raise WorkflowValidationError(
        "Agent node input must be a message map, list, or string."
    )


def _extract_messages_from_chunk(chunk: Any) -> list[BaseMessage]:
    if isinstance(chunk, BaseMessage):
        return [chunk]

    if isinstance(chunk, tuple):
        return [item for item in chunk if isinstance(item, BaseMessage)]

    if isinstance(chunk, list):
        return [item for item in chunk if isinstance(item, BaseMessage)]

    if not isinstance(chunk, dict):
        return []

    messages: list[BaseMessage] = []
    for key in ("messages", "chunk", "output"):
        value = chunk.get(key)
        if isinstance(value, BaseMessage):
            messages.append(value)
        elif isinstance(value, tuple):
            messages.extend(item for item in value if isinstance(item, BaseMessage))
        elif isinstance(value, list):
            messages.extend(item for item in value if isinstance(item, BaseMessage))

    # If no direct keys match, attempt one level deeper for event-style payloads.
    if not messages:
        data = chunk.get("data") if isinstance(chunk, dict) else None
        if isinstance(data, dict):
            for key in ("chunk", "output", "message", "messages"):
                value = data.get(key)
                if isinstance(value, BaseMessage):
                    messages.append(value)
                elif isinstance(value, tuple):
                    messages.extend(item for item in value if isinstance(item, BaseMessage))
                elif isinstance(value, list):
                    messages.extend(item for item in value if isinstance(item, BaseMessage))

    return messages


def _extract_output_payload(result: Any, spec: Any, *, state: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
    if spec is None:
        return result
    if not isinstance(spec, str):
        return _render_value(spec, state=state, inputs=inputs)

    token_match = TOKEN_RE.fullmatch(spec.strip())
    if token_match:
        return _resolve_template_expr(token_match.group(1), state=state, inputs=inputs)
    return _render_value(spec, state=state, inputs=inputs)


@dataclass(frozen=True)
class WorkflowNode:
    name: str
    kind: str
    agent: str | None
    function: str | None
    input: Any = None
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_data(cls, name: str, data: Mapping[str, Any]) -> "WorkflowNode":
        if not isinstance(data, Mapping):
            raise WorkflowValidationError(f"Invalid workflow node '{name}': expected mapping")

        kind = data.get("type")
        if kind not in {"agent", "function"}:
            raise WorkflowValidationError(
                f"Invalid workflow node '{name}': type must be 'agent' or 'function'."
            )

        agent_name = data.get("agent") if kind == "agent" else None
        function_ref = data.get("function") if kind == "function" else None
        if kind == "agent" and not isinstance(agent_name, str):
            raise WorkflowValidationError(
                f"Invalid workflow node '{name}': agent type requires 'agent' reference."
            )
        if kind == "function" and not isinstance(function_ref, str):
            raise WorkflowValidationError(
                f"Invalid workflow node '{name}': function type requires 'function' reference."
            )

        output = data.get("output")
        if output is None:
            output = {}
        if not isinstance(output, dict):
            raise WorkflowValidationError(
                f"Invalid workflow node '{name}': output must be mapping."
            )

        return cls(
            name=name,
            kind=kind,
            agent=agent_name if isinstance(agent_name, str) else None,
            function=function_ref if isinstance(function_ref, str) else None,
            input=data.get("input", {}),
            output=output,
        )


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    version: int | None
    nodes: list[WorkflowNode]
    edges: list[tuple[str, str]]
    raw_inputs: dict[str, Any]
    outputs: dict[str, Any]
    payload: dict[str, Any]
    path: Path
    definition_root: Path

    @classmethod
    def from_mapping(
        cls,
        name: str,
        path: Path,
        payload: Mapping[str, Any],
        definition_root: Path,
    ) -> "WorkflowDefinition":
        if not isinstance(payload, Mapping):
            raise WorkflowValidationError(f"Invalid workflow '{name}': expected mapping payload.")

        version = payload.get("version")
        raw_nodes = payload.get("nodes")
        if raw_nodes is None:
            raise WorkflowValidationError(f"Workflow '{name}' is missing required 'nodes'.")

        nodes: list[WorkflowNode] = []
        if isinstance(raw_nodes, dict):
            for node_name, node_data in raw_nodes.items():
                if not isinstance(node_name, str):
                    raise WorkflowValidationError(
                        "Workflow node names must be strings."
                    )
                nodes.append(WorkflowNode.from_data(node_name, node_data))
        elif isinstance(raw_nodes, list):
            for item in raw_nodes:
                if not isinstance(item, Mapping):
                    raise WorkflowValidationError("Workflow node list entries must be mappings.")
                node_name = item.get("name")
                if not isinstance(node_name, str):
                    raise WorkflowValidationError("Workflow node list entries must include string 'name'.")
                nodes.append(WorkflowNode.from_data(node_name, item))
        else:
            raise WorkflowValidationError(
                f"Workflow '{name}' has unsupported nodes format: expected mapping or list."
            )

        raw_edges: list[tuple[str, str]] = []
        for edge in payload.get("edges", []):
            if not isinstance(edge, Mapping):
                raise WorkflowValidationError("Workflow edges must be mapping objects.")
            source = edge.get("from")
            target = edge.get("to")
            if not isinstance(source, str) or not isinstance(target, str):
                raise WorkflowValidationError("Workflow edge must have string 'from' and 'to' fields.")
            raw_edges.append((source, target))

        raw_inputs = payload.get("inputs")
        if raw_inputs is None:
            raw_inputs = {}
        elif not isinstance(raw_inputs, dict):
            raise WorkflowValidationError("Workflow 'inputs' must be a mapping.")

        outputs = payload.get("outputs")
        if outputs is None:
            outputs = {}
        elif not isinstance(outputs, dict):
            raise WorkflowValidationError("Workflow 'outputs' must be a mapping.")

        declared_name = payload.get("name")
        resolved_name = declared_name if isinstance(declared_name, str) and declared_name else name

        return cls(
            name=resolved_name,
            version=version if isinstance(version, int) else None,
            nodes=nodes,
            edges=raw_edges,
            raw_inputs=raw_inputs,
            outputs=outputs,
            payload=dict(payload),
            path=path,
            definition_root=definition_root,
        )

    @property
    def node_names(self) -> list[str]:
        return [node.name for node in self.nodes]

    def validate(self, entrypoints: Sequence[str] | None = None) -> list[Any]:
        issues = []
        node_set = set(self.node_names)

        for source, target in self.edges:
            if source not in node_set:
                issues.append(error(f"Unknown edge source '{source}'", path=str(self.path)))
            if target not in node_set:
                issues.append(error(f"Unknown edge target '{target}'", path=str(self.path)))

        if entrypoints:
            for required in entrypoints:
                if required not in node_set:
                    issues.append(error(f"Missing required workflow node '{required}'", path=str(self.path)))
        return issues


class WorkflowFunctionResolver:
    """Load and cache functions from workflow bundle files."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Path, str, str], tuple[Callable[..., Any], Path]] = {}

    @staticmethod
    def _safe_module_name(bundle_root: Path, module: str) -> str:
        # Deterministic digest (unlike the PYTHONHASHSEED-salted builtin ``hash``)
        # so the synthetic module name is stable across processes.
        seed = f"{bundle_root.resolve()}::{module}".encode("utf-8")
        digest = hashlib.sha256(seed).hexdigest()[:16]
        return f"andromeda_runtime_fn_{digest}"

    @staticmethod
    def _resolve_module_path(workflow_root: Path, module_name: str) -> Path:
        root = workflow_root.resolve()
        module_path = Path(module_name.replace(".", "/"))
        if module_path.is_absolute():
            raise WorkflowValidationError(
                f"Invalid workflow function module reference '{module_name}'. Use a path relative to the workflow directory."
            )

        candidate = (root / f"{module_path}.py").resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:  # pragma: no cover - defensive, defensive in edge cases
            raise WorkflowValidationError(
                f"Invalid function module reference '{module_name}.py' for workflow in {workflow_root}"
            ) from exc
        return candidate

    def resolve(self, workflow_root: Path, function_ref: str) -> Callable[..., Any]:
        if ":" not in function_ref:
            raise WorkflowValidationError(
                f"Invalid function reference '{function_ref}'. Expected 'module:function'."
            )

        module_name, function_name = function_ref.split(":", 1)
        module_name = module_name.strip()
        function_name = function_name.strip()
        if not module_name or not function_name:
            raise WorkflowValidationError(
                f"Invalid function reference '{function_ref}'."
            )

        module_path = self._resolve_module_path(workflow_root, module_name)
        if not module_path.exists():
            raise WorkflowValidationError(
                f"Could not find function module '{module_name}.py' for workflow in {workflow_root}"
            )

        cache_key = (workflow_root.resolve(), module_name, function_name)
        if cache_key in self._cache:
            return self._cache[cache_key][0]

        try:
            spec = importlib.util.spec_from_file_location(
                self._safe_module_name(workflow_root, module_name), module_path
            )
            if spec is None or spec.loader is None:
                raise WorkflowValidationError(
                    f"Unable to create import spec for module '{module_name}' at {module_path}"
                )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            raise WorkflowValidationError(
                f"Failed to import workflow function module '{module_path}': {exc}"
            ) from exc

        if not hasattr(module, function_name):
            raise WorkflowValidationError(
                f"Module '{module_name}' in {workflow_root} does not define '{function_name}'."
            )

        fn = getattr(module, function_name)
        if not callable(fn):
            raise WorkflowValidationError(
                f"Workflow function '{function_name}' is not callable in '{module_path}'."
            )

        self._cache[cache_key] = (fn, module_path)
        return fn


def _call_function_node(
    fn: Callable[..., Any],
    node_input: Any,
    *,
    state: Mapping[str, Any],
    context: RuntimeContext,
    thread_id: str | None,
    metadata: Mapping[str, Any] | None,
) -> Any:
    """Invoke a workflow function with a small, versioned signature policy.

    Supported forms (in precedence order):
    - ``fn(input, state, context_payload)``
    - ``fn(input, state, context)``
    - ``fn(input, state)``
    - ``fn(input)``
    - keyword-based variants for flexible naming
    """

    context_payload = _function_context(context, thread_id=thread_id, metadata=metadata)

    candidate_calls = [
        ((node_input, state, context_payload), {}),
        ((node_input, state, context), {}),
        ((node_input, context), {}),
        ((node_input, state), {}),
        ((node_input,), {}),
        ((), {
            "input": node_input,
            "state": state,
            "context": context,
            "runtime_context": context,
            "thread_id": thread_id,
            "metadata": metadata,
        }),
        ((), {
            "input": node_input,
            "state": state,
            "runtime_context": context_payload,
        }),
        ((), {
            "input": node_input,
            "runtime_context": context,
            "thread_id": thread_id,
            "metadata": metadata,
        }),
    ]

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        # Select the first call form that fully binds to the function signature,
        # then invoke it exactly once. Any error raised by the function *body*
        # (including TypeError) propagates unchanged rather than being misreported
        # as a signature mismatch.
        for args, kwargs in candidate_calls:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            return fn(*args, **kwargs)

        raise WorkflowValidationError(
            f"Unsupported signature for workflow function '{fn.__name__}'."
        )

    # Signature could not be introspected (e.g. some builtins/C callables): fall
    # back to trying each form until one binds without a TypeError.
    last_exc: TypeError | None = None
    for args, kwargs in candidate_calls:
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue

    raise WorkflowValidationError(
        f"Unsupported signature for workflow function '{fn.__name__}': {last_exc}"
    )


def _topological_order(
    nodes: list[WorkflowNode],
    edges: list[tuple[str, str]],
    *,
    source_path: Path,
) -> list[WorkflowNode]:
    order: list[WorkflowNode] = []
    name_to_node = {node.name: node for node in nodes}

    indegree: dict[str, int] = {node.name: 0 for node in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for source, target in edges:
        if source not in name_to_node or target not in name_to_node:
            raise WorkflowValidationError(
                f"Invalid edge in {source_path}: '{source}' -> '{target}' references unknown nodes."
            )
        adjacency[source].append(target)
        indegree[target] = indegree.get(target, 0) + 1

    queue: deque[str] = deque([name for name, in_degree in indegree.items() if in_degree == 0])

    while queue:
        current = queue.popleft()
        order.append(name_to_node[current])
        for nxt in adjacency.get(current, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(nodes):
        raise WorkflowValidationError(f"Workflow graph at {source_path} contains a cycle.")

    return order


def _apply_node_outputs(
    state: MutableMapping[str, Any],
    node_result: Any,
    output_cfg: dict[str, Any],
    *,
    node_name: str,
    context_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> None:
    if not output_cfg:
        return

    if "set" in output_cfg:
        raw_set = output_cfg["set"]
        if not isinstance(raw_set, dict):
            raise WorkflowValidationError(
                f"Workflow node '{node_name}' output.set must be a mapping."
            )
        for key, value in raw_set.items():
            state[key] = _render_value(value, state=context_state, inputs=inputs)

    if "merge" in output_cfg:
        raw_merge = output_cfg["merge"]
        merged = _extract_output_payload(node_result, raw_merge, state=context_state, inputs=inputs)
        if not isinstance(merged, dict):
            raise WorkflowValidationError(
                f"Workflow node '{node_name}' output.merge must evaluate to a mapping."
            )
        state.update(merged)

    if "assign" in output_cfg:
        raw_assign = output_cfg["assign"]
        if not isinstance(raw_assign, dict):
            raise WorkflowValidationError(
                f"Workflow node '{node_name}' output.assign must be a mapping."
            )
        for key, value in raw_assign.items():
            state[key] = _render_value(value, state=context_state, inputs=inputs)


@dataclass
class WorkflowRunResult:
    state: dict[str, Any]
    messages: list[Any] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    text: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeWorkflow:
    name: str
    definition: WorkflowDefinition
    defaults: Mapping[str, Any]
    context: RuntimeContext
    agent_builder: Callable[[str], Any] | None = None

    def run(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowRunResult:
        final_result: WorkflowRunResult | None = None
        for event in self._execution_events(inputs=inputs, thread_id=thread_id, metadata=metadata):
            if event["type"] == "completed":
                final_result = WorkflowRunResult(
                    state=event["state"],
                    messages=event.get("messages", []),
                    raw=event.get("raw"),
                    text=event.get("text"),
                    outputs=event.get("outputs", {}),
                )
        if final_result is None:
            raise RuntimeError("Workflow stream did not yield completed result")
        return final_result

    def stream(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream_mode: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        resolved_mode = stream_mode or "values"
        for chunk in self._execution_events(
            inputs=inputs,
            thread_id=thread_id,
            metadata=metadata,
            stream_mode=resolved_mode,
        ):
            yield chunk

    def _execution_events(
        self,
        *,
        inputs: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        stream_mode: str = "values",
    ) -> Iterator[dict[str, Any]]:
        if inputs is None:
            inputs = {}

        # Merge defaults for workflow-level metadata, but do not mutate input objects.
        merged_inputs: Dict[str, Any] = _deep_merge(dict(self.defaults), dict(inputs))

        state: dict[str, Any] = dict(merged_inputs)
        state.setdefault("__inputs", dict(inputs))
        messages: list[Any] = []
        run_results: dict[str, Any] = {}

        yield {
            "type": "workflow.started",
            "name": self.name,
            "inputs": merged_inputs,
        }

        execution_order = (
            _topological_order(self.definition.nodes, self.definition.edges, source_path=self.definition.path)
            if self.definition.edges
            else list(self.definition.nodes)
        )

        fn_resolver = WorkflowFunctionResolver()
        agent_cache: dict[str, Any] = {}

        for node in execution_order:
            node_input = _render_value(node.input, state=state, inputs=merged_inputs)
            yield {
                "type": "node.started",
                "node": node.name,
                "node_type": node.kind,
                "node_ref": node.agent if node.kind == "agent" else node.function,
                "input": _to_json_compatible(node_input),
            }
            if node.kind == "agent":
                if not node.agent:
                    raise WorkflowValidationError(
                        f"Workflow node '{node.name}' is missing agent name."
                    )

                if node.agent not in agent_cache:
                    if self.agent_builder is None:
                        raise WorkflowValidationError(
                            "Runtime workflow lacks an agent builder."
                        )

                    # Build-once cache for workflow execution.
                    agent_cache[node.agent] = self.agent_builder(node.agent)

                workspace_agent = agent_cache[node.agent]
                node_messages = _parse_messages(node_input, state=state, inputs=merged_inputs)
                # NOTE: Agent nodes drive the agent's single-agent graph via
                # ``_run_workflow`` (which still has the workspace tools and prompt
                # bound at construction), not the supervisor's multi-agent
                # ``supervise()`` loop used by ``WorkspaceAgent.run``. Team
                # delegation / long-horizon orchestration are intentionally skipped
                # for workflow nodes in this MVP.
                if not hasattr(workspace_agent, "stream"):
                    result_payload = workspace_agent._run_workflow(
                        node_messages,
                        remember="last",
                        thread_id=thread_id,
                        metadata=metadata,
                    )
                else:
                    prior_memory: list[Any] = []
                    if hasattr(workspace_agent, "memory"):
                        prior_memory = list(getattr(workspace_agent, "memory") or [])

                    stream_mode_for_node = stream_mode or "updates"
                    result_messages: list[Any] = []
                    for chunk in workspace_agent.stream(
                        node_messages,
                        remember="last",
                        thread_id=thread_id,
                        stream_mode=stream_mode_for_node,
                        metadata=metadata,
                    ):
                        messages_from_chunk = _extract_messages_from_chunk(chunk)
                        if messages_from_chunk:
                            result_messages = messages_from_chunk

                        yield {
                            "type": "node.chunk",
                            "node": node.name,
                            "node_type": node.kind,
                            "node_ref": node.agent,
                            "chunk": _to_json_compatible(chunk),
                        }

                    if hasattr(workspace_agent, "memory"):
                        post_memory = list(getattr(workspace_agent, "memory") or [])
                        if len(post_memory) > len(prior_memory):
                            result_messages = list(
                                item
                                for item in post_memory[len(prior_memory) :]
                                if isinstance(item, BaseMessage)
                            )

                    result_payload = {"messages": result_messages}

                node_messages = result_payload.get("messages", [])
                messages.extend(node_messages)

                if isinstance(node_messages, list) and node_messages:
                    result_value = node_messages[-1].content
                else:
                    result_value = None

                if not isinstance(result_payload, dict):
                    result_payload = {"messages": node_messages}
                node_result = {
                    "result": result_value,
                    "payload": result_payload,
                    "messages": node_messages,
                }
                run_results[node.name] = node_result
            else:
                if not node.function:
                    raise WorkflowValidationError(
                        f"Workflow node '{node.name}' is missing function reference."
                    )
                fn = fn_resolver.resolve(self.definition.definition_root, node.function)
                node_result = _call_function_node(
                    fn,
                    node_input,
                    state=state,
                    context=self.context,
                    thread_id=thread_id,
                    metadata=metadata,
                )
                run_results[node.name] = node_result

            state["__result"] = node_result
            state["__node"] = node.name
            _apply_node_outputs(
                state,
                node_result=node_result,
                output_cfg=node.output,
                node_name=node.name,
                context_state=state,
                inputs=merged_inputs,
            )

            yield {
                "type": "node.completed",
                "node": node.name,
                "node_type": node.kind,
                "result": _to_json_compatible(node_result),
                "state": _to_json_compatible(state),
            }

        workflow_output = _render_value(self.definition.outputs, state=state, inputs=merged_inputs)
        text = None
        if isinstance(workflow_output, str):
            text = workflow_output
        elif "result" in workflow_output:
            text = str(workflow_output["result"])

        raw = {"outputs": workflow_output, "nodes": run_results, "state": state}

        yield {
            "type": "completed",
            "state": state,
            "text": text,
            "raw": raw,
            "messages": messages,
            "outputs": workflow_output if isinstance(workflow_output, dict) else {},
        }


def parse_workflow_definition(
    path: Path,
    payload: Mapping[str, Any],
    context: RuntimeContext,
    *,
    defaults: Mapping[str, Any],
    agent_builder: Callable[[str], Any] | None = None,
) -> RuntimeWorkflow:
    definition_name = path.stem
    definition = WorkflowDefinition.from_mapping(
        definition_name,
        path,
        payload,
        definition_root=path.parent,
    )
    definition.validate()

    merged_defaults = _deep_merge(dict(defaults), dict(definition.payload))
    return RuntimeWorkflow(
        name=definition.name,
        definition=definition,
        defaults=merged_defaults,
        context=context,
        agent_builder=agent_builder,
    )
