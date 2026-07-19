Workflow Builder
================

``WorkflowBuilder`` is Andromeda's low-level orchestration API for deterministic,
state-driven pipelines.

Use it when you need explicit step control instead of full agent/team automation.

What it gives you:

- explicit step graph (start -> then -> finish)
- branching and route control
- parallel branches
- checkpoints and resume
- sync/async execution and streaming

Core Concepts
-------------

- Workflow state is a mutable mapping (usually a ``dict``).
- Each step function receives state and returns partial/full state updates.
- The builder compiles to a LangGraph workflow under the hood.

Simple Linear Workflow
----------------------

Use this pattern for straightforward multi-step processing.

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder

   def ingest(state: Dict[str, Any]) -> Dict[str, Any]:
       numbers = state.get("numbers", [])
       return {"numbers": numbers}

   def compute_sum(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"sum": sum(state.get("numbers", []))}

   def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
       return state | {"summary": f"Total = {state['sum']}"}

   workflow = WorkflowBuilder(name="SimpleLinearWorkflow")
   (
       workflow
       .start("ingest").run(ingest)
       .then("sum").run(compute_sum)
       .finish("finalize").run(finalize)
   )

   result = workflow.execute(state={"numbers": [1, 2, 3, 4]})
   print("Summary:", result["summary"])

Conditional Routing
-------------------

Use ``if_succeeds`` / ``if_fails`` to route based on step outcome.

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder

   def risky_step(state: Dict[str, Any]) -> Dict[str, Any]:
       if state.get("should_fail"):
           raise RuntimeError("Simulated failure")
       return {"status": "ok"}

   def on_failure(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"status": "failed", "handled": True}

   def on_success(state: Dict[str, Any]) -> Dict[str, Any]:
       return {"status": "completed"}

   wf = WorkflowBuilder(name="RoutingExample")
   (
       wf
       .start("do_risky_thing")
           .run(risky_step)
           .if_fails().goto("failure_handler")
           .if_succeeds().goto("success_handler")
       .then("failure_handler")
           .run(on_failure)
       .then("success_handler")
           .run(on_success)
   )

   final_state = wf.execute(state={"should_fail": True})
   print(final_state["status"], final_state.get("handled"))

Parallel Branches
-----------------

Use ``branch(...).parallel(...)`` for independent computations.

.. code-block:: python

   from typing import Dict, Any
   from andromeda.core.workflow import WorkflowBuilder

   def compute_sum(state: Dict[str, Any]) -> Dict[str, Any]:
       numbers = state.get("numbers", [])
       return {"sum": sum(numbers)}

   def compute_count(state: Dict[str, Any]) -> Dict[str, Any]:
       numbers = state.get("numbers", [])
       return {"count": len(numbers)}

   wf = WorkflowBuilder(name="ParallelExample")
   (
       wf
       .start("prepare")
           .run(lambda s: {"numbers": [1, 2, 3, 4, 5]})
       .branch("analytics")
           .parallel([
               ("sum_step", compute_sum),
               ("count_step", compute_count),
           ])
           .merge_results()
       .finish("done")
           .run(lambda s: s)
   )

   out = wf.execute(state={})
   print("Sum:", out["sum"], "Count:", out["count"])

Checkpoint and Resume (Human-in-the-loop)
-----------------------------------------

Use checkpoints when you need approval before continuing.

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, WorkflowBase, Command

   builder = WorkflowBuilder(name="checkpoint_flow")
   (
       builder
       .start("init")
       .run(lambda state: {"messages": ["started"]})
       .checkpoint("human_review")
       .pause_for_approval("Approve or redirect")
       .finish("complete")
       .run(lambda state: state | {"messages": state["messages"] + ["completed"]})
   )

   # First run pauses at checkpoint
   initial = WorkflowBase.run(builder, state=None)
   thread_id = initial.context.thread_id

   # Resume execution from "complete"
   resumed = builder.execute(
       resume=Command(goto="complete"),
       thread_id=thread_id,
   )
   print(resumed["messages"])

Operator Style (Expression DSL)
-------------------------------

Operator workflows focus on transforming a single "current value" and work well for functional composition:

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, task, conditional

   @task
   def add_one(value: int) -> int:
       return value + 1

   @task
   def multiply_two(value: int) -> int:
       return value * 2

   flow = add_one >> multiply_two
   workflow = WorkflowBuilder.from_expression(flow, name="operator_chain")
   print(workflow.execute(state=1))  # 4

   # Conditional expression branch
   expr = add_one >> conditional(
       true_branch=lambda x: x * 10,
       false_branch=lambda x: x,
       condition=lambda x: x > 1,
   )
   conditional_workflow = WorkflowBuilder.from_expression(expr, name="operator_conditional")
   print(conditional_workflow.execute(state=1))  # 20

Streaming Workflow Execution
----------------------------

Use streaming when you want progressive state updates.

.. code-block:: python

   import asyncio
   from andromeda.core.workflow import WorkflowBuilder
   
   def step_one(state):
       return state | {"messages": state.get("messages", []) + ["first"]}

   def step_two(state):
       return state | {"messages": state["messages"] + ["second"]}

   workflow = WorkflowBuilder(name="StreamingExample")
   (
       workflow
       .start("one").run(step_one)
       .finish("two").run(step_two)
   )

   # Sync streaming
   for chunk in workflow.stream(state={"messages": []}):
       print("Chunk:", chunk)

   # Async streaming
   async def main():
       async for chunk in workflow.astream(state={"messages": []}):
           print("AChunk:", chunk)

   asyncio.run(main())

Tips
----

- Keep step functions pure where possible (state in, state out).
- Use clear step names to make traces and debugging easier.
- Prefer ``execute``/``aexecute`` for final results and ``stream``/``astream`` for UIs.
- Use checkpoints for safe human approval points in critical workflows.

