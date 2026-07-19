Langfuse Tracing & Evaluation
=============================

Andromeda integrates with `Langfuse <https://langfuse.com/>`_ for:

- **Tracing**: see LLM/tool activity on a trace timeline.
- **Evaluation / scoring**: ingest custom scores back into Langfuse.

Install
-------

.. code-block:: bash

   pip install langfuse

Configure
---------

Set the Langfuse environment variables (example values):

.. code-block:: bash

   export LANGFUSE_PUBLIC_KEY=pk-lf-...
   export LANGFUSE_SECRET_KEY=sk-lf-...
   export LANGFUSE_HOST=https://agentops.valuemomentum.studio 

Trace IDs (distributed tracing)
-------------------------------

By default, Langfuse generates random IDs for traces/spans. For distributed tracing and
for linking scores back to the correct trace, Andromeda brings its own Langfuse trace ID.

- Andromeda derives a deterministic Langfuse ``trace_id`` from the workflow ``thread_id``.
- The derived Langfuse trace id is a **32-character lowercase hex string** (it will not equal ``thread_id``).
- To ensure stable correlation across services, pass a stable domain id as ``thread_id`` when executing:

.. code-block:: python

   result = workflow.execute(state=initial_state, thread_id="request-123")

Trace input/output
------------------

The root workflow span (``workflow:<workflow_name>``) updates the Langfuse trace with the full
workflow input and output state:

- ``update_trace(input=<initial_state>)`` at the beginning
- ``update_trace(output=<final_state>)`` at the end

This uses best-effort serialization (messages and objects are converted to JSON-friendly forms
where possible).

Per-node spans
--------------

Each workflow node also runs inside a Langfuse span named:

- ``workflow.step:<step_name>``

This improves trace-context propagation when the workflow engine executes nodes in worker
threads (for example, parallel branches).

Evaluation / scoring
--------------------

Attach evaluators to a node using ``with_evaluators``:

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, correctness, hallucination

   wf = WorkflowBuilder(name="ScoringExample")
   (
       wf.start("generate")
         .run(lambda s: s)
         .with_evaluators(
             [
                 correctness(sample_rate=1.0),
                 hallucination(sample_rate=0.5),
             ],
             # Bounded background scheduler (best-effort)
             scheduler={"max_workers": 2, "max_pending": 128},
             # Model used by built-in LLM evaluators
             model={
                 "name": "gpt-oss:20b",
                 "provider": "litellm",
                 "temperature": 0.0,
                 "other_args": {"base_url": "http://localhost:11434"},
             },
         )
   )

Notes:

- Evaluators run in the background and do not block the workflow.
- Scores are **automatically scoped by step name**, e.g. ``refine:correctness``.

Built-in evaluators
-------------------

Andromeda ships a few evaluators you can attach directly:

- ``correctness`` (LLM-based)
- ``hallucination`` (LLM-based)
- ``relevance`` (LLM-based)
- ``tool_usage`` (heuristic)

Custom evaluators
-----------------

Custom evaluators are regular callables that receive a ``WorkflowEvaluationInput`` and return one of:

- ``None`` (skip)
- a numeric / categorical / boolean value
- ``(value, comment)``
- a ``LangfuseScore`` or a list of ``LangfuseScore``

The input object contains:

- ``step_name``
- ``before`` / ``after`` payload snapshots
- ``trajectory`` (messages/tools/results summary)
- ``input`` / ``output`` (selected via presets)

Choosing evaluator input/output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can choose what ends up in ``inp.input`` and ``inp.output`` by setting ``input_preset`` /
``output_preset`` on the evaluator.

Supported presets include: ``raw_payload``, ``first_message``, ``last_message``, ``all_messages``,
``tools_called``, ``tool_results``, and ``trajectory``.

If you want an evaluator to use *arbitrary state*, you have two options:

- Use ``raw_payload`` so ``inp.input`` / ``inp.output`` are the full state dicts, then read any keys you need.
- Ignore presets entirely and read directly from ``inp.before`` / ``inp.after`` (these always contain full payload snapshots).

Example: custom state-based evaluator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This evaluator scores whether the state contains a non-empty ``summary`` after the step runs.
It uses ``raw_payload`` so it can access arbitrary keys in the state.

.. code-block:: python

   from andromeda.core.workflow import LangfuseEvaluator

   def has_summary(inp):
       summary = (inp.output or {}).get("summary")
       ok = bool(summary and str(summary).strip())
       return (1.0 if ok else 0.0, "summary present" if ok else "missing/empty summary")

   evaluator = LangfuseEvaluator(
       name="has_summary",
       eval_fn=has_summary,
       data_type="NUMERIC",
       input_preset="raw_payload",
       output_preset="raw_payload",
   )

   # Attach per node:
   # wf.then("summarize").run(...).with_evaluators([evaluator])

Types: ``from andromeda.core.workflow import WorkflowEvaluationInput, LangfuseScore, LangfuseEvaluator``


Workflow Evaluation and Langfuse Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Attach evaluators to workflow nodes to ingest Langfuse scores (non-blocking, background).
Scores are automatically scoped by node name (for example ``refine:correctness``).

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, correctness, hallucination, relevance
   from andromeda import HumanMessage

   wf = WorkflowBuilder(name="ScoringExample")
   (
       wf.start("generate")
         .run(lambda s: s)
         .with_evaluators(
             [correctness(), hallucination(), relevance()],
             scheduler={"max_workers": 2, "max_pending": 128},
             model={"name": "gpt-oss:20b", "provider": "litellm", "temperature": 0.0, "other_args": {"base_url": "http://localhost:11434"}},
         )
   )

   out = wf.execute(state={"messages": [HumanMessage(content="Write a haiku about databases.")]})

Custom evaluators can also read arbitrary keys from the workflow state. Use the ``raw_payload``
preset to pass the full state dict into ``inp.input`` / ``inp.output`` (or read from
``inp.before`` / ``inp.after`` directly).

.. code-block:: python

   from andromeda.core.workflow import WorkflowBuilder, LangfuseEvaluator

   def has_answer(inp):
       # Because output_preset="raw_payload", inp.output is the full state dict after the step.
       answer = (inp.output or {}).get("answer")
       ok = bool(answer and str(answer).strip())
       return (1.0 if ok else 0.0, "answer present" if ok else "missing/empty answer")

   evaluator = LangfuseEvaluator(
       name="has_answer",
       eval_fn=has_answer,
       data_type="NUMERIC",
       input_preset="raw_payload",
       output_preset="raw_payload",
   )

   wf = WorkflowBuilder(name="CustomEvaluatorExample")
   (
       wf.start("generate")
         .run(lambda s: {"answer": "hello"})
         .with_evaluators([evaluator])
   )
   wf.execute(state={})

