Teams
=====

``Team`` is Andromeda's high-level orchestrator for multi-agent workflows.

It combines:

- a planner
- a supervisor
- one or more worker agents
- optional report generation

Use ``Team`` when you want an end-to-end pipeline (plan, delegate, synthesize)
instead of manually coordinating components.

How Team Executes
-----------------

A team run follows this order:

1. planner step (creates/updates plan)
2. supervisor step (routes and coordinates workers)
3. optional report step (if ``report.enabled=True``)

Entry points:

- ``team.begin(user_message)`` for sync execution
- ``await team.abegin(user_message)`` for async execution

Returned state usually includes:

- ``messages``
- ``plan``
- ``report_output`` (when reporting is enabled)

Minimal Team
------------

Start here for the simplest working multi-agent setup.

.. code-block:: python

    from andromeda.config import AndromedaConfig, AgentConfig, SupervisorConfig, ModelConfig, PlannerConfig
    from andromeda.core.team import Team
    from andromeda.tools.filesystem import make_filesystem_tools

    # Define model once
    model_cfg = ModelConfig(name="gpt-oss:20b", provider="litellm")

    # Create filesystem tools (make_filesystem_tools returns dict, convert to list)
    fs_tools_dict = make_filesystem_tools(allowed_dirs=["."])
    fs_tools_list = list(fs_tools_dict.values())

    # Agents with tools and stop conditions
    research_agent = AgentConfig(
        name="researcher",
        model=model_cfg,
        tools=fs_tools_list,
        prompt="""You are a research agent. Your task is to research a topic and provide a concise summary.
    When you have gathered the information and provided a clear response, STOP.
    Do not ask for more information or iterate endlessly.
    Focus on accuracy and clarity. Complete your response and await next instruction."""
    )

    # Supervisor with routing prompt and stop condition
    supervisor = SupervisorConfig(
        name="supervisor",
        model=model_cfg,
        prompt="""You are a supervisor. Route the user's question to the appropriate agent (researcher).
    Wait for their response. Once the agent provides findings, return the result to the user.
    Do not iterate endlessly. Complete the task when the agent has responded."""
    )

    # Planner for decomposition
    planner = PlannerConfig(
        model=model_cfg,
        task_type="research",
    )

    # Create config with all required components
    config = AndromedaConfig(
        agents={"researcher": research_agent},
        supervisor=supervisor,
        planner=planner,
    )

    # Create team and execute
    team = Team(config)
    print("Starting team execution...")
    result = team.begin("Summarize the impact AI has on modern logistics in 2 sentences.")
    print("\nResult:")
    if isinstance(result, dict) and "messages" in result:
        print(result["messages"][-1].content)
    else:
        print(result)
        
Async Team Run
--------------

Use ``abegin`` when integrating Team in async services or background workers.

.. code-block:: python

   import asyncio

    from andromeda.config import AndromedaConfig, AgentConfig, SupervisorConfig, ModelConfig, PlannerConfig
    from andromeda.core.team import Team


    async def main() -> None:
        cfg = AndromedaConfig(
            agents={
                "researcher": AgentConfig(
                    name="researcher",
                    model=ModelConfig(name="qwen3:8b", provider="litellm"),
                )
            },
            supervisor=SupervisorConfig(
                name="supervisor",
                model=ModelConfig(name="qwen3:8b", provider="litellm"),
            ),
            planner=PlannerConfig(
                model=ModelConfig(name="qwen3:8b", provider="litellm"),
                task_type="research",
            ),
        )

        team = Team(cfg)
        state = await team.abegin("Compare top open-source agent frameworks.")
        print(state["messages"][-1].content)


    if __name__ == "__main__":
        asyncio.run(main())

Team with Planner and Reporting
-------------------------------

Enable reporting to produce a structured final output in state, file, or both.

.. code-block:: python

   from andromeda.config import (
       AndromedaConfig,
       AgentConfig,
       SupervisorConfig,
       PlannerConfig,
       ReportConfig,
       ModelConfig,
       ValidationConfig,
       CitationConfig,
   )
   from andromeda.core.team import Team

   base_model = ModelConfig(name="llama3.1:8b", provider="litellm")

   researcher = AgentConfig(
       name="researcher",
       model=base_model,
   )

   supervisor_cfg = SupervisorConfig(
       name="supervisor",
       model=base_model,
       prompt="Coordinate agents and keep the plan up to date.",
       enable_planning=True,
   )

   planner_cfg = PlannerConfig(
       model=base_model,
       task_type="research",
       report_structure="Introduction, Current Landscape, Key Players, Risks, Conclusion",
   )

   report_cfg = ReportConfig(
       enabled=True,
       model=base_model,
       format="markdown",
       citations=CitationConfig(required=True, min_density=0.2),
       validation=ValidationConfig(
           enabled=True,
           min_sufficiency_score=0.8,
           model=base_model,
       ),
       output_mode="both",  # save in state and file
       output_path="team_report.md",
   )

   cfg = AndromedaConfig(
       agents={"researcher": researcher},
       supervisor=supervisor_cfg,
       planner=planner_cfg,
       report=report_cfg,
   )

   team = Team(cfg)
   state = team.begin("Deep dive on AI chip shortages and their impact on cloud providers.")
   print("Report:\n", state.get("report_output"))

Configuration Tips
------------------

- ``Team`` forces supervisor planning on internally for team workflows.
- Keep worker prompts specialized by role (researcher, writer, reviewer, etc.).
- Use ``report.enabled=True`` only when you need a final generated report stage.
- For iterative experimentation, use smaller models first, then scale up.

Important Team Behavior
-----------------------

At runtime, Team creates a temporary working directory and injects filesystem
tools into agents/supervisor so they can collaborate through shared artifacts.

This is especially useful for long tasks where agents need to write notes,
draft sections, or intermediate outputs before final synthesis.
