from textwrap import dedent
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
import shutil

from andromeda.config.config import AgentConfig, SupervisorConfig
from andromeda.core.agent import Agent
from andromeda.core.workflow import Command, WorkflowBuilder
from andromeda.core.supervisor import Supervisor
from andromeda.core.planner import PlannerAgent
from andromeda.reporting.writer import ReportWriter
from andromeda.config import AndromedaConfig
from andromeda import HumanMessage
from andromeda.core.middleware import LLMToolSelectorMiddleware
from andromeda.workspace import WorkspacePolicy, WorkspaceSession
from pyeztrace.tracer import trace

from andromeda.utils.logger import log_error


@trace(exclude=["log_*"])
class Team:
    """
    Team orchestrates a group of agents and a supervisor to collaboratively solve complex tasks.

    This class manages the initialization and coordination of multiple agents, a supervisor agent,
    an event system for tracking progress, and a state graph for workflow execution. It provides
    both synchronous and asynchronous interfaces for running agent workflows, handles error reporting,
    and supports report generation.

    Attributes:
        config (andromedaConfig): The configuration object containing all settings for agents, supervisor, and reporting.
        agents (Dict[str, Agent]): Dictionary of agent instances managed by the team.
        supervisor (SupervisorAgent): The supervisor agent responsible for routing and overseeing tasks.
        graph (StateGraph): The state graph representing the workflow of agents and supervisor.
        compiled (CompiledStateGraph): The compiled version of the state graph for efficient execution.
    """

    def __init__(self, config: AndromedaConfig) -> None:
        """
        Initializes the Team with agents, supervisor, event system, and state graph.

        This constructor sets up the event emitter for tracking progress, instantiates all agents
        according to the provided configuration (including validation and citation settings), and
        creates the supervisor agent. It then builds a state graph connecting all agents and the
        supervisor, compiling it for later execution.

        Args:
            config (andromedaConfig): The configuration object containing settings for agents, supervisor, and reporting.
        """
        self.config: AndromedaConfig = config

        # Normalize agent configs to a list for downstream components.
        raw_agents = config.agents
        if isinstance(raw_agents, dict):
            agent_list = list(raw_agents.values())
        else:
            agent_list = raw_agents

        for agent in agent_list:
            if isinstance(agent, AgentConfig) and LLMToolSelectorMiddleware not in agent.middleware.custom:
                agent.middleware.custom.append(LLMToolSelectorMiddleware(
                    system_prompt=dedent(
                        """
                        List the relevant tools for this task.
                        At least 2 research tools are required.
                        Include filesystem tools always.
                        """),
                    always_include=['write_file']
                ))
            elif isinstance(agent, Agent) and LLMToolSelectorMiddleware not in agent.middleware:
                agent.middleware.append(LLMToolSelectorMiddleware(
                    system_prompt=dedent(
                        """
                        List the relevant tools for this task.
                        At least 2 research tools are required.
                        Include relevant filesystem tools always.
                        """),
                    always_include=['write_file'])
                )

        tmp_run_id = str(uuid.uuid4())[:6]
        tmp_dir = Path(f"~/.andromeda/tmp/{tmp_run_id}").expanduser()
        try:
            # Create the temporary directory first. Only assign to self.tmp_dir after creation
            # succeeded so other code can safely assume existence when present.
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self.tmp_dir = tmp_dir
            self.workspace_session = WorkspaceSession.create(
                backend="local_fs",
                root=tmp_dir,
                policy=WorkspacePolicy(
                    read_only=False,
                    enable_shell=False,
                    tool_profile="full_compatibility",
                ),
            )
            fs_tools = self.workspace_session.tools()
            for agent in agent_list:
                for tool in fs_tools.values():
                    if tool not in agent.tools:
                        agent.tools.append(tool)
            
            for tool in fs_tools.values():
                if tool not in config.supervisor.tools:
                    config.supervisor.tools.append(tool)

            config.supervisor.middleware.tool_error_handler = True # must be true for a Team

            for agent in agent_list:
                if isinstance(agent, AgentConfig):
                    agent.prompt += (
                        f"\n\nYou are working in a temporary directory: {tmp_dir}.\n"
                        "This is your ONLY workspace. You MUST write ALL research findings and output files directly in the root of this directory. "
                        "Do NOT create any subdirectories or folders for your work; place ALL outputs, regardless of type, in the root.\n"
                        "If instructed to create or save a file, always use the root of this temporary directory.\n"
                        "The files will be the main research findings memory for your future tasks and must be thorough and detailed. "
                        "Avoid high-level-only notes: include concrete evidence, key data points, assumptions, alternatives considered, risks, and clear conclusions. "
                        "When making claims, tie them to available evidence and keep source tags in [Search #n] format when applicable. "
                        "When evidence is missing, explicitly record what is missing and why it matters; do not fill gaps with invented specifics. "
                        "Prefer primary sources and capture enough raw detail (numbers, dates, entity names, direct evidence) for downstream reporting. "
                        "If the same file already exists, use append_to_file or edit_file tool to update the file. "
                        "For every file created, provide in your final response to user (not in files):\n"
                        "• The file name (in the root dir)\n"
                        "• A description of its content and purpose\n"
                        "• A summary of what was learned or demonstrated\n"
                        "Do not simply summarize the workflow, but include the actual research results and analysis in your response. "
                        "Important: Do not output .png or .csv content directly, and never attempt to create or reference folders or non-root paths. "
                        "Respond to user with a clearly itemized list of any files (if created), all at the root."
                    )

            config.supervisor.prompt += (
                f"\n\nYou are working in a temporary directory: {tmp_dir}.\n"
                "This directory is used as shared working memory for research artifacts and results. "
                "Sub-agents will create ALL output files ONLY at the root of this directory—never in subdirectories. Reinforce this rule in your instructions.\n"
                "When instructing sub-agents, give precise and explicit guidance for each research step required. "
                "Clearly specify what markdown (.md) files to produce, ensuring they are all written to the root, and what detailed content to include for each (instructions, data, results, and conclusions as appropriate). "
                "Disallow the creation of any subdirectories, folders, or nested paths. "
                "After delegating, review returned progress to ensure it contains the agent’s actual research findings and outputs, not just summaries.\n"
                "You are an autonomous agent; if you lack information or clarity, you must make reasonable assumptions and proceed. "
                "Never ask a follow-up question to the user.\n"
                "Drive the process step-by-step: assign tasks that produce meaningful research findings, explicit data, and detailed written outputs (in root-only files). "
                "Always provide structured, granular instructions for sub-agents based on the tools available and check that results include true research findings and analysis, not just a summary or file names.\n"
                "Keep the size of the task/research limited and small per agent assignment. For example, you could break it apart by sub topics or break the plan item into sub items."
                "Actively close information gaps: if key primary data is missing, instruct sub-agents to attempt additional targeted retrieval and document outcomes.\n"
                "If data remains unavailable after retrieval attempts, require explicit 'Insufficient Evidence' notes with precise missing items and impact.\n"
                "Reject and rework any high-level or generic output that lacks concrete support.\n\n"
                "Important: Ensure the findings and analysis are thorough, evidence-grounded, and professionally written."
            )

            # Mandatory planning for team workflows
            config.supervisor.enable_planning = True

            # Initialize supervisor with configuration
            self.supervisor = Supervisor(
                agents=agent_list,
                config=config.supervisor,
            )

            self.planner = PlannerAgent(
                config=config.planner,
            )

            self.report_writer: Optional[ReportWriter] = None
            if getattr(config, "report", None) and config.report.enabled:
                self.report_writer = ReportWriter(
                    report_format=config.report.format,
                    report_model_config=config.report.model,
                    supervisor_config=config.supervisor,
                    tmp_dir=tmp_dir,
                    output_mode=config.report.output_mode,
                    output_path=str(config.report.output_path)
                    if config.report.output_path
                    else None,
                    base_dir=config.report.base_dir,
                )

            # Build declarative workflow
            self.workflow = self._build_workflow()
        except Exception as e:
            # Attempt to cleanup the temporary directory if it was created.
            # Use the local tmp_dir reference and ignore cleanup errors so we don't mask
            # the original exception that caused initialization to fail.
            try:
                if tmp_dir is not None:
                    shutil.rmtree(tmp_dir)
            except (FileNotFoundError, AttributeError):
                # directory already gone or tmp_dir not a proper path-like object; ignore
                pass
            except Exception as cleanup_err:
                log_error(f"Unexpected error while cleaning up temporary directory during __init__: {cleanup_err}")
            raise e

    def begin(self, user_message: str) -> Dict[str, Any] | Any:
        try:
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "plan": [],
                "report_output": None,
            }
            result = self.workflow.execute(state=initial_state)
            if isinstance(result, dict):
                return result
            return initial_state
        finally:
            # Defensive cleanup: only attempt to remove tmp_dir if it exists on self.
            tmp = getattr(self, "tmp_dir", None)
            if tmp is not None:
                try:
                    shutil.rmtree(tmp)
                except (FileNotFoundError, AttributeError):
                    pass
                except Exception as cleanup_err:
                    log_error(f"Unexpected error while cleaning up temporary directory during begin(): {cleanup_err}")

    async def abegin(self, user_message: str) -> Dict[str, Any] | Any:
        try:
            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "plan": [],
                "report_output": None,
            }
            result = await self.workflow.aexecute(state=initial_state)
            if isinstance(result, dict):
                return result
            return initial_state
        finally:
            tmp = getattr(self, "tmp_dir", None)
            if tmp is not None:
                try:
                    shutil.rmtree(tmp)
                except (FileNotFoundError, AttributeError):
                    pass
                except Exception as cleanup_err:
                    log_error(f"Unexpected error while cleaning up temporary directory during abegin(): {cleanup_err}")

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------
    def _build_workflow(self) -> WorkflowBuilder:
        workflow = WorkflowBuilder(
            name="TeamWorkflow",
            checkpointer=self.config.supervisor.checkpointer,
        )
        chain = (
            workflow.start("planner")
            .run(self._planner_step)
            .then("supervisor")
            .run(self._supervisor_step)
        )

        # Attach report step only when reporting is enabled.
        if self.report_writer is not None:
            chain = chain.then("report_writer").run(self._report_writer_step)

        return workflow

    def _planner_step(self, state: Dict[str, Any]) -> Dict[str, Any] | Command:
        update = self.planner.plan_of_action(state)
        return self._merge_state(state, update)

    def _supervisor_step(self, state: Dict[str, Any]) -> Dict[str, Any] | Command:
        result = self.supervisor.supervise(state)
        return self._merge_state(state, result)

    def _report_writer_step(self, state: Dict[str, Any]) -> Dict[str, Any] | Command:
        result = self.report_writer.report_generator(state)
        return self._merge_state(state, result)

    def _merge_state(
        self, base_state: Dict[str, Any] | None, update: Any
    ) -> Dict[str, Any] | Command:
        if isinstance(update, Command):
            return update

        merged: Dict[str, Any] = dict(base_state or {})

        if update is None:
            return merged

        if isinstance(update, dict):
            for key, value in update.items():
                if key == "messages" and isinstance(value, list):
                    merged[key] = value
                elif key == "plan" and value is not None:
                    merged[key] = value
                elif key == "report_output" and value is not None:
                    merged[key] = value
                else:
                    merged[key] = value
            return merged

        if isinstance(update, list):
            merged["messages"] = update
            return merged

        return merged
