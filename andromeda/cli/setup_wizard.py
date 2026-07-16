"""Setup wizard for Andromeda CLI."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from andromeda.cli.config_generator import (
    generate_default_config,
    generate_interactive_config,
    tune_config_interactive,
)
from andromeda.cli.env_generator import generate_example_env
from andromeda.config.yaml_utils import yaml_dump
from andromeda.cli.helpers import (
    HAS_QUESTIONARY,
    ask_bool,
    console,
    slugify_identifier,
)
from andromeda.cli.prebuilt_templates import (
    PREBUILT_AGENT_TEMPLATES,
    apply_prebuilt_agent_templates,
)

# Use the availability flag from helpers; import questionary only when available
if HAS_QUESTIONARY:
    import questionary  # type: ignore


def _iter_agent_dicts(config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    agents_cfg = config_data.get("agents", [])
    if isinstance(agents_cfg, list):
        return [a for a in agents_cfg if isinstance(a, dict)]
    if isinstance(agents_cfg, dict):
        return [a for a in agents_cfg.values() if isinstance(a, dict)]
    return []


def _configure_optional_middleware(config_data: Dict[str, Any]) -> None:
    """Optionally configure LangChain middleware blocks for agents/supervisor."""
    if HAS_QUESTIONARY:
        enable = questionary.confirm(
            "Enable agent middleware in generated config?", default=False
        ).ask()
    else:
        enable = (
            input("Enable agent middleware in generated config? (y/N): ")
            .strip()
            .lower()
            in ["y", "yes"]
        )

    agents = _iter_agent_dicts(config_data)
    supervisor_cfg = config_data.get("supervisor")
    if not isinstance(supervisor_cfg, dict):
        supervisor_cfg = {}
        config_data["supervisor"] = supervisor_cfg

    if not enable:
        for agent_cfg in agents:
            agent_cfg.pop("middleware", None)
        supervisor_cfg.pop("middleware", None)
        return

    if HAS_QUESTIONARY:
        include_summarization = questionary.confirm(
            "Include SummarizationMiddleware?", default=False
        ).ask()
    else:
        include_summarization = (
            input("Include SummarizationMiddleware? (y/N): ")
            .strip()
            .lower()
            in ["y", "yes"]
        )

    trigger_tokens = 1000
    if include_summarization:
        if HAS_QUESTIONARY:
            raw = questionary.text(
                "Summarization trigger tokens", default="1000"
            ).ask()
        else:
            raw = input("Summarization trigger tokens (default: 1000): ").strip() or "1000"
        try:
            trigger_tokens = max(1, int(raw))
        except Exception:
            trigger_tokens = 1000

    if HAS_QUESTIONARY:
        include_hitl = questionary.confirm(
            "Include HumanInTheLoopMiddleware scaffold (disabled until interrupt map is filled)?",
            default=False,
        ).ask()
    else:
        include_hitl = (
            input(
                "Include HumanInTheLoopMiddleware scaffold (disabled until interrupt map is filled)? (y/N): "
            )
            .strip()
            .lower()
            in ["y", "yes"]
        )

    middleware_block: Dict[str, Any] = {
        "tool_error_handler": True,
        "guardrails": {"input": False, "output": False, "tool": False},
        "masking": {"input": False, "output": False, "tool": False},
        "custom": [],
    }
    if include_summarization:
        middleware_block["summarization"] = {
            "trigger_tokens": trigger_tokens,
        }
    if include_hitl:
        middleware_block["hitl"] = {"interrupt_on": {}}

    for agent_cfg in agents:
        agent_cfg["middleware"] = dict(middleware_block)
    supervisor_cfg["middleware"] = dict(middleware_block)


def run_setup_wizard():
    """Run the interactive setup wizard"""
    console.print("[bold cyan]Welcome to Andromeda Setup Wizard![/bold cyan]")
    console.print("This wizard will help you set up a new Andromeda project.\n")

    # Ask for project name
    if HAS_QUESTIONARY:
        project_name = questionary.text(
            "What's your project name?", default="my-andromeda-project"
        ).ask()
    else:
        project_name = (
            input("What's your project name? (default: my-andromeda-project): ").strip()
            or "my-andromeda-project"
        )

    if not project_name:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return

    # Choose project type with guidance
    console.print(
        "\n[bold cyan]Select project type:[/bold cyan]\n"
        "- [bold]Team[/bold]: Long-horizon tasks that benefit from planning and optional reporting.\n"
        "- [bold]Supervisor & Sub-Agents[/bold]: Concierge-style chatbot with specialist agents; great for high-context routing.\n"
        "- [bold]Custom Workflow[/bold]: Build everything from the ground up using WorkflowBuilder.\n"
    )

    if HAS_QUESTIONARY:
        project_type = questionary.select(
            "What kind of project would you like to set up?",
            choices=[
                "Team",
                "Supervisor & Sub-Agents",
                "Custom Workflow",
            ],
            default="Team",
        ).ask()
    else:
        print("1) Team\n2) Supervisor & Sub-Agents\n3) Custom Workflow")
        choice = (
            input("Enter choice (1/2/3, default: 1): ").strip()
            or "1"
        )
        mapping = {"1": "Team", "2": "Supervisor & Sub-Agents", "3": "Custom Workflow"}
        project_type = mapping.get(choice, "Team")

    # Normalize to simple identifier
    if project_type.startswith("Team"):
        project_type_id = "team"
    elif project_type.startswith("Supervisor"):
        project_type_id = "supervisor"
    else:
        project_type_id = "workflow"

    # Ask for project structure
    if HAS_QUESTIONARY:
        create_structure = questionary.confirm(
            f"Create project structure for '{project_name}'?", default=True
        ).ask()
    else:
        create_structure = input(
            f"Create project structure for '{project_name}'? (y/N): "
        ).strip().lower() in ["y", "yes"]

    if create_structure:
        create_project_structure(project_name)

    # Collect workflow nodes specification upfront for workflow-type projects.
    workflow_nodes: Optional[List[Dict[str, str]]] = None
    workflow_spec: Optional[Dict[str, Any]] = None
    if project_type_id == "workflow":
        workflow_nodes = collect_workflow_nodes()
        workflow_spec = collect_workflow_topology(workflow_nodes)

    # Ask about configuration
    if HAS_QUESTIONARY:
        create_config = questionary.confirm(
            "Generate configuration files?", default=True
        ).ask()
    else:
        create_config = input(
            "Generate configuration files? (Y/n): "
        ).strip().lower() not in ["n", "no"]

    config_generated = False
    if create_config and project_type_id != "workflow":
        # Ask about interactive config generation
        if HAS_QUESTIONARY:
            interactive_config = questionary.confirm(
                "Use interactive configuration generation?", default=False
            ).ask()
        else:
            interactive_config = input(
                "Use interactive configuration generation? (y/N): "
            ).strip().lower() in ["y", "yes"]

        if interactive_config:
            config_data = generate_interactive_config()
        else:
            config_data = generate_default_config()

        # Populate prebuilt agent templates when agents are named "prebuilt_*".
        config_data = apply_prebuilt_agent_templates(config_data)
        applied_prebuilts = []
        agents_cfg = config_data.get("agents", [])
        if isinstance(agents_cfg, list):
            for agent_cfg in agents_cfg:
                if isinstance(agent_cfg, dict):
                    name = str(agent_cfg.get("name", "")).strip()
                    if name in PREBUILT_AGENT_TEMPLATES:
                        applied_prebuilts.append(name)
        if applied_prebuilts:
            console.print(
                "[cyan]Applied prebuilt agent templates:[/cyan] "
                + ", ".join(applied_prebuilts)
            )

        # Optional toggles for validation and reporting (quick path)
        if HAS_QUESTIONARY:
            enable_validation = questionary.confirm(
                "Enable validation for agents and supervisor?", default=True
            ).ask()
        else:
            enable_validation = (
                input("Enable validation for agents and supervisor? (Y/n): ")
                .strip()
                .lower()
                not in ["n", "no"]
            )

        if not enable_validation:
            # Disable validation on agents (list-style)
            agents_cfg = config_data.get("agents", [])
            if isinstance(agents_cfg, list):
                for agent_cfg in agents_cfg:
                    if isinstance(agent_cfg, dict):
                        validation = agent_cfg.get("validation", {})
                        if isinstance(validation, dict):
                            validation["enabled"] = False
                            agent_cfg["validation"] = validation

            # Disable supervisor validation
            supervisor_cfg = config_data.get("supervisor", {})
            if isinstance(supervisor_cfg, dict):
                validation = supervisor_cfg.get("validation", {})
                if isinstance(validation, dict):
                    validation["enabled"] = False
                    supervisor_cfg["validation"] = validation

        if project_type_id == "team":
            # Reporting is optional for teams.
            if HAS_QUESTIONARY:
                enable_report = questionary.confirm(
                    "Enable report generation for this team?", default=False
                ).ask()
            else:
                enable_report = (
                    input("Enable report generation? (y/N): ")
                    .strip()
                    .lower()
                    in ["y", "yes"]
                )
            if not enable_report:
                # Explicitly disable report; model is only required when enabled.
                config_data["report"] = {"enabled": False}

        _configure_optional_middleware(config_data)

        # Optional advanced configuration of prompts, citations, planner, and report.
        if ask_bool(
            "Open advanced configuration (prompts, citations, planner, report)?",
            default=False,
        ):
            config_data = tune_config_interactive(config_data, project_type=project_type_id)

        Path(project_name).mkdir(parents=True, exist_ok=True)
        config_path = Path(project_name) / "config.yaml"
        with open(config_path, "w") as f:
            yaml_dump(config_data, f, default_flow_style=False, sort_keys=False)

        console.print(f"[green]✓[/green] Generated {config_path}")
        config_generated = True

    # Ask about environment
    if HAS_QUESTIONARY:
        create_env = questionary.confirm(
            "Generate .env.example file?", default=True
        ).ask()
    else:
        create_env = input(
            "Generate .env.example file? (Y/n): "
        ).strip().lower() not in ["n", "no"]

    if create_env:
        env_data = generate_example_env(include_pyeztrace=True, include_optional=True)
        Path(project_name).mkdir(parents=True, exist_ok=True)
        env_path = Path(project_name) / ".env.example"
        with open(env_path, "w") as f:
            for key, value in env_data.items():
                f.write(f"{key}={value}\n")

        console.print(f"[green]✓[/green] Generated {env_path}")

    # Ask about requirements
    if HAS_QUESTIONARY:
        create_requirements = questionary.confirm(
            "Create requirements.txt with Andromeda dependencies?", default=True
        ).ask()
    else:
        create_requirements = input(
            "Create requirements.txt with Andromeda dependencies? (Y/n): "
        ).strip().lower() not in ["n", "no"]

    if create_requirements:
        Path(project_name).mkdir(parents=True, exist_ok=True)
        requirements_path = Path(project_name) / "requirements.txt"
        with open(requirements_path, "w") as f:
            f.write("# Andromeda Framework Requirements\n")
            f.write("andromeda>=1.0.0\n")

        console.print(f"[green]✓[/green] Generated {requirements_path}")

    # Optionally generate a single quickstart file tailored to project type.
    if HAS_QUESTIONARY:
        create_quickstart = questionary.confirm(
            "Generate a quickstart script for this project?", default=True
        ).ask()
    else:
        create_quickstart = (
            input(
                "Generate a quickstart script for this project? (Y/n): "
            ).strip().lower()
            not in ["n", "no"]
        )

    if create_quickstart:
        create_quickstart_file(
            project_name,
            project_type_id,
            config_generated=config_generated,
            workflow_nodes=workflow_nodes,
            workflow_spec=workflow_spec,
        )

    console.print("[bold green]Setup complete![/bold green]")
    console.print(f"Your project '{project_name}' is ready to use.")
    console.print("\nNext steps:")
    console.print(f"  1. cd {project_name}")
    console.print("  2. cp .env.example .env")
    console.print("  3. Edit .env with your API keys")
    console.print("  4. Edit config.yaml for your needs")
    console.print("  5. Run your Andromeda application!")


def create_project_structure(project_name: str):
    """Create basic project structure"""
    project_path = Path(project_name)

    # Create directories
    directories = [
        "agents",
        "tools",
        "config",
        "data",
        "logs",
        "scripts",
        "tests",
        "docs",
    ]

    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)

    # Create basic files
    files = {
        "README.md": f"# {project_name}\n\nAndromeda created this empty project for you. This is intended to standardize project structure and provide a starting point for your project.\n",
        "pyproject.toml": f"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name.replace('-', '_')}"
version = "0.1.0"
description = "Andromeda multi-agent project"
dependencies = [
    "andromeda>=1.0.0"
]

[project.scripts]
{project_name.replace('-', '_')} = "{project_name.replace('-', '_')}.main:main"
""",
        ".gitignore": """__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.mypy_cache
.pytest_cache
.hypothesis/
.env
.DS_Store
""",
    }

    for file_path, content in files.items():
        (project_path / file_path).write_text(content)

    console.print(f"[green]✓[/green] Created project structure for '{project_name}'")

def collect_workflow_nodes() -> List[Dict[str, str]]:
    """Interactively collect workflow node definitions from the user."""

    console.print(
        "\n[bold cyan]Define your workflow steps:[/bold cyan]\n"
        "You will create a linear workflow where each step is a Python function.\n"
        "At minimum, define two steps (start and finish). You can add more as needed.\n"
    )

    nodes: List[Dict[str, str]] = []
    idx = 1
    while True:
        default_name = f"step_{idx}"
        if HAS_QUESTIONARY:
            raw_name = questionary.text(
                f"Name for step {idx} (Python identifier):", default=default_name
            ).ask()
            desc = questionary.text(
                f"Short description for step {idx}:", default="TODO: describe this step"
            ).ask()
        else:
            raw_name = (
                input(f"Name for step {idx} (Python identifier, default: {default_name}): ")
                .strip()
                or default_name
            )
            desc = (
                input(f"Short description for step {idx} (default: TODO): ").strip()
                or "TODO: describe this step"
            )

        fn_name = slugify_identifier(raw_name)
        nodes.append({"name": fn_name, "description": desc})

        if idx >= 2:
            if HAS_QUESTIONARY:
                more = questionary.confirm(
                    "Add another step to the workflow?", default=False
                ).ask()
            else:
                more = (
                    input("Add another step to the workflow? (y/N): ")
                    .strip()
                    .lower()
                    in ["y", "yes"]
                )
            if not more:
                break

        idx += 1

    return nodes


def collect_workflow_topology(
    workflow_nodes: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Ask how the user would like the workflow to be wired (linear/parallel/conditional).

    This does not attempt to cover every possible graph, but provides
    sensible skeletons that follow the idioms in ``tests/test_workflows*.py``.
    """

    step_names = [node["name"] for node in workflow_nodes if "name" in node]
    if len(step_names) < 2:
        # Not enough information to build anything more complex than linear.
        return {"pattern": "linear"}

    console.print(
        "\n[bold cyan]Workflow topology[/bold cyan]\n"
        "How should these steps be connected in the initial quickstart?\n"
        "- [bold]Linear[/bold]: run each step one after another.\n"
        "- [bold]Parallel fan-out[/bold]: run middle steps in parallel between a start and finish.\n"
        "- [bold]Conditional branch[/bold]: route from a branch step to different follow-ups on success/failure.\n"
    )

    if HAS_QUESTIONARY:
        choice = questionary.select(
            "Select workflow topology:",
            choices=[
                "Linear",
                "Parallel fan-out",
                "Conditional branch",
            ],
            default="Linear",
        ).ask()
    else:
        print("1) Linear\n2) Parallel fan-out\n3) Conditional branch")
        raw = input("Enter choice (1/2/3, default: 1): ").strip() or "1"
        mapping = {"1": "Linear", "2": "Parallel fan-out", "3": "Conditional branch"}
        choice = mapping.get(raw, "Linear")

    if not choice or choice.startswith("Linear"):
        return {"pattern": "linear"}

    if choice.startswith("Parallel"):
        # Use first step as pre-parallel, last as post-merge, and middles as branches.
        before = step_names[0]
        after = step_names[-1]
        parallel_steps = step_names[1:-1] or step_names
        return {
            "pattern": "parallel",
            "before": before,
            "after": after,
            "parallel": parallel_steps,
            "branch_name": "analytics",
        }

    # Conditional branch: start -> branch; on success go to last step; on failure go to previous step.
    if len(step_names) < 3:
        console.print(
            "[yellow]Not enough steps for a conditional pattern; falling back to linear.[/yellow]"
        )
        return {"pattern": "linear"}

    return {
        "pattern": "conditional",
        "start": step_names[0],
        "branch_step": step_names[1],
        "failure_step": step_names[-2],
        "success_step": step_names[-1],
    }


def create_quickstart_file(
    project_name: str,
    project_type: str,
    *,
    config_generated: bool,
    workflow_nodes: Optional[List[Dict[str, str]]] = None,
    workflow_spec: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate a single quickstart script tailored to the selected project type."""

    project_path = Path(project_name)
    project_path.mkdir(parents=True, exist_ok=True)
    module_name = project_name.replace("-", "_")

    if project_type == "team":
        if not config_generated:
            console.print(
                "[yellow]No config.yaml was generated; team quickstart will not work until you add one.[/yellow]"
            )

        content = f'''"""
Quickstart script for running a Team-based workflow with Andromeda.

Usage:
  python -m {module_name}.main
"""

from andromeda.config import AndromedaConfig
from andromeda.core.team import Team


def main() -> None:
    cfg = AndromedaConfig.load_from_file("config.yaml")
    team = Team(cfg)

    task = input("Enter a task for the team to work on: ")
    if not task:
        print("No task provided, exiting.")
        return

    state = team.begin(task)
    report = state.get("report_output")
    if report:
        print("\\n=== Report ===\\n")
        print(report)
    else:
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", last)
            print("\\n=== Response ===\\n")
            print(content)


if __name__ == "__main__":
    main()
'''
    elif project_type == "supervisor":
        if not config_generated:
            console.print(
                "[yellow]No config.yaml was generated; supervisor quickstart will not work until you add one.[/yellow]"
            )

        content = f'''"""
Quickstart script for running a Supervisor directly without the Team wrapper.

This is useful when you want a routing/orchestration layer over a set of
specialist agents but don't need separate planning/reporting.
"""

from andromeda.config import AndromedaConfig
from andromeda.core.supervisor import Supervisor
from andromeda import HumanMessage


def main() -> None:
    cfg = AndromedaConfig.load_from_file("config.yaml")

    # Normalize agents from config (dict or list) into a list of AgentConfig.
    raw_agents = cfg.agents
    if isinstance(raw_agents, dict):
        agents_cfg = list(raw_agents.values())
    else:
        agents_cfg = raw_agents

    supervisor = Supervisor(agents=agents_cfg, config=cfg.supervisor)

    print("Type an empty line to exit.\\n")
    while True:
        user_input = input("You: ")
        if not user_input:
            break

        state = {{"messages": [HumanMessage(content=user_input)], "plan": []}}
        result = supervisor.supervise(state)
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            content = getattr(last, "content", last)
            print(f"Supervisor: {{content}}\\n")


if __name__ == "__main__":
    main()
'''
    else:  # workflow
        # Ensure workflow directory exists for node placeholders.
        (project_path / "workflows").mkdir(parents=True, exist_ok=True)

        # Generate node placeholders in workflows/nodes.py
        nodes_spec = workflow_nodes or [
            {"name": "ingest", "description": "Ingest input and prepare state."},
            {"name": "process", "description": "Process state and produce output."},
        ]
        nodes_lines = [
            "from typing import Any, Dict",
            "",
            "# Auto-generated placeholders for workflow steps.",
            "",
        ]
        for node in nodes_spec:
            n = node["name"]
            d = node.get("description", "").replace('"""', '\\"\\"\\"')
            nodes_lines.append(f"def {n}(state: Dict[str, Any]) -> Dict[str, Any]:")
            if d:
                nodes_lines.append(f'    """{d}"""')
            else:
                nodes_lines.append('    """TODO: describe this step."""')
            nodes_lines.append("    # TODO: implement this step")
            nodes_lines.append("    return state")
            nodes_lines.append("")

        (project_path / "workflows" / "nodes.py").write_text("\n".join(nodes_lines).rstrip() + "\n")

        # Build WorkflowBuilder wiring using the defined nodes in order.
        step_names = [n["name"] for n in nodes_spec]
        first_step = step_names[0]
        last_step = step_names[-1]
        pattern = (workflow_spec or {}).get("pattern", "linear")

        builder_lines: List[str] = []
        builder_lines.append(
            '    wf = WorkflowBuilder(name="QuickstartWorkflow", state_schema=QuickstartState)'
        )

        if pattern == "parallel" and len(step_names) >= 2:
            before = (workflow_spec or {}).get("before", first_step)
            after = (workflow_spec or {}).get("after", last_step)
            parallel_steps = (workflow_spec or {}).get(
                "parallel", step_names[1:-1] or step_names
            )
            branch_name = (workflow_spec or {}).get("branch_name", "analytics")

            builder_lines.append(
                f'    chain = wf.start("{before}").run(wf_nodes.{before})'
            )
            builder_lines.append(f'    chain = chain.branch("{branch_name}").parallel([')
            for p_step in parallel_steps:
                builder_lines.append(f'        ("{p_step}", wf_nodes.{p_step}),')
            builder_lines.append("    ])")
            builder_lines.append("    chain = chain.merge_results()")
            builder_lines.append(
                f'    chain = chain.finish("{after}").run(wf_nodes.{after})'
            )
            builder_lines.append("    return wf")
        elif pattern == "conditional" and len(step_names) >= 3:
            start = (workflow_spec or {}).get("start", first_step)
            branch_step = (workflow_spec or {}).get("branch_step", step_names[1])
            failure_step = (workflow_spec or {}).get("failure_step", step_names[-2])
            success_step = (workflow_spec or {}).get("success_step", last_step)

            builder_lines.append(
                f'    chain = wf.start("{start}").run(wf_nodes.{start})'
            )
            builder_lines.append(
                f'    chain = chain.then("{branch_step}").run(wf_nodes.{branch_step})'
            )
            builder_lines.append(
                f'    chain = chain.if_succeeds().goto("{success_step}")'
            )
            builder_lines.append(
                f'    chain = chain.if_fails().goto("{failure_step}")'
            )
            if failure_step != success_step:
                builder_lines.append(
                    f'    chain = chain.then("{failure_step}").run(wf_nodes.{failure_step})'
                )
            builder_lines.append(
                f'    chain = chain.finish("{success_step}").run(wf_nodes.{success_step})'
            )
            builder_lines.append("    return wf")
        else:
            # Default linear chain
            builder_lines.append(
                f'    chain = wf.start("{first_step}").run(wf_nodes.{first_step})'
            )
            for step in step_names[1:-1]:
                builder_lines.append(
                    f'    chain = chain.then("{step}").run(wf_nodes.{step})'
                )
            builder_lines.append(
                f'    chain = chain.finish("{last_step}").run(wf_nodes.{last_step})'
            )
            builder_lines.append("    return wf")

        workflow_builder_code = "\n".join(builder_lines)

        content = f'''"""
Quickstart skeleton for building a custom Workflow with Andromeda.

Edit the steps to match your use case, then run:
  python main.py
"""

from typing import Any, Dict, TypedDict

from andromeda.core.workflow import WorkflowBuilder
from workflows import nodes as wf_nodes


class QuickstartState(TypedDict):
    """Shape of the workflow state.

    Extend this with strongly-typed keys as your workflow evolves.
    """
    pass


def build_workflow() -> WorkflowBuilder:
{workflow_builder_code}


def main() -> None:
    workflow = build_workflow()
    user_query = input("Enter an input for the workflow (e.g., topic or query): ") or "example topic"
    state = workflow.execute(state={{"query": user_query}}) # this has to be updated to reflect your state and nodes
    print("Final state:", state)


if __name__ == "__main__":
    main()
'''

    (project_path / "main.py").write_text(content)
    console.print(f"[green]✓[/green] Generated quickstart script 'main.py' in '{project_name}'")
