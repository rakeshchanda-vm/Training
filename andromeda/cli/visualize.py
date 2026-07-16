"""Modular visualization system for Andromeda workflows.

Supports visualization of Team, Supervisor, and Custom Workflow structures.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from andromeda.config.config import AndromedaConfig


def _sanitize_id(raw: str) -> str:
    """Sanitize a label into a Mermaid-safe node id.

    Ensures an initial letter and allows only [A-Za-z0-9_.-].
    """
    import re

    # Replace spaces with underscores for readability
    s = re.sub(r"\s+", "_", str(raw))
    # Remove invalid characters
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    # Ensure starts with a letter
    if not s or not s[0].isalpha():
        s = f"N_{s}" if s else "N"
    return s


def infer_source_kind(path: Path) -> str:
    """Classify an input path as config, python, or directory.

    This is used by the CLI to decide how to interpret the user-provided path.
    """
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        return "config"
    if suffix == ".py":
        return "python"
    return "unknown"

def detect_workflow_type(config: AndromedaConfig, config_file: Optional[str] = None) -> Optional[str]:
    """Detect workflow type from config or project structure.
    
    Args:
        config: The loaded AndromedaConfig
        config_file: Optional path to config file or project root for detecting workflows
        
    Returns:
        'team', 'supervisor', 'workflow', or None if cannot determine
    """
    project_root: Optional[Path] = None
    if config_file:
        base_path = Path(config_file)
        project_root = base_path.parent if base_path.is_file() else base_path
    
    # Inspect project Python files (if available) for explicit usage patterns first.
    # This respects the user-provided config location and does not try to guess
    # unrelated files elsewhere on disk.
    if project_root:
        workflow_files = []
        for pattern in ["workflows/*.py", "main.py", "*workflow*.py"]:
            workflow_files.extend(project_root.rglob(pattern))
        
        for wf_file in workflow_files:
            try:
                content = wf_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            
            # Detect explicit Team-based orchestration
            if (
                "from andromeda.core.team import Team" in content
                or "from andromeda.core import Team" in content
                or "Team(" in content
            ):
                return "team"
            
            # Detect custom WorkflowBuilder-based workflows
            if "WorkflowBuilder" in content and (
                "from andromeda.core.workflow import WorkflowBuilder" in content
                or "from andromeda.core import WorkflowBuilder" in content
            ):
                return "workflow"
    
    # Config-based hints: enabled reporting implies a Team workflow.
    if hasattr(config, "report") and getattr(config.report, "enabled", False):
        return "team"
    
    # Default to supervisor-style routing when we have agents and a supervisor.
    if hasattr(config, "agents") and hasattr(config, "supervisor"):
        agents = config.agents
        if isinstance(agents, dict) and len(agents) > 0:
            return "supervisor"
        elif isinstance(agents, list) and len(agents) > 0:
            return "supervisor"
    
    return None


def detect_python_workflow_type(py_file: Path) -> Optional[str]:
    """Detect workflow type from a Python entrypoint.

    Heuristics:
        - Imports/usage of Team -> 'team'
        - Imports/usage of WorkflowBuilder -> 'workflow'
        - Imports/usage of Supervisor -> 'supervisor'
    """
    try:
        source = py_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    # Team-based orchestration
    if (
        "from andromeda.core.team import Team" in source
        or "from andromeda.core import Team" in source
        or " Team(" in source
    ):
        return "team"

    # Custom WorkflowBuilder-based workflow
    if "WorkflowBuilder" in source and (
            "from andromeda.core.workflow import WorkflowBuilder" in source
            or "from andromeda.core import WorkflowBuilder" in source
        ):
        return "workflow"

    # Direct Supervisor orchestration
    if (
        "from andromeda.core.supervisor import Supervisor" in source
        or "from andromeda.core import Supervisor" in source
        or " Supervisor(" in source
    ):
        return "supervisor"

    return None


def extract_config_path_from_python(py_file: Path) -> Optional[Path]:
    """Best-effort extraction of a config file path from a Python entrypoint.

    Looks for patterns like:
        AndromedaConfig.load_from_file("config.yaml")

    Returns an absolute Path if one can be resolved and appears to exist.
    """
    import re

    try:
        source = py_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    pattern = r"AndromedaConfig\.load_from_file\(\s*['\"]([^'\"]+)['\"]\s*\)"
    match = re.search(pattern, source)
    if not match:
        return None

    rel_path = match.group(1)
    candidate = (py_file.parent / rel_path).resolve()
    if candidate.exists():
        return candidate
    return None


def visualize_team_workflow(config: AndromedaConfig) -> str:
    """Generate mermaid diagram for Team workflow.
    
    Team workflow: Planner -> Supervisor -> (Agents) -> Reporter (optional)
    
    Args:
        config: The AndromedaConfig for the team
        
    Returns:
        Mermaid diagram code as string
    """
    lines = ["graph TD"]
    
    # Start node
    lines.append("    Start([Start])")
    
    # Planner node
    planner_cfg = getattr(config, "planner", None)
    planner_label = "Planner"
    planner_model = getattr(getattr(planner_cfg, "model", None), "name", "unknown")
    planner_id = _sanitize_id(planner_label)
    lines.append(f'    {planner_id}["{planner_label}<br/>{planner_model}"]')
    lines.append(f"    Start --> {planner_id}")
    
    # Supervisor node
    supervisor_cfg = getattr(config, "supervisor", None)
    supervisor_label = getattr(supervisor_cfg, "name", "Supervisor")
    supervisor_model = getattr(getattr(supervisor_cfg, "model", None), "name", "unknown")
    supervisor_id = _sanitize_id(supervisor_label)
    lines.append(f'    {supervisor_id}["{supervisor_label}<br/>{supervisor_model}"]')
    lines.append(f"    {planner_id} --> {supervisor_id}")
    
    # Agents (as a subgraph)
    agents = getattr(config, "agents", [])
    if isinstance(agents, dict):
        agent_list = list(agents.values())
    else:
        agent_list = agents
    
    if agent_list:
        lines.append("    subgraph Agents[Specialist Agents]")
        for agent_cfg in agent_list:
            agent_label = getattr(agent_cfg, "name", "agent")
            agent_model = getattr(getattr(agent_cfg, "model", None), "name", "unknown")
            node_id = _sanitize_id(agent_label)
            lines.append(f'        {node_id}["{agent_label}<br/>{agent_model}"]')
            lines.append(f"        {supervisor_id} -.->|routes| {node_id}")
            lines.append(f"        {node_id} -.->|returns| {supervisor_id}")
        lines.append("    end")
    
    # Reporter (optional)
    if hasattr(config, "report") and getattr(config.report, "enabled", False):
        reporter_name = "Reporter"
        reporter_model = getattr(config.report.model, "name", "unknown") if hasattr(config.report, "model") and config.report.model else "unknown"
        lines.append(f'    {reporter_name}["{reporter_name}<br/>{reporter_model}"]')
        lines.append(f"    {supervisor_id} --> {reporter_name}")
        lines.append(f"    {reporter_name} --> End([End])")
    else:
        lines.append("    End([End])")
        lines.append(f"    {supervisor_id} --> End")
    
    # Styling
    lines.append("    classDef planner fill:#e1f5ff,stroke:#01579b,stroke-width:2px")
    lines.append("    classDef supervisor fill:#fff3e0,stroke:#e65100,stroke-width:2px")
    lines.append("    classDef agent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px")
    lines.append("    classDef reporter fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px")
    lines.append(f"    class {planner_id} planner")
    lines.append(f"    class {supervisor_id} supervisor")
    if agent_list:
        for agent_cfg in agent_list:
            node_name = getattr(agent_cfg, "name", "agent").replace(" ", "_").replace("-", "_")
            lines.append(f"    class {node_name} agent")
    if hasattr(config, "report") and getattr(config.report, "enabled", False):
        lines.append(f"    class {reporter_name} reporter")
    
    return "\n".join(lines)


def visualize_supervisor_workflow(config: AndromedaConfig) -> str:
    """Generate mermaid diagram for Supervisor workflow.
    
    Supervisor workflow: Supervisor -> (Agents)
    
    Args:
        config: The AndromedaConfig for the supervisor
        
    Returns:
        Mermaid diagram code as string
    """
    lines = ["graph TD"]
    
    # Start node
    lines.append("    Start([Start])")
    
    # Supervisor node
    supervisor_cfg = getattr(config, "supervisor", None)
    supervisor_label = getattr(supervisor_cfg, "name", "Supervisor")
    supervisor_model = getattr(getattr(supervisor_cfg, "model", None), "name", "unknown")
    planning_enabled = getattr(supervisor_cfg, "enable_planning", False)
    planning_label = " (with planning)" if planning_enabled else ""
    supervisor_id = _sanitize_id(supervisor_label)
    lines.append(f'    {supervisor_id}["{supervisor_label}{planning_label}<br/>{supervisor_model}"]')
    lines.append(f"    Start --> {supervisor_id}")
    
    # Agents
    agents = getattr(config, "agents", [])
    if isinstance(agents, dict):
        agent_list = list(agents.values())
    else:
        agent_list = agents
    
    if agent_list:
        lines.append("    subgraph Agents[Specialist Agents]")
        for agent_cfg in agent_list:
            agent_name = getattr(agent_cfg, "name", "agent")
            agent_model = getattr(agent_cfg.model, "name", "unknown") if hasattr(agent_cfg, "model") else "unknown"
            # Get tools count
            tools = getattr(agent_cfg, "tools", [])
            tools_count = len(tools) if isinstance(tools, list) else 0
            tools_label = f" ({tools_count} tools)" if tools_count > 0 else ""
            
            node_id = _sanitize_id(agent_name)
            lines.append(f'        {node_id}["{agent_name}{tools_label}<br/>{agent_model}"]')
            lines.append(f"        {supervisor_id} -.->|routes| {node_id}")
            lines.append(f"        {node_id} -.->|returns| {supervisor_id}")
        lines.append("    end")
    
    # End node
    lines.append("    End([End])")
    lines.append(f"    {supervisor_id} --> End")
    
    # Styling
    lines.append("    classDef supervisor fill:#fff3e0,stroke:#e65100,stroke-width:2px")
    lines.append("    classDef agent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px")
    lines.append(f"    class {supervisor_id} supervisor")
    if agent_list:
        for agent_cfg in agent_list:
            node_id = _sanitize_id(getattr(agent_cfg, "name", "agent"))
            lines.append(f"    class {node_id} agent")
    
    return "\n".join(lines)


def visualize_custom_workflow(config_file: str) -> str:
    """Generate mermaid diagram for Custom Workflow.
    
    Attempts to parse workflow structure from Python files in the project.
    
    Args:
        config_file: Path to config file (used to find workflow files)
        
    Returns:
        Mermaid diagram code as string
    """
    base_path = Path(config_file)
    # Treat either a config file path or a project root directory uniformly.
    config_path = base_path.parent if base_path.is_file() else base_path
    
    # Look for workflow files
    workflow_files = []
    for pattern in ["workflows/*.py", "main.py", "*workflow*.py"]:
        workflow_files.extend(config_path.rglob(pattern))
    
    if not workflow_files:
        # Fallback: simple linear workflow diagram
        return _generate_simple_workflow_diagram()
    
    # Try to parse workflow structure from main.py or workflow files
    for wf_file in workflow_files:
        try:
            content = wf_file.read_text(encoding="utf-8")
            if "WorkflowBuilder" in content:
                # Try to extract workflow steps
                steps = _extract_workflow_steps(content)
                if steps:
                    return _generate_workflow_diagram_from_steps(steps)
        except (OSError, UnicodeDecodeError):
            continue
    
    # Fallback
    return _generate_simple_workflow_diagram()


def _extract_workflow_steps(content: str) -> List[Dict[str, Any]]:
    """Extract workflow step information from Python code.
    
    Looks for patterns like:
    - wf.start("step1")
    - chain.then("step2")
    - chain.finish("step3")
    - wf.branch("branch_name").parallel([...])
    
    Args:
        content: Python source code
        
    Returns:
        List of step dictionaries with name, type, and connections
    """
    import re
    
    steps = []
    
    # Pattern for start/then/finish
    step_pattern = r'\.(start|then|finish|branch)\(["\']([^"\']+)["\']\)'
    matches = re.findall(step_pattern, content)
    
    for step_type, step_name in matches:
        steps.append({
            "name": step_name,
            "type": step_type,
            "kind": "task" if step_type in ["start", "then", "finish"] else "branch"
        })
    
    # Pattern for parallel branches
    parallel_pattern = r'\.parallel\(\[(.*?)\]\)'
    parallel_matches = re.findall(parallel_pattern, content, re.DOTALL)
    for match in parallel_matches:
        # Extract tuple pairs like ("name", func)
        tuple_pattern = r'\("([^"]+)",\s*([^)]+)\)'
        branch_matches = re.findall(tuple_pattern, match)
        for branch_name, _ in branch_matches:
            steps.append({
                "name": branch_name,
                "type": "parallel",
                "kind": "parallel"
            })
    
    return steps


def _generate_workflow_diagram_from_steps(steps: List[Dict[str, Any]]) -> str:
    """Generate mermaid diagram from extracted workflow steps.
    
    Args:
        steps: List of step dictionaries
        
    Returns:
        Mermaid diagram code
    """
    if not steps:
        return _generate_simple_workflow_diagram()
    
    lines = ["graph TD"]
    lines.append("    Start([Start])")
    
    prev_node = "Start"
    parallel_groups = []
    current_parallel = None
    
    for i, step in enumerate(steps):
        step_name = step["name"].replace(" ", "_").replace("-", "_")
        step_display = step["name"]
        step_type = step.get("type", "task")
        step_kind = step.get("kind", "task")
        
        if step_kind == "parallel" and current_parallel is None:
            # Start parallel group
            current_parallel = []
            parallel_groups.append(current_parallel)
        elif step_kind != "parallel" and current_parallel is not None:
            # End parallel group
            if current_parallel:
                # Connect all parallel branches to next step
                for branch in current_parallel:
                    lines.append(f"    {branch} --> {step_name}")
            current_parallel = None
        
        if step_kind == "parallel":
            if current_parallel is not None:
                current_parallel.append(step_name)
                lines.append(f'    {step_name}["{step_display}"]')
                lines.append(f"    {prev_node} --> {step_name}")
        else:
            lines.append(f'    {step_name}["{step_display}"]')
            if step_type == "start":
                lines.append(f"    {prev_node} --> {step_name}")
            elif step_type == "then":
                lines.append(f"    {prev_node} --> {step_name}")
            elif step_type == "finish":
                lines.append(f"    {prev_node} --> {step_name}")
                lines.append(f"    {step_name} --> End([End])")
                prev_node = "End"
                continue
            elif step_type == "branch":
                lines.append(f"    {prev_node} --> {step_name}")
            
            prev_node = step_name
    
    # Connect any remaining parallel groups
    if current_parallel and prev_node != "End":
        for branch in current_parallel:
            lines.append(f"    {branch} --> End([End])")
    elif prev_node != "End":
        lines.append(f"    {prev_node} --> End([End])")
    
    # Styling
    lines.append("    classDef step fill:#e3f2fd,stroke:#1565c0,stroke-width:2px")
    lines.append("    classDef branch fill:#fff9c4,stroke:#f57f17,stroke-width:2px")
    for step in steps:
        step_name = step["name"].replace(" ", "_").replace("-", "_")
        if step.get("kind") == "branch":
            lines.append(f"    class {step_name} branch")
        else:
            lines.append(f"    class {step_name} step")
    
    return "\n".join(lines)


def _generate_simple_workflow_diagram() -> str:
    """Generate a simple placeholder workflow diagram.
    
    Returns:
        Mermaid diagram code
    """
    return """graph TD
    Start([Start])
    Step1["Step 1"]
    Step2["Step 2"]
    Step3["Step 3"]
    End([End])
    
    Start --> Step1
    Step1 --> Step2
    Step2 --> Step3
    Step3 --> End
    
    classDef step fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    class Step1,Step2,Step3 step
"""
