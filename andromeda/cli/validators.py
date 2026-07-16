"""Validation and diagnostics for Andromeda CLI."""

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from andromeda.config.config import AndromedaConfig
from andromeda.config.yaml_utils import yaml_load
from andromeda.cli.helpers import console, discover_config_files

def run_tool_diagnostics() -> None:
    """Inspect the global tool registry and how config files reference tools."""

    # Import built-in tools so they self-register with the Toolkit.
    try:  # noqa: F401
        import andromeda.tools.tools as _builtin_tools  # type: ignore[unused-import]
    except ImportError:
        _builtin_tools = None  # type: ignore[assignment]

    try:
        from andromeda.tools.toolkit import get_default_toolkit
    except ImportError as exc:
        console.print(
            f"[red]✗[/red] Unable to import tool registry (andromeda.tools.toolkit): {exc}"
        )
        return

    toolkit = get_default_toolkit()
    tools = toolkit.all()

    if not tools:
        console.print(
            "[yellow]No tools are currently registered in the Toolkit.[/yellow]"
        )
        console.print(
            "  Import [bold]andromeda.tools.tools[/bold] at startup or register "
            "custom tools via [bold]register_tool[/bold]."
        )
    else:
        names = sorted(tools.keys())
        preview = ", ".join(names[:10])
        suffix = " ..." if len(names) > 10 else ""
        console.print(
            f"[green]✓[/green] {len(names)} tools registered in the global Toolkit."
        )
        if names:
            console.print(f"  Tools: {preview}{suffix}")

    # Cross-check with any config files in the current directory.
    config_files = discover_config_files()
    if not config_files:
        console.print(
            "No configuration files found to inspect tool wiring "
            "(run [bold]andromeda generate-config[/bold] or [bold]andromeda setup[/bold])."
        )
        return

    def _collect_tool_specs(config_data: Dict[str, Any]) -> List[Any]:
        specs: List[Any] = []

        def _from_agent(agent_cfg: Dict[str, Any]) -> None:
            tools_field = agent_cfg.get("tools")
            if isinstance(tools_field, list):
                for item in tools_field:
                    specs.append(item)

        agents = config_data.get("agents")
        if isinstance(agents, list):
            for agent_cfg in agents:
                if isinstance(agent_cfg, dict):
                    _from_agent(agent_cfg)
        elif isinstance(agents, dict):
            for agent_cfg in agents.values():
                if isinstance(agent_cfg, dict):
                    _from_agent(agent_cfg)

        supervisor_cfg = config_data.get("supervisor")
        if isinstance(supervisor_cfg, dict):
            tools_field = supervisor_cfg.get("tools")
            if isinstance(tools_field, list):
                specs.extend(tools_field)

        return specs

    import json as _json

    for path in config_files:
        console.print(f"\n[bold]Tools referenced in config:[/bold] {path}")
        try:
            if path.suffix in (".yaml", ".yml"):
                with open(path) as f:
                    config_data = yaml_load(f) or {}
            else:
                with open(path) as f:
                    config_data = _json.load(f) or {}
        except FileNotFoundError as exc:
            console.print(f"  [red]✗[/red] File not found: {exc}")
            continue
        except PermissionError as exc:
            console.print(f"  [red]✗[/red] Permission error: {exc}")
            continue
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            console.print(f"  [red]✗[/red] YAML parsing error: {exc}")
            continue
        except _json.JSONDecodeError as exc:  # type: ignore[attr-defined]
            console.print(f"  [red]✗[/red] JSON parsing error at pos {exc.pos}: {exc.msg}")
            continue

        tool_specs = _collect_tool_specs(config_data)
        if not tool_specs:
            console.print("  [yellow]No tools referenced in this config.[/yellow]")
            continue

        for spec in tool_specs:
            if isinstance(spec, str):
                if spec in tools:
                    console.print(f"  [green]✓[/green] Tool '{spec}' is registered.")
                else:
                    console.print(
                        f"  [yellow]•[/yellow] Tool name '{spec}' is not registered in the Toolkit "
                        "(may be a custom tool or missing registration)."
                    )
            else:
                console.print(
                    f"  [yellow]•[/yellow] Non-string tool spec {spec!r} found; "
                    "ensure it maps to a registered LangChain tool instance."
                )


def validate_configuration_file(config_file: str, format: str = None) -> Dict[str, Any]:
    """Validate a configuration file"""
    try:
        path = Path(config_file)

        # Determine format if not provided
        if not format:
            if path.suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif path.suffix == ".json":
                format = "json"
            else:
                return {
                    "valid": False,
                    "errors": [f"Unsupported file format: {path.suffix}"],
                    "config": None,
                }

        # Load configuration
        if format == "yaml":
            with open(path) as f:
                config_data = yaml_load(f)
        else:  # json
            with open(path) as f:
                config_data = json.load(f)

        if config_data is None:
            return {
                "valid": False,
                "errors": ["Configuration file is empty or invalid"],
                "config": None,
            }

        # Validate against expected structure
        errors = []

        required_sections = ["agents", "supervisor", "planner"]
        for section in required_sections:
            if section not in config_data:
                errors.append(f"Missing required section: '{section}'")
            else:
                console.print(f"  [green]✓[/green] Section '{section}' found")

        # Validate agents (allow list or mapping)
        if "agents" in config_data:
            agents = config_data["agents"]
            if isinstance(agents, list):
                iterable = list(enumerate(agents))
            elif isinstance(agents, dict):
                iterable = list(agents.items())
            else:
                errors.append("'agents' must be a list or a mapping")
                iterable = []

            for key, agent in iterable:
                if not isinstance(agent, dict):
                    errors.append(f"Agent {key} must be a dictionary")
                    continue

                required_fields = ["name", "model"]
                for field in required_fields:
                    if field not in agent:
                        errors.append(
                            f"Agent {key} missing required field: '{field}'"
                        )

        # Validate model configurations (basic check for required fields on models)
        model_sections = ["agents", "supervisor", "planner"]
        for section in model_sections:
            if section not in config_data:
                continue

            section_data = config_data[section]
            items: List[tuple[Any, Any]] = []

            if section == "agents":
                if isinstance(section_data, list):
                    items = list(enumerate(section_data))
                elif isinstance(section_data, dict):
                    items = list(section_data.items())
            else:
                items = [("", section_data)]

            for key, entry in items:
                if not isinstance(entry, dict) or "model" not in entry:
                    continue

                model_config = entry["model"]
                if not isinstance(model_config, dict):
                    continue

                required_model_fields = ["name", "provider"]
                for field in required_model_fields:
                    if field not in model_config:
                        errors.append(
                            f"{section}[{key}].model missing field: '{field}'"
                            if key != ""
                            else f"{section}.model missing field: '{field}'"
                        )

        return {"valid": len(errors) == 0, "errors": errors, "config": config_data}
    except FileNotFoundError as exc:
        return {"valid": False, "errors": [f"File not found: {exc}"], "config": None}
    except PermissionError as exc:
        return {"valid": False, "errors": [f"Permission error: {exc}"], "config": None}
    except yaml.YAMLError as exc:  # type: ignore[attr-defined]
        return {"valid": False, "errors": [f"YAML parsing error: {exc}"], "config": None}
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "errors": [f"JSON parsing error at pos {exc.pos}: {exc.msg}"],
            "config": None,
        }


def run_config_diagnostics() -> None:
    """Run structural + schema-based diagnostics for configuration files."""

    config_files = discover_config_files()
    if not config_files:
        console.print(
            "[yellow]No configuration files found in current directory.[/yellow]"
        )
        console.print(
            "Use [bold]andromeda generate-config[/bold] or "
            "[bold]andromeda setup[/bold] to create one."
        )
        return

    # Ensure built-in tools are registered so AndromedaConfig.load_from_file
    # can resolve tool names referenced in config.
    try:  # noqa: F401
        import andromeda.tools.tools as _builtin_tools  # type: ignore[unused-import]
    except ImportError:
        _builtin_tools = None  # type: ignore[assignment]

    for path in config_files:
        console.print(f"\n[bold]File:[/bold] {path}")

        # Structural validation
        result = validate_configuration_file(str(path))
        errors = list(result["errors"])
        if errors:
            console.print("  [red]✗ Structural issues detected:[/red]")
            for err in errors:
                console.print(f"    [red]•[/red] {err}")
        else:
            console.print("  [green]✓[/green] Basic structure looks valid.")

        # Schema + tool resolution via AndromedaConfig
        try:
            cfg = AndromedaConfig.load_from_file(str(path))
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            console.print("  [red]✗[/red] Failed to parse as AndromedaConfig.")
            console.print(f"    {exc}")
            continue

        agents_cfg = cfg.agents
        if isinstance(agents_cfg, dict):
            agent_count = len(agents_cfg)
        else:
            agent_count = len(agents_cfg)

        console.print(
            "  [green]✓[/green] Parsed as AndromedaConfig "
            f"({agent_count} agents, supervisor='{cfg.supervisor.name}')"
        )

        # High-level hints about optional components.
        if getattr(cfg.report, "enabled", False):
            console.print("  [green]✓[/green] Reporting is [bold]enabled[/bold].")
        else:
            console.print("  [yellow]•[/yellow] Reporting is currently disabled.")

        if getattr(cfg.supervisor, "enable_planning", False):
            console.print("  [green]✓[/green] Supervisor planning is enabled.")
        else:
            console.print("  [yellow]•[/yellow] Supervisor planning is disabled.")




def display_config_summary(config: Dict[str, Any]):
    """Display a summary of the configuration"""
    if not config:
        return

    console.print("\n[bold cyan]Configuration Summary:[/bold cyan]")

    # Count agents (support both list and mapping styles)
    if "agents" in config:
        agents = config["agents"]
        if isinstance(agents, list):
            count = len(agents)
        elif isinstance(agents, dict):
            count = len(agents)
        else:
            count = 0
        console.print(f"  [green]Agents:[/green] {count} configured")

    # Show supervisor status
    if "supervisor" in config:
        console.print("  [green]Supervisor:[/green] Configured")

    # Show planner status
    if "planner" in config:
        console.print("  [green]Planner:[/green] Configured")

    # Show report status
    if "report" in config:
        console.print("  [green]Report:[/green] Configured")
