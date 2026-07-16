"""CLI command definitions for Andromeda."""

import json
from contextlib import suppress
import sys
import os
from pathlib import Path
from typing import Any, Optional

import click
from rich.panel import Panel
from rich.table import Table

from andromeda.config.config import AndromedaConfig
from andromeda.config.yaml_utils import yaml_dump
from andromeda.cli.config_generator import generate_example_config
from andromeda.cli.env_generator import generate_example_env
from andromeda.cli.validators import (
    validate_configuration_file,
    run_config_diagnostics,
    run_tool_diagnostics,
    display_config_summary,
)
from andromeda.cli.display import (
    display_config_options_help,
    display_env_vars_help,
    display_env_table,
)
from andromeda.cli.diagnostics import (
    check_dependencies,
    test_service_connections,
    check_environment_setup,
)
from andromeda.cli.setup_wizard import run_setup_wizard
from andromeda.cli.helpers import console, discover_config_files
from andromeda.cli.visualize_helpers import mermaid_to_text, render_mermaid_to_image
from andromeda import BaseMessage
from andromeda.runtime import AndromedaRuntime
from andromeda.runtime.json_utils import (
    message_content,
    to_json_compatible as _to_json_compatible,
)
from andromeda.runtime.validation import RunnableAmbiguousError, RunnableNotFoundError


_BASE_MESSAGE_TYPE = BaseMessage if isinstance(BaseMessage, type) else ()


def _load_runtime_inputs(
    value: str | None = None,
    file_path: Path | None = None,
) -> dict[str, Any]:
    if value is None and file_path is None:
        return {}
    if value is not None and file_path is not None:
        raise click.BadParameter("Use either --input or --input-file, not both")

    if value is None and file_path is not None:
        if not file_path.exists():
            raise click.BadParameter(f"Input file not found: {file_path}")
        value = file_path.read_text(encoding="utf-8")

    assert value is not None
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise click.BadParameter(f"Invalid JSON input: {exc}") from exc

    if not isinstance(parsed, dict):
        raise click.BadParameter("Input must be a JSON object")
    return parsed


def _build_runtime(
    root: str | None,
    *,
    include_global: bool,
    global_root: str | None,
) -> AndromedaRuntime:
    return AndromedaRuntime.discover(
        root=root,
        include_global=include_global,
        global_root=global_root,
    )


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(_to_json_compatible(value), **kwargs)


def _render_stream_chunk(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, (dict, list, tuple)):
        return _json_dumps(chunk, indent=2)
    if _BASE_MESSAGE_TYPE and isinstance(chunk, _BASE_MESSAGE_TYPE):
        return _json_dumps(chunk, indent=2)
    return str(chunk)


def _close_runtime(runtime: Any) -> None:
    closer = getattr(runtime, "close", None)
    if callable(closer):
        with suppress(Exception):
            closer()


def register_commands(cli):
    """Register all CLI commands with the click group."""
    
    @cli.command()
    @click.option(
        "--format",
        "-f",
        type=click.Choice(["yaml", "json", "both"]),
        default="yaml",
        help="Output format for config files",
    )
    @click.option(
        "--output-dir",
        "-o",
        type=click.Path(),
        default=".",
        help="Output directory for generated files",
    )
    @click.option(
        "--interactive",
        "-i",
        is_flag=True,
        help="Use interactive prompts to customize configuration",
    )
    def generate_config(format, output_dir, interactive):
        """Generate example configuration files (config.yaml and/or config.json)

        This command creates comprehensive example configuration files that demonstrate
        all available configuration options for the Andromeda framework.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Configuration Generator[/bold blue]",
                subtitle="Generate example configuration files",
            )
        )

        # Generate configuration data
        config_data = generate_example_config(interactive)

        output_path = Path(output_dir)

        if format in ["yaml", "both"]:
            yaml_path = output_path / "config.example.yaml"
            with open(yaml_path, "w") as f:
                yaml_dump(config_data, f, default_flow_style=False, sort_keys=False)
            console.print(f"[green]✓[/green] Generated {yaml_path}")

        if format in ["json", "both"]:
            json_path = output_path / "config.example.json"
            with open(json_path, "w") as f:
                json.dump(config_data, f, indent=2)
            console.print(f"[green]✓[/green] Generated {json_path}")

        if format == "both":
            console.print(
                f"\n[bold green]Success![/bold green] Generated both YAML and JSON configuration examples in {output_path}"
            )
        else:
            console.print(
                f"\n[bold green]Success![/bold green] Generated {format.upper()} configuration example in {output_path}"
            )



    @cli.command()
    @click.option(
        "--output-dir",
        "-o",
        type=click.Path(),
        default=".",
        help="Output directory for generated .env file",
    )
    @click.option(
        "--interactive",
        "-i",
        is_flag=True,
        help="Use interactive prompts to customize environment variables",
    )
    @click.option(
        "--include-pyeztrace",
        is_flag=True,
        default=True,
        help="Include PyEzTrace environment variables",
    )
    @click.option(
        "--include-optional",
        is_flag=True,
        default=True,
        help="Include optional environment variables",
    )
    def generate_env(output_dir, interactive, include_pyeztrace, include_optional):
        """Generate example .env file with all environment variables

        This command creates a comprehensive .env.example file that includes all
        environment variables used by the Andromeda framework and its dependencies.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Environment Generator[/bold blue]",
                subtitle="Generate example .env file",
            )
        )

        env_data = generate_example_env(interactive, include_pyeztrace, include_optional)

        output_path = Path(output_dir) / ".env.example"

        with open(output_path, "w") as f:
            for key, value in env_data.items():
                f.write(f"{key}={value}\n")

        console.print(f"[green]✓[/green] Generated {output_path}")

        # Display the generated content in a nice table
        display_env_table(env_data)

        console.print(
            f"\n[bold green]Success![/bold green] Generated .env.example file in {output_path}"
        )



    @cli.command()
    def show_config_options():
        """Display all available configuration options with descriptions

        This command shows a comprehensive overview of all configuration options
        available in the Andromeda framework.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Configuration Options[/bold blue]",
                subtitle="Complete reference of all configuration options",
            )
        )

        display_config_options_help()



    @cli.command()
    @click.argument("source", type=click.Path(exists=True), required=False)
    @click.option(
        "--format",
        "-f",
        type=click.Choice(["yaml", "json"]),
        help="Format of the config file to validate",
    )
    def validate_config(source, format):
        """Validate configuration files for correctness and completeness

        This command validates existing configuration files and provides detailed
        feedback about any issues found.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Configuration Validator[/bold blue]",
                subtitle="Validate configuration files",
            )
        )

        if not source:
            # Look for config files in current directory
            config_files = []
            for ext in [".yaml", ".yml", ".json"]:
                config_files.extend(Path(".").glob(f"*config*{ext}"))
                config_files.extend(Path(".").glob(f"config*{ext}"))

            if not config_files:
                console.print(
                    "[yellow]No configuration files found in current directory.[/yellow]"
                )
                console.print(
                    "Use [bold]andromeda generate-config[/bold] to create example configs."
                )
                return

            source = str(config_files[0])
            console.print(f"[green]Found configuration file:[/green] {source}")

        validation_result = validate_configuration_file(source, format)
        errors = list(validation_result["errors"])

    

        if not errors:
            console.print(f"\n[bold green]✓ Configuration is valid![/bold green]")
        
            display_config_summary(validation_result["config"])
        else:
            console.print(f"\n[bold red]✗ Configuration has errors:[/bold red]")
            for error in errors:
                console.print(f"  [red]•[/red] {error}")
            console.print(
                f"\n[yellow]Fix the errors above and run validation again.[/yellow]"
            )



    @cli.command()
    def setup():
        """Interactive setup wizard for new projects

        This command guides users through the initial setup process, helping them
        configure their project structure, dependencies, and basic configuration.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Setup Wizard[/bold blue]",
                subtitle="Interactive setup for new projects",
            )
        )

        run_setup_wizard()



    @cli.command()
    @click.option("--check-deps", is_flag=True, help="Check system dependencies")
    @click.option(
        "--test-connections", is_flag=True, help="Test external service connections"
    )
    @click.option("--check-env", is_flag=True, help="Check environment setup")
    @click.option(
        "--check-config",
        is_flag=True,
        help="Validate configuration files and summarize any issues",
    )
    @click.option(
        "--check-tools",
        is_flag=True,
        help="Inspect tool registry and config/tool wiring",
    )
    def diagnose(
        check_deps: bool,
        test_connections: bool,
        check_env: bool,
        check_config: bool,
        check_tools: bool,
    ) -> None:
        """Run system diagnostics to check setup, configuration, and dependencies.

        This command performs several diagnostic checks to help debug common issues:

        - Python/system dependencies
        - Connectivity to external services
        - Environment variables and .env presence
        - Config file structure and AndromedaConfig parsing
        - Tool registry status and any mismatches with config tool names
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda System Diagnostics[/bold blue]",
                subtitle="Check system setup, configuration, and dependencies",
            )
        )

        any_flag = any([check_deps, test_connections, check_env, check_config, check_tools])

        if check_deps or not any_flag:
            console.print("\n[bold cyan]Checking dependencies...[/bold cyan]")
            check_dependencies()

        if test_connections or not any_flag:
            console.print("\n[bold cyan]Testing external service connections...[/bold cyan]")
            test_service_connections()

        if check_env or not any_flag:
            console.print("\n[bold cyan]Checking environment setup...[/bold cyan]")
            check_environment_setup()

        if check_config or not any_flag:
            console.print("\n[bold cyan]Validating configuration files...[/bold cyan]")
            run_config_diagnostics()

        if check_tools or not any_flag:
            console.print("\n[bold cyan]Inspecting tool registry and wiring...[/bold cyan]")
            run_tool_diagnostics()

        console.print(f"\n[bold green]Diagnostics complete![/bold green]")



    @cli.command()
    def show_env_vars():
        """Display all environment variables with descriptions

        This command shows a comprehensive overview of all environment variables
        used by the Andromeda framework and its dependencies.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Environment Variables[/bold blue]",
                subtitle="Complete reference of all environment variables",
            )
        )

        display_env_vars_help()



    @cli.command()
    def show_tools():
        """List tools registered in the global Toolkit registry.

        This is useful when wiring tools via ``config.yaml`` using their string
        names (e.g., ``tools: [web_search]``).
        """

        # Import built-in tools so they self-register with the Toolkit.
        try:  # noqa: F401
            import andromeda.tools.tools as _builtin_tools  # type: ignore[unused-import]
        except ImportError:
            _builtin_tools = None  # type: ignore[assignment]

        from andromeda.tools.toolkit import get_default_toolkit

        toolkit = get_default_toolkit()
        tools = toolkit.all()

        console.print(
            Panel.fit(
                "[bold blue]Andromeda Tool Registry[/bold blue]",
                subtitle="Built-in tools",
            )
        )

        if not tools:
            console.print("[yellow]No tools are currently registered in the Toolkit.[/yellow]")
            console.print(
                "Register tools at startup using "
                "[bold]andromeda.tools.toolkit.register_tool[/bold] "
                "or by importing modules that register tools."
            )
            return

        table = Table(title="All Tools")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description/Prompt", style="white")

        for name, tool in sorted(tools.items(), key=lambda item: item[0]):
            # LangChain tools expose ``description``; fall back to docstring.
            desc = getattr(tool, "description", None) or (tool.__doc__ or "").strip()
            table.add_row(name, desc or "-")
            # divider
            table.add_row("", "-" * 10, style="dim")

        console.print(table)


    @cli.command()
    @click.argument("name")
    @click.argument("prompt", required=False)
    @click.option(
        "--kind",
        type=click.Choice(["agent", "workflow"]),
        default=None,
        help="Force kind when names collide.",
    )
    @click.option(
        "--input",
        "input_value",
        type=str,
        help="JSON object of runtime inputs.",
    )
    @click.option(
        "--input-file",
        "input_file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Path to JSON file with runtime inputs.",
    )
    @click.option(
        "--state-file",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Optional workflow state file (read and write).",
    )
    @click.option("--root", type=click.Path(file_okay=False, path_type=Path), help="Project root for .andromeda")
    @click.option("--global-root", type=click.Path(file_okay=False, path_type=Path), help="Override global config root")
    @click.option("--no-global", is_flag=True, default=False, help="Exclude ~/.andromeda")
    @click.option("--dry-run", is_flag=True, help="Run validation-only mode")
    @click.option(
        "--json",
        "as_json",
        is_flag=True,
        help="Emit result as JSON.",
    )
    @click.option(
        "--raw",
        "include_raw",
        is_flag=True,
        help="Include the verbose 'raw' field in JSON output.",
    )
    @click.option(
        "--stream",
        is_flag=True,
        help="Stream execution progress as events.",
    )
    def run(
        name,
        prompt,
        kind,
        input_value,
        input_file,
        state_file,
        root,
        global_root,
        no_global,
        dry_run,
        as_json,
        include_raw,
        stream,
    ):
        """Execute an agent or workflow from .andromeda definitions."""
        runtime = _build_runtime(
            str(root) if root else None,
            include_global=not no_global,
            global_root=str(global_root) if global_root else None,
        )
        try:
            inputs = _load_runtime_inputs(input_value, input_file)
        except click.BadParameter as exc:
            raise click.ClickException(str(exc)) from exc

        try:
            try:
                if stream:
                    if state_file is not None:
                        raise click.ClickException("--state-file is not supported with --stream.")
                    if dry_run:
                        raise click.ClickException("--dry-run is not supported with --stream.")

                    for chunk in runtime.stream(
                        name,
                        prompt=prompt,
                        inputs=inputs,
                        kind=kind,
                        thread_id=None,
                        metadata=None,
                    ):
                        if as_json:
                            click.echo(_json_dumps(chunk))
                        else:
                            click.echo(_render_stream_chunk(chunk))
                        try:
                            sys.stdout.flush()
                        except Exception:
                            pass
                    return

                result = runtime.run(
                    name,
                    prompt=prompt,
                    inputs=inputs,
                    kind=kind,
                    state_file=state_file,
                    dry_run=dry_run,
                )
            except (RunnableNotFoundError, RunnableAmbiguousError) as exc:
                raise click.ClickException(str(exc))

            if as_json:
                click.echo(_json_dumps(result.to_dict(verbose=include_raw), indent=2))
                return

            if result.kind == "agent":
                if result.text:
                    console.print(result.text)
                    return
                if result.structured_response is not None:
                    console.print(_json_dumps(result.structured_response, indent=2))
                    return
                if result.messages:
                    console.print(message_content(result.messages[-1]))
                    return
                if result.raw is not None:
                    console.print(_json_dumps(result.raw, indent=2))
                    return
                console.print("[yellow]No content returned from agent.[/yellow]")
                return

            if result.text:
                console.print(result.text)
                return
            if result.structured_response is not None:
                console.print(_json_dumps(result.structured_response, indent=2))
                return
            if result.state is not None:
                console.print(_json_dumps(result.state, indent=2))
                return
            console.print(_json_dumps(result.raw or {}, indent=2))
        finally:
            _close_runtime(runtime)


    @cli.command()
    @click.option(
        "--kind",
        type=click.Choice(["agent", "workflow"]),
        default=None,
        help="Filter by kind.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit result as JSON")
    @click.option("--root", type=click.Path(file_okay=False, path_type=Path), help="Project root for .andromeda")
    @click.option("--global-root", type=click.Path(file_okay=False, path_type=Path), help="Override global config root")
    @click.option("--no-global", is_flag=True, default=False, help="Exclude ~/.andromeda")
    def list(
        kind,
        as_json,
        root,
        global_root,
        no_global,
    ):
        """List runtime agents and workflows."""
        runtime = _build_runtime(
            str(root) if root else None,
            include_global=not no_global,
            global_root=str(global_root) if global_root else None,
        )
        try:
            items = runtime.list(kind=kind)

            if as_json:
                click.echo(_json_dumps(items, indent=2))
                return

            if not items:
                console.print("[yellow]No runtime definitions found.[/yellow]")
                return

            table = Table()
            table.add_column("name")
            table.add_column("kind")
            table.add_column("scope")
            table.add_column("source")

            for item in sorted(items, key=lambda item: (item.get("name", ""), item.get("kind", ""))):
                table.add_row(item.get("name", ""), item.get("kind", ""), item.get("scope", ""), item.get("source", ""))
            console.print(table)
        finally:
            _close_runtime(runtime)


    @cli.command()
    @click.argument("name")
    @click.option(
        "--kind",
        type=click.Choice(["agent", "workflow"]),
        default=None,
        help="Filter by kind when names collide.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit result as JSON")
    @click.option("--root", type=click.Path(file_okay=False, path_type=Path), help="Project root for .andromeda")
    @click.option("--global-root", type=click.Path(file_okay=False, path_type=Path), help="Override global config root")
    @click.option("--no-global", is_flag=True, default=False, help="Exclude ~/.andromeda")
    def inspect(name, kind, as_json, root, global_root, no_global):
        """Show resolved definition metadata."""
        runtime = _build_runtime(
            str(root) if root else None,
            include_global=not no_global,
            global_root=str(global_root) if global_root else None,
        )
        try:
            try:
                info = runtime.inspect(name, kind=kind)
            except (RunnableNotFoundError, RunnableAmbiguousError) as exc:
                raise click.ClickException(str(exc))

            if as_json:
                click.echo(_json_dumps(info, indent=2))
                return

            table = Table(show_header=False, box=None)
            table.add_column(style="cyan")
            table.add_column()
            for key in ("name", "kind", "scope", "source"):
                table.add_row(key, str(info.get(key, "")))
            console.print(table)
        finally:
            _close_runtime(runtime)


    @cli.command()
    @click.argument("name", required=False)
    @click.option(
        "--kind",
        type=click.Choice(["agent", "workflow"]),
        default=None,
        help="Validate a specific kind.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Emit result as JSON")
    @click.option("--root", type=click.Path(file_okay=False, path_type=Path), help="Project root for .andromeda")
    @click.option("--global-root", type=click.Path(file_okay=False, path_type=Path), help="Override global config root")
    @click.option("--no-global", is_flag=True, default=False, help="Exclude ~/.andromeda")
    def validate(name, kind, as_json, root, global_root, no_global):
        """Validate definitions or all definitions in scope."""
        runtime = _build_runtime(
            str(root) if root else None,
            include_global=not no_global,
            global_root=str(global_root) if global_root else None,
        )
        try:
            result = runtime.validate(name=name, kind=kind)
            payload = result.to_dict()
            if as_json:
                click.echo(_json_dumps(payload, indent=2))
                return

            if result.valid:
                if name:
                    console.print(f"[green]Valid {kind or 'runnable'}:[/green] {name}")
                else:
                    console.print("[green]Runtime registry is valid.[/green]")
                return

            console.print("[red]Validation failed:[/red]")
            for issue in payload["issues"]:
                console.print(f"- [{issue.get('severity')}] {issue.get('message')} ({issue.get('path', 'no path')})")
        finally:
            _close_runtime(runtime)



    @cli.command()
    @click.argument("source", type=click.Path(exists=True), required=False)
    @click.option(
        "--output",
        "-o",
        type=click.Path(),
        help="Output file path (supports .mmd, .mermaid, .svg, .png, .txt)",
    )
    @click.option(
        "--format",
        "-f",
        type=click.Choice(["mermaid", "svg", "png", "text"]),
        default="mermaid",
        help="Output format (default: mermaid code)",
    )
    @click.option(
        "--type",
        "-t",
        "wf_type",
        type=click.Choice(["auto", "team", "supervisor", "workflow"]),
        default="auto",
        help="Workflow type (default: auto-detect from config or Python entrypoint)",
    )
    def visualize(source, output, format, wf_type):
        """Generate visual diagrams of workflow structures.
        
        Supports Team, Supervisor, and Custom Workflow visualizations.
        The positional argument can be either:
        
        - A config file (YAML/JSON), e.g. config.yaml
        - A Python entrypoint (e.g. main.py) that orchestrates Team/Supervisor/WorkflowBuilder
        
        When no path is provided, the command will:
        - Prefer a config file discovered in the current directory, or
        - Fall back to main.py for custom workflows.
        """
        console.print(
            Panel.fit(
                "[bold blue]Andromeda Workflow Visualizer[/bold blue]",
                subtitle="Generate workflow diagrams",
            )
        )
        
        # Import visualization modules and helpers
        from andromeda.cli.visualize import (
            visualize_team_workflow,
            visualize_supervisor_workflow,
            visualize_custom_workflow,
            detect_workflow_type,
            detect_python_workflow_type,
            infer_source_kind,
            extract_config_path_from_python,
        )
        
        cfg = None
        input_path: Path
        
        # Resolve the primary input path: explicit source > discovered config > main.py
        if source:
            input_path = Path(source).resolve()
        else:
            config_files = discover_config_files()
            if config_files:
                input_path = config_files[0].resolve()
                console.print(f"[green]Found configuration file:[/green] {input_path}")
            else:
                main_py = Path("main.py").resolve()
                if not main_py.exists():
                    console.print(
                        "[yellow]No configuration file or main.py found in current directory.[/yellow]"
                    )
                    console.print(
                        "Pass a specific file (e.g. config.yaml or main.py), or use "
                        "[bold]andromeda generate-config[/bold] / [bold]andromeda setup[/bold] to create one."
                    )
                    return
                input_path = main_py
                console.print(f"[green]Using Python entrypoint:[/green] {input_path}")
        
        project_root = input_path if input_path.is_dir() else input_path.parent
        source_kind = infer_source_kind(input_path)
        
        # Optionally resolve a config file associated with a Python entrypoint.
        config_path: Optional[Path] = None
        if source_kind == "config":
            config_path = input_path
        elif source_kind == "python":
            resolved = extract_config_path_from_python(input_path)
            if resolved is not None:
                config_path = resolved
        
        # Load config when we have one and the requested / auto-detected workflow
        # type may rely on it (team/supervisor).
        if config_path is not None and wf_type in {"auto", "team", "supervisor"}:
            try:
                # Ensure built-in tools are registered
                try:  # noqa: F401
                    import andromeda.tools.tools as _builtin_tools  # type: ignore[unused-import]
                except ImportError:
                    _builtin_tools = None  # type: ignore[assignment]
                
                cfg = AndromedaConfig.load_from_file(str(config_path))
            except (ValueError, FileNotFoundError, PermissionError) as exc:
                console.print(f"[red]✗[/red] Failed to load config: {exc}")
                return
        
        # Detect workflow type if auto
        if wf_type == "auto":
            workflow_type: Optional[str] = None
            if cfg is not None and config_path is not None:
                workflow_type = detect_workflow_type(cfg, str(config_path))
            elif source_kind == "python":
                workflow_type = detect_python_workflow_type(input_path)
            
            if not workflow_type:
                console.print(
                    "[yellow]Could not auto-detect workflow type from the provided source. "
                    "Please specify with --type (team, supervisor, or workflow).[/yellow]"
                )
                return
            console.print(f"[cyan]Detected workflow type:[/cyan] {workflow_type}")
        else:
            workflow_type = wf_type
        
        # Generate visualization based on type
        try:
            if workflow_type == "team":
                if cfg is None:
                    console.print(
                        "[red]✗[/red] Cannot visualize a Team workflow without a config file."
                    )
                    return
                diagram_code = visualize_team_workflow(cfg)
            elif workflow_type == "supervisor":
                if cfg is None:
                    console.print(
                        "[red]✗[/red] Cannot visualize a Supervisor workflow without a config file."
                    )
                    return
                diagram_code = visualize_supervisor_workflow(cfg)
            elif workflow_type == "workflow":
                # For custom workflows, use the project root (or explicit directory/file)
                base_path = str(project_root)
                diagram_code = visualize_custom_workflow(base_path)
            else:
                console.print(f"[red]Unknown workflow type: {workflow_type}[/red]")
                return

            # Output the diagram
            if output:
                output_path = Path(output)
            
                # Determine format from extension or format parameter
                if output_path.suffix in [".mmd", ".mermaid"]:
                    output_format = "mermaid"
                elif output_path.suffix == ".txt":
                    output_format = "text"
                elif output_path.suffix in [".svg", ".png"]:
                    output_format = output_path.suffix[1:]  # Remove dot
                else:
                    # Use format parameter, and update extension if needed
                    output_format = format
                    if output_format in ["svg", "png"] and not output_path.suffix:
                        output_path = output_path.with_suffix(f".{output_format}")
            
                if output_format == "mermaid":
                    output_path.write_text(diagram_code, encoding="utf-8")
                    console.print(f"[green]✓[/green] Saved mermaid diagram to {output_path}")
                elif output_format == "text":
                    # Simple text representation
                    text_diagram = mermaid_to_text(diagram_code)
                    output_path.write_text(text_diagram, encoding="utf-8")
                    console.print(f"[green]✓[/green] Saved text diagram to {output_path}")
                elif output_format in ["svg", "png"]:
                    # Try to render to SVG/PNG using mmdc if available
                    render_mermaid_to_image(diagram_code, output_path, output_format)
                else:
                    console.print(f"[yellow]⚠[/yellow] Unknown format '{output_format}', saving as mermaid code")
                    output_path = output_path.with_suffix(".mmd")
                    output_path.write_text(diagram_code, encoding="utf-8")
                    console.print(f"[green]✓[/green] Saved mermaid diagram to {output_path}")
            else:
                # Print mermaid code
                console.print("\n[bold cyan]Mermaid Diagram Code:[/bold cyan]\n")
                console.print("```mermaid")
                console.print(diagram_code)
                console.print("```")
                console.print(
                    "\n[yellow]Tip:[/yellow] Copy the code above and paste it into "
                    "https://mermaid.live or use --output to save to a file."
                )

        except (RuntimeError, OSError, ValueError) as exc:
            console.print(f"[red]✗[/red] Error generating visualization: {exc}")
            import traceback

            if os.getenv("DEBUG"):
                console.print(traceback.format_exc())
