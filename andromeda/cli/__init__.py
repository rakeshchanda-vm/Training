#!/usr/bin/env python3
"""
Andromeda Framework CLI - Development and Setup Tool [A Work in Progress - in pre-alpha]

A CLI tool to help developers get started with the Andromeda multi-agent framework.
Provides templates, configuration management, and setup utilities.
"""

import sys

def _check_cli_dependencies():
    """Check if CLI dependencies are installed and return (has_deps, missing_packages)."""
    missing = []
    for pkg, import_name in [("click", "click"), ("rich", "rich"), ("questionary", "questionary")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    return len(missing) == 0, missing


# Lazy console accessor (created on first use)
_console = None


def _get_console():
    """Get the Rich console instance (lazy initialization)."""
    global _console
    if _console is None:
        from rich.console import Console
        _console = Console()
    return _console


# Cached module-level attributes (populated on first access via __getattr__)
_cli = None
_commands = None


def _build_cli():
    """Build and return the Click group (lazy import)."""
    import click

    @click.group()
    @click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
    @click.version_option(version="1.1.10", message="Andromeda Framework CLI - %(version)s")
    def cli_group(verbose):
        """Andromeda Framework CLI - Development and Setup Tool

        A comprehensive CLI tool to help developers get started with the Andromeda
        multi-agent framework. Provides configuration management, setup utilities,
        and development tools.
        """
        if verbose:
            _get_console().print(
                "[bold blue]Andromeda CLI[/bold blue] - Verbose mode enabled", style="info"
            )

    return cli_group


def _init_cli():
    """Initialize and cache the Click CLI group. Checks deps first."""
    global _cli
    if _cli is not None:
        return

    has_deps, missing = _check_cli_dependencies()
    if not has_deps:
        missing_str = ", ".join(missing)
        raise ImportError(
            "CLI dependencies are not installed.\n"
            f"Missing packages: {missing_str}\n"
            "Install them with: pip install 'andromeda[cli]'"
        )

    _cli = _build_cli()
    from andromeda.cli.commands import register_commands
    register_commands(_cli)


def _init_commands():
    """Initialize and cache the commands module."""
    global _commands
    if _commands is not None:
        return

    has_deps, missing = _check_cli_dependencies()
    if not has_deps:
        missing_str = ", ".join(missing)
        raise ImportError(
            "CLI dependencies are not installed.\n"
            f"Missing packages: {missing_str}\n"
            "Install them with: pip install 'andromeda[cli]'"
        )

    import andromeda.cli.commands
    _commands = andromeda.cli.commands


def __getattr__(name):
    if name == "cli":
        _init_cli()
        return _cli
    if name == "commands":
        _init_commands()
        return _commands
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main():
    """Main entry point for the CLI"""
    try:
        # Access cli through __getattr__ (which checks deps and builds lazily)
        # Must use getattr on the module to trigger __getattr__ for lazy loading
        _cli_entry = getattr(sys.modules[__name__], "cli")
        _cli_entry()
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        _get_console().print("\n[yellow]Operation cancelled by user.[/yellow]")
        return 1
    except Exception as e:
        _get_console().print(f"\n[red]Error:[/red] {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
