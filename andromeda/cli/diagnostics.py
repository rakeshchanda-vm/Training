"""System diagnostics for Andromeda CLI."""

import os
from pathlib import Path

from andromeda.cli.helpers import console

def check_dependencies():
    """Check system dependencies"""
    dependencies = [
        ("python", "3.11", "Programming language"),
        ("pip", "20.0", "Package manager"),
        ("git", "2.0", "Version control"),
    ]

    try:
        import importlib

        python_deps = [
            ("pydantic", "2.0.0", "Data validation"),
            ("langchain", "0.1.0", "LLM framework"),
            ("rich", "13.0.0", "Terminal formatting"),
            ("click", "8.0.0", "CLI framework"),
            ("questionary", "1.10.0", "Interactive prompts"),
        ]

        all_deps = dependencies + python_deps
    except ImportError:
        all_deps = dependencies

    for name, version, description in all_deps:
        if name == "python":
            import sys

            # Compare version tuples to avoid lexicographic errors
            try:
                req_major, req_minor = (int(x) for x in version.split(".")[:2])
            except ValueError:
                req_major, req_minor = 0, 0
            cur = (sys.version_info.major, sys.version_info.minor)
            req = (req_major, req_minor)
            if cur >= req:
                console.print(
                    f"  [green]✓[/green] {description}: {name} {cur[0]}.{cur[1]}"
                )
            else:
                console.print(
                    f"  [yellow]⚠[/yellow] {description}: {name} {cur[0]}.{cur[1]} (need {version}+)"
                )
        elif name in ["pip", "git"]:
            # Simple check for system commands
            import subprocess

            try:
                result = subprocess.run(
                    [name, "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    console.print(f"  [green]✓[/green] {description}: {name}")
                else:
                    console.print(f"  [red]✗[/red] {description}: {name} not working")
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                console.print(f"  [red]✗[/red] {description}: {name} not found")
        else:
            # Python package check
            try:
                module = importlib.import_module(name)
                current_version = getattr(module, "__version__", "unknown")
                console.print(
                    f"  [green]✓[/green] {description}: {name} {current_version}"
                )
            except ImportError:
                console.print(f"  [red]✗[/red] {description}: {name} not installed")


def test_service_connections():
    """Test connections to external services"""
    services = [
        ("localhost:11434", "Ollama server", "http"),
        ("api.tavily.com", "Tavily API", "https"),
    ]

    for host_port, service_name, protocol in services:
        # If a port is provided, try a TCP socket; otherwise fall back to HTTP(S)
        if ":" in host_port:
            import socket

            try:
                host, port_s = host_port.split(":", 1)
                port = int(port_s)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                if result == 0:
                    console.print(
                        f"  [green]✓[/green] {service_name}: Connected successfully"
                    )
                else:
                    console.print(
                        f"  [yellow]⚠[/yellow] {service_name}: Connection failed (service may not be running)"
                    )
            except (OSError, ValueError):
                console.print(
                    f"  [yellow]⚠[/yellow] {service_name}: Connection check failed"
                )
        else:
            try:
                import requests  # type: ignore

                url = f"{protocol}://{host_port}"
                resp = requests.get(url, timeout=5)
                if 200 <= resp.status_code < 500:
                    console.print(
                        f"  [green]✓[/green] {service_name}: HTTP reachable ({resp.status_code})"
                    )
                else:
                    console.print(
                        f"  [yellow]⚠[/yellow] {service_name}: HTTP check failed ({resp.status_code})"
                    )
            except ImportError:
                console.print(
                    f"  [yellow]•[/yellow] requests not installed; skipping {service_name} HTTP check"
                )
            except Exception:
                console.print(
                    f"  [yellow]⚠[/yellow] {service_name}: HTTP check failed"
                )


def check_environment_setup():
    """Check environment setup"""
    env_vars = [
        # these are optional, keeping this commented for now
        # "TAVILY_API_KEY",
        # "LANGFUSE_PUBLIC_KEY",
        # "LANGFUSE_SECRET_KEY",
        # "LANGFUSE_HOST",
    ]

    missing_vars = []
    for var in env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        console.print("  [yellow]⚠[/yellow] Missing environment variables:")
        for var in missing_vars:
            console.print(f"    - {var}")
        console.print(
            "  Use [bold]andromeda generate-env[/bold] to create .env.example"
        )
    else:
        console.print("  [green]✓[/green] All required environment variables are set")

    # Check for .env file
    if Path(".env").exists():
        console.print("  [green]✓[/green] .env file found")
    elif Path(".env.example").exists():
        console.print(
            "  [yellow]⚠[/yellow] .env.example found - copy to .env and configure"
        )
    else:
        console.print("  [red]✗[/red] No .env file found")
