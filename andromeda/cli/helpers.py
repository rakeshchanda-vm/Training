"""Helper utilities for the Andromeda CLI."""

import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    # Fallback: plain print-based console
    class _PlainConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = _PlainConsole()

# Optional import for interactive features
try:
    import questionary

    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    console.print(
        "[yellow]⚠[/yellow] questionary not installed. Interactive features will use basic prompts."
    )


def ask_bool(prompt: str, default: bool = False) -> bool:
    """Helper to ask a yes/no question via questionary or basic input."""

    if HAS_QUESTIONARY:
        return bool(
            questionary.confirm(
                prompt,
                default=default,
            ).ask()
        )

    if default:
        answer = input(f"{prompt} (Y/n): ").strip().lower()
        return answer not in ["n", "no"]

    answer = input(f"{prompt} (y/N): ").strip().lower()
    return answer in ["y", "yes"]


def ask_text(prompt: str, default: str = "") -> str:
    """Helper to ask for free-form text with a default."""

    if HAS_QUESTIONARY:
        return str(
            questionary.text(
                prompt,
                default=default,
            ).ask()
            or default
        )

    suffix = f" (default: {default})" if default else ""
    return input(f"{prompt}{suffix}: ").strip() or default


def ask_float(prompt: str, default: float) -> float:
    """Helper to ask for a float value with basic validation."""

    while True:
        raw = ask_text(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")


def ask_int(prompt: str, default: int) -> int:
    """Helper to ask for an int value with basic validation."""

    while True:
        raw = ask_text(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            console.print("[red]Please enter a valid integer.[/red]")


def slugify_identifier(name: str) -> str:
    """Convert an arbitrary string into a safe Python identifier."""

    slug = name.strip().lower()
    slug = re.sub(r"[\s\-]+", "_", slug)
    slug = re.sub(r"[^0-9a-zA-Z_]", "", slug)
    if not slug or slug[0].isdigit():
        slug = f"step_{slug}" if slug else "step_1"
    return slug


def discover_config_files() -> List[Path]:
    """Find candidate configuration files in the current directory."""

    config_files: List[Path] = []
    for ext in [".yaml", ".yml", ".json"]:
        config_files.extend(Path(".").glob(f"*config*{ext}"))
        config_files.extend(Path(".").glob(f"config*{ext}"))

    # Deduplicate while preserving order
    seen = set()
    unique_files: List[Path] = []
    for path in config_files:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)
    return unique_files


def iter_agent_configs(agents_cfg: Any) -> List[Dict[str, Any]]:
    """Normalize agents config (list or dict) into a list of dicts."""

    if isinstance(agents_cfg, list):
        return [cfg for cfg in agents_cfg if isinstance(cfg, dict)]
    if isinstance(agents_cfg, dict):
        return [cfg for cfg in agents_cfg.values() if isinstance(cfg, dict)]
    return []

