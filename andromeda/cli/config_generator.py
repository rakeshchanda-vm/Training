"""Configuration generation for Andromeda CLI."""

from typing import Any, Dict, Optional

import click

from andromeda.cli.helpers import (
    HAS_QUESTIONARY,
    ask_bool,
    ask_float,
    ask_text,
    console,
    iter_agent_configs,
)

# Import questionary if available
if HAS_QUESTIONARY:
    import questionary
else:
    questionary = None


def _default_checkpointer_config() -> Dict[str, Any]:
    return {"backend": "in-memory"}


def generate_example_config(interactive: bool = False) -> Dict[str, Any]:
    """Generate example configuration data.

    When ``interactive`` is True, the user can configure all major
    options exposed in :mod:`andromeda.config.config` (agents, supervisor,
    planner, and report) via the CLI.
    """

    if not interactive:
        return generate_default_config()

    config = generate_interactive_config()
    # Let the user optionally tune prompts, citations, planner and report.
    return tune_config_interactive(config, project_type=None)


def generate_default_config() -> Dict[str, Any]:
    """Generate default configuration with sensible defaults."""

    return {
        "agents": [
            {
                "name": "researcher",
                "model": {
                    "name": "qwen3:8b",
                    "provider": "litellm",
                    "temperature": 0.3,
                    "other_args": {},
                },
                "tools": [],
                "prompt": "You are a research assistant. Be thorough and cite your sources.",
                "validation": {
                    "enabled": True,
                    "model": {
                        "name": "gpt-oss:20b",
                        "provider": "litellm",
                        "temperature": 0.3,
                        "other_args": {},
                    },
                    "skip_after_attempts": 3,
                    "min_sufficiency_score": 0.7,
                },
                "citations": {
                    "required": True,
                    "min_density": 0.1,
                    "require_reference_section": True,
                },
                "return_direct": False,
                "next": "analyst",
                "type": "react",
                "debug": 2,
                "middleware": {
                    "enabled": False,
                },
                "checkpointer": _default_checkpointer_config(),
            },
            {
                "name": "analyst",
                "model": {
                    "name": "qwen3:8b",
                    "provider": "litellm",
                    "temperature": 0.2,
                    "other_args": {},
                },
                "tools": [],
                "prompt": "You are a data analyst. Provide detailed analysis with evidence.",
                "validation": {
                    "enabled": True,
                    "model": {
                        "name": "gpt-oss:20b",
                        "provider": "litellm",
                        "temperature": 0.3,
                        "other_args": {},
                    },
                    "skip_after_attempts": 3,
                    "min_sufficiency_score": 0.8,
                },
                "citations": {
                    "required": True,
                    "min_density": 0.15,
                    "require_reference_section": True,
                },
                "return_direct": False,
                "next": "reporter",
                "type": "react",
                "debug": 2,
                "middleware": {
                    "enabled": False,
                },
                "checkpointer": _default_checkpointer_config(),
            },
            {
                "name": "reporter",
                "model": {
                    "name": "qwen3:8b",
                    "provider": "litellm",
                    "temperature": 0.4,
                    "other_args": {},
                },
                "tools": [],
                "prompt": "You are a technical writer. Create clear, comprehensive reports.",
                "validation": {
                    "enabled": True,
                    "model": {
                        "name": "gpt-oss:20b",
                        "provider": "litellm",
                        "temperature": 0.3,
                        "other_args": {},
                    },
                    "skip_after_attempts": 2,
                    "min_sufficiency_score": 0.9,
                },
                "citations": {
                    "required": True,
                    "min_density": 0.2,
                    "require_reference_section": True,
                },
                "return_direct": False,
                "next": None,
                "type": "react",
                "debug": 2,
                "middleware": {
                    "enabled": False,
                },
                "checkpointer": _default_checkpointer_config(),
            },
        ],
        "supervisor": {
            "name": "supervisor",
            "model": {
                "name": "qwen3:8b",
                "provider": "litellm",
                "temperature": 0.4,
                "other_args": {},
            },
            "tools": [],
            "prompt": "You are the supervisor agent. Coordinate between agents and ensure quality.",
            "enable_planning": True,
            "allow_parallel_agents": False,
            "allow_async_tasks": False,
            "validation": {
                "enabled": True,
                "model": {
                    "name": "gpt-oss:20b",
                    "provider": "litellm",
                    "temperature": 0.3,
                    "other_args": {},
                },
                "skip_after_attempts": 3,
                "min_sufficiency_score": 0.8,
            },
            "citations": {
                "required": True,
                "min_density": 0.1,
                "require_reference_section": True,
            },
            "return_direct": False,
            "next": None,
            "type": "react",
            "debug": 2,
            "middleware": {
                "enabled": False,
            },
            "checkpointer": _default_checkpointer_config(),
        },
        "planner": {
            "model": {
                "name": "qwen3:8b",
                "provider": "litellm",
                "temperature": 0.9,
                "other_args": {},
            },
            "task_type": "research",
            "report_structure": None,
        },
        "report": {
            "format": None,
            "citations": {
                "required": True,
                "min_density": 0.1,
                "require_reference_section": True,
            },
            "validation": {
                "enabled": True,
                "model": {
                    "name": "gpt-oss:20b",
                    "provider": "litellm",
                    "temperature": 0.3,
                    "other_args": {},
                },
                "skip_after_attempts": 3,
                "min_sufficiency_score": 0.7,
            },
        },
    }


def generate_interactive_config() -> Dict[str, Any]:
    """Generate configuration interactively using prompts"""
    console.print("[bold cyan]Interactive Configuration Generator[/bold cyan]")
    console.print(
        "We'll create a configuration step by step. Press Ctrl+C to cancel.\n"
    )

    if not HAS_QUESTIONARY:
        console.print(
            "[yellow]Using basic prompts (install questionary for enhanced experience)[/yellow]"
        )

    # Ask about number of agents
    if HAS_QUESTIONARY:
        num_agents = questionary.select(
            "How many agents would you like to configure?",
            choices=["1", "2", "3", "4", "5"],
            default="2",
        ).ask()
    else:
        num_agents = (
            input(
                "How many agents would you like to configure? (1-5, default: 2): "
            ).strip()
            or "2"
        )

    if not num_agents:
        raise click.Abort()

    agents = []
    for i in range(int(num_agents)):
        console.print(f"\n[yellow]Agent {i+1} Configuration[/yellow]")

        if HAS_QUESTIONARY:
            agent_name = questionary.text(
                f"Agent {i+1} name:", default=f"agent_{i+1}"
            ).ask()
        else:
            agent_name = (
                input(f"Agent {i+1} name (default: agent_{i+1}): ").strip()
                or f"agent_{i+1}"
            )

        if HAS_QUESTIONARY:
            model_name = questionary.text(
                f"Model name for {agent_name}:", default="qwen3:8b"
            ).ask()
        else:
            model_name = (
                input(f"Model name for {agent_name} (default: qwen3:8b): ").strip()
                or "qwen3:8b"
            )

        if HAS_QUESTIONARY:
            provider = questionary.select(
                f"Provider for {agent_name}:",
                choices=["ollama", "openai", "anthropic", "bedrock", "azure_openai", "litellm"],
                default="litellm",
            ).ask()
        else:
            provider = (
                input(
                    f"Provider for {agent_name} (ollama/openai/anthropic/bedrock/azure_openai/litellm, default: ollama): "
                ).strip()
                or "litellm"
            )

        if HAS_QUESTIONARY:
            temperature = questionary.text(
                f"Temperature for {agent_name} (0.0-1.0):", default="0.3"
            ).ask()
        else:
            temperature = (
                input(f"Temperature for {agent_name} (0.0-1.0, default: 0.3): ").strip()
                or "0.3"
            )

        agents.append(
            {
                "name": agent_name,
                "model": {
                    "name": model_name,
                    "provider": provider,
                    "temperature": float(temperature),
                    "other_args": {},
                },
                "tools": [],
                "prompt": f"You are {agent_name}. Be helpful and accurate.",
                "validation": {
                    "enabled": True,
                    "model": {
                        "name": "gpt-oss:20b",
                        "provider": "litellm",
                        "temperature": 0.3,
                        "other_args": {},
                    },
                    "skip_after_attempts": 3,
                    "min_sufficiency_score": 0.7,
                },
                "citations": {
                    "required": True,
                    "min_density": 0.1,
                    "require_reference_section": True,
                },
                "return_direct": False,
                "next": f"agent_{i+2}" if i < int(num_agents) - 1 else None,
                "type": "react",
                "debug": 2,
                "middleware": {
                    "enabled": False,
                },
                "checkpointer": _default_checkpointer_config(),
            }
        )

    return {
        "agents": agents,
        "supervisor": {
            "name": "supervisor",
            "model": {
                "name": "qwen3:8b",
                "provider": "litellm",
                "temperature": 0.4,
                "other_args": {},
            },
            "tools": [],
            "prompt": "You are the supervisor agent. Coordinate between agents and ensure quality.",
            "enable_planning": True,
            "allow_parallel_agents": False,
            "allow_async_tasks": False,
            "validation": {
                "enabled": True,
                "model": {
                    "name": "gpt-oss:20b",
                    "provider": "litellm",
                    "temperature": 0.3,
                    "other_args": {},
                },
                "skip_after_attempts": 3,
                "min_sufficiency_score": 0.8,
            },
            "citations": {
                "required": True,
                "min_density": 0.1,
                "require_reference_section": True,
            },
            "return_direct": False,
            "next": None,
            "type": "react",
            "debug": 2,
            "middleware": {
                "enabled": False,
            },
            "checkpointer": _default_checkpointer_config(),
        },
        "planner": {
            "model": {
                "name": "qwen3:8b",
                "provider": "litellm",
                "temperature": 0.9,
                "other_args": {},
            },
            "task_type": "research",
            "report_structure": None,
        },
        "report": {
            "format": None,
            "citations": {
                "required": True,
                "min_density": 0.1,
                "require_reference_section": True,
            },
            "validation": {
                "enabled": True,
                "model": {
                    "name": "gpt-oss:20b",
                    "provider": "litellm",
                    "temperature": 0.3,
                    "other_args": {},
                },
                "skip_after_attempts": 3,
                "min_sufficiency_score": 0.7,
            },
        },
    }


def tune_config_interactive(
    config_data: Dict[str, Any],
    project_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Interactively tune advanced configuration options.

    This helper is used by both the standalone ``generate-config`` command
    and the setup wizard so that high‑level knobs in
    :mod:`andromeda.config.config` (agents, supervisor, planner, report,
    citations, validation, prompts, etc.) can be reached from the CLI.
    """

    console.print(
        "\n[bold cyan]Advanced configuration[/bold cyan]\n"
        "You can now tweak prompts, citations, planner, and report settings.\n"
        "Press Enter to accept sensible defaults if you're not sure."
    )

    # ------------------------------------------------------------------
    # Agent and supervisor prompts
    # ------------------------------------------------------------------
    if ask_bool("Customize prompts for agents and supervisor?", default=False):
        agents_cfg_raw = config_data.get("agents", [])
        agents_cfg = iter_agent_configs(agents_cfg_raw)

        for idx, agent_cfg in enumerate(agents_cfg, start=1):
            name = str(agent_cfg.get("name", f"agent_{idx}"))
            default_prompt = agent_cfg.get(
                "prompt", f"You are {name}. Be helpful and accurate."
            )
            agent_cfg["prompt"] = ask_text(
                f"Prompt for agent '{name}'", default_prompt
            )

        supervisor_cfg = config_data.get("supervisor")
        if isinstance(supervisor_cfg, dict):
            default_sup_prompt = supervisor_cfg.get(
                "prompt",
                "You are the supervisor agent. Coordinate between agents and ensure quality.",
            )
            supervisor_cfg["prompt"] = ask_text(
                "Prompt for supervisor", default_sup_prompt
            )

    # ------------------------------------------------------------------
    # Supervisor routing/delegation controls
    # ------------------------------------------------------------------
    supervisor_cfg = config_data.get("supervisor")
    if isinstance(supervisor_cfg, dict) and ask_bool(
        "Configure supervisor routing and delegation controls?", default=False
    ):
        supervisor_cfg["enable_planning"] = ask_bool(
            "Enable supervisor planning/todo tools?",
            bool(supervisor_cfg.get("enable_planning", True)),
        )
        supervisor_cfg["allow_parallel_agents"] = ask_bool(
            "Allow supervisor to run multiple agent tasks in parallel?",
            bool(supervisor_cfg.get("allow_parallel_agents", False)),
        )
        supervisor_cfg["allow_async_tasks"] = ask_bool(
            "Allow supervisor to start background agent tasks?",
            bool(supervisor_cfg.get("allow_async_tasks", False)),
        )

    # ------------------------------------------------------------------
    # LangGraph checkpoint persistence
    # ------------------------------------------------------------------
    if ask_bool("Configure LangGraph checkpointer persistence?", default=False):
        if HAS_QUESTIONARY:
            backend = questionary.select(
                "Checkpointer backend:",
                choices=["in-memory", "postgres", "none"],
                default="in-memory",
            ).ask()
        else:
            backend = ask_text(
                "Checkpointer backend (in-memory/postgres/none)", "in-memory"
            )
        if backend not in {"in-memory", "postgres", "none"}:
            backend = "in-memory"

        checkpointer: Dict[str, Any] = {"backend": backend}
        if backend == "postgres":
            checkpointer["connection_string"] = ask_text(
                "Postgres connection string (supports ${DATABASE_URL})",
                "${DATABASE_URL}",
            )
            checkpointer["setup"] = ask_bool(
                "Run Postgres checkpointer setup at startup?", default=False
            )

        agents_cfg_raw = config_data.get("agents", [])
        for agent_cfg in iter_agent_configs(agents_cfg_raw):
            agent_cfg["checkpointer"] = dict(checkpointer)

        supervisor_cfg = config_data.get("supervisor")
        if isinstance(supervisor_cfg, dict):
            supervisor_cfg["checkpointer"] = dict(checkpointer)

    # ------------------------------------------------------------------
    # Citations (agents, supervisor, report)
    # ------------------------------------------------------------------
    if ask_bool(
        "Configure citation behaviour (agents, supervisor, and report)?",
        default=False,
    ):
        required = ask_bool("Require citations by default?", default=True)
        min_density = ask_float("Minimum citation density (0.0–1.0):", 0.1)
        require_section = ask_bool(
            "Require a reference section in long outputs?", default=True
        )

        def _apply_citations(target: Dict[str, Any]) -> None:
            citations = dict(target.get("citations") or {})
            citations["required"] = required
            citations["min_density"] = float(min_density)
            citations["require_reference_section"] = require_section
            target["citations"] = citations

        agents_cfg_raw = config_data.get("agents", [])
        for agent_cfg in iter_agent_configs(agents_cfg_raw):
            _apply_citations(agent_cfg)

        supervisor_cfg = config_data.get("supervisor")
        if isinstance(supervisor_cfg, dict):
            _apply_citations(supervisor_cfg)

        report_cfg = config_data.get("report")
        if isinstance(report_cfg, dict):
            _apply_citations(report_cfg)

    # ------------------------------------------------------------------
    # Planner (task_type, report_structure)
    # ------------------------------------------------------------------
    if ask_bool("Configure planner settings (task type, report structure)?", False):
        planner_cfg = dict(config_data.get("planner") or {})

        default_task_type = str(planner_cfg.get("task_type", "research"))
        if HAS_QUESTIONARY:
            task_type = questionary.select(
                "Planner task type:",
                choices=["research", "general", "code"],
                default=default_task_type,
            ).ask()
        else:
            task_type = ask_text(
                "Planner task type (research/general/code)", default_task_type
            )
        if task_type not in {"research", "general", "code"}:
            task_type = default_task_type
        planner_cfg["task_type"] = task_type

        default_structure = planner_cfg.get("report_structure") or ""
        structure = ask_text(
            "Optional high-level report structure (blank to skip)",
            default_structure,
        )
        planner_cfg["report_structure"] = structure or None
        config_data["planner"] = planner_cfg

    # ------------------------------------------------------------------
    # Report (enable + basic parameters)
    # ------------------------------------------------------------------
    if project_type == "team" and ask_bool(
        "Configure report generation settings?", default=False
    ):
        report_cfg = dict(config_data.get("report") or {})
        enabled_default = bool(report_cfg.get("enabled", False))
        enabled = ask_bool("Enable report generation by default?", enabled_default)
        report_cfg["enabled"] = enabled

        # Only ask for model + format when reports are enabled.
        if enabled:
            agents_cfg_raw = config_data.get("agents", [])
            first_agent_model: Dict[str, Any] = {}
            agents_list = iter_agent_configs(agents_cfg_raw)
            if agents_list:
                first_agent_model = dict(agents_list[0].get("model", {}))

            current_model = dict(report_cfg.get("model") or {})
            use_agent_model = ask_bool(
                "Use the same model as your primary agent for reports?",
                default=not bool(current_model),
            )

            if use_agent_model and first_agent_model:
                report_cfg["model"] = first_agent_model
            else:
                default_name = str(current_model.get("name") or first_agent_model.get("name", "qwen3:8b"))
                default_provider = str(
                    current_model.get("provider") or first_agent_model.get("provider", "ollama")
                )
                default_temp = float(
                    current_model.get("temperature")
                    or first_agent_model.get("temperature", 0.3)
                )

                name = ask_text("Report model name", default_name)
                provider = ask_text(
                    "Report model provider (ollama/openai/anthropic/bedrock/azure_openai)",
                    default_provider,
                )
                temperature = ask_float(
                    "Report model temperature (0.0–1.0)", default_temp
                )

                report_cfg["model"] = {
                    "name": name,
                    "provider": provider,
                    "temperature": float(temperature),
                    "other_args": current_model.get(
                        "other_args", first_agent_model.get("other_args", {})
                    ),
                }

            default_format = report_cfg.get("format") or ""
            report_cfg["format"] = ask_text(
                "Optional report format/outline (blank for none)", default_format
            ) or None

        # Output mode can be tuned regardless of enabled flag.
        default_output_mode = str(report_cfg.get("output_mode", "state"))
        if HAS_QUESTIONARY:
            output_mode = questionary.select(
                "Report output mode:",
                choices=["state", "file", "both"],
                default=default_output_mode,
            ).ask()
        else:
            output_mode = ask_text(
                "Report output mode (state/file/both)", default_output_mode
            )
        if output_mode not in {"state", "file", "both"}:
            output_mode = default_output_mode
        report_cfg["output_mode"] = output_mode

        config_data["report"] = report_cfg

    return config_data
