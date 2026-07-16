"""Prebuilt agent templates for the CLI setup wizard.

These templates are applied when an agent is named with the special
``prebuilt_*`` identifiers (e.g. ``prebuilt_research_agent``).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


PREBUILT_AGENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "prebuilt_claims_agent": {
        "prompt": (
            "You are a claims assistant.\n"
            "- Use the `get_claim_details` tool to fetch claim details.\n"
            "- Ask clarifying questions when a claim_id is missing or ambiguous.\n"
            "- Summarize findings clearly and highlight any missing information.\n"
        ),
        "tools": ["get_claim_details"],
    },
    "prebuilt_file_agent": {
        "prompt": (
            "You are a file and codebase assistant.\n"
            "- Use filesystem tools to inspect and modify files safely.\n"
            "- Prefer targeted reads (small ranges) and avoid unnecessary writes.\n"
            "- When editing, preserve formatting and minimize unrelated changes.\n"
        ),
        "tools": [
            "read_file",
            "grep_file",
            "list_directory",
            "directory_tree",
            "search_files",
            "write_file",
            "edit_file",
            "search_and_replace_file_edit",
            "create_directory",
        ],
    },
    "prebuilt_research_agent": {
        "prompt": (
            "You are a research agent.\n"
            "- Use `web_search` for general information.\n"
            "- Use `news_search` for recent news and time-sensitive updates.\n"
            "- Cite sources clearly and avoid unsupported claims.\n"
        ),
        "tools": ["web_search", "news_search", "search_historical"],
    },
}


def apply_prebuilt_agent_templates(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply prebuilt templates to any matching agents in a config mapping.

    Merges template keys into existing agent config dicts while preserving fields
    that are typically user-selected (e.g. model, next, debug).
    """

    agents = config_data.get("agents")
    if not isinstance(agents, list):
        return config_data

    for agent_cfg in agents:
        if not isinstance(agent_cfg, dict):
            continue

        name = str(agent_cfg.get("name", "")).strip()
        template: Mapping[str, Any] | None = PREBUILT_AGENT_TEMPLATES.get(name)
        if not template:
            continue

        # Only overlay the fields defined by the template; keep the rest.
        for key, value in template.items():
            agent_cfg[key] = value
        agent_cfg['name'] = agent_cfg['name'].replace('prebuilt_','')

    return config_data
