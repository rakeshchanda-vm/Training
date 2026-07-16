# Repository Guidelines

## Project Structure & Module Organization

- `andromeda/`: main Python package.
  - `andromeda/core/`: orchestration (team/supervisor/planner/workflows)
  - `andromeda/tools/`: tool adapters (filesystem, MCP, integrations)
  - `andromeda/cli/`: Click-based CLI (`andromeda …`)
  - `andromeda/reporting/`: report synthesis, mermaid helpers
  - `andromeda/config/`: config models + YAML helpers
  - `andromeda/utils/`: logging, prompts, schemas, sandbox helpers
- `tests/`: `pytest` suite (unit + integration-style tests).
- `docs/`: Sphinx docs (`make -C docs html`).
- `andromeda/examples/`: runnable scripts (e.g. `andromeda/examples/ollama_workflow_scoring.py`).

## Build, Test, and Development Commands

Run from the repo root (one level above this file): `cd ..`

- `python -m pip install -e ".[dev]"`: editable install with formatter/type-check deps.
- `python -m pip install -r requirements.txt`: install pinned runtime deps (non-editable).
- `andromeda --help`: CLI entrypoint; try `andromeda generate-config` / `andromeda generate-env`.
- `pytest`: run test suite (configured in `pyproject.toml`).
- `black .` / `isort .`: auto-format code (Black 88-char lines, isort Black profile).
- `mypy andromeda`: strict type-checking (tests are excluded from strictness).

## Coding Style & Naming Conventions

- Python 3.11+; prefer explicit types (mypy is configured with strict options).
- 4-space indentation; `snake_case` for functions/files, `PascalCase` for classes.
- Prefer absolute imports (`from andromeda.<module> import …`) and keep modules small.

## Testing Guidelines

- Use `pytest`; name tests `test_*.py` and group by feature under `tests/`.
- Add/adjust tests for bug fixes and behavior changes; keep network calls mocked unless
  explicitly testing integrations.

## Commit & Pull Request Guidelines

- Follow existing subject patterns: `feat: …`, `Fix: …`, `Refactor: …`, `Doc …`; include
  issue/PR references when relevant (e.g. `(#42)`).
- PRs should include: purpose + scope, reproduction/verification steps, linked issues,
  and screenshots for CLI output changes.

## Security & Configuration Tips

- Never commit secrets. Use a local `.env` (e.g. `TAVILY_API_KEY=…`; optional `LANGFUSE_*`).
- Prefer generating templates via CLI: `andromeda generate-env` and `andromeda generate-config`.
