Overview
========

Andromeda is a multi-agent orchestration framework built on top of LangGraph and LangChain. 
It helps you orchestrate language models, tools, and workflows to execute complex tasks and generate structured outputs with minimal boilerplate.

At a high level, Andromeda provides:

- **Multi-agent orchestration**: Planner, supervisor, and specialist agents that coordinate complex tasks.
- **Workflow engine**: Declarative and operator-based workflows for composing reusable steps over shared state.
- **Model and configuration system**: A simple way to manage models from different providers and share configuration across agents, planners, and reporting.
- **Pre-built tooling**: Built-in tools for web search, news, context management, and report generation with validation and citations.

You can start with a single chat model or a small workflow, and grow into full agentic systems that automatically plan, route work, validate results, and complete end-to-end tasks.

For LLMs, see the `LLMs documentation <../andromeda/_static/llms.txt>`_.

