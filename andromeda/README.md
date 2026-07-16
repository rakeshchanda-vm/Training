# Andromeda Framework

A powerful multi-agent framework for conducting comprehensive research and analysis using LLMs and search tools.

## Overview

Andromeda is a sophisticated framework that orchestrates multiple AI agents to collaboratively work on complex research and analysis tasks. It leverages LangChain, LangGraph, and various LLMs to create a hierarchical team structure with specialized agents working under a supervisor.

## Architecture

High-level workflow:
```
                                       +----------------+
                                       |   User Input   |
                                       +----------------+
                                              |
                                              v
                                       +----------------+
                                       |     Team       |
                                       +----------------+
                                              |
                                              v
                                  +------------------------+
                                  |    Supervisor Agent    |
                                  +------------------------+
                                              |
                    +-------------------------+-------------------------+
                    |                         |                        |
                    v                         v                        v
            +--------------+          +--------------+         +--------------+
            |   Agent 1    |          |   Agent 2    |         |   Agent 3    |
            +--------------+          +--------------+         +--------------+
                    |                         |                        |
                    |                         |                        |
                    +-------------------------+------------------------+
                                              |
                                              v
                                  +------------------------+
                                  |    Report Writer       |
                                  +------------------------+
                                              |
                                              v
                                  +------------------------+
                                  |    Final Report        |
                                  +------------------------+
```

Detailed component interaction:
```
+------------------------------------------------------------------------------+
|                                FASTAGENTS WORKFLOW                              |
+------------------------------------------------------------------------------+
|                                                                              |
|  [User Input] --> [Team] ----+                                              |
|                              |                                               |
|                              v                                               |
|                     +------------------+                                     |
|                     | Supervisor Agent |<-----------------+                  |
|                     +------------------+                  |                  |
|                              |                           |                   |
|                              v                           |                   |
|                        [Task Router]                     |                   |
|                              |                           |                   |
|               +-----------------------------+            |                   |
|               |              |              |            |                   |
|               v              v              v            |                   |
|         [Agent 1]      [Agent 2]      [Agent 3]         |                  |
|               |              |              |            |                   |
|               +-----------------------------+            |                   |
|                              |                          |                   |
|                              v                          |                   |
|                     [Result Validator]------------------+                   |
|                              |                                              |
|                              v                                              |
|                     [Report Writer]                                         |
|                              |                                              |
|                              v                                              |
|                     [Final Report]                                          |
|                                                                            |
+----------------------------------------------------------------------------+
```

State flow diagram:
```
     +--------+    Plan     +-----------+   Tasks    +--------+
     |  Task  |----------->| Supervisor |---------->| Agents |
     +--------+            +-----------+            +--------+
                               ^  |                     |
                               |  |                     |
                          Facts|  |Validation           |Results
                               |  v                     v
                          +---------+              +----------+
                          | Planner |              | Writer   |
                          +---------+              +----------+
                                                       |
                                                       v
                                                  [Report]
```

### Core Components

1. **Team**
   - Entry point for the framework
   - Manages agent workflow and coordination
   - Creates a state graph for task execution

2. **Supervisor Agent**
   - Orchestrates the workflow
   - Routes tasks to appropriate agents
   - Validates task completion
   - Handles task summarization and context management

3. **Planner**
   - Generates detailed action plans
   - Validates plan feasibility
   - Ensures comprehensive task coverage

4. **Agents**
   - Specialized workers with specific expertise
   - Execute assigned tasks using tools
   - Validate and cite their findings
   - Support recursive feedback loops for quality improvement

5. **Report Writer**
   - Synthesizes agent findings into coherent reports
   - Manages citations and references
   - Structures content with sections and subsections
   - Handles token limits and context windows

### Tools System

The framework includes various tools for research and analysis:
- Internet search
- News article search
- Website scraping
- Domain-specific search capabilities

## Workflow

1. **Task Initialization**
   - User submits a research/analysis task
   - Team initializes workflow state

2. **Planning Phase**
   - Planner generates detailed action steps
   - Supervisor validates and approves plan

3. **Execution Phase**
   - Supervisor routes tasks to appropriate agents
   - Agents execute tasks using available tools
   - Results are validated for citations and quality

4. **Report Generation**
   - Writer consolidates findings
   - Structures content into sections
   - Validates citations and references
   - Produces final markdown report

## Features

- **Hierarchical Management**: Supervisor-worker model for efficient task distribution
- **Quality Control**: Multiple validation layers for accuracy and completeness
- **Citation Management**: Automated tracking and standardization of sources
- **Context Management**: Smart handling of token limits and conversation history
- **Flexible Architecture**: Easy to add new agents and tools

## Usage

```python
from andromeda.agent import Agent
from andromeda.team import Team
from andromeda.tools import news_search, web_search

# Initialize agents with specific roles and tools
workers = {
    "finance_analyst": Agent("finance_analyst", tools=[news_search, web_search]),
    "market_analyst": Agent("market_analyst", tools=[news_search, web_search]),
    # Add more specialized agents as needed
}

# Create team and execute task
team = Team(workers)
response = team.begin("Your research task here")
```

## Dependencies

- langchain
- langgraph
- tiktoken
- colorama
- pydantic

## Installation

```bash
pip install -r requirements.txt
```

## Environment Setup

Create a .env file with required API keys:
```
TAVILY_API_KEY=your_api_key_here
```

### Langfuse (optional)

If you want Langfuse tracing/scoring, install `langfuse` and set:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### WorkflowBuilder scoring (optional)

Langfuse score ingestion is opt-in per node by attaching evaluators. Evaluators run in the background and do not block workflow execution:
```python
from andromeda.core.workflow import WorkflowBuilder, LangfuseEvaluator

def correctness_eval(inp):
    # Return NUMERIC, CATEGORICAL, or BOOLEAN (bool / 0/1) values.
    return 0.9

wf = WorkflowBuilder(name="MyWorkflow")
(
    wf.start("step_1")
      .run(lambda payload: payload)
      .with_evaluators(
          [LangfuseEvaluator(name="correctness", data_type="NUMERIC", eval_fn=correctness_eval)],
          scheduler={"max_workers": 2, "max_pending": 128},
          model={"name": "gpt-4.1-mini", "provider": "openai", "temperature": 0.0, "other_args": {}},
      )
)
```

Tuning for production:
```
# Evaluator scheduler + model are configured per node via with_evaluators(...).
```

Notes:
- Score names are automatically scoped by node name (e.g. `refine:correctness`) to keep per-node metrics easy to interpret.
- When Langfuse tracing is enabled, Andromeda derives a deterministic Langfuse `trace_id` from the workflow `thread_id` and uses it for both tracing and scoring (the Langfuse trace id is a 32-hex string; it won’t equal `thread_id`).
- The root workflow span updates the trace with the full input and output state (`update_trace(input=..., output=...)`).

## Extending the Framework

### Adding New Agents

```python
new_agent = Agent(
    name="custom_agent",
    model_name="your_model",
    tools=[your_tools],
    prompt="Custom prompt"
)
```

### Creating Custom Tools

```python
from andromeda.tools import tool

@tool
def custom_tool(query: str) -> str:
    """Tool description"""
    # Implementation
    return result
```

## Best Practices

1. Provide detailed tasks for better planning
2. Configure appropriate token limits
3. Include relevant tools for each agent
4. Monitor execution with logging
5. Validate outputs and citations

## Logging

The framework includes comprehensive logging with different levels:
- Supervisor logs (blue)
- Agent logs (green)
- Tool logs (yellow)
- Output logs (magenta)
- Error logs (red)
