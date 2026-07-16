"""Display and help functions for Andromeda CLI."""

from typing import Dict

from rich.table import Table

from andromeda.cli.helpers import console

def display_config_options_help():
    """Display comprehensive help for configuration options"""
    table = Table(title="Andromeda Configuration Options")

    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Option", style="magenta")
    table.add_column("Type", style="green")
    table.add_column("Default", style="yellow")
    table.add_column("Description", style="white")

    # Agent configuration
    table.add_row("Agent", "name", "str", "N/A", "Unique name for the agent")
    table.add_row(
        "Agent", "model", "ModelConfig/dict", "N/A", "Model configuration for the agent"
    )
    table.add_row(
        "Agent", "tools", "List[Callable]", "[]", "List of tools available to the agent"
    )
    table.add_row("Agent", "prompt", "str", "None", "Custom prompt for the agent")
    table.add_row(
        "Agent", "validation", "ValidationConfig", "Default", "Validation settings"
    )
    table.add_row(
        "Agent", "citations", "CitationConfig", "Default", "Citation requirements"
    )
    table.add_row("Agent", "return_direct", "bool", "False", "Return results directly")
    table.add_row("Agent", "next", "str", "None", "Next agent in chain")
    table.add_row("Agent", "type", "str", "react", "Agent type (react/codeact)")

    # Model configuration
    table.add_row("Model", "name", "str", "qwen3:8b", "Model name/identifier")
    table.add_row("Model", "provider", "str", "ollama", "Model provider")
    table.add_row("Model", "temperature", "float", "1.0", "Sampling temperature")
    table.add_row("Model", "other_args", "dict", "{}", "Additional provider args")

    # Validation configuration
    table.add_row("Validation", "enabled", "bool", "False", "Enable validation")
    table.add_row(
        "Validation", "model", "ModelConfig/str", "qwen3:4b", "Validation model"
    )
    table.add_row(
        "Validation", "skip_after_attempts", "int", "3", "Skip after N failures"
    )
    table.add_row(
        "Validation", "min_sufficiency_score", "float", "0.7", "Minimum score"
    )

    # Citation configuration
    table.add_row("Citation", "required", "bool", "False", "Require citations")
    table.add_row("Citation", "min_density", "float", "0.1", "Minimum citation density")
    table.add_row(
        "Citation", "require_reference_section", "bool", "True", "Require references"
    )

    # Supervisor configuration
    table.add_row("Supervisor", "name", "str", "supervisor", "Supervisor name")
    table.add_row("Supervisor", "model", "ModelConfig", "Default", "Supervisor model")
    table.add_row("Supervisor", "tools", "List[Callable]", "[]", "Supervisor tools")
    table.add_row("Supervisor", "prompt", "str", "Default", "Supervisor prompt")

    # Planner configuration
    table.add_row(
        "Planner", "model", "ModelConfig", "qwen3:8b, temp=0.9", "Planning model"
    )
    table.add_row("Planner", "task_type", "str", "general", "Task optimization type")
    table.add_row("Planner", "report_structure", "str", "None", "Report structure")

    # Report configuration
    table.add_row("Report", "format", "str", "None", "Output format")
    table.add_row(
        "Report", "citations", "CitationConfig", "Default", "Citation settings"
    )
    table.add_row(
        "Report", "validation", "ValidationConfig", "Default", "Validation settings"
    )

    # (No global Event configuration section in current schema)

    console.print(table)


def display_env_vars_help():
    """Display comprehensive help for environment variables"""
    table = Table(title="Andromeda Environment Variables")

    table.add_column("Variable", style="cyan", no_wrap=True)
    table.add_column("Required", style="red")
    table.add_column("Default", style="yellow")
    table.add_column("Description", style="white")

    # Core variables
    table.add_row("TAVILY_API_KEY", "Yes", "N/A", "API key for Tavily search service")
    # PyEzTrace variables [optional]
    table.add_row(
        "EZTRACE_LOG_FORMAT",
        "No",
        "json",
        "Logging format (json/color/plain/csv/logfmt)",
    )
    table.add_row(
        "EZTRACE_LOG_LEVEL", "No", "INFO", "Logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    table.add_row("EZTRACE_LOG_FILE", "No", "logs/andromeda.log", "Path to log file")
    table.add_row("EZTRACE_MAX_SIZE", "No", "10485760", "Max log file size in bytes")
    table.add_row("EZTRACE_BACKUP_COUNT", "No", "5", "Number of backup log files")
    table.add_row("EZTRACE_OTEL_ENABLED", "No", "false", "Enable OpenTelemetry tracing")
    table.add_row(
        "EZTRACE_OTEL_EXPORTER",
        "No",
        "otlp",
        "Tracing exporter (otlp/console/s3/azure)",
    )
    table.add_row(
        "EZTRACE_OTLP_ENDPOINT",
        "No",
        "http://localhost:4318/v1/traces",
        "OTLP endpoint",
    )
    table.add_row(
        "EZTRACE_OTLP_HEADERS", "No", "", "Headers for OTLP (key=value,key=value)"
    )
    table.add_row("EZTRACE_SERVICE_NAME", "No", "andromeda", "Service name for tracing")
    table.add_row("EZTRACE_S3_BUCKET", "No", "", "S3 bucket for trace exports")
    table.add_row("EZTRACE_S3_PREFIX", "No", "traces/", "S3 key prefix")
    table.add_row("EZTRACE_S3_REGION", "No", "us-east-1", "AWS region for S3")
    table.add_row("EZTRACE_COMPRESS", "No", "true", "Compress trace files")
    table.add_row("EZTRACE_AZURE_CONTAINER", "No", "", "Azure container for traces")
    table.add_row("EZTRACE_AZURE_PREFIX", "No", "traces/", "Azure blob prefix")
    table.add_row(
        "EZTRACE_AZURE_CONNECTION_STRING", "No", "", "Azure connection string"
    )
    table.add_row("EZTRACE_AZURE_ACCOUNT_URL", "No", "", "Azure storage account URL")

    # Optional variables
    table.add_row("OPENAI_API_KEY", "No", "", "OpenAI API key")
    table.add_row("ANTHROPIC_API_KEY", "No", "", "Anthropic API key")
    table.add_row("AWS_ACCESS_KEY_ID", "No", "", "AWS access key for Bedrock")
    table.add_row("AWS_SECRET_ACCESS_KEY", "No", "", "AWS secret key for Bedrock")
    table.add_row("AWS_REGION", "No", "us-east-1", "AWS region")
    table.add_row("AZURE_OPENAI_ENDPOINT", "No", "", "Azure OpenAI endpoint")
    table.add_row("AZURE_OPENAI_API_KEY", "No", "", "Azure OpenAI API key")

    console.print(table)




def display_env_table(env_data: Dict[str, str]):
    """Display environment variables in a formatted table"""
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in env_data.items():
        # Handle commented keys
        if "\n" in key:
            parts = key.split("\n")
            comment = parts[0]
            var_name = parts[1] if len(parts) > 1 else key
            table.add_row(var_name, value, style="dim")
            # Add comment as a separate row
            table.add_row("", comment, style="yellow")
        else:
            table.add_row(key, value)

    console.print(table)
