"""Environment variable generation for Andromeda CLI."""

from typing import Dict

def generate_example_env(
    interactive: bool = False,
    include_pyeztrace: bool = False,
    include_optional: bool = True,
) -> Dict[str, str]:
    """Generate example environment variables"""

    # Core Andromeda variables
    core_vars = {}

    # PyEzTrace variables
    pyeztrace_vars = {}
    if include_pyeztrace:
        pyeztrace_vars = {
            # Logging configuration
            "EZTRACE_LOG_FORMAT": "json",
            "EZTRACE_LOG_LEVEL": "INFO",
            "EZTRACE_LOG_FILE": "logs/andromeda.log",
            "EZTRACE_MAX_SIZE": "10485760",  # 10MB
            "EZTRACE_BACKUP_COUNT": "5",
            # OpenTelemetry configuration
            "EZTRACE_OTEL_ENABLED": "false",
            "EZTRACE_OTEL_EXPORTER": "otlp",
            "EZTRACE_OTLP_ENDPOINT": "http://localhost:4318/v1/traces",
            "EZTRACE_OTLP_HEADERS": "",
            "EZTRACE_SERVICE_NAME": "andromeda",
            # S3 exporter (optional)
            "EZTRACE_S3_BUCKET": "your-trace-bucket",
            "EZTRACE_S3_PREFIX": "traces/",
            "EZTRACE_S3_REGION": "us-east-1",
            "EZTRACE_COMPRESS": "true",
            # Azure Blob exporter (optional)
            "EZTRACE_AZURE_CONTAINER": "trace-container",
            "EZTRACE_AZURE_PREFIX": "traces/",
            "EZTRACE_AZURE_CONNECTION_STRING": "your_connection_string",
            "EZTRACE_AZURE_ACCOUNT_URL": "https://youraccount.blob.core.windows.net",
        }

    # Optional development and testing variables
    optional_vars = {}
    if include_optional:
        optional_vars = {
            "TAVILY_API_KEY": "your_tavily_api_key_here",
            # Langfuse configuration
            "LANGFUSE_PUBLIC_KEY": "your_langfuse_public_key_here",
            "LANGFUSE_SECRET_KEY": "your_langfuse_secret_key_here",
            "LANGFUSE_HOST": "https://your-langfuse-host.com",
            # Model provider configurations
            "OPENAI_API_KEY": "your_openai_api_key_here",
            "ANTHROPIC_API_KEY": "your_anthropic_api_key_here",
            "AWS_ACCESS_KEY_ID": "your_aws_access_key",
            "AWS_SECRET_ACCESS_KEY": "your_aws_secret_key",
            "AWS_REGION": "us-east-1",
            "AZURE_OPENAI_ENDPOINT": "https://your-endpoint.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "your_azure_openai_api_key",
        }

    # Combine all variables
    env_vars = {**core_vars, **pyeztrace_vars, **optional_vars}

    # Add comments for interactive mode
    if interactive:
        return add_env_comments(env_vars)
    else:
        return env_vars


def add_env_comments(env_vars: Dict[str, str]) -> Dict[str, str]:
    """Add descriptive comments to environment variables"""
    comments = {
        "TAVILY_API_KEY": "# Required: API key for Tavily search service if using internet search",
        # PyEzTrace comments
        "EZTRACE_LOG_FORMAT": "# Logging format: json, color, plain, csv, logfmt",
        "EZTRACE_LOG_LEVEL": "# Logging level: DEBUG, INFO, WARNING, ERROR",
        "EZTRACE_LOG_FILE": "# Path to log file",
        "EZTRACE_MAX_SIZE": "# Max log file size in bytes",
        "EZTRACE_BACKUP_COUNT": "# Number of backup log files to keep",
        "EZTRACE_OTEL_ENABLED": "# Enable OpenTelemetry tracing (true/false)",
        "EZTRACE_OTEL_EXPORTER": "# Tracing exporter: otlp, console, s3, azure",
        "EZTRACE_OTLP_ENDPOINT": "# OTLP endpoint for tracing",
        "EZTRACE_OTLP_HEADERS": "# Headers for OTLP (comma-separated key=value pairs)",
        "EZTRACE_SERVICE_NAME": "# Service name for tracing",
        "EZTRACE_S3_BUCKET": "# S3 bucket for trace exports",
        "EZTRACE_S3_PREFIX": "# S3 key prefix for traces",
        "EZTRACE_S3_REGION": "# AWS region for S3",
        "EZTRACE_COMPRESS": "# Compress trace files (true/false)",
        "EZTRACE_AZURE_CONTAINER": "# Azure container for trace exports",
        "EZTRACE_AZURE_PREFIX": "# Azure blob prefix for traces",
        "EZTRACE_AZURE_CONNECTION_STRING": "# Azure storage connection string",
        "EZTRACE_AZURE_ACCOUNT_URL": "# Azure storage account URL",
        # Optional comments
        "LANGFUSE_PUBLIC_KEY": "# Langfuse public key (optional)",
        "LANGFUSE_SECRET_KEY": "# Langfuse secret key (optional)",
        "LANGFUSE_HOST": "# Langfuse host (optional)",
        "OPENAI_API_KEY": "# OpenAI API key (optional)",
        "ANTHROPIC_API_KEY": "# Anthropic API key (optional)",
        "AWS_ACCESS_KEY_ID": "# AWS access key for Bedrock (optional)",
        "AWS_SECRET_ACCESS_KEY": "# AWS secret key for Bedrock (optional)",
        "AWS_REGION": "# AWS region (optional)",
        "AZURE_OPENAI_ENDPOINT": "# Azure OpenAI endpoint (optional)",
        "AZURE_OPENAI_API_KEY": "# Azure OpenAI API key (optional)",
    }

    result = {}
    for key, value in env_vars.items():
        if key in comments:
            result[f"{comments[key]}\n{key}"] = value
        else:
            result[key] = value

    return result
