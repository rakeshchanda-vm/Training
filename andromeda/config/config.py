from __future__ import annotations
from typing import Dict, List, Literal, Optional, Any, Type, Union, Mapping
from pathlib import Path
from langchain.tools import BaseTool
from langchain.agents import AgentState
from pydantic import BaseModel, Field, ValidationError, model_validator

from andromeda.config.yaml_utils import yaml_dump, yaml_load


def _normalize_config_object(
    data: Any,
    *,
    source: Union[str, Path],
    interpolate_env: bool = True,
    register_mcp: bool = True,
    resolve_tools: bool = True,
    env: Optional[Mapping[str, str]] = None,
    toolkit: Optional[Any] = None,
    execution_context: Optional[Any] = None,
    mcp_runtime: Optional[Any] = None,
) -> Any:
    """
    Normalize a parsed JSON/YAML object for Andromeda config consumption.

    This function is intentionally tolerant: it will only apply transforms to
    keys/sections that exist, allowing partial configs to be processed and reused
    across different entrypoints (e.g. loading just a SupervisorConfig).
    """
    import os
    import re

    src = str(source)
    effective_env = env
    if effective_env is None and execution_context is not None:
        effective_env = getattr(execution_context, "env", None)
    effective_toolkit = toolkit
    if effective_toolkit is None and execution_context is not None:
        effective_toolkit = getattr(execution_context, "toolkit", None)
    effective_mcp_runtime = mcp_runtime
    if effective_mcp_runtime is None and execution_context is not None:
        effective_mcp_runtime = getattr(execution_context, "mcp_runtime", None)

    def _ensure_execution_toolkit() -> Any:
        nonlocal effective_toolkit
        if effective_toolkit is not None:
            return effective_toolkit
        if execution_context is None and effective_mcp_runtime is None:
            return None

        from andromeda.tools.toolkit import Toolkit, get_default_toolkit

        effective_toolkit = Toolkit(parent=get_default_toolkit())
        if execution_context is not None:
            execution_context.toolkit = effective_toolkit
        return effective_toolkit

    def _tool_resolution_toolkit() -> Any:
        if effective_toolkit is None:
            return None

        from andromeda.tools.toolkit import Toolkit, get_default_toolkit

        layered_toolkit = Toolkit(parent=get_default_toolkit())
        for name, tool in effective_toolkit.all().items():
            layered_toolkit.register(tool, name=name, override=True)
        return layered_toolkit

    def _format_path(p: List[Union[str, int]]) -> str:
        if not p:
            return "<root>"
        out: List[str] = []
        for part in p:
            if isinstance(part, int):
                out.append(f"[{part}]")
            else:
                out.append(f".{part}")
        s = "".join(out)
        return s[1:] if s.startswith(".") else s

    _DOLLAR_BRACE_ENV_PATTERN = re.compile(r"""\$\{([A-Za-z_][A-Za-z0-9_]*)\}""")

    def _interpolate_string(value: str, path_parts: List[Union[str, int]]) -> str:
        def _brace_repl(match: re.Match[str]) -> str:
            var = match.group(1)
            env_val = None
            if effective_env is not None:
                env_val = effective_env.get(var)
            if env_val is None:
                env_val = os.environ.get(var)
            if env_val is not None:
                return env_val
            raise ValueError(
                "Missing environment variable while loading config.\n"
                f"- source: {src}\n"
                f"- variable: {var}\n"
                f"- location: {_format_path(path_parts)}\n"
                f"- expression: {match.group(0)}"
            )

        return _DOLLAR_BRACE_ENV_PATTERN.sub(_brace_repl, value)

    def _interpolate_env(obj: Any, path_parts: Optional[List[Union[str, int]]] = None) -> Any:
        """
        Recursively interpolate env vars in a parsed JSON/YAML object.

        Supported forms:
        - "${VAR}"
        """
        path_parts = [] if path_parts is None else path_parts
        if isinstance(obj, dict):
            return {k: _interpolate_env(v, [*path_parts, str(k)]) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_interpolate_env(v, [*path_parts, idx]) for idx, v in enumerate(obj)]
        if isinstance(obj, tuple):
            return tuple(_interpolate_env(v, [*path_parts, idx]) for idx, v in enumerate(obj))
        if isinstance(obj, str):
            return _interpolate_string(obj, path_parts)
        return obj

    def _resolve_dotted_object(obj: Any) -> Any:
        if isinstance(obj, str):
            import importlib
            return importlib.import_module(obj)
        return obj

    if interpolate_env and isinstance(data, (dict, list)):
        data = _interpolate_env(data)

    if not isinstance(data, dict):
        # Nothing further to normalize (MCP/tool resolution is keyed off mappings).
        return data

    # Allow MCP adapters to register their tools into the global Toolkit
    # before we resolve tool specifications by name. This enables users to
    # reference MCP-provided tools in the same way as built-in tools.
    if register_mcp:
        mcp_servers_raw = data.get("mcp_servers")
        if isinstance(mcp_servers_raw, dict) and mcp_servers_raw:
            if (
                effective_toolkit is None
                and effective_mcp_runtime is None
                and execution_context is None
            ):
                from andromeda.tools.mcp_adapter import register_mcp as _register_mcp

                _register_mcp(mcp_servers_raw)
            else:
                from andromeda.tools.mcp_adapter import build_mcp_tools as _build_mcp_tools

                scoped_toolkit = _ensure_execution_toolkit()

                _build_mcp_tools(
                    mcp_servers_raw,
                    execution_context=execution_context,
                    toolkit=scoped_toolkit,
                    runtime=effective_mcp_runtime,
                    register=True,
                )
        elif isinstance(mcp_servers_raw, list) and mcp_servers_raw:
            merged: Dict[str, Any] = {}
            for mcp_server in mcp_servers_raw:
                if isinstance(mcp_server, dict):
                    if (
                        effective_toolkit is None
                        and effective_mcp_runtime is None
                        and execution_context is None
                    ):
                        from andromeda.tools.mcp_adapter import register_mcp as _register_mcp

                        _register_mcp(mcp_server)
                    else:
                        from andromeda.tools.mcp_adapter import build_mcp_tools as _build_mcp_tools

                        scoped_toolkit = _ensure_execution_toolkit()

                        _build_mcp_tools(
                            mcp_server,
                            execution_context=execution_context,
                            toolkit=scoped_toolkit,
                            runtime=effective_mcp_runtime,
                            register=True,
                        )
                    for name, cfg in mcp_server.items():
                        merged[name] = cfg
            data["mcp_servers"] = merged

    if resolve_tools:
        # Importing here to avoid circular import at module import time
        from andromeda.tools.toolkit import resolve_tool_spec

        # Import built-in tools so they self-register with the Toolkit.
        try:  # noqa: BLE001
            import andromeda.tools.tools as _builtin_tools  # type: ignore[unused-import]
        except ImportError:
            _builtin_tools = None  # type: ignore[assignment]
        resolution_toolkit = _tool_resolution_toolkit()

        def _resolve_tools_for_mapping(agent_name: str, mapping: Mapping[str, Any]) -> None:
            tools = mapping.get("tools")
            if not tools:
                return
            if not isinstance(tools, list):
                raise ValueError(
                    f"Invalid tools specification for '{agent_name}' in {src}: "
                    f"expected a list, got {type(tools).__name__}."
                )

            resolved_tools: List[Any] = []
            for spec in tools:
                try:
                    resolved_tools.append(resolve_tool_spec(spec, toolkit=resolution_toolkit))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to resolve tool specification {spec!r} for '{agent_name}' "
                        f"from {src}. Details: {exc}"
                    ) from exc

            # mapping is a Mapping at type-level, but in practice it's dicts from JSON/YAML.
            if isinstance(mapping, dict):
                mapping["tools"] = resolved_tools

        # Root can itself be a (supervisor/agent) config dict.
        if "tools" in data:
            name = str(data.get("name", "<root>"))
            _resolve_tools_for_mapping(name, data)

        # Common Andromeda top-level sections.
        agents_raw = data.get("agents")
        
        def _resolve_agent_data(agent_data: Dict[str, Any]) -> None:
            agent_name = str(agent_data.get("name", "<root>"))
            state_schema = agent_data.get("state_schema")
            if state_schema is not None:
                try:
                    agent_data["state_schema"] = _resolve_dotted_object(state_schema)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        f"Failed to resolve state_schema {state_schema!r} for agent "
                        f"'{agent_name}'. Use a dotted import path, e.g. "
                        f"'my_pkg.schemas.CustomAgentState'. Details: {exc}"
                    ) from exc

            middleware = agent_data.get("middleware")
            if isinstance(middleware, dict):
                custom = middleware.get("custom")
                if isinstance(custom, list):
                    resolved_custom = []
                    for entry in custom:
                        try:
                            resolved_custom.append(_resolve_dotted_object(entry))
                        except Exception as exc:  # noqa: BLE001
                            raise ValueError(
                                f"Failed to resolve middleware.custom entry {entry!r} for "
                                f"agent '{agent_name}'. Use a dotted import path, e.g. "
                                f"'my_pkg.middleware.custom_guardrail'. Details: {exc}"
                            ) from exc
                    middleware["custom"] = resolved_custom
                agent_data["middleware"] = middleware

        if isinstance(agents_raw, list):
            for idx, agent_cfg in enumerate(agents_raw):
                if isinstance(agent_cfg, dict):
                    name = str(agent_cfg.get("name", f"agent_{idx}"))
                    _resolve_tools_for_mapping(name, agent_cfg)
                    _resolve_agent_data(agent_cfg)

        elif isinstance(agents_raw, dict):
            for name, agent_cfg in agents_raw.items():
                if isinstance(agent_cfg, dict):
                    _resolve_tools_for_mapping(str(name), agent_cfg)
                    _resolve_agent_data(agent_cfg)

        supervisor_raw = data.get("supervisor")
        if isinstance(supervisor_raw, dict):
            _resolve_tools_for_mapping(str(supervisor_raw.get("name", "supervisor")), supervisor_raw)
            _resolve_agent_data(supervisor_raw)

    return data


class ModelConfig(BaseModel):
    """
    Configuration model for specifying language model parameters.
    
    Attributes:
        name (str): The name or identifier of the model.
        provider (str): The provider or backend service for the model. 
        base_url (str): The base URL for the model API endpoint. Defaults to an empty string.
        output_version (str): The version of the model output format. Defaults to "v1". Use v0 for legacy langchain output format.
        context_window (int): Maximum number of tokens in the model's context window. Defaults to 40960.

    Deprecation Warning:
        - temperature is deprecated in favor of other_args.
    """

    name: str
    provider: str
    output_version: str = Field(default="v1")
    temperature: float = Field(default=1.0)
    other_args: dict = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class ValidationConfig(BaseModel):
    """
    Configuration class for validation settings.

    Attributes:
        enabled (bool): Flag to enable or disable validation. Defaults to True.
        model (Optional[ModelConfig]): The model configuration to use for validation.
        skip_after_attempts (int): Number of failed attempts after which validation is skipped. Defaults to 3.
        min_sufficiency_score (float): Minimum score required to consider validation sufficient. Defaults to 0.7.

    Config:
        arbitrary_types_allowed (bool): Allows arbitrary types in Pydantic model fields.
    """

    enabled: bool = Field(default=False)
    model: Optional[ModelConfig] = Field(
        default=None,
        description="Model config to use for validation when enabled.",
    )
    skip_after_attempts: int = Field(default=3)
    min_sufficiency_score: float = Field(default=0.7)
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_model_when_enabled(self) -> "ValidationConfig":
        """Returns an error when validation is enabled but no model is set."""

        if self.enabled and self.model is None:
            raise ValueError(
                "Invalid ValidationConfig: 'enabled' is True but 'model' is not set.\n"
                "Either disable validation (validation.enabled = False) or provide a "
                "ModelConfig in validation.model."
            )
        return self


class CitationConfig(BaseModel):
    """
    Configuration settings for citation requirements.

    Attributes:
        required (bool): Indicates whether citations are required. Defaults to True.
        min_density (float): Minimum citation density (as a fraction) required in the content. Defaults to 0.1.
        require_reference_section (bool): Specifies if a reference section is mandatory. Defaults to True.
    """

    required: bool = Field(default=False)
    min_density: float = Field(default=0.1)
    require_reference_section: bool = Field(default=False)
    model_config = {"arbitrary_types_allowed": True}


class PlannerConfig(BaseModel):
    """
    Configuration class for the planner component.

    Attributes:
        model (ModelConfig): The model configuration to use for planning.
        report_structure (Optional[str]): Optional string specifying the structure of the report,
            typically used for research-style long-horizon tasks.
        task_type (Literal["research", "general", "code"]): High-level type of long-horizon
            task the planner should optimize for. This is surfaced into prompts so that the
            same planner can be reused for different kinds of workflows. Defaults to "general".
    """

    model: ModelConfig
    report_structure: Optional[str] = Field(default=None)
    task_type: Literal["research", "general", "code"] = Field(default="general")
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_model(self) -> "PlannerConfig":
        """Returns an error when model is not set."""
        if self.model is None:
            raise ValueError("Invalid PlannerConfig: 'model' is not set.")
        return self


class DataPatternsConfig(BaseModel):
    """
    Regex patterns used by DataPrivacyMiddleware.

    Keep defaults minimal to reduce false positives; extend via extra_patterns.
    """

    email: str = Field(
        default=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    ssn: str = Field(default=r"\b\d{3}-\d{2}-\d{4}\b")
    phone: str = Field(
        default=r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    )
    credit_card: str = Field(default=r"\b(?:\d[ -]*?){13,19}\b")
    extra_patterns: Dict[str, str] = Field(default_factory=dict)
    model_config = {"arbitrary_types_allowed": True}


class PromptInjectionPatternsConfig(BaseModel):
    """
    Regex patterns used by PromptInjectionMiddleware.
    """

    patterns: List[str] = Field(
        default_factory=lambda: [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"disregard\s+(the\s+)?system\s+prompt",
            r"reveal\s+(the\s+)?system\s+prompt",
            r"bypass\s+(safety|guardrails|policy|security)",
            r"disable\s+(safety|guardrails|policy|security)",
        ]
    )
    model_config = {"arbitrary_types_allowed": True}


class CompliancePatternsConfig(BaseModel):
    """
    Regex patterns used by ComplianceMiddleware.
    """

    patterns: List[str] = Field(
        default_factory=lambda: [
            r"\bguaranteed\s+(approval|coverage|payout|returns?)\b",
            r"\bno\s+(exclusions|conditions|limitations)\b",
            r"\bcannot\s+be\s+denied\b",
            r"\bfalsif(?:y|ied|ication)\b",
            r"\bmisrepresent(?:ation)?\b",
        ]
    )
    model_config = {"arbitrary_types_allowed": True}


class MiddlewareConfig(BaseModel):
    """
    Optional middleware settings for LangChain v1 agents.

    All middleware is opt-in. When ``enabled`` is False, agent middleware is disabled.
    """

    class SummarizationOptions(BaseModel):
        keep: int = Field(default=20)
        trigger_tokens: int = Field(default=1000, ge=1)
        model: Optional[Union[str, "ModelConfig"]] = Field(default=None)
        model_config = {"arbitrary_types_allowed": True}

    class HITLOptions(BaseModel):
        interrupt_on: Optional[Dict[str, Any]] = Field(default=None)
        model_config = {"arbitrary_types_allowed": True}

    class GuardrailOptions(BaseModel):
        input: bool = Field(default=False)
        output: bool = Field(default=False)
        tool: bool = Field(default=False)
        data_patterns: DataPatternsConfig = Field(
            default_factory=DataPatternsConfig
        )
        prompt_injection_patterns: PromptInjectionPatternsConfig = Field(
            default_factory=PromptInjectionPatternsConfig
        )
        compliance_patterns: CompliancePatternsConfig = Field(
            default_factory=CompliancePatternsConfig
        )
        blocked_message: str = Field(default="This request was flagged by Andromeda and cannot be processed.")
        model_config = {"arbitrary_types_allowed": True}

    class MaskingOptions(BaseModel):
        input: bool = Field(default=False)
        output: bool = Field(default=False)
        tool: bool = Field(default=False)
        strategy: Literal["redact", "mask", "hash", "tokenize"] = Field(default="redact")
        data_patterns: DataPatternsConfig = Field(
            default_factory=DataPatternsConfig
        )
        token_prefix: str = Field(default="pii")
        token_ttl_seconds: Optional[int] = Field(default=24 * 60 * 60)
        model_config = {"arbitrary_types_allowed": True}

    enabled: Optional[bool] = Field(default=None)
    tool_error_handler: bool = Field(default=False)
    summarization: Optional[SummarizationOptions] = Field(default=None)
    hitl: Optional[HITLOptions] = Field(default=None)
    guardrails: GuardrailOptions = Field(default_factory=GuardrailOptions)
    masking: MaskingOptions = Field(default_factory=MaskingOptions)
    custom: List[Any] = Field(default_factory=list)
    model_config = {"arbitrary_types_allowed": True}


class CheckpointerConfig(BaseModel):
    """
    Configuration for LangGraph persistence checkpointers.

    ``in-memory`` preserves the existing ephemeral checkpoint behavior.
    ``none`` disables LangGraph checkpoint persistence.
    ``postgres`` requires ``connection_string`` and is resolved by the runtime
    checkpointing layer.
    """

    backend: Literal["none", "in-memory", "postgres"] = Field(default="in-memory")
    connection_string: Optional[str] = Field(default=None)
    setup: bool = Field(default=False)
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def _normalize_shorthand(cls, data: Any) -> Any:
        """Allow ``checkpointer: in-memory`` YAML shorthand."""

        if isinstance(data, str):
            return {"backend": data}
        return data

    @model_validator(mode="after")
    def _check_postgres_connection(self) -> "CheckpointerConfig":
        if self.backend == "postgres" and not self.connection_string:
            raise ValueError(
                "Invalid CheckpointerConfig: 'connection_string' is required "
                "when checkpointer.backend is 'postgres'."
            )
        return self


class AgentConfig(BaseModel):
    """
    Configuration class for defining an agent's settings.

    Attributes:
        name (str): The name of the agent.
        model (Union[ModelConfig, str]): The model configuration or model name to use. Defaults to a new ModelConfig instance.
        tools (List[Callable]): A list of callable tools available to the agent. Defaults to an empty list.
        prompt (Optional[str]): An optional prompt string for the agent.
        debug (int): Debug level. Defaults to 0. Legend=0: none, 1: io only, 2: full tracing, 3: full tracing with langgraph debug
        recursion_limit (Optional[int]): Maximum recursion limit. Defaults to 25.
        response_format (Optional[Any]): An optional response format for structured output. Can be a TypedDict or Pydantic model.
        validation (ValidationConfig): Configuration for validation. Defaults to a new ValidationConfig instance.
        citations (CitationConfig): Configuration for citations. Defaults to a new CitationConfig instance.
        return_direct (bool): Whether the agent should return results directly. Defaults to True.
        next (Optional[str]): The name of the next agent or step, if any.
        checkpointer (CheckpointerConfig): LangGraph persistence settings.
    """

    name: str
    model: Union[ModelConfig, str]
    tools: List[BaseTool] = Field(default_factory=list)
    prompt: Optional[str] = Field(default="")
    response_format: Optional[Union[Type[BaseModel], Dict, None]] = Field(default=None)
    debug: int = Field(default=0)
    recursion_limit: Optional[int] = Field(default=25)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    citations: CitationConfig = Field(default_factory=CitationConfig)
    return_direct: bool = Field(default=True)
    next: Optional[str] = Field(default=None)
    type: Literal["react", "codeact"] = Field(default="react")
    output_standard: Literal["andromeda", "langchain"] = Field(default="andromeda")
    state_schema: Optional[Any] = Field(
        default=None,
        description="Must be a subclass of AgentState."
    )
    middleware: MiddlewareConfig = Field(default_factory=MiddlewareConfig)
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_model(self) -> "AgentConfig":
        """Returns an error when model is not set."""

        if self.model is None:
            raise ValueError(
                "Invalid AgentConfig: 'model' is not set."
            )
        return self


class SupervisorConfig(AgentConfig):
    """
    Configuration class for the supervisor agent.
    
    This extends :class:`AgentConfig` with supervisor-specific behavior while
    reusing all core agent settings (model, tools, debug, validation, etc.).
    """
    
    # Override some defaults for the supervisor role.
    name: str = Field(default="supervisor")
    recursion_limit: Optional[int] = Field(default=30)
    type: Literal["react"] = Field(default="react")
    allowed_route_types: List[Literal["chat", "task", "research", "handoff"]] = Field(default=["task", "research"])
    allow_parallel_agents: bool = Field(default=False)
    allow_async_tasks: bool = Field(default=False)
    enable_planning: bool = Field(default=True)
    middleware: MiddlewareConfig = Field(
        default_factory=lambda: MiddlewareConfig(tool_error_handler=True)
    )

    @classmethod
    def load_from_file(
        cls,
        path: str,
        *,
        interpolate_env: bool = True,
        register_mcp: bool = True,
        resolve_tools: bool = True,
        env: Optional[Mapping[str, str]] = None,
        toolkit: Optional[Any] = None,
        execution_context: Optional[Any] = None,
        mcp_runtime: Optional[Any] = None,
    ) -> "SupervisorConfig":
        """
        Load a SupervisorConfig from a JSON/YAML file.

        The file can either be:
        - a supervisor-only config mapping (fields of SupervisorConfig at the root), or
        - a full Andromeda-style config mapping containing a top-level "supervisor" key.
        """
        import json

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path_obj}")

        if path_obj.suffix == ".json":
            with open(path_obj) as f:
                data = json.load(f)
        elif path_obj.suffix in (".yml", ".yaml"):
            with open(path_obj) as f:
                data = yaml_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")

        return cls.load_from_config(
            data,
            source=path_obj,
            interpolate_env=interpolate_env,
            register_mcp=register_mcp,
            resolve_tools=resolve_tools,
            env=env,
            toolkit=toolkit,
            execution_context=execution_context,
            mcp_runtime=mcp_runtime,
        )

    @classmethod
    def load_from_config(
        cls,
        data: Any,
        *,
        source: Optional[Union[str, Path]] = None,
        interpolate_env: bool = True,
        register_mcp: bool = True,
        resolve_tools: bool = True,
        env: Optional[Mapping[str, str]] = None,
        toolkit: Optional[Any] = None,
        execution_context: Optional[Any] = None,
        mcp_runtime: Optional[Any] = None,
    ) -> "SupervisorConfig":
        """
        Load a SupervisorConfig from an in-memory object (dict from JSON/YAML).

        Supports both shapes:
        - {"supervisor": {...}, "mcp_servers": ...}  (extracts "supervisor")
        - {...supervisor fields...}                 (treats root as supervisor config)
        """
        src = source if source is not None else "<in-memory config>"
        resolved = _normalize_config_object(
            data,
            source=src,
            interpolate_env=interpolate_env,
            register_mcp=register_mcp,
            resolve_tools=resolve_tools,
            env=env,
            toolkit=toolkit,
            execution_context=execution_context,
            mcp_runtime=mcp_runtime,
        )

        if isinstance(resolved, dict) and isinstance(resolved.get("supervisor"), dict):
            payload: Any = resolved["supervisor"]
        else:
            payload = resolved

        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid SupervisorConfig parsed from {src}: expected a mapping (dict)."
            )

        try:
            return cls(**payload)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid SupervisorConfig parsed from {src}.\n"
                f"Please check required fields on the supervisor config.\n"
                f"Original validation error:\n{exc}"
            ) from exc


class WorkspaceAgentConfig(SupervisorConfig):
    """
    Configuration class for the workspace agent.

    This extends :class:`SupervisorConfig` with workspace-specific behavior while reusing
    all core agent and supervisor settings (model, tools, routing, debug, validation, etc.).
    A workspace agent drives long-horizon tasks inside an isolated workspace session that
    provisions the concrete tools (filesystem, shell, ...) it operates with, and — like a
    supervisor — can route sub tasks to specialist agents.

    Attributes:
        coworker_tools (List[BaseTool]): Tools handed to the auto-spawned coworker agents.
            These are merged with the coworker's workspace session tools. When empty
            (default), coworkers receive only the session tools selected by
            ``coworker_tool_profile``.
        coworker_tool_profile (Literal["default", "read_only"]): Workspace tool profile
            for auto-spawned coworkers. ``"default"`` gives coworkers the same session
            tools as the supervisor; ``"read_only"`` gives coworkers read-only tools for
            the same workspace session.
        read_only (bool): Place the workspace in read-only mode. When enabled, the workspace
            is created with ``read_only=True`` and ``enable_shell=False``.
        skill_sources (Optional[List[str]]): Skill source paths (e.g. ``['/skills']``).
            When provided, the skills middleware is attached so the agent can discover and use
            skills from these paths; when None (default) the skills middleware is not added.
        skills_backend (Literal["filesystem", "in-memory"]): Backend used by the skills
            middleware. Defaults to "filesystem".
        workspace_backend (str): Backend used when the agent has to auto-create a workspace
            session (i.e. when the caller does not pass one). Defaults to ``"auto"``, which
            picks the ``"bubblewrap_process"`` sandbox for real isolation when the host
            supports it and otherwise falls back to ``"ephemeral_fs"`` (a managed local
            workspace, no isolation). Supported values:

            - ``"auto"``: bubblewrap sandbox when available, else ``"ephemeral_fs"`` (default).
            - ``"local_fs"``: persistent local filesystem rooted at ``workspace_root``
              (required); no isolation, shell supported.
            - ``"ephemeral_fs"``: managed local workspace under the agent home; no isolation,
              shell supported.
            - ``"s3_snapshot"``: S3-backed snapshot workspace; no isolation, shell supported.
            - ``"postgres_vfs"``: Postgres-backed virtual filesystem; file tools only (no
              shell). Requires ``PostgresVFSSettings``.
            - ``"bubblewrap_process"``: bubblewrap process sandbox; real isolation, shell
              supported.
            - ``"gvisor_container"``: gVisor container sandbox; real isolation, shell
              supported. Requires ``GVisorContainerSettings`` and host/runtime deps.
            - ``"microvm"``: microVM sandbox; real isolation, shell supported. Requires
              ``NerdctlDevSettings``/``ContainerdKataSettings`` and host/runtime deps.
        workspace_root (Optional[str]): Filesystem root for an auto-created session. When
            None, the session manager picks a managed location under the agent home.
    """

    name: str = Field(default="workspace_agent")
    recursion_limit: Optional[int] = Field(default=200)
    coworker_tools: List[BaseTool] = Field(default_factory=list)
    coworker_tool_profile: Literal["default", "read_only"] = Field(default="default")
    read_only: bool = Field(default=False)
    skill_sources: Optional[List[str]] = Field(default=None)
    skills_backend: Literal["filesystem", "in-memory"] = Field(default="filesystem")
    workspace_backend: str = Field(default="auto")
    workspace_root: Optional[str] = Field(default=None)


class ReportConfig(BaseModel):
    """
    Configuration class for report generation.

    Attributes:
        enabled (bool): Flag to enable or disable report generation in team workflows.
            When disabled, the Team will skip the report step entirely.
        model (Optional[ModelConfig]): Model configuration for report generation. Required
            only when ``enabled`` is True.
        format (Optional[str]): The output format/structure of the report (e.g., a markdown
            template or high-level outline).
        citations (CitationConfig): Configuration for citations within the report.
        validation (ValidationConfig): Configuration for report validation.
        output_mode (Literal["state", "file", "both"]): Controls how the final report is
            surfaced. ``state`` stores it on the workflow state as ``report_output``,
            ``file`` saves it to disk, and ``both`` does both. Defaults to ``state``.
        output_path (Optional[Path]): Optional explicit file path for saving the report
            when ``output_mode`` includes file output. If not provided, a sensible default
            will be derived at runtime.
        base_dir (Optional[Path]): The base directory to save the report to. Defaults to None.
    """

    enabled: bool = Field(default=False)
    model: Optional[ModelConfig] = Field(default=None)
    format: Optional[str] = None
    citations: CitationConfig = Field(default_factory=CitationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    output_mode: Literal["state", "file", "both"] = Field(default="state")
    output_path: Optional[Path] = Field(default=None)
    base_dir: Optional[Path] = Field(default=None)
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_model(self) -> "ReportConfig":
        """Returns an error when a model is required but not set."""
        if self.enabled and self.model is None:
            raise ValueError(
                "Invalid ReportConfig: 'model' must be set when report.enabled is True."
            )
        return self
    
class MCPServerConfig(BaseModel):
    """
    Configuration for a single MCP (Model Context Protocol) server.

    The exact client implementation is provided by :mod:`andromeda.tools.mcp_adapter`.
    This model captures transport and process-level configuration so that MCP
    servers can be described alongside agents in Andromeda config files.
    """

    transport: Literal["stdio", "http"] = Field(
        default="stdio",
        description=(
            "Transport used to connect to the MCP server. "
            "'stdio' spawns a local process, 'http' connects to a remote MCP "
            "endpoint via StreamableHTTP."
        ),
    )

    # HTTP transport settings
    url: Optional[str] = Field(
        default=None,
        description=(
            "MCP endpoint URL when using 'http' transport, e.g. "
            "'https://api.githubcopilot.com/mcp/'."
        ),
    )
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional HTTP headers for 'http' transport, for example "
            "{'Authorization': 'Bearer <token>'}."
        ),
    )

    # stdio transport settings
    command: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "Executable (and optionally initial arguments) used to start the "
            "MCP server process when using 'stdio' transport. When a string is "
            "provided it is treated as the executable name and combined with "
            "any values in 'args'."
        ),
    )
    args: List[str] = Field(
        default_factory=list,
        description="Additional command-line arguments for the MCP server.",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the MCP server process.",
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Optional working directory for the MCP server process.",
    )
    include_tools: Optional[List[str]] = Field(
        default=None,
        description=(
            "If set, only tools whose names appear in this list will be "
            "registered into the Andromeda Toolkit."
        ),
    )
    exclude_tools: List[str] = Field(
        default_factory=list,
        description=(
            "Tools whose names appear in this list will be skipped during "
            "registration."
        ),
    )
    prefix: Optional[str] = Field(
        default=None,
        description=(
            "Optional prefix for registered tool names. When omitted, the key "
            "used in the top-level 'mcp_servers' mapping will be used as the "
            "prefix, resulting in tool names like 'server_name.tool_name'."
        ),
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_transport(self) -> "MCPServerConfig":
        """Ensure required fields are present for the chosen transport."""

        if self.transport == "stdio":
            if not self.command:
                raise ValueError(
                    "MCPServerConfig: 'command' is required when transport is 'stdio'."
                )
        elif self.transport == "http":
            if not self.url:
                raise ValueError(
                    "MCPServerConfig: 'url' is required when transport is 'http'."
                )

        return self


class AndromedaConfig(BaseModel):
    """
    AndromedaConfig is a configuration model for the andromeda system, encapsulating settings for agents, supervisor, planner, report.

    Attributes:
        agents (Dict[str, Union[AgentConfig, Any]]): A dictionary mapping agent names to their configuration objects.
        supervisor (SupervisorConfig): Configuration for the supervisor component. Defaults to a new SupervisorConfig instance.
        planner (PlannerConfig): Configuration for the planner component. Defaults to a new PlannerConfig instance.
        report (ReportConfig): Configuration for the reporting component. Defaults to a new ReportConfig instance.

    Class Attributes:
        model_config (dict): Pydantic model configuration allowing arbitrary types.

    Methods:
        load_from_file(path: str) -> "andromedaConfig":
            Class method to load configuration from a JSON or YAML file.

        save(path: str):
            Instance method to save the current configuration to a file in JSON or YAML format.
    """

    agents: Union[Dict[str, Union[AgentConfig, Any]], List[AgentConfig]]
    supervisor: SupervisorConfig = Field(default_factory=SupervisorConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_agents(self) -> "AndromedaConfig":
        """Returns an error when agents are not set."""
        if self.agents is None:
            raise ValueError(
                "Invalid AndromedaConfig: 'agents' is not set."
            )
        return self

    @model_validator(mode="after")
    def _check_supervisor(self) -> "AndromedaConfig":
        """Returns an error when supervisor is not set."""
        if self.supervisor is None:
            raise ValueError(
                "Invalid AndromedaConfig: 'supervisor' is not set."
            )
        return self

    @model_validator(mode="after")
    def _check_planner(self) -> "AndromedaConfig":
        """Returns an error when planner is not set."""
        if self.planner is None:
            raise ValueError(
                "Invalid AndromedaConfig: 'planner' is not set."
            )
        return self

    @model_validator(mode="after")
    def _check_report(self) -> "AndromedaConfig":
        """
        Ensure a report configuration object always exists.

        If a user explicitly sets ``report=None``, normalise it to a disabled
        ``ReportConfig`` so that downstream components can rely on the attribute
        being present while still treating reporting as optional.
        """
        if self.report is None:
            self.report = ReportConfig()  # disabled by default
        return self


    @classmethod
    def load_from_file(
        cls,
        path: str,
        *,
        interpolate_env: bool = True,
        register_mcp: bool = True,
        resolve_tools: bool = True,
        env: Optional[Mapping[str, str]] = None,
        toolkit: Optional[Any] = None,
        execution_context: Optional[Any] = None,
        mcp_runtime: Optional[Any] = None,
    ) -> "AndromedaConfig":
        """
        Load configuration from a JSON or YAML file.
        Args:
            path (str): The path to the configuration file. Supported formats are JSON (.json) and YAML (.yml, .yaml).
        Returns:
            andromedaConfig: An instance of the configuration class populated with data from the file.
        Raises:
            FileNotFoundError: If the specified configuration file does not exist.
            ValueError: If the file extension is not .json, .yml, or .yaml.
        """
        import importlib
        import json

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path_obj}")

        if path_obj.suffix == ".json":
            with open(path_obj) as f:
                data = json.load(f)
        elif path_obj.suffix in (".yml", ".yaml"):
            with open(path_obj) as f:
                data = yaml_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")

        return cls.load_from_config(
            data,
            source=path_obj,
            interpolate_env=interpolate_env,
            register_mcp=register_mcp,
            resolve_tools=resolve_tools,
            env=env,
            toolkit=toolkit,
            execution_context=execution_context,
            mcp_runtime=mcp_runtime,
        )

    @classmethod
    def load_from_config(
        cls,
        data: Any,
        *,
        source: Optional[Union[str, Path]] = None,
        strict: bool = True,
        interpolate_env: bool = True,
        register_mcp: bool = True,
        resolve_tools: bool = True,
        env: Optional[Mapping[str, str]] = None,
        toolkit: Optional[Any] = None,
        execution_context: Optional[Any] = None,
        mcp_runtime: Optional[Any] = None,
    ) -> Union["AndromedaConfig", Any]:
        """
        Load (and optionally normalize) configuration from an in-memory object.

        This is a reusable version of :meth:`load_from_file` for cases where you
        want to:
        - build config dicts programmatically
        - load partial configs (omit top-level keys like agents/supervisor/planner)
        - still benefit from env interpolation + tool resolution

        Args:
            data: Parsed configuration object (typically a dict from JSON/YAML).
            source: Optional identifier used in error messages (file path, etc.).
            strict: When True, validates and returns an AndromedaConfig instance.
                When False, returns the resolved config object (typically a dict).
            interpolate_env: When True, interpolates env vars in strings using "${VAR}".
            register_mcp: When True, registers MCP servers if the "mcp_servers" key exists.
            resolve_tools: When True, resolves tool specifications for any present agent/supervisor entries.
        """
        src = source if source is not None else "<in-memory config>"
        resolved = _normalize_config_object(
            data,
            source=src,
            interpolate_env=interpolate_env,
            register_mcp=register_mcp,
            resolve_tools=resolve_tools,
            env=env,
            toolkit=toolkit,
            execution_context=execution_context,
            mcp_runtime=mcp_runtime,
        )

        if not strict:
            return resolved

        if not isinstance(resolved, dict):
            raise ValueError(
                f"Invalid AndromedaConfig parsed from {src}: expected a mapping (dict) at the config root."
            )

        try:
            return cls(**resolved)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid AndromedaConfig parsed from {src}.\n"
                f"Please check required fields on 'agents', 'supervisor', 'planner', "
                f"'report', and nested model/validation configs.\n"
                f"Original validation error:\n{exc}"
            ) from exc

    def save(self, path: str):
        """
        Save the current configuration to a file in JSON or YAML format.

        Args:
            path (str): The file path where the configuration will be saved. The file extension
                determines the format: '.json' for JSON, '.yml' or '.yaml' for YAML.

        Raises:
            ValueError: If the file extension is not supported.
        """

        import json

        path = Path(path)
        data = self.model_dump()

        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in (".yml", ".yaml"):
            with open(path, "w") as f:
                yaml_dump(data, f)
        else:
            raise ValueError("Config file must be JSON or YAML")
