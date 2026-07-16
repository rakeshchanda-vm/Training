from __future__ import annotations

"""
MCP (Model Context Protocol) integration for Andromeda.

This module provides a thin bridge between MCP servers and Andromeda's
existing LangChain-style tool registry. The goal is to make MCP tools
feel like any other tool in configuration files:

    mcp_servers:
      filesystem:
        command: ["python", "servers/fs_server.py"]
        cwd: "./servers"

    agents:
      - name: researcher
        model: ...
        tools:
          - filesystem.read_file   # MCP tool exposed as an Andromeda tool

Configuration is parsed in :meth:`AndromedaConfig.load_from_file`, which
calls :func:`register_mcp` before resolving tool
specifications so that MCP tools can be referenced by name in ``tools``.

The actual wire protocol implementation is delegated to the optional
``mcp`` Python package. When it is not installed, this module will raise
an informative error at registration time.
"""

import asyncio
import importlib
import threading
from concurrent.futures import Future
from typing import Any, AsyncIterator, Awaitable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, TypeVar

from contextlib import AsyncExitStack, asynccontextmanager

from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool # exception for this one to not use langchain.tools

from andromeda.tools.toolkit import Toolkit, register_tool

# Cache for lazily-imported MCP client classes. We intentionally avoid static
# imports so that type-checkers do not require the optional dependency.
_MCP_CLIENT: Optional[Tuple[Any, Any]] = None
_HTTP_CLIENT: Optional[Any] = None

T = TypeVar("T")


def _run_coro_blocking(coro: Awaitable[T]) -> T:
    """Run an async coroutine from sync code without assuming loop state.

    When no event loop is running in the current thread, this uses
    :func:`asyncio.run`. If a loop *is* running (e.g. in notebooks or
    async web servers), the coroutine is executed in a dedicated worker
    thread with its own event loop to avoid ``RuntimeError``.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread: safe to use asyncio.run directly.
        return asyncio.run(coro)

    # An event loop is already running in this thread; offload the coroutine
    # to a background thread and block until it completes.
    result: List[T] = []
    error: List[BaseException] = []
    done = threading.Event()

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    done.wait()

    if error:
        raise error[0]
    return result[0] if result else None  # type: ignore[return-value]


def _resolve_execution_toolkit(
    *,
    toolkit: Optional[Toolkit] = None,
    execution_context: Optional[Any] = None,
) -> Optional[Toolkit]:
    if toolkit is not None:
        return toolkit
    if execution_context is not None:
        maybe_toolkit = getattr(execution_context, "toolkit", None)
        if isinstance(maybe_toolkit, Toolkit):
            return maybe_toolkit
    return None


def _resolve_execution_runtime(
    *,
    runtime: Optional["MCPRuntime"] = None,
    execution_context: Optional[Any] = None,
) -> Optional["MCPRuntime"]:
    if runtime is not None:
        return runtime
    if execution_context is not None:
        maybe_runtime = getattr(execution_context, "mcp_runtime", None)
        if isinstance(maybe_runtime, MCPRuntime):
            return maybe_runtime
    return None


def _apply_execution_context_to_server_cfg(
    cfg: Mapping[str, Any],
    *,
    execution_context: Optional[Any] = None,
) -> Dict[str, Any]:
    resolved = dict(cfg)
    if execution_context is None:
        return resolved

    execution_env = getattr(execution_context, "env", None)
    if isinstance(execution_env, Mapping) and execution_env:
        merged_env = {str(key): str(value) for key, value in execution_env.items()}
        raw_env = _extract_env(cfg) or {}
        merged_env.update(raw_env)
        resolved["env"] = merged_env
    return resolved


def _normalize_tool_filters(
    server_name: str,
    raw_cfg: Mapping[str, Any],
) -> Tuple[Optional[List[str]], set[str], Optional[str]]:
    include_tools_raw = raw_cfg.get("include_tools")
    exclude_tools_raw = raw_cfg.get("exclude_tools") or []
    prefix = raw_cfg.get("prefix")
    if prefix is not None:
        prefix = str(prefix)

    if include_tools_raw is None:
        include_tools: Optional[List[str]] = None
    elif isinstance(include_tools_raw, str):
        include_tools = [include_tools_raw]
    elif isinstance(include_tools_raw, Iterable):
        include_tools = [str(name) for name in include_tools_raw]
    else:
        raise TypeError(
            f"'include_tools' for MCP server '{server_name}' must be a "
            "string or list of strings."
        )

    if isinstance(exclude_tools_raw, str):
        exclude_tools = {exclude_tools_raw}
    elif isinstance(exclude_tools_raw, Iterable):
        exclude_tools = {str(name) for name in exclude_tools_raw}
    else:
        raise TypeError(
            f"'exclude_tools' for MCP server '{server_name}' must be a "
            "string or list of strings."
        )

    return include_tools, exclude_tools, prefix


class _LoopDispatcher:
    """Own a dedicated event loop so MCP sessions stay bound to one loop."""

    def __init__(self, *, name: str) -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._closed = False

    def start(self) -> None:
        if self._loop is not None:
            return

        def runner() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._ready.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    with contextlib.suppress(Exception):
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                loop.close()

        import contextlib

        self._thread = threading.Thread(target=runner, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()

    def submit(self, coro: Awaitable[T]) -> Future[T]:
        if self._closed:
            raise RuntimeError("Dispatcher is closed.")
        self.start()
        if self._loop is None:
            raise RuntimeError("Dispatcher loop failed to start.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def call(self, coro: Awaitable[T]) -> T:
        return await asyncio.wrap_future(self.submit(coro))

    def call_blocking(self, coro: Awaitable[T]) -> T:
        return self.submit(coro).result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None


class MCPRuntime:
    """Execution-scoped MCP session manager with a dedicated event loop."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        execution_context: Optional[Any] = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError(
                "Expected MCP runtime config to be a mapping from server names to configuration dictionaries."
            )
        self._raw_config = dict(config)
        self._execution_context = execution_context
        self._dispatcher = _LoopDispatcher(name="andromeda-mcp-runtime")
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._opened = False

    @classmethod
    async def open(
        cls,
        config: Mapping[str, Any],
        *,
        execution_context: Optional[Any] = None,
    ) -> "MCPRuntime":
        runtime = cls(config, execution_context=execution_context)
        await runtime._open()
        return runtime

    async def _open(self) -> None:
        if self._opened:
            return
        await self._dispatcher.call(self._open_inner())
        self._opened = True

    async def _open_inner(self) -> None:
        if self._sessions:
            return

        ClientSession, stdio_client = _ensure_mcp_installed()
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            for server_name, original_cfg in self._raw_config.items():
                if not isinstance(original_cfg, Mapping):
                    continue

                raw_cfg = _apply_execution_context_to_server_cfg(
                    original_cfg,
                    execution_context=self._execution_context,
                )
                if _is_http_transport(raw_cfg):
                    streamablehttp_client = _ensure_http_client()
                    url = raw_cfg.get("url")
                    if not url:
                        raise ValueError(
                            f"MCP server '{server_name}' is configured for HTTP transport "
                            "but no 'url' was provided."
                        )
                    headers = _extract_headers(raw_cfg)
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(url, headers=headers)
                    )
                    session = ClientSession(read, write)  # type: ignore[call-arg]
                    session = await stack.enter_async_context(session)
                    await session.initialize()
                else:
                    argv = _build_command(server_name, raw_cfg)
                    if not argv:
                        raise ValueError(
                            f"MCP server '{server_name}' produced an empty command."
                        )
                    cmd = argv[0]
                    cmd_args = argv[1:]
                    env = _extract_env(raw_cfg)
                    cwd = _extract_cwd(raw_cfg)

                    from mcp.client.stdio import StdioServerParameters

                    server_params = StdioServerParameters(
                        command=cmd,
                        args=cmd_args,
                        env=env,
                        cwd=cwd,
                    )
                    read, write = await stack.enter_async_context(
                        stdio_client(server=server_params)  # type: ignore[call-arg]
                    )
                    session = ClientSession(read, write)  # type: ignore[call-arg]
                    session = await stack.enter_async_context(session)
                    await session.initialize()

                self._sessions[server_name] = {"session": session, "config": raw_cfg}
        except Exception:
            await stack.aclose()
            raise

        self._sessions["__stack__"] = {"session": stack, "config": {}}

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("MCPRuntime is not open. Call MCPRuntime.open(...) first.")

    async def list_tools(self, server_name: str) -> List[Mapping[str, Any]]:
        self._require_open()
        return await self._dispatcher.call(self._list_tools_inner(server_name))

    def list_tools_blocking(self, server_name: str) -> List[Mapping[str, Any]]:
        self._require_open()
        return self._dispatcher.call_blocking(self._list_tools_inner(server_name))

    async def _list_tools_inner(self, server_name: str) -> List[Mapping[str, Any]]:
        session = self._sessions[server_name]["session"]
        tool_objs = await _list_all_tools(session)
        return _normalize_tools_list(tool_objs)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        self._require_open()
        return await self._dispatcher.call(
            self._call_tool_inner(server_name, tool_name, arguments)
        )

    def call_tool_blocking(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        self._require_open()
        return self._dispatcher.call_blocking(
            self._call_tool_inner(server_name, tool_name, arguments)
        )

    async def _call_tool_inner(
        self,
        server_name: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        session = self._sessions[server_name]["session"]
        return await session.call_tool(tool_name, arguments)

    async def build_tools(
        self,
        *,
        toolkit: Optional[Toolkit] = None,
        register: bool = False,
    ) -> List[BaseTool]:
        return build_mcp_tools(
            self._raw_config,
            execution_context=self._execution_context,
            toolkit=toolkit,
            runtime=self,
            register=register,
        )

    async def aclose(self) -> None:
        if not self._opened:
            self._dispatcher.close()
            return
        await self._dispatcher.call(self._aclose_inner())
        self._opened = False
        self._dispatcher.close()

    async def _aclose_inner(self) -> None:
        stack_holder = self._sessions.pop("__stack__", None)
        self._sessions.clear()
        if stack_holder is not None:
            stack = stack_holder["session"]
            await stack.aclose()


def _ensure_mcp_installed() -> Tuple[Any, Any]:
    """Return (ClientSession, stdio_client) or raise a clear error.

    The underlying classes are loaded lazily via :mod:`importlib` so that
    Andromeda can be installed without the optional ``mcp`` dependency unless
    users explicitly enable MCP adapters in their configuration.
    """

    global _MCP_CLIENT

    if _MCP_CLIENT is not None:
        return _MCP_CLIENT

    try:
        session_mod = importlib.import_module("mcp.client.session")
        stdio_mod = importlib.import_module("mcp.client.stdio")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "MCP integration requires the optional 'mcp' Python package to be "
            "installed. Install it with `pip install <andromeda_package>[mcp]` (or the official "
            "Model Context Protocol Python client via `pip install mcp`) and then re-run your "
            "Andromeda workflow.\n\n"
            "If you do not intend to use MCP, remove the 'mcp_servers' section "
            "from your configuration file."
        ) from exc

    client_session = getattr(session_mod, "ClientSession", None)
    stdio_client = getattr(stdio_mod, "stdio_client", None)

    if client_session is None or stdio_client is None:
        raise RuntimeError(
            "The 'mcp' package is installed but does not expose the expected "
            "client interfaces (ClientSession, stdio_client). Please verify "
            "your MCP Python client version."
        )

    _MCP_CLIENT = (client_session, stdio_client)
    return _MCP_CLIENT


def _ensure_http_client() -> Any:
    """Return ``streamablehttp_client`` from the MCP SDK or raise a clear error."""

    global _HTTP_CLIENT

    if _HTTP_CLIENT is not None:
        return _HTTP_CLIENT

    try:
        http_mod = importlib.import_module("mcp.client.streamable_http")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "HTTP MCP integration requires the 'mcp' Python package with "
            "StreamableHTTP support installed. Install/upgrade it with "
            "`pip install -U <andromeda_package>[mcp]` (or the official "
            "Model Context Protocol Python client via `pip install mcp`) and then re-run your "
            "Andromeda workflow."
        ) from exc

    http_client = getattr(http_mod, "streamablehttp_client", None)
    if http_client is None:
        raise RuntimeError(
            "The 'mcp' package does not expose 'streamablehttp_client'. "
            "Please upgrade to a recent MCP Python SDK."
        )

    _HTTP_CLIENT = http_client
    return _HTTP_CLIENT


def _build_command(server_name: str, cfg: Mapping[str, Any]) -> List[str]:
    """Normalise the configured command/args into a concrete argv list."""

    command = cfg.get("command")
    args = cfg.get("args", [])

    if command is None:
        raise ValueError(
            f"MCP server '{server_name}' is missing required 'command' in "
            "configuration."
        )

    cmd_list: List[str]
    if isinstance(command, str):
        cmd_list = [command]
    elif isinstance(command, Iterable):
        cmd_list = [str(part) for part in command]
    else:
        raise TypeError(
            f"Invalid 'command' type for MCP server '{server_name}': "
            f"{type(command)!r}. Expected string or list of strings."
        )

    if isinstance(args, str):
        # Allow a single string argument; do not attempt shell-splitting.
        extra = [args]
    elif isinstance(args, Iterable):
        extra = [str(part) for part in args]
    else:
        raise TypeError(
            f"Invalid 'args' type for MCP server '{server_name}': "
            f"{type(args)!r}. Expected string or list of strings."
        )

    return cmd_list + extra


def _extract_env(cfg: Mapping[str, Any]) -> Optional[Dict[str, str]]:
    """Return environment mapping for the MCP process, if any."""

    env_raw = cfg.get("env")
    if env_raw is None:
        return None
    if not isinstance(env_raw, MutableMapping):
        raise TypeError(
            "Expected 'env' for MCP server configuration to be a mapping of "
            "string keys to string values."
        )
    env: Dict[str, str] = {}
    for key, value in env_raw.items():
        env[str(key)] = str(value)
    return env


def _extract_cwd(cfg: Mapping[str, Any]) -> Optional[str]:
    """Return working directory for the MCP process, if configured."""

    cwd = cfg.get("cwd")
    if cwd is None:
        return None
    return str(cwd)


def _extract_headers(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """Return HTTP headers mapping for StreamableHTTP transport, if any."""

    headers_raw = cfg.get("headers") or {}
    if not isinstance(headers_raw, MutableMapping):
        raise TypeError(
            "Expected 'headers' for HTTP-based MCP server configuration to be "
            "a mapping of string keys to string values."
        )
    headers: Dict[str, str] = {}
    for key, value in headers_raw.items():
        headers[str(key)] = str(value)
    return headers


def _is_http_transport(cfg: Mapping[str, Any]) -> bool:
    """Return True when the configuration indicates HTTP transport should be used."""

    transport = str(cfg.get("transport", "stdio")).lower()
    if transport == "http":
        return True
    # Heuristic fallback: if a URL is provided and no command is set, assume HTTP.
    if "url" in cfg and "command" not in cfg:
        return True
    return False


async def _list_all_tools(session: Any) -> List[Any]:
    """List all tools from an MCP session, following pagination cursors."""

    current_cursor: str | None = None
    all_tools: List[Any] = []
    iterations = 0
    max_iterations = 1000

    while True:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError(
                f"Reached max of {max_iterations} iterations while listing MCP tools."
            )

        # Use the simple cursor-based form for broad SDK compatibility.
        result = await session.list_tools(cursor=current_cursor)
        if result.tools:
            all_tools.extend(result.tools)

        # Pagination spec allows None/"" when finished.
        if not getattr(result, "nextCursor", None):
            break
        current_cursor = result.nextCursor

    return all_tools


async def _discover_tools_stdio(
    server_name: str,
    cfg: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Discover tools from a stdio-based MCP server."""

    ClientSession, stdio_client = _ensure_mcp_installed()

    argv = _build_command(server_name, cfg)
    if not argv:
        raise ValueError(f"MCP server '{server_name}' produced an empty command.")

    cmd = argv[0]
    cmd_args = argv[1:]
    env = _extract_env(cfg)
    cwd = _extract_cwd(cfg)
    
    from mcp.client.stdio import StdioServerParameters

    server_params = StdioServerParameters(
        command=cmd,
        args=cmd_args,
        env=env,
        cwd=cwd,
    )
    async with stdio_client(server=server_params) as (read, write):  # type: ignore[call-arg]
        async with ClientSession(read, write) as session:  # type: ignore[call-arg]
            await session.initialize()
            tools_obj = await _list_all_tools(session)

    return _normalize_tools_list(tools_obj)


async def _discover_tools_http(
    server_name: str,
    cfg: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Discover tools from an HTTP-based MCP server."""

    ClientSession, _ = _ensure_mcp_installed()
    streamablehttp_client = _ensure_http_client()

    url = cfg.get("url")
    if not url:
        raise ValueError(
            f"MCP server '{server_name}' is configured for HTTP transport but "
            "no 'url' was provided."
        )
    headers = _extract_headers(cfg)

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:  # type: ignore[call-arg]
            await session.initialize()
            tools_obj = await _list_all_tools(session)

    return _normalize_tools_list(tools_obj)


def _normalize_tools_list(tools_obj: Any) -> List[Mapping[str, Any]]:
    """Normalize ListToolsResult or list of tools into a list of plain dicts."""

    # Accept either a ListToolsResult-like object or a plain list of tools.
    if isinstance(tools_obj, list):
        raw_tools = tools_obj
    else:
        raw_tools = getattr(tools_obj, "tools", tools_obj)

    tools: List[Mapping[str, Any]] = []
    for entry in raw_tools:
        if isinstance(entry, Mapping):
            # Normalise key names to a consistent snake_case variant.
            name = entry.get("name")
            if not name:
                continue
            description = entry.get("description") or ""
            input_schema = (
                entry.get("input_schema")
                or entry.get("inputSchema")
                or {}
            )
            tools.append(
                {
                    "name": str(name),
                    "description": str(description),
                    "input_schema": input_schema,
                }
            )
            continue

        # Fallback for dataclass/attribute style tool descriptions.
        name_attr = getattr(entry, "name", None)
        if not name_attr:
            continue
        description_attr = getattr(entry, "description", "") or ""
        input_schema_attr = (
            getattr(entry, "input_schema", None)
            or getattr(entry, "inputSchema", None)
            or {}
        )
        tools.append(
            {
                "name": str(name_attr),
                "description": str(description_attr),
                "input_schema": input_schema_attr,
            }
        )

    return tools


async def _discover_tools_for_server(
    server_name: str,
    cfg: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    """Connect to an MCP server and return its advertised tools."""

    if _is_http_transport(cfg):
        return await _discover_tools_http(server_name, cfg)
    return await _discover_tools_stdio(server_name, cfg)


async def _call_mcp_tool_stdio(
    server_name: str,
    cfg: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any],
) -> Any:
    """Invoke a specific tool on a stdio-based MCP server and return its result."""

    ClientSession, stdio_client = _ensure_mcp_installed()

    argv = _build_command(server_name, cfg)
    if not argv:
        raise ValueError(f"MCP server '{server_name}' produced an empty command.")

    cmd = argv[0]
    cmd_args = argv[1:]
    env = _extract_env(cfg)
    cwd = _extract_cwd(cfg)

    from mcp.client.stdio import StdioServerParameters

    server_params = StdioServerParameters(
        command=cmd,
        args=cmd_args,
        env=env,
        cwd=cwd,
    )
    async with stdio_client(server=server_params) as (read, write):  # type: ignore[call-arg]
        async with ClientSession(read, write) as session:  # type: ignore[call-arg]
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    # The exact type of ``result`` depends on the MCP client; we simply return
    # it to the caller so the LLM can reason over the structured payload.
    return result


async def _call_mcp_tool_http(
    server_name: str,
    cfg: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any],
) -> Any:
    """Invoke a specific tool on an HTTP-based MCP server and return its result."""

    ClientSession, _ = _ensure_mcp_installed()
    streamablehttp_client = _ensure_http_client()

    url = cfg.get("url")
    if not url:
        raise ValueError(
            f"MCP server '{server_name}' is configured for HTTP transport but "
            "no 'url' was provided."
        )
    headers = _extract_headers(cfg)

    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:  # type: ignore[call-arg]
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    return result


async def _call_mcp_tool(
    server_name: str,
    cfg: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any],
    session: Any | None = None,
) -> Any:
    """Invoke a specific tool on an MCP server and return its result.

    When ``session`` is provided, it is assumed to be a live ``ClientSession``
    already connected to the appropriate server. In that case, no additional
    transports are created and the call is executed directly on the session.
    """

    if session is not None:
        return await session.call_tool(tool_name, arguments)

    if _is_http_transport(cfg):
        return await _call_mcp_tool_http(server_name, cfg, tool_name, arguments)
    return await _call_mcp_tool_stdio(server_name, cfg, tool_name, arguments)


def _render_call_tool_result_to_text(result: Any) -> str:
    """Convert an MCP CallToolResult into a human-readable text string.

    This flattens multiple TextContent blocks and appends a brief summary
    of any non-text artifacts. When the result represents an error, a
    RuntimeError is raised with the tool's textual error message.
    """

    try:
        from mcp.types import TextContent  # type: ignore[import]
    except Exception:  # pragma: no cover - very unlikely
        return str(result)

    contents = getattr(result, "content", None)
    is_error = bool(getattr(result, "isError", False))

    texts: List[str] = []
    artifacts_info: List[str] = []

    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, TextContent):
                texts.append(item.text)
            else:
                # Lightweight representation for non-text content; the goal is
                # to surface that artifacts exist without requiring callers to
                # understand MCP's rich content types.
                artifacts_info.append(repr(item))

    if not texts:
        text = ""
    elif len(texts) == 1:
        text = texts[0]
    else:
        text = "\n".join(texts)

    if is_error:
        raise RuntimeError(text or "MCP tool returned an error.")

    if artifacts_info:
        text = (
            text
            + ("\n\n" if text else "")
            + "[Artifacts]\n"
            + "\n".join(artifacts_info)
        )

    return text


def _build_args_model(tool_fq_name: str, schema: Mapping[str, Any]) -> type:
    """Create a minimal Pydantic model from a JSON Schema mapping.

    We intentionally keep type inference simple and treat all fields as
    ``Any``. The primary goal is to surface field names and descriptions to
    the LLM, not to enforce strict runtime validation.
    """

    # Deferred import to avoid imposing Pydantic at module import time.
    from pydantic import BaseModel, create_model  # type: ignore

    if not isinstance(schema, Mapping):
        return create_model(f"{tool_fq_name.replace('.', '_')}_Args", __base__=BaseModel)

    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping) or not properties:
        return create_model(f"{tool_fq_name.replace('.', '_')}_Args", __base__=BaseModel)

    required = set(schema.get("required") or [])

    def _json_schema_type_to_py_type(prop_schema: Mapping[str, Any]) -> type:
        """Best-effort mapping from JSON Schema types to Python types."""

        t = prop_schema.get("type")
        # Handle union types like ["string", "null"] by picking a non-null type.
        if isinstance(t, list):
            non_null = [v for v in t if v != "null"]
            t = non_null[0] if non_null else None

        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "array":
            return list[Any]
        if t == "object":
            return Dict[str, Any]
        return Any

    fields: Dict[str, tuple[type, Any]] = {}
    for prop_name, prop_schema in properties.items():
        py_type = _json_schema_type_to_py_type(prop_schema if isinstance(prop_schema, Mapping) else {})
        default = ... if prop_name in required else None
        fields[str(prop_name)] = (py_type, default)

    return create_model(
        f"{tool_fq_name.replace('.', '_')}_Args",
        __base__=BaseModel,
        **fields,  # type: ignore[arg-type]
    )


def _make_langchain_tool(
    server_name: str,
    cfg: Mapping[str, Any],
    tool_desc: Mapping[str, Any],
    *,
    prefix: Optional[str],
    runtime: MCPRuntime | None = None,
    session: Any | None = None,
) -> BaseTool:
    """Convert an MCP tool description into a LangChain ``BaseTool``."""

    short_name = str(tool_desc.get("name"))
    if not short_name:
        raise ValueError("MCP tool description is missing a 'name' field.")

    # Use '<server>.<tool>' by default so that tools are namespaced when
    # multiple MCP servers expose similarly named tools.
    effective_prefix = prefix or server_name
    full_name = f"{effective_prefix}_{short_name}" if effective_prefix else short_name

    description = str(
        tool_desc.get("description")
        or f"MCP tool '{short_name}' provided by server '{server_name}'."
    )
    input_schema = tool_desc.get("input_schema") or {}
    args_model = _build_args_model(full_name, input_schema)

    async def _acall(**kwargs: Any) -> Any:
        if runtime is not None:
            raw_result = await runtime.call_tool(server_name, short_name, kwargs)
        else:
            raw_result = await _call_mcp_tool(
                server_name,
                cfg,
                short_name,
                kwargs,
                session=session,
            )
        return _render_call_tool_result_to_text(raw_result)

    def _call(**kwargs: Any) -> Any:
        if runtime is not None:
            raw_result = runtime.call_tool_blocking(server_name, short_name, kwargs)
            return _render_call_tool_result_to_text(raw_result)

        # Best-effort bridge from sync tool interface to the async MCP client,
        # handling both plain scripts and environments with a running event loop.
        return _run_coro_blocking(_acall(**kwargs))

    return StructuredTool(
        name=full_name,
        description=description,
        func=_call,
        coroutine=_acall,
        args_schema=args_model,
    )


def build_mcp_tools(
    config: Mapping[str, Any],
    *,
    execution_context: Optional[Any] = None,
    toolkit: Optional[Toolkit] = None,
    runtime: Optional[MCPRuntime] = None,
    register: bool = False,
) -> List[BaseTool]:
    """Build MCP tools directly from config, optionally registering them into a toolkit."""

    if not isinstance(config, Mapping):
        raise TypeError(
            "Expected 'mcp_servers' configuration to be a mapping from server "
            "names to configuration dictionaries."
        )

    resolved_toolkit = _resolve_execution_toolkit(
        toolkit=toolkit,
        execution_context=execution_context,
    )
    resolved_runtime = _resolve_execution_runtime(
        runtime=runtime,
        execution_context=execution_context,
    )
    should_register = resolved_toolkit is not None or (
        register and resolved_runtime is None
    )

    all_tools: List[BaseTool] = []
    for server_name, original_cfg in config.items():
        if original_cfg is None:
            continue
        if not isinstance(original_cfg, Mapping):
            raise TypeError(
                f"Configuration for MCP server '{server_name}' must be a "
                f"mapping, got {type(original_cfg)!r}."
            )

        raw_cfg = _apply_execution_context_to_server_cfg(
            original_cfg,
            execution_context=execution_context,
        )
        include_tools, exclude_tools, prefix = _normalize_tool_filters(
            server_name,
            raw_cfg,
        )

        try:
            if resolved_runtime is not None:
                tools = resolved_runtime.list_tools_blocking(server_name)
            else:
                tools = _run_coro_blocking(
                    _discover_tools_for_server(server_name, raw_cfg)
                )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to discover tools for MCP server '{server_name}': {exc}"
            ) from exc

        for desc in tools:
            name = desc.get("name")
            if not name:
                continue

            if include_tools is not None and name not in include_tools:
                continue
            if name in exclude_tools:
                continue

            tool = _make_langchain_tool(
                server_name,
                raw_cfg,
                desc,
                prefix=prefix,
                runtime=resolved_runtime,
            )
            if should_register:
                if resolved_toolkit is not None:
                    resolved_toolkit.register(tool)
                else:
                    register_tool(tool)
            all_tools.append(tool)

    return all_tools


def register_mcp(
    config: Mapping[str, Any],
    *,
    toolkit: Optional[Toolkit] = None,
    execution_context: Optional[Any] = None,
    runtime: Optional[MCPRuntime] = None,
) -> List[BaseTool]:
    """Register MCP tools from configuration into the target toolkit."""

    return build_mcp_tools(
        config,
        execution_context=execution_context,
        toolkit=toolkit,
        runtime=runtime,
        register=True,
    )


@asynccontextmanager
async def open_mcp_sessions(
    config: Mapping[str, Any],
) -> AsyncIterator[Dict[str, Any]]:
    """Open long-lived MCP sessions for all configured servers.

    This is an advanced helper for callers that want to manage MCP sessions
    explicitly (for example, to amortize initialization across many tool
    calls within a single Andromeda Team run). It does **not** change the
    behaviour of tools returned by :func:`register_mcp`, which continue to
    create short-lived sessions per call.

    Example:
        async with open_mcp_sessions(mcp_cfg) as sessions:
            langgraph_session = sessions["langgraph-docs-mcp"]
            result = await langgraph_session.call_tool("some_tool", {"arg": "value"})
    """

    if not isinstance(config, Mapping):
        raise TypeError(
            "Expected 'config' for open_mcp_sessions to be a mapping from server "
            "names to configuration dictionaries."
        )

    ClientSession, stdio_client = _ensure_mcp_installed()
    streamablehttp_client = _ensure_http_client()

    sessions: Dict[str, Dict[str, Any]] = {}
    async with AsyncExitStack() as stack:
        for server_name, raw_cfg in config.items():
            if not isinstance(raw_cfg, Mapping):
                continue

            if _is_http_transport(raw_cfg):
                url = raw_cfg.get("url")
                if not url:
                    raise ValueError(
                        f"MCP server '{server_name}' is configured for HTTP transport "
                        "but no 'url' was provided."
                    )
                headers = _extract_headers(raw_cfg)
                read, write, _ = await stack.enter_async_context(
                    streamablehttp_client(url, headers=headers)
                )
                session = ClientSession(read, write)  # type: ignore[call-arg]
                session = await stack.enter_async_context(session)
                await session.initialize()
                sessions[server_name] = {"session": session, "config": raw_cfg}
            else:
                # stdio transport
                argv = _build_command(server_name, raw_cfg)
                if not argv:
                    raise ValueError(
                        f"MCP server '{server_name}' produced an empty command."
                    )
                cmd = argv[0]
                cmd_args = argv[1:]
                env = _extract_env(raw_cfg)
                cwd = _extract_cwd(raw_cfg)

                from mcp.client.stdio import StdioServerParameters

                server_params = StdioServerParameters(
                    command=cmd,
                    args=cmd_args,
                    env=env,
                    cwd=cwd,
                )
                read, write = await stack.enter_async_context(
                    stdio_client(server=server_params)  # type: ignore[call-arg]
                )
                session = ClientSession(read, write)  # type: ignore[call-arg]
                session = await stack.enter_async_context(session)
                await session.initialize()
                sessions[server_name] = {"session": session, "config": raw_cfg}

        try:
            yield sessions
        finally:
            # AsyncExitStack will close all sessions and transports.
            ...


async def load_from_sessions(
    sessions: Dict[str, Dict[str, Any]],
) -> List[BaseTool]:
    """Build StructuredTools bound to long-lived MCP sessions.

    This helper is intended for advanced usage where you explicitly manage
    MCP sessions (e.g., via :func:`open_mcp_sessions`) and want to build
    Andromeda/LangChain tools that reuse those sessions instead of spawning
    a new process or HTTP connection per call.

    Example:
        async with open_mcp_sessions(mcp_cfg) as sessions:
            tools_by_server = await load_mcp_tools_from_sessions(sessions, mcp_cfg)
            github_tools = tools_by_server["github"]
    """

    all_tools: List[BaseTool] = []

    for server_name, data in sessions.items():
        session = data["session"]
        raw_cfg = data["config"]
        if not isinstance(raw_cfg, Mapping):
            raw_cfg = {}

        include_tools_raw = raw_cfg.get("include_tools")
        exclude_tools_raw = raw_cfg.get("exclude_tools") or []
        prefix = raw_cfg.get("prefix")
        if prefix is not None:
            prefix = str(prefix)

        if include_tools_raw is None:
            include_tools: Optional[List[str]] = None
        elif isinstance(include_tools_raw, str):
            include_tools = [include_tools_raw]
        elif isinstance(include_tools_raw, Iterable):
            include_tools = [str(name) for name in include_tools_raw]
        else:
            raise TypeError(
                f"'include_tools' for MCP server '{server_name}' must be a "
                "string or list of strings."
            )

        if isinstance(exclude_tools_raw, str):
            exclude_tools = {exclude_tools_raw}
        elif isinstance(exclude_tools_raw, Iterable):
            exclude_tools = {str(name) for name in exclude_tools_raw}
        else:
            raise TypeError(
                f"'exclude_tools' for MCP server '{server_name}' must be a "
                "string or list of strings."
            )

        # Use the already-open session to list tools; no extra processes/HTTP
        # connections are spawned here.
        tool_objs = await _list_all_tools(session)
        tool_descs = _normalize_tools_list(tool_objs)

        server_tools: List[BaseTool] = []
        for desc in tool_descs:
            name = desc.get("name")
            if not name:
                continue

            if include_tools is not None and name not in include_tools:
                continue
            if name in exclude_tools:
                continue

            # NOTE: we intentionally do NOT bind the long-lived ``session`` into
            # the tool wrapper here. LangGraph executes tools inside thread
            # pools, and ``ClientSession`` instances are bound to the event loop
            # that created them. Reusing the same session across threads/loops
            # can lead to subtle deadlocks. Instead, tools created here behave
            # like those from :func:`register_mcp`: they open short-lived
            # connections per call while we still use the shared session only
            # for upfront tool discovery.
            tool = _make_langchain_tool(
                server_name,
                raw_cfg,
                desc,
                prefix=prefix,
            )
            register_tool(tool)
            server_tools.append(tool)
            all_tools.append(tool)

    return all_tools
