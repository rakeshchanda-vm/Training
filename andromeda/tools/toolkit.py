from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

from langchain.tools import BaseTool


class Toolkit:
    """Registry for LangChain/Andromeda tools.

    This registry is used when loading configuration from YAML so that
    string tool names (e.g. ``tools: [web_search]``) can be resolved
    into concrete ``BaseTool`` instances.
    """

    def __init__(
        self,
        tools: Optional[Iterable[BaseTool]] = None,
        *,
        parent: Optional["Toolkit"] = None,
    ) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._parent = parent
        if tools:
            for tool in tools:
                self.register(tool)

    # ------------------------------------------------------------------
    # Core registry operations
    # ------------------------------------------------------------------
    def register(
        self,
        tool: BaseTool,
        *,
        name: Optional[str] = None,
        override: bool = False,
    ) -> None:
        """Register a tool in the registry.

        Args:
            tool: The tool instance to register.
            name: Optional explicit registry name. Defaults to ``tool.name``.
            override: When False (default), existing registrations are kept.
        """

        tool_name = name or getattr(tool, "name", None)
        if not tool_name:
            raise ValueError(
                "Tool must have a 'name' attribute or an explicit name when registering."
            )

        name_exists = tool_name in self._tools or (
            self._parent.has(tool_name) if self._parent is not None else False
        )
        if not override and name_exists:
            # Do not silently overwrite by default; keep first registration.
            return

        self._tools[tool_name] = tool

    def get(self, name: str) -> BaseTool:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as exc:  # noqa: B904
            if self._parent is not None:
                return self._parent.get(name)
            raise KeyError(f"Tool '{name}' is not registered in the Toolkit.") from exc

    def has(self, name: str) -> bool:
        """Check if a tool with the given name exists."""

        return name in self._tools or (
            self._parent.has(name) if self._parent is not None else False
        )

    def all(self) -> Dict[str, BaseTool]:
        """Return a shallow copy of the registry mapping."""

        merged = self._parent.all() if self._parent is not None else {}
        merged.update(self._tools)
        return merged


# ----------------------------------------------------------------------
# Module-level default registry helpers
# ----------------------------------------------------------------------
_DEFAULT_TOOLKIT = Toolkit()


def get_default_toolkit() -> Toolkit:
    """Return the process-wide default ``Toolkit`` instance."""

    return _DEFAULT_TOOLKIT


def register_tool(
    tool: BaseTool,
    *,
    name: Optional[str] = None,
    override: bool = False,
    toolkit: Optional[Toolkit] = None,
) -> None:
    """Convenience wrapper to register a tool on the default registry."""

    (toolkit or get_default_toolkit()).register(tool, name=name, override=override)

def register_tools(
    tools: Iterable[Union[BaseTool, Tuple[BaseTool, Optional[str], bool]]],
    *,
    toolkit: Optional[Toolkit] = None,
) -> None:
    """Convenience wrapper to register a list of tools on the default registry.

    Accepts either ``BaseTool`` instances or 3-tuples of
    ``(tool, name, override)`` with appropriate types.
    """

    target_toolkit = toolkit or get_default_toolkit()
    for tool in tools:
        if isinstance(tool, BaseTool):
            target_toolkit.register(tool)
        elif isinstance(tool, tuple):
            if len(tool) != 3:
                raise ValueError(
                    "Tool tuple must be (tool, name, override) with length 3."
                )
            tool_obj, name, override = tool
            if not isinstance(tool_obj, BaseTool):
                raise TypeError("First element of tuple must be a BaseTool instance.")
            if name is not None and not isinstance(name, str):
                raise TypeError("Second element (name) must be a string or None.")
            if not isinstance(override, bool):
                raise TypeError("Third element (override) must be a bool.")
            target_toolkit.register(tool_obj, name=name, override=override)
        else:
            raise TypeError(
                f"Unsupported tool specification of type {type(tool)!r}. Expected a BaseTool instance or a tuple of (tool, name, override)."
            )

def resolve_tool_spec(spec: Any, *, toolkit: Optional[Toolkit] = None) -> BaseTool:
    """Resolve a generic tool specification into a concrete ``BaseTool``.

    This is primarily used when loading configuration from YAML where tools
    are specified by name. Currently supported forms:

    - A ``BaseTool`` instance.
    - A string matching a name registered in the default ``Toolkit``.

    For more advanced use-cases, users can register tools at startup via
    :func:`register_tool` and then reference them by name in ``config.yaml``.
    """

    # Already a tool
    if isinstance(spec, BaseTool):
        return spec

    # Look up by registered name
    if isinstance(spec, str):
        target_toolkit = toolkit or get_default_toolkit()
        if target_toolkit.has(spec):
            return target_toolkit.get(spec)
        raise ValueError(
            f"Unknown tool '{spec}'. Ensure it is either one of Andromeda's built-in "
            "tools or that you have registered it via andromeda.tools.toolkit.register_tool "
            "before loading configuration."
        )

    raise TypeError(
        f"Unsupported tool specification of type {type(spec)!r}. "
        "Expected a BaseTool instance or a registered tool name string."
    )
