"""Workspace agent.

A workspace agent is essentially a single, long-horizon agent that operates inside an
isolated *workspace session*. It is the core building block behind the Autoforge SDK.

It is a :class:`~andromeda.core.supervisor.Supervisor`, so on top of routing work to
specialist sub-agents it adds:
    - Access to a workspace session, which provisions the concrete tools it works with
      (filesystem, shell, ...). The session is configurable and truly isolated; its
      backend can be microvm / filesystem / postgres / docker / fargate / ... and it may
      be read-only.
    - Long-horizon middleware (todo/plan tracking + context editing) so it can stay on
      task across many iterations.
    - Optional skills middleware (prompt/skills) and MCP-provided tools (resolved through
      the standard config tool resolution).

Subclassing :class:`Supervisor` (rather than the plain :class:`~andromeda.core.agent.Agent`)
means a workspace agent can both act directly inside its workspace and orchestrate other
agents; pass those sub-agents via the ``agents`` argument (empty by default).

The *workspace session* itself lives in :mod:`andromeda.workspace` (the session manager);
see :class:`andromeda.workspace.WorkspaceSession`, which this module imports and consumes
directly. A caller may pass a pre-built session, or omit it. This will auto-create a
default one from ``config.workspace_backend`` -- which defaults to ``"auto"``: a bubblewrap
sandbox when the host supports it, otherwise a managed local filesystem (``ephemeral_fs``).
"""
from __future__ import annotations

import asyncio
import threading
import warnings
import weakref
from contextlib import suppress
from typing import Any, List, Mapping, Optional, Union

import random

from andromeda import HumanMessage
from andromeda.config.config import AgentConfig, MiddlewareConfig, WorkspaceAgentConfig
from andromeda.core.agent import Agent
from andromeda.core.supervisor import Supervisor
from andromeda.core.middleware import (
    ContextEditingMiddleware,
    SkillsMiddleware,
)
from andromeda.tools import BaseTool
from andromeda.utils.logger import log_agent
from andromeda.utils.prompts import coworker_agent_prompt, workspace_supervisor_prompt
from andromeda.workspace import WorkspaceSession, WorkspacePolicy, ProviderSettings


def _dedupe_tools(tools: List[BaseTool]) -> List[BaseTool]:
    """Return ``tools`` with duplicates removed, keeping the first occurrence so 
    caller-supplied tools win over session-provided ones.
    Tools without a ``name`` are passed through untouched.
    """
    seen: set[str] = set()
    deduped: List[BaseTool] = []
    for t in tools:
        name = getattr(t, "name", None)
        if name is not None:
            if name in seen:
                continue
            seen.add(name)
        deduped.append(t)
    return deduped


def _release_session(session: Optional[Any]) -> None:
    """Best-effort teardown of a workspace session via ``cleanup()`` or ``close()``.

    Shared by :meth:`WorkspaceAgent.close` and the auto-create path so both release a
    session the same way (and so a half-built session can be reclaimed on error).
    """
    if session is None:
        return
    cleanup = getattr(session, "cleanup", None)
    if callable(cleanup):
        cleanup()
        return
    close = getattr(session, "close", None)
    if callable(close):
        close()


def _warn_unclosed_and_release(session: Any, agent_name: str) -> None:
    """:func:`weakref.finalize` callback for an auto-created session close() never reached.

    Registered only for sessions the agent owns. It runs at most once and only via the
    garbage-collection / interpreter-exit path: an explicit :meth:`WorkspaceAgent.close`
    (or context-manager exit) detaches the finalizer first and releases the session itself,
    so the deterministic path does NOT pass through here and does not warn. When this does
    fire it means the caller dropped an owning agent without closing it, so we emit the
    Pythonic ``ResourceWarning`` for a leaked external resource and still release the
    workspace as a safety net (a live sandbox/process would otherwise linger).
    """
    warnings.warn(
        f"WorkspaceAgent {agent_name!r} was not closed; releasing its auto-created "
        "workspace session via the garbage collector. Use "
        "`with WorkspaceAgent(...) as agent:` or call agent.close() to release the "
        "workspace deterministically.",
        ResourceWarning,
        stacklevel=2,
    )
    _release_session(session)


COWORKER_NAMES: tuple[str, ...] = (
    # Constellation neighborhood (bordering Andromeda)
    "Pegasus",       # great constellation bordering Andromeda
    "Cassiopeia",    # W-shaped constellation, mother of Andromeda
    "Perseus",       # hero who rescued Andromeda
    "Cepheus",       # king of Ethiopia
    "Pisces",        # zodiac constellation near Andromeda
    "Cetus",         # whale constellation that threatened Andromeda
    "Aries",         # ram constellation bordering Andromeda
    "Phoenix",       # southern constellation of the same equatorial family
    # Stars of the Andromeda region
    "Mirach",        # Beta Andromedae — brightest star in Andromeda
    "Alpheratz",     # Alpha Andromedae — corner of Great Square of Pegasus
    "Almach",        # Gamma Andromedae — blue-white triple system
    "Markab",        # Alpha Pegasi — corner of Great Square
    "Mira",          # Omicron Ceti — "The Wonderful One"
    "Atlas",         # Eta Tauri — mythological figure
    # Deep-sky objects in the Andromeda region
    "California",    # California Nebula in Perseus
    "Heart",         # IC 1805 nebula in Cassiopeia
    "Soul",          # IC 1848 nebula in Cassiopeia
    "Bubble",        # NGC 7635 nebula in Cassiopeia
    "Fishhead",      # NGC 2244 nebula in Orion region
    "Owl",           # NGC 457 owl cluster in Cassiopeia
    "Iris",          # NGC 7023 nebula in Cepheus
    "Skull",         # Sh 2-129 nebula in Cassiopeia
)

# Shuffled copy used by _coworker_name so coworkers get names in random order
_COWORKER_ROSTER: list[str] = list(COWORKER_NAMES)
random.shuffle(_COWORKER_ROSTER)

def _coworker_name(index: int) -> str:
    """Name the ``index``-th (0-based) auto-spawned coworker from :data:`COWORKER_NAMES`.

    The first pass through the roster uses the bare names; once it is exhausted the names wrap
    with an alphabetical cycle suffix so large teams still get unique names. I don't know if we will ever need this many names, but it is good to have I think.
    """
    roster = _COWORKER_ROSTER
    base = roster[index % len(roster)]
    cycle = index // len(roster)
    if cycle == 0:
        return base
    
    #alphabetical suffix: 1 -> "b", 25 -> "z", 26 -> "aa", ...
    letters: List[str] = []
    n = cycle
    while n >= 0:
        letters.append(chr(ord("a") + n % 26))
        n = n // 26 - 1
    return f"{base}_{''.join(reversed(letters))}"


class WorkspaceAgent(Supervisor):
    #: Default minimum team size. A workspace agent always operates with at least this
    #: many agents on its team. Override per-instance with the ``min_agents`` constructor
    #: argument; the value in effect for an instance is read via ``self.MIN_AGENTS``.
    MIN_AGENTS: int = 4

    def __init__(
        self,
        config: WorkspaceAgentConfig,
        session: Optional[WorkspaceSession] = None,
        agents: Optional[List[Union[AgentConfig, Agent]]] = None,
        *,
        min_agents: Optional[int] = None,
        policy: Optional[WorkspacePolicy] = None,
        settings: Optional[ProviderSettings] = None,
    ) -> None:
        """
        Workspace agent: a long-horizon supervisor bound to an isolated workspace session.

        It can:
            - Use the tools the session provisions (filesystem, shell, ...) to carry out
              tasks end to end inside the workspace.
            - Route sub tasks to specialist agents (inherited supervisor behavior).
            - Track progress across many iterations via long-horizon middleware.
            - Discover and use skills, and use MCP-provided tools resolved from config.

        The workspace agent always runs with a team. If fewer than :attr:`MIN_AGENTS`
        agents are supplied, it spins up coworker agents to fill the team out to
        :attr:`MIN_AGENTS`; if :attr:`MIN_AGENTS` or more are supplied, it adds one extra
        coworker to assist in parallel. Coworkers share the workspace session's tools. The
        minimum defaults to 4 and can be tuned per instance via ``min_agents``.

        One instance models a single conversation. Use a
        separate ``WorkspaceAgent`` per independent conversation (see :meth:`run`).

        Args:
            config (WorkspaceAgentConfig): The configuration for the workspace agent.
            session (Optional[WorkspaceSession]): The workspace session that provisions the
                agent's tools. When omitted, the agent auto-creates a default session using
                ``config.workspace_backend``. The default ``"auto"`` backend prefers a
                bubblewrap sandbox and falls back to ``ephemeral_fs`` (no isolation) when the
                host lacks bubblewrap. A backend named explicitly is honored as-is: if it is
                unavailable on the host, auto-creation raises ``WorkspaceProviderError``
                rather than falling back to a less-isolated backend.
            agents (Optional[List[Union[AgentConfig, Agent]]]): Specialist agents the
                workspace agent can route work to. Defaults to no caller-supplied agents
                (the team is then made up entirely of coworker agents).
            min_agents (Optional[int]): Minimum team size for this instance. Must be a
                positive integer (>= 1). When None (default), the class default
                :attr:`MIN_AGENTS` (4) is used. Supplying a value overrides it for this
                instance only (it does not mutate the class attribute), so fewer supplied
                agents are padded with coworkers up to this minimum.
            policy (Optional[WorkspacePolicy]): Policy applied to an auto-created session
                (read-only, shell limits, ...). Ignored when ``session`` is supplied. When
                None, a shell-enabled default policy is used.
            settings (Optional[ProviderSettings]): Backend-specific settings for an
                auto-created session (e.g. ``BubblewrapProcessSettings``). Ignored when
                ``session`` is supplied. When None, the sandbox backends with a fully
                defaulted settings object -- ``bubblewrap_process`` (BubblewrapProcessSettings())
                and ``gvisor_container`` (GVisorContainerSettings()) -- get their defaults
                filled in, and the filesystem backends (``ephemeral_fs``/``local_fs``) need
                none. Backends that require un-guessable config (``postgres_vfs``, ``microvm``)
                must be given settings explicitly and will error otherwise.
        Returns:
            None
        """
        # Work on a private copy so we never mutate the caller's config object.
        config = config.model_copy(
            update={
                "tools": list(config.tools),
                "middleware": config.middleware.model_copy(
                    update={
                        "custom": list(config.middleware.custom),
                        "tool_error_handler": True # must be true for the workspace agent to work
                    }
                ),
            }
        )
        self.workspace_config: WorkspaceAgentConfig = config
        # Per-instance minimum team size. When supplied, shadow the class-level default
        # via an instance attribute so `self.MIN_AGENTS` (read in _build_team) picks it up
        # while WorkspaceAgent.MIN_AGENTS stays at its class default for everyone else.
        if min_agents is not None:
            if not isinstance(min_agents, int) or isinstance(min_agents, bool):
                raise TypeError(
                    f"min_agents must be an int; got {type(min_agents).__name__}."
                )
            if min_agents < 1:
                raise ValueError(
                    f"min_agents must be a positive integer (>= 1); got {min_agents}."
                )
            self.MIN_AGENTS = min_agents
        # Serializes run()/arun() on this instance: the supervisor's memory/plan and the
        # coworkers' memory are shared mutable state, so concurrent runs would race. The
        # lock makes overlapping calls queue rather than corrupt each other.
        self._run_lock = threading.RLock()
        # Set by close() before it waits for _run_lock. That ordering prevents a queued
        # run/arun worker from starting after deterministic teardown has begun.
        self._closed = False

        # Bind to a workspace session, auto-creating a default one if the caller omitted it.
        session, owns_session = self._ensure_session(
            session, policy=policy, settings=settings
        )
        self.session: Optional[WorkspaceSession] = session
        #: Whether this agent created the session (and so should clean it up on close()).
        self._owns_session: bool = owns_session
        
        # Leak safety net: if the caller drops an agent that owns its session without
        # close()/the context manager, release the session on GC (and at interpreter exit)
        # rather than orphaning a live sandbox/process. weakref.finalize fires the callback
        # at most once; close() detaches it for the deterministic, warning-free path. The
        # callback must NOT capture self -- that would keep the agent alive and defeat the
        # finalizer -- so it closes over the session and name only.
        self._session_finalizer: Optional[weakref.finalize] = (
            weakref.finalize(
                self, _warn_unclosed_and_release, session, self.workspace_config.name
            )
            if owns_session and session is not None
            else None
        )

        # Once we own a session, any failure while building the underlying agent (team,
        # middleware, super().__init__) would orphan it. Tear down an owned session on
        # any construction error before re-raising. close() only touches the session
        # attributes set above, so it is safe to call before super().__init__ has run.
        try:
            backend = str(getattr(session, "backend", "local_fs")) if session else "local_fs"
            read_only = (
                bool(getattr(getattr(session, "policy", None), "read_only", False))
                if session
                else False
            )
            workspace_path = getattr(session, "root", None) if session else None
            prompt_workspace_path = self._agent_visible_workspace_path(session)

            # Provision the session's tools and merge them onto any already on the config,
            # de-duplicating by name so a tool the caller listed and the session also exposes
            # isn't handed to the model twice.
            session_tools = self._provision_session_tools(session)
            config.tools = _dedupe_tools(list(config.tools) + session_tools)

            # Tools the coworker agents receive. By default coworkers get the workspace
            # session tools only; explicit coworker_tools are the opt-in path for extra
            # non-session tools.
            coworker_tools = self._build_coworker_toolset(
                config,
                session=session,
                session_tools=session_tools,
            )

            # Build the team, padding with coworker agents so the workspace agent always has
            # help. Done before super().__init__ so the routing tools/prompt see the full team,
            # and using coworker_tools so coworkers get the workspace tools but not the
            # supervisor's routing tools (those are added inside super().__init__).
            team = self._build_team(
                agents,
                tools=coworker_tools,
                backend=backend,
                read_only=read_only or config.coworker_tool_profile == "read_only",
                workspace_path=prompt_workspace_path,
            )

            self._configure_middleware(config, workspace_path)
            self._workspace_prompt_context = {
                "backend": backend,
                "read_only": read_only,
                "workspace_path": prompt_workspace_path,
            }

            super().__init__(agents=team, config=config)
        except BaseException:
            # Never let a cleanup failure mask the original construction error.
            try:
                self.close()
            except Exception:
                pass
            raise

    def _build_system_prompt(self, *, extended_prompt: str) -> str:
        context = getattr(self, "_workspace_prompt_context", {})
        return workspace_supervisor_prompt(
            agents=self.agent_map,
            backend=str(context.get("backend", "local_fs")),
            read_only=bool(context.get("read_only", False)),
            workspace_path=context.get("workspace_path"),
            extended_prompt=extended_prompt,
            enable_planning=self.config.enable_planning,
            allow_parallel_agents=self.allow_parallel_agents,
            allow_async_tasks=self.allow_async_tasks,
            allowed_route_types=list(self.allowed_route_types),
        )

    def _build_team(
        self,
        agents: Optional[List[Union[AgentConfig, Agent]]],
        *,
        tools: List[BaseTool],
        backend: str,
        read_only: bool,
        workspace_path: Optional[Any],
    ) -> List[Union[AgentConfig, Agent]]:
        """Pad the caller-supplied agents with coworker agents.

        Fewer than :attr:`MIN_AGENTS` supplied -> fill out to :attr:`MIN_AGENTS`.
        :attr:`MIN_AGENTS` or more supplied -> add one extra coworker to assist in parallel.
        """
        provided: List[Union[AgentConfig, Agent]] = list(agents or [])
        # The Supervisor keys its agent_map by name, so duplicate names collapse into one
        # entry and silently shrink the team below MIN_AGENTS. Reject them up front rather
        # than quietly violating the team-size invariant.
        reserved_names: set[str] = set()
        for entry in provided:
            if entry.name in reserved_names:
                raise ValueError(
                    f"Duplicate agent name {entry.name!r} in supplied agents; agent names "
                    "must be unique (the supervisor keys its team by name)."
                )
            reserved_names.add(entry.name)
        # With unique names, list length == team size, so padding math is exact.
        coworker_count = max(self.MIN_AGENTS - len(provided), 1)
        coworkers = self._build_coworker_agents(
            coworker_count,
            reserved_names=reserved_names,
            tools=tools,
            backend=backend,
            read_only=read_only,
            workspace_path=workspace_path,
        )
        if self.workspace_config.debug == 1:
            log_agent(
                self.workspace_config.name,
                f"Team: {len(provided)} supplied + {len(coworkers)} coworker(s) "
                f"= {len(provided) + len(coworkers)} agent(s)",
            )
        return provided + coworkers

    def _build_coworker_agents(
        self,
        count: int,
        *,
        reserved_names: Optional[set[str]] = None,
        tools: List[BaseTool],
        backend: str,
        read_only: bool,
        workspace_path: Optional[Any],
    ) -> List[Agent]:
        """Create ``count`` coworker agents that share the workspace's model and tools."""
        
        taken: set[str] = set(reserved_names or set())
        coworkers: List[Agent] = []
        index = 0
        while len(coworkers) < count:
            name = _coworker_name(index)
            prompt = coworker_agent_prompt(
                name=name,
                backend=backend,
                read_only=read_only,
                workspace_path=str(workspace_path) if workspace_path else None,
            )
            index += 1
            if name in taken:
                continue
            taken.add(name)
            coworker_config = AgentConfig(
                name=name,
                model=self.workspace_config.model,
                tools=list(tools),
                prompt=prompt,
                debug=self.workspace_config.debug,
                recursion_limit=self.workspace_config.recursion_limit,
                middleware=MiddlewareConfig(tool_error_handler=True),
            )
            coworkers.append(Agent(coworker_config))
        return coworkers

    def _ensure_session(
        self,
        session: Optional[WorkspaceSession],
        *,
        policy: Optional[Any],
        settings: Optional[Any],
    ) -> tuple[Optional[WorkspaceSession], bool]:
        """Return the session to use and whether this agent created (and so owns) it.

        A caller-supplied session is used as-is (the caller owns its lifecycle). Otherwise a
        default session is built via :class:`andromeda.workspace.WorkspaceSession` using
        ``config.workspace_backend``. The default ``"auto"`` backend resolves to the bubblewrap
        sandbox when available and otherwise falls back to ``ephemeral_fs``.
        """
        if session is not None:
            if policy is not None or settings is not None:
                warnings.warn(
                    "policy/settings are ignored when an explicit session is supplied; "
                    "configure them on the session before passing it in.",
                    stacklevel=2,
                )
            return session, False

        # Imported lazily: keeps the andromeda.workspace dependency (and its heavier
        # provider imports) off the andromeda.core import path until a session is needed.
        from andromeda.workspace import (
            BubblewrapProcessSettings,
            GVisorContainerSettings,
            WorkspacePolicy,
            WorkspaceProviderError,
            WorkspaceSession as WorkspaceSessionImpl,
            check_provider_availability,
        )

        backend = self.workspace_config.workspace_backend
        effective_settings = settings

        if backend == "auto":
            bubblewrap = check_provider_availability("bubblewrap_process")
            if bubblewrap.available:
                backend, availability = "bubblewrap_process", bubblewrap
            else:
                backend = "ephemeral_fs"
                availability = check_provider_availability(backend)
        else:
            availability = check_provider_availability(backend)
        if not availability.available:
            raise WorkspaceProviderError(
                f"Workspace backend {backend!r} is unavailable on this host "
                f"({availability.reason})."
            )

        # Fill in default settings for sandbox backends whose settings object is fully
        # defaulted and usable out of the box 
        if effective_settings is None:
            default_settings_factories = {
                "bubblewrap_process": BubblewrapProcessSettings,
                "gvisor_container": GVisorContainerSettings,
            }
            default_factory = default_settings_factories.get(backend)
            if default_factory is not None:
                effective_settings = default_factory()

        created = WorkspaceSessionImpl.create(
            backend=backend,
            root=self.workspace_config.workspace_root,
            policy=policy
            or WorkspacePolicy(
                read_only=self.workspace_config.read_only,
                enable_shell=not self.workspace_config.read_only,
            ),
            settings=effective_settings,
        )
        # Reclaim the freshly created session if anything between here and returning it
        # fails, so a transient error never orphans a live sandbox/process. (This window is
        # outside __init__'s construction guard, since self.session is not set yet.)
        try:
            if self.workspace_config.debug == 1:
                log_agent(
                    self.workspace_config.name,
                    f"Auto-created workspace session (backend={backend}, root={created.root})",
                )
        except BaseException:
            with suppress(Exception):
                _release_session(created)
            raise
        return created, True

    def _provision_session_tools(
        self,
        session: Optional[WorkspaceSession],
        *,
        read_only: Optional[bool] = None,
        enable_shell: Optional[bool] = None,
        tool_profile: Optional[Any] = None,
    ) -> List[BaseTool]:
        """Pull the tool set the session exposes, flattened to a list of tools."""
        if session is None:
            return []

        tool_kwargs = {
            key: value
            for key, value in {
                "read_only": read_only,
                "enable_shell": enable_shell,
                "tool_profile": tool_profile,
            }.items()
            if value is not None
        }
        provided = session.tools(**tool_kwargs)
        tools = list(provided.values()) if hasattr(provided, "values") else list(provided)

        if self.workspace_config.debug == 1:
            log_agent(
                self.workspace_config.name,
                f"Provisioned {len(tools)} tool(s) from {getattr(session, 'backend', '?')} session",
            )
        return tools

    def _build_coworker_toolset(
        self,
        config: WorkspaceAgentConfig,
        *,
        session: Optional[WorkspaceSession],
        session_tools: List[BaseTool],
    ) -> List[BaseTool]:
        """Return tools for auto-spawned coworkers according to coworker_tool_profile."""
        if config.coworker_tool_profile == "read_only":
            read_only_session_tools = self._provision_session_tools(
                session,
                read_only=True,
                enable_shell=False,
            )
            return _dedupe_tools(list(config.coworker_tools) + read_only_session_tools)

        return _dedupe_tools(list(config.coworker_tools) + session_tools)

    def _agent_visible_workspace_path(
        self,
        session: Optional[WorkspaceSession],
    ) -> Optional[str]:
        """Return the workspace path to expose in prompts.

        Local/materialized backends operate directly on ``session.root``. Provider-backed
        sandboxes mount or copy that root to a sandbox-internal workspace path, and that is
        the path the agent should reason about when using shell and aliased file tools.
        """
        if session is None:
            return None

        provider_state = getattr(session, "provider_state", None)
        if getattr(provider_state, "provider_backed_shell", False):
            provider = getattr(session, "provider", None)
            settings = getattr(provider, "settings", None)
            for attribute in ("workspace_mount", "workspace_path"):
                value = getattr(settings, attribute, None)
                if value:
                    return str(value)

            metadata = getattr(provider_state, "metadata", {}) or {}
            if isinstance(metadata, Mapping):
                sandbox = metadata.get("sandbox")
                if isinstance(sandbox, Mapping) and sandbox.get("workspace_path"):
                    return str(sandbox["workspace_path"])
                if metadata.get("workspace_path"):
                    return str(metadata["workspace_path"])

        workspace_path = getattr(session, "root", None)
        return str(workspace_path) if workspace_path else None

    def _configure_middleware(
        self, config: WorkspaceAgentConfig, workspace_path: Optional[Any]
    ) -> None:
        """Append long-horizon and skills middleware to the agent's custom middleware."""
        custom = config.middleware.custom

        if not any(isinstance(m, ContextEditingMiddleware) for m in custom):
            custom.append(ContextEditingMiddleware())

        # Skills middleware is attached only when skill sources are configured
        if config.skill_sources and not any(
            isinstance(m, SkillsMiddleware) for m in custom
        ):
            custom.append(
                SkillsMiddleware(
                    backend=config.skills_backend,
                    sources=config.skill_sources,
                    repo_root=str(workspace_path) if workspace_path else None,
                )
            )

    def run(
        self,
        prompt: str,
        *,
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Union[str, list]]:
        """Run a task to completion inside the workspace and return the final report.

        Drives the inherited supervisor orchestration loop (:meth:`Supervisor.supervise`)
        so the workspace agent can both act directly with its workspace tools and assign
        sub tasks to its team of agents, the way a supervisor does.

        Runs under a per-instance lock: the supervisor's memory/plan and the coworkers'
        memory are shared mutable state, so overlapping ``run``/``arun`` calls on the same
        agent queue rather than racing. The returned value is the final message content,
        usually a string but a list for structured/multimodal content.

        ``thread_id`` and ``metadata`` are forwarded to the underlying model invocation
        (checkpointer/tracing). They do NOT scope the agent's conversation: the supervisor's
        ``memory`` and ``plan`` are instance-level and not keyed by ``thread_id``, so two
        runs on the same agent share and grow the same history no matter what ``thread_id``
        is passed. The lock keeps that shared state from being corrupted, not from being
        intermingled -- for genuinely isolated conversations, use one agent per thread.
        """
        with self._run_lock:
            if getattr(self, "_closed", False):
                raise RuntimeError("WorkspaceAgent is closed and cannot run new tasks.")
            result = self.supervise(
                {"messages": [HumanMessage(content=prompt)], "plan": []},
                thread_id=thread_id,
                metadata=metadata,
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            return messages[-1].content if messages else None

    async def arun(
        self,
        prompt: str,
        *,
        thread_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Union[str, list]]:
        """Async variant of :meth:`run`.

        :meth:`Supervisor.supervise` is synchronous, so it is run off the event loop to
        avoid blocking it. Like :meth:`run`, concurrent calls on the same agent are
        serialized by the instance lock rather than allowed to race on shared state, and the
        same one-instance-per-conversation caveat applies (``thread_id`` does not isolate
        conversations on a shared agent -- see :meth:`run`).
        """
        return await asyncio.to_thread(
            self.run, prompt, thread_id=thread_id, metadata=metadata
        )

    def close(self) -> None:
        """Release an auto-created workspace session via its lifecycle hook.

        Only sessions this agent created are torn down; a caller-supplied session is left
        alone, since its lifecycle belongs to the caller. Safe to call repeatedly and from
        multiple threads: teardown is gated on :meth:`weakref.finalize.detach`, which hands
        the session to exactly one caller, so a second close() (e.g. an explicit call
        followed by __exit__) or a concurrent one is a no-op rather than a double teardown.
        The close path marks the agent closed before waiting on the run lock, then releases
        the session under that lock so a cancelled ``arun()`` worker cannot keep using
        workspace tools while cleanup tears them down.
        """
        # getattr guard: a caller-supplied (unowned) session has no finalizer, and the
        # attribute may be absent if close() runs via the construction guard before it was
        # set. Either way there is nothing this agent owns to release.
        finalizer = getattr(self, "_session_finalizer", None)
        if finalizer is None:
            return
        # detach() atomically marks the finalizer dead and returns its callback args to the
        # first caller only (None thereafter, and to losers of a concurrent race). It both
        # disarms the GC path so it can't re-release and serves as the at-most-once gate for
        # close() itself. We release directly here -- bypassing the finalizer callback -- so
        # this deterministic path does not emit the unclosed-resource warning.
        self._closed = True
        if finalizer.detach() is None:
            return
        with self._run_lock:
            self._owns_session = False
            _release_session(self.session)

    # Allow `with WorkspaceAgent(...) as agent:` to scope the session lifecycle.
    def __enter__(self) -> "WorkspaceAgent":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # Async-context variant for `async with WorkspaceAgent(...) as agent:`; close() is a
    # synchronous teardown and may wait for an in-flight run to leave the workspace.
    async def __aenter__(self) -> "WorkspaceAgent":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
