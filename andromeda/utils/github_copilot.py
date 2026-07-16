"""GitHub Copilot chat model for Andromeda.

Exposes :class:`ChatGithubCopilot` and :class:`GithubCopilotEmbeddings`, thin
subclasses of ``langchain_openai.ChatOpenAI`` / ``OpenAIEmbeddings`` pointed at
GitHub Copilot's OpenAI-compatible ``/chat/completions`` and ``/embeddings`` APIs
(``https://api.githubcopilot.com``). Because the results are real LangChain
``BaseChatModel`` / ``Embeddings`` objects, they slot into Andromeda's existing
orchestration, tools, memory, callbacks and streaming with no caller changes —
constructed via :func:`andromeda.utils.langtils.get_chat_model` and
:func:`get_embedding_model` exactly like every other model.

The defining feature is **automatic authentication from the environment**: when a
developer already has a Copilot subscription signed in inside VSCode (the target
use case — Andromeda running as a CLI subagent that Copilot invokes), this module
discovers the editor's OAuth token, exchanges it for a short-lived Copilot token,
and uses it transparently. No API keys, no litellm, no manual setup. Overrides are
accepted via ``ModelConfig.other_args`` (``github_token``, ``copilot_token``,
``base_url``, ``editor_version``, ``default_headers``, ``auto_login``).

When no auth can be discovered, the model can fall back to a one-time interactive
device-flow login (see :func:`device_login`) instead of raising. This is on by
default in an interactive terminal and off otherwise; force it either way with
``other_args={"auto_login": True/False}`` or the ``ANDROMEDA_COPILOT_AUTO_LOGIN``
env var.

Note: this reuses Copilot's internal token-exchange endpoint and editor headers,
which are reverse-engineered rather than a sanctioned public API. It can break if
GitHub changes those internals, and programmatic use sits in a grey area of the
Copilot terms of service. Intended for personal/subscription-backed CLI use.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple

import httpx
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from andromeda.utils.logger import log_output

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BASE_URL = "https://api.githubcopilot.com"
TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"

# OAuth App client id used by the VSCode Copilot Chat extension (device flow).
CLIENT_ID = "Iv1.b507a08c87ecfe98"

# Where we cache exchanged Copilot tokens (separate from the editor's own files).
CACHE_PATH = Path.home() / ".andromeda-copilot.json"
_CACHE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR

# Network calls should fail promptly instead of hanging an agent indefinitely.
_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=60.0)

# Buffer (seconds) before token expiry to trigger a proactive refresh.
_TOKEN_REFRESH_BUFFER_SECS = 60

# GitHub token prefixes that can be exchanged for a short-lived Copilot token.
# Copilot tokens themselves start with "tid=" and must NOT be re-exchanged.
_EXCHANGEABLE_TOKEN_PREFIXES = ("gho_", "ghp_", "ghu_", "github_pat_")

# Env vars that may carry a *ready* Copilot token (used as-is, no exchange).
_READY_TOKEN_ENV_VARS = ("GITHUB_COPILOT_TOKEN", "COPILOT_API_KEY")
# Env vars that may carry an exchangeable GitHub OAuth/PAT token.
_GITHUB_TOKEN_ENV_VARS = ("GITHUB_TOKEN", "GH_TOKEN")

# Editor version/identity headers. Copilot rejects requests without these.
# These version values are validated loosely (a minimum-supported floor, not an
# exact match), so the pinned defaults below act as a fallback; see
# _resolve_editor_version for the auto-refresh chain.
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_PLUGIN_VERSION = "copilot-chat/0.26.7"
COPILOT_INTEGRATION_ID = "vscode-chat"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.7"

# Env var to override Editor-Version fleet-wide without a code change.
_EDITOR_VERSION_ENV_VAR = "GITHUB_COPILOT_EDITOR_VERSION"

# Env var to opt in/out of an automatic one-time device-flow login when no
# Copilot auth can be discovered. Unset -> auto-enabled only in an interactive
# terminal (so a human can see the verification URL/code); non-interactive
# processes (CI, daemons, tests) stay non-blocking.
_AUTO_LOGIN_ENV_VAR = "ANDROMEDA_COPILOT_AUTO_LOGIN"


def _resolve_editor_version(explicit: Optional[str] = None) -> str:
    """Resolve the ``Editor-Version`` header value.

    Priority order, so the header stays fresh without code edits:
    1. ``explicit`` (``other_args["editor_version"]``)
    2. ``GITHUB_COPILOT_EDITOR_VERSION`` env var (ops can bump fleet-wide)
    3. ``vscode/<TERM_PROGRAM_VERSION>`` when launched from a VSCode terminal
    4. the pinned :data:`COPILOT_EDITOR_VERSION` fallback
    """
    if explicit:
        return explicit
    env_version = os.environ.get(_EDITOR_VERSION_ENV_VAR)
    if env_version:
        return env_version
    if os.environ.get("TERM_PROGRAM") == "vscode":
        term_version = os.environ.get("TERM_PROGRAM_VERSION")
        if term_version:
            return f"vscode/{term_version}"
    return COPILOT_EDITOR_VERSION


def _build_default_headers(editor_version: Optional[str] = None) -> Dict[str, str]:
    """Build the editor-identity headers every Copilot API call must carry.

    Copilot's backend rejects requests that don't look like they came from a
    supported editor, so we present the VSCode Copilot Chat integration id,
    user-agent and editor/plugin versions. The header names are duplicated in
    both ``Title-Case`` and ``lower-case`` because different Copilot endpoints
    look for different casings.
    """
    ev = _resolve_editor_version(editor_version)
    return {
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "User-Agent": COPILOT_USER_AGENT,
        "Editor-Version": ev,
        "Editor-Plugin-Version": COPILOT_PLUGIN_VERSION,
        "editor-version": ev,
        "editor-plugin-version": COPILOT_PLUGIN_VERSION,
        "copilot-vision-request": "true",
    }


# --------------------------------------------------------------------------- #
# Token cache
# --------------------------------------------------------------------------- #


def save_tokens_to_cache(
    github_token: Optional[str],
    copilot_token: str,
    expires_at: Optional[float] = None,
) -> None:
    """Persist the exchanged Copilot token (best-effort)."""
    try:
        fd = os.open(
            CACHE_PATH,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            _CACHE_FILE_MODE,
        )
        with os.fdopen(fd, "w") as f:
            os.chmod(CACHE_PATH, _CACHE_FILE_MODE)
            json.dump(
                {
                    "github_token": github_token,
                    "copilot_token": copilot_token,
                    "expires_at": expires_at,
                },
                f,
                indent=2,
            )
    except OSError as exc:
        log_output(f"Failed to save Copilot token cache to {CACHE_PATH}: {exc}")


def _github_token_only(data: Dict[str, Any]) -> Dict[str, Any]:
    github_token = data.get("github_token")
    if isinstance(github_token, str) and github_token:
        return {"github_token": github_token}
    return {}


def load_tokens_from_cache() -> Dict[str, Any]:
    """Load cached tokens, dropping the Copilot token once it has expired.

    The long-lived ``github_token`` (OAuth) is preserved even when the
    short-lived Copilot token has expired, so it can be transparently
    re-exchanged for a fresh Copilot token without forcing another login.
    """
    try:
        os.chmod(CACHE_PATH, _CACHE_FILE_MODE)
        with open(CACHE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        if data.get("copilot_token"):
            expires_at = data.get("expires_at")
            if expires_at is None or time.time() > float(expires_at):
                return _github_token_only(data)
        return data
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        log_output(f"Failed to load Copilot token cache from {CACHE_PATH}: {exc}")
        return {}


# --------------------------------------------------------------------------- #
# Environment auth discovery
# --------------------------------------------------------------------------- #


def _find_editor_config_dirs() -> List[Path]:
    """Candidate directories where editors persist Copilot OAuth tokens."""
    dirs: List[Path] = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        dirs.append(Path(xdg) / "github-copilot")
    dirs.append(Path.home() / ".config" / "github-copilot")
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        dirs.append(Path(local_appdata) / "github-copilot")
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "github-copilot")
    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: List[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _extract_oauth_token(entry: Any) -> Optional[str]:
    """Pull the ``oauth_token`` string out of one editor auth-file entry.

    Returns ``None`` if the entry isn't a dict or has no usable token.
    """
    if isinstance(entry, dict):
        tok = entry.get("oauth_token")
        if isinstance(tok, str) and tok:
            return tok
    return None


def _read_oauth_token_from_editor_files() -> Optional[str]:
    """Read a GitHub OAuth token from the Copilot editor auth files.

    Supports both the newer ``apps.json`` (keys like ``github.com:Iv1.<id>``)
    and the older ``hosts.json`` (keys like ``github.com``) layouts written by
    the official Copilot editor plugins.
    """
    for directory in _find_editor_config_dirs():
        for filename in ("apps.json", "hosts.json"):
            path = Path(directory) / filename
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except FileNotFoundError:
                continue
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                log_output(f"Failed to read Copilot auth file {path}: {exc}")
                continue
            if not isinstance(data, dict):
                continue
            for key, entry in data.items():
                if isinstance(key, str) and key.startswith("github.com"):
                    tok = _extract_oauth_token(entry)
                    if tok:
                        log_output(f"Discovered Copilot OAuth token from {path}")
                        return tok
    return None


def resolve_github_oauth_token(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a GitHub OAuth/PAT token from the environment, in priority order.

    1. ``explicit`` (e.g. ``other_args["github_token"]``)
    2. ``GITHUB_TOKEN`` / ``GH_TOKEN`` env vars
    3. Editor Copilot auth files (``~/.config/github-copilot/{apps,hosts}.json``,
       ``%LOCALAPPDATA%/github-copilot`` on Windows)
    4. Our own token cache
    """
    if explicit:
        return explicit
    for var in _GITHUB_TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    from_files = _read_oauth_token_from_editor_files()
    if from_files:
        return from_files
    cached = load_tokens_from_cache().get("github_token")
    return cached if isinstance(cached, str) and cached else None


def _is_exchangeable_github_token(token: str) -> bool:
    """True if ``token`` is a GitHub OAuth/PAT that can be swapped for a Copilot token.

    Distinguishes exchangeable GitHub tokens (``gho_``/``ghp_``/...) from a
    ready-to-use Copilot token (``tid=...``), which must never be re-exchanged.
    """
    return token.startswith(_EXCHANGEABLE_TOKEN_PREFIXES)


def fetch_copilot_token(github_token: str) -> Tuple[Optional[str], Optional[float]]:
    """Exchange a GitHub OAuth/PAT token for a short-lived Copilot token."""
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **_build_default_headers(),
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        res = client.get(TOKEN_EXCHANGE_URL, headers=headers)
    if res.status_code == 200:
        data = res.json()
        return data.get("token"), data.get("expires_at")
    log_output(f"Copilot token exchange failed (HTTP {res.status_code}).")
    return None, None


def fetch_copilot_session(github_token: str) -> Dict[str, Any]:
    """Return the full Copilot token-exchange response.

    Beyond the token itself this payload carries account metadata — notably
    ``copilot_plan`` and quota fields (``quota_snapshots`` / ``quota_reset_date``
    on current plans, ``limited_user_quotas`` on the free plan).
    """
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **_build_default_headers(),
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        res = client.get(TOKEN_EXCHANGE_URL, headers=headers)
    res.raise_for_status()
    return res.json()


def get_copilot_limits(github_token: Optional[str] = None) -> Dict[str, Any]:
    """Summarize the account's Copilot plan and remaining quota.

    Resolves a GitHub OAuth token (override or environment) and reads the quota
    fields from the token-exchange response. Raises ``ValueError`` if no
    exchangeable token is available.
    """
    gh = resolve_github_oauth_token(github_token)
    if not gh or not _is_exchangeable_github_token(gh):
        raise ValueError(
            "Need an exchangeable GitHub token to read Copilot limits. "
            "Run device_login() or set GITHUB_TOKEN."
        )
    data = fetch_copilot_session(gh)
    return {
        "copilot_plan": data.get("copilot_plan"),
        "sku": data.get("sku"),
        "chat_enabled": data.get("chat_enabled"),
        "quota_reset_date": data.get("quota_reset_date")
        or data.get("limited_user_reset_date"),
        "quota_snapshots": data.get("quota_snapshots"),
        "limited_user_quotas": data.get("limited_user_quotas"),
    }


def _supports_chat(model: Dict[str, Any]) -> bool:
    """Whether a model entry supports the /chat/completions endpoint.

    Models that omit ``supported_endpoints`` are legacy chat models, so treat a
    missing field as chat-capable.
    """
    endpoints = model.get("supported_endpoints")
    if endpoints is None:
        return True
    return "/chat/completions" in endpoints


_MODEL_NOT_SUPPORTED_CODE = "model_not_supported"

# How many models to probe for live access concurrently. Each probe is a
# single cheap request (1 output token / 1-item embedding input), so this is
# tuned for latency, not for protecting any particular quota.
_ACCESS_PROBE_WORKERS = 8


def _policy_disabled(model: Dict[str, Any]) -> bool:
    """True if the account/org has explicitly toggled this model off.

    A ``policy.state`` of ``"disabled"`` is a reliable exclude signal (verified
    live: every disabled-policy model 400s with ``model_not_supported``), so
    callers can skip probing those and save a round trip. The converse is
    *not* true — ``"enabled"`` or a missing ``policy`` block does not imply
    the account can actually call the model (see :func:`_probe_model_access`).
    """
    policy = model.get("policy")
    return isinstance(policy, dict) and policy.get("state") == "disabled"


def _probe_model_access(
    client: httpx.Client, headers: Dict[str, str], model: Dict[str, Any]
) -> Optional[bool]:
    """Best-effort live check of whether this account can actually call ``model``.

    GitHub's ``/models`` catalog is not a reliable source of truth for
    per-account access: on a free/limited plan, legacy models with no
    ``policy`` block at all (e.g. ``gpt-3.5-turbo``, ``gpt-4``) 400 with
    ``model_not_supported`` while sibling models in the same shape (``gpt-4o``,
    ``gpt-4.1``) work, and premium models with ``policy.state == "enabled"``
    (e.g. ``gpt-5-mini``, ``claude-haiku-4.5``) 400 as well because GitHub
    additionally gates them to its first-party clients. So the only accurate
    signal is a live, minimal call to the actual endpoint.

    Returns ``True``/``False`` when the endpoint gives an unambiguous answer,
    or ``None`` if the probe itself failed (network error, unexpected status)
    — callers should treat ``None`` as "keep it" rather than hide a model due
    to a transient problem.
    """
    mtype = (model.get("capabilities") or {}).get("type")
    model_id = model.get("id")
    if not model_id:
        return None
    try:
        if mtype == "embeddings":
            res = client.post(
                f"{BASE_URL}/embeddings",
                headers=headers,
                json={"model": model_id, "input": ["ping"]},
            )
        else:
            res = client.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
    except httpx.HTTPError as exc:
        log_output(f"Copilot access probe failed for {model_id!r}: {exc}")
        return None

    if res.status_code == 200:
        return True
    if res.status_code == 400:
        try:
            code = res.json().get("error", {}).get("code")
        except (json.JSONDecodeError, AttributeError):
            code = None
        if code == _MODEL_NOT_SUPPORTED_CODE:
            return False
    # Any other status (429 rate-limited, 5xx, a different 400 reason, ...)
    # doesn't tell us the model is unsupported, so don't filter it out.
    return None


def _filter_accessible_models(
    models: List[Dict[str, Any]], token: str
) -> List[Dict[str, Any]]:
    """Filter ``models`` down to those this account can actually call.

    Skips the live probe for models GitHub has already told us are disabled
    (see :func:`_policy_disabled`); probes the rest concurrently.
    """
    candidates = [m for m in models if not _policy_disabled(m)]
    if not candidates:
        return []

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        **_build_default_headers(),
    }
    accessible: List[Dict[str, Any]] = []
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_ACCESS_PROBE_WORKERS
        ) as pool:
            results = pool.map(
                lambda m: _probe_model_access(client, headers, m), candidates
            )
            for model, result in zip(candidates, results):
                if result is not False:
                    accessible.append(model)
    return accessible


def list_models(
    github_token: Optional[str] = None,
    chat_only: bool = True,
    check_access: bool = False,
) -> List[Dict[str, Any]]:
    """List the models the account can use via the Copilot ``/models`` endpoint.

    Set ``chat_only=False`` to include non-chat (e.g. embedding) models.

    Set ``check_access=True`` to filter the catalog down to models this
    account can actually invoke right now. This is *not* derivable from the
    catalog's own metadata (see :func:`_probe_model_access` for why), so it
    costs one extra minimal live request per remaining candidate model
    (parallelized, but still real API calls) — leave it off (the default) to
    just list the raw catalog GitHub advertises.
    """
    token, _ = discover_copilot_token(github_token=github_token)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        **_build_default_headers(),
    }
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        res = client.get(f"{BASE_URL}/models", headers=headers)
    res.raise_for_status()
    models = res.json().get("data", [])
    if chat_only:
        models = [m for m in models if _supports_chat(m)]
    if check_access:
        models = _filter_accessible_models(models, token)
    return models


def _format_token_limit(n: Any) -> str:
    """Compact a token count, e.g. 128000 -> '128k'."""
    if not isinstance(n, (int, float)):
        return "-"
    n = int(n)
    return f"{n // 1000}k" if n >= 1000 else str(n)


def format_models(models: List[Dict[str, Any]]) -> str:
    """Render :func:`list_models` output as a readable, grouped table.

    Groups by capability type (chat / embeddings / ...) and shows the model id,
    display name, vendor, context window and notable capability flags.
    """
    if not models:
        return "No GitHub Copilot models available."

    # Bucket models by capability type so chat and embedding models print apart.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for m in models:
        mtype = (m.get("capabilities") or {}).get("type") or "other"
        groups.setdefault(mtype, []).append(m)

    lines = [f"GitHub Copilot models ({len(models)} available)"]
    for mtype in sorted(groups):
        entries = sorted(groups[mtype], key=lambda m: m.get("id", ""))
        # Size each text column to its widest value (header included) for alignment.
        id_w = max([len("ID")] + [len(m.get("id", "")) for m in entries])
        name_w = max([len("NAME")] + [len(m.get("name", "")) for m in entries])
        ven_w = max([len("VENDOR")] + [len(m.get("vendor", "")) for m in entries])

        lines.append("")
        lines.append(f"{mtype.upper()} ({len(entries)})")
        lines.append(
            f"  {'ID':<{id_w}}  {'NAME':<{name_w}}  {'VENDOR':<{ven_w}}  CONTEXT  CAPABILITIES"
        )
        for m in entries:
            caps = m.get("capabilities") or {}
            limits = caps.get("limits") or {}
            supports = caps.get("supports") or {}
            # Chat models report a context window; embeddings report a max input count.
            ctx = _format_token_limit(
                limits.get("max_context_window_tokens") or limits.get("max_inputs")
            )
            flags = [k for k in ("tool_calls", "vision", "streaming", "dimensions") if supports.get(k)]
            lines.append(
                f"  {m.get('id', ''):<{id_w}}  {m.get('name', ''):<{name_w}}  "
                f"{m.get('vendor', ''):<{ven_w}}  {ctx:>7}  {', '.join(flags)}"
            )
    return "\n".join(lines)


def print_models(
    github_token: Optional[str] = None,
    chat_only: bool = False,
    check_access: bool = True,
) -> None:
    """Fetch and print the Copilot models this account can actually use.

    A friendly companion to :func:`list_models` (which returns raw dicts).
    Defaults to ``check_access=True``, so the table only shows models this
    account can currently invoke rather than GitHub's full catalog (many
    catalog entries 400 with ``model_not_supported`` for a given plan; pass
    ``check_access=False`` to see the raw, unfiltered catalog instead).
    """
    try:
        plan = get_copilot_limits(github_token)
        sku = plan.get("sku") or plan.get("copilot_plan") or "unknown"
        print(f"GitHub Copilot plan: {sku}\n")
    except ValueError:
        pass  # No exchangeable token to inspect the plan; still list models.
    print(
        format_models(
            list_models(
                github_token=github_token,
                chat_only=chat_only,
                check_access=check_access,
            )
        )
    )


async def afetch_copilot_token(
    github_token: str,
) -> Tuple[Optional[str], Optional[float]]:
    """Async variant of :func:`fetch_copilot_token`."""
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **_build_default_headers(),
    }
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        res = await client.get(TOKEN_EXCHANGE_URL, headers=headers)
    if res.status_code == 200:
        data = res.json()
        return data.get("token"), data.get("expires_at")
    log_output(f"Copilot token exchange failed (HTTP {res.status_code}).")
    return None, None


def _resolve_auto_login(explicit: Optional[bool] = None) -> bool:
    """Decide whether to fall back to interactive device login on auth failure.

    Priority: explicit override (``other_args["auto_login"]``) > the
    ``ANDROMEDA_COPILOT_AUTO_LOGIN`` env var > interactive-terminal autodetect.
    Defaults to on only when stdin/stdout are a TTY, so the blocking device
    prompt never hangs a non-interactive process (CI, daemons, tests).
    """
    if explicit is not None:
        return bool(explicit)
    env = os.environ.get(_AUTO_LOGIN_ENV_VAR)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def discover_copilot_token(
    github_token: Optional[str] = None,
    copilot_token: Optional[str] = None,
    auto_login: bool = False,
) -> Tuple[str, Optional[str]]:
    """Resolve a usable Copilot bearer token from overrides + the environment.

    Returns ``(copilot_token, github_oauth_token)``; the second value is the
    underlying OAuth token (when known) so callers can refresh later. When
    ``auto_login`` is true and nothing else resolves, fall back to a one-time
    interactive :func:`device_login` (which blocks until the user authorizes)
    before giving up. Raises ``ValueError`` with an actionable message if
    nothing yields a token.
    """
    # 1. Explicit ready Copilot token.
    if copilot_token:
        return copilot_token, github_token
    for var in _READY_TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val, github_token

    # 2. Valid cached Copilot token.
    cached = load_tokens_from_cache()
    cached_token = cached.get("copilot_token")
    cached_exp = cached.get("expires_at")
    if cached_token and (
        cached_exp is not None
        and time.time() < float(cached_exp) - _TOKEN_REFRESH_BUFFER_SECS
    ):
        return cached_token, github_token or cached.get("github_token")

    # 3. Resolve a GitHub OAuth token and exchange it.
    gh_token = resolve_github_oauth_token(github_token)
    if gh_token and _is_exchangeable_github_token(gh_token):
        new_token, expires_at = fetch_copilot_token(gh_token)
        if new_token:
            save_tokens_to_cache(gh_token, new_token, expires_at)
            return new_token, gh_token

    # 4. Fall back to using the raw token directly (fine-grained PATs / enterprise).
    if gh_token:
        return gh_token, gh_token

    # 5. Last resort (opt-in): interactive one-time device login, then re-read cache.
    if auto_login:
        log_output("No GitHub Copilot auth found; starting one-time device login.")
        device_login()
        refreshed = load_tokens_from_cache()
        token = refreshed.get("copilot_token")
        if token:
            return token, refreshed.get("github_token")

    raise ValueError(
        "Could not authenticate with GitHub Copilot. Sign in to Copilot in VSCode, "
        "or set GITHUB_TOKEN (a token with Copilot access), or pass "
        "other_args={'github_token': ...} / other_args={'copilot_token': ...}. "
        "Run device_login() once to cache a token, or set "
        "ANDROMEDA_COPILOT_AUTO_LOGIN=1 to authenticate automatically on first use."
    )


def device_login(
    client_id: str = CLIENT_ID,
    on_message: Optional[Callable[[str], None]] = None,
) -> str:
    """Authenticate via GitHub's device flow and cache a Copilot token.

    Intended for headless/remote CLI use (e.g. a VSCode remote-server shell where
    no Copilot auth file is exposed). Prints a verification URL + user code, blocks
    until the user authorizes in a browser, exchanges the resulting OAuth token for
    a short-lived Copilot token, caches it to :data:`CACHE_PATH`, and returns it.
    After this, :func:`discover_copilot_token` resolves automatically from cache.
    """
    emit = on_message or print
    with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
        res = client.post(
            "https://github.com/login/device/code",
            headers={"Accept": "application/json"},
            data={"client_id": client_id, "scope": "read:user"},
        )
        res.raise_for_status()
        data = res.json()

    device_code = data["device_code"]
    interval = int(data.get("interval", 5))
    emit(
        f"\nOpen {data['verification_uri']} and enter the code: {data['user_code']}\n"
        f"Waiting for authorization (polling every {interval}s)..."
    )

    # Poll GitHub's token endpoint until the user finishes authorizing in the
    # browser. GitHub asks us to wait `interval` seconds between polls and to
    # back off further whenever it answers "slow_down".

    #we do not want the standard timeout here as the user should be given an ample amount of time to authenticate/login/deal with security measures
    access_token: Optional[str] = None
    with httpx.Client() as client:
        while access_token is None:
            time.sleep(interval)
            tok = client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": client_id,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            ).json()
            if "access_token" in tok:
                access_token = tok["access_token"]
            elif tok.get("error") == "authorization_pending":
                continue
            elif tok.get("error") == "slow_down":
                interval += 5
            else:
                raise RuntimeError(f"Device authorization failed: {tok}")

    copilot_token, expires_at = fetch_copilot_token(access_token)
    if not copilot_token:
        raise RuntimeError(
            "Authorized with GitHub, but failed to obtain a Copilot token. "
            "Is GitHub Copilot enabled for this account?"
        )
    save_tokens_to_cache(access_token, copilot_token, expires_at)
    emit("Copilot token acquired and cached.")
    return copilot_token


# --------------------------------------------------------------------------- #
# Chat model
# --------------------------------------------------------------------------- #

try:  # langchain-openai is an optional extra; only import when this module loads.
    import openai
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import Field, SecretStr, model_validator
except ImportError as exc:  # pragma: no cover - surfaced with an actionable message
    raise ImportError(
        "The 'github_copilot' provider requires langchain-openai. Install it with: "
        "pip install 'andromeda[github-copilot]'"
    ) from exc


# Locks to avoid concurrent token-refresh stampedes.
_sync_refresh_lock = threading.Lock()
_sync_refresh_result = False
_sync_refresh_token: Optional[str] = None
_async_refresh_lock: Optional[asyncio.Lock] = None


def _get_async_refresh_lock() -> asyncio.Lock:
    """Return the process-wide async refresh lock, creating it on first use.

    The lock is created lazily (not at import time) so it binds to whatever
    event loop is actually running.
    """
    global _async_refresh_lock
    if _async_refresh_lock is None:
        _async_refresh_lock = asyncio.Lock()
    return _async_refresh_lock


def _is_auth_error(exc: Exception) -> bool:
    """Whether an OpenAI client error indicates an expired/invalid Copilot token.

    A 401 (``AuthenticationError``) is unambiguous. Copilot also surfaces an
    expired token as a 400 (``BadRequestError``) whose message mentions
    authorization / a "badly formatted" token, so we sniff those too — that is
    the signal to refresh and retry rather than give up.
    """
    if isinstance(exc, openai.AuthenticationError):
        return True
    if isinstance(exc, openai.BadRequestError):
        msg = str(exc).lower()
        return "authorization" in msg or "badly formatted" in msg
    return False


class _CopilotAuthMixin:
    """Shared GitHub Copilot auth + token-refresh for chat and embeddings models.

    Resolves a Copilot bearer token automatically from the environment (see
    :func:`discover_copilot_token`), points the underlying OpenAI-compatible
    client at Copilot's API with the required editor headers, and refreshes the
    short-lived token proactively (near expiry) and reactively (on a 401 / auth
    400). Mixed into a ``ChatOpenAI`` / ``OpenAIEmbeddings`` subclass.
    """

    github_token: Optional[SecretStr] = Field(default=None)
    copilot_token: Optional[SecretStr] = Field(default=None)
    editor_version: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _setup_copilot_auth(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve Copilot auth and rewrite it into ChatOpenAI/OpenAIEmbeddings kwargs.

        Runs before pydantic builds the model, so by the time the underlying
        OpenAI client is constructed it already points at Copilot's API with a
        valid token and the required editor headers. Caller-supplied overrides
        (``github_token``/``copilot_token``/``base_url``/``editor_version``/
        ``default_headers``/``auto_login``) all flow through here.
        """
        if not isinstance(values, dict):
            return values

        # Accept either plain strings or already-wrapped SecretStr values.
        def _unwrap(v: Any) -> Optional[str]:
            if v is None:
                return None
            return v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)

        github_token = _unwrap(values.get("github_token")) or None
        copilot_token = _unwrap(values.get("copilot_token")) or None
        editor_version = values.get("editor_version")
        # auto_login isn't an OpenAI field, so pop it before it reaches super().
        auto_login = _resolve_auto_login(values.pop("auto_login", None))

        api_token, resolved_github = discover_copilot_token(
            github_token, copilot_token, auto_login=auto_login
        )

        # Persist resolved values so refresh helpers can reuse them.
        if resolved_github:
            values["github_token"] = resolved_github
        values["copilot_token"] = api_token

        # Configure the underlying ChatOpenAI client.
        values["openai_api_key"] = api_token
        base_url = values.pop("base_url", None) or values.get("openai_api_base")
        values["openai_api_base"] = base_url or BASE_URL

        user_headers: Dict[str, str] = values.get("default_headers") or {}
        values["default_headers"] = {**_build_default_headers(editor_version), **user_headers}
        return values

    # ---- token refresh ---------------------------------------------------- #

    def _get_github_token(self) -> str:
        """Return the OAuth token used to mint Copilot tokens (cached, else env)."""
        if self.github_token:
            return self.github_token.get_secret_value()
        return resolve_github_oauth_token() or ""

    def _rebuild_clients(self) -> None:
        """Recreate the underlying OpenAI clients after the API key changes.

        The openai SDK captures the api key when its client is built, so after a
        token refresh we must drop the cached clients and let langchain-openai's
        ``validate_environment`` rebuild them with the new key.
        """
        self.client = None
        self.async_client = None
        self.root_client = None
        self.root_async_client = None
        # validate_environment re-creates the OpenAI clients from current fields.
        rebuild = getattr(self, "validate_environment", None)
        if callable(rebuild):
            rebuild()  # type: ignore[operator]
        else:  # pragma: no cover - guards against future langchain-openai renames
            log_output("Could not rebuild OpenAI clients after token refresh.")

    def _install_copilot_token(self, new_token: str) -> None:
        self.openai_api_key = SecretStr(new_token)
        self._rebuild_clients()

    def _apply_new_token(self, new_token: str, expires_at: Optional[float]) -> None:
        """Cache a freshly exchanged Copilot token and wire it into the client."""
        gh = self._get_github_token() or None
        save_tokens_to_cache(gh, new_token, expires_at)
        self._install_copilot_token(new_token)

    def _refresh_copilot_token(self) -> bool:
        """Exchange the OAuth token for a fresh Copilot token. Returns success.

        Guarded by a non-blocking lock so concurrent calls don't stampede the
        exchange endpoint: if another thread already holds it, wait for that
        refresh to finish and return the result it observed.
        """
        global _sync_refresh_result, _sync_refresh_token

        if not _sync_refresh_lock.acquire(blocking=False):
            with _sync_refresh_lock:
                refreshed = _sync_refresh_result
                new_token = _sync_refresh_token
            if refreshed and new_token:
                self._install_copilot_token(new_token)
            return refreshed
        try:
            _sync_refresh_result = False
            _sync_refresh_token = None

            gh = self._get_github_token()
            if not gh or not _is_exchangeable_github_token(gh):
                log_output("Cannot refresh Copilot token: no exchangeable token.")
                return False
            new_token, expires_at = fetch_copilot_token(gh)
            if not new_token:
                return False
            self._apply_new_token(new_token, expires_at)
            _sync_refresh_result = True
            _sync_refresh_token = new_token
            return True
        finally:
            _sync_refresh_lock.release()

    async def _arefresh_copilot_token(self) -> bool:
        """Async variant of :meth:`_refresh_copilot_token`."""
        async with _get_async_refresh_lock():
            gh = self._get_github_token()
            if not gh or not _is_exchangeable_github_token(gh):
                log_output("Cannot refresh Copilot token: no exchangeable token.")
                return False
            new_token, expires_at = await afetch_copilot_token(gh)
            if not new_token:
                return False
            self._apply_new_token(new_token, expires_at)
            return True

    def _maybe_refresh_proactively(self) -> None:
        """Refresh the token before a request if it's within the expiry buffer.

        Cheaper than waiting for a 401 mid-request: avoids a failed call and an
        interrupted stream when the token is about to lapse.
        """
        exp = load_tokens_from_cache().get("expires_at")
        if exp is not None and time.time() >= float(exp) - _TOKEN_REFRESH_BUFFER_SECS:
            self._refresh_copilot_token()

    async def _amaybe_refresh_proactively(self) -> None:
        """Async variant of :meth:`_maybe_refresh_proactively`."""
        exp = load_tokens_from_cache().get("expires_at")
        if exp is not None and time.time() >= float(exp) - _TOKEN_REFRESH_BUFFER_SECS:
            await self._arefresh_copilot_token()


class ChatGithubCopilot(_CopilotAuthMixin, ChatOpenAI):
    """GitHub Copilot chat model via the OpenAI-compatible ``/chat/completions`` API.

    Auth is resolved automatically from the environment (see
    :func:`discover_copilot_token`). Construct via ``get_chat_model`` with
    ``ModelConfig(provider="github_copilot", name="gpt-4o")``.
    """

    @property
    def _llm_type(self) -> str:
        return "github-copilot"

    # ---- generation overrides (preserve streaming semantics) -------------- #
    #
    # Each override wraps the ChatOpenAI implementation with the same two-step
    # token-refresh dance so long-running agents survive Copilot's short token
    # lifetime: refresh proactively before the call if near expiry, and if the
    # call still fails with an auth error, refresh once and retry. _stream /
    # _agenerate / _astream below mirror _generate (sync/async, batch/stream).

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._maybe_refresh_proactively()
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            # Only a genuine auth failure that we can refresh is worth retrying.
            if not _is_auth_error(exc) or not self._refresh_copilot_token():
                raise
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self._maybe_refresh_proactively()
        try:
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not self._refresh_copilot_token():
                raise
            yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self._amaybe_refresh_proactively()
        try:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not await self._arefresh_copilot_token():
                raise
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        await self._amaybe_refresh_proactively()
        try:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not await self._arefresh_copilot_token():
                raise
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk


class GithubCopilotEmbeddings(_CopilotAuthMixin, OpenAIEmbeddings):
    """GitHub Copilot embeddings via the OpenAI-compatible ``/embeddings`` API.

    Auth is resolved automatically from the environment (see
    :func:`discover_copilot_token`), identically to :class:`ChatGithubCopilot`.
    Construct via ``get_embedding_model`` with
    ``ModelConfig(provider="github_copilot", name="text-embedding-3-small")``.
    """

    # Copilot's /embeddings endpoint accepts string inputs but rejects the
    # tiktoken integer-token arrays OpenAIEmbeddings sends when this is True, so
    # default it off (callers can still override via other_args).
    check_embedding_ctx_length: bool = False

    # The four embed methods below mirror the chat model's generation overrides:
    # proactive refresh before the call, and one refresh-and-retry on auth error.

    def embed_documents(self, texts: List[str], *args: Any, **kwargs: Any) -> List[List[float]]:
        self._maybe_refresh_proactively()
        try:
            return super().embed_documents(texts, *args, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not self._refresh_copilot_token():
                raise
            return super().embed_documents(texts, *args, **kwargs)

    def embed_query(self, text: str, *args: Any, **kwargs: Any) -> List[float]:
        self._maybe_refresh_proactively()
        try:
            return super().embed_query(text, *args, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not self._refresh_copilot_token():
                raise
            return super().embed_query(text, *args, **kwargs)

    async def aembed_documents(
        self, texts: List[str], *args: Any, **kwargs: Any
    ) -> List[List[float]]:
        await self._amaybe_refresh_proactively()
        try:
            return await super().aembed_documents(texts, *args, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not await self._arefresh_copilot_token():
                raise
            return await super().aembed_documents(texts, *args, **kwargs)

    async def aembed_query(self, text: str, *args: Any, **kwargs: Any) -> List[float]:
        await self._amaybe_refresh_proactively()
        try:
            return await super().aembed_query(text, *args, **kwargs)
        except (openai.AuthenticationError, openai.BadRequestError) as exc:
            if not _is_auth_error(exc) or not await self._arefresh_copilot_token():
                raise
            return await super().aembed_query(text, *args, **kwargs)


__all__ = [
    "ChatGithubCopilot",
    "GithubCopilotEmbeddings",
    "discover_copilot_token",
    "resolve_github_oauth_token",
    "fetch_copilot_token",
    "afetch_copilot_token",
    "fetch_copilot_session",
    "get_copilot_limits",
    "list_models",
    "format_models",
    "print_models",
    "device_login",
    "BASE_URL",
]
