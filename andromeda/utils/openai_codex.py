"""ChatGPT/Codex (subscription) chat model for Andromeda.

Exposes :class:`ChatOpenAICodex`, a LangChain chat model that talks to OpenAI's
**Codex backend** (``https://chatgpt.com/backend-api/codex``) authenticated with
a **ChatGPT subscription** (Plus/Pro/Team) instead of an OpenAI API key. No
``OPENAI_API_KEY``, no per-token billing: sign in once with your ChatGPT
account and every request rides the subscription.

It builds on ``langchain_openai``'s experimental Codex support
(``_ChatOpenAICodex`` + ``chatgpt_oauth``, shipped since ``langchain-openai``
1.3.1): an OAuth 2.0 Authorization Code + PKCE flow stores a refresh-aware
token bundle at ``~/.langchain/chatgpt-auth.json`` (deliberately *not*
``~/.codex/auth.json``, so it never invalidates your Codex CLI / VS Code
session), and the model injects fresh ``Authorization`` /
``ChatGPT-Account-Id`` headers on every request.

Because :class:`ChatOpenAICodex` is a real ``ChatOpenAI`` subclass (Responses
API), it slots into Andromeda's orchestration, tools, memory, callbacks and
streaming with no caller changes — including **native tool calling** and
``with_structured_output``. Construct it via
:func:`andromeda.utils.langtils.get_chat_model` with
``ModelConfig(provider="openai_codex", name="gpt-5.5")`` like every other
model. The name must be a **Codex model slug** (regular API names like
``gpt-4`` are rejected by this backend) — see :func:`print_models` for what
your plan can call.

Setup
-----
1. ``pip install 'andromeda[openai-codex]'`` (needs ``langchain-openai>=1.3.1``).
2. Sign in once via the browser-based Authorization Code + PKCE flow (opens
   your system browser to the OpenAI sign-in page; the URL is also printed
   as a fallback)::

       from andromeda.utils.openai_codex import login_device
       login_device()

   Or just construct the model in an interactive terminal and skip this step
   entirely: with no token on disk, ``ChatOpenAICodex`` runs :func:`login_device`
   for you automatically (see ``auto_login`` below).

Notes
-----
- The Codex backend forces ``use_responses_api=True``, ``store=False`` and
  ``streaming=True`` at the wire level; ``invoke`` still returns a single
  aggregated ``AIMessage``.
- Codex rejects system-role chat turns; ``SystemMessage`` content is lifted
  into the Responses-API top-level ``instructions`` field automatically.
- There is no embeddings counterpart — the subscription does not expose an
  embeddings endpoint.
- Use it only where your OpenAI account, plan, and OpenAI's terms permit ChatGPT-authenticated Codex access.

Common ``other_args``: ``instructions`` (default system prompt),
``token_store_path`` (alternate token file), ``reasoning`` (e.g.
``{"effort": "high"}``), ``originator`` (telemetry header, defaults to
``"andromeda"``), ``auto_login`` (force the automatic browser-based sign-in
on/off), plus anything ``ChatOpenAI`` accepts *except* ``api_key`` /
``base_url`` (both are managed by the OAuth layer and rejected upstream).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# --------------------------------------------------------------------------- #
# langchain-openai Codex import (optional extra)
# --------------------------------------------------------------------------- #
#
# ``_ChatOpenAICodex`` / ``chatgpt_oauth`` first shipped in langchain-openai
# 1.3.1 and are underscore-prefixed upstream (experimental). This module is the
# single place that touches those names, so a future upstream rename only has
# to be absorbed here.

try:
    from langchain_openai.chat_models.codex import (  # type: ignore
        CHATGPT_CODEX_BASE_URL,
        _ChatOpenAICodex,
    )
    from langchain_openai.chatgpt_oauth import (  # type: ignore
        DEFAULT_STORE_PATH,
        _FileChatGPTOAuthTokenProvider,
        login_chatgpt,
    )
except ImportError as exc:  # pragma: no cover - surfaced with guidance
    raise ImportError(
        "The 'openai_codex' provider requires langchain-openai>=1.3.1. Install "
        "it with: pip install 'andromeda[openai-codex]', then sign in once with "
        "andromeda.utils.openai_codex.login_device()."
    ) from exc

_ORIGINATOR_ENV_VAR = "LANGCHAIN_CODEX_ORIGINATOR"
_ANDROMEDA_ORIGINATOR = "andromeda"
_AUTO_LOGIN_ENV_VAR = "ANDROMEDA_CODEX_AUTO_LOGIN"


def _default_originator() -> str:
    """Andromeda's ``originator`` header default; the env var still wins."""
    return os.environ.get(_ORIGINATOR_ENV_VAR) or _ANDROMEDA_ORIGINATOR


def _resolve_auto_login(explicit: Optional[bool] = None) -> bool:
    """Decide whether to fall back to the browser-based sign-in flow when signed out.

    Priority: explicit override (``other_args["auto_login"]``) > the
    ``ANDROMEDA_CODEX_AUTO_LOGIN`` env var > interactive-terminal autodetect.
    Defaults to on only when stdin/stdout are a TTY, so the blocking browser
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




# --------------------------------------------------------------------------- #
# Chat model
# --------------------------------------------------------------------------- #


class ChatOpenAICodex(_ChatOpenAICodex):
    """ChatGPT-subscription-authed ``ChatOpenAI`` over the Codex backend.

    Auth comes from the OAuth token store written by :func:`login_device`
    (default: ``~/.langchain/chatgpt-auth.json``); tokens are refreshed
    automatically. Construct via ``get_chat_model`` with
    ``ModelConfig(provider="openai_codex", name="gpt-5.5")`` — the name must
    be a Codex slug from :func:`list_models`.

    Additions over the upstream class:

    - **Automatic sign-in**: if no token is found on disk, runs
      :func:`login_device` for you instead of raising ``FileNotFoundError`` on
      first request. See ``auto_login`` below to control this.
    - ``token_store_path``: config-friendly alternative to passing a live
      ``token_provider`` object (usable from YAML/``other_args``).
    - ``auto_login``: force the automatic sign-in on/off. Resolution order:
      explicit kwarg > ``ANDROMEDA_CODEX_AUTO_LOGIN`` env var
      (``1``/``true``/``yes``/``on`` to force on) > interactive-terminal
      autodetect (on only when stdin *and* stdout are a TTY, so this never
      blocks CI/daemons/tests waiting on a browser sign-in nobody will complete).
    - The ``originator`` telemetry header defaults to ``"andromeda"`` instead
      of ``"langchain"`` (``LANGCHAIN_CODEX_ORIGINATOR`` still overrides).
    """

    def __init__(self, **kwargs: Any) -> None:
        explicit_provider = kwargs.get("token_provider")
        store_path = kwargs.pop("token_store_path", None)
        path = Path(store_path) if store_path is not None else DEFAULT_STORE_PATH
        if store_path is not None and explicit_provider is None:
            kwargs["token_provider"] = _FileChatGPTOAuthTokenProvider(path=path)

        auto_login = _resolve_auto_login(kwargs.pop("auto_login", None))
        # Only auto-trigger for the default file-backed provider: a caller who
        # hands in their own token_provider is managing auth themselves.
        if explicit_provider is None and auto_login and not logged_in(path):
            login_device(store_path=path)

        kwargs.setdefault("originator", _default_originator())
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "openai-codex-subscription"


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #


def login_device(
    store_path: Optional[Path] = None, **kwargs: Any
) -> _FileChatGPTOAuthTokenProvider:
    """Sign in via the ChatGPT browser-based OAuth Authorization Code + PKCE flow.

    Binds a local loopback callback server (default ``localhost:1455``) and
    opens your system browser to the OpenAI sign-in page (the URL is also
    printed as a fallback if the browser can't be launched, e.g. no display).
    Blocks for up to ``timeout`` seconds (default 300) until the callback is
    received, then persists the token bundle to ``store_path`` (default
    ``~/.langchain/chatgpt-auth.json``) — which :class:`ChatOpenAICodex` reads
    from automatically, so subsequent constructions need nothing further.

    Accepts the same keyword arguments as upstream ``login_chatgpt``
    """
    if store_path is not None:
        kwargs["store_path"] = Path(store_path)
    return login_chatgpt(**kwargs)


def logged_in(store_path: Optional[Path] = None) -> bool:
    """Whether a ChatGPT OAuth token bundle exists on disk.

    Only checks presence — an expired token still counts, since the model
    refreshes it transparently on first use (a *revoked* refresh token is the
    one case that needs a fresh :func:`login_device`).
    """
    return Path(store_path or DEFAULT_STORE_PATH).is_file()


def auth_status(store_path: Optional[Path] = None) -> Dict[str, Any]:
    """Best-effort snapshot of the stored ChatGPT sign-in, for diagnostics.

    Returns a dict with ``logged_in``, ``store_path`` and — when the token file
    is readable — ``account_id``, ``plan_type``, ``expires_at`` and
    ``expired``. Never raises for a missing/corrupt store; a parse problem is
    reported under ``error``.
    """
    path = Path(store_path or DEFAULT_STORE_PATH)
    status: Dict[str, Any] = {"logged_in": False, "store_path": str(path)}
    if not path.is_file():
        return status
    status["logged_in"] = True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        status["account_id"] = data.get("account_id")
        status["plan_type"] = data.get("plan_type")
        expires_raw = data.get("expires_at")
        if isinstance(expires_raw, str):
            expires_at = datetime.fromisoformat(expires_raw)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            status["expires_at"] = expires_at.isoformat()
            status["expired"] = datetime.now(timezone.utc) >= expires_at
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        status["error"] = f"Token store unreadable: {exc}"
    return status


# --------------------------------------------------------------------------- #
# Model discovery
# --------------------------------------------------------------------------- #

CODEX_MODELS_CLIENT_VERSION = "1.0.0"
"""``client_version`` sent to the Codex ``/models`` endpoint.

The backend gates the response on this query param: versions below 1.0.0 get
an empty ``models`` list. Bump if the server starts requiring something newer.
"""


def _auth_headers(token: Any) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token.access_token}",
        "originator": _default_originator(),
        "Accept": "application/json",
    }
    if token.account_id:
        headers["ChatGPT-Account-Id"] = token.account_id
    return headers


def list_models(
    store_path: Optional[Path] = None,
    *,
    client_version: str = CODEX_MODELS_CLIENT_VERSION,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Return the raw Codex model list for the signed-in ChatGPT account.

    Each entry is the backend's model dict — notable keys: ``slug`` (the value
    to use as ``ModelConfig.name``), ``display_name``, ``description``,
    ``context_window``, ``supported_reasoning_levels``, and
    ``available_in_plans`` (compare against your plan from :func:`auth_status`;
    the endpoint returns the catalog, not just your entitlement). Requires a
    prior :func:`login_device`; the token is refreshed automatically.
    """
    provider = (
        _FileChatGPTOAuthTokenProvider(path=Path(store_path))
        if store_path
        else _FileChatGPTOAuthTokenProvider()
    )
    token = provider.get_token()
    response = httpx.get(
        f"{CHATGPT_CODEX_BASE_URL}/models",
        params={"client_version": client_version},
        headers=_auth_headers(token),
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Codex model listing failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    return list(response.json().get("models", []))


def print_models(store_path: Optional[Path] = None, *, show_hidden: bool = False) -> None:
    """Print a readable table of Codex models and your plan's access to each.

    Models with ``visibility: "hide"`` (internal/UI-only entries like
    ``codex-auto-review``) are skipped unless ``show_hidden=True``.
    """
    provider = (
        _FileChatGPTOAuthTokenProvider(path=Path(store_path))
        if store_path
        else _FileChatGPTOAuthTokenProvider()
    )
    plan = provider.get_token().plan_type
    models = list_models(store_path)
    print(f"ChatGPT plan: {plan or 'unknown'}\n")
    header = f"{'MODEL (slug)':<22} {'PLAN OK':<8} {'CONTEXT':<9} {'EFFORTS':<25} DESCRIPTION"
    print(header)
    print("-" * len(header))
    for model in models:
        if model.get("visibility") == "hide" and not show_hidden:
            continue
        slug = model.get("slug", "?")
        plans = model.get("available_in_plans") or []
        plan_ok = "yes" if (plan and plan in plans) else ("?" if not plans else "no")
        context = str(model.get("context_window") or "?")
        efforts = ",".join(
            level.get("effort", "?")
            for level in model.get("supported_reasoning_levels") or []
        )
        description = (model.get("description") or "")[:60]
        print(f"{slug:<22} {plan_ok:<8} {context:<9} {efforts:<25} {description}")


__all__ = [
    "CHATGPT_CODEX_BASE_URL",
    "DEFAULT_STORE_PATH",
    "ChatOpenAICodex",
    "auth_status",
    "list_models",
    "logged_in",
    "login_device",
    "print_models",
]
