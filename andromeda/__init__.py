"""Top-level package exports and shared warning hygiene."""

import warnings

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except Exception:  # pragma: no cover - fallback for environments without this import path
    LangChainPendingDeprecationWarning = DeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
)

try:
    from langchain.messages import HumanMessage, AIMessage, SystemMessage
except ImportError:
    HumanMessage = None
    AIMessage = None
    SystemMessage = None

try:
    from langchain_core.messages import BaseMessage
except ImportError:
    BaseMessage = None

if all(x is not None for x in (HumanMessage, AIMessage, SystemMessage, BaseMessage)):
    __all__ = ["HumanMessage", "AIMessage", "SystemMessage", "BaseMessage"]
else:
    __all__ = []
