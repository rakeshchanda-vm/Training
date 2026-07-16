from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProviderAvailability:
    available: bool
    provider: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


KNOWN_PROVIDERS = (
    "local_fs",
    "ephemeral_fs",
    "postgres_vfs",
    "bubblewrap_process",
    "gvisor_container",
    "microvm",
)


def check_provider_availability(provider: str) -> ProviderAvailability:
    if provider == "bubblewrap_process":
        from andromeda.workspace.sandbox_providers import BubblewrapProcessProvider

        return BubblewrapProcessProvider.check_available()
    if provider == "gvisor_container":
        from andromeda.workspace.sandbox_providers import GVisorContainerProvider

        return GVisorContainerProvider.check_available()
    if provider in {"local_fs", "ephemeral_fs"}:
        return ProviderAvailability(available=True, provider=provider)
    if provider == "postgres_vfs":
        return ProviderAvailability(
            available=True,
            provider=provider,
            reason="Requires PostgresVFSSettings at session creation.",
        )
    if provider == "microvm":
        return ProviderAvailability(
            available=False,
            provider=provider,
            reason="Requires NerdctlDevSettings or ContainerdKataSettings and host/runtime deps.",
        )
    return ProviderAvailability(
        available=False,
        provider=provider,
        reason=f"Unknown provider: {provider}",
    )


def check_all_providers() -> dict[str, ProviderAvailability]:
    return {name: check_provider_availability(name) for name in KNOWN_PROVIDERS}
