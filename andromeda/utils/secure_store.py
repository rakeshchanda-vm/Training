from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from cryptography.fernet import Fernet


def _derive_fernet_key() -> bytes:
    """
    Derive/load a Fernet key from environment.

    Supported env vars:
    - ANDROMEDA_ENCRYPTION_KEY: Fernet key (urlsafe base64-encoded 32-byte key)
    - ANDROMEDA_ENCRYPTION_SECRET: passphrase used to derive a Fernet key via SHA-256
    """
    raw_key = os.getenv("ANDROMEDA_ENCRYPTION_KEY")
    if raw_key:
        return raw_key.encode("utf-8")

    secret = os.getenv("ANDROMEDA_ENCRYPTION_SECRET")
    if not secret:
        raise ValueError(
            "Missing encryption secret for secure token store. Set either "
            "ANDROMEDA_ENCRYPTION_KEY or ANDROMEDA_ENCRYPTION_SECRET."
        )

    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


@dataclass(frozen=True)
class TokenRecord:
    ciphertext: bytes
    expires_at: Optional[float]


class InMemoryEncryptedTokenStore:
    """Short-token, in-memory encrypted storage for sensitive value recovery."""

    def __init__(
        self,
        *,
        token_prefix: str = "pii",
        ttl_seconds: Optional[int] = 24 * 60 * 60,
        max_entries: int = 50_000,
    ) -> None:
        self._fernet = Fernet(_derive_fernet_key())
        self._token_prefix = token_prefix
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._records: Dict[str, TokenRecord] = {}
        self._lock = threading.RLock()

    def _new_token(self) -> str:
        # 11 chars from token_urlsafe(8) + prefix keeps tokens short and URL-safe.
        short = secrets.token_urlsafe(8).rstrip("=")
        return f"{self._token_prefix}_{short}"

    def _evict_if_needed(self) -> None:
        if len(self._records) < self._max_entries:
            return

        self.purge_expired()
        if len(self._records) < self._max_entries:
            return

        # Best-effort bounded memory: drop oldest arbitrary entries.
        remove_count = max(1, len(self._records) // 10)
        for token in list(self._records.keys())[:remove_count]:
            self._records.pop(token, None)

    def put(self, value: str) -> str:
        now = time.time()
        expires_at = (now + self._ttl_seconds) if self._ttl_seconds else None
        payload = value.encode("utf-8")
        encrypted = self._fernet.encrypt(payload)

        with self._lock:
            self._evict_if_needed()
            token = self._new_token()
            while token in self._records:
                token = self._new_token()
            self._records[token] = TokenRecord(ciphertext=encrypted, expires_at=expires_at)
            return token

    def get(self, token: str) -> Optional[str]:
        with self._lock:
            record = self._records.get(token)
            if not record:
                return None

            if record.expires_at is not None and record.expires_at < time.time():
                self._records.pop(token, None)
                return None

            decrypted = self._fernet.decrypt(record.ciphertext)
            return decrypted.decode("utf-8")

    def delete(self, token: str) -> None:
        with self._lock:
            self._records.pop(token, None)

    def purge_expired(self) -> int:
        with self._lock:
            now = time.time()
            expired = [
                token
                for token, rec in self._records.items()
                if rec.expires_at is not None and rec.expires_at < now
            ]
            for token in expired:
                self._records.pop(token, None)
            return len(expired)


_STORES: Dict[str, InMemoryEncryptedTokenStore] = {}
_STORE_LOCK = threading.Lock()


def get_secure_store(
    *,
    token_prefix: str = "pii",
    ttl_seconds: Optional[int] = 24 * 60 * 60,
) -> InMemoryEncryptedTokenStore:
    store = _STORES.get(token_prefix)
    if store is not None:
        return store

    with _STORE_LOCK:
        store = _STORES.get(token_prefix)
        if store is None:
            store = InMemoryEncryptedTokenStore(
                token_prefix=token_prefix,
                ttl_seconds=ttl_seconds,
            )
            _STORES[token_prefix] = store
        return store


def detokenize_value(token: str) -> Optional[str]:
    """Resolve a secure token back to plaintext using the matching prefix store."""
    prefix = token.split("_", 1)[0] if "_" in token else "pii"
    store = _STORES.get(prefix)
    if store is None:
        return None
    return store.get(token)
