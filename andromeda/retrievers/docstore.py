from __future__ import annotations

import asyncio
import json
from functools import lru_cache
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

from andromeda.retrievers.core import Document, DocumentStoreBackend

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine
else:
    Engine = Any
    AsyncEngine = Any


def _sqlalchemy():
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise ImportError(
            "Postgres-backed retriever storage requires optional dependencies. "
            "Install with `pip install \"andromeda[retrievers-postgres]\"`."
        ) from exc
    return create_engine, text


def _async_sqlalchemy():
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise ImportError(
            "Postgres-backed retriever storage requires optional dependencies. "
            "Install with `pip install \"andromeda[retrievers-postgres]\"`."
        ) from exc
    return create_async_engine, text


class InMemoryDocumentStore(DocumentStoreBackend):
    def __init__(self):
        self._docs: Dict[str, Document] = {}

    def upsert_documents(self, docs: List[Document]) -> None:
        for doc in docs:
            self._docs[doc.id] = doc

    def delete_documents(self, ids: List[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)

    def get_documents(self, ids: List[str]) -> Dict[str, Document]:
        return {doc_id: self._docs[doc_id] for doc_id in ids if doc_id in self._docs}


@dataclass(frozen=True)
class SqliteDocumentStoreConfig:
    path: str
    table: str = "documents"


class SqliteDocumentStore(DocumentStoreBackend):
    """
    Minimal sqlite-backed docstore.

    Stores full chunk text + metadata keyed by chunk id. Intended to support GraphRAG
    expansions that start from KG edges (chunk ids) instead of relying on short snippets.
    """

    def __init__(self, config: SqliteDocumentStoreConfig):
        self._path = config.path
        self._table = config.table
        # This docstore can be shared by a long-lived service that may handle requests
        # across multiple threads (e.g., FastAPI / tool servers).
        # sqlite3 defaults to check_same_thread=True which raises if the connection is
        # used from a different thread than the one that created it.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    def upsert_documents(self, docs: List[Document]) -> None:
        rows = [
            (d.id, d.text, json.dumps(d.metadata or {}, ensure_ascii=False))
            for d in docs
        ]
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    f"INSERT OR REPLACE INTO {self._table} (id, text, metadata_json) VALUES (?, ?, ?)",
                    rows,
                )

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        unique_ids = list(dict.fromkeys(ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    f"DELETE FROM {self._table} WHERE id IN ({placeholders})",
                    unique_ids,
                )

    def get_documents(self, ids: List[str]) -> Dict[str, Document]:
        if not ids:
            return {}
        unique_ids = list(dict.fromkeys(ids))
        placeholders = ",".join("?" for _ in unique_ids)
        with self._lock:
            cur = self._conn.execute(
                f"SELECT id, text, metadata_json FROM {self._table} WHERE id IN ({placeholders})",
                unique_ids,
            )
        out: Dict[str, Document] = {}
        for doc_id, text, meta_json in cur.fetchall():
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                meta = {}
            out[str(doc_id)] = Document(id=str(doc_id), text=str(text), metadata=dict(meta))
        return out


@dataclass(frozen=True)
class PostgresDocumentStoreConfig:
    connection_string: str
    namespace: str
    chunks_table: str = "rag_corpus_chunks"


@lru_cache(maxsize=8)
def _postgres_engine(connection_string: str) -> Engine:
    create_engine, _ = _sqlalchemy()
    return create_engine(connection_string, pool_pre_ping=True, future=True)


@lru_cache(maxsize=8)
def _postgres_async_engine(connection_string: str) -> AsyncEngine:
    create_async_engine, _ = _async_sqlalchemy()
    return create_async_engine(connection_string, pool_pre_ping=True, future=True)


class PostgresDocumentStore(DocumentStoreBackend):
    """
    Postgres-backed docstore keyed by corpus namespace and chunk id.

    This stores full chunk text + metadata for GraphRAG expansion and lexical restore.
    It intentionally shares the same `rag_corpus_chunks` table used by the ingestion
    index so state stays centralized in Postgres rather than per-corpus sqlite files.
    """

    def __init__(self, config: PostgresDocumentStoreConfig):
        self._namespace = config.namespace
        self._chunks_table = config.chunks_table
        self._engine = _postgres_engine(config.connection_string)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._chunks_table} (
                        namespace TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        chunk_hash TEXT,
                        text_content TEXT,
                        metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (namespace, chunk_id)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self._chunks_table}_namespace_source
                    ON {self._chunks_table} (namespace, source_id)
                    """
                )
            )

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass

    def upsert_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        _, text = _sqlalchemy()
        rows = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            source_id = str(metadata.get("source_id") or doc.id.split("::")[0])
            rows.append(
                {
                    "namespace": self._namespace,
                    "chunk_id": doc.id,
                    "source_id": source_id,
                    "text_content": doc.text,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                }
            )

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self._chunks_table} (
                        namespace,
                        chunk_id,
                        source_id,
                        text_content,
                        metadata_json,
                        updated_at
                    ) VALUES (
                        :namespace,
                        :chunk_id,
                        :source_id,
                        :text_content,
                        CAST(:metadata_json AS JSONB),
                        NOW()
                    )
                    ON CONFLICT (namespace, chunk_id) DO UPDATE
                    SET
                        source_id = EXCLUDED.source_id,
                        text_content = EXCLUDED.text_content,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = NOW()
                    """
                ),
                rows,
            )

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        _, text = _sqlalchemy()
        unique_ids = list(dict.fromkeys(ids))
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {self._chunks_table}
                    WHERE namespace = :namespace
                      AND chunk_id = ANY(:chunk_ids)
                    """
                ),
                {"namespace": self._namespace, "chunk_ids": unique_ids},
            )

    def get_documents(self, ids: List[str]) -> Dict[str, Document]:
        if not ids:
            return {}
        _, text = _sqlalchemy()
        unique_ids = list(dict.fromkeys(ids))
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT chunk_id, text_content, metadata_json
                    FROM {self._chunks_table}
                    WHERE namespace = :namespace
                      AND chunk_id = ANY(:chunk_ids)
                    """
                ),
                {"namespace": self._namespace, "chunk_ids": unique_ids},
            ).fetchall()

        out: Dict[str, Document] = {}
        for chunk_id, text_content, metadata_json in rows:
            meta = metadata_json if isinstance(metadata_json, dict) else {}
            out[str(chunk_id)] = Document(
                id=str(chunk_id),
                text=str(text_content or ""),
                metadata=dict(meta),
            )
        return out


class AsyncPostgresDocumentStore:
    """Async Postgres-backed docstore keyed by corpus namespace and chunk id."""

    def __init__(self, config: PostgresDocumentStoreConfig):
        self._namespace = config.namespace
        self._chunks_table = config.chunks_table
        self._engine = _postgres_async_engine(config.connection_string)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            _, text = _async_sqlalchemy()
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self._chunks_table} (
                            namespace TEXT NOT NULL,
                            chunk_id TEXT NOT NULL,
                            source_id TEXT NOT NULL,
                            chunk_hash TEXT,
                            text_content TEXT,
                            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (namespace, chunk_id)
                        )
                        """
                    )
                )
                await conn.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{self._chunks_table}_namespace_source
                        ON {self._chunks_table} (namespace, source_id)
                        """
                    )
                )
            self._initialized = True

    async def aupsert_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        rows = []
        for doc in docs:
            metadata = dict(doc.metadata or {})
            source_id = str(metadata.get("source_id") or doc.id.split("::")[0])
            rows.append(
                {
                    "namespace": self._namespace,
                    "chunk_id": doc.id,
                    "source_id": source_id,
                    "text_content": doc.text,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                }
            )

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._chunks_table} (
                        namespace,
                        chunk_id,
                        source_id,
                        text_content,
                        metadata_json,
                        updated_at
                    ) VALUES (
                        :namespace,
                        :chunk_id,
                        :source_id,
                        :text_content,
                        CAST(:metadata_json AS JSONB),
                        NOW()
                    )
                    ON CONFLICT (namespace, chunk_id) DO UPDATE
                    SET
                        source_id = EXCLUDED.source_id,
                        text_content = EXCLUDED.text_content,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = NOW()
                    """
                ),
                rows,
            )

    async def adelete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        unique_ids = list(dict.fromkeys(ids))
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._chunks_table}
                    WHERE namespace = :namespace
                      AND chunk_id = ANY(:chunk_ids)
                    """
                ),
                {"namespace": self._namespace, "chunk_ids": unique_ids},
            )

    async def aget_documents(self, ids: List[str]) -> Dict[str, Document]:
        if not ids:
            return {}
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        unique_ids = list(dict.fromkeys(ids))
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT chunk_id, text_content, metadata_json
                        FROM {self._chunks_table}
                        WHERE namespace = :namespace
                          AND chunk_id = ANY(:chunk_ids)
                        """
                    ),
                    {"namespace": self._namespace, "chunk_ids": unique_ids},
                )
            ).fetchall()

        out: Dict[str, Document] = {}
        for chunk_id, text_content, metadata_json in rows:
            meta = metadata_json if isinstance(metadata_json, dict) else {}
            out[str(chunk_id)] = Document(
                id=str(chunk_id),
                text=str(text_content or ""),
                metadata=dict(meta),
            )
        return out
