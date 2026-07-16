from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from andromeda.retrievers.sqlalchemy_url import sqlalchemy_async_postgres_url

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
            "Postgres-backed retriever ingestion state requires optional dependencies. "
            "Install with `pip install \"andromeda[retrievers-postgres]\"`."
        ) from exc
    return create_engine, text


def _async_sqlalchemy():
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise ImportError(
            "Postgres-backed retriever ingestion state requires optional dependencies. "
            "Install with `pip install \"andromeda[retrievers-postgres]\"`."
        ) from exc
    return create_async_engine, text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceState:
    source_id: str
    doc_hash: str
    chunks: Dict[str, str]  # chunk_id -> chunk_hash


class SqliteIngestionIndex:
    """
    Tracks per-source ingest state so re-ingestion can do add/update/delete.

    This is intentionally minimal and stores only hashes and ids, not full text.
    """

    def __init__(self, path: str):
        self._path = path
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA journal_mode=WAL;")
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT PRIMARY KEY,
                    doc_hash TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id)"
            )
            self._conn.commit()

    def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    def get_source_state(self, source_id: str) -> Optional[SourceState]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT doc_hash FROM documents WHERE source_id = ?",
                (source_id,),
            )
        row = cur.fetchone()
        if not row:
            return None
        doc_hash = str(row[0])
        with self._lock:
            cur = self._conn.execute(
                "SELECT chunk_id, chunk_hash FROM chunks WHERE source_id = ?",
                (source_id,),
            )
        chunks = {str(cid): str(ch) for (cid, ch) in cur.fetchall()}
        return SourceState(source_id=source_id, doc_hash=doc_hash, chunks=chunks)

    def list_sources(self) -> List[str]:
        with self._lock:
            cur = self._conn.execute("SELECT source_id FROM documents")
            return [str(r[0]) for r in cur.fetchall()]

    def list_chunk_ids(self, source_id: Optional[str] = None) -> List[str]:
        with self._lock:
            if source_id:
                cur = self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE source_id = ?",
                    (source_id,),
                )
            else:
                cur = self._conn.execute("SELECT chunk_id FROM chunks")
            return [str(r[0]) for r in cur.fetchall()]

    def delete_source(self, source_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
                self._conn.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))

    def upsert_source(self, source_id: str, doc_hash: str, chunks: Dict[str, str]) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO documents(source_id, doc_hash) VALUES(?, ?)",
                    (source_id, doc_hash),
                )
                self._conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
                self._conn.executemany(
                    "INSERT OR REPLACE INTO chunks(chunk_id, source_id, chunk_hash) VALUES(?, ?, ?)",
                    [(cid, source_id, chash) for cid, chash in chunks.items()],
                )


@dataclass(frozen=True)
class PostgresIngestionIndexConfig:
    connection_string: str
    namespace: str
    sources_table: str = "rag_corpus_sources"
    chunks_table: str = "rag_corpus_chunks"


@lru_cache(maxsize=8)
def _postgres_engine(connection_string: str) -> Engine:
    create_engine, _ = _sqlalchemy()
    return create_engine(connection_string, pool_pre_ping=True, future=True)


@lru_cache(maxsize=8)
def _postgres_async_engine(connection_string: str) -> AsyncEngine:
    create_async_engine, _ = _async_sqlalchemy()
    async_url = sqlalchemy_async_postgres_url(connection_string)
    return create_async_engine(async_url, pool_pre_ping=True, future=True)


class PostgresIngestionIndex:
    """
    Tracks per-source ingest state in Postgres so re-ingestion can do add/update/delete.

    Storage is shared across corpora using a logical namespace rather than per-corpus files.
    """

    def __init__(self, config: PostgresIngestionIndexConfig):
        self._namespace = config.namespace
        self._sources_table = config.sources_table
        self._chunks_table = config.chunks_table
        self._engine = _postgres_engine(config.connection_string)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._sources_table} (
                        namespace TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        doc_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (namespace, source_id)
                    )
                    """
                )
            )
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

    def get_source_state(self, source_id: str) -> Optional[SourceState]:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            row = conn.execute(
                text(
                    f"""
                    SELECT doc_hash
                    FROM {self._sources_table}
                    WHERE namespace = :namespace
                      AND source_id = :source_id
                    """
                ),
                {"namespace": self._namespace, "source_id": source_id},
            ).fetchone()
            if not row:
                return None
            chunk_rows = conn.execute(
                text(
                    f"""
                    SELECT chunk_id, chunk_hash
                    FROM {self._chunks_table}
                    WHERE namespace = :namespace
                      AND source_id = :source_id
                    """
                ),
                {"namespace": self._namespace, "source_id": source_id},
            ).fetchall()

        chunks = {str(cid): str(ch or "") for cid, ch in chunk_rows}
        return SourceState(source_id=source_id, doc_hash=str(row[0]), chunks=chunks)

    def list_sources(self) -> List[str]:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT source_id
                    FROM {self._sources_table}
                    WHERE namespace = :namespace
                    """
                ),
                {"namespace": self._namespace},
            ).fetchall()
        return [str(r[0]) for r in rows]

    def list_chunk_ids(self, source_id: Optional[str] = None) -> List[str]:
        _, text = _sqlalchemy()
        params = {"namespace": self._namespace}
        query = (
            f"""
            SELECT chunk_id
            FROM {self._chunks_table}
            WHERE namespace = :namespace
            """
        )
        if source_id:
            query += " AND source_id = :source_id"
            params["source_id"] = source_id
        with self._engine.begin() as conn:
            rows = conn.execute(text(query), params).fetchall()
        return [str(r[0]) for r in rows]

    def delete_source(self, source_id: str) -> None:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    DELETE FROM {self._chunks_table}
                    WHERE namespace = :namespace
                      AND source_id = :source_id
                    """
                ),
                {"namespace": self._namespace, "source_id": source_id},
            )
            conn.execute(
                text(
                    f"""
                    DELETE FROM {self._sources_table}
                    WHERE namespace = :namespace
                      AND source_id = :source_id
                    """
                ),
                {"namespace": self._namespace, "source_id": source_id},
            )

    def upsert_source(self, source_id: str, doc_hash: str, chunks: Dict[str, str]) -> None:
        _, text = _sqlalchemy()
        chunk_ids = list(chunks.keys())
        rows = [
            {
                "namespace": self._namespace,
                "chunk_id": chunk_id,
                "source_id": source_id,
                "chunk_hash": chunk_hash,
            }
            for chunk_id, chunk_hash in chunks.items()
        ]

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self._sources_table} (
                        namespace,
                        source_id,
                        doc_hash,
                        updated_at
                    ) VALUES (
                        :namespace,
                        :source_id,
                        :doc_hash,
                        NOW()
                    )
                    ON CONFLICT (namespace, source_id) DO UPDATE
                    SET
                        doc_hash = EXCLUDED.doc_hash,
                        updated_at = NOW()
                    """
                ),
                {
                    "namespace": self._namespace,
                    "source_id": source_id,
                    "doc_hash": doc_hash,
                },
            )

            if chunk_ids:
                conn.execute(
                    text(
                        f"""
                        DELETE FROM {self._chunks_table}
                        WHERE namespace = :namespace
                          AND source_id = :source_id
                          AND chunk_id <> ALL(:chunk_ids)
                        """
                    ),
                    {
                        "namespace": self._namespace,
                        "source_id": source_id,
                        "chunk_ids": chunk_ids,
                    },
                )
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._chunks_table} (
                            namespace,
                            chunk_id,
                            source_id,
                            chunk_hash,
                            updated_at
                        ) VALUES (
                            :namespace,
                            :chunk_id,
                            :source_id,
                            :chunk_hash,
                            NOW()
                        )
                        ON CONFLICT (namespace, chunk_id) DO UPDATE
                        SET
                            source_id = EXCLUDED.source_id,
                            chunk_hash = EXCLUDED.chunk_hash,
                            updated_at = NOW()
                        """
                    ),
                    rows,
                )
            else:
                conn.execute(
                    text(
                        f"""
                        DELETE FROM {self._chunks_table}
                        WHERE namespace = :namespace
                          AND source_id = :source_id
                        """
                    ),
                    {"namespace": self._namespace, "source_id": source_id},
                )


class AsyncPostgresIngestionIndex:
    """Async Postgres-backed ingest state using a logical namespace."""

    def __init__(self, config: PostgresIngestionIndexConfig):
        self._namespace = config.namespace
        self._sources_table = config.sources_table
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
                        CREATE TABLE IF NOT EXISTS {self._sources_table} (
                            namespace TEXT NOT NULL,
                            source_id TEXT NOT NULL,
                            doc_hash TEXT NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (namespace, source_id)
                        )
                        """
                    )
                )
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

    async def aget_source_state(self, source_id: str) -> Optional[SourceState]:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        async with self._engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        f"""
                        SELECT doc_hash
                        FROM {self._sources_table}
                        WHERE namespace = :namespace
                          AND source_id = :source_id
                        """
                    ),
                    {"namespace": self._namespace, "source_id": source_id},
                )
            ).fetchone()
            if not row:
                return None
            chunk_rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT chunk_id, chunk_hash
                        FROM {self._chunks_table}
                        WHERE namespace = :namespace
                          AND source_id = :source_id
                        """
                    ),
                    {"namespace": self._namespace, "source_id": source_id},
                )
            ).fetchall()

        chunks = {str(cid): str(ch or "") for cid, ch in chunk_rows}
        return SourceState(source_id=source_id, doc_hash=str(row[0]), chunks=chunks)

    async def alist_sources(self) -> List[str]:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT source_id
                        FROM {self._sources_table}
                        WHERE namespace = :namespace
                        """
                    ),
                    {"namespace": self._namespace},
                )
            ).fetchall()
        return [str(r[0]) for r in rows]

    async def alist_chunk_ids(self, source_id: Optional[str] = None) -> List[str]:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        params = {"namespace": self._namespace}
        query = (
            f"""
            SELECT chunk_id
            FROM {self._chunks_table}
            WHERE namespace = :namespace
            """
        )
        if source_id:
            query += " AND source_id = :source_id"
            params["source_id"] = source_id
        async with self._engine.begin() as conn:
            rows = (await conn.execute(text(query), params)).fetchall()
        return [str(r[0]) for r in rows]

    async def adelete_source(self, source_id: str) -> None:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._chunks_table}
                    WHERE namespace = :namespace
                      AND source_id = :source_id
                    """
                ),
                {"namespace": self._namespace, "source_id": source_id},
            )
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._sources_table}
                    WHERE namespace = :namespace
                      AND source_id = :source_id
                    """
                ),
                {"namespace": self._namespace, "source_id": source_id},
            )

    async def aupsert_source(self, source_id: str, doc_hash: str, chunks: Dict[str, str]) -> None:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        chunk_ids = list(chunks.keys())
        rows = [
            {
                "namespace": self._namespace,
                "chunk_id": chunk_id,
                "source_id": source_id,
                "chunk_hash": chunk_hash,
            }
            for chunk_id, chunk_hash in chunks.items()
        ]

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._sources_table} (
                        namespace,
                        source_id,
                        doc_hash,
                        updated_at
                    ) VALUES (
                        :namespace,
                        :source_id,
                        :doc_hash,
                        NOW()
                    )
                    ON CONFLICT (namespace, source_id) DO UPDATE
                    SET
                        doc_hash = EXCLUDED.doc_hash,
                        updated_at = NOW()
                    """
                ),
                {
                    "namespace": self._namespace,
                    "source_id": source_id,
                    "doc_hash": doc_hash,
                },
            )

            if chunk_ids:
                await conn.execute(
                    text(
                        f"""
                        DELETE FROM {self._chunks_table}
                        WHERE namespace = :namespace
                          AND source_id = :source_id
                          AND chunk_id <> ALL(:chunk_ids)
                        """
                    ),
                    {
                        "namespace": self._namespace,
                        "source_id": source_id,
                        "chunk_ids": chunk_ids,
                    },
                )
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._chunks_table} (
                            namespace,
                            chunk_id,
                            source_id,
                            chunk_hash,
                            updated_at
                        ) VALUES (
                            :namespace,
                            :chunk_id,
                            :source_id,
                            :chunk_hash,
                            NOW()
                        )
                        ON CONFLICT (namespace, chunk_id) DO UPDATE
                        SET
                            source_id = EXCLUDED.source_id,
                            chunk_hash = EXCLUDED.chunk_hash,
                            updated_at = NOW()
                        """
                    ),
                    rows,
                )
            else:
                await conn.execute(
                    text(
                        f"""
                        DELETE FROM {self._chunks_table}
                        WHERE namespace = :namespace
                          AND source_id = :source_id
                        """
                    ),
                    {"namespace": self._namespace, "source_id": source_id},
                )
