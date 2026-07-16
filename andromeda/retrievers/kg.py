from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypedDict, TYPE_CHECKING

from andromeda.retrievers.core import Document, KnowledgeGraphBackend, KnowledgeGraphFactRecord
from andromeda.config import ModelConfig

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from sqlalchemy.ext.asyncio import AsyncEngine
else:
    Engine = Any
    AsyncEngine = Any


@dataclass(frozen=True)
class KGFact:
    subject: str
    predicate: str
    object: str
    metadata: Dict[str, str]


@dataclass(frozen=True)
class KGExtraction:
    """
    Bundle of extraction outputs for a single chunk.

    For LLM-based extraction, returning entities + triples in one call reduces
    inconsistency between separately-generated lists.
    """

    entities: List[str]
    triples: List[Tuple[str, str, str]]


class InMemoryKnowledgeGraph(KnowledgeGraphBackend):
    """
    Minimal in-memory graph for demo and debugging.
    Stores directed edges (plus reverse lookup) and supports neighborhood traversal.
    """

    def __init__(self):
        self._adj: Dict[str, List[KGFact]] = defaultdict(list)
        self._rev: Dict[str, List[KGFact]] = defaultdict(list)

    def _norm(self, value: str) -> str:
        return value.strip().lower()

    def upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        meta = metadata or {}
        fact = KGFact(subject=subject, predicate=predicate, object=object_, metadata=meta)
        nsubj = self._norm(subject)
        nobj = self._norm(object_)

        if fact not in self._adj[nsubj]:
            self._adj[nsubj].append(fact)
        # Also track reverse adjacency for incoming edges.
        if fact not in self._rev[nobj]:
            self._rev[nobj].append(fact)

    def upsert_facts(self, facts: List[KnowledgeGraphFactRecord]) -> None:
        for fact in facts:
            self.upsert_fact(fact.subject, fact.predicate, fact.object, dict(fact.metadata or {}))

    def neighborhood(
        self, node_id: str, hops: int = 2, limit: int = 50
    ) -> List[Dict[str, str]]:
        visited: Set[str] = set()
        queue: deque[Tuple[str, int, str]] = deque([(self._norm(node_id), 0, node_id)])
        results: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        while queue and len(results) < limit:
            current_norm, depth, display_id = queue.popleft()
            if depth > hops or current_norm in visited:
                continue
            visited.add(current_norm)

            def _append_neighbors(facts: List[KGFact], outgoing: bool) -> None:
                nonlocal results
                for fact in facts:
                    target = fact.object if outgoing else fact.subject
                    edge_key = (display_id, fact.predicate, target)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    node = {
                        "id": target,
                        "predicate": fact.predicate if outgoing else f"inv:{fact.predicate}",
                        # Always point 'source' to the node we expanded from (useful for debugging).
                        "source": display_id,
                        "depth": depth + 1,
                        "direction": "out" if outgoing else "in",
                        **fact.metadata,
                    }
                    results.append(node)
                    queue.append((self._norm(target), depth + 1, target))

            _append_neighbors(self._adj.get(current_norm, []), outgoing=True)
            if len(results) >= limit:
                break
            _append_neighbors(self._rev.get(current_norm, []), outgoing=False)
        return results[:limit]

    def query(
        self, cypher_or_graph_query: str, params: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        # Placeholder: real implementations would dispatch to a graph DB.
        # For the demo, return a flat list of all facts.
        out: List[Dict[str, str]] = []
        for subject, facts in self._adj.items():
            for fact in facts:
                out.append(
                    {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        **fact.metadata,
                    }
                )
        return out

    def delete_facts_for_chunk(self, chunk_id: str) -> None:
        """
        Best-effort deletion for re-ingestion: removes facts whose subject is chunk_id
        (e.g. mentions edges) or whose metadata doc_id is chunk_id (extracted triples).
        """
        def _keep(f: KGFact) -> bool:
            if f.subject == chunk_id:
                return False
            if isinstance(f.metadata, dict) and f.metadata.get("doc_id") == chunk_id:
                return False
            return True

        for key in list(self._adj.keys()):
            facts = self._adj.get(key, [])
            kept = [f for f in facts if _keep(f)]
            if kept:
                self._adj[key] = kept
            else:
                self._adj.pop(key, None)

        for key in list(self._rev.keys()):
            facts = self._rev.get(key, [])
            kept = [f for f in facts if _keep(f)]
            if kept:
                self._rev[key] = kept
            else:
                self._rev.pop(key, None)

    def delete_facts_for_chunks(self, chunk_ids: List[str]) -> None:
        for chunk_id in chunk_ids:
            self.delete_facts_for_chunk(chunk_id)


@dataclass(frozen=True)
class PostgresKnowledgeGraphConfig:
    connection_string: str
    namespace: str
    table: str = "rag_corpus_kg_facts"


def _sqlalchemy():
    try:
        from sqlalchemy import create_engine, text
    except ImportError as exc:
        raise ImportError(
            "Postgres-backed retriever graph storage requires optional dependencies. "
            "Install with `pip install \"andromeda[retrievers-postgres]\"`."
        ) from exc
    return create_engine, text


def _async_sqlalchemy():
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise ImportError(
            "Postgres-backed retriever graph storage requires optional dependencies. "
            "Install with `pip install \"andromeda[retrievers-postgres]\"`."
        ) from exc
    return create_async_engine, text


@lru_cache(maxsize=8)
def _postgres_engine(connection_string: str) -> Engine:
    create_engine, _ = _sqlalchemy()
    return create_engine(connection_string, pool_pre_ping=True, future=True)


@lru_cache(maxsize=8)
def _postgres_async_engine(connection_string: str) -> AsyncEngine:
    create_async_engine, _ = _async_sqlalchemy()
    return create_async_engine(connection_string, pool_pre_ping=True, future=True)


class PostgresKnowledgeGraph(KnowledgeGraphBackend):
    def __init__(self, config: PostgresKnowledgeGraphConfig):
        self._namespace = config.namespace
        self._table = config.table
        self._engine = _postgres_engine(config.connection_string)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        namespace TEXT NOT NULL,
                        fact_key TEXT NOT NULL,
                        source_id TEXT,
                        doc_id TEXT,
                        subject TEXT NOT NULL,
                        subject_norm TEXT NOT NULL,
                        predicate TEXT NOT NULL,
                        object TEXT NOT NULL,
                        object_norm TEXT NOT NULL,
                        snippet TEXT,
                        metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (namespace, fact_key)
                    )
                    """
                )
            )
            for index_name, columns in (
                (f"idx_{self._table}_namespace_source", "(namespace, source_id)"),
                (f"idx_{self._table}_namespace_doc", "(namespace, doc_id)"),
                (f"idx_{self._table}_namespace_subject_norm", "(namespace, subject_norm)"),
                (f"idx_{self._table}_namespace_object_norm", "(namespace, object_norm)"),
            ):
                conn.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self._table} {columns}
                        """
                    )
                )

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower()

    def _fact_key(self, subject: str, predicate: str, object_: str, metadata: Dict[str, Any]) -> str:
        source_id = str(metadata.get("source_id") or "")
        doc_id = str(metadata.get("doc_id") or "")
        raw = "\x1f".join(
            [
                self._namespace,
                self._norm(subject),
                predicate.strip().lower(),
                self._norm(object_),
                source_id,
                doc_id,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        self.upsert_facts(
            [
                KnowledgeGraphFactRecord(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    metadata=dict(metadata or {}),
                )
            ]
        )

    def upsert_facts(self, facts: List[KnowledgeGraphFactRecord]) -> None:
        if not facts:
            return
        _, text = _sqlalchemy()
        rows = []
        for fact in facts:
            metadata = dict(fact.metadata or {})
            rows.append(
                {
                    "namespace": self._namespace,
                    "fact_key": self._fact_key(fact.subject, fact.predicate, fact.object, metadata),
                    "source_id": str(metadata.get("source_id") or "") or None,
                    "doc_id": str(metadata.get("doc_id") or "") or None,
                    "subject": fact.subject,
                    "subject_norm": self._norm(fact.subject),
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "object_norm": self._norm(fact.object),
                    "snippet": str(metadata.get("snippet") or "") or None,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                }
            )
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table} (
                        namespace,
                        fact_key,
                        source_id,
                        doc_id,
                        subject,
                        subject_norm,
                        predicate,
                        object,
                        object_norm,
                        snippet,
                        metadata_json
                    )
                    VALUES (
                        :namespace,
                        :fact_key,
                        :source_id,
                        :doc_id,
                        :subject,
                        :subject_norm,
                        :predicate,
                        :object,
                        :object_norm,
                        :snippet,
                        CAST(:metadata_json AS JSONB)
                    )
                    ON CONFLICT (namespace, fact_key) DO UPDATE SET
                        source_id = EXCLUDED.source_id,
                        doc_id = EXCLUDED.doc_id,
                        subject = EXCLUDED.subject,
                        subject_norm = EXCLUDED.subject_norm,
                        predicate = EXCLUDED.predicate,
                        object = EXCLUDED.object,
                        object_norm = EXCLUDED.object_norm,
                        snippet = EXCLUDED.snippet,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = NOW()
                    """
                ),
                rows,
            )

    def _delete_for_chunks(self, conn, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        _, text = _sqlalchemy()
        conn.execute(
            text(
                f"""
                DELETE FROM {self._table}
                WHERE namespace = :namespace
                  AND (doc_id = ANY(:chunk_ids) OR subject = ANY(:chunk_ids))
                """
            ),
            {"namespace": self._namespace, "chunk_ids": chunk_ids},
        )

    def delete_facts_for_chunk(self, chunk_id: str) -> None:
        self.delete_facts_for_chunks([chunk_id])

    def delete_facts_for_chunks(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        with self._engine.begin() as conn:
            self._delete_for_chunks(conn, list(dict.fromkeys(chunk_ids)))

    def _neighbor_rows(self, node_norm: str) -> List[Dict[str, Any]]:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT subject, predicate, object, metadata_json
                    FROM {self._table}
                    WHERE namespace = :namespace
                      AND (subject_norm = :node_norm OR object_norm = :node_norm)
                    """
                ),
                {"namespace": self._namespace, "node_norm": node_norm},
            ).mappings()
            return [dict(row) for row in rows]

    def neighborhood(
        self, node_id: str, hops: int = 2, limit: int = 50
    ) -> List[Dict[str, str]]:
        visited: Set[str] = set()
        queue: deque[Tuple[str, int, str]] = deque([(self._norm(node_id), 0, node_id)])
        results: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        while queue and len(results) < limit:
            current_norm, depth, display_id = queue.popleft()
            if depth > hops or current_norm in visited:
                continue
            visited.add(current_norm)

            for row in self._neighbor_rows(current_norm):
                metadata_raw = row.get("metadata_json")
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                    except Exception:
                        metadata = {}
                else:
                    metadata = dict(metadata_raw or {})

                subject = str(row.get("subject") or "")
                predicate = str(row.get("predicate") or "")
                object_ = str(row.get("object") or "")
                if not subject or not predicate or not object_:
                    continue

                outgoing = self._norm(subject) == current_norm
                target = object_ if outgoing else subject
                edge_key = (display_id, predicate if outgoing else f"inv:{predicate}", target)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                results.append(
                    {
                        "id": target,
                        "predicate": predicate if outgoing else f"inv:{predicate}",
                        "source": display_id,
                        "depth": depth + 1,
                        "direction": "out" if outgoing else "in",
                        **metadata,
                    }
                )
                queue.append((self._norm(target), depth + 1, target))
                if len(results) >= limit:
                    break

        return results[:limit]

    def query(
        self, cypher_or_graph_query: str, params: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        _, text = _sqlalchemy()
        with self._engine.begin() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT subject, predicate, object, source_id, doc_id, snippet, metadata_json
                    FROM {self._table}
                    WHERE namespace = :namespace
                    ORDER BY updated_at ASC, created_at ASC
                    """
                ),
                {"namespace": self._namespace},
            ).mappings()

            out: List[Dict[str, str]] = []
            for row in rows:
                metadata_raw = row.get("metadata_json")
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                    except Exception:
                        metadata = {}
                else:
                    metadata = dict(metadata_raw or {})
                metadata.update(
                    {
                        "subject": str(row.get("subject") or ""),
                        "predicate": str(row.get("predicate") or ""),
                        "object": str(row.get("object") or ""),
                    }
                )
                if row.get("source_id") is not None:
                    metadata["source_id"] = row.get("source_id")
                if row.get("doc_id") is not None:
                    metadata["doc_id"] = row.get("doc_id")
                if row.get("snippet") is not None:
                    metadata["snippet"] = row.get("snippet")
                out.append(metadata)
            return out


class AsyncPostgresKnowledgeGraph:
    def __init__(self, config: PostgresKnowledgeGraphConfig):
        self._namespace = config.namespace
        self._table = config.table
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
                        CREATE TABLE IF NOT EXISTS {self._table} (
                            namespace TEXT NOT NULL,
                            fact_key TEXT NOT NULL,
                            source_id TEXT,
                            doc_id TEXT,
                            subject TEXT NOT NULL,
                            subject_norm TEXT NOT NULL,
                            predicate TEXT NOT NULL,
                            object TEXT NOT NULL,
                            object_norm TEXT NOT NULL,
                            snippet TEXT,
                            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (namespace, fact_key)
                        )
                        """
                    )
                )
                for index_name, columns in (
                    (f"idx_{self._table}_namespace_source", "(namespace, source_id)"),
                    (f"idx_{self._table}_namespace_doc", "(namespace, doc_id)"),
                    (f"idx_{self._table}_namespace_subject_norm", "(namespace, subject_norm)"),
                    (f"idx_{self._table}_namespace_object_norm", "(namespace, object_norm)"),
                ):
                    await conn.execute(
                        text(
                            f"""
                            CREATE INDEX IF NOT EXISTS {index_name}
                            ON {self._table} {columns}
                            """
                        )
                    )
            self._initialized = True

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower()

    def _fact_key(self, subject: str, predicate: str, object_: str, metadata: Dict[str, Any]) -> str:
        source_id = str(metadata.get("source_id") or "")
        doc_id = str(metadata.get("doc_id") or "")
        raw = "\x1f".join(
            [
                self._namespace,
                self._norm(subject),
                predicate.strip().lower(),
                self._norm(object_),
                source_id,
                doc_id,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def aupsert_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        await self.aupsert_facts(
            [
                KnowledgeGraphFactRecord(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    metadata=dict(metadata or {}),
                )
            ]
        )

    async def aupsert_facts(self, facts: List[KnowledgeGraphFactRecord]) -> None:
        if not facts:
            return
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        rows = []
        for fact in facts:
            metadata = dict(fact.metadata or {})
            rows.append(
                {
                    "namespace": self._namespace,
                    "fact_key": self._fact_key(fact.subject, fact.predicate, fact.object, metadata),
                    "source_id": str(metadata.get("source_id") or "") or None,
                    "doc_id": str(metadata.get("doc_id") or "") or None,
                    "subject": fact.subject,
                    "subject_norm": self._norm(fact.subject),
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "object_norm": self._norm(fact.object),
                    "snippet": str(metadata.get("snippet") or "") or None,
                    "metadata_json": json.dumps(metadata, ensure_ascii=False),
                }
            )
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table} (
                        namespace,
                        fact_key,
                        source_id,
                        doc_id,
                        subject,
                        subject_norm,
                        predicate,
                        object,
                        object_norm,
                        snippet,
                        metadata_json
                    )
                    VALUES (
                        :namespace,
                        :fact_key,
                        :source_id,
                        :doc_id,
                        :subject,
                        :subject_norm,
                        :predicate,
                        :object,
                        :object_norm,
                        :snippet,
                        CAST(:metadata_json AS JSONB)
                    )
                    ON CONFLICT (namespace, fact_key) DO UPDATE SET
                        source_id = EXCLUDED.source_id,
                        doc_id = EXCLUDED.doc_id,
                        subject = EXCLUDED.subject,
                        subject_norm = EXCLUDED.subject_norm,
                        predicate = EXCLUDED.predicate,
                        object = EXCLUDED.object,
                        object_norm = EXCLUDED.object_norm,
                        snippet = EXCLUDED.snippet,
                        metadata_json = EXCLUDED.metadata_json,
                        updated_at = NOW()
                    """
                ),
                rows,
            )

    async def adelete_facts_for_chunks(self, chunk_ids: List[str]) -> None:
        if not chunk_ids:
            return
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._table}
                    WHERE namespace = :namespace
                      AND (doc_id = ANY(:chunk_ids) OR subject = ANY(:chunk_ids))
                    """
                ),
                {"namespace": self._namespace, "chunk_ids": list(dict.fromkeys(chunk_ids))},
            )

    async def _aneighbor_rows(self, node_norm: str) -> List[Dict[str, Any]]:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT subject, predicate, object, metadata_json
                        FROM {self._table}
                        WHERE namespace = :namespace
                          AND (subject_norm = :node_norm OR object_norm = :node_norm)
                        """
                    ),
                    {"namespace": self._namespace, "node_norm": node_norm},
                )
            ).mappings()
            return [dict(row) for row in rows]

    async def aneighborhood(
        self, node_id: str, hops: int = 2, limit: int = 50
    ) -> List[Dict[str, str]]:
        visited: Set[str] = set()
        queue: deque[Tuple[str, int, str]] = deque([(self._norm(node_id), 0, node_id)])
        results: List[Dict[str, str]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        while queue and len(results) < limit:
            current_norm, depth, display_id = queue.popleft()
            if depth > hops or current_norm in visited:
                continue
            visited.add(current_norm)

            for row in await self._aneighbor_rows(current_norm):
                metadata_raw = row.get("metadata_json")
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                    except Exception:
                        metadata = {}
                else:
                    metadata = dict(metadata_raw or {})

                subject = str(row.get("subject") or "")
                predicate = str(row.get("predicate") or "")
                object_ = str(row.get("object") or "")
                if not subject or not predicate or not object_:
                    continue

                outgoing = self._norm(subject) == current_norm
                target = object_ if outgoing else subject
                edge_key = (display_id, predicate if outgoing else f"inv:{predicate}", target)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                results.append(
                    {
                        "id": target,
                        "predicate": predicate if outgoing else f"inv:{predicate}",
                        "source": display_id,
                        "depth": depth + 1,
                        "direction": "out" if outgoing else "in",
                        **metadata,
                    }
                )
                queue.append((self._norm(target), depth + 1, target))
                if len(results) >= limit:
                    break

        return results[:limit]

    async def aquery(
        self, cypher_or_graph_query: str, params: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        await self._ensure_schema()
        _, text = _async_sqlalchemy()
        async with self._engine.begin() as conn:
            rows = (
                await conn.execute(
                    text(
                        f"""
                        SELECT subject, predicate, object, source_id, doc_id, snippet, metadata_json
                        FROM {self._table}
                        WHERE namespace = :namespace
                        ORDER BY updated_at ASC, created_at ASC
                        """
                    ),
                    {"namespace": self._namespace},
                )
            ).mappings()

            out: List[Dict[str, str]] = []
            for row in rows:
                metadata_raw = row.get("metadata_json")
                if isinstance(metadata_raw, str):
                    try:
                        metadata = json.loads(metadata_raw)
                    except Exception:
                        metadata = {}
                else:
                    metadata = dict(metadata_raw or {})
                metadata.update(
                    {
                        "subject": str(row.get("subject") or ""),
                        "predicate": str(row.get("predicate") or ""),
                        "object": str(row.get("object") or ""),
                    }
                )
                if row.get("source_id") is not None:
                    metadata["source_id"] = row.get("source_id")
                if row.get("doc_id") is not None:
                    metadata["doc_id"] = row.get("doc_id")
                if row.get("snippet") is not None:
                    metadata["snippet"] = row.get("snippet")
                out.append(metadata)
            return out


# --- Extractors ------------------------------------------------------------


def simple_entity_extractor(text: str, max_entities: int = 10) -> List[str]:
    """
    Pulls capitalized noun-ish phrases as entities.
    This is intentionally lightweight; plug a real NER model for production.
    """
    candidates = re.findall(r"\b([A-Z][A-Za-z0-9\-/ ]{2,})\b", text)
    seen: Set[str] = set()
    entities: List[str] = []
    for cand in candidates:
        norm = _clean_phrase(cand)
        if not _is_valid_entity_name(norm, max_words=4):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        entities.append(norm)
        if len(entities) >= max_entities:
            break
    return entities


def simple_kg_extractor(doc: Document) -> List[tuple[str, str, str, Dict[str, str]]]:
    """
    Heuristic triple extractor:
    - Captures patterns like "X is Y", "X are Y", "X supports Y".
    - Also extracts "X: Y" style bullet/heading relationships.
    """
    triples: List[tuple[str, str, str, Dict[str, str]]] = []
    sentences = re.split(r"(?<=[.!?])\s+", doc.text)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Pattern: "X is Y"
        m = re.match(r"(.{2,80})\s+(is|are)\s+(an?\s+)?(.{2,120})", sent, flags=re.IGNORECASE)
        if m:
            subj = m.group(1).strip()
            pred = m.group(2).lower()
            obj = m.group(4).strip()
            if _is_valid_entity_name(subj) and _is_valid_entity_name(obj):
                triples.append(
                    (
                        subj,
                        pred,
                        obj,
                        {
                            "doc_id": doc.id,
                            "source_id": doc.metadata.get("source_id", ""),
                            "snippet": doc.text[:200],
                        },
                    )
                )
            continue

        # Pattern: "X supports Y" / "X enables Y"
        m2 = re.match(r"(.{2,80})\s+(supports|enables|drives|powers)\s+(.{2,120})", sent, flags=re.IGNORECASE)
        if m2:
            subj = m2.group(1).strip()
            pred = m2.group(2).lower()
            obj = m2.group(3).strip()
            if _is_valid_entity_name(subj) and _is_valid_entity_name(obj):
                triples.append(
                    (
                        subj,
                        pred,
                        obj,
                        {
                            "doc_id": doc.id,
                            "source_id": doc.metadata.get("source_id", ""),
                            "snippet": doc.text[:200],
                        },
                    )
                )
            continue

        # Pattern: "Heading: detail"
        if ":" in sent:
            parts = sent.split(":", 1)
            if len(parts[0]) > 3 and len(parts[1]) > 3:
                subj = parts[0].strip()
                pred = "describes"
                obj = parts[1].strip()
                if _is_valid_entity_name(subj) and _is_valid_entity_name(obj):
                    triples.append(
                        (
                            subj,
                            pred,
                            obj,
                            {
                                "doc_id": doc.id,
                                "source_id": doc.metadata.get("source_id", ""),
                                "snippet": doc.text[:200],
                            },
                        )
                    )
    return triples


# --- NLP-assisted extractors ----------------------------------------------

_SPACY_MODEL = None
_QUESTION_WORDS = {"what", "how", "why", "who", "when", "where"}
_QUESTION_PATTERNS = [
    r"\bwhat is\b(.*)",
    r"\bwhat are\b(.*)",
    r"\bhow does\b(.*)",
    r"\bhow do\b(.*)",
    r"\bdefine\b(.*)",
]
_STOPWORDS = {
    "the",
    "and",
    "for",
    "but",
    "of",
    "in",
    "on",
    "to",
    "at",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "our",
    "we",
    "us",
    "you",
    "your",
    "yours",
    "i",
    "me",
    "my",
    "mine",
    "they",
    "them",
    "their",
    "theirs",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
}


def _question_focus(text: str) -> Optional[str]:
    """
    Extracts the main noun phrase from a question like
    "What is the enterprise AI architecture?" -> "enterprise AI architecture".
    """
    lowered = text.lower().strip(" ?!.")
    for pattern in _QUESTION_PATTERNS:
        m = re.search(pattern, lowered)
        if m and m.group(1):
            focus = m.group(1).strip(" ?!.")
            # Drop leading determiners/pronouns
            focus = re.sub(r"^(the|a|an|this|that|these|those)\s+", "", focus)
            if len(focus) >= 3 and focus not in _QUESTION_WORDS:
                return focus
    return None


def _clean_phrase(text: str) -> str:
    out = text.strip(" \n\t\r.,:;?!\"'()[]{}")
    return re.sub(r"\s+", " ", out)


def normalize_entity_name(text: str) -> str:
    """
    Shared normalizer for entity identifiers across KG + metadata.
    """
    return _clean_phrase(text).lower()


def _is_valid_entity_name(text: str, max_words: int = 6, min_chars: int = 3) -> bool:
    cleaned = _clean_phrase(text)
    if not cleaned:
        return False
    lower = cleaned.lower()
    if len(cleaned) < min_chars:
        return False
    words = cleaned.split()
    if len(words) > max_words:
        return False
    if lower in _QUESTION_WORDS or lower in _STOPWORDS:
        return False
    if words and words[0].lower() in _QUESTION_WORDS:
        return False
    if any(token in cleaned for token in ["##", "***", "->", "+", "|"]):
        return False
    # Require at least one alphabetic token; avoid glued headings like "introductionmodern"
    if not any(any(ch.isalpha() for ch in w) for w in words):
        return False
    # Heuristic: reject long single-token strings that look glued
    if len(words) == 1 and len(words[0]) > 20:
        return False
    return True


def _load_spacy_model():
    """
    Attempts to load a spaCy English model; falls back to a blank pipeline with sentencizer.
    """
    global _SPACY_MODEL
    if _SPACY_MODEL is not None:
        return _SPACY_MODEL
    try:
        import spacy  # type: ignore

        try:
            _SPACY_MODEL = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback: blank pipeline with sentencizer only (no POS/DEP but still tokenizes)
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            _SPACY_MODEL = nlp
    except Exception:
        _SPACY_MODEL = None
    return _SPACY_MODEL


def nlp_entity_extractor(text: str, max_entities: int = 15) -> List[str]:
    """
    Uses spaCy noun chunks/ents if available; otherwise falls back to regex heuristic.
    """
    nlp = _load_spacy_model()
    entities: List[str] = []
    seen: Set[str] = set()
    if nlp:
        doc = nlp(text)
        # Prefer named entities; then noun chunks
        try:
            for ent in doc.ents:
                norm = ent.text.strip()
                if not _is_valid_entity_name(norm, max_words=4):
                    continue
                if norm.lower() in seen:
                    continue
                seen.add(norm.lower())
                entities.append(norm)
                if len(entities) >= max_entities:
                    return entities
        except Exception:
            pass

        try:
            for chunk in doc.noun_chunks:
                norm = chunk.text.strip()
                if not _is_valid_entity_name(norm, max_words=4):
                    continue
                if any(token.pos_ in {"PRON", "DET"} for token in chunk):
                    continue
                if norm.lower() in seen:
                    continue
                seen.add(norm.lower())
                entities.append(norm)
                if len(entities) >= max_entities:
                    return entities
        except Exception:
            pass
    # Fallback to regex if not enough entities were found
    entities.extend(simple_entity_extractor(text, max_entities=max_entities - len(entities)))
    # Deduplicate while preserving order
    deduped: List[str] = []
    seen_order: Set[str] = set()
    for e in entities:
        key = e.lower()
        if key in seen_order:
            continue
        seen_order.add(key)
        deduped.append(e)
    # Add question focus if applicable and missing
    focus = _question_focus(text)
    if focus and focus.lower() not in seen_order:
        deduped.append(focus)
    return deduped[:max_entities]


def nlp_kg_extractor(doc: Document) -> List[tuple[str, str, str, Dict[str, str]]]:
    """
    SpaCy-assisted triple extraction with rule-based fallback.
    """
    triples = simple_kg_extractor(doc)
    nlp = _load_spacy_model()
    if not nlp or not hasattr(nlp, "pipeline") or len(nlp.pipeline) == 0:
        return triples

    parsed = nlp(doc.text)
    added: Set[tuple[str, str, str]] = set()
    max_triples = 50
    snippet_preview = doc.text[:200]

    def _add(subj: str, pred: str, obj: str):
        nonlocal added, triples
        s = _clean_phrase(subj)
        o = _clean_phrase(obj)
        if not _is_valid_entity_name(s) or not _is_valid_entity_name(o):
            return
        key = (s.lower(), pred.lower(), o.lower())
        if key in added:
            return
        added.add(key)
        triples.append(
            (
                s,
                pred.lower(),
                o,
                {
                    "doc_id": doc.id,
                    "source_id": doc.metadata.get("source_id", ""),
                    "snippet": snippet_preview,
                },
            )
        )

    for sent in parsed.sents:
        # Verb-driven relations
        verbs = [t for t in sent if t.pos_ in {"VERB", "AUX"}]
        for v in verbs:
            subjects = [c for c in v.children if c.dep_ in {"nsubj", "nsubjpass"} and c.pos_ not in {"PRON", "DET"}]
            dobjs = [c for c in v.children if c.dep_ in {"dobj", "obj", "attr"}]
            pobj = [c for c in v.children if c.dep_ == "prep" for c in c.children if c.dep_ == "pobj"]
            targets = dobjs + pobj
            if not subjects or not targets:
                continue
            subj_text = subjects[0].text
            for tgt in targets:
                obj_text = tgt.text
                pred = v.lemma_.lower()
                _add(subj_text, pred, obj_text)
                if len(triples) >= max_triples:
                    return triples

        # Co-occurrence edges between noun chunks to capture related concepts
        try:
            noun_chunks = [nc for nc in sent.noun_chunks if len(nc.text.strip()) >= 3]
        except Exception:
            noun_chunks = []
        filtered_chunks = []
        for nc in noun_chunks:
            if any(tok.pos_ in {"PRON", "DET"} for tok in nc):
                continue
            cleaned = _clean_phrase(nc.text)
            if len(cleaned.split()) < 2:
                continue
            if cleaned.lower() in _STOPWORDS or cleaned.lower() in _QUESTION_WORDS:
                continue
            filtered_chunks.append(cleaned)

        for i in range(len(filtered_chunks)):
            for j in range(i + 1, len(filtered_chunks)):
                _add(filtered_chunks[i], "co_occurs", filtered_chunks[j])
                if len(triples) >= max_triples:
                    return triples

    return triples

class Triple(TypedDict):
    subject: str
    predicate: str
    object: str

# --- LLM-based extractors --------------------------------------------------


class EntitiesOut(TypedDict):
    entities: List[str]


class TriplesOut(TypedDict):
    triples: List[Triple]


class KGExtractionOut(TypedDict):
    entities: List[str]
    triples: List[Triple]

def make_llm_entity_extractor(
    model_config: ModelConfig,
    max_entities: int = 10,
) -> Callable[[str], List[str]]:
    """
    Returns an entity extractor backed by an LLM (Andromeda get_chat_model).
    If the model cannot be loaded, falls back to the heuristic extractor.
    """
    try:
        from andromeda.utils import get_chat_model  # type: ignore
        from andromeda.config import ModelConfig  # type: ignore
        chat = get_chat_model(model_config)
    except Exception:
        def fallback(text: str) -> List[str]:
            return nlp_entity_extractor(text, max_entities=max_entities)
        return fallback

    def _extract(text: str) -> List[str]:
        prompt = (
            "Extract up to {max_entities} clean factual entities from the text.\n"
            "Return structured output with key 'entities'.\n"
            "Rules: entities <= 6 words, no headings/bullets, no question words."
        ).format(max_entities=max_entities)
        text = text[:3500]  # guardrail
        msg = f"{prompt}\n\nTEXT:\n{text}"
        try:
            resp = chat.with_structured_output(EntitiesOut).invoke(msg)
            entities_raw = resp.get("entities") if isinstance(resp, dict) else None
            if not isinstance(entities_raw, list):
                entities_raw = []
        except Exception:
            return nlp_entity_extractor(text, max_entities=max_entities)

        entities: List[str] = []
        seen: Set[str] = set()
        for entity in entities_raw:
            if not _is_valid_entity_name(entity):
                continue
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(entity)
            if len(entities) >= max_entities:
                break
        if not entities:
            return nlp_entity_extractor(text, max_entities=max_entities)
        return entities

    return _extract

def make_llm_kg_extractor(
    model_config: ModelConfig,
    max_triples: int = 10,
) -> Callable[[Document], List[tuple[str, str, str, Dict[str, str]]]]:
    """
    Returns a triple extractor backed by an LLM (Andromeda get_chat_model).
    If the model cannot be loaded, falls back to the heuristic extractor.
    """
    try:
        from andromeda.utils import get_chat_model  # type: ignore
        from andromeda.config import ModelConfig  # type: ignore

        chat = get_chat_model(model_config)
    except Exception:
        def fallback(doc: Document):
            return nlp_kg_extractor(doc)
        return fallback

    def _extract(doc: Document) -> List[tuple[str, str, str, Dict[str, str]]]:
        prompt = (
            "Extract up to {max_triples} clean factual triples from the text. "
            "Return structured output with key 'triples' (each item has subject,predicate,object). "
            "Rules: subjects/objects <= 6 words, no headings/bullets, no question words, "
            "predicate should be a verb-like relation (e.g., 'is', 'enables', 'relates_to'), "
            "drop noisy spans like section titles or markup."
        ).format(max_triples=max_triples)
        text = doc.text[:3500]  # guardrail
        msg = f"{prompt}\n\nTEXT:\n{text}"
        try:
            resp = chat.with_structured_output(TriplesOut).invoke(msg)
            triples_raw = resp.get("triples") if isinstance(resp, dict) else None
            if not isinstance(triples_raw, list):
                triples_raw = []
        except Exception:
            return nlp_kg_extractor(doc)

        triples: List[tuple[str, str, str, Dict[str, str]]] = []
        seen: Set[tuple[str, str, str]] = set()
        for item in triples_raw:
            if not isinstance(item, dict):
                continue
            subj = item.get("subject")
            pred = item.get("predicate")
            obj = item.get("object")
            if not (subj and pred and obj):
                continue
            if not (_is_valid_entity_name(subj) and _is_valid_entity_name(obj)):
                continue
            key = (subj.lower(), pred.lower(), obj.lower())
            if key in seen:
                continue
            seen.add(key)
            triples.append(
                (
                    _clean_phrase(subj),
                    pred.lower(),
                    _clean_phrase(obj),
                    {
                        "doc_id": doc.id,
                        "source_id": doc.metadata.get("source_id", ""),
                        "snippet": doc.text[:200],
                    },
                )
            )
            if len(triples) >= max_triples:
                break
        if not triples:
            return nlp_kg_extractor(doc)
        return triples

    return _extract


def bundle_kg_extractor_fallback(doc: Document, max_entities: int = 15) -> KGExtraction:
    """
    Deterministic fallback bundling: entities come from NLP entity extraction and are
    augmented with subjects/objects from extracted triples.
    """
    entities = nlp_entity_extractor(doc.text, max_entities=max_entities)
    triples_with_meta = nlp_kg_extractor(doc)
    triples: List[Tuple[str, str, str]] = [(s, p, o) for (s, p, o, _m) in triples_with_meta]

    ent_seen: Set[str] = set()
    ent_out: List[str] = []
    for e in entities:
        key = normalize_entity_name(e)
        if not key or key in ent_seen:
            continue
        ent_seen.add(key)
        ent_out.append(e)

    # Add subjects/objects (best-effort) so entities and triples stay aligned.
    for s, _p, o in triples:
        for candidate in (s, o):
            if not _is_valid_entity_name(candidate):
                continue
            key = normalize_entity_name(candidate)
            if key in ent_seen:
                continue
            ent_seen.add(key)
            ent_out.append(candidate)
            if len(ent_out) >= max_entities:
                break
        if len(ent_out) >= max_entities:
            break

    return KGExtraction(entities=ent_out, triples=triples)


def make_llm_kg_bundle_extractor(
    model_config: ModelConfig,
    max_entities: int = 12,
    max_triples: int = 12,
) -> Callable[[Document], KGExtraction]:
    """
    Single-call LLM extractor returning both entities and triples for a chunk.

    Output schema (JSON object):
      {"entities":[...], "triples":[{"subject":"...","predicate":"...","object":"..."}]}
    """
    try:
        from andromeda.utils import get_chat_model  # type: ignore
        from andromeda.config import ModelConfig  # type: ignore

        chat = get_chat_model(model_config)
    except Exception:
        def fallback(doc: Document) -> KGExtraction:
            return bundle_kg_extractor_fallback(doc, max_entities=max_entities)
        return fallback

    def _extract(doc: Document) -> KGExtraction:
        prompt = (
            "Extract entities and factual triples from the text.\n"
            "Return structured output with keys:\n"
            f"- entities: up to {max_entities} strings\n"
            f"- triples: up to {max_triples} objects with subject,predicate,object\n"
            "Rules:\n"
            "- entities/subjects/objects <= 6 words\n"
            "- predicate should be a verb-like relation (e.g., is, enables, uses, relates_to)\n"
            "- no headings/bullets/section titles\n"
            "- do not include pronouns/determiners like 'our', 'we', 'their'\n"
            "- do not invent facts not present in the text\n"
        )
        text = doc.text[:3500]
        msg = f"{prompt}\nTEXT:\n{text}"
        try:
            resp = chat.with_structured_output(KGExtractionOut).invoke(msg)
            entities_raw = resp.get("entities") if isinstance(resp, dict) else None
            triples_raw = resp.get("triples") if isinstance(resp, dict) else None
            if not isinstance(entities_raw, list):
                entities_raw = []
            if not isinstance(triples_raw, list):
                triples_raw = []
        except Exception:
            return bundle_kg_extractor_fallback(doc, max_entities=max_entities)

        entities: List[str] = []
        seen: Set[str] = set()
        for e in entities_raw:
            if not isinstance(e, str):
                continue
            if not _is_valid_entity_name(e):
                continue
            key = normalize_entity_name(e)
            if not key or key in seen:
                continue
            seen.add(key)
            entities.append(str(e))
            if len(entities) >= max_entities:
                break

        triples: List[Tuple[str, str, str]] = []
        seen_t: Set[Tuple[str, str, str]] = set()
        for t in triples_raw:
            if not isinstance(t, dict):
                continue
            s = t.get("subject")
            p = t.get("predicate")
            o = t.get("object")
            if not (s and p and o):
                continue
            if not (_is_valid_entity_name(s) and _is_valid_entity_name(o)):
                continue
            key = (normalize_entity_name(s), str(p).lower(), normalize_entity_name(o))
            if key in seen_t:
                continue
            seen_t.add(key)
            triples.append((_clean_phrase(str(s)), str(p).lower(), _clean_phrase(str(o))))
            if len(triples) >= max_triples:
                break

        if not entities and not triples:
            return bundle_kg_extractor_fallback(doc, max_entities=max_entities)
        return KGExtraction(entities=entities, triples=triples)

    return _extract
