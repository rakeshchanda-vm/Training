
from __future__ import annotations

import asyncio
import importlib
import threading
from typing import Any, Dict, List

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from andromeda.ports.bm25 import BM25Retriever
from andromeda.retrievers.core import (
    Document,
    LexicalBackend,
    MetadataFilter,
    ScoredChunk,
    VectorStoreBackend,
    metadata_matches_filter,
)


def _load_symbol(module_name: str, symbol: str, extra_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{symbol} requires optional retriever dependencies. "
            f"Install with `pip install \"andromeda[{extra_name}]\"`."
        ) from exc
    return getattr(module, symbol)


def _to_lc_docs(docs: List[Document]) -> List[LCDocument]:
    return [
        LCDocument(
            id=d.id,
            page_content=d.text,
            metadata={"_id": d.id, **(d.metadata or {})},
        )
        for d in docs
    ]


def _from_lc_docs(results) -> List[ScoredChunk]:
    out: List[ScoredChunk] = []
    for item in results:
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc, score = item, 0.0
        meta = dict(doc.metadata or {})
        doc_id = (
            getattr(doc, "id", None)
            or meta.pop("_id", None)
            or meta.get("id")
            or ""
        )
        out.append(
            ScoredChunk(
                doc_id=doc_id,
                text=doc.page_content,
                metadata=meta,
                score=float(score),
            )
        )
    return out


class ChromaBackend(VectorStoreBackend):
    def __init__(self, collection_name: str, embedding: Embeddings, persist_dir: str):
        chroma_cls = _load_symbol(
            "langchain_chroma",
            "Chroma",
            "retrievers-chroma",
        )
        self._vs = chroma_cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_dir,
        )

    def add_documents(self, docs: List[Document]) -> None:
        # Use stable ids to avoid duplicate vectors on re-ingest.
        self._vs.add_documents(_to_lc_docs(docs), ids=[d.id for d in docs])

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self._vs.delete(ids=ids)
        except Exception:
            pass

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[ScoredChunk]:
        docs_and_scores = self._vs.similarity_search_with_score(query, k=k, **kwargs)
        return _from_lc_docs(docs_and_scores)


class FaissBackend(VectorStoreBackend):
    def __init__(
        self,
        embedding: Embeddings,
        index_path: str | None = None,
        allow_dangerous_deserialization: bool = False,
    ):
        self._embedding = embedding
        self._index_path = index_path
        faiss_cls = _load_symbol(
            "andromeda.ports.faiss",
            "FAISS",
            "retrievers-faiss",
        )
        if index_path:
            self._vs = faiss_cls.load_local(
                index_path,
                embeddings=embedding,
                allow_dangerous_deserialization=allow_dangerous_deserialization,
            )
        else:
            self._vs = None

    def add_documents(self, docs: List[Document]) -> None:
        faiss_cls = _load_symbol(
            "andromeda.ports.faiss",
            "FAISS",
            "retrievers-faiss",
        )
        lc_docs = _to_lc_docs(docs)
        if self._vs is None:
            self._vs = faiss_cls.from_documents(
                lc_docs, self._embedding, ids=[d.id for d in docs]
            )
        else:
            self._vs.add_documents(lc_docs, ids=[d.id for d in docs])
        if self._index_path:
            self._vs.save_local(self._index_path)

    def delete_documents(self, ids: List[str]) -> None:
        if not ids or self._vs is None:
            return
        try:
            self._vs.delete(ids=ids)
        except Exception:
            return
        if self._index_path:
            self._vs.save_local(self._index_path)

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[ScoredChunk]:
        if self._vs is None:
            return []
        docs_and_scores = self._vs.similarity_search_with_score(query, k=k, **kwargs)
        return _from_lc_docs(docs_and_scores)


class AzureSearchBackend(VectorStoreBackend):
    def __init__(
        self,
        embedding: Embeddings,
        azure_search_endpoint: str,
        azure_search_key: str,
        index_name: str,
    ):
        azure_cls = _load_symbol(
            "langchain_azure_ai.vectorstores",
            "AzureSearch",
            "retrievers-azure",
        )
        self._vs = azure_cls(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            index_name=index_name,
            embedding_function=embedding,
        )

    def add_documents(self, docs: List[Document]) -> None:
        self._vs.add_documents(_to_lc_docs(docs))

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self._vs.delete(ids=ids)
        except Exception:
            pass

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[ScoredChunk]:
        docs_and_scores = self._vs.similarity_search_with_score(query, k=k, **kwargs)
        return _from_lc_docs(docs_and_scores)


class MongoAtlasBackend(VectorStoreBackend):
    def __init__(
        self,
        embedding: Embeddings,
        connection_string: str,
        db_name: str,
        collection_name: str,
        index_name: str,
    ):
        mongo_cls = _load_symbol(
            "langchain_mongodb",
            "MongoDBAtlasVectorSearch",
            "retrievers-mongo",
        )
        self._vs = mongo_cls.from_connection_string(
            connection_string=connection_string,
            namespace=f"{db_name}.{collection_name}",
            index_name=index_name,
            embedding=embedding,
        )

    def add_documents(self, docs: List[Document]) -> None:
        self._vs.add_documents(_to_lc_docs(docs))

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self._vs.delete(ids=ids)
        except Exception:
            pass

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[ScoredChunk]:
        docs_and_scores = self._vs.similarity_search_with_score(query, k=k, **kwargs)
        return _from_lc_docs(docs_and_scores)


class PgVectorBackend(VectorStoreBackend):
    def __init__(
        self,
        embedding: Embeddings,
        connection_string: str,
        collection_table: str,
    ):
        pgvector_cls = _load_symbol(
            "langchain_postgres",
            "PGVector",
            "retrievers-postgres",
        )
        self._vs = pgvector_cls(
            embeddings=embedding,
            connection=connection_string,
            collection_name=collection_table,
        )

    def add_documents(self, docs: List[Document]) -> None:
        self._vs.add_documents(_to_lc_docs(docs))

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self._vs.delete(ids=ids)
        except Exception:
            pass

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[ScoredChunk]:
        docs_and_scores = self._vs.similarity_search_with_score(query, k=k, **kwargs)
        return _from_lc_docs(docs_and_scores)


class AsyncPgVectorBackend:
    def __init__(
        self,
        embedding: Embeddings,
        connection_string: str,
        collection_table: str,
    ):
        pgvector_cls = _load_symbol(
            "langchain_postgres",
            "PGVector",
            "retrievers-postgres",
        )
        self._vs = pgvector_cls(
            embeddings=embedding,
            connection=connection_string,
            collection_name=collection_table,
            async_mode=True,
        )
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            apost_init = getattr(self._vs, "__apost_init__", None)
            if callable(apost_init):
                await apost_init()
            self._initialized = True

    async def aadd_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        await self._ensure_initialized()
        await self._vs.aadd_documents(_to_lc_docs(docs), ids=[d.id for d in docs])

    async def adelete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        await self._ensure_initialized()
        try:
            await self._vs.adelete(ids=ids)
        except Exception:
            pass

    async def asimilarity_search(
        self,
        query: str,
        k: int = 10,
        **kwargs,
    ) -> List[ScoredChunk]:
        await self._ensure_initialized()
        docs_and_scores = await self._vs.asimilarity_search_with_score(
            query,
            k=k,
            **kwargs,
        )
        return _from_lc_docs(docs_and_scores)


class OpenSearchBackend(VectorStoreBackend):
    def __init__(
        self,
        embedding: Embeddings,
        opensearch_url: str,
        index_name: str,
        http_auth: tuple[str, str],
    ):
        opensearch_cls = _load_symbol(
            "andromeda.ports.opensearch",
            "OpenSearchVectorSearch",
            "retrievers-opensearch",
        )
        self._vs = opensearch_cls(
            embedding_function=embedding,
            index_name=index_name,
            opensearch_url=opensearch_url,
            http_auth=http_auth,
        )

    def add_documents(self, docs: List[Document]) -> None:
        self._vs.add_documents(_to_lc_docs(docs))

    def delete_documents(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self._vs.delete(ids=ids)
        except Exception:
            pass

    def similarity_search(self, query: str, k: int = 10, **kwargs) -> List[ScoredChunk]:
        docs_and_scores = self._vs.similarity_search_with_score(query, k=k, **kwargs)
        return _from_lc_docs(docs_and_scores)


class BM25LexicalBackend(LexicalBackend):
    def __init__(self):
        self._retriever: BM25Retriever | None = None
        self._docs: Dict[str, LCDocument] = {}
        self._dirty: bool = False
        self._search_lock = threading.RLock()

    def index_documents(self, docs: List[Document]) -> None:
        lc_docs = _to_lc_docs(docs)
        for d in lc_docs:
            doc_id = (d.metadata or {}).get("_id")
            if doc_id:
                self._docs[str(doc_id)] = d
        self._dirty = True

    def delete_documents(self, ids: List[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)
        self._dirty = True

    def _ensure(self) -> None:
        if not self._docs:
            # Fresh process where delta-ingest had no changes.
            self._retriever = None
            self._dirty = False
            return
        if self._retriever is None or self._dirty:
            self._retriever = BM25Retriever.from_documents(list(self._docs.values()))
            self._dirty = False

    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
        **kwargs,
    ) -> List[ScoredChunk]:
        with self._search_lock:
            self._ensure()
            if self._retriever is None:
                return []
            prior_k = getattr(self._retriever, "k", None)
            self._retriever.k = len(self._docs) if metadata_filter else k
            try:
                results = self._retriever.invoke(query, **kwargs)
            finally:
                if prior_k is not None:
                    self._retriever.k = prior_k
        scored: List[ScoredChunk] = []
        for d in results:
            if len(scored) >= k:
                break
            meta = dict(d.metadata or {})
            if not metadata_matches_filter(meta, metadata_filter):
                continue
            doc_id = (
                getattr(d, "id", None)
                or meta.pop("_id", None)
                or meta.get("id")
                or ""
            )
            scored.append(
                ScoredChunk(
                    doc_id=doc_id,
                    text=d.page_content,
                    metadata=meta,
                    score=float(max(k - len(scored), 0)),
                )
            )
        return scored

    async def aindex_documents(self, docs: List[Document]) -> None:
        await asyncio.to_thread(self.index_documents, docs)

    async def adelete_documents(self, ids: List[str]) -> None:
        await asyncio.to_thread(self.delete_documents, ids)

    async def asearch(
        self,
        query: str,
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
        **kwargs,
    ) -> List[ScoredChunk]:
        return await asyncio.to_thread(
            self.search,
            query,
            k,
            metadata_filter=metadata_filter,
            **kwargs,
        )
