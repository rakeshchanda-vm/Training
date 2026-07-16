
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from andromeda.retrievers.core import (
    LexicalBackend,
    MetadataFilter,
    Reranker,
    ScoredChunk,
    VectorStoreBackend,
)


def rrf_fuse(
    ranked_lists: List[List[ScoredChunk]],
    k_final: int = 10,
    k_constant: int = 60,
) -> List[ScoredChunk]:
    scores: Dict[str, float] = {}
    payload: Dict[str, ScoredChunk] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            key = item.doc_id
            payload[key] = item
            scores[key] = scores.get(key, 0.0) + 1.0 / (k_constant + rank + 1)

    sorted_ids = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k_final]
    fused: List[ScoredChunk] = []
    for doc_id, fused_score in sorted_ids:
        c = payload[doc_id]
        fused.append(
            ScoredChunk(
                doc_id=c.doc_id,
                text=c.text,
                metadata=c.metadata,
                score=fused_score,
            )
        )
    return fused


class DenseRetriever:
    def __init__(self, vector_backend: VectorStoreBackend):
        self._vs = vector_backend

    def retrieve(
        self,
        query: str,
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        return self._vs.similarity_search(
            query,
            k=k,
            **_vector_search_kwargs(metadata_filter=metadata_filter, kwargs=kwargs),
        )


class AsyncDenseRetriever:
    def __init__(self, vector_backend):
        self._vs = vector_backend

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        search_kwargs = _vector_search_kwargs(
            metadata_filter=metadata_filter,
            kwargs=kwargs,
        )
        async_search = getattr(self._vs, "asimilarity_search", None)
        if callable(async_search):
            return await async_search(query, k=k, **search_kwargs)
        return await asyncio.to_thread(
            self._vs.similarity_search,
            query,
            k,
            **search_kwargs,
        )


class HybridRetriever:
    def __init__(
        self,
        vector_backend: VectorStoreBackend,
        lexical_backend: LexicalBackend,
        rrf_k: int = 60,
    ):
        self._vs = vector_backend
        self._lex = lexical_backend
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k_vector: int = 20,
        k_lexical: int = 20,
        k_final: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        dense = self._vs.similarity_search(
            query,
            k=k_vector,
            **_vector_search_kwargs(metadata_filter=metadata_filter),
        )
        sparse = self._lex.search(query, k=k_lexical, metadata_filter=metadata_filter)
        return rrf_fuse([dense, sparse], k_final=k_final, k_constant=self._rrf_k)


class AsyncHybridRetriever:
    def __init__(
        self,
        vector_backend,
        lexical_backend,
        rrf_k: int = 60,
    ):
        self._vs = vector_backend
        self._lex = lexical_backend
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        k_vector: int = 20,
        k_lexical: int = 20,
        k_final: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        search_kwargs = _vector_search_kwargs(metadata_filter=metadata_filter)
        async_search = getattr(self._vs, "asimilarity_search", None)
        if callable(async_search):
            dense_task = async_search(query, k=k_vector, **search_kwargs)
        else:
            dense_task = asyncio.to_thread(
                self._vs.similarity_search,
                query,
                k_vector,
                **search_kwargs,
            )

        async_lex = getattr(self._lex, "asearch", None)
        if callable(async_lex):
            sparse_task = async_lex(query, k=k_lexical, metadata_filter=metadata_filter)
        else:
            sparse_task = asyncio.to_thread(
                self._lex.search,
                query,
                k_lexical,
                metadata_filter=metadata_filter,
            )

        dense, sparse = await asyncio.gather(dense_task, sparse_task)
        return rrf_fuse([dense, sparse], k_final=k_final, k_constant=self._rrf_k)


class NoopReranker:
    def rerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        return candidates[:k] if k else candidates

    async def arerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        return self.rerank(query, candidates, k=k)


def _vector_search_kwargs(
    *,
    metadata_filter: MetadataFilter | None,
    kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    search_kwargs = dict(kwargs or {})
    if metadata_filter:
        search_kwargs["filter"] = dict(metadata_filter)
    return search_kwargs
