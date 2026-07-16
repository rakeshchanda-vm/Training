from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from andromeda.config.config import ModelConfig
from andromeda.retrievers.core import Reranker, ScoredChunk


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


class EmbeddingCosineReranker(Reranker):
    """
    Lightweight reranker that scores candidates by cosine similarity between
    query embedding and candidate embeddings.

    Works with any LangChain-compatible Embeddings object (embed_query/embed_documents).
    """

    def __init__(
        self,
        embedding_model: Any,
        max_chars_per_candidate: int = 4000,
    ):
        self._emb = embedding_model
        self._max_chars = max_chars_per_candidate

    def rerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        if not candidates:
            return []

        embed_query = getattr(self._emb, "embed_query", None)
        embed_docs = getattr(self._emb, "embed_documents", None)
        if not callable(embed_query) or not callable(embed_docs):
            return candidates[:k] if k else candidates

        texts: List[str] = [(c.text or "")[: self._max_chars] for c in candidates]

        try:
            qv = embed_query(query)
            dvs = embed_docs(texts)
        except Exception:
            return candidates[:k] if k else candidates

        scored: List[ScoredChunk] = []
        for c, dv in zip(candidates, dvs):
            try:
                sim = float(_cosine(qv, dv))
            except Exception:
                sim = 0.0
            scored.append(
                ScoredChunk(
                    doc_id=c.doc_id,
                    text=c.text,
                    metadata=c.metadata,
                    score=sim,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k] if k else scored

    async def arerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        if not candidates:
            return []

        embed_query = getattr(self._emb, "aembed_query", None)
        embed_docs = getattr(self._emb, "aembed_documents", None)
        if callable(embed_query) and callable(embed_docs):
            texts: List[str] = [(c.text or "")[: self._max_chars] for c in candidates]
            try:
                qv = await embed_query(query)
                dvs = await embed_docs(texts)
            except Exception:
                return candidates[:k] if k else candidates

            scored: List[ScoredChunk] = []
            for c, dv in zip(candidates, dvs):
                try:
                    sim = float(_cosine(qv, dv))
                except Exception:
                    sim = 0.0
                scored.append(
                    ScoredChunk(
                        doc_id=c.doc_id,
                        text=c.text,
                        metadata=c.metadata,
                        score=sim,
                    )
                )

            scored.sort(key=lambda x: x.score, reverse=True)
            return scored[:k] if k else scored

        return await asyncio.to_thread(self.rerank, query, candidates, k)


@dataclass(frozen=True)
class LLMRerankerConfig:
    model: ModelConfig
    max_candidates: int = 30
    max_chars_per_candidate: int = 800


class RankingOut(TypedDict):
    ranking: List[str]


class LLMListwiseReranker(Reranker):
    """
    Listwise reranker using a single LLM call that returns a JSON ranking of ids.

    This is slower/more expensive than embedding reranking, but can be higher quality.
    """

    def __init__(self, config: LLMRerankerConfig):
        self._cfg = config
        try:
            from andromeda.utils import get_chat_model  # type: ignore
            from andromeda.config import ModelConfig  # type: ignore

            self._chat = get_chat_model(config.model)
        except Exception:
            self._chat = None

    def rerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        if not candidates:
            return []
        if self._chat is None:
            return candidates[:k] if k else candidates

        limit = min(len(candidates), self._cfg.max_candidates)
        pool = candidates[:limit]

        items = []
        for c in pool:
            meta = c.metadata or {}
            src = meta.get("retrieval_source", "base")
            depth = meta.get("depth")
            seed = meta.get("kg_seed_entity")
            text = (c.text or "")[: self._cfg.max_chars_per_candidate]
            items.append(
                {
                    "id": c.doc_id,
                    "source": src,
                    "depth": depth,
                    "seed": seed,
                    "text": text,
                }
            )

        prompt = (
            "You are reranking retrieval candidates for answering a user query.\n"
            "Return structured output with key 'ranking' (a list of ids).\n"
            "Rules:\n"
            "- ranking must contain only ids from the provided candidates\n"
            "- rank by relevance to the query; prioritize factual coverage and directness\n"
            "- you may omit ids you consider irrelevant\n"
            "\n"
            f"QUERY:\n{query}\n\n"
            f"CANDIDATES (max {limit}):\n{json.dumps(items, ensure_ascii=False)}\n"
        )

        try:
            resp = self._chat.with_structured_output(RankingOut).invoke(prompt)
            ranking = resp.get("ranking") if isinstance(resp, dict) else None
            if not isinstance(ranking, list):
                ranking = []
        except Exception:
            return candidates[:k] if k else candidates

        if not isinstance(ranking, list):
            return candidates[:k] if k else candidates

        wanted = [str(x) for x in ranking if isinstance(x, (str, int, float))]
        by_id = {c.doc_id: c for c in candidates}
        ordered: List[ScoredChunk] = []
        for doc_id in wanted:
            c = by_id.get(doc_id)
            if c and c not in ordered:
                ordered.append(c)

        # Fill with remaining candidates in original order.
        for c in candidates:
            if c not in ordered:
                ordered.append(c)

        return ordered[:k] if k else ordered

    async def arerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        if not candidates:
            return []
        if self._chat is None:
            return candidates[:k] if k else candidates

        limit = min(len(candidates), self._cfg.max_candidates)
        pool = candidates[:limit]

        items = []
        for c in pool:
            meta = c.metadata or {}
            src = meta.get("retrieval_source", "base")
            depth = meta.get("depth")
            seed = meta.get("kg_seed_entity")
            text = (c.text or "")[: self._cfg.max_chars_per_candidate]
            items.append(
                {
                    "id": c.doc_id,
                    "source": src,
                    "depth": depth,
                    "seed": seed,
                    "text": text,
                }
            )

        prompt = (
            "You are reranking retrieval candidates for answering a user query.\n"
            "Return structured output with key 'ranking' (a list of ids).\n"
            "Rules:\n"
            "- ranking must contain only ids from the provided candidates\n"
            "- rank by relevance to the query; prioritize factual coverage and directness\n"
            "- you may omit ids you consider irrelevant\n"
            "\n"
            f"QUERY:\n{query}\n\n"
            f"CANDIDATES (max {limit}):\n{json.dumps(items, ensure_ascii=False)}\n"
        )

        try:
            async_invoke = getattr(self._chat.with_structured_output(RankingOut), "ainvoke", None)
            if callable(async_invoke):
                resp = await async_invoke(prompt)
            else:
                return await asyncio.to_thread(self.rerank, query, candidates, k)
            ranking = resp.get("ranking") if isinstance(resp, dict) else None
            if not isinstance(ranking, list):
                ranking = []
        except Exception:
            return candidates[:k] if k else candidates

        wanted = [str(x) for x in ranking if isinstance(x, (str, int, float))]
        by_id = {c.doc_id: c for c in candidates}
        ordered: List[ScoredChunk] = []
        for doc_id in wanted:
            c = by_id.get(doc_id)
            if c and c not in ordered:
                ordered.append(c)

        for c in candidates:
            if c not in ordered:
                ordered.append(c)

        return ordered[:k] if k else ordered
