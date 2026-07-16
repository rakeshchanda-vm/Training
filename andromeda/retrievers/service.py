
from __future__ import annotations

import asyncio
from typing import Literal, List, Dict, Any

from andromeda.retrievers.config import RAGRegistry
from andromeda.retrievers.core import MetadataFilter, ScoredChunk


RetrievalMode = Literal["dense", "hybrid", "hybrid+rerank", "graphrag"]


class RetrievalService:
    def __init__(self, registry: RAGRegistry):
        self.registry = registry
        self.reranker = registry.reranker
        self.entity_extractor = getattr(registry, "entity_extractor", None)

    def retrieve(
        self,
        corpus: str,
        query: str,
        mode: RetrievalMode = "hybrid+rerank",
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        if mode == "dense":
            base = self.registry.get_dense(corpus).retrieve(
                query,
                k=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return self.reranker.rerank(query, base, k=k)

        if mode == "hybrid":
            base = self.registry.get_hybrid(corpus).retrieve(
                query,
                k_vector=k,
                k_lexical=k,
                k_final=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return base

        if mode == "hybrid+rerank":
            base = self.registry.get_hybrid(corpus).retrieve(
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=3 * k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return self.reranker.rerank(query, base, k=k)

        if mode == "graphrag":
            gr = self.registry.get_graphrag(corpus)
            k_base = 2 * k
            k_graph = 4 * k
            base = gr.retrieve(
                query,
                k_final=k_base,
                max_expanded=k_graph,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return self.reranker.rerank(query, base, k=k)

        raise ValueError(f"Unknown retrieval mode: {mode}")

    def retrieve_with_debug(
        self,
        corpus: str,
        query: str,
        mode: RetrievalMode = "hybrid+rerank",
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ):
        """
        Returns results plus trace metadata showing how results were produced
        and any KG relationships pulled in.
        """
        debug: Dict[str, Any] = {
            "query": query,
            "mode": mode,
        }

        if mode == "dense":
            base = self.registry.get_dense(corpus).retrieve(
                query,
                k=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            final = self.reranker.rerank(query, base, k=k)
            debug["base_candidates"] = base
            debug["reranked"] = final
            return final, debug

        if mode == "hybrid":
            base = self.registry.get_hybrid(corpus).retrieve(
                query,
                k_vector=k,
                k_lexical=k,
                k_final=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            debug["base_candidates"] = base
            return base, debug

        if mode == "hybrid+rerank":
            base = self.registry.get_hybrid(corpus).retrieve(
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=3 * k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            final = self.reranker.rerank(query, base, k=k)
            debug["base_candidates"] = base
            debug["reranked"] = final
            return final, debug

        if mode == "graphrag":
            gr = self.registry.get_graphrag(corpus)
            kg_backend = self.registry.get_kg_backend(corpus)
            graph_hops = getattr(gr, "_hops", 2)
            k_base = 2 * k
            k_graph = 4 * k
            base, gr_debug = gr.retrieve_with_debug(
                query,
                k_final=k_base,
                max_expanded=k_graph,
                **_metadata_filter_kwargs(metadata_filter),
            )
            final = self.reranker.rerank(query, base, k=k)
            debug["base_candidates"] = base
            debug["reranked"] = final
            debug["graphrag"] = gr_debug

            # Attach KG neighborhood for entities in the query.
            if kg_backend:
                entities = []
                if isinstance(gr_debug, dict):
                    entities = gr_debug.get("seed_entities") or []
                neighborhoods = {}
                for ent in entities:
                    neighborhoods[ent] = kg_backend.neighborhood(
                        ent,
                        hops=graph_hops,
                        limit=10,
                    )
                # Also include neighborhoods of entities from results (metadata)
                result_entities = []
                for c in base:
                    if isinstance(c.metadata, dict):
                        ent_meta = c.metadata.get("entity_ids")
                        if isinstance(ent_meta, str):
                            result_entities.extend(
                                [e for e in ent_meta.split(",") if e]
                            )
                        elif isinstance(ent_meta, list):
                            result_entities.extend(ent_meta)
                for ent in set(result_entities):
                    neighborhoods[ent] = kg_backend.neighborhood(
                        ent,
                        hops=graph_hops,
                        limit=10,
                    )
                debug["kg_neighborhood"] = neighborhoods
            return final, debug

        raise ValueError(f"Unknown retrieval mode: {mode}")


def _metadata_filter_kwargs(
    metadata_filter: MetadataFilter | None,
) -> Dict[str, MetadataFilter]:
    if not metadata_filter:
        return {}
    return {"metadata_filter": metadata_filter}


class AsyncRetrievalService:
    def __init__(self, registry: RAGRegistry):
        self.registry = registry
        self.reranker = registry.reranker
        self.entity_extractor = getattr(registry, "entity_extractor", None)

    def _get_dense(self, corpus: str):
        getter = getattr(self.registry, "get_async_dense", None)
        if callable(getter):
            return getter(corpus)
        return self.registry.get_dense(corpus)

    def _get_hybrid(self, corpus: str):
        getter = getattr(self.registry, "get_async_hybrid", None)
        if callable(getter):
            return getter(corpus)
        return self.registry.get_hybrid(corpus)

    def _get_graphrag(self, corpus: str):
        getter = getattr(self.registry, "get_async_graphrag", None)
        if callable(getter):
            return getter(corpus)
        return self.registry.get_graphrag(corpus)

    def _get_kg_backend(self, corpus: str):
        getter = getattr(self.registry, "get_async_kg_backend", None)
        if callable(getter):
            return getter(corpus)
        if hasattr(self.registry, "get_kg_backend"):
            return self.registry.get_kg_backend(corpus)
        return getattr(self.registry, "kg_backend", None)

    async def _arerank(
        self,
        query: str,
        candidates: List[ScoredChunk],
        k: int,
    ) -> List[ScoredChunk]:
        async_rerank = getattr(self.reranker, "arerank", None)
        if callable(async_rerank):
            return await async_rerank(query, candidates, k=k)
        return await asyncio.to_thread(self.reranker.rerank, query, candidates, k)

    async def _aretrieve(self, retriever, *args, **kwargs):
        method = getattr(retriever, "retrieve")
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return await asyncio.to_thread(method, *args, **kwargs)

    async def _aretrieve_with_debug(self, retriever, *args, **kwargs):
        method = getattr(retriever, "retrieve_with_debug")
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        return await asyncio.to_thread(method, *args, **kwargs)

    async def _aneighborhood(
        self,
        kg_backend,
        ent: str,
        graph_hops: int,
    ) -> List[Dict[str, Any]]:
        async_neighborhood = getattr(kg_backend, "aneighborhood", None)
        if callable(async_neighborhood):
            return await async_neighborhood(ent, hops=graph_hops, limit=10)
        return await asyncio.to_thread(kg_backend.neighborhood, ent, graph_hops, 10)

    async def retrieve(
        self,
        corpus: str,
        query: str,
        mode: RetrievalMode = "hybrid+rerank",
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        if mode == "dense":
            base = await self._aretrieve(
                self._get_dense(corpus),
                query,
                k=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return await self._arerank(query, base, k=k)

        if mode == "hybrid":
            base = await self._aretrieve(
                self._get_hybrid(corpus),
                query,
                k_vector=k,
                k_lexical=k,
                k_final=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return base

        if mode == "hybrid+rerank":
            base = await self._aretrieve(
                self._get_hybrid(corpus),
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=3 * k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return await self._arerank(query, base, k=k)

        if mode == "graphrag":
            gr = self._get_graphrag(corpus)
            k_base = 2 * k
            k_graph = 4 * k
            base = await self._aretrieve(
                gr,
                query,
                k_final=k_base,
                max_expanded=k_graph,
                **_metadata_filter_kwargs(metadata_filter),
            )
            return await self._arerank(query, base, k=k)

        raise ValueError(f"Unknown retrieval mode: {mode}")

    async def retrieve_with_debug(
        self,
        corpus: str,
        query: str,
        mode: RetrievalMode = "hybrid+rerank",
        k: int = 10,
        metadata_filter: MetadataFilter | None = None,
    ):
        debug: Dict[str, Any] = {
            "query": query,
            "mode": mode,
        }

        if mode == "dense":
            base = await self._aretrieve(
                self._get_dense(corpus),
                query,
                k=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            final = await self._arerank(query, base, k=k)
            debug["base_candidates"] = base
            debug["reranked"] = final
            return final, debug

        if mode == "hybrid":
            base = await self._aretrieve(
                self._get_hybrid(corpus),
                query,
                k_vector=k,
                k_lexical=k,
                k_final=k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            debug["base_candidates"] = base
            return base, debug

        if mode == "hybrid+rerank":
            base = await self._aretrieve(
                self._get_hybrid(corpus),
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=3 * k,
                **_metadata_filter_kwargs(metadata_filter),
            )
            final = await self._arerank(query, base, k=k)
            debug["base_candidates"] = base
            debug["reranked"] = final
            return final, debug

        if mode == "graphrag":
            gr = self._get_graphrag(corpus)
            kg_backend = self._get_kg_backend(corpus)
            graph_hops = getattr(gr, "_hops", 2)
            k_base = 2 * k
            k_graph = 4 * k
            base, gr_debug = await self._aretrieve_with_debug(
                gr,
                query,
                k_final=k_base,
                max_expanded=k_graph,
                **_metadata_filter_kwargs(metadata_filter),
            )
            final = await self._arerank(query, base, k=k)
            debug["base_candidates"] = base
            debug["reranked"] = final
            debug["graphrag"] = gr_debug

            if kg_backend:
                entities = []
                if isinstance(gr_debug, dict):
                    entities = gr_debug.get("seed_entities") or []
                neighborhoods = {}
                for ent in entities:
                    neighborhoods[ent] = await self._aneighborhood(
                        kg_backend,
                        ent,
                        graph_hops,
                    )
                result_entities = []
                for c in base:
                    if isinstance(c.metadata, dict):
                        ent_meta = c.metadata.get("entity_ids")
                        if isinstance(ent_meta, str):
                            result_entities.extend(
                                [e for e in ent_meta.split(",") if e]
                            )
                        elif isinstance(ent_meta, list):
                            result_entities.extend(ent_meta)
                for ent in set(result_entities):
                    neighborhoods[ent] = await self._aneighborhood(
                        kg_backend,
                        ent,
                        graph_hops,
                    )
                debug["kg_neighborhood"] = neighborhoods
            return final, debug

        raise ValueError(f"Unknown retrieval mode: {mode}")
