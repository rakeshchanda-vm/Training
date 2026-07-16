
from __future__ import annotations

import asyncio
from typing import Callable, Dict, List, Optional, Set

from andromeda.retrievers.core import DocumentStoreBackend
from andromeda.retrievers.core import KnowledgeGraphBackend, MetadataFilter, ScoredChunk
from andromeda.retrievers.core import metadata_matches_filter
from andromeda.retrievers.retrievers import (
    AsyncDenseRetriever,
    AsyncHybridRetriever,
    DenseRetriever,
    HybridRetriever,
)
from andromeda.retrievers.kg import _is_valid_entity_name, normalize_entity_name


EntityExtractor = Callable[[str], List[str]]


class GraphRAGRetriever:
    def __init__(
        self,
        base_retriever: HybridRetriever | DenseRetriever,
        kg: KnowledgeGraphBackend,
        entity_extractor: EntityExtractor,
        docstore: Optional[DocumentStoreBackend] = None,
        max_neighbors: int = 50,
        hops: int = 2,
        max_query_entities: int = 12,
    ):
        self._base = base_retriever
        self._kg = kg
        self._extract_entities = entity_extractor
        self._docstore = docstore
        self._max_neighbors = max_neighbors
        self._hops = hops
        self._max_query_entities = max_query_entities

    def _base_retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        if isinstance(self._base, HybridRetriever):
            # Keep hybrid recall high; GraphRAG will add additional candidates.
            return self._base.retrieve(
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=k,
                metadata_filter=metadata_filter,
            )
        return self._base.retrieve(query, k=k, metadata_filter=metadata_filter)

    def _retrieve_internal(
        self,
        query: str,
        k_final: int,
        max_expanded: int,
        metadata_filter: MetadataFilter | None = None,
    ):
        base_results = self._base_retrieve(
            query,
            k=k_final,
            metadata_filter=metadata_filter,
        )

        seeds: List[str] = []
        try:
            seeds.extend(self._extract_entities(query) or [])
        except Exception:
            seeds = []

        # Add entities already computed at ingest time.
        for c in base_results:
            meta_ents = (
                c.metadata.get("entity_ids")
                if isinstance(c.metadata, dict)
                else None
            )
            if isinstance(meta_ents, str) and meta_ents:
                seeds.extend([e for e in meta_ents.split(",") if e])
            elif isinstance(meta_ents, list):
                seeds.extend([str(e) for e in meta_ents if e])

        seen_entities: Set[str] = set()
        unique_entities: List[str] = []
        for ent in seeds:
            ent_str = str(ent).strip()
            if not ent_str or not _is_valid_entity_name(ent_str):
                continue
            key = normalize_entity_name(ent_str)
            if not key or key in seen_entities:
                continue
            seen_entities.add(key)
            unique_entities.append(ent_str)
            if len(unique_entities) >= self._max_query_entities:
                break

        expanded_ids: List[str] = []
        expanded_meta: Dict[str, Dict] = {}
        traversal_edges: List[Dict[str, object]] = []

        for ent in unique_entities:
            neigh = self._kg.neighborhood(
                ent,
                hops=self._hops,
                limit=self._max_neighbors,
            )
            for node in neigh:
                src = node.get("source")
                pred = node.get("predicate")
                dst = node.get("id")
                if src and pred and dst:
                    traversal_edges.append(
                        {
                            "source": src,
                            "predicate": pred,
                            "id": dst,
                            "depth": node.get("depth"),
                            "direction": node.get("direction"),
                        }
                    )
            for node in neigh:
                pred = str(node.get("predicate", "")).lower()
                # Expand only incoming mention edges to chunk ids.
                if pred != "inv:mentions":
                    continue
                chunk_id = str(node.get("id") or "")
                if not chunk_id or "::chunk-" not in chunk_id:
                    continue
                if chunk_id in expanded_meta:
                    continue
                meta = dict(node)
                meta.setdefault("kg_seed_entity", ent)
                expanded_meta[chunk_id] = meta
                expanded_ids.append(chunk_id)
                if len(expanded_ids) >= max_expanded:
                    break
            if len(expanded_ids) >= max_expanded:
                break

        fetched = (
            self._docstore.get_documents(expanded_ids)
            if (self._docstore and expanded_ids)
            else {}
        )
        expanded_chunks: List[ScoredChunk] = []
        for chunk_id in expanded_ids:
            node = expanded_meta.get(chunk_id, {})
            doc = fetched.get(chunk_id)
            text = ""
            meta = {k: v for k, v in node.items() if k not in {"snippet"}}
            meta["retrieval_source"] = "kg"

            if doc is not None:
                if not metadata_matches_filter(doc.metadata, metadata_filter):
                    continue
                text = doc.text
                meta = dict(doc.metadata or {}) | meta
            else:
                if metadata_filter and not metadata_matches_filter(
                    meta,
                    metadata_filter,
                ):
                    continue
                text = str(node.get("snippet") or "")
                if not text:
                    continue

            expanded_chunks.append(
                ScoredChunk(
                    doc_id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=0.0,
                )
            )

        merged: Dict[str, ScoredChunk] = {}
        positions: Dict[str, int] = {}
        ordered: List[ScoredChunk] = []
        for c in base_results + expanded_chunks:
            if not c.doc_id:
                continue
            if c.doc_id in merged:
                existing = merged[c.doc_id]
                existing_meta = dict(existing.metadata or {})
                new_meta = dict(c.metadata or {})

                def _sources(meta: Dict[str, object]) -> Set[str]:
                    val = meta.get("retrieval_source")
                    if isinstance(val, str) and val:
                        return {s for s in val.split("+") if s}
                    return set()

                sources = _sources(existing_meta) | _sources(new_meta)
                if sources:
                    existing_meta["retrieval_source"] = "+".join(sorted(sources))

                # Preserve kg attribution if it exists.
                for k in ("kg_seed_entity", "depth", "predicate"):
                    if k in new_meta and k not in existing_meta:
                        existing_meta[k] = new_meta[k]

                updated = ScoredChunk(
                    doc_id=existing.doc_id,
                    text=existing.text,
                    metadata=existing_meta,
                    score=existing.score,
                )
                merged[c.doc_id] = updated
                pos = positions.get(c.doc_id)
                if pos is not None:
                    ordered[pos] = updated
                continue

            merged[c.doc_id] = c
            positions[c.doc_id] = len(ordered)
            ordered.append(c)

        # Build directed adjacency from neighborhood edges for path reconstruction.
        adjacency: Dict[str, List[tuple[str, str]]] = {}
        for e in traversal_edges:
            src = str(e.get("source") or "")
            dst = str(e.get("id") or "")
            pred = str(e.get("predicate") or "")
            if not src or not dst or not pred:
                continue
            adjacency.setdefault(src, []).append((dst, pred))

        def _find_path(seed: str, target: str, max_steps: int = 6):
            # BFS shortest path on the local neighborhood edges.
            from collections import deque

            q = deque([seed])
            prev: Dict[str, tuple[str, str]] = {}  # node -> (prev_node, predicate)
            seen: Set[str] = {seed}
            while q:
                cur = q.popleft()
                if cur == target:
                    break
                if len(prev) > 2000:  # guardrail
                    break
                for nxt, pred in adjacency.get(cur, []):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    prev[nxt] = (cur, pred)
                    q.append(nxt)

            if target not in prev and target != seed:
                return []

            # Reconstruct.
            steps: List[Dict[str, str]] = []
            node = target
            if node == seed:
                return steps
            while node != seed:
                p = prev.get(node)
                if not p:
                    return []
                parent, pred = p
                steps.append({"from": parent, "predicate": pred, "to": node})
                node = parent
                if len(steps) >= max_steps:
                    break
            steps.reverse()
            return steps

        expanded_paths: Dict[str, List[Dict[str, str]]] = {}
        for cid in expanded_ids:
            seed = str(expanded_meta.get(cid, {}).get("kg_seed_entity") or "")
            if not seed:
                continue
            expanded_paths[cid] = _find_path(seed, cid)

        debug = {
            "seed_entities": unique_entities,
            "base_count": len(base_results),
            "expanded_count": len(expanded_chunks),
            "expanded_chunk_ids": expanded_ids,
            "expanded_sample": [
                {
                    "chunk_id": cid,
                    "seed": expanded_meta.get(cid, {}).get("kg_seed_entity"),
                    "depth": expanded_meta.get(cid, {}).get("depth"),
                    "predicate": expanded_meta.get(cid, {}).get("predicate"),
                }
                for cid in expanded_ids[:10]
            ],
            "expanded_paths": expanded_paths,
            "traversal_edges_sample": traversal_edges[:50],
            "returned_count": len(ordered),
        }
        return ordered, debug

    def retrieve(
        self,
        query: str,
        k_final: int = 20,
        max_expanded: Optional[int] = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        max_expanded_int = max_expanded if max_expanded is not None else k_final
        results, _debug = self._retrieve_internal(
            query=query,
            k_final=k_final,
            max_expanded=max_expanded_int,
            metadata_filter=metadata_filter,
        )
        return results

    def retrieve_with_debug(
        self,
        query: str,
        k_final: int = 20,
        max_expanded: Optional[int] = None,
        metadata_filter: MetadataFilter | None = None,
    ):
        max_expanded_int = max_expanded if max_expanded is not None else k_final
        results, debug = self._retrieve_internal(
            query=query,
            k_final=k_final,
            max_expanded=max_expanded_int,
            metadata_filter=metadata_filter,
        )
        debug = dict(debug)
        debug.update(
            {
                "query": query,
                "k_final": k_final,
                "max_expanded": max_expanded_int,
            }
        )
        return results, debug


async def _aget_documents(
    docstore: Optional[DocumentStoreBackend],
    ids: List[str],
) -> Dict[str, object]:
    if docstore is None or not ids:
        return {}
    async_get = getattr(docstore, "aget_documents", None)
    if callable(async_get):
        return await async_get(ids)
    return await asyncio.to_thread(docstore.get_documents, ids)


async def _aneighborhood(
    kg: KnowledgeGraphBackend,
    node_id: str,
    hops: int,
    limit: int,
) -> List[Dict[str, object]]:
    async_neighborhood = getattr(kg, "aneighborhood", None)
    if callable(async_neighborhood):
        return await async_neighborhood(node_id, hops=hops, limit=limit)
    return await asyncio.to_thread(kg.neighborhood, node_id, hops, limit)


class AsyncGraphRAGRetriever:
    def __init__(
        self,
        base_retriever: (
            AsyncHybridRetriever
            | AsyncDenseRetriever
            | HybridRetriever
            | DenseRetriever
        ),
        kg: KnowledgeGraphBackend,
        entity_extractor: EntityExtractor,
        docstore: Optional[DocumentStoreBackend] = None,
        max_neighbors: int = 50,
        hops: int = 2,
        max_query_entities: int = 12,
    ):
        self._base = base_retriever
        self._kg = kg
        self._extract_entities = entity_extractor
        self._docstore = docstore
        self._max_neighbors = max_neighbors
        self._hops = hops
        self._max_query_entities = max_query_entities

    async def _aextract_entities(self, query: str) -> List[str]:
        extractor = self._extract_entities
        if asyncio.iscoroutinefunction(extractor):
            return list(await extractor(query) or [])
        return list(await asyncio.to_thread(extractor, query) or [])

    async def _base_retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        if isinstance(self._base, AsyncHybridRetriever):
            return await self._base.retrieve(
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=k,
                metadata_filter=metadata_filter,
            )
        if isinstance(self._base, HybridRetriever):
            return await asyncio.to_thread(
                self._base.retrieve,
                query,
                k_vector=3 * k,
                k_lexical=3 * k,
                k_final=k,
                metadata_filter=metadata_filter,
            )
        if isinstance(self._base, AsyncDenseRetriever):
            return await self._base.retrieve(
                query,
                k=k,
                metadata_filter=metadata_filter,
            )
        return await asyncio.to_thread(
            self._base.retrieve,
            query,
            k,
            metadata_filter=metadata_filter,
        )

    async def _retrieve_internal(
        self,
        query: str,
        k_final: int,
        max_expanded: int,
        metadata_filter: MetadataFilter | None = None,
    ):
        base_results = await self._base_retrieve(
            query,
            k=k_final,
            metadata_filter=metadata_filter,
        )

        seeds: List[str] = []
        try:
            seeds.extend(await self._aextract_entities(query))
        except Exception:
            seeds = []

        for c in base_results:
            meta_ents = (
                c.metadata.get("entity_ids")
                if isinstance(c.metadata, dict)
                else None
            )
            if isinstance(meta_ents, str) and meta_ents:
                seeds.extend([e for e in meta_ents.split(",") if e])
            elif isinstance(meta_ents, list):
                seeds.extend([str(e) for e in meta_ents if e])

        seen_entities: Set[str] = set()
        unique_entities: List[str] = []
        for ent in seeds:
            ent_str = str(ent).strip()
            if not ent_str or not _is_valid_entity_name(ent_str):
                continue
            key = normalize_entity_name(ent_str)
            if not key or key in seen_entities:
                continue
            seen_entities.add(key)
            unique_entities.append(ent_str)
            if len(unique_entities) >= self._max_query_entities:
                break

        expanded_ids: List[str] = []
        expanded_meta: Dict[str, Dict] = {}
        traversal_edges: List[Dict[str, object]] = []

        for ent in unique_entities:
            neigh = await _aneighborhood(
                self._kg,
                ent,
                hops=self._hops,
                limit=self._max_neighbors,
            )
            for node in neigh:
                src = node.get("source")
                pred = node.get("predicate")
                dst = node.get("id")
                if src and pred and dst:
                    traversal_edges.append(
                        {
                            "source": src,
                            "predicate": pred,
                            "id": dst,
                            "depth": node.get("depth"),
                            "direction": node.get("direction"),
                        }
                    )
            for node in neigh:
                pred = str(node.get("predicate", "")).lower()
                if pred != "inv:mentions":
                    continue
                chunk_id = str(node.get("id") or "")
                if not chunk_id or "::chunk-" not in chunk_id:
                    continue
                if chunk_id in expanded_meta:
                    continue
                meta = dict(node)
                meta.setdefault("kg_seed_entity", ent)
                expanded_meta[chunk_id] = meta
                expanded_ids.append(chunk_id)
                if len(expanded_ids) >= max_expanded:
                    break
            if len(expanded_ids) >= max_expanded:
                break

        fetched = await _aget_documents(self._docstore, expanded_ids)
        expanded_chunks: List[ScoredChunk] = []
        for chunk_id in expanded_ids:
            node = expanded_meta.get(chunk_id, {})
            doc = fetched.get(chunk_id)
            text = ""
            meta = {k: v for k, v in node.items() if k not in {"snippet"}}
            meta["retrieval_source"] = "kg"

            if doc is not None:
                if not metadata_matches_filter(doc.metadata, metadata_filter):
                    continue
                text = doc.text
                meta = dict(doc.metadata or {}) | meta
            else:
                if metadata_filter and not metadata_matches_filter(
                    meta,
                    metadata_filter,
                ):
                    continue
                text = str(node.get("snippet") or "")
                if not text:
                    continue

            expanded_chunks.append(
                ScoredChunk(
                    doc_id=chunk_id,
                    text=text,
                    metadata=meta,
                    score=0.0,
                )
            )

        merged: Dict[str, ScoredChunk] = {}
        positions: Dict[str, int] = {}
        ordered: List[ScoredChunk] = []
        for c in base_results + expanded_chunks:
            if not c.doc_id:
                continue
            if c.doc_id in merged:
                existing = merged[c.doc_id]
                existing_meta = dict(existing.metadata or {})
                new_meta = dict(c.metadata or {})

                def _sources(meta: Dict[str, object]) -> Set[str]:
                    val = meta.get("retrieval_source")
                    if isinstance(val, str) and val:
                        return {s for s in val.split("+") if s}
                    return set()

                sources = _sources(existing_meta) | _sources(new_meta)
                if sources:
                    existing_meta["retrieval_source"] = "+".join(sorted(sources))

                for k in ("kg_seed_entity", "depth", "predicate"):
                    if k in new_meta and k not in existing_meta:
                        existing_meta[k] = new_meta[k]

                updated = ScoredChunk(
                    doc_id=existing.doc_id,
                    text=existing.text,
                    metadata=existing_meta,
                    score=existing.score,
                )
                merged[c.doc_id] = updated
                pos = positions.get(c.doc_id)
                if pos is not None:
                    ordered[pos] = updated
                continue

            merged[c.doc_id] = c
            positions[c.doc_id] = len(ordered)
            ordered.append(c)

        adjacency: Dict[str, List[tuple[str, str]]] = {}
        for e in traversal_edges:
            src = str(e.get("source") or "")
            dst = str(e.get("id") or "")
            pred = str(e.get("predicate") or "")
            if not src or not dst or not pred:
                continue
            adjacency.setdefault(src, []).append((dst, pred))

        def _find_path(seed: str, target: str, max_steps: int = 6):
            from collections import deque

            q = deque([seed])
            prev: Dict[str, tuple[str, str]] = {}
            seen: Set[str] = {seed}
            while q:
                cur = q.popleft()
                if cur == target:
                    break
                if len(prev) > 2000:
                    break
                for nxt, pred in adjacency.get(cur, []):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    prev[nxt] = (cur, pred)
                    q.append(nxt)

            if target not in prev and target != seed:
                return []

            steps: List[Dict[str, str]] = []
            node = target
            if node == seed:
                return steps
            while node != seed:
                p = prev.get(node)
                if not p:
                    return []
                parent, pred = p
                steps.append({"from": parent, "predicate": pred, "to": node})
                node = parent
                if len(steps) >= max_steps:
                    break
            steps.reverse()
            return steps

        expanded_paths: Dict[str, List[Dict[str, str]]] = {}
        for cid in expanded_ids:
            seed = str(expanded_meta.get(cid, {}).get("kg_seed_entity") or "")
            if not seed:
                continue
            expanded_paths[cid] = _find_path(seed, cid)

        debug = {
            "seed_entities": unique_entities,
            "base_count": len(base_results),
            "expanded_count": len(expanded_chunks),
            "expanded_chunk_ids": expanded_ids,
            "expanded_sample": [
                {
                    "chunk_id": cid,
                    "seed": expanded_meta.get(cid, {}).get("kg_seed_entity"),
                    "depth": expanded_meta.get(cid, {}).get("depth"),
                    "predicate": expanded_meta.get(cid, {}).get("predicate"),
                }
                for cid in expanded_ids[:10]
            ],
            "expanded_paths": expanded_paths,
            "traversal_edges_sample": traversal_edges[:50],
            "returned_count": len(ordered),
        }
        return ordered, debug

    async def retrieve(
        self,
        query: str,
        k_final: int = 20,
        max_expanded: Optional[int] = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> List[ScoredChunk]:
        max_expanded_int = max_expanded if max_expanded is not None else k_final
        results, _debug = await self._retrieve_internal(
            query=query,
            k_final=k_final,
            max_expanded=max_expanded_int,
            metadata_filter=metadata_filter,
        )
        return results

    async def retrieve_with_debug(
        self,
        query: str,
        k_final: int = 20,
        max_expanded: Optional[int] = None,
        metadata_filter: MetadataFilter | None = None,
    ):
        max_expanded_int = max_expanded if max_expanded is not None else k_final
        results, debug = await self._retrieve_internal(
            query=query,
            k_final=k_final,
            max_expanded=max_expanded_int,
            metadata_filter=metadata_filter,
        )
        debug = dict(debug)
        debug.update(
            {
                "query": query,
                "k_final": k_final,
                "max_expanded": max_expanded_int,
            }
        )
        return results, debug
