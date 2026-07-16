
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
from typing import Iterable, Callable, List, Dict, Any, Optional

from andromeda.retrievers.core import Document, KnowledgeGraphFactRecord
from andromeda.retrievers.config import RAGRegistry
from andromeda.retrievers.ingestion_index import SqliteIngestionIndex, sha256_text
from andromeda.retrievers.kg import KGExtraction, normalize_entity_name, _is_valid_entity_name
from andromeda.retrievers.processing import DocumentProcessingEngine, RawDocument


Chunker = Callable[[str, Dict[str, Any]], List[Document]]
KGExtractor = Callable[[Document], List[tuple[str, str, str, Dict[str, Any]]]]
KGBundleExtractor = Callable[[Document], KGExtraction | Dict[str, Any]]
_INDEX_FLUSH_CHUNK_COUNT = 100


def _coerce_raw(doc: RawDocument | Document) -> RawDocument:
    if isinstance(doc, RawDocument):
        return doc
    return RawDocument(id=doc.id, text=doc.text, metadata=doc.metadata)


def ingest_corpus(
    registry: RAGRegistry,
    corpus_name: str,
    raw_docs: Iterable[RawDocument | Document],
    processor: DocumentProcessingEngine | None = None,
    chunker: Chunker | None = None,
    kg_extractor: KGExtractor | None = None,
    kg_bundle_extractor: KGBundleExtractor | None = None,
    kg_chunks_per_doc: Optional[int] = 3,
    entity_extractor: Optional[Callable[[str], List[str]]] = None,
    prefer_bundle_entities: bool = True,
    prune_missing_sources: bool = False,
    *,
    kg_parallelism: int = 1,
) -> List[Document]:
    """
    Ingests an iterable of raw docs using either a processing engine or a simple chunker.

    - processor: production-ready pipeline (normalization, chunking, dedupe).
    - chunker: legacy/simple chunker kept for compatibility.
    """
    corpus_cfg = registry.config.corpora[corpus_name]
    vb = registry.vector_backends[corpus_name]
    lb = registry.lex_backends.get(corpus_name)
    kg = (
        registry.get_kg_backend(corpus_name)  # type: ignore[attr-defined]
        if corpus_cfg.enable_graph and hasattr(registry, "get_kg_backend")
        else registry.kg_backend if corpus_cfg.enable_graph else None
    )
    docstore = getattr(registry, "docstores", {}).get(corpus_name)
    index: SqliteIngestionIndex | None = getattr(registry, "ingestion_indexes", {}).get(corpus_name)

    if kg_extractor and kg_bundle_extractor:
        raise ValueError("Provide either kg_extractor or kg_bundle_extractor, not both")

    docs_list: List[RawDocument | Document] = list(raw_docs)
    all_chunks: List[Document] = []
    entity_fn = entity_extractor or getattr(registry, "entity_extractor", None)

    if processor:
        raw_inputs: List[RawDocument] = [_coerce_raw(d) for d in docs_list]
        raw_by_source_id = {r.id: r for r in raw_inputs}
        all_chunks = processor.process(raw_inputs)
    elif chunker:
        raw_by_source_id = {}
        for doc in docs_list:
            raw = _coerce_raw(doc)
            raw_by_source_id[raw.id] = raw
            all_chunks.extend(chunker(raw.text, raw.metadata))
    else:
        raise ValueError("Provide either a processor or chunker")

    # If configured, support full delta sync: add/update/delete by source_id.
    # Otherwise fall back to always-upsert.
    by_source: Dict[str, List[Document]] = {}
    for c in all_chunks:
        src = c.metadata.get("source_id") or c.id.split("::")[0]
        by_source.setdefault(src, []).append(c)

    if index and prune_missing_sources:
        incoming_sources = set(by_source.keys())
        for old_source_id in index.list_sources():
            if old_source_id not in incoming_sources:
                old_state = index.get_source_state(old_source_id)
                if old_state:
                    _delete_chunk_ids(
                        chunk_ids=list(old_state.chunks.keys()),
                        vb=vb,
                        lb=lb,
                        kg=kg,
                        docstore=docstore,
                    )
                index.delete_source(old_source_id)

    if not all_chunks:
        return []

    # Delta-ingest per source to avoid duplicates and to delete stale chunks.
    pending_index_chunks: List[Document] = []
    for src, chunks in by_source.items():
        doc = raw_by_source_id.get(src)
        doc_hash = sha256_text(doc.text) if doc else sha256_text("".join(c.text for c in chunks))
        new_hashes = {c.id: sha256_text(c.text) for c in chunks}
        bundle_entities_by_chunk: Dict[str, List[str]] = {}
        pending_facts: List[KnowledgeGraphFactRecord] = []

        old_state = index.get_source_state(src) if index else None
        old_hashes = old_state.chunks if old_state else {}

        deleted_ids = [cid for cid in old_hashes.keys() if cid not in new_hashes]
        updated_ids = [cid for cid, h in new_hashes.items() if old_hashes.get(cid) and old_hashes.get(cid) != h]

        if deleted_ids or updated_ids:
            _delete_chunk_ids(
                chunk_ids=list(dict.fromkeys(deleted_ids + updated_ids)),
                vb=vb,
                lb=lb,
                kg=kg,
                docstore=docstore,
            )

        # Only (re)process chunks that are new or changed.
        changed = [c for c in chunks if (c.id not in old_hashes) or (c.id in updated_ids)]
        if not changed:
            if index:
                index.upsert_source(src, doc_hash=doc_hash, chunks=new_hashes)
            continue

        if kg and hasattr(kg, "delete_facts_for_chunks"):
            try:
                kg.delete_facts_for_chunks([c.id for c in changed])  # type: ignore[attr-defined]
            except Exception:
                pass
        elif kg and hasattr(kg, "delete_facts_for_chunk"):
            for c in changed:
                try:
                    kg.delete_facts_for_chunk(c.id)  # type: ignore[attr-defined]
                except Exception:
                    pass

        # Extract KG facts for a limited number of chunks per source.
        if kg and kg_extractor:
            limited_chunks = changed[: (kg_chunks_per_doc or len(changed))]
            for chunk, facts in _parallel_map_chunks(
                limited_chunks,
                kg_extractor,
                parallelism=kg_parallelism,
            ):
                for fact in facts:
                    subj, pred, obj, fmeta = fact
                    pending_facts.append(
                        KnowledgeGraphFactRecord(
                            subject=subj,
                            predicate=pred,
                            object=obj,
                            metadata=dict(fmeta or {}),
                        )
                    )

        if kg and kg_bundle_extractor:
            limited_chunks = changed[: (kg_chunks_per_doc or len(changed))]
            for chunk, extracted in _parallel_map_chunks(
                limited_chunks,
                kg_bundle_extractor,
                parallelism=kg_parallelism,
            ):
                extracted = extracted or {}
                entities = []
                triples = []
                if isinstance(extracted, KGExtraction):
                    entities = extracted.entities or []
                    triples = extracted.triples or []
                elif isinstance(extracted, dict):
                    entities = extracted.get("entities") or extracted.get("entity_ids") or []
                    triples = extracted.get("triples") or extracted.get("relations") or []
                if isinstance(entities, list):
                    bundle_entities_by_chunk[chunk.id] = [str(e) for e in entities]
                for t in triples:
                    try:
                        subj, pred, obj = t
                    except Exception:
                        continue
                    pending_facts.append(
                        KnowledgeGraphFactRecord(
                            subject=str(subj),
                            predicate=str(pred),
                            object=str(obj),
                            metadata={
                                "doc_id": chunk.id,
                                "source_id": chunk.metadata.get("source_id", ""),
                                "snippet": chunk.text[:200],
                            },
                        )
                    )

        # Attach entity_ids metadata and mention edges before indexing.
        if entity_fn or bundle_entities_by_chunk:
            for chunk in changed:
                entities = []
                if prefer_bundle_entities and chunk.id in bundle_entities_by_chunk:
                    entities = bundle_entities_by_chunk.get(chunk.id, []) or []
                else:
                    entities = entity_fn(chunk.text) if entity_fn else []
                norm_entities: List[str] = []
                seen_e: set[str] = set()
                for e in entities:
                    if not _is_valid_entity_name(e):
                        continue
                    norm = normalize_entity_name(e)
                    if not norm or norm in seen_e:
                        continue
                    seen_e.add(norm)
                    norm_entities.append(norm)
                    if kg:
                        pending_facts.append(
                            KnowledgeGraphFactRecord(
                                subject=chunk.id,
                                predicate="mentions",
                                object=norm,
                                metadata={
                                    "source_id": chunk.metadata.get("source_id", ""),
                                    "snippet": chunk.text[:200],
                                },
                            )
                        )
                if norm_entities:
                    chunk.metadata = dict(chunk.metadata)
                    chunk.metadata["entity_ids"] = ",".join(norm_entities)

        if kg and pending_facts:
            if hasattr(kg, "upsert_facts"):
                kg.upsert_facts(pending_facts)  # type: ignore[attr-defined]
            else:
                for fact in pending_facts:
                    kg.upsert_fact(fact.subject, fact.predicate, fact.object, fact.metadata)
        if index:
            index.upsert_source(src, doc_hash=doc_hash, chunks=new_hashes)
        pending_index_chunks.extend(changed)
        if len(pending_index_chunks) >= _INDEX_FLUSH_CHUNK_COUNT:
            _flush_changed_chunks(
                changed=pending_index_chunks,
                vb=vb,
                lb=lb,
                docstore=docstore,
            )
            pending_index_chunks = []

    _flush_changed_chunks(
        changed=pending_index_chunks,
        vb=vb,
        lb=lb,
        docstore=docstore,
    )

    return all_chunks


async def aingest_corpus(
    registry: RAGRegistry,
    corpus_name: str,
    raw_docs: Iterable[RawDocument | Document],
    processor: DocumentProcessingEngine | None = None,
    chunker: Chunker | None = None,
    kg_extractor: KGExtractor | None = None,
    kg_bundle_extractor: KGBundleExtractor | None = None,
    kg_chunks_per_doc: Optional[int] = 3,
    entity_extractor: Optional[Callable[[str], List[str]]] = None,
    prefer_bundle_entities: bool = True,
    prune_missing_sources: bool = False,
    *,
    kg_parallelism: int = 1,
) -> List[Document]:
    corpus_cfg = registry.config.corpora[corpus_name]
    vb = getattr(registry, "async_vector_backends", {}).get(corpus_name) or registry.vector_backends[corpus_name]
    lb = registry.lex_backends.get(corpus_name)
    kg = (
        registry.get_async_kg_backend(corpus_name)  # type: ignore[attr-defined]
        if corpus_cfg.enable_graph and hasattr(registry, "get_async_kg_backend")
        else registry.get_kg_backend(corpus_name) if corpus_cfg.enable_graph and hasattr(registry, "get_kg_backend")
        else registry.kg_backend if corpus_cfg.enable_graph else None
    )
    docstore = getattr(registry, "async_docstores", {}).get(corpus_name) or getattr(registry, "docstores", {}).get(corpus_name)
    index = getattr(registry, "async_ingestion_indexes", {}).get(corpus_name) or getattr(registry, "ingestion_indexes", {}).get(corpus_name)

    if kg_extractor and kg_bundle_extractor:
        raise ValueError("Provide either kg_extractor or kg_bundle_extractor, not both")

    docs_list: List[RawDocument | Document] = list(raw_docs)
    all_chunks: List[Document] = []
    entity_fn = entity_extractor or getattr(registry, "entity_extractor", None)

    if processor:
        raw_inputs: List[RawDocument] = [_coerce_raw(d) for d in docs_list]
        raw_by_source_id = {r.id: r for r in raw_inputs}
        all_chunks = await asyncio.to_thread(processor.process, raw_inputs)
    elif chunker:
        raw_by_source_id = {}
        for doc in docs_list:
            raw = _coerce_raw(doc)
            raw_by_source_id[raw.id] = raw
            all_chunks.extend(await _acall_chunk_builder(chunker, raw.text, raw.metadata))
    else:
        raise ValueError("Provide either a processor or chunker")

    by_source: Dict[str, List[Document]] = {}
    for c in all_chunks:
        src = c.metadata.get("source_id") or c.id.split("::")[0]
        by_source.setdefault(src, []).append(c)

    if index and prune_missing_sources:
        incoming_sources = set(by_source.keys())
        for old_source_id in await _aindex_list_sources(index):
            if old_source_id not in incoming_sources:
                old_state = await _aindex_get_source_state(index, old_source_id)
                if old_state:
                    await _adelete_chunk_ids(
                        chunk_ids=list(old_state.chunks.keys()),
                        vb=vb,
                        lb=lb,
                        kg=kg,
                        docstore=docstore,
                    )
                await _aindex_delete_source(index, old_source_id)

    if not all_chunks:
        return []

    pending_index_chunks: List[Document] = []
    for src, chunks in by_source.items():
        doc = raw_by_source_id.get(src)
        doc_hash = sha256_text(doc.text) if doc else sha256_text("".join(c.text for c in chunks))
        new_hashes = {c.id: sha256_text(c.text) for c in chunks}
        bundle_entities_by_chunk: Dict[str, List[str]] = {}
        pending_facts: List[KnowledgeGraphFactRecord] = []

        old_state = await _aindex_get_source_state(index, src) if index else None
        old_hashes = old_state.chunks if old_state else {}

        deleted_ids = [cid for cid in old_hashes.keys() if cid not in new_hashes]
        updated_ids = [cid for cid, h in new_hashes.items() if old_hashes.get(cid) and old_hashes.get(cid) != h]

        if deleted_ids or updated_ids:
            await _adelete_chunk_ids(
                chunk_ids=list(dict.fromkeys(deleted_ids + updated_ids)),
                vb=vb,
                lb=lb,
                kg=kg,
                docstore=docstore,
            )

        changed = [c for c in chunks if (c.id not in old_hashes) or (c.id in updated_ids)]
        if not changed:
            if index:
                await _aindex_upsert_source(index, src, doc_hash=doc_hash, chunks=new_hashes)
            continue

        if kg and hasattr(kg, "adelete_facts_for_chunks"):
            try:
                await kg.adelete_facts_for_chunks([c.id for c in changed])  # type: ignore[attr-defined]
            except Exception:
                pass
        elif kg and hasattr(kg, "delete_facts_for_chunks"):
            try:
                await asyncio.to_thread(kg.delete_facts_for_chunks, [c.id for c in changed])  # type: ignore[attr-defined]
            except Exception:
                pass
        elif kg and hasattr(kg, "delete_facts_for_chunk"):
            for c in changed:
                try:
                    await asyncio.to_thread(kg.delete_facts_for_chunk, c.id)  # type: ignore[attr-defined]
                except Exception:
                    pass

        if kg and kg_extractor:
            limited_chunks = changed[: (kg_chunks_per_doc or len(changed))]
            for chunk, facts in await _aparallel_map_chunks(
                limited_chunks,
                kg_extractor,
                parallelism=kg_parallelism,
            ):
                for fact in facts:
                    subj, pred, obj, fmeta = fact
                    pending_facts.append(
                        KnowledgeGraphFactRecord(
                            subject=subj,
                            predicate=pred,
                            object=obj,
                            metadata=dict(fmeta or {}),
                        )
                    )

        if kg and kg_bundle_extractor:
            limited_chunks = changed[: (kg_chunks_per_doc or len(changed))]
            for chunk, extracted in await _aparallel_map_chunks(
                limited_chunks,
                kg_bundle_extractor,
                parallelism=kg_parallelism,
            ):
                extracted = extracted or {}
                entities = []
                triples = []
                if isinstance(extracted, KGExtraction):
                    entities = extracted.entities or []
                    triples = extracted.triples or []
                elif isinstance(extracted, dict):
                    entities = extracted.get("entities") or extracted.get("entity_ids") or []
                    triples = extracted.get("triples") or extracted.get("relations") or []
                if isinstance(entities, list):
                    bundle_entities_by_chunk[chunk.id] = [str(e) for e in entities]
                for t in triples:
                    try:
                        subj, pred, obj = t
                    except Exception:
                        continue
                    pending_facts.append(
                        KnowledgeGraphFactRecord(
                            subject=str(subj),
                            predicate=str(pred),
                            object=str(obj),
                            metadata={
                                "doc_id": chunk.id,
                                "source_id": chunk.metadata.get("source_id", ""),
                                "snippet": chunk.text[:200],
                            },
                        )
                    )

        if entity_fn or bundle_entities_by_chunk:
            for chunk in changed:
                entities = []
                if prefer_bundle_entities and chunk.id in bundle_entities_by_chunk:
                    entities = bundle_entities_by_chunk.get(chunk.id, []) or []
                else:
                    entities = await _acall_entity_extractor(entity_fn, chunk.text) if entity_fn else []
                norm_entities: List[str] = []
                seen_e: set[str] = set()
                for e in entities:
                    if not _is_valid_entity_name(e):
                        continue
                    norm = normalize_entity_name(e)
                    if not norm or norm in seen_e:
                        continue
                    seen_e.add(norm)
                    norm_entities.append(norm)
                    if kg:
                        pending_facts.append(
                            KnowledgeGraphFactRecord(
                                subject=chunk.id,
                                predicate="mentions",
                                object=norm,
                                metadata={
                                    "source_id": chunk.metadata.get("source_id", ""),
                                    "snippet": chunk.text[:200],
                                },
                            )
                        )
                if norm_entities:
                    chunk.metadata = dict(chunk.metadata)
                    chunk.metadata["entity_ids"] = ",".join(norm_entities)

        if kg and pending_facts:
            if hasattr(kg, "aupsert_facts"):
                await kg.aupsert_facts(pending_facts)  # type: ignore[attr-defined]
            elif hasattr(kg, "upsert_facts"):
                await asyncio.to_thread(kg.upsert_facts, pending_facts)
            else:
                for fact in pending_facts:
                    if hasattr(kg, "aupsert_fact"):
                        await kg.aupsert_fact(fact.subject, fact.predicate, fact.object, fact.metadata)  # type: ignore[attr-defined]
                    else:
                        await asyncio.to_thread(kg.upsert_fact, fact.subject, fact.predicate, fact.object, fact.metadata)
        if index:
            await _aindex_upsert_source(index, src, doc_hash=doc_hash, chunks=new_hashes)
        pending_index_chunks.extend(changed)
        if len(pending_index_chunks) >= _INDEX_FLUSH_CHUNK_COUNT:
            await _aflush_changed_chunks(
                changed=pending_index_chunks,
                vb=vb,
                lb=lb,
                docstore=docstore,
            )
            pending_index_chunks = []

    await _aflush_changed_chunks(
        changed=pending_index_chunks,
        vb=vb,
        lb=lb,
        docstore=docstore,
    )

    return all_chunks


def _parallel_map_chunks(
    chunks: List[Document],
    fn: Callable[[Document], Any],
    *,
    parallelism: int,
) -> List[tuple[Document, Any]]:
    if not chunks:
        return []

    worker_count = max(1, min(parallelism, len(chunks)))
    if worker_count == 1:
        return [(chunk, fn(chunk)) for chunk in chunks]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(fn, chunks))
    return list(zip(chunks, results))


async def _aparallel_map_chunks(
    chunks: List[Document],
    fn: Callable[[Document], Any],
    *,
    parallelism: int,
) -> List[tuple[Document, Any]]:
    if not chunks:
        return []

    worker_count = max(1, min(parallelism, len(chunks)))
    if worker_count == 1:
        out = []
        for chunk in chunks:
            out.append((chunk, await _acall_chunk_transform(fn, chunk)))
        return out

    semaphore = asyncio.Semaphore(worker_count)

    async def _run(chunk: Document) -> tuple[Document, Any]:
        async with semaphore:
            return chunk, await _acall_chunk_transform(fn, chunk)

    return list(await asyncio.gather(*(_run(chunk) for chunk in chunks)))


def _delete_chunk_ids(
    chunk_ids: List[str],
    vb,
    lb,
    kg,
    docstore,
) -> None:
    if not chunk_ids:
        return
    try:
        vb.delete_documents(chunk_ids)
    except Exception:
        pass
    if lb:
        try:
            lb.delete_documents(chunk_ids)
        except Exception:
            pass
    if docstore:
        try:
            docstore.delete_documents(chunk_ids)
        except Exception:
            pass
    if kg and hasattr(kg, "delete_facts_for_chunks"):
        try:
            kg.delete_facts_for_chunks(chunk_ids)  # type: ignore[attr-defined]
        except Exception:
            pass
    elif kg and hasattr(kg, "delete_facts_for_chunk"):
        for cid in chunk_ids:
            try:
                kg.delete_facts_for_chunk(cid)  # type: ignore[attr-defined]
            except Exception:
                    pass


def _flush_changed_chunks(
    changed: List[Document],
    vb,
    lb,
    docstore,
) -> None:
    if not changed:
        return
    if docstore:
        try:
            docstore.upsert_documents(changed)
        except Exception:
            pass
    vb.add_documents(changed)
    if lb:
        lb.index_documents(changed)


async def _adelete_chunk_ids(
    chunk_ids: List[str],
    vb,
    lb,
    kg,
    docstore,
) -> None:
    if not chunk_ids:
        return
    try:
        async_delete = getattr(vb, "adelete_documents", None)
        if callable(async_delete):
            await async_delete(chunk_ids)
        else:
            await asyncio.to_thread(vb.delete_documents, chunk_ids)
    except Exception:
        pass
    if lb:
        try:
            async_delete = getattr(lb, "adelete_documents", None)
            if callable(async_delete):
                await async_delete(chunk_ids)
            else:
                await asyncio.to_thread(lb.delete_documents, chunk_ids)
        except Exception:
            pass
    if docstore:
        try:
            async_delete = getattr(docstore, "adelete_documents", None)
            if callable(async_delete):
                await async_delete(chunk_ids)
            else:
                await asyncio.to_thread(docstore.delete_documents, chunk_ids)
        except Exception:
            pass
    if kg and hasattr(kg, "adelete_facts_for_chunks"):
        try:
            await kg.adelete_facts_for_chunks(chunk_ids)  # type: ignore[attr-defined]
        except Exception:
            pass
    elif kg and hasattr(kg, "delete_facts_for_chunks"):
        try:
            await asyncio.to_thread(kg.delete_facts_for_chunks, chunk_ids)  # type: ignore[attr-defined]
        except Exception:
            pass
    elif kg and hasattr(kg, "delete_facts_for_chunk"):
        for cid in chunk_ids:
            try:
                await asyncio.to_thread(kg.delete_facts_for_chunk, cid)  # type: ignore[attr-defined]
            except Exception:
                pass


async def _aflush_changed_chunks(
    changed: List[Document],
    vb,
    lb,
    docstore,
) -> None:
    if not changed:
        return
    if docstore:
        try:
            async_upsert = getattr(docstore, "aupsert_documents", None)
            if callable(async_upsert):
                await async_upsert(changed)
            else:
                await asyncio.to_thread(docstore.upsert_documents, changed)
        except Exception:
            pass
    async_add = getattr(vb, "aadd_documents", None)
    if callable(async_add):
        await async_add(changed)
    else:
        await asyncio.to_thread(vb.add_documents, changed)
    if lb:
        async_index = getattr(lb, "aindex_documents", None)
        if callable(async_index):
            await async_index(changed)
        else:
            await asyncio.to_thread(lb.index_documents, changed)


async def _aindex_list_sources(index) -> List[str]:
    async_list = getattr(index, "alist_sources", None)
    if callable(async_list):
        return await async_list()
    return await asyncio.to_thread(index.list_sources)


async def _aindex_get_source_state(index, source_id: str):
    async_get = getattr(index, "aget_source_state", None)
    if callable(async_get):
        return await async_get(source_id)
    return await asyncio.to_thread(index.get_source_state, source_id)


async def _aindex_delete_source(index, source_id: str) -> None:
    async_delete = getattr(index, "adelete_source", None)
    if callable(async_delete):
        await async_delete(source_id)
        return
    await asyncio.to_thread(index.delete_source, source_id)


async def _aindex_upsert_source(index, source_id: str, *, doc_hash: str, chunks: Dict[str, str]) -> None:
    async_upsert = getattr(index, "aupsert_source", None)
    if callable(async_upsert):
        await async_upsert(source_id, doc_hash=doc_hash, chunks=chunks)
        return
    await asyncio.to_thread(index.upsert_source, source_id, doc_hash, chunks)


async def _acall_chunk_builder(fn: Callable[[str, Dict[str, Any]], List[Document]], text: str, metadata: Dict[str, Any]) -> List[Document]:
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
        result = fn(text, metadata)
        return list(await result) if inspect.isawaitable(result) else list(result)
    return list(await asyncio.to_thread(fn, text, metadata))


async def _acall_chunk_transform(fn: Callable[[Document], Any], chunk: Document) -> Any:
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
        result = fn(chunk)
        return await result if inspect.isawaitable(result) else result
    return await asyncio.to_thread(fn, chunk)


async def _acall_entity_extractor(fn: Callable[[str], List[str]], text: str) -> List[str]:
    if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None)):
        result = fn(text)
        value = await result if inspect.isawaitable(result) else result
        return list(value or [])
    return list(await asyncio.to_thread(fn, text) or [])
