
from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Optional


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class ScoredChunk:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    score: float


MetadataFilter = Mapping[str, Any]


def metadata_matches_filter(
    metadata: Mapping[str, Any] | None,
    metadata_filter: MetadataFilter | None,
) -> bool:
    """Return whether metadata satisfies a simple exact-match style filter."""

    if not metadata_filter:
        return True
    metadata = metadata or {}
    return all(
        _metadata_value_matches(metadata.get(key), expected)
        for key, expected in metadata_filter.items()
    )


def _metadata_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping) and any(
        str(key).startswith("$") for key in expected
    ):
        plain_keys = [k for k in expected if not str(k).startswith("$")]
        if plain_keys:
            raise ValueError(
                f"Filter value mixes operator keys ($-prefixed) with plain keys "
                f"{plain_keys!r}. Use all operators or all plain equality checks."
            )
        return all(
            _metadata_operator_matches(actual, str(operator), operand)
            for operator, operand in expected.items()
        )
    if _is_non_string_sequence(expected):
        expected_values = list(expected)
        if _is_non_string_sequence(actual):
            return any(value in expected_values for value in actual)
        return actual in expected_values
    if _is_non_string_sequence(actual):
        return any(value == expected for value in actual)
    return actual == expected


def _metadata_operator_matches(actual: Any, operator: str, operand: Any) -> bool:
    if operator == "$eq":
        return _metadata_value_matches(actual, operand)
    if operator == "$ne":
        return not _metadata_value_matches(actual, operand)
    if operator == "$in":
        values = list(operand) if _is_non_string_sequence(operand) else [operand]
        if _is_non_string_sequence(actual):
            return any(value in values for value in actual)
        return actual in values
    if operator == "$nin":
        values = list(operand) if _is_non_string_sequence(operand) else [operand]
        if _is_non_string_sequence(actual):
            return all(value not in values for value in actual)
        return actual not in values
    if operator == "$contains":
        if isinstance(actual, str):
            return str(operand) in actual
        if _is_non_string_sequence(actual):
            return operand in actual
        return False
    raise ValueError(f"Unsupported metadata filter operator: {operator}")


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Collection) and not isinstance(
        value,
        (str, bytes, bytearray, Mapping),
    )


@dataclass(frozen=True)
class KnowledgeGraphFactRecord:
    subject: str
    predicate: str
    object: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStoreBackend(Protocol):
    """Abstract vector store interface."""

    def add_documents(self, docs: List[Document]) -> None:
        ...

    def delete_documents(self, ids: List[str]) -> None:
        ...

    def similarity_search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[ScoredChunk]:
        ...


class AsyncVectorStoreBackend(Protocol):
    """Async vector store interface."""

    async def aadd_documents(self, docs: List[Document]) -> None:
        ...

    async def adelete_documents(self, ids: List[str]) -> None:
        ...

    async def asimilarity_search(
        self, query: str, k: int = 10, **kwargs: Any
    ) -> List[ScoredChunk]:
        ...


class DocumentStoreBackend(Protocol):
    """Abstract document store for fetching chunks by id."""

    def upsert_documents(self, docs: List[Document]) -> None:
        ...

    def delete_documents(self, ids: List[str]) -> None:
        ...

    def get_documents(self, ids: List[str]) -> Dict[str, Document]:
        ...


class AsyncDocumentStoreBackend(Protocol):
    """Async document store for fetching chunks by id."""

    async def aupsert_documents(self, docs: List[Document]) -> None:
        ...

    async def adelete_documents(self, ids: List[str]) -> None:
        ...

    async def aget_documents(self, ids: List[str]) -> Dict[str, Document]:
        ...


class LexicalBackend(Protocol):
    """Abstract lexical retriever interface (BM25 / keyword / full-text)."""

    def index_documents(self, docs: List[Document]) -> None:
        ...

    def delete_documents(self, ids: List[str]) -> None:
        ...

    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional["MetadataFilter"] = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        ...


class AsyncLexicalBackend(Protocol):
    """Async lexical retriever interface."""

    async def aindex_documents(self, docs: List[Document]) -> None:
        ...

    async def adelete_documents(self, ids: List[str]) -> None:
        ...

    async def asearch(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional["MetadataFilter"] = None,
        **kwargs: Any,
    ) -> List[ScoredChunk]:
        ...


class Reranker(Protocol):
    """Second-stage reranker."""

    def rerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        ...


class AsyncReranker(Protocol):
    """Async second-stage reranker."""

    async def arerank(
        self, query: str, candidates: List[ScoredChunk], k: Optional[int] = None
    ) -> List[ScoredChunk]:
        ...


class KnowledgeGraphBackend(Protocol):
    """Minimal KG interface for GraphRAG."""

    def upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def upsert_facts(self, facts: List[KnowledgeGraphFactRecord]) -> None:
        ...

    def delete_facts_for_chunks(self, chunk_ids: List[str]) -> None:
        ...

    def neighborhood(
        self, node_id: str, hops: int = 2, limit: int = 50
    ) -> List[Dict[str, Any]]:
        ...

    def query(
        self, cypher_or_graph_query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        ...


class AsyncKnowledgeGraphBackend(Protocol):
    """Async minimal KG interface for GraphRAG."""

    async def aupsert_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    async def aupsert_facts(self, facts: List[KnowledgeGraphFactRecord]) -> None:
        ...

    async def adelete_facts_for_chunks(self, chunk_ids: List[str]) -> None:
        ...

    async def aneighborhood(
        self, node_id: str, hops: int = 2, limit: int = 50
    ) -> List[Dict[str, Any]]:
        ...

    async def aquery(
        self, cypher_or_graph_query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        ...
