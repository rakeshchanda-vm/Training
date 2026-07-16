
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Callable, List

from andromeda.retrievers.core import (
    VectorStoreBackend,
    LexicalBackend,
    KnowledgeGraphBackend,
    Reranker,
    DocumentStoreBackend,
)
from andromeda.retrievers.backends import (
    ChromaBackend,
    FaissBackend,
    AzureSearchBackend,
    MongoAtlasBackend,
    PgVectorBackend,
    AsyncPgVectorBackend,
    OpenSearchBackend,
    BM25LexicalBackend,
)
from andromeda.retrievers.retrievers import DenseRetriever, HybridRetriever, AsyncDenseRetriever, AsyncHybridRetriever, NoopReranker
from andromeda.retrievers.graph import GraphRAGRetriever, AsyncGraphRAGRetriever
from andromeda.retrievers.docstore import (
    InMemoryDocumentStore,
    PostgresDocumentStore,
    AsyncPostgresDocumentStore,
    PostgresDocumentStoreConfig,
    SqliteDocumentStore,
    SqliteDocumentStoreConfig,
)
from andromeda.retrievers.ingestion_index import PostgresIngestionIndex, AsyncPostgresIngestionIndex, PostgresIngestionIndexConfig, SqliteIngestionIndex
from andromeda.retrievers.processing import ChunkingConfig
from andromeda.retrievers.kg import (
    InMemoryKnowledgeGraph,
    PostgresKnowledgeGraph,
    AsyncPostgresKnowledgeGraph,
    PostgresKnowledgeGraphConfig,
    normalize_entity_name,
)


@dataclass
class CorpusConfig:
    name: str
    backend_type: Literal[
        "chroma", "faiss", "azure", "mongo", "pgvector", "opensearch"
    ]
    params: Dict[str, str]
    enable_lexical: bool = True
    enable_graph: bool = False
    ingestion: Optional[ChunkingConfig] = None
    # Optional docstore (recommended for GraphRAG). When unset, uses an in-memory store.
    docstore_path: Optional[str] = None
    docstore_table: str = "documents"
    # Optional ingestion index path (sqlite). When unset and docstore_path is set,
    # a sibling file "<docstore_path>.ingest.sqlite" is used.
    ingestion_index_path: Optional[str] = None
    # Optional shared state backend for docstore + ingestion index.
    state_backend_type: Optional[Literal["sqlite", "postgres", "memory"]] = None
    state_backend_params: Dict[str, str] = field(default_factory=dict)
    # Optional graph backend selection. When unset, defaults to Postgres if state_backend_type
    # is Postgres; otherwise falls back to in-memory.
    graph_backend_type: Optional[Literal["memory", "postgres"]] = None
    graph_backend_params: Dict[str, str] = field(default_factory=dict)
    # GraphRAG traversal controls.
    graph_hops: int = 2
    graph_max_neighbors: int = 50
    graph_max_query_entities: int = 12


@dataclass
class RAGConfig:
    corpora: Dict[str, CorpusConfig]


class RAGRegistry:
    def __init__(
        self,
        config: RAGConfig,
        embedding_model,
        kg_backend: Optional[KnowledgeGraphBackend] = None,
        reranker: Optional[Reranker] = None,
        entity_extractor: Optional[Callable[[str], List[str]]] = None,
    ):
        self.config = config
        self.embedding_model = embedding_model
        self.kg_backend = kg_backend
        self.reranker = reranker or NoopReranker()
        self.entity_extractor = entity_extractor

        self.vector_backends: Dict[str, VectorStoreBackend] = {}
        self.lex_backends: Dict[str, LexicalBackend] = {}
        self.docstores: Dict[str, DocumentStoreBackend] = {}
        self.ingestion_indexes: Dict[str, object] = {}
        self.kg_backends: Dict[str, KnowledgeGraphBackend] = {}
        self.dense_retrievers: Dict[str, DenseRetriever] = {}
        self.hybrid_retrievers: Dict[str, HybridRetriever] = {}
        self.graph_retrievers: Dict[str, GraphRAGRetriever] = {}
        self.async_vector_backends: Dict[str, object] = {}
        self.async_docstores: Dict[str, object] = {}
        self.async_ingestion_indexes: Dict[str, object] = {}
        self.async_kg_backends: Dict[str, object] = {}
        self.async_dense_retrievers: Dict[str, AsyncDenseRetriever] = {}
        self.async_hybrid_retrievers: Dict[str, AsyncHybridRetriever] = {}
        self.async_graph_retrievers: Dict[str, AsyncGraphRAGRetriever] = {}

        self._build()

    def _build_vector_backend(self, corpus: CorpusConfig) -> VectorStoreBackend:
        p = corpus.params
        if corpus.backend_type == "chroma":
            return ChromaBackend(
                collection_name=p["collection_name"],
                embedding=self.embedding_model,
                persist_dir=p["persist_dir"],
            )
        if corpus.backend_type == "faiss":
            return FaissBackend(
                embedding=self.embedding_model,
                index_path=p.get("index_path"),
                allow_dangerous_deserialization=str(
                    p.get("allow_dangerous_deserialization", "false")
                ).lower()
                in {"1", "true", "yes"},
            )
        if corpus.backend_type == "azure":
            return AzureSearchBackend(
                embedding=self.embedding_model,
                azure_search_endpoint=p["endpoint"],
                azure_search_key=p["key"],
                index_name=p["index_name"],
            )
        if corpus.backend_type == "mongo":
            return MongoAtlasBackend(
                embedding=self.embedding_model,
                connection_string=p["connection_string"],
                db_name=p["db_name"],
                collection_name=p["collection_name"],
                index_name=p["index_name"],
            )
        if corpus.backend_type == "pgvector":
            return PgVectorBackend(
                embedding=self.embedding_model,
                connection_string=p["connection_string"],
                collection_table=p["table_name"],
            )
        if corpus.backend_type == "opensearch":
            return OpenSearchBackend(
                embedding=self.embedding_model,
                opensearch_url=p["url"],
                index_name=p["index_name"],
                http_auth=(p["username"], p["password"]),
            )
        raise ValueError(f"Unsupported backend_type {corpus.backend_type}")

    def _build_kg_backend(self, corpus_name: str, corpus: CorpusConfig) -> Optional[KnowledgeGraphBackend]:
        if not corpus.enable_graph:
            return None
        if self.kg_backend is not None:
            return self.kg_backend

        backend_type = corpus.graph_backend_type or (
            "postgres" if corpus.state_backend_type == "postgres" else "memory"
        )
        if backend_type == "memory":
            return InMemoryKnowledgeGraph()
        if backend_type == "postgres":
            params = corpus.graph_backend_params or corpus.state_backend_params or {}
            connection_string = params.get("connection_string")
            if not connection_string:
                raise ValueError(
                    f"Corpus {corpus_name} requires graph_backend_params.connection_string for Postgres graph storage."
                )
            namespace = params.get("namespace", corpus_name)
            table = params.get("table", "rag_corpus_kg_facts")
            return PostgresKnowledgeGraph(
                PostgresKnowledgeGraphConfig(
                    connection_string=connection_string,
                    namespace=namespace,
                    table=table,
                )
            )
        raise ValueError(f"Unsupported graph_backend_type {backend_type}")

    def get_kg_backend(self, corpus: str) -> Optional[KnowledgeGraphBackend]:
        return self.kg_backends.get(corpus) or self.kg_backend

    def get_async_vector_backend(self, corpus: str):
        return self.async_vector_backends[corpus]

    def get_async_docstore(self, corpus: str):
        return self.async_docstores.get(corpus) or self.docstores.get(corpus)

    def get_async_ingestion_index(self, corpus: str):
        return self.async_ingestion_indexes.get(corpus) or self.ingestion_indexes.get(corpus)

    def get_async_kg_backend(self, corpus: str):
        return self.async_kg_backends.get(corpus) or self.kg_backends.get(corpus) or self.kg_backend

    def _build(self) -> None:
        def _batched(items: List[str], size: int = 200) -> List[List[str]]:
            return [items[i : i + size] for i in range(0, len(items), size)]

        for name, corpus in self.config.corpora.items():
            kg_backend = self._build_kg_backend(name, corpus)
            if kg_backend is not None:
                self.kg_backends[name] = kg_backend
                if self.kg_backend is None:
                    self.kg_backend = kg_backend
            async_kg_backend = None

            vb = self._build_vector_backend(corpus)
            self.vector_backends[name] = vb
            self.dense_retrievers[name] = DenseRetriever(vb)
            self.async_vector_backends[name] = vb
            if corpus.backend_type == "pgvector":
                p = corpus.params
                self.async_vector_backends[name] = AsyncPgVectorBackend(
                    embedding=self.embedding_model,
                    connection_string=p["connection_string"],
                    collection_table=p["table_name"],
                )
            self.async_dense_retrievers[name] = AsyncDenseRetriever(self.async_vector_backends[name])

            state_backend = corpus.state_backend_type
            if state_backend == "postgres":
                params = corpus.state_backend_params or {}
                connection_string = params["connection_string"]
                namespace = params.get("namespace", name)
                self.docstores[name] = PostgresDocumentStore(
                    PostgresDocumentStoreConfig(
                        connection_string=connection_string,
                        namespace=namespace,
                    )
                )
                self.async_docstores[name] = AsyncPostgresDocumentStore(
                    PostgresDocumentStoreConfig(
                        connection_string=connection_string,
                        namespace=namespace,
                    )
                )
                self.ingestion_indexes[name] = PostgresIngestionIndex(
                    PostgresIngestionIndexConfig(
                        connection_string=connection_string,
                        namespace=namespace,
                    )
                )
                self.async_ingestion_indexes[name] = AsyncPostgresIngestionIndex(
                    PostgresIngestionIndexConfig(
                        connection_string=connection_string,
                        namespace=namespace,
                    )
                )
            elif corpus.docstore_path:
                self.docstores[name] = SqliteDocumentStore(
                    SqliteDocumentStoreConfig(
                        path=corpus.docstore_path,
                        table=corpus.docstore_table or "documents",
                    )
                )
                self.async_docstores[name] = self.docstores[name]
                index_path = corpus.ingestion_index_path or f"{corpus.docstore_path}.ingest.sqlite"
                self.ingestion_indexes[name] = SqliteIngestionIndex(index_path)
                self.async_ingestion_indexes[name] = self.ingestion_indexes[name]
            else:
                self.docstores[name] = InMemoryDocumentStore()
                self.async_docstores[name] = self.docstores[name]

            if corpus.enable_lexical:
                lb = BM25LexicalBackend()
                self.lex_backends[name] = lb
                self.hybrid_retrievers[name] = HybridRetriever(vb, lb)
                self.async_hybrid_retrievers[name] = AsyncHybridRetriever(self.async_vector_backends[name], lb)
                # Best-effort restore of BM25 index from docstore + ingestion index so that
                # delta-ingest runs (with no changes) still have lexical search.
                docstore = self.docstores.get(name)
                idx = self.ingestion_indexes.get(name)
                if docstore is not None and idx is not None:
                    try:
                        all_ids = idx.list_chunk_ids()
                        for batch in _batched(all_ids, size=200):
                            docs_map = docstore.get_documents(batch)
                            if docs_map:
                                lb.index_documents(list(docs_map.values()))
                    except Exception:
                        pass

            if corpus.enable_graph and kg_backend is not None:
                # Best-effort restore of mention edges for in-memory KG from docstore so GraphRAG
                # still works across process restarts when delta-ingest has no changes.
                docstore = self.docstores.get(name)
                idx = self.ingestion_indexes.get(name)
                if isinstance(kg_backend, InMemoryKnowledgeGraph) and docstore is not None and idx is not None:
                    try:
                        all_ids = idx.list_chunk_ids()
                        for batch in _batched(all_ids, size=200):
                            docs_map = docstore.get_documents(batch)
                            for d in docs_map.values():
                                meta = d.metadata or {}
                                ent_csv = meta.get("entity_ids")
                                if not isinstance(ent_csv, str) or not ent_csv:
                                    continue
                                entities = [e for e in ent_csv.split(",") if e]
                                for ent in entities:
                                    kg_backend.upsert_fact(
                                        subject=d.id,
                                        predicate="mentions",
                                        object_=normalize_entity_name(ent),
                                        metadata={
                                            "source_id": meta.get("source_id", ""),
                                            "snippet": (d.text or "")[:200],
                                        },
                                    )
                    except Exception:
                        pass

                try:
                    from andromeda.retrievers.kg import simple_entity_extractor
                except Exception:
                    def simple_entity_extractor(text: str):
                        return []

                extractor = self.entity_extractor or simple_entity_extractor
                base_ret = self.hybrid_retrievers.get(name) or self.dense_retrievers[name]
                self.graph_retrievers[name] = GraphRAGRetriever(
                    base_retriever=base_ret,
                    kg=kg_backend,
                    entity_extractor=extractor,
                    docstore=self.docstores.get(name),
                    max_query_entities=corpus.graph_max_query_entities,
                    max_neighbors=corpus.graph_max_neighbors,
                    hops=corpus.graph_hops,
                )
                if self.kg_backend is not None:
                    async_kg_backend = kg_backend
                elif corpus.graph_backend_type == "postgres" or (
                    corpus.graph_backend_type is None and corpus.state_backend_type == "postgres"
                ):
                    params = corpus.graph_backend_params or corpus.state_backend_params or {}
                    connection_string = params.get("connection_string")
                    namespace = params.get("namespace", name)
                    table = params.get("table", "rag_corpus_kg_facts")
                    async_kg_backend = AsyncPostgresKnowledgeGraph(
                        PostgresKnowledgeGraphConfig(
                            connection_string=connection_string,
                            namespace=namespace,
                            table=table,
                        )
                    )
                else:
                    async_kg_backend = kg_backend
                self.async_kg_backends[name] = async_kg_backend
                async_base_ret = self.async_hybrid_retrievers.get(name) or self.async_dense_retrievers[name]
                self.async_graph_retrievers[name] = AsyncGraphRAGRetriever(
                    base_retriever=async_base_ret,
                    kg=async_kg_backend,
                    entity_extractor=extractor,
                    docstore=self.async_docstores.get(name),
                    max_query_entities=corpus.graph_max_query_entities,
                    max_neighbors=corpus.graph_max_neighbors,
                    hops=corpus.graph_hops,
                )
            elif kg_backend is not None:
                self.async_kg_backends[name] = kg_backend
            else:
                self.async_kg_backends[name] = None
    
    def get_dense(self, corpus: str) -> DenseRetriever:
        return self.dense_retrievers[corpus]

    def get_hybrid(self, corpus: str) -> HybridRetriever:
        return self.hybrid_retrievers[corpus]

    def get_graphrag(self, corpus: str) -> GraphRAGRetriever:
        return self.graph_retrievers[corpus]

    def get_async_dense(self, corpus: str) -> AsyncDenseRetriever:
        return self.async_dense_retrievers[corpus]

    def get_async_hybrid(self, corpus: str) -> AsyncHybridRetriever:
        return self.async_hybrid_retrievers[corpus]

    def get_async_graphrag(self, corpus: str) -> AsyncGraphRAGRetriever:
        return self.async_graph_retrievers[corpus]
