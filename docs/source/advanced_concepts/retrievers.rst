Retrievers
==========
.. admonition:: New in Andromeda 1.1.1

   The retriever subsystem was introduced in version 1.1.1.

Andromeda includes a retriever subsystem for document ingestion, chunking, dense retrieval,
hybrid retrieval, reranking, and GraphRAG-style expansion.

This is an advanced feature because retrieval introduces:

- stateful indexes
- backend-specific dependencies
- relevance tuning tradeoffs
- ingestion lifecycle concerns
- optional graph enrichment

At a high level, the retriever flow looks like this:

1. Raw documents are normalized and chunked.
2. Chunks are written into one or more retrieval backends.
3. Queries are executed through dense, hybrid, reranked, or graph-augmented retrieval.
4. Retrieved chunks are passed into prompts, tools, or higher-level workflows.

Installation
------------

The base ``andromeda`` installation offers the retriever source code, but does **not** include heavier backend provider dependencies by default. This design provides multiple installation options so you can choose only the extras you need, keeping the standard install and Docker image leaner.

Common install patterns:

.. code-block:: bash

   pip install "andromeda[retrievers]"
   pip install "andromeda[retrievers,retrievers-faiss]"
   pip install "andromeda[retrievers,retrievers-chroma]"
   pip install "andromeda[retrievers,retrievers-postgres]"

Available extras:

- ``retrievers``: common retrieval helpers
- ``retrievers-faiss``: FAISS-backed vector search
- ``retrievers-chroma``: Chroma-backed vector search
- ``retrievers-postgres``: PGVector plus Postgres-backed state/indexing
- ``retrievers-azure``: Azure AI Search integration
- ``retrievers-mongo``: Mongo Atlas vector search
- ``retrievers-opensearch``: OpenSearch vector search
- ``retrievers-nlp``: optional NLP helpers such as spaCy-backed extraction

Core Building Blocks
--------------------

The retriever package is split into a few main concepts:

- ``Document`` / ``ScoredChunk`` in ``andromeda.retrievers.core``
- ``DocumentProcessingEngine`` in ``andromeda.retrievers.processing``
- ``RAGRegistry`` and ``CorpusConfig`` in ``andromeda.retrievers.config``
- ``ingest_corpus(...)`` in ``andromeda.retrievers.ingest``
- ``RetrievalService`` in ``andromeda.retrievers.service``

These play different roles:

- **Processing** converts raw text into normalized chunks.
- **Registry** builds the configured backend objects and retrievers.
- **Ingest** performs indexing, lexical updates, and optional graph enrichment.
- **Service** provides a simpler runtime retrieval API for querying a corpus.

Minimal End-to-End Example
--------------------------

This example shows the basic programmatic flow: configure a corpus, ingest raw text,
and query it through the retrieval service.

.. code-block:: python

   from andromeda.utils import get_embedding_model
   from andromeda.config import ModelConfig
   from andromeda.retrievers import (
       ChunkingConfig,
       CorpusConfig,
       DocumentProcessingEngine,
       RAGConfig,
       RAGRegistry,
       RawDocument,
       RetrievalService,
       ingest_corpus,
   )


   embeddings = get_embedding_model(ModelConfig(name="nomic-embed-text", provider="litellm"))

   rag_config = RAGConfig(
       corpora={
           "knowledge": CorpusConfig(
               name="knowledge",
               backend_type="faiss",
               params={},
               enable_lexical=True,
           )
       }
   )

   registry = RAGRegistry(
       config=rag_config,
       embedding_model=embeddings,
   )

   processor = DocumentProcessingEngine(
       chunking=ChunkingConfig(
           strategy="recursive",
           size=800,
           overlap=120,
       )
   )

   ingest_corpus(
       registry=registry,
       corpus_name="knowledge",
       raw_docs=[
           RawDocument(
               id="doc-1",
               text="Andromeda supports agents, teams, middleware, and retrieval.",
               metadata={"title": "overview"},
           )
       ],
       processor=processor,
   )

   service = RetrievalService(registry)
   chunks = service.retrieve(
       corpus="knowledge",
       query="How does Andromeda handle retrieval?",
       mode="hybrid+rerank",
       k=5,
   )

   for chunk in chunks:
       print(chunk.doc_id, chunk.score, chunk.text[:120])

Choosing a Retrieval Method
---------------------------

Andromeda currently supports four runtime retrieval modes:

- ``dense``
- ``hybrid``
- ``hybrid+rerank``
- ``graphrag``

These are not just different names. They reflect different tradeoffs in recall,
precision, latency, and complexity.

Retrieval Mode Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 25 23 18 18

   * - Mode
     - Best for
     - Strengths
     - Tradeoffs
     - Recommended default
   * - ``dense``
     - Semantic matching when phrasing varies a lot
     - Good semantic recall, simplest setup
     - May miss exact identifiers or keywords
     - Good baseline, not the default for most production corpora
   * - ``hybrid``
     - Search over technical docs, identifiers, product names, mixed language
     - Combines semantic and lexical retrieval
     - Ranking quality may still be rough at the top
     - Strong default if you do not want reranking cost
   * - ``hybrid+rerank``
     - Most production QA and knowledge retrieval
     - Best overall balance of recall and final precision
     - More latency and more moving parts
     - Best starting default for most teams
   * - ``graphrag``
     - Entity-centric corpora with meaningful relationships across chunks
     - Can pull connected evidence that base retrieval misses
     - Highest complexity; can over-expand if badly tuned
     - Use only when relationships materially improve answers

Rule of Thumb
~~~~~~~~~~~~~

- Start with ``hybrid+rerank`` if you want the safest general-purpose choice.
- Use ``dense`` for a lightweight semantic baseline or small prototypes.
- Use ``hybrid`` when exact terms, codes, and names matter but you want lower cost than reranking.
- Use ``graphrag`` only after the base retrieval pipeline is already working well.

Dense Retrieval
---------------

What it is
~~~~~~~~~~

Dense retrieval uses embeddings to find semantically similar chunks. It works well when:

- users ask the same thing in many different ways
- the right answer uses different wording than the query
- you want the simplest retrieval pipeline possible

What it is good for
~~~~~~~~~~~~~~~~~~~

- conceptual or natural-language questions
- small or medium corpora
- a first retrieval baseline before adding lexical search or reranking

Where it struggles
~~~~~~~~~~~~~~~~~~

- exact identifiers such as ticket numbers, policy codes, product SKUs, class names
- acronym-heavy corpora
- corpora where keyword presence matters more than semantic similarity

How to set it up
~~~~~~~~~~~~~~~~

Dense retrieval only requires:

- an embedding model
- a vector backend

.. code-block:: python

   chunks = service.retrieve(
       corpus="knowledge",
       query="How is middleware applied?",
       mode="dense",
       k=5,
   )

Recommended when:

- you are just starting
- the corpus is mostly prose
- you want fewer retrieval knobs to tune

Hybrid Retrieval
----------------

What it is
~~~~~~~~~~

Hybrid retrieval combines:

- dense vector retrieval
- lexical keyword retrieval

Andromeda fuses the two ranked lists with reciprocal-rank fusion (RRF). This gives you both:

- semantic recall from embeddings
- exact-term recall from lexical search

What it is good for
~~~~~~~~~~~~~~~~~~~

- technical documentation
- API references
- policy corpora
- corpora containing product names, IDs, abbreviations, or field names

Where it is stronger than dense
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hybrid retrieval is usually better than dense retrieval when users search for:

- exact strings
- rare terms
- compound identifiers
- semi-structured phrasing

How to set it up
~~~~~~~~~~~~~~~~

Hybrid retrieval requires:

- a vector backend
- lexical indexing enabled on the corpus

.. code-block:: python

   rag_config = RAGConfig(
       corpora={
           "knowledge": CorpusConfig(
               name="knowledge",
               backend_type="faiss",
               params={},
               enable_lexical=True,
           )
       }
   )

   chunks = service.retrieve(
       corpus="knowledge",
       query="middleware input masking",
       mode="hybrid",
       k=8,
   )

Recommended when:

- exact terms matter
- dense retrieval alone misses obvious hits
- you want better recall without yet paying reranker cost

Hybrid + Rerank
---------------

What it is
~~~~~~~~~~

This mode first retrieves a wider candidate pool using hybrid retrieval, then applies a reranker
to improve the final ordering.

In the current implementation, ``RetrievalService`` expands the initial hybrid recall window to
roughly ``3 * k`` before reranking the candidates back down to ``k``.

Why it is often the best default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This mode gives you:

- the semantic + exact-term coverage of hybrid retrieval
- improved top-of-list quality from reranking

That usually makes it the best default for:

- RAG answer generation
- agent context retrieval
- user-facing question answering

Tradeoffs
~~~~~~~~~

- higher latency than dense or hybrid alone
- more compute cost
- an additional model or embedding dependency if you use a non-noop reranker

How to set it up
~~~~~~~~~~~~~~~~

Hybrid+rerank requires:

- hybrid retrieval to be available
- a reranker configured on the registry if you want real reranking

.. code-block:: python

   from andromeda.retrievers.rerankers import EmbeddingCosineReranker

   registry = RAGRegistry(
       config=rag_config,
       embedding_model=embeddings,
       reranker=EmbeddingCosineReranker(embedding_model=embeddings),
   )

   chunks = service.retrieve(
       corpus="knowledge",
       query="How is middleware applied?",
       mode="hybrid+rerank",
       k=5,
   )

Recommended when:

- answer quality matters more than minimal latency
- your corpus mixes prose plus exact technical terms
- agents need strong context selection rather than just broad recall

GraphRAG
--------

What it is
~~~~~~~~~~

GraphRAG starts with normal retrieval, extracts or reuses query entities, traverses the knowledge
graph, and expands the result set with graph-connected chunks.

In practice, it is useful when relevant evidence is distributed across multiple chunks that are
connected by entities or relationships.

What it is good for
~~~~~~~~~~~~~~~~~~~

- enterprise knowledge graphs
- product and system relationship documentation
- corpora where entity mentions link evidence across otherwise weakly similar chunks

What it needs
~~~~~~~~~~~~~

GraphRAG requires more infrastructure than the other modes:

- a base retriever
- a knowledge graph backend
- an entity extractor
- ideally a docstore so graph-discovered results can return full chunk text

Andromeda can enrich graph usage further by:

- attaching ``entity_ids`` during ingestion
- restoring mention edges from persisted state
- expanding from query entities into related chunk IDs

Important tuning knobs
~~~~~~~~~~~~~~~~~~~~~~

``CorpusConfig`` includes GraphRAG-specific controls:

.. list-table::
   :header-rows: 1
   :widths: 24 14 40 22

   * - Option
     - Default
     - Meaning
     - Recommendation
   * - ``enable_graph``
     - ``False``
     - Enables graph-aware retrieval for that corpus
     - Turn on only for corpora that truly benefit from graph expansion
   * - ``graph_hops``
     - ``2``
     - How many graph hops are traversed
     - Start with ``1`` or ``2``; higher values expand aggressively
   * - ``graph_max_neighbors``
     - ``50``
     - Maximum neighbors returned per graph lookup
     - Lower this if graph expansion becomes noisy
   * - ``graph_max_query_entities``
     - ``12``
     - Cap on unique entities used as graph seeds
     - Keep modest unless your entity extractor is highly precise

How to set it up
~~~~~~~~~~~~~~~~

.. code-block:: python

   from andromeda.retrievers.kg import InMemoryKnowledgeGraph, simple_entity_extractor

   registry = RAGRegistry(
       config=RAGConfig(
           corpora={
               "knowledge": CorpusConfig(
                   name="knowledge",
                   backend_type="faiss",
                   params={},
                   enable_lexical=True,
                   enable_graph=True,
                   docstore_path="./knowledge.sqlite",
                   graph_hops=1,
                   graph_max_neighbors=25,
                   graph_max_query_entities=8,
               )
           }
       ),
       embedding_model=embeddings,
       kg_backend=InMemoryKnowledgeGraph(),
       entity_extractor=simple_entity_extractor,
   )

   chunks = service.retrieve(
       corpus="knowledge",
       query="How does Alpha service connect to Policy API?",
       mode="graphrag",
       k=6,
   )

When to avoid it
~~~~~~~~~~~~~~~~

Avoid GraphRAG when:

- the corpus has weak or noisy entities
- relationships are not actually useful for answering questions
- you have not yet validated dense or hybrid retrieval

GraphRAG should usually be the **last** retrieval method you adopt, not the first.

Backend Selection
-----------------

Different vector backends change operational characteristics, but not the high-level retrieval API.

Backend Types
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 22 32 30

   * - ``backend_type``
     - Extra to install
     - Required ``params``
     - Typical use case
   * - ``faiss``
     - ``retrievers-faiss``
     - Optional ``index_path``, optional ``allow_dangerous_deserialization``
     - Local or embedded deployments, lightweight experimentation
   * - ``chroma``
     - ``retrievers-chroma``
     - ``collection_name``, ``persist_dir``
     - Local persisted vector search with simple setup
   * - ``azure``
     - ``retrievers-azure``
     - ``endpoint``, ``key``, ``index_name``
     - Azure-hosted enterprise search deployments
   * - ``mongo``
     - ``retrievers-mongo``
     - ``connection_string``, ``db_name``, ``collection_name``, ``index_name``
     - Mongo Atlas-based retrieval stacks
   * - ``pgvector``
     - ``retrievers-postgres``
     - ``connection_string``, ``table_name``
     - Postgres-centric architectures and shared infra
   * - ``opensearch``
     - ``retrievers-opensearch``
     - ``url``, ``index_name``, ``username``, ``password``
     - OpenSearch-based deployments

How to choose a backend
~~~~~~~~~~~~~~~~~~~~~~~

- Choose ``faiss`` for local development or simple embedded deployments.
- Choose ``chroma`` when you want local persistence and a convenient local store.
- Choose ``pgvector`` when Postgres is already part of your operational platform.
- Choose managed search backends when your organization already standardizes on them.

Corpus Configuration Reference
------------------------------

``CorpusConfig`` is where most retrieval behavior is defined.

Common fields:

.. list-table::
   :header-rows: 1
   :widths: 22 16 36 26

   * - Field
     - Required
     - Purpose
     - Typical recommendation
   * - ``name``
     - Yes
     - Corpus identifier
     - Keep stable; use a business-meaningful name
   * - ``backend_type``
     - Yes
     - Which vector backend to build
     - Start with ``faiss`` or ``chroma``
   * - ``params``
     - Yes
     - Backend-specific parameters
     - Keep provider credentials outside code when possible
   * - ``enable_lexical``
     - No
     - Enables BM25 lexical retrieval
     - ``True`` for most document corpora
   * - ``enable_graph``
     - No
     - Enables graph retrieval support
     - ``False`` unless you are intentionally building GraphRAG
   * - ``ingestion``
     - No
     - Chunking config stored on the corpus object
     - Useful when you want chunking tied directly to corpus config
   * - ``docstore_path``
     - No
     - SQLite docstore path for chunk persistence
     - Recommended for GraphRAG or restart-safe lexical restore
   * - ``docstore_table``
     - No
     - Docstore table name
     - Leave default unless you have naming constraints
   * - ``ingestion_index_path``
     - No
     - SQLite path for ingest state
     - Usually let Andromeda derive it from ``docstore_path``
   * - ``state_backend_type``
     - No
     - Shared state backend such as ``postgres``
     - Use ``postgres`` when centralizing state across processes
   * - ``state_backend_params``
     - No
     - Parameters for shared state backend
     - Required only when using non-default state storage

Recommended Starting Configurations
-----------------------------------

Use these as practical defaults:

.. list-table::
   :header-rows: 1
   :widths: 26 24 30 20

   * - Scenario
     - Retrieval mode
     - Suggested setup
     - Why
   * - Small prototype or notebook
     - ``dense`` or ``hybrid``
     - ``faiss`` + local embeddings
     - Low setup friction
   * - Internal docs assistant
     - ``hybrid+rerank``
     - ``faiss`` or ``chroma`` + lexical + embedding reranker
     - Best balance of precision and simplicity
   * - Production QA over technical docs
     - ``hybrid+rerank``
     - lexical enabled, persistent store, reranker configured
     - Strong top-k quality for answer generation
   * - Entity-linked enterprise corpus
     - ``graphrag``
     - persistent docstore, graph enabled, tuned entity extraction
     - Useful when relationship expansion matters

Rerankers
---------

What reranking is
~~~~~~~~~~~~~~~~~

Reranking happens **after** an initial candidate set has been retrieved.

The idea is:

1. retrieve a broader set of candidate chunks
2. score those candidates more carefully
3. keep only the best final ranking

Reranking is useful because first-pass retrieval often optimizes for recall, not perfect ordering.

When to use a reranker
~~~~~~~~~~~~~~~~~~~~~~

Use a reranker when:

- your top 3 to 10 results are often relevant but in the wrong order
- hybrid retrieval gets the right candidates but answer quality still feels unstable
- you want stronger precision for user-facing answers

Do not start with a reranker if:

- your base retrieval is not yet returning the right candidate set at all
- latency or cost is the primary concern
- the corpus is tiny and retrieval is already obviously good

Built-in Reranker Options
~~~~~~~~~~~~~~~~~~~~~~~~~

Andromeda currently includes these rerankers:

- ``NoopReranker``
- ``EmbeddingCosineReranker``
- ``LLMListwiseReranker``

Reranker Comparison
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 24 22 16 16

   * - Reranker
     - Best for
     - Strengths
     - Tradeoffs
     - Recommended use
   * - ``NoopReranker``
     - Baseline or no-rerank setups
     - No cost, no latency
     - No ranking improvement
     - Use when validating base retrieval first
   * - ``EmbeddingCosineReranker``
     - Most practical reranking setups
     - Cheap, simple, deterministic
     - Less nuanced than an LLM reranker
     - Best first reranker to try
   * - ``LLMListwiseReranker``
     - High-value queries where ranking quality matters a lot
     - Can reason over the whole candidate list
     - Highest cost and latency
     - Use selectively for premium or critical workflows

EmbeddingCosineReranker
~~~~~~~~~~~~~~~~~~~~~~~

This reranker embeds:

- the query
- each candidate chunk

Then it sorts candidates by cosine similarity.

Use it when:

- you want a strong, low-friction reranker
- you already have an embeddings model available
- you want lower latency than LLM reranking

Setup example:

.. code-block:: python

   from andromeda.retrievers.rerankers import EmbeddingCosineReranker

   reranker = EmbeddingCosineReranker(
       embedding_model=embeddings,
       max_chars_per_candidate=4000,
   )

   registry = RAGRegistry(
       config=rag_config,
       embedding_model=embeddings,
       reranker=reranker,
   )

Recommended starting value:

- ``max_chars_per_candidate=4000`` is a good default unless chunks are much larger than normal

LLMListwiseReranker
~~~~~~~~~~~~~~~~~~~

This reranker sends the candidate list to an LLM and asks it to return a final ranked ordering.

It is more expressive than embedding reranking because it can reason over:

- directness of answer
- factual coverage
- relation between multiple candidates
- graph-derived context metadata

Use it when:

- ranking quality matters more than latency
- queries are high-value
- candidate sets are still noisy after hybrid retrieval

Setup example:

.. code-block:: python

   from andromeda.config import ModelConfig
   from andromeda.retrievers.rerankers import (
       LLMListwiseReranker,
       LLMRerankerConfig,
   )

   reranker = LLMListwiseReranker(
       LLMRerankerConfig(
           model=ModelConfig(name="llama3.1:8b", provider="litellm"),
           max_candidates=30,
           max_chars_per_candidate=800,
           temperature=0.0,
       )
   )

   registry = RAGRegistry(
       config=rag_config,
       embedding_model=embeddings,
       reranker=reranker,
   )

Recommended starting values:

- ``max_candidates``: start around ``20`` to ``30``
- ``max_chars_per_candidate``: start around ``500`` to ``1000`` to control cost
- ``temperature``: keep at ``0.0`` for stable ranking behavior

How to choose a reranker
~~~~~~~~~~~~~~~~~~~~~~~~

- Choose ``EmbeddingCosineReranker`` first for most deployments.
- Choose ``LLMListwiseReranker`` only when there is a clear quality reason to pay the cost.
- Use ``NoopReranker`` while validating the rest of the stack or if latency must stay minimal.

Chunking and Ingestion Notes
----------------------------

``DocumentProcessingEngine`` handles normalization and chunking before indexing.
The defaults are reasonable, but they matter:

- chunk size affects recall vs context density
- overlap affects whether context survives chunk boundaries
- deduplication avoids repeated chunks within a processing run
- token-aware or sentence-aware chunking may perform better for long technical documents

``ChunkingConfig`` options include:

- ``strategy``: ``recursive``, ``sentence``, ``sentence_window``, ``markdown``, ``tokens``, ``fixed``
- ``size``: chunk size target
- ``overlap``: chunk overlap
- ``min_length``: minimum chunk length
- ``max_chunks``: cap on produced chunks
- ``merge_shorter_than``: merge overly short adjacent chunks
- ``sentence_overlap``: overlap control for sentence-window mode

Recommended chunking defaults:

.. list-table::
   :header-rows: 1
   :widths: 24 18 18 40

   * - Corpus type
     - Strategy
     - Starting size
     - Notes
   * - General prose and docs
     - ``recursive``
     - ``800``
     - Best general-purpose default
   * - Sentence-sensitive content
     - ``sentence`` or ``sentence_window``
     - ``600`` to ``1000``
     - Better when sentence boundaries matter
   * - Markdown docs
     - ``markdown``
     - ``800``
     - Preserves section structure better
   * - Token-budgeted pipelines
     - ``tokens``
     - model-dependent
     - Useful when aligning to context-window behavior

``ingest_corpus(...)`` supports more than simple upsert behavior. Depending on configuration,
it can also:

- maintain lexical indexes
- update docstores used by GraphRAG expansion
- perform delta-style re-ingestion
- attach extracted entity metadata
- populate knowledge-graph facts

Using Retrievers with Agents
----------------------------

Retrievers are not automatically exposed to agents. The usual pattern is to wrap retrieval
in an Andromeda tool and pass that tool into an agent or team.

Conceptually:

.. code-block:: python

   from andromeda.tools import tool


   @tool
   def retrieve_context(query: str) -> str:
       chunks = service.retrieve(
           corpus="knowledge",
           query=query,
           mode="hybrid+rerank",
           k=5,
       )
       return "\n\n".join(chunk.text for chunk in chunks)

This keeps the retriever lifecycle in your application code while giving agents controlled access
to retrieval results.

Metadata Filtering
------------------

``RetrievalService.retrieve(...)`` and ``retrieve_with_debug(...)`` accept an optional
``metadata_filter`` argument. Use it to constrain retrieval to a tenant, document, subdocument,
classification label, or any other metadata field attached during ingestion.

Example:

.. code-block:: python

   chunks = service.retrieve(
       corpus="claims",
       query="What is the policy deductible?",
       mode="hybrid+rerank",
       k=8,
       metadata_filter={
           "tenant_id": "tenant-a",
           "document_id": "doc-123",
           "final_group_id": "subdoc-002",
       },
   )

The filter is applied to dense vector retrieval and the in-memory BM25 lexical retriever for
``hybrid`` and ``hybrid+rerank`` modes. GraphRAG applies the filter to the base retrieval pass and
to docstore-backed graph expansions. Dense backends receive the filter through their underlying
LangChain vector store ``filter`` argument, so exact backend support and advanced operator behavior
can vary by provider.

The built-in in-memory lexical filter supports exact equality and simple operator forms:

- ``{"field": "value"}``
- ``{"field": {"$eq": "value"}}``
- ``{"field": {"$ne": "value"}}``
- ``{"field": {"$in": ["a", "b"]}}``
- ``{"field": {"$nin": ["a", "b"]}}``
- ``{"field": {"$contains": "value"}}``

Keep frequently filtered values as scalar metadata where possible. For example, prefer
``tenant_id``, ``document_id``, ``final_group_id``, and ``classification_label`` as direct fields.

Best Practices
--------------

- Start with one corpus and one backend before adding multi-corpus routing.
- Keep backend installs explicit; only install the extras you actually use.
- Prefer ``hybrid+rerank`` as the default retrieval mode for general knowledge work.
- Validate base retrieval before adding GraphRAG or LLM reranking.
- Keep graph expansion narrow; excessive neighborhood traversal hurts relevance quickly.
- Treat docstore and ingestion state as production data, not temporary scratch state.
- Wrap retrieval in tools if the consumer is an Andromeda agent.

When to Go Further
------------------

If retrieval becomes central to your application, the next step is usually one of these:

- add a dedicated retrieval tool layer for agents
- standardize corpus construction in your application bootstrap code
- later, introduce first-class config/CLI support if you want retrievers to behave like a
  top-level Andromeda framework feature
