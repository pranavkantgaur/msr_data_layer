# RAG Pipeline — Internal Architecture

This diagram details the multi-step Retrieval-Augmented Generation pipeline
implemented in `msr_digital_twin_with_rag.py`.

---

## Ingestion Pipeline

```mermaid
flowchart TD
    subgraph ING["Ingestion Pipeline — MSRDigitalTwinRAG.add_document()"]
        direction TB
        RAW["Raw document text\n(ORNL OCR / OpenAlex abstract /\nplant sensor snapshot)"]

        subgraph CHUNK_STAGE["Stage 1 · Text chunking  _chunk_text()"]
            SC["Split on sentence boundaries\n(re.split on .!?)"]
            WC["Accumulate up to 300 words per chunk"]
            OV["60-word overlap between\nconsecutive chunks"]
            SC --> WC --> OV
        end

        subgraph EMBED_STAGE["Stage 2 · Embedding  _embed()"]
            ENG{"Embedding\nengine\nselection"}
            E1["OpenAI-compatible API\n(MSR_OPENAI_API_KEY)\ntext-embedding-3-small\ndim = 1536"]
            E2["GitHub Models API\n(MSR_GITHUB_TOKEN)\ntext-embedding-3-small\ndim = 1536"]
            E3["Local GPU\n(MSR_USE_LOCAL_GPU=true)\nall-MiniLM-L6-v2\ndim = 384"]
            E4["Random projection\n(numpy, no keys needed)\ndim = 256"]
            ENG -->|key present| E1
            ENG -->|token present| E2
            ENG -->|GPU=true| E3
            ENG -->|fallback| E4
        end

        subgraph INSIGHT_STAGE["Stage 3 · Source insight extraction  _extract_source_insight()"]
            LLM_INS["LLM call (if key available)\nGenerates:\n• 1-sentence summary\n• topic tags\n• 3-5 key facts\n• source type"]
            STUB_INS["Stub insight\n(if no LLM key)\nFirst 200 chars as summary"]
            LLM_INS & STUB_INS --> INS_OBJ["SourceInsight dataclass"]
        end

        HASH["SHA-256 hash of text\nfor deduplication"]

        subgraph PERSIST["Stage 4 · KB persistence"]
            CJ["chunks.json\n{id, text, source_id, data_type,\n embedding_index, timestamp}"]
            ENJ["embeddings.npy\nnumpy float32 matrix\n[n_chunks × dim]"]
            IJ["insights.json\n{source_id → SourceInsight}"]
            TF["tfidf.json\nvocab + doc-freq counts\nfor BM25-style scoring"]
        end

        RAW --> HASH
        HASH -->|skip if already in KB| CHUNK_STAGE
        CHUNK_STAGE --> EMBED_STAGE
        EMBED_STAGE --> INSIGHT_STAGE
        INSIGHT_STAGE --> PERSIST
        EMBED_STAGE --> PERSIST
        CHUNK_STAGE --> PERSIST
    end

    classDef stage fill:#fff8e1,stroke:#ff9800,color:#000
    classDef store fill:#fafafa,stroke:#607d8b,color:#000
    class CHUNK_STAGE,EMBED_STAGE,INSIGHT_STAGE stage
    class CJ,ENJ,IJ,TF store
```

---

## Retrieval Pipeline

```mermaid
flowchart TD
    subgraph RET["Retrieval Pipeline — MSRDigitalTwinRAG.answer()"]
        direction TB
        Q["Natural-language question\n(from agent or operator)"]

        subgraph DECOMP["Step 1 · Query decomposition  _decompose_query()"]
            LLM_D["LLM call\nBreaks question into\n≤ 5 targeted sub-queries\neach with extraction instructions"]
            SQ["Sub-query list\n[{query, instructions}, ...]"]
            LLM_D --> SQ
        end

        subgraph SEARCH["Step 2 · Parallel hybrid search  _hybrid_search()  ×N sub-queries"]
            direction LR
            DENSE["Dense retrieval\ncosine_similarity(\n  embed(sub_query),\n  embeddings.npy\n)\ntop-k chunks"]
            SPARSE["Sparse retrieval\nTF-IDF BM25 scoring\nover tokenised chunks\ntop-k chunks"]
            MERGE["Score fusion\nRRF (Reciprocal Rank Fusion)\nor weighted sum\nde-duplicated final set"]
            DENSE & SPARSE --> MERGE
        end

        subgraph SUBANS["Step 3 · Sub-answer extraction  _extract_sub_answer()  ×N"]
            LLM_SA["LLM call per sub-query\nContext: ranked chunks\nExtracts focused partial answer\nfollowing sub-query instructions"]
        end

        subgraph SYNTH["Step 4 · Final synthesis  _synthesise()"]
            LIVE["Live plant state\n_get_current_state()\n(from MSR_PLANT_DATA_URL\n or dev stub)"]
            LLM_F["LLM call\nCombines:\n• All sub-answers\n• Live plant data\n• Source citations\ninto comprehensive final answer"]
        end

        Q --> DECOMP
        DECOMP --> SEARCH
        SEARCH --> SUBANS
        SUBANS --> SYNTH
        LIVE --> SYNTH
        SYNTH --> ANS["Final answer\n(plain text + citations)"]
    end

    classDef step fill:#e8f4f8,stroke:#2196f3,color:#000
    class DECOMP,SEARCH,SUBANS,SYNTH step
```

---

## Embedding engine priority

```mermaid
flowchart LR
    START(["Startup\n_build_embed_fn()"])
    C1{"MSR_OPENAI_API_KEY\nset?"}
    C2{"MSR_GITHUB_TOKEN\nset?"}
    C3{"MSR_USE_LOCAL_GPU\n= true?"}
    E1["OpenAI API\nmodel: text-embedding-3-small\ndim: 1536"]
    E2["GitHub Models API\nendpoint: models.inference.ai.azure.com\ndim: 1536"]
    E3["sentence-transformers\nall-MiniLM-L6-v2\ndim: 384\n(requires torch)"]
    E4["Random projection\nnumpy PRNG matrix\ndim: 256\nzero external dependencies"]

    START --> C1
    C1 -->|yes| E1
    C1 -->|no| C2
    C2 -->|yes| E2
    C2 -->|no| C3
    C3 -->|yes| E3
    C3 -->|no| E4

    classDef engine fill:#fff8e1,stroke:#ff9800,color:#000
    classDef check  fill:#e8f4f8,stroke:#2196f3,color:#000
    class E1,E2,E3,E4 engine
    class C1,C2,C3 check
```

---

## KB store file layout

```
kb_store/
├── chunks.json          — list of chunk dicts
│                           {id, text, source_id, data_type,
│                            embedding_index, timestamp}
├── embeddings.npy       — float32 numpy array [n_chunks × embed_dim]
├── insights.json        — dict {source_id → {summary, topics, key_facts, source_type}}
├── tfidf.json           — {vocab: {term→idx}, idf: [...], doc_freq: {...}}
├── archive_state.json   — {ingested_files: [...]}
├── openalex_state.json  — {ingested_ids: [...]}
├── arxiv_state.json     — {ingested_ids: [...]}
├── semanticscholar_state.json — {ingested_ids: [...]}
└── plant_data_state.json — {ingested_ids: [...]}
```

All files are regenerated from scratch if missing.  Re-running any loader
only adds genuinely new content — already-seen IDs are skipped.
