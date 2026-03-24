# End-to-End System Architecture — MSR Data Layer

This diagram shows every component of the `msr_data_layer` system and how
data flows between them: from external sources, through the knowledge-base
pipeline, into the MCP server, and out to consuming agents or operators.

---

## Full System Diagram

```mermaid
flowchart TD
    %% ── External data sources ─────────────────────────────────────────────
    subgraph EXT["External Sources"]
        direction TB
        ORNL["🏛️ ORNL MSR Archive\n(pranavkantgaur/msr-archive)\nOCR text — 1960s MSRE/MSBR reports"]
        OA["📚 OpenAlex API\nmolten salt reactor papers\n(TMSR-LF1 / SINAP focus)"]
        AX["📄 arXiv Atom XML\nMSR + TMSR-LF1 preprints\n≥ 3 s between requests (ToS)"]
        S2["🔬 Semantic Scholar\nS2 Graph API\nOptional API key for 100 req/s"]
        SCADA["⚙️ Plant SCADA / Historian\nMSR_PLANT_DATA_URL\n(REST JSON; stub when unset)"]
        PUSH["📊 Sensor push (operator/agent)\ntimestamped readings\n{sensor_name, value, unit, timestamp}"]
    end

    %% ── Knowledge-base loaders ────────────────────────────────────────────
    subgraph KBS["msr_kb_sources.py — Source Loaders"]
        direction TB
        AL["MSRArchiveLoader\nfetches OCR files via\nGitHub Contents API\nstate: archive_state.json"]
        OAL["OpenAlexLoader\nfetches academic papers\ndeduplication + state\nstate: openalex_state.json"]
        AXL["ArXivLoader\nfetches Atom XML\ndeduplication + state\nstate: arxiv_state.json"]
        S2L["SemanticScholarLoader\nfetches paper metadata\ndeduplication + state\nstate: semanticscholar_state.json"]
        PDL["PlantDataLoader\naccepts operator/agent pushes\nsensor snapshots, event logs\nstate: plant_data_state.json"]
        TSS["TimeseriesStore\nSQLite sqlite3 stdlib\nplant_timeseries.db\ntime-range / aggregate / NL→SQL\nstate: timeseries_state.json"]
        KBM["KBSourceManager\nupdate_all() orchestrates\nall four text loaders\ningest_timeseries() / query_timeseries()\nquery_timeseries_nl() NL→SQL"]
    end

    ORNL -->|HTTPS fetch| AL
    OA   -->|REST/HTTPS| OAL
    AX   -->|Atom XML| AXL
    S2   -->|REST/HTTPS| S2L
    PUSH -->|insert_readings()| TSS
    AL & OAL & AXL & S2L --> KBM

    %% ── RAG pipeline ──────────────────────────────────────────────────────
    subgraph RAG["msr_digital_twin_with_rag.py — RAG Pipeline"]
        direction TB
        CHUNK["1 · Sentence-aware\ntext chunking\n(300 words, 60-word overlap)"]
        EMBED["2 · Dense vector\nembedding\n(see Embedding Engines diagram)"]
        INSIGHT["3 · Source-insight\nextraction\n(LLM summary + topics)"]
        KB[("📦 KB Store\n./kb_store/\nchunks.json\nembeddings.npy\ninsights.json\ntfidf.json")]
        QDECOMP["4 · Query decomposition\n≤ 5 sub-queries (LLM)"]
        HYBRID["5 · Hybrid retrieval\ncosine similarity\n+ TF-IDF BM25"]
        SYNTH["6 · Multi-step synthesis\nsub-answer extraction\n→ final answer (LLM)"]
    end

    KBM  -->|"add_document()"| CHUNK
    PDL  -->|"ingest_text() / ingest_sensor_snapshot()"| CHUNK
    CHUNK --> EMBED --> INSIGHT --> KB
    KB   -->|chunks + embeddings| HYBRID
    QDECOMP --> HYBRID
    HYBRID --> SYNTH

    %% ── MCP server ────────────────────────────────────────────────────────
    subgraph MCP["msr_mcp_server.py — MCP Tool Surface"]
        direction LR
        READ["Read tools\n• get_reactor_status\n• get_sensor_reading\n• get_all_sensor_readings\n• get_sensor_history\n• get_active_alarms\n• get_data_source_info"]
        TSTOOLS["Timeseries tools\n• query_sensor_timeseries\n• get_sensor_stats\n• query_plant_data_nl (NL→SQL)"]
        WRITE["Write tool\n• ingest_plant_data\n  → PlantDataLoader\n    → RAG pipeline"]
    end

    SCADA -->|"GET MSR_PLANT_DATA_URL (or dev stub)"| READ
    SYNTH -->|"rag.answer()"| READ
    TSS   -->|"query_range / aggregate / execute_safe_select"| TSTOOLS
    WRITE -->|calls| PDL

    %% ── Transport / deployment variants ───────────────────────────────────
    subgraph TRANS["Transport Variants"]
        STDIO["msr_mcp_server_main.py\nstdio transport\nfor local agents /\nClaude Desktop / Copilot"]
        LAMBDA["lambda_function.py\nHTTP transport (AWS)\nAPI Gateway + Lambda\nPOST /mcp\nPOST /query\nPOST /kb/update\nPOST /data/ingest\nPOST /timeseries/ingest\nPOST /timeseries/query\nGET  /health"]
    end

    MCP --> STDIO
    MCP --> LAMBDA

    %% ── AWS infrastructure ────────────────────────────────────────────────
    subgraph AWS["AWS Infrastructure (template.yaml)"]
        APIGW["API Gateway\nHTTP API v2\nCORS enabled"]
        LFUNC["Lambda Function\nMSRKBFunction\npython3.12 / x86_64\n1 GB RAM, 15 min timeout\nX-Ray tracing"]
        S3["S3 Bucket\nKBStoreBucket\nVersioned + AES-256\nKB persistence across\nLambda cold starts"]
        EB["EventBridge Rule\nDaily schedule\n→ /kb/update\n(auto KB refresh)"]
        CW["CloudWatch Logs\n+ X-Ray traces"]
    end

    LAMBDA --> APIGW --> LFUNC
    LFUNC  <-->|"sync_kb_from_s3() sync_kb_to_s3()"| S3
    EB     -->|scheduled invoke| LFUNC
    LFUNC  --> CW

    %% ── GPU variant ───────────────────────────────────────────────────────
    subgraph GPU["GPU Container Variant (Dockerfile.gpu)"]
        GFUNC["MSRKBGPUFunction\nECS / EC2 / ECR\nsentence-transformers\nall-MiniLM-L6-v2\nTinyLlama-1.1B-Chat"]
    end

    LAMBDA -.->|optional GPU container path| GFUNC

    %% ── Consumers ─────────────────────────────────────────────────────────
    subgraph CONS["Consumers — LLM Agents & Operators"]
        CC["Claude Code /\nClaude Desktop"]
        GHC["GitHub Copilot /\nOpenAI Codex"]
        PY["Custom Python agent\nmsr_digital_twin_client.py\nMSRDataLayerClient"]
        OPS["Human operators\nvia HTTP REST"]
    end

    STDIO  --> CC & GHC & PY
    APIGW  --> CC & GHC & PY & OPS

    %% ── Physical AI integration ───────────────────────────────────────────
    subgraph PAI["Physical AI Layer Integration"]
        ROBOT["msr_physical_ai_layer\n12 robotic areas\n(PLMR-01 … SPR-01)"]
        TRAIN["Foundation model\nfine-tuning corpus\n(use_cases/physical_ai/)"]
    end

    ROBOT -->|robotic task episodes PlantDataLoader| PDL
    KB    -->|"rag.answer() for structured queries"| TRAIN

    %% ── Styling ───────────────────────────────────────────────────────────
    classDef extNode   fill:#e8f4f8,stroke:#2196f3,color:#000
    classDef kbNode    fill:#e8f8e8,stroke:#4caf50,color:#000
    classDef ragNode   fill:#fff8e1,stroke:#ff9800,color:#000
    classDef mcpNode   fill:#fce4ec,stroke:#e91e63,color:#000
    classDef awsNode   fill:#fff3e0,stroke:#ff9800,color:#000
    classDef consNode  fill:#f3e5f5,stroke:#9c27b0,color:#000
    classDef paiNode   fill:#e0f2f1,stroke:#009688,color:#000
    classDef storeNode fill:#fafafa,stroke:#607d8b,color:#000

    class ORNL,OA,AX,S2,SCADA extNode
    class AL,OAL,AXL,S2L,PDL,KBM kbNode
    class CHUNK,EMBED,INSIGHT,QDECOMP,HYBRID,SYNTH ragNode
    class KB storeNode
    class READ,WRITE mcpNode
    class APIGW,LFUNC,S3,EB,CW awsNode
    class CC,GHC,PY,OPS consNode
    class ROBOT,TRAIN paiNode
```

---

## Data-flow summary

| Stage | Module | In | Out |
|---|---|---|---|
| Source fetch | `msr_kb_sources.py` | ORNL archive, OpenAlex, arXiv, Semantic Scholar | Raw document text |
| Chunking | `msr_digital_twin_with_rag.py` | Raw document text | 300-word overlapping chunks |
| Embedding | `msr_digital_twin_with_rag.py` | Text chunks | Dense float vectors (dim 256–1536) |
| Insight extraction | `msr_digital_twin_with_rag.py` | Text chunks | LLM-generated summaries + topics |
| KB persistence | `./kb_store/` JSON files | Chunks + embeddings + insights | On-disk vector store |
| Plant data ingest (text) | `msr_kb_sources.py` `PlantDataLoader` | Sensor snapshots / event logs | Chunks → KB store |
| Plant data ingest (timeseries) | `msr_kb_sources.py` `TimeseriesStore` | `{sensor_name, value, unit, timestamp}` rows | SQLite `plant_timeseries.db` |
| Timeseries query (structured) | `TimeseriesStore.query_range/aggregate` | sensor name + optional time bounds | Rows / aggregate stats |
| Timeseries query (NL→SQL) | `KBSourceManager.query_timeseries_nl` | Natural-language question | LLM-generated SQL → executed rows |
| Retrieval | `msr_digital_twin_with_rag.py` | Natural-language question | Ranked relevant chunks |
| Synthesis | `msr_digital_twin_with_rag.py` | Ranked chunks + live plant state | Final answer text |
| MCP tools | `msr_mcp_server.py` | Agent/operator requests | JSON tool responses |
| HTTP transport | `lambda_function.py` | HTTP POST/GET | JSON responses |
| stdio transport | `msr_mcp_server_main.py` | MCP JSON-RPC messages | MCP JSON-RPC responses |
| S3 sync | `lambda_function.py` | `/tmp/kb_store/` files | S3 bucket objects |

See the individual component diagrams for internals of each stage.
