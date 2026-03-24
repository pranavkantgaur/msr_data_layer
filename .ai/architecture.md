# MSR Data Layer — System Architecture

This document is the authoritative architecture reference for AI agents and
human contributors. Read this alongside `requirements.md` before making
structural changes.

---

## High-level data flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SOURCES                             │
│                                                                     │
│  ORNL MSR Archive ──┐                                               │
│  OpenAlex papers  ──┼──► msr_kb_sources.py ──► KB store            │
│  Plant documents  ──┘     (loaders)          (./kb_store/)          │
│                                                    │                │
│  Live SCADA/historian ──► MSR_PLANT_DATA_URL ──────┼──────────────┐ │
│  (or development stub)                             │              │ │
│                                                    │              │ │
│  Push sensor readings ──► PlantDataLoader ──► TimeseriesStore     │ │
│  (operator / agent)        (ingest_text /      SQLite             │ │
│                            ingest_timeseries)  plant_timeseries.db│ │
└─────────────────────────────────────────────────────────────────────┘
                                                     │              │
                                  ┌──────────────────▼──────────────▼─┐
                                  │   msr_digital_twin_with_rag.py    │
                                  │                                   │
                                  │  1. Sentence-aware chunking       │
                                  │  2. Dense vector embeddings       │
                                  │     (GitHub Models / OpenAI /     │
                                  │      local GPU / random-proj)     │
                                  │  3. Source-insight extraction     │
                                  │  4. Hybrid retrieval              │
                                  │     (cosine + TF-IDF)             │
                                  │  5. Multi-step RAG:               │
                                  │     decompose → search → synth    │
                                  └──────────────┬────────────────────┘
                                                 │
                               ┌─────────────────▼─────────────────┐
                               │       msr_mcp_server.py           │
                               │                                   │
                               │  Read tools:                      │
                               │   get_reactor_status              │
                               │   get_sensor_reading              │
                               │   get_all_sensor_readings         │
                               │   get_sensor_history              │
                               │   get_active_alarms               │
                               │   get_data_source_info            │
                               │                                   │
                               │  Timeseries tools (SQLite):       │
                               │   query_sensor_timeseries         │
                               │   get_sensor_stats                │
                               │   query_plant_data_nl (NL→SQL)    │
                               │                                   │
                               │  Write tools:                     │
                               │   ingest_plant_data               │
                               └───────┬───────────────┬───────────┘
                                       │               │
                  ┌────────────────────▼──┐   ┌────────▼────────────────┐
                  │  msr_mcp_server_main  │   │   lambda_function.py   │
                  │  (stdio transport)    │   │   (HTTP + EventBridge  │
                  │  for local agents /   │   │    + /data/ingest      │
                  │  Claude Desktop /     │   │    + /timeseries/ingest│
                  │  OpenClaw             │   │    + /timeseries/query) │
                  └────────────────────┬─┘   └────────────────────────┘
                                       │
                  ┌────────────────────▼──────────────────────────────┐
                  │             LLM AGENTS / OPERATORS                │
                  │                                                   │
                  │  Claude Code · OpenClaw · GitHub Copilot ·        │
                  │  OpenAI Codex · Cursor · custom Python agents      │
                  │  using msr_digital_twin_client.py                 │
                  └───────────────────────────────────────────────────┘
```

---

## Module responsibilities

### `msr_mcp_server.py`
Single source of truth for the MCP tool surface. Imports the RAG pipeline
and KB loaders at startup. All tool handlers are thin wrappers — they call
into `msr_digital_twin_with_rag.py` or directly read the plant data state.

**Key invariant:** this module must work with zero external services
(all paths have stubs) so unit tests never make network calls.

### `msr_digital_twin_with_rag.py`
The intelligence layer. Implements:
- `MSRDigitalTwinRAG` class — KB load, ingest, and multi-step query
- Four embedding engines (GitHub Models → OpenAI → local GPU → random-projection),
  selected at construction time based on available env vars
- Persistent KB store (`./kb_store/`) — JSON files for chunks, embeddings,
  and source insights

### `msr_kb_sources.py`
Document loaders and data stores that populate the KB:
- `MSRArchiveLoader` — fetches and parses ORNL MSR technical reports
- `OpenAlexLoader` — fetches academic papers via the OpenAlex API
- `PlantDataLoader` — ingests real-time plant operational data snapshots;
  state persisted in `plant_data_state.json`
- `TimeseriesStore` — **SQLite-backed timeseries store** (`sqlite3` stdlib);
  append-only table `sensor_readings(id, timestamp, sensor_name, value, unit, source_id, data_type, inserted_at)`;
  supports time-range queries, aggregate statistics, and NL→SQL via
  `execute_safe_select()` + `get_schema_description()`;
  state persisted in `timeseries_state.json`

### `lambda_function.py`
AWS Lambda entry point. Routes HTTP requests to:
- `GET /health` → health check
- `POST /query` → RAG query
- `POST /kb/update` → KB refresh
- `POST /data/ingest` → plant data ingest (calls `PlantDataLoader`)
- `POST /timeseries/ingest` → ingest timestamped sensor readings into `TimeseriesStore`
- `POST /timeseries/query` → structured or NL query against `TimeseriesStore`

### `msr_digital_twin_client.py`
Python client that wraps the MCP server tools as Python methods. Used for
direct integration without an MCP host.

---

## Deployment variants

```
Variant         Transport    Embedding engine          Deployment
─────────────────────────────────────────────────────────────────────
Local MCP       stdio        GitHub Models / random    python msr_mcp_server_main.py
Local HTTP      HTTP/3000    GitHub Models / random    make local-api
Lambda (CPU)    HTTP         GitHub Models / OpenAI    make deploy
Lambda (GPU)    HTTP         local GPU (sentence-t)    make deploy-gpu
GPU container   HTTP         local GPU (sentence-t)    make run-gpu-local
```

---

## Physical AI integration

The `use_cases/physical_ai/` directory documents how this data layer feeds
foundation-model training for each of the 12 robotic operational areas
defined in `msr_physical_ai_layer`:

```
msr_physical_ai_layer  ──► robotic task episodes ──► PlantDataLoader.ingest_sensor_snapshot()
                                                                    │
                                                           kb_store/ (training corpus)
                                                                    │
                       foundation model fine-tuning ◄── rag.answer(structured query)
```

Each area has a dedicated use-case file in `use_cases/physical_ai/` showing
the three-stage pipeline: ORNL archive retrieval → sensor stream ingestion →
labelled episode export.

---

## Data persistence

| Artefact | Location | Git-tracked |
|---|---|---|
| Vector embeddings + chunks | `./kb_store/` | No (`.gitignore`) |
| Source insights | `./kb_store/` | No |
| Plant data state | `plant_data_state.json` | No (generated at runtime) |
| ORNL reference docs | `./docs/` | No (downloaded at runtime) |
| SAM deploy config | `samconfig.toml` | No (git-ignored) |

---

## Security boundaries

- No credentials in source code. All secrets via environment variables.
- The MCP server is **read + limited-write** (ingest only). No control actuation.
- Lambda IAM role is minimal — read/write to its own S3 bucket only.
- All ingest paths carry `source_id` for audit traceability.
