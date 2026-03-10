# MSR Data Layer MCP Interface – Full Guide

## Overview

The MSR data layer exposes MSR (Molten Salt Reactor) plant sensor data and a
knowledge base through the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io).
This allows LLM agents and human operators to:

* Read live sensor readings and active alarms from an external plant data source
  (SCADA system, historian API, or digital twin) — or from a built-in development
  stub when no external URL is configured.
* Query reference documents — historical ORNL MSR reports, academic papers,
  and accumulated plant operational data — via an enhanced multi-step
  Retrieval-Augmented Generation (RAG) pipeline.
* Ingest operational data — push real-time sensor snapshots, event logs, and
  maintenance reports into the knowledge base so future queries can incorporate
  actual plant experience.

> **The data layer is read-only for simulation and control.**  It does not contain
> a built-in reactor simulator and does not accept control-actuation commands.
> Simulation and control capabilities live in the digital twin or process-control
> system that the data layer sits in front of.

---

## Architecture

```
┌─────────────────────────┐        stdin / stdout
│   MCP Host              │◄──────────────────────►┌───────────────────────────┐
│  (Claude Desktop,       │    JSON-RPC 2.0         │  msr_mcp_server_main.py  │
│   VS Code, agent, …)    │                         │  (stdio transport)       │
└─────────────────────────┘                         │                          │
                                                    │  msr_mcp_server.py       │
                                                    │  ┌──────────────────┐    │
                                                    │  │ Tool handlers    │    │
                                                    │  │  • read sensors  │    │
                                                    │  │  • read alarms   │    │
                                                    │  │  • ingest data   │    │
                                                    │  └──────────────────┘    │
                                                    └───────────────────────────┘
                                                               │
                              External plant data API ◄────────┤ (MSR_PLANT_DATA_URL)
                              (SCADA / historian / DT)         │  or development stub

          ┌─────────────────────────────────────────────────────────────────┐
          │  msr_digital_twin_with_rag.py  (Enhanced RAG pipeline)          │
          │                                                                  │
          │  Ingestion:                                                      │
          │    Text → sentence-aware chunking → dense embeddings            │
          │             → source insights (LLM summary/topics/key_facts)    │
          │             → KnowledgeBase (JSON + numpy, persistent)          │
          │                                                                  │
          │  Retrieval (multi-step, inspired by open-notebook ask.py):      │
          │    Question → QueryDecomposer (≤5 sub-queries via LLM)          │
          │             → parallel hybrid search (dense + TF-IDF)           │
          │             → sub-answer extraction per search                  │
          │             → synthesis (sub-answers + live plant data)         │
          └─────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `msr_mcp_server.py`

Core library.  Contains:

* **`_BASE_STATE`** – development stub with representative FLiBe-MSR sensor values
  (used when `MSR_PLANT_DATA_URL` is not set)
* **Tool handler functions** – one Python function per MCP tool
* **`TOOLS`** – list of MCP tool descriptors with JSON schemas
* **`handle_message(raw)`** – JSON-RPC 2.0 dispatcher

#### Available Sensors

| Sensor Key | Unit | Stub Value |
|---|---|---|
| `reactor_power_mw` | MW | 100 |
| `core_temperature_c` | °C | 700 |
| `salt_flow_rate_kg_s` | kg/s | 250 |
| `fuel_salt_level_pct` | % | 87.5 |
| `coolant_salt_level_pct` | % | 91.2 |
| `neutron_flux_n_cm2_s` | n/cm²/s | 2.5 × 10¹³ |
| `primary_loop_pressure_bar` | bar | 1.1 |
| `heat_exchanger_outlet_c` | °C | 565 |
| `turbine_inlet_temp_c` | °C | 540 |
| `turbine_output_mwe` | MWe | 42 |
| `tritium_production_rate_g_day` | g/day | 0.12 |
| `off_gas_activity_bq_m3` | Bq/m³ | 3.8 × 10⁶ |

When `MSR_PLANT_DATA_URL` is set, all read tools fetch from that URL instead of
the stub.

### `msr_mcp_server_main.py`

Entry point.  Reads newline-delimited JSON-RPC messages from **stdin**
and writes responses to **stdout**.  Log messages go to **stderr**.

### `msr_digital_twin_client.py`

Python client class `MSRDataLayerClient`.  Spawns the server as a
subprocess and wraps each MCP tool in a typed Python method.

### `msr_digital_twin_with_rag.py`

Enhanced multi-step RAG pipeline (inspired by
[open-notebook](https://github.com/lfnovo/open-notebook)):

#### Key components

| Class / Function | Role |
|---|---|
| `EmbeddingEngine` | Abstract base for text → dense vector |
| `RandomProjectionEmbeddingEngine` | Numpy random-projection fallback |
| `OpenAIEmbeddingEngine` | OpenAI-compatible /embeddings API |
| `LocalGPUEmbeddingEngine` | sentence-transformers local GPU engine |
| `KnowledgeBase` | Persistent hybrid (dense + TF-IDF) store |
| `SourceInsight` | LLM-generated summary, topics, key_facts |
| `SubQuery` | One decomposed search term + instructions |
| `_chunk_text()` | Sentence-aware overlapping chunking |
| `_decompose_query()` | LLM → ≤5 targeted sub-queries |
| `_extract_insight()` | LLM → SourceInsight for a document |
| `MSRDigitalTwinRAG` | Orchestrates the full pipeline |

#### RAG Pipeline Stages

```
1. Query Decomposition
   question ──► LLM ──► [SubQuery₁, SubQuery₂, …, SubQuery₅]
                         (term + extraction instructions per query)

2. Parallel Search (ThreadPoolExecutor)
   SubQuery₁ ──► KnowledgeBase.search() ──► LLM ──► partial answer₁
   SubQuery₂ ──► KnowledgeBase.search() ──► LLM ──► partial answer₂
   …

3. Synthesis
   [partial answers + live plant data] ──► LLM ──► final answer
```

#### Knowledge base persistence

Chunks, embeddings, insights, and TF-IDF statistics are saved under
`MSR_KB_DIR` (default `./kb_store`) as:

```
kb_store/
  chunks.json       ← chunk text + metadata
  embeddings.npy    ← dense embedding matrix (numpy)
  insights.json     ← LLM-generated source insights
  tfidf.json        ← document-frequency counts
```

The store is loaded on startup so re-embedding is skipped for already-
indexed sources.

---

## Tool Reference

### `get_reactor_status`

Returns the current operational status, power level, core temperature, and
data source identifier.

**Input:** _(none)_

**Output example:**
```json
{
  "status": "NOMINAL",
  "reactor_power_mw": 100.2,
  "core_temperature_c": 701.4,
  "last_updated": "2026-03-09T06:00:00+00:00",
  "data_source": "external"
}
```

---

### `get_sensor_reading`

**Input:**
```json
{ "sensor_name": "core_temperature_c" }
```

**Output example:**
```json
{ "sensor": "core_temperature_c", "value": 701.4, "unit": "°C", "timestamp": "…" }
```

---

### `get_all_sensor_readings`

Returns a map of every sensor with its value and unit.

---

### `get_sensor_history`

**Input:**
```json
{ "sensor_name": "reactor_power_mw", "last_n": 20 }
```

Returns the last N readings stored in the in-memory session buffer (up to 100).

---

### `get_active_alarms`

Returns all currently active safety/operational alarms.

**Alarm thresholds (checked against live data):**

| Sensor | Condition | Alarm ID |
|---|---|---|
| `core_temperature_c` | > 750 °C | `CORE_TEMP_HIGH` |
| `reactor_power_mw` | > 110 MW | `POWER_HIGH` |
| `fuel_salt_level_pct` | < 70 % | `FUEL_LEVEL_LOW` |
| `primary_loop_pressure_bar` | > 1.5 bar | `PRIMARY_PRESSURE_HIGH` |

---

### `get_data_source_info`

Returns information about the active data source (external URL or stub) and
its connectivity status.

**Output example:**
```json
{
  "mode": "external",
  "url": "https://scada.example.com/api/plant/state",
  "reachable": true,
  "plant_status": "NOMINAL"
}
```

---

### `ingest_plant_data`

Pushes operational data into the RAG knowledge base.

**Input:**
```json
{
  "content":   "Core temp 702°C at 14:32 UTC, flow 248 kg/s",
  "data_type": "sensor_snapshot",
  "source_id": "shift-log-2024-01-15-1432"
}
```

`data_type` must be one of: `sensor_snapshot`, `event_log`,
`maintenance_report`, `operational_data`.

**Output example:**
```json
{
  "source_id": "shift-log-2024-01-15-1432",
  "chunks_added": 2
}
```

---

## What the Data Layer Does NOT Do

The following capabilities are intentionally absent – they belong in a
simulation tool or process-control system, not in the data layer:

| Absent capability | Correct location |
|---|---|
| Thermal-hydraulic simulation | Digital twin / physics engine |
| Control rod actuation | Plant control system |
| Alarm acknowledgement | SCADA / DCS |
| Transient calculation | Neutronics / thermalhydraulics solver |

---

## Extending the Server

### Adding a new read tool

1. Write a handler function in `msr_mcp_server.py`.
2. Add an entry to the `TOOLS` list with a JSON schema.
3. The `TOOL_MAP` and `handle_message` dispatcher pick it up automatically.

### Connecting to a real data source

Set `MSR_PLANT_DATA_URL` to the base URL of your SCADA historian, OPC-UA
gateway, or digital twin REST API:

```bash
export MSR_PLANT_DATA_URL=https://your-scada.example.com/api/plant/state
python msr_mcp_server_main.py
```

The API is expected to return a flat JSON object whose keys match the sensor
names listed in the table above.

### Using a real vector database for RAG

Replace `KnowledgeBase` with a ChromaDB or Qdrant client for large corpora
(> 100 000 chunks).

---

See also → [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md)
