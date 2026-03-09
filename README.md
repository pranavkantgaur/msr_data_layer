# MSR Data Layer

A **data layer for Molten Salt Reactor (MSR) design, construction, and operations**,
exposed through the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io).

The repo serves as a **knowledge-base and live-data interface** so that LLM agents
(Claude, GitHub Copilot, custom agents) and human operators can:

* **Query reference documents** – historical ORNL MSR reports, academic papers,
  maintenance logs, and plant operational data via an enhanced multi-step
  Retrieval-Augmented Generation (RAG) pipeline.
* **Read live plant data** – sensor readings and alarms from an external plant data
  source (SCADA, historian, or digital twin API), or a development stub when no
  external URL is configured.
* **Ingest operational data** – push real-time sensor snapshots, event logs, and
  maintenance reports into the knowledge base so future queries incorporate actual
  plant experience.

The repo is **not** a digital twin itself.  It is designed to work alongside an
existing digital twin or simulation tool by acting as its knowledge and data layer.

The RAG implementation is inspired by the
[open-notebook](https://github.com/lfnovo/open-notebook) project,
adopting its multi-query decomposition, source-insight extraction, and
parallel-search approach.

---

## Repository Contents

| File | Description |
|---|---|
| `msr_mcp_server.py` | MCP server – read-only data tools + `ingest_plant_data`; configurable external data source |
| `msr_mcp_server_main.py` | Entry point – stdio transport server |
| `msr_digital_twin_client.py` | Python client for the MCP server (`MSRDataLayerClient`) |
| `msr_digital_twin_with_rag.py` | Enhanced multi-step RAG pipeline (+ GPU engines) |
| `msr_kb_sources.py` | KB source loaders: msr-archive, OpenAlex, and `PlantDataLoader` |
| `lambda_function.py` | AWS Lambda handler (HTTP API + EventBridge + `/data/ingest`) |
| `template.yaml` | AWS SAM template (Lambda + API Gateway + S3 + GPU variant) |
| `Dockerfile.gpu` | GPU-capable container image (CUDA + sentence-transformers + transformers) |
| `Makefile` | Build / deploy / local-dev / GPU container targets |
| `requirements_mcp.txt` | Core Python dependencies |
| `requirements_lambda.txt` | Lambda-specific dependencies (adds boto3) |
| `requirements_gpu.txt` | GPU-specific dependencies (torch, sentence-transformers, transformers) |
| `00_MCP_START_HERE.md` | Quick-start guide |
| `MSR_DIGITAL_TWIN_MCP_GUIDE.md` | Full architecture and tool reference |
| `MSR_MCP_DEPLOYMENT_GUIDE.md` | Deployment and production guide |
| `test_msr_mcp_server.py` | Unit tests for MCP server |
| `test_msr_rag.py` | Unit tests for RAG pipeline (including GPU engine tests) |
| `test_msr_kb_sources.py` | Unit tests for KB source loaders (including PlantDataLoader) |
| `test_lambda_function.py` | Unit tests for Lambda handler |

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements_mcp.txt
```

### 2 – Run the demo client

```bash
python msr_digital_twin_client.py
```

### 3 – Query the knowledge base with RAG

```bash
# Without LLM – returns enriched context (hybrid vector + TF-IDF retrieval)
python msr_digital_twin_with_rag.py "What is the current core temperature?"

# With LLM – multi-step RAG: decompose → parallel search → synthesize
export MSR_OPENAI_API_KEY=sk-...
python msr_digital_twin_with_rag.py "Is the plant operating within safe limits?"
```

### 4 – Connect to a live plant data source

```bash
# Set the URL of your SCADA/historian/digital twin REST API
export MSR_PLANT_DATA_URL=https://your-scada.example.com/api/plant/state

# The MCP server will now read live data from that URL instead of the stub
python msr_mcp_server_main.py
```

### 5 – Connect Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "msr-data-layer": {
      "command": "python",
      "args": ["/path/to/msr_mcp_server_main.py"]
    }
  }
}
```

---

## MCP Tools

The data layer exposes 7 tools via the Model Context Protocol:

| Tool | Type | Description |
|---|---|---|
| `get_reactor_status` | Read | Current plant operational status (power, temperature, last-updated) |
| `get_sensor_reading` | Read | Latest value of a named sensor |
| `get_all_sensor_readings` | Read | All sensor values in one call |
| `get_sensor_history` | Read | Last N readings for a sensor (in-memory session buffer) |
| `get_active_alarms` | Read | Currently active safety/operational alarms |
| `get_data_source_info` | Read | Data source configuration and connectivity status |
| `ingest_plant_data` | Write | Push operational data into the RAG knowledge base |

> **No simulation or control-actuation tools are exposed.**  This is a data
> layer – it reads from and ingests into external sources, not from a
> built-in reactor model.

### External data source (`MSR_PLANT_DATA_URL`)

When `MSR_PLANT_DATA_URL` is set, sensor read tools fetch from that URL:

```
GET MSR_PLANT_DATA_URL
→ { "core_temperature_c": 700.5, "reactor_power_mw": 99.8, ... }
```

If the URL is unreachable or unset, a **development stub** with representative
FLiBe-MSR parameters is used so the service can be exercised without a live
data connection.

---

## Enhanced RAG Pipeline

The RAG pipeline adopts the multi-step approach from
[open-notebook](https://github.com/lfnovo/open-notebook):

```
Ingestion:
  Document ──► sentence-aware chunking
           ──► dense embeddings (local GPU / OpenAI API / random-projection)
           ──► source insights (LLM: summary + topics + key_facts)
           ──► persistent KnowledgeBase (JSON + numpy)

Retrieval (per question):
  Question ──► [Step 1] QueryDecomposer (LLM) ──► up to 5 SubQueries
           ──► [Step 2] Parallel hybrid search (dense cosine + TF-IDF)
           ──► [Step 3] Sub-answer extraction (LLM per sub-query)
           ──► [Step 4] Synthesis (LLM: sub-answers + live plant data)
```

### Embedding engines

| Condition | Engine used |
|---|---|
| `MSR_USE_LOCAL_GPU=true` | `LocalGPUEmbeddingEngine` – sentence-transformers, CUDA/MPS/CPU |
| `MSR_OPENAI_API_KEY` set | `OpenAIEmbeddingEngine` – real semantic embeddings |
| No API key, no GPU | `RandomProjectionEmbeddingEngine` – numpy, no external deps |

### Knowledge base sources

| Source | How to trigger |
|---|---|
| Local documents (`MSR_DOCS_DIR`) | Auto-loaded at startup |
| msr-archive OCR files (ORNL reports) | `rag.load_msr_archive()` or `POST /kb/update` |
| OpenAlex academic papers | `rag.update_openalex()` or `POST /kb/update` |
| Plant operational data | `rag.add_document()` / `PlantDataLoader.ingest_text()` / `POST /data/ingest` |

### Loading documents

Place `.md` or `.txt` files in `./docs/` (or set `MSR_DOCS_DIR`):

```bash
mkdir docs
cp my_safety_manual.pdf docs/   # convert to .txt first
python msr_digital_twin_with_rag.py "What are the temperature safety limits?"
```

The knowledge base is persisted to `./kb_store/` so documents are only
embedded once.

---

## Knowledge Base Sources (`msr_kb_sources.py`)

The knowledge base is populated from two external sources via
`msr_kb_sources.py`:

### 1 – Static source: pranavkantgaur/msr-archive

Historical ORNL Molten Salt Reactor reports, OCR-transcribed to text,
from the [msr-archive](https://github.com/pranavkantgaur/msr-archive)
repository (`ocr/` directory).

```bash
# Ingest all new OCR files (first run fetches everything; subsequent runs add only new files)
python msr_kb_sources.py --update-archive

# Limit to 20 files per run
python msr_kb_sources.py --update-archive --max-docs 20
```

Or from Python:

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG
rag = MSRDigitalTwinRAG()
rag.load_msr_archive()          # ingest new ORNL documents
```

### 2 – Dynamic source: OpenAlex academic papers

Papers from the [OpenAlex](https://openalex.org) API matching:
- `"molten salt reactors experimental data"` (broad MSR coverage)
- `"TMSR-LF1"` (targeted: TMSR-LF1 reactor, SINAP, China)

```bash
# Ingest new papers (default: up to 100 per run)
python msr_kb_sources.py --update-openalex

# Use both sources at once
python msr_kb_sources.py --update-all

# Show current ingestion state without fetching
python msr_kb_sources.py --status
```

Or from Python:

```python
rag.update_openalex()           # ingest new OpenAlex papers
```

### State tracking

All loaders write state files to `MSR_KB_DIR` (default `./kb_store`):

```
kb_store/
  archive_state.json    ← URLs of ingested msr-archive OCR files
  openalex_state.json   ← IDs of ingested OpenAlex works
  plant_data_state.json ← IDs of ingested plant operational data records
```

Re-running the updater only adds truly new content.

### 3 – Plant operational data (`PlantDataLoader`)

Push real-time plant data into the knowledge base so future RAG queries
incorporate actual plant experience:

```bash
# CLI
python msr_kb_sources.py --ingest-plant-data \
    --content "Core temperature 702°C at 14:32 UTC, flow 248 kg/s" \
    --data-type sensor_snapshot \
    --source-id "shift-log-20240115T1432Z"

python msr_kb_sources.py --ingest-plant-data \
    --content-file /path/to/maintenance_report.txt \
    --data-type maintenance_report
```

Or from Python:

```python
from msr_kb_sources import PlantDataLoader
loader = PlantDataLoader()
loader.ingest_text(rag, "Inspection complete. HX-1 clean.", source_id="maint-hx1-001")

# Structured sensor snapshot (list or dict)
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2024-01-15T14:00Z", "sensor": "core_temperature_c",
     "value": 702.1, "unit": "°C"},
], source_id="snap-20240115T1400Z")
```

Via the Lambda endpoint:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"content": "Core temp 702°C", "data_type": "sensor_snapshot"}' \
     https://<api>.execute-api.us-east-1.amazonaws.com/prod/data/ingest
```

### Periodic updates (cron example)

```cron
# Update knowledge base daily at 02:00 UTC
0 2 * * *  cd /opt/msr && python msr_kb_sources.py --update-all >> logs/kb_update.log 2>&1
```

---

## GPU-Accelerated Local Models (`LocalGPUEmbeddingEngine` + `LocalGPULLM`)

When `MSR_USE_LOCAL_GPU=true` the pipeline replaces the OpenAI API calls with
fully local, GPU-accelerated models:

| Component | Default model | Library | VRAM |
|---|---|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | sentence-transformers | ~200 MB |
| **Text generation** | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 🤗 transformers | ~2 GB (fp16) |

Both models auto-detect CUDA → MPS (Apple Silicon) → CPU and fall back
gracefully.

### Engine-selection priority

```
MSR_USE_LOCAL_GPU=true  →  LocalGPUEmbeddingEngine + LocalGPULLM
MSR_OPENAI_API_KEY set  →  OpenAIEmbeddingEngine  + OpenAI chat API
(neither)               →  RandomProjectionEmbeddingEngine  (no LLM synthesis)
```

### Using GPU models locally

```bash
pip install -r requirements_gpu.txt
export MSR_USE_LOCAL_GPU=true
# Optional: choose different models
export MSR_LOCAL_EMBED_MODEL=BAAI/bge-small-en-v1.5
export MSR_LOCAL_LLM_MODEL=microsoft/phi-2
python msr_digital_twin_with_rag.py "What is the safe core temperature?"
```

### GPU container image

A CUDA-capable Docker container image is provided in `Dockerfile.gpu`:

```bash
# Build (pre-downloads model weights into the image)
make build-gpu-container

# Run locally on CPU (no GPU required for testing)
make run-gpu-local

# Run with NVIDIA GPU
make run-gpu-cuda

# Build with different models
docker build -f Dockerfile.gpu \
  --build-arg EMBED_MODEL=BAAI/bge-small-en-v1.5 \
  --build-arg LLM_MODEL=microsoft/phi-2 \
  -t msr-kb-gpu .
```

The container bundles model weights so that cold-start latency is minimal (no
HuggingFace Hub download on first request).

### GPU Lambda deployment

```bash
# 1. Push GPU image to ECR
make create-ecr-repo
make push-gpu-container ECR_REPO=<your-ecr-uri>

# 2. Deploy GPU Lambda variant (hosted on /gpu/* routes)
make deploy-gpu ECR_REPO=<your-ecr-uri>

# 3. Query the GPU endpoint
make remote-gpu-query QUESTION="What is the thermal efficiency of TMSR-LF1?"
```

> **Note:** AWS Lambda does not currently support GPU instances.  The GPU
> container image runs on CPU within Lambda (still beneficial for
> self-contained deployments without the OpenAI API key).  For true GPU
> inference, deploy the same container image to **Amazon ECS** with the
> NVIDIA Container Toolkit, **AWS Batch** GPU compute environments, or an
> **EC2 G/P instance** running the Lambda Runtime Interface Emulator.

### `/health` reports GPU status

```json
{
  "service": "msr-knowledge-base",
  "status": "healthy",
  "gpu": {
    "torch_available": true,
    "device": "cuda",
    "local_gpu_mode": true,
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  }
}
```

---

## AWS Lambda Deployment (`lambda_function.py` + `template.yaml`)

The MSR knowledge base can be hosted as a **serverless HTTPS service on AWS
Lambda**, making it accessible to any agent or tool over the internet.

### Architecture

```
Agent / MCP client
       │  HTTPS
       ▼
Amazon API Gateway (HTTP API)
       │
       ▼
AWS Lambda  ─── GET  /health
(lambda_function.py)
               ├── POST /mcp        MCP JSON-RPC 2.0 (any MCP client)
               ├── POST /query      Plain-text RAG query
               └── POST /kb/update  Trigger KB ingestion

S3 Bucket  ←─── KB persistence (chunks + embeddings + state)
      ↑
EventBridge (daily schedule)
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check – returns reactor status and service version |
| `/mcp` | POST | Full MCP JSON-RPC 2.0 – for Claude Desktop, VS Code Copilot, custom agents |
| `/query` | POST | Plain-text question → RAG answer (no MCP client needed) |
| `/kb/update` | POST | Trigger ingestion from msr-archive / OpenAlex |

### Prerequisites

```bash
pip install aws-sam-cli    # https://docs.aws.amazon.com/serverless-application-model/
aws configure              # set AWS credentials
```

### Build and deploy

```bash
# First-time deploy (interactive – saves config to samconfig.toml)
make deploy-guided

# Subsequent deploys
make deploy

# Full help
make help
```

### Local development

```bash
make local-api             # start local server at http://127.0.0.1:3000
make local-health          # GET  /health
make local-query           # POST /query  (test question)
make local-mcp             # POST /mcp    (tools/list)
make local-update          # POST /kb/update (first 5 archive files)
```

### Querying the deployed service

```bash
# Health check
curl https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/health

# RAG query (plain text)
curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $MSR_API_KEY" \
     -d '{"question": "What is the thermal efficiency of the TMSR-LF1 reactor?"}' \
     https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/query

# MCP tools/list (for MCP-compatible clients)
curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $MSR_API_KEY" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
     https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/mcp

# Trigger KB update from both sources
curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $MSR_API_KEY" \
     -d '{"source": "all"}' \
     https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/kb/update

# Ingest plant operational data (sensor snapshot)
curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-Api-Key: $MSR_API_KEY" \
     -d '{"content": "Core temperature 702°C, flow 248 kg/s at 14:32 UTC", "data_type": "sensor_snapshot"}' \
     https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/data/ingest
```

### Knowledge base persistence

The Lambda function stores the KB files in `/tmp/kb_store` during execution
and syncs them to an **S3 bucket** on each update so they survive across
cold starts.  The SAM template creates the bucket automatically.

### Scheduled KB refresh

The same Lambda is also triggered daily by an **EventBridge rule** to
automatically ingest new documents from both sources (configurable via the
`KBUpdateSchedule` CloudFormation parameter, e.g. `rate(1 day)` or
`cron(0 2 * * ? *)`).

### Lambda-specific environment variables

| Variable | Default | Description |
|---|---|---|
| `MSR_KB_S3_BUCKET` | (created by SAM) | S3 bucket for KB persistence |
| `MSR_KB_S3_PREFIX` | `kb-prod/` | Key prefix inside the bucket |
| `MSR_API_KEY` | _(unset)_ | Shared API key (X-Api-Key header) |

All [existing env vars](#environment-variables) (`MSR_OPENAI_API_KEY`, etc.)
are passed through via the SAM template parameters.

---

## MCP Tools (read-only + ingestion)

| Tool | Type | Description |
|---|---|---|
| `get_reactor_status` | Read | Current status, power, core temperature, data source |
| `get_sensor_reading` | Read | Single named sensor value |
| `get_all_sensor_readings` | Read | All sensors at once |
| `get_sensor_history` | Read | Historical readings (up to last 100 samples) |
| `get_active_alarms` | Read | List active alarms |
| `get_data_source_info` | Read | Data source mode, URL, connectivity status |
| `ingest_plant_data` | Write | Push operational data into the RAG knowledge base |

> Simulation tools (`run_thermal_simulation`) and control-actuation tools
> (`set_control_rod_position`, `acknowledge_alarm`) are **not** included.
> This is a data layer; simulation and control live in the digital twin or
> process-control system.

---

## Architecture

```
MCP Host (Claude / agent)
        │  stdin/stdout JSON-RPC 2.0
        ▼
msr_mcp_server_main.py
        │
        ▼
msr_mcp_server.py  ── read tools ── external plant data API (MSR_PLANT_DATA_URL)
                              │                └── development stub (fallback)
                              └── ingest_plant_data ──► MSRDigitalTwinRAG

msr_digital_twin_with_rag.py  (inspired by open-notebook)
  ├── LocalGPUEmbeddingEngine (sentence-transformers, CUDA/MPS/CPU)
  ├── OpenAIEmbeddingEngine   (OpenAI Embeddings API)
  ├── RandomProjectionEmbeddingEngine (numpy fallback, no external deps)
  ├── KnowledgeBase (hybrid dense+TF-IDF, persistent)
  ├── SourceInsight (LLM-generated summary/topics/key_facts)
  ├── _decompose_query() (multi-query strategy via LLM)
  ├── MSRDigitalTwinRAG._run_sub_query() (parallel search + extraction)
  └── MSRDigitalTwinRAG._synthesize() (final answer synthesis)
```

---

## Testing

```bash
pytest -v
```

53 unit tests covering the MCP server (19) and enhanced RAG pipeline (53, including 19 GPU engine tests), plus 55 tests for the KB source loaders (including 15 PlantDataLoader tests) and 55 tests for the Lambda handler (174 total).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MSR_OPENAI_API_KEY` | _(unset)_ | API key for LLM + embeddings |
| `MSR_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `MSR_OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `MSR_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `MSR_DOCS_DIR` | `./docs` | Reference documents directory |
| `MSR_KB_DIR` | `./kb_store` | Persistent knowledge-base directory |
| `MSR_ARCHIVE_REPO` | `pranavkantgaur/msr-archive` | GitHub owner/repo for static source |
| `MSR_ARCHIVE_BRANCH` | `master` | Branch of the archive repo |
| `MSR_ARCHIVE_MAX_DOCS` | `0` (unlimited) | Max OCR files per archive run |
| `MSR_OPENALEX_MAX_RESULTS` | `100` | Max OpenAlex papers per run |
| `MSR_OPENALEX_EMAIL` | _(unset)_ | Email for OpenAlex polite pool |
| `MSR_GITHUB_TOKEN` | _(unset)_ | GitHub PAT for higher API rate limits |
| `MSR_KB_S3_BUCKET` | _(unset)_ | S3 bucket for Lambda KB persistence |
| `MSR_KB_S3_PREFIX` | `kb-prod/` | S3 key prefix inside the bucket |
| `MSR_API_KEY` | _(unset)_ | Shared API key for Lambda HTTP auth |
| `MSR_PLANT_DATA_URL` | _(unset)_ | External plant data REST API URL (SCADA/historian/DT) |
| `MSR_USE_LOCAL_GPU` | `false` | `true` to use local GPU models |
| `MSR_LOCAL_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `MSR_LOCAL_LLM_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Local generation model |
| `MSR_HF_CACHE_DIR` | `/tmp/hf_cache` | HuggingFace model cache directory |

---

## Integrating with MSR Digital Twin Architectures

This data layer is intentionally architecture-agnostic: it exposes a stable MCP
interface on top of whatever plant data source and knowledge base you configure.
The sections below describe the concrete interfaces you need to implement to
connect it to the major MSR digital twin patterns documented in the literature,
grouped by **reactor lifecycle phase**.

> **Key principle:** The data layer reads, stores, and surfaces information.
> Simulation, control, and physics calculations remain in the digital twin.
> The data layer never *drives* the twin — it feeds it and learns from it.

---

### Phase 1 – Design

During the design phase agents and engineers use the data layer primarily as a
**knowledge retrieval and multi-physics data hub**.  The relevant digital twin
architectures are model-based design tools (MOOSE, SAM, ARMI) and early-stage
FPGA/hardware emulation.

#### 1a. Python-Based Unified API (ARMI-style)

[ARMI](https://github.com/terrapower/armi) provides a single authoritative
data model that lets disparate physics kernels (neutronics, thermal-hydraulics,
fuel performance) share geometry and material data.  To use the data layer as
the knowledge backbone for an ARMI-driven design workflow:

```python
# Ingest ARMI run outputs into the knowledge base
from msr_kb_sources import PlantDataLoader
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
loader = PlantDataLoader()

# After each ARMI case run, push the result summary into the KB
armi_summary = """
ARMI case: FLiBe-MSR nominal design
  neutron_flux_peak: 2.5e13 n/cm²/s
  core_outlet_temp: 704°C
  cycle_length_efpd: 365
"""
loader.ingest_text(rag, armi_summary,
                   source_id="armi-nominal-v1",
                   data_type="operational_data")

# Now query design trade-offs via RAG
answer = rag.answer("What core outlet temperature maximises cycle length "
                    "without exceeding FLiBe freezing limits?")
```

#### 1b. RESTful / GraphQL Hub (Deep Lynx–style)

For large design projects that integrate multiple engineering tools via a data
warehouse (e.g. [Deep Lynx](https://github.com/idaholab/Deep-Lynx)), configure
`MSR_PLANT_DATA_URL` to point at your warehouse's sensor/parameter export
endpoint and post design-phase documents to the `/data/ingest` HTTP endpoint:

```bash
# Point the data layer at Deep Lynx's plant-parameter export
export MSR_PLANT_DATA_URL=https://deep-lynx.example.com/api/v1/containers/<id>/data/query

# Ingest a MOOSE thermal-hydraulics result report
curl -X POST https://<api>/prod/data/ingest \
     -H "Content-Type: application/json" \
     -d '{
       "content": "MOOSE TH result: peak cladding temp 712°C at 110% power",
       "data_type": "operational_data",
       "source_id": "moose-th-110pct-power"
     }'
```

**Metadata and tagging:** include a structured `source_id` that encodes the
equipment tag and version (e.g. `moose-HX1-rev3`) so the RAG pipeline can
filter results by component.

#### 1c. RTL I/O Port Interface (FPGA-HLS HIL)

For FPGA-based real-time digital twins that use High-Level Synthesis (HLS),
the data layer acts as the software-side partner that feeds plant parameter
snapshots into the HLS testbench.  Map the data layer's JSON sensor output to
your RTL I/O ports:

```python
import json, urllib.request

# Pull a snapshot from the data layer
resp = urllib.request.urlopen("http://localhost:3000/mcp", ...)
state = json.loads(resp.read())

# Map to HLS C++ struct / RTL port values
hls_input = {
    "core_temp_in":    state["core_temperature_c"],   # ap_fixed<32,10>
    "flow_rate_in":    state["salt_flow_rate_kg_s"],  # ap_fixed<32,10>
    "power_in":        state["reactor_power_mw"],     # ap_fixed<32,10>
}
# Write hls_input to shared memory or a FIFO consumed by the HLS testbench
```

**Buffer memory mapping:** for fission time-delay models that rely on BRAM-backed
ring buffers, use `get_sensor_history` (returns the last *N* readings as a list)
to fill the initial buffer state before the FPGA simulation starts:

```python
from msr_digital_twin_client import MSRDataLayerClient
with MSRDataLayerClient() as client:
    history = client.get_sensor_history("neutron_flux_n_cm2_s", last_n=64)
ring_buffer = history["values"]   # push into BRAM via driver
```

---

### Phase 2 – Construction

During construction the data layer bridges **commissioning test records** and
**as-built inspection reports** into the knowledge base, so agents can answer
questions like "were all welds on loop A pressure-tested?" or "what was the
baseline core temperature during cold commissioning?".

#### 2a. Ingesting Commissioning and Inspection Records

Use `PlantDataLoader` to ingest commissioning test records as they are
completed.  Pair each record with a stable `source_id` that encodes the
equipment tag from the Bill of Materials (BOM):

```python
from msr_kb_sources import PlantDataLoader
loader = PlantDataLoader()

# Hydrostatic pressure test result for loop A
loader.ingest_text(
    rag,
    "Loop-A hydrostatic test: 12 bar for 4 h. No leaks. Date: 2025-03-01.",
    source_id="commissioning-loop-A-hydro-20250301",
    data_type="maintenance_report",
)

# Baseline sensor calibration record
loader.ingest_text(
    rag,
    "TC-101 calibrated against NIST standard. Offset: +0.3°C. Cal date: 2025-03-05.",
    source_id="cal-TC-101-20250305",
    data_type="maintenance_report",
)
```

#### 2b. Bitstream / Firmware Configuration Tracking

For digital twins that reconfigure virtual logic between commissioning phases
(e.g. switching from pre-operational to operational control logic), log the
configuration change through the data layer so the knowledge base retains a
history of which bitstream/firmware was active during each test:

```python
loader.ingest_text(
    rag,
    "FPGA bitstream v2.1.0 loaded 2025-03-10. "
    "Includes updated fission time-delay model for full-power operation.",
    source_id="fpga-bitstream-v2.1.0",
    data_type="operational_data",
)
```

#### 2c. Asynchronous Metadata Exchange

For multi-organization projects the data layer's HTTP endpoints support
**fire-and-forget ingestion** without tight time-step coupling.  Configure
your construction management system to `POST /data/ingest` whenever a
milestone is completed:

```bash
# Triggered by a CI/CD pipeline or work-order system
curl -X POST https://<api>/prod/data/ingest \
     -H "X-Api-Key: $MSR_API_KEY" \
     -d '{
       "content": "Primary pump PP-01 hydro test passed. WO#: 48291.",
       "data_type": "maintenance_report",
       "source_id": "wo-48291-PP01-hydro"
     }'
```

---

### Phase 3 – Operations

During steady-state operations the data layer is the **live sensor gateway and
operational memory**.  The relevant digital twin patterns are health-aware
supervisory control (Ensemble Kalman Filter recalibration) and reinforcement
learning–based O&M scheduling.

#### 3a. Physical-to-Virtual (P2V) Streaming — Ensemble Kalman Filter

For architectures that recalibrate a virtual model using an Ensemble Kalman
Filter (EnKF), the data layer provides the observational measurement stream
*d* and the associated measurement noise covariance Γ.  Implement an adapter
that polls `get_all_sensor_readings` at your required assimilation frequency:

```python
import time, numpy as np
from msr_digital_twin_client import MSRDataLayerClient

def stream_to_enkf(enkf, dt_seconds: float = 60.0):
    """
    Stream live sensor readings (d) and a diagonal measurement-noise
    covariance matrix (Γ) to an EnKF at the requested frequency.
    """
    with MSRDataLayerClient() as client:
        while True:
            snapshot = client.get_all_sensor_readings()

            # Build observation vector d
            sensors = ["core_temperature_c", "reactor_power_mw",
                       "salt_flow_rate_kg_s", "primary_loop_pressure_bar"]
            d = np.array([snapshot["readings"][s]["value"] for s in sensors])

            # Measurement-noise covariance Γ (instrument uncertainties, ±1σ)
            noise_1sigma = np.array([0.5, 0.2, 1.0, 0.005])  # °C, MW, kg/s, bar
            gamma = np.diag(noise_1sigma ** 2)

            # Hand off to the EnKF
            enkf.assimilate(d, gamma)

            time.sleep(dt_seconds)
```

> **Variable time-step:** for transient capture (seconds) vs. long-term
> monitoring (hours), simply change `dt_seconds`.  The data layer's in-memory
> history buffer (`get_sensor_history`) retains up to 1 000 samples so you
> can backfill missed assimilation steps.

#### 3b. Parameter Augmentation Portal

When the virtual model's trainable coefficients (e.g. VARMAX parameters or
neural-network weights) are updated after each EnKF cycle, log the new
parameters through the data layer so the knowledge base tracks model evolution:

```python
updated_params = {"varmax_ar_coef": [0.91, 0.07], "varmax_ma_coef": [0.12]}
loader.ingest_text(
    rag,
    f"EnKF parameter update cycle 47: {updated_params}",
    source_id="enkf-params-cycle-047",
    data_type="operational_data",
)
```

#### 3c. Virtual-to-Physical (V2P) Decision Loop — RL Set-Points

For architectures that use a Reinforcement Learning supervisor (e.g. Soft
Actor Critic) to generate optimised power set-points, the data layer acts as
the **audit log** for V2P commands.  Before a set-point is dispatched to the
plant control system, record it through the data layer for traceability:

```python
def dispatch_setpoint(rl_agent, plant_control_api):
    action = rl_agent.select_action(current_state)

    # Log the proposed set-point in the knowledge base
    loader.ingest_text(
        rag,
        f"RL set-point proposed: power={action['power_mw']:.1f} MW, "
        f"flow={action['flow_kg_s']:.1f} kg/s at {datetime.utcnow().isoformat()}Z",
        source_id=f"rl-setpoint-{int(time.time())}",
        data_type="event_log",
    )

    # Send to plant
    plant_control_api.send_setpoint(action)
```

#### 3d. Constraint Enforcement (Reference Governor)

Before dispatching a set-point you can query the knowledge base via RAG for
safety-boundary context, providing a lightweight pre-check without calling the
full physics model:

```python
safety_context = rag.answer(
    f"Is a mass flow rate of {action['flow_kg_s']:.1f} kg/s within the "
    "safe operating envelope at the proposed power level?"
)
# Log the safety check result alongside the set-point
loader.ingest_text(rag, f"Safety pre-check: {safety_context}",
                   source_id=f"safety-check-{int(time.time())}",
                   data_type="event_log")
```

---

### Phase 4 – Monitoring (Long-Term Health and Degradation)

Long-term monitoring requires the data layer to accumulate **aging and
degradation evidence** and surface it on demand, without requiring tight
time-step integration with the physics model.

#### 4a. Surrogate Compression Interface (RL Offline Data)

For training RL agents on compressed hourly propagators the data layer serves
as the **offline data archive**.  Ingest the propagator outputs as they are
generated so they are searchable during future design or re-commissioning work:

```python
# After each surrogate batch run
for batch_id, summary in surrogate_results.items():
    loader.ingest_text(
        rag,
        f"Surrogate propagator batch {batch_id}: {summary}",
        source_id=f"surrogate-batch-{batch_id}",
        data_type="operational_data",
    )
```

#### 4b. Aging and Degradation Records

Use `maintenance_report` ingestion to build a longitudinal record of
component health indicators.  The RAG pipeline can then synthesise trend
analyses across this history:

```python
# Monthly heat-exchanger fouling check
loader.ingest_text(
    rag,
    "HX-1 thermal resistance 2025-04: 0.00021 m²·K/W (+3% vs. baseline). "
    "Fouling index: MODERATE. Recommended cleaning in 90 days.",
    source_id="hx1-fouling-2025-04",
    data_type="maintenance_report",
)

# Query degradation trend
trend = rag.answer("What is the fouling trend for HX-1 over the past 12 months "
                   "and when should the next cleaning be scheduled?")
```

#### 4c. Consistency Metadata Tagging

To resolve data availability pain points and link physical parts to their
virtual representations, adopt a structured `source_id` convention that
encodes the equipment tag:

```
<system>-<component>-<data_type>-<ISO_date>
  examples:
    primary-HX1-fouling-2025-04
    secondary-pump-PP01-vibration-2025-Q1
    fpga-bitstream-v2.1.0
    enkf-params-cycle-047
```

This makes it straightforward for agents to retrieve all data about a specific
component:

```python
# Retrieve all knowledge-base entries tagged to HX-1
results = rag._kb.search("HX-1 heat exchanger primary loop", top_k=20)
hx1_entries = [r for r in results if "HX1" in r.get("source", "")]
```

---

### Summary: Interface Matrix by Lifecycle Phase

| Interface | Design | Construction | Operations | Monitoring |
|---|:---:|:---:|:---:|:---:|
| `MSR_PLANT_DATA_URL` (SCADA/historian) | | ✓ | ✓ | ✓ |
| `PlantDataLoader.ingest_text` (maintenance/commissioning reports) | | ✓ | ✓ | ✓ |
| `PlantDataLoader.ingest_sensor_snapshot` (sensor snapshots) | | ✓ | ✓ | ✓ |
| `POST /data/ingest` (async HTTP ingestion) | | ✓ | ✓ | ✓ |
| `rag.answer()` / `POST /query` (RAG knowledge retrieval) | ✓ | ✓ | ✓ | ✓ |
| `POST /kb/update` (archive + OpenAlex ingestion) | ✓ | ✓ | | |
| Sensor history buffer (`get_sensor_history`) | | | ✓ | ✓ |
| Structured `source_id` tagging (BOM / equipment tags) | | ✓ | ✓ | ✓ |
| P2V streaming adapter (EnKF assimilation loop) | | | ✓ | ✓ |
| V2P audit log (RL set-point traceability) | | | ✓ | |
| ARMI / Deep Lynx connector (`MSR_PLANT_DATA_URL` + ingestion) | ✓ | | | |
| RTL I/O / BRAM bridge (`get_sensor_history` → HLS buffer) | ✓ | | | |
| Surrogate offline archive (`operational_data` ingestion) | | | ✓ | ✓ |

---

## Design Critique: A Rickover-Memo Perspective

Admiral Hyman G. Rickover's famous memo on paper versus real reactors offers a
useful lens for any engineering system that exists primarily as a design
document or software prototype.  Rickover's core observation was that a *paper*
reactor is always simple, cheap, elegant, and satisfies every requirement,
whereas a *real* reactor is complex, expensive, messy, and barely satisfies the
requirements that actually matter.  The same distinction applies here.

> *"The educational and intellectual value of a paper reactor is zero.  It is
> the actual reactor that teaches you things."*
> — paraphrased from Rickover's January 1953 development-philosophy memo

Applied honestly to this repository, that philosophy surfaces six serious gaps
between what the README describes and what would be required in an actual
nuclear plant environment.

---

### Critique 1 — The Development Stub IS the Product

**The paper version:** "The data layer reads live sensor data from your
SCADA/historian via `MSR_PLANT_DATA_URL`."

**The real version:** When `MSR_PLANT_DATA_URL` is unset — which is the default
and the only mode exercised by every test and demo in this repository — the
system returns a hard-coded Python dictionary (`_STUB_STATE` in
`msr_mcp_server.py`) with a core temperature of 700 °C, a reactor power of
100 MW, and a flow rate of 250 kg/s.  These numbers are never validated against
any physical model.  The stub is not a temporary scaffold; it is the only
reactor this system has ever actually "read from."

**What would be required:** A real integration requires a formally specified,
version-controlled interface contract with the plant data historian, including
schema validation, connection-health monitoring, timeout and retry policies,
and an explicit failure mode that surfaces — not silently substitutes — stale
or missing data.  The failure mode of "return the stub instead of raising an
error" (`except Exception` swallowed in `_get_current_state()`) is the
opposite of what nuclear data systems require.

---

### Critique 2 — LLM Hallucination as a Feature

**The paper version:** "Multi-step RAG synthesizes a comprehensive final answer
from sub-answers and live plant data."

**The real version:** The synthesis step calls `gpt-4o-mini` (or any other
OpenAI-compatible model) with a prompt that includes live sensor readings and
asks it to answer questions such as *"Is the plant operating within safe limits?"*
(the literal example in the `__main__` block of `msr_digital_twin_with_rag.py`).
A large language model is a stochastic, non-deterministic system with no
physical model of MSR thermodynamics, no formal verification, no nuclear
qualification, and a well-documented propensity to generate confident but
incorrect statements.  Presenting its output to an operator as an answer to a
safety question is not a data layer — it is a liability.

**What would be required:** LLM output must be clearly labelled as
*unverified suggestion*, never as a safety determination.  Safety-boundary
checks must be performed by deterministic, qualified code against validated
sensor readings — not by a language model reasoning over retrieved text chunks.
The system should refuse, not attempt to answer, safety-critical queries with
an unqualified LLM.

---

### Critique 3 — Random Projections Labelled as Semantic Search

**The paper version:** "Hybrid vector + TF-IDF retrieval over the knowledge base."

**The real version:** When no `MSR_OPENAI_API_KEY` is set and no GPU is
available — again, the default state — the embedding engine is
`RandomProjectionEmbeddingEngine`, which multiplies the TF-IDF bag-of-words
vector by a fixed random matrix (Johnson–Lindenstrauss projection,
`msr_digital_twin_with_rag.py:216`).  The resulting vectors encode no learned
semantic relationships.  A query for *"fission product release during loss of
cooling"* and a query for *"FLiBe viscosity at 700 °C"* are likely to retrieve
the same documents.  The system reports no signal to the caller that semantic
search is degraded or absent; it returns results silently as if they were
meaningful.

**What would be required:** The system must make its retrieval quality
transparent.  When operating without qualified embeddings the response should
include an explicit quality warning, or the system should require a configured
embedding engine before serving queries that influence operational decisions.

---

### Critique 4 — Alarm State Exists Only in Process Memory

**The paper version:** "Get the list of currently active alarms."

**The real version:** `_ALARMS` is a Python module-level list
(`msr_mcp_server.py`).  Every time the MCP server process restarts — for any
reason, including a Lambda cold start, an OOM kill, or a deployment update —
the alarm list is wiped.  An alarm that was active before the restart does not
reappear unless the sensor threshold is re-crossed after the process starts.
There is no alarm persistence, no acknowledgement record, no alarm log, and no
operator notification pathway.

**What would be required:** In a qualified nuclear I&C system, alarm state is
held in a persistent, redundant historian with a complete audit trail of
assertion, acknowledgement, and clearance timestamps.  Process-memory alarm
state is disqualifying for any system that describes itself as serving
"safety/operational alarms."

---

### Critique 5 — The Knowledge Base is Scraped, Uncontrolled Content

**The paper version:** "Query reference documents — historical ORNL MSR reports,
academic papers, maintenance logs, and plant operational data."

**The real version:** Documents enter the knowledge base by fetching OCR text
from a public GitHub repository and downloading abstracts from the OpenAlex
academic paper API.  There is no document control, no version-locked corpus,
no verification that the retrieved text matches the authoritative source
document, and no mechanism to retract a document once ingested.  An
OCR error in a 1960s ORNL report, a paper retracted from OpenAlex, or a
corrupted JSON chunk file will silently degrade retrieval quality with no
indication to the caller.

**What would be required:** A nuclear plant knowledge base must operate under
formal document configuration management: controlled issue numbers, formal
approval signatures, explicit supersession records, and a mechanism to
invalidate or correct ingested content.  The answer to "what documents are in
the knowledge base?" must be reproducible and auditable, not "whatever the
GitHub and OpenAlex APIs returned the last time the cron job ran."

---

### Critique 6 — Third-Party Cloud Dependency During Operations

**The paper version:** "The data layer can be deployed as a serverless HTTPS
service."

**The real version:** In its primary intended configuration, the RAG synthesis
step sends plant sensor readings, alarm states, and potentially maintenance
records to `api.openai.com` — a commercial service operated by a private
company, subject to their terms of service, their data retention policies,
their network availability, and their model update schedule (which can silently
change model behaviour between calls).  No nuclear plant data governance
framework permits operational data to be sent to an unqualified, uncontrolled
external service during routine operations.

**What would be required:** Either (a) all LLM inference must be fully
air-gapped using the local GPU mode (`MSR_USE_LOCAL_GPU=true`) with a
qualified, version-pinned, formally evaluated model, or (b) the synthesis
step must be limited to retrieval-only (no LLM) during any mode where
plant-operational data is present in context.  The current default of "send
everything to OpenAI unless someone deliberately sets the local GPU flag" is
the wrong fail-safe direction.

---

### The Rickover Summary

A paper data layer is elegant, flexible, and satisfies all requirements.

A real data layer for a nuclear plant must answer four questions that this
repository does not currently address:

1. **What happens when it fails?**  Every external dependency — the plant URL,
   the OpenAI API, the GitHub archive, the S3 bucket — must have a documented,
   tested, and safe failure mode.  Silent substitution of stub data is not a
   safe failure mode.

2. **Who is responsible for the answers it gives?**  An LLM synthesising safety
   assessments from retrieved text has no responsible engineer behind it.  Every
   answer that influences an operational decision must have a qualified, named
   human or a formally certified deterministic algorithm behind it.

3. **Has it been tested under the conditions it will actually face?**  Every
   test in this repository runs against the stub, with a mock LLM, and with
   random-projection embeddings.  None of those conditions resemble a deployed
   nuclear plant environment.

4. **What does it not know, and does it say so?**  A system that returns
   confident-sounding LLM-synthesised answers without uncertainty quantification
   or retrieval-quality indicators is more dangerous than one that returns
   nothing at all.

This repository is a useful research prototype and a starting point for
thinking about MSR data architecture.  It is not — and should not be claimed
to be — production-ready for any role that touches nuclear plant operations.
Rickover would have rejected it at the first design review and sent it back
with the instruction to return when it had actually been built and run.

---

## How the MSR Data Layer Assists the Lucas et al. (2025) Corrosion Study

> **Paper:** Lucas N., Woods R., Crombleholme S., Vandanapu H., Beer C.,
> Sobel J., Steenberg T., Patel M.K. — *"Effect of Salt Purity on the
> Corrosion of 316L SS: Long-Term Studies in Molten FLiNaK and ThF₄–LiF"*,
> *Journal of Nuclear Materials* (2025), PII S0022311525007913.
> Copenhagen Atomics A/S & University of Liverpool.
>
> The paper reports 18 static-immersion corrosion tests of **316L stainless
> steel** coupons (Cr 16.9 wt%, Ni 10.7 wt%, Mo 2.6 wt%) in two molten
> fluoride salt systems — **FLiNaK at 600 °C** and **LiThF (ThF₄-LiF) at
> 700 °C** — comparing purified versus untreated salt over **1 000, 2 000,
> and 3 000 h**.  Analysis methods include ICP-OES (Cr/Fe/Ni in salt),
> mass change, SEM/EDS cross-sections, and GIXRD phase identification.

The sections below show exactly where the data layer plugs into each phase of
this experimental programme.

---

### 1 — Design Phase: Retrieving ORNL Baselines for 316L SS and INOR-8

**Paper connection:** The Introduction positions Copenhagen Atomics' 316L
work against the MSRE/MSBR heritage of **Inconel** and **INOR-8** data —
roughly 1 mm corrosion per 20 000 h.  Before running expensive 3 000 h tests
the researchers must know what the ORNL reports actually measured, and under
which salt-chemistry conditions those rates were achieved.

**Data-layer capability:** Ingest the ORNL OCR archive and query it directly.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()   # one-time; re-runs add only new files

answer = rag.answer(
    "What mass-loss or corrosion-depth data exist in the ORNL reports for "
    "316 stainless steel or INOR-8 coupons in FLiNaK at 600–700 °C? "
    "Include experiment duration, temperature, and salt purity conditions."
)
print(answer)
```

```bash
python msr_digital_twin_with_rag.py \
  "Summarise ORNL MSRE container-material corrosion data for austenitic steels"
```

This surfaces specific ORNL report numbers and measured values — avoiding
manual archive searches across dozens of 1960s technical reports — and
establishes a traceable baseline against which the new 316L data can be
benchmarked.

---

### 2 — Design Phase: Surveying Recent 316L / FLiNaK Literature

**Paper connection:** The paper cites competing work on chromium depletion in
FLiNaK and on the role of moisture-derived HF in stainless steel attack.  The
researchers need to know what corrosion depths and ICP-OES concentrations have
already been reported for 316L in fluoride salts so they can frame their
contribution and choose appropriate exposure durations.

**Data-layer capability:** The OpenAlex loader ingests papers matching MSR
corrosion queries into the same vector store as the ORNL archive.

```bash
python msr_kb_sources.py --update-openalex
```

```python
answer = rag.answer(
    "What chromium and iron dissolution rates have been reported for 316L SS "
    "in FLiNaK or FLiBe in the last 10 years? "
    "Include temperature, exposure time, and whether the salt was purified."
)
```

A single query now spans six decades of literature — ORNL reports from the
1960s and peer-reviewed papers from the 2010s–2020s — in one step.

---

### 3 — During Experiment: Logging Furnace Conditions for All 18 Tests

**Paper connection:** The 18 immersion tests (9 purified FLiNaK, 9 untreated
FLiNaK) were run at **600 °C** under **0.3 bar Ar overpressure** inside an
argon glovebox (&lt;10 ppm O₂ and H₂O).  Long-duration experiments (up to
3 000 h ≈ 125 days) accumulate furnace controller logs, thermocouple readings,
and glovebox atmosphere readings that must be preserved alongside the coupon
results.

**Data-layer capability:** `PlantDataLoader.ingest_sensor_snapshot()` stores
periodic furnace-condition records in the RAG knowledge base so they can be
co-queried with characterisation data.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Call from the DAQ script every 4 h for the duration of each test
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2024-06-01T04:00Z",
     "sensor": "furnace_temperature_c",   "value": 600.2, "unit": "°C",
     "test_id": "FLiNaK-purified-3000h"},
    {"timestamp": "2024-06-01T04:00Z",
     "sensor": "ar_overpressure_bar",     "value": 0.302, "unit": "bar",
     "test_id": "FLiNaK-purified-3000h"},
    {"timestamp": "2024-06-01T04:00Z",
     "sensor": "glovebox_o2_ppm",         "value": 7.1,   "unit": "ppm",
     "test_id": "FLiNaK-purified-3000h"},
], source_id="FLiNaK-purified-3000h-2024-06-01T04Z")
```

After 3 000 h a query like *"Were there any periods during the purified-FLiNaK
3 000 h run where the glovebox O₂ exceeded 10 ppm, and what was the furnace
temperature at those times?"* can be answered without manually scanning
controller log files.

---

### 4 — During Experiment: Ingesting Salt-Preparation and Purification Records

**Paper connection:** The key experimental variable is salt purity.  The
Copenhagen Atomics purification method (high-temperature treatment under inert
gas, resulting in moisture below detection and oxides &lt;10 ppm) is what
separates the two coupon populations.  Documenting the purification batch
record for every test tube is essential for traceability and for future
statistical analysis of purity vs. corrosion depth.

**Data-layer capability:** Free-text purification and preparation records are
ingested as event logs so they are retrievable alongside the coupon results.

```python
loader.ingest_text(
    rag,
    text=(
        "Salt batch CA-FLiNaK-P-007 (purified). "
        "Composition: LiF 46.5 mol%, NaF 11.5 mol%, KF 42 mol%. "
        "Purification: 24 h at 500 °C under Ar flow, followed by HF/H₂ sparging. "
        "Post-purification assay: moisture below detection limit (<1 ppm), "
        "oxide impurities 6 ppm (ICP-OES). "
        "Loaded into test tube T-P-07 on 2024-05-15; "
        "four 316L coupons (IDs: P07-A, P07-B, P07-C, P07-D) suspended. "
        "Test start: 2024-05-15T10:00Z. Target exposure: 3000 h at 600 °C."
    ),
    source_id="salt-batch-CA-FLiNaK-P-007",
    data_type="salt_preparation_record",
)
```

This links a coupon ID to a specific salt batch, enabling a future query like
*"Which coupons came from salt batches with oxide impurities above 8 ppm, and
what were their corrosion depths?"*

---

### 5 — After Each Time-Point: Storing ICP-OES Results

**Paper connection:** Post-test salt samples were dissolved in nitric/
hydrochloric acid and analysed by ICP-OES for **Cr, Fe, and Ni**.  The paper
reports average concentrations across the three exposure durations (Table in
Section 4.3):

| Metal | Untreated (mg/kg) | Purified (mg/kg) |
|---|---|---|
| Cr | 1 200 ± 100 | 110 ± 6 |
| Fe | 800 ± 300  | 22 ± 7  |
| Ni | 80 ± 32    | below detection |

**Data-layer capability:** Each ICP-OES result set is ingested as a structured
characterisation record so it can be queried alongside corrosion-depth and
mass-change data.

```python
loader.ingest_text(
    rag,
    text=(
        "ICP-OES salt analysis — test tube T-U-03 (untreated FLiNaK, 1000 h, 600 °C). "
        "Dissolved metals: Cr 1185 mg/kg, Fe 510 mg/kg, Ni 48 mg/kg. "
        "Analysis date: 2024-07-20. Lab: Copenhagen Atomics internal. "
        "Coupons: U03-A, U03-B, U03-C, U03-D."
    ),
    source_id="icp-oes/T-U-03/1000h",
    data_type="characterisation_report",
)
```

After all 18 tests are ingested, a single RAG query surfaces the full Cr/Fe/Ni
dissolution trend across the purified vs. untreated series and all three
time-points — the same analysis the paper presents in Section 4.3, but
queryable in plain language.

---

### 6 — After Each Time-Point: Storing Mass-Change and Corrosion-Depth Records

**Paper connection:** The paper reports coupon mass change (untreated salt
~194× greater loss than purified) and SEM-measured corrosion depths (untreated
68.5 → 112.1 µm vs. purified 2.1 → 3.0 µm over 1 000–3 000 h), with
ImageJ used for depth measurements on cross-sectional SEM images.

**Data-layer capability:** Mass and depth records are ingested per coupon per
time-point with a consistent `source_id` scheme enabling cross-series
comparisons.

```python
# Mass-change record after the 1000 h untreated-FLiNaK test teardown
loader.ingest_text(
    rag,
    text=(
        "Mass change — coupon U03-A (untreated FLiNaK, 1000 h, 600 °C). "
        "Pre-exposure mass: 14.823 g. Post-exposure mass: 14.695 g. "
        "Mass loss: 128 mg. Coupon area: 13.24 cm². "
        "Specific mass loss: 9.67 mg/cm²."
    ),
    source_id="mass-change/T-U-03/U03-A/1000h",
    data_type="characterisation_report",
)

# SEM corrosion depth from ImageJ measurement
loader.ingest_text(
    rag,
    text=(
        "SEM cross-section — coupon U03-A (untreated FLiNaK, 1000 h, 600 °C). "
        "Intergranular corrosion observed. Maximum corrosion depth (ImageJ): 71 µm. "
        "Mean corrosion depth: 68.5 µm. "
        "Attack mode: intergranular; no uniform dissolution. "
        "Cr-depleted zone confirmed by EDS line scan."
    ),
    source_id="sem/T-U-03/U03-A/1000h",
    data_type="characterisation_report",
)
```

---

### 7 — Post-Exposure: Storing GIXRD Phase-Identification Results

**Paper connection:** GIXRD identified key phases that explain the mechanistic
difference between purified and untreated salts:

* **Purified FLiNaK coupons:** Cr₇C₃ and Cr₂₃C₆ (chromium carbides) —
  hypothesised to act as diffusion barriers.
* **Untreated FLiNaK coupons:** FeCr₂O₄ spinel, KF/K-Cr-F compounds, and
  γ-Fe → α-Fe transformation (austenite to ferrite) from Cr and Ni depletion.

**Data-layer capability:** Phase-identification results are ingested as
structured text, linking phase names to the coupon, salt condition, and
exposure duration, so that future queries can reason over mechanism.

```python
loader.ingest_text(
    rag,
    text=(
        "GIXRD phase identification — coupon P07-B (purified FLiNaK, 3000 h, 600 °C). "
        "Phases detected: Cr₇C₃ (chromium carbide), Cr₂₃C₆ (chromium carbide), "
        "γ-Fe (austenite, matrix retained). "
        "No FeCr₂O₄ detected. No KF or K-Cr-F compounds. "
        "Interpretation: Cr carbide surface film may act as diffusion barrier "
        "limiting further Cr dissolution into the salt."
    ),
    source_id="gixrd/T-P-07/P07-B/3000h",
    data_type="characterisation_report",
)

loader.ingest_text(
    rag,
    text=(
        "GIXRD phase identification — coupon U03-C (untreated FLiNaK, 3000 h, 600 °C). "
        "Phases detected: FeCr₂O₄ (spinel), KF, K-Cr-F compounds, α-Fe (ferrite). "
        "Original γ-Fe austenite peak greatly reduced — consistent with "
        "Cr and Ni depletion driving γ-to-α transformation. "
        "Interpretation: impurity-driven oxide dissolution removes passive Cr₂O₃ "
        "layer, exposing alloy to further fluoride attack."
    ),
    source_id="gixrd/T-U-03/U03-C/3000h",
    data_type="characterisation_report",
)
```

Once all GIXRD records are ingested, a query like *"Which coupons retained
austenite after 3 000 h and what were their corresponding salt-Cr concentrations?"*
draws on both the GIXRD records and the ICP-OES records in one answer.

---

### 8 — Cross-Experiment Analysis: Querying the Full 18-Test Dataset

**Paper connection:** The paper's main finding is the ~33× difference in
corrosion depth and ~194× difference in mass loss between untreated and
purified salt, and the saturation of corrosion depth in untreated salt between
2 000 and 3 000 h.  This is derived by cross-referencing results from all 18
tests.

**Data-layer capability:** Once all ICP-OES, mass-change, SEM, and GIXRD
records are ingested, the RAG pipeline can synthesise cross-test comparisons in
plain language.

```python
answer = rag.answer(
    "Summarise the chromium depletion depth and dissolved Cr concentration "
    "in the salt for all 316L SS coupons tested in FLiNaK, grouped by "
    "salt condition (purified vs. untreated) and exposure time. "
    "Does the untreated-salt corrosion depth appear to plateau after 2000 h?"
)
print(answer)
```

```python
answer = rag.answer(
    "Compare the GIXRD phases found in purified vs. untreated FLiNaK coupons. "
    "Which phases are unique to untreated-salt coupons and which are unique "
    "to purified-salt coupons? What mechanistic interpretation does this support?"
)
```

This supports the paper's Discussion section — and also supports future
researchers replicating or extending the work, who need to understand whether
new data points are consistent with the existing dataset.

---

### 9 — Future Work: Contextualising UF₃/UF₄ and Fission-Product Extensions

**Paper connection:** Section 6 (Conclusion) explicitly lists four directions
for future work: radiation effects, fission-product chemistry, **UF₃/UF₄
additions**, and temperature/flow gradients.  These represent the next
experimental programme.

**Data-layer capability:** The same data layer serves as the foundation for the
follow-on programme.  ORNL archive reports on uranium-bearing salts (MSRE ran
with UF₄ dissolved in FLiBe) and OpenAlex papers on fission-product speciation
are already in the knowledge base.

```python
# Before designing the UF4-doped FLiNaK tests:
answer = rag.answer(
    "What effect did UF4 additions have on the corrosion rate of structural "
    "alloys in the MSRE? What U/U4+ redox ratio was maintained, and how was "
    "it controlled? Were there 316 SS or stainless steel tests in uranium-bearing salts?"
)

answer = rag.answer(
    "What tellurium and cesium speciation data exists for FLiNaK at 600 °C "
    "in the ORNL reports? How were fission-product impurities handled "
    "during the MSRE purification cycles?"
)
```

This gives the Copenhagen Atomics team a head-start on experimental design for
the follow-on UF₄ and fission-product tests — grounded in six decades of ORNL
operational data — before a single new test begins.

---

### Summary: Lucas et al. (2025) Experimental Workflow × Data-Layer Capability

| Experimental step (paper section) | Data ingested | Data-layer capability |
|---|---|---|
| Design — ORNL baseline (§1 Intro) | ORNL MSRE/MSBR reports | `rag.load_msr_archive()` |
| Design — recent literature (§1 Intro) | OpenAlex 316L/FLiNaK papers | `python msr_kb_sources.py --update-openalex` |
| During — furnace conditions (§2.1) | Temp, Ar pressure, glovebox O₂ per test | `loader.ingest_sensor_snapshot()` |
| During — salt prep records (§2.1) | Batch IDs, purity assay, impurity levels | `loader.ingest_text()` (salt_preparation_record) |
| Post-test — ICP-OES (§4.3) | Dissolved Cr/Fe/Ni per test tube per time-point | `loader.ingest_text()` (characterisation_report) |
| Post-test — mass change (§4.5) | Pre/post mass, specific mass loss per coupon | `loader.ingest_text()` (characterisation_report) |
| Post-test — SEM depth (§4.4) | Max and mean corrosion depth, attack mode | `loader.ingest_text()` (characterisation_report) |
| Post-test — GIXRD phases (§4.6) | Phase names, mechanistic interpretation | `loader.ingest_text()` (characterisation_report) |
| Analysis — cross-test comparison (§5) | Full 18-test dataset | `rag.answer()` |
| Future work — UF₄ / fission products (§6) | ORNL uranium-salt + fission-product data | `rag.answer()` over existing archive |

---

## Documentation

* [00_MCP_START_HERE.md](00_MCP_START_HERE.md) – five-minute quick start
* [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md) – full guide with RAG pipeline details
* [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md) – deployment guide

---

## License

MIT
