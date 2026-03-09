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

## Supporting MSR Experimental Work: J. Nucl. Mater. (2025) Mapping

> **Article reference:** *Journal of Nuclear Materials*, 2025,
> doi:10.1016/j.jnucmat.2025.155737 (PII S0022311525007913)
>
> The Journal of Nuclear Materials (ISSN 0022-3115) publishes experimental
> materials-science work directly relevant to molten salt reactor development:
> corrosion of structural alloys in fluoride/chloride salts, fission-product
> speciation and transport, thermophysical property measurements of fuel and
> coolant salts, redox-potential electrochemistry, and radiation-effects
> characterisation.  The sections below map each major experimental workflow
> phase to a specific data-layer capability.

---

### 1 — Pre-Experiment: Historical Context from ORNL Reports

**Experimental need:** Before designing a new corrosion coupon experiment or
salt-chemistry study, researchers must survey prior ORNL results — loop
corrosion data from the 1960s–70s MSRE/MSBR programmes, tellurium
embrittlement studies, fission-product volatility measurements — to avoid
re-discovering known failure modes and to set realistic baseline values.

**Data-layer capability:** The RAG pipeline pre-loads OCR transcripts from the
`pranavkantgaur/msr-archive` GitHub repository, which contains the primary
ORNL MSR technical reports.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()   # one-time ingest; re-run adds only new files

# Query historical corrosion measurements
answer = rag.answer(
    "What corrosion rates were measured for Hastelloy N coupons in "
    "FLiBe-UF4 at 650–700 °C during MSRE loop tests?"
)
print(answer)
```

```bash
# CLI equivalent
python msr_digital_twin_with_rag.py \
  "Summarise ORNL fission-product volatility data for tellurium in FLiBe"
```

This surfaces specific ORNL report numbers, measured values, and experimental
conditions — giving the experimentalist a traceable prior-art baseline in
seconds rather than hours of manual literature search.

---

### 2 — Pre-Experiment: Recent Literature Survey via OpenAlex

**Experimental need:** Knowing the current state of the art — which alloy
compositions have been tested in FLiNaK since 2010, what redox-control
strategies have been validated at bench scale — helps the researcher calibrate
measurement uncertainty and choose appropriate control conditions.

**Data-layer capability:** The OpenAlex loader queries the academic paper API
for MSR-relevant experimental papers and ingests their abstracts into the
knowledge base alongside the ORNL archive.

```bash
# Pull all new MSR experimental papers from OpenAlex
python msr_kb_sources.py --update-openalex
```

```python
# After update, query spans both ORNL reports and recent literature
answer = rag.answer(
    "What chromium depletion depths have been reported for 316L SS "
    "after 500 h immersion in FLiBe at 700 °C?"
)
```

Because both the ORNL archive and OpenAlex papers live in the same vector
store, a single query retrieves corroborating or contradicting evidence from
six decades of published experimental work.

---

### 3 — During Experiment: Continuous Sensor Data Ingestion

**Experimental need:** A molten-salt loop experiment typically runs continuously
for hundreds to thousands of hours, generating time-series data from multiple
sensors: furnace temperature controllers, thermal mass-flow meters, in-line
redox probes (Pt/Ni reference electrodes), pressure transducers, and
occasionally online gamma spectrometers tracking noble-metal fission products.
This data must be stored in a queryable form alongside the experimental
narrative.

**Data-layer capability:** `PlantDataLoader.ingest_sensor_snapshot()` ingests
structured sensor readings directly into the RAG knowledge base so that
subsequent natural-language queries can reason over measured time-series.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# After each measurement cycle (e.g., every 4 h)
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2025-03-01T04:00Z", "sensor": "loop_temperature_c",
     "value": 698.4, "unit": "°C", "location": "hot-leg"},
    {"timestamp": "2025-03-01T04:00Z", "sensor": "redox_potential_mv",
     "value": -342.1, "unit": "mV", "electrode": "Pt/Ni"},
    {"timestamp": "2025-03-01T04:00Z", "sensor": "corrosion_current_ua",
     "value": 8.3, "unit": "µA", "coupon": "IN617-A"},
], source_id="loop-run-007-2025-03-01T04Z")
```

Via the Lambda HTTP endpoint from a LabVIEW or Python DAQ script:

```bash
curl -X POST https://<api-gw>/prod/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "04:00 UTC — hot-leg 698.4°C, redox −342 mV, coupon IN617-A at 8.3 µA",
    "data_type": "sensor_snapshot",
    "source_id": "loop-run-007-2025-03-01T04Z"
  }'
```

Once ingested, any future query such as *"When did the redox potential cross
−350 mV and what was the simultaneous corrosion current?"* is answerable by
the RAG pipeline without manual log-file searching.

---

### 4 — During Experiment: Shift-Log and Observation Ingestion

**Experimental need:** In a long-duration loop test, shift technicians record
qualitative observations that are critical for interpretation but difficult to
query retrospectively: *"salt level in the pump bowl dropped 3 mm; likely
micro-leak at the upper flange gasket"*, *"faint white deposit on the cold-leg
viewing window"*.  These notes are typically kept in paper log books or flat
text files and are rarely cross-referenced with sensor data.

**Data-layer capability:** Free-text maintenance and event logs can be ingested
alongside structured sensor data so that the RAG pipeline surfaces them in
context.

```python
loader.ingest_text(
    rag,
    text=(
        "Shift log 2025-03-15 06:00 UTC — Operator noted faint white "
        "crystalline deposit on cold-leg observation window (location CL-4). "
        "Loop temperature 690 °C. Salt level unchanged. Suspect UF4 "
        "precipitation at cold-leg minimum temperature 585 °C.  "
        "Salt sample drawn (SSC-007-015)."
    ),
    source_id="shift-log-20250315T0600Z",
    data_type="event_log",
)
```

A subsequent RAG query such as *"List all instances of solid precipitation
observed in the cold leg and the associated temperature at the time"* will
surface this entry alongside any matching historical ORNL observations.

---

### 5 — Post-Experiment: Coupon Characterisation Data Storage

**Experimental need:** After loop teardown, post-irradiation examination (PIE)
or post-exposure characterisation generates structured results: SEM/EDS
elemental profiles, mass-loss measurements, XRD phase identification, tensile
test curves.  These results need to be stored in a form that supports
cross-experiment comparison.

**Data-layer capability:** Characterisation reports are ingested as structured
documents with equipment-tagged `source_id` values following the convention
`<experiment-id>/<coupon-id>/<technique>`.

```python
loader.ingest_text(
    rag,
    text=(
        "Post-exposure SEM/EDS of coupon IN617-A (loop-run-007, 1000 h at "
        "700 °C in FLiBe-UF4 with U/U4+ = 0.01). Cr depletion zone: "
        "42 ± 3 µm. Mo enrichment at grain boundaries observed to 8 at%. "
        "Mass loss: 1.34 mg/cm². No evidence of intergranular attack. "
        "Phase ID by XRD: gamma-Ni matrix + CrF2 surface film."
    ),
    source_id="loop-run-007/IN617-A/SEM-EDS",
    data_type="characterisation_report",
)
```

After ingesting results from multiple coupons and multiple runs, a query such
as *"How does the chromium depletion depth in IN617 vary with U/U4+ ratio
across all loop runs?"* synthesises the stored characterisation records into a
comparison table.

---

### 6 — Post-Experiment: AI-Assisted Interpretation via MCP Tools

**Experimental need:** Once sensor time-series, shift logs, and characterisation
data are all in the knowledge base, the experimentalist needs to query across
all of them simultaneously — correlating redox-probe readings with observed
corrosion rates, identifying the onset time of accelerated attack, and
contextualising the results against ORNL historical benchmarks.

**Data-layer capability:** The seven MCP tools are the primary interface for
AI-agent-assisted interpretation.  An LLM agent connected via the Model Context
Protocol can call these tools in sequence to synthesise a complete picture:

```
Agent workflow for post-experiment interpretation:

1. get_all_sensor_readings      → confirm current loop status / end-of-run state
2. get_active_alarms            → check for any threshold violations in sensor history
3. rag.answer(question)         → query across archive + papers + ingested experiment data
4. ingest_plant_data            → store the agent's synthesised interpretation report
```

Example Claude/Copilot prompt, with MCP tools active:

```
Using the MSR data layer tools:
1. Search the knowledge base for all corrosion-rate measurements on IN617
   in FLiBe-UF4 between 650 °C and 720 °C.
2. Retrieve the sensor history for the redox_potential_mv sensor from
   loop-run-007.
3. Correlate the redox excursions above −300 mV with the peak corrosion
   current events and compare against the ORNL MSRE baseline.
4. Store a summary of this analysis in the knowledge base for future reference.
```

---

### 7 — Data Governance: Reproducible Experimental Record

**Experimental need:** A peer-reviewed experimental paper requires a
reproducible data record — every ingested document must be traceable to an
exact source, and the knowledge base state at the time of the paper's analysis
must be reconstructable.

**Data-layer capability:** The state-tracking files (`archive_state.json`,
`openalex_state.json`, `plant_data_state.json`) record the exact URL or ID of
every ingested document.  Committing the `kb_store/` state files alongside the
paper's supplementary data provides a complete provenance record.

```bash
# Show exactly what is in the knowledge base
python msr_kb_sources.py --status

# Example output:
# msr-archive:  127 files ingested
# openalex:      84 papers ingested
# plant-data:   312 records ingested
```

> **Caveat (Rickover):** State-file provenance records *which URLs were fetched*
> but not *what those URLs returned at the time of fetch*.  For archival
> reproducibility, snapshots of the raw OCR text and paper abstracts should be
> committed to a version-controlled supplementary data repository.

---

### Summary: Experimental Workflow × Data-Layer Capability Matrix

| Experimental phase | Data-layer capability | Key API / CLI |
|---|---|---|
| Pre-experiment: ORNL context | RAG over msr-archive OCR | `rag.load_msr_archive()` |
| Pre-experiment: literature survey | RAG over OpenAlex papers | `python msr_kb_sources.py --update-openalex` |
| During: sensor time-series | `ingest_sensor_snapshot()` | `POST /data/ingest` |
| During: shift-log observations | `ingest_text()` | `POST /data/ingest` |
| Post: characterisation reports | `ingest_text()` with tagged `source_id` | `PlantDataLoader` |
| Post: cross-experiment query | RAG `answer()` over full KB | `rag.answer(question)` |
| Post: AI-agent interpretation | MCP tools (read + ingest) | `get_sensor_history`, `get_active_alarms` |
| Publication: provenance | State-file snapshot | `python msr_kb_sources.py --status` |

---

## Documentation

* [00_MCP_START_HERE.md](00_MCP_START_HERE.md) – five-minute quick start
* [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md) – full guide with RAG pipeline details
* [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md) – deployment guide

---

## License

MIT
