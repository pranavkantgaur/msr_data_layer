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

## Documentation

* [00_MCP_START_HERE.md](00_MCP_START_HERE.md) – five-minute quick start
* [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md) – full guide with RAG pipeline details
* [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md) – deployment guide

---

## License

MIT
