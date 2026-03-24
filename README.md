# MSR Data Layer

A **reference data layer for Molten Salt Reactor (MSR) design, construction, and
operations** — powered by your GitHub Copilot subscription, zero cloud bills.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/pranavkantgaur/msr_data_layer)

> **One-click demo** — click the badge above, wait ~60 s, and a public
> `https://<codespace>-8000.app.github.dev` URL appears in the *Ports* panel.
> Paste it into the [demo notebook](use_cases/lucas_et_al_2025_demo.ipynb) and
> run it end-to-end in Colab.  No AWS account, no API key purchase needed —
> your GitHub Copilot Pro subscription covers everything.

---

## What it does

The data layer gives LLM agents and human operators a single interface to:

* **Query research knowledge** — ORNL MSR reports (full text), auto-fetched
  paper abstracts (OpenAlex / arXiv / Semantic Scholar), and user-provided
  full paper text — all searchable via multi-step RAG.
* **Ingest plant timeseries** — push timestamped sensor readings from
  SCADA/instruments and query them with structured filters or plain English
  (NL→SQL backed by SQLite).
* **Ingest operational records** — event logs, maintenance reports, ICP-OES
  results, corrosion measurements, etc.
* **Read live sensor state** — from an external plant REST API or a built-in
  development stub.

> **Not** a digital twin, simulation, or control system.  This is read +
> ingest only.  No actuation tools.

---

## Two interfaces, one KB

```
┌───────────────────────────────────────────────────────────────────┐
│                     Shared KB + Timeseries Store                   │
│          (./kb_store/  +  plant_timeseries.db  — SQLite)          │
└──────────────────────┬────────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   ┌──────▼──────┐           ┌──────▼──────────────┐
   │  server.py  │           │ msr_mcp_server_      │
   │  HTTP :8000 │           │ main.py  (stdio)     │
   │             │           │                      │
   │ Human ops   │           │ AI agents:           │
   │ Notebooks   │           │ GitHub Copilot Chat  │
   │ REST APIs   │           │ Claude Desktop       │
   └─────────────┘           │ Custom MCP clients   │
                             └──────────────────────┘
```

| Component | Start command | Used by |
|---|---|---|
| `server.py` | `make serve` or `python server.py` | Demo notebooks, REST clients, human operators |
| `msr_mcp_server_main.py` | `make serve-mcp` or `python msr_mcp_server_main.py` | AI agents via MCP protocol |

Both components share the same KB files — you can run one, the other, or both
simultaneously.

---

## Quick start (GitHub Codespaces — recommended)

1. Click **"Open in GitHub Codespaces"** above (or badge in the GitHub UI).
2. Wait ~60 s for the container to build and `server.py` to start.
3. In the *Ports* tab, the port 8000 URL is already **public** — copy it.
4. Open [`use_cases/lucas_et_al_2025_demo.ipynb`](use_cases/lucas_et_al_2025_demo.ipynb)
   in Google Colab, paste the URL, and run all cells.

The server uses your Codespace's `GITHUB_TOKEN` automatically — no extra
setup required.

## Quick start (local)

```bash
# Install dependencies
pip install -r requirements_mcp.txt

# Start the HTTP server (port 8000)
make serve              # same as: python server.py

# Or start the MCP stdio server for AI agents
make serve-mcp          # same as: python msr_mcp_server_main.py

# Run tests
make test
```

Set `MSR_GITHUB_TOKEN` to your GitHub PAT (with `models:read` scope) or
`MSR_OPENAI_API_KEY` for LLM-backed RAG synthesis and NL→SQL.  Without these,
the service works in stub mode (random-projection embeddings, no synthesis).

---

## Repository layout

| File / dir | Description |
|---|---|
| `server.py` | **Primary HTTP server** — wraps lambda_function.py; start with `make serve` |
| `msr_mcp_server_main.py` | **MCP stdio server** — for AI agents; start with `make serve-mcp` |
| `msr_mcp_server.py` | MCP tool definitions + handler functions |
| `msr_digital_twin_with_rag.py` | Multi-step RAG pipeline (chunking, embeddings, retrieval, synthesis) |
| `msr_kb_sources.py` | KB loaders: ORNL archive, OpenAlex, arXiv, Semantic Scholar, plant data, `TimeseriesStore` |
| `lambda_function.py` | HTTP router (used by `server.py`; also AWS Lambda handler — optional) |
| `template.yaml` | AWS SAM template (optional cloud deployment) |
| `Dockerfile.gpu` | GPU container image (optional) |
| `Makefile` | All build/test/serve targets |
| `requirements_mcp.txt` | Core Python deps (no GPU, no AWS) |
| `requirements_lambda.txt` | Lambda-specific deps (adds boto3) |
| `requirements_gpu.txt` | GPU deps (torch, sentence-transformers) |
| `.devcontainer/` | GitHub Codespaces auto-start config |
| `use_cases/` | Demo notebooks and worked examples |
| `docs/architecture/` | Architecture diagrams (Mermaid) |
| `.ai/` | Agent grounding documents |
| `test_*.py` | Unit tests (pytest) |

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

## GitHub Codespaces Deployment (GitHub Copilot Pro – no extra subscription required)

If you have a **GitHub Copilot Pro** subscription you can deploy the MSR data
layer on a publicly accessible HTTPS URL using **GitHub Codespaces** and the
**GitHub Models API**, with no AWS, Azure, or OpenAI accounts required.

### How it works

| GitHub feature | Role |
|---|---|
| **GitHub Codespaces** | Runs the Python HTTP server in a managed cloud VM |
| **Port forwarding** | Exposes `server.py` on a public `*.preview.app.github.dev` URL |
| **GitHub Models API** | Provides `gpt-4o-mini` (chat) and `text-embedding-3-small` (embeddings) via `GITHUB_TOKEN` |

The `GITHUB_TOKEN` secret is injected automatically into every Codespace.
`server.py` forwards it to `MSR_GITHUB_TOKEN`, which the RAG pipeline uses to
call `https://models.inference.ai.azure.com` – the same OpenAI-compatible
endpoint used by GitHub Copilot Chat.

### Deploy in 4 steps

**Step 1 – Open a Codespace**

Go to your fork of this repository on GitHub, click **Code → Codespaces →
Create codespace on main** (or your working branch).  The
`.devcontainer/devcontainer.json` configuration installs Python dependencies
and starts `server.py` automatically when the Codespace boots.  Port 8000 is
marked as **public** automatically, so no extra "Make Public" step is needed.

**Step 2 – Wait for the server to start**

The startup script (`.devcontainer/start-server.sh`) runs `nohup` so the
server process keeps running after the postStartCommand shell exits.  Check
that the server is up:

```bash
curl http://localhost:8000/health
# → {"status":"healthy", ...}

# Server log:
tail -f /tmp/msr_server.log
```

**Step 3 – Get the public URL**

In VS Code's **Ports** panel (or the Codespace details page) look for
port **8000 – MSR Data Layer API**.  Its public URL is:

```
https://<codespace-name>-8000.app.github.dev
```

**Step 4 – Query the data layer**

```bash
BASE_URL="https://<codespace-name>-8000.app.github.dev"

# Health check
curl "$BASE_URL/health"

# RAG query (uses GitHub Models gpt-4o-mini for synthesis)
curl -X POST "$BASE_URL/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key operational parameters of the TMSR-LF1 reactor?"}'

# MCP JSON-RPC (for MCP-compatible clients)
curl -X POST "$BASE_URL/mcp" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_reactor_status","arguments":{}}}'
```

**Step 5 – (Optional) Protect with an API key**

Add a Codespace secret named `MSR_API_KEY` in
**GitHub → Settings → Codespaces → Secrets**.  All requests will then require
an `X-Api-Key: <your-key>` header.

### Manual start / restart

If the server is not running (e.g. after the Codespace was paused), restart it
from the terminal:

```bash
bash .devcontainer/start-server.sh
# Stops any previous instance, starts a fresh one with nohup.
# Logs: /tmp/msr_server.log   PID: /tmp/msr_server.pid
```

### Environment variables

All variables are configured via `.devcontainer/devcontainer.json` or as
Codespace secrets:

| Variable | Default | Description |
|---|---|---|
| `MSR_GITHUB_TOKEN` | auto from `GITHUB_TOKEN` | GitHub PAT for GitHub Models API |
| `MSR_OPENAI_MODEL` | `gpt-4o-mini` | Chat model (any GitHub Models chat model) |
| `MSR_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `MSR_API_KEY` | (unset) | Optional bearer token for endpoint auth |
| `MSR_PLANT_DATA_URL` | (unset) | URL of external SCADA/historian REST API |
| `MSR_KB_DIR` | `/workspaces/msr_data_layer/kb_store` | Persistent KB directory |
| `MSR_SERVER_PORT` | `8000` | HTTP server port |

### GitHub Models rate limits

GitHub Copilot Pro includes access to GitHub Models with the following
approximate limits (subject to change; see
[GitHub Models docs](https://docs.github.com/en/github-models/prototyping-with-ai-models)):

- **gpt-4o-mini**: ~150 requests/day, ~15 requests/minute
- **text-embedding-3-small**: ~150 requests/day, ~15 requests/minute

For higher throughput, set `MSR_OPENAI_API_KEY` to use a paid OpenAI key
instead (the same `MSR_OPENAI_BASE_URL`/`MSR_OPENAI_MODEL` env vars apply).

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

## MCP Tools (read + timeseries + ingestion)

| Tool | Type | Description |
|---|---|---|
| `get_reactor_status` | Read | Current status, power, core temperature, data source |
| `get_sensor_reading` | Read | Single named sensor value |
| `get_all_sensor_readings` | Read | All sensors at once |
| `get_sensor_history` | Read | Historical readings (up to last 100 samples) |
| `get_active_alarms` | Read | List active alarms |
| `get_data_source_info` | Read | Data source mode, URL, connectivity status |
| `query_sensor_timeseries` | Timeseries | Time-range or latest-N raw sensor readings from SQLite |
| `get_sensor_stats` | Timeseries | avg / min / max / count aggregates over a sensor |
| `query_plant_data_nl` | Timeseries | Natural-language question → LLM-generated SQL → results |
| `ingest_plant_data` | Write | Push operational text/logs into the RAG knowledge base |
| `ingest_full_paper_text` | Write | Upgrade an abstract-only KB entry to full paper text |

> Simulation tools (`run_thermal_simulation`) and control-actuation tools
> (`set_control_rod_position`, `acknowledge_alarm`) are **not** included.
> This is a data layer; simulation and control live in the digital twin or
> process-control system.

---

## HTTP Endpoints

All endpoints available at the Codespace public URL or `http://localhost:8000`:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/mcp` | JSON-RPC 2.0 MCP protocol |
| `POST` | `/query` | Plain-text RAG question → answer |
| `POST` | `/kb/update` | Trigger KB ingestion (archive / papers) |
| `POST` | `/data/ingest` | Push plant operational data (text / event logs) |
| `POST` | `/timeseries/ingest` | Push timestamped sensor readings |
| `POST` | `/timeseries/query` | Query sensor timeseries (structured or natural language) |

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║               Shared Knowledge Base + Timeseries Store            ║
║   ./kb_store/ (vector + TF-IDF)  +  plant_timeseries.db (SQLite) ║
╚══════════════╤═══════════════════════════════╤════════════════════╝
               │                               │
    ┌──────────▼──────────┐        ┌───────────▼──────────────┐
    │    server.py        │        │  msr_mcp_server_main.py  │
    │    HTTP :8000       │        │  (stdio MCP transport)   │
    │    make serve       │        │  make serve-mcp          │
    │                     │        │                          │
    │  POST /query        │        │  GitHub Copilot Chat     │
    │  POST /data/ingest  │        │  Claude Desktop          │
    │  POST /timeseries/* │        │  Custom MCP agents       │
    │  GET  /health       │        └──────────────────────────┘
    └─────────────────────┘

msr_digital_twin_with_rag.py  (RAG pipeline)
  ├── LocalGPUEmbeddingEngine   (optional – sentence-transformers)
  ├── OpenAIEmbeddingEngine     (GitHub Models / OpenAI API)
  ├── RandomProjectionEngine    (numpy fallback – zero deps)
  └── KnowledgeBase             (hybrid dense+TF-IDF, persistent)

msr_kb_sources.py  (data loaders)
  ├── MSRArchiveLoader     – ORNL reports full text (msr-archive)
  ├── OpenAlexLoader       – paper abstracts (auto; full text on request)
  ├── ArXivLoader          – paper abstracts (auto; full text on request)
  ├── SemanticScholarLoader– paper abstracts (auto; full text on request)
  ├── PlantDataLoader      – operational records, event logs
  └── TimeseriesStore      – SQLite: timestamped sensor readings
```

---

## Testing

```bash
make test
# or: python -m pytest test_*.py -v
```

204+ unit tests covering all components. Run with `--tb=short` for brevity.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MSR_GITHUB_TOKEN` | _(unset)_ | **Recommended**: GitHub PAT for GitHub Models API (free with Copilot Pro) |
| `MSR_OPENAI_API_KEY` | _(unset)_ | OpenAI-compatible API key (overrides GITHUB_TOKEN) |
| `MSR_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `MSR_OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `MSR_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `MSR_DOCS_DIR` | `./docs` | Reference documents directory |
| `MSR_KB_DIR` | `./kb_store` | Persistent knowledge-base directory |
| `MSR_PLANT_DATA_URL` | _(unset)_ | External plant data REST API URL |
| `MSR_USE_LOCAL_GPU` | `false` | `true` to use local GPU (sentence-transformers) |
| `MSR_ARXIV_MAX_RESULTS` | `100` | Max arXiv papers per run |
| `MSR_S2_API_KEY` | _(unset)_ | Semantic Scholar API key (100 req/s vs 1 req/s) |
| `MSR_S2_MAX_RESULTS` | `100` | Max S2 papers per run |
| `MSR_API_KEY` | _(unset)_ | Shared secret for HTTP endpoint auth (`X-Api-Key` header) |
| `MSR_SERVER_HOST` | `0.0.0.0` | `server.py` bind address |
| `MSR_SERVER_PORT` | `8000` | `server.py` TCP port |

In GitHub Codespaces, `GITHUB_TOKEN` is injected automatically and forwarded
to `MSR_GITHUB_TOKEN` by both the devcontainer config and `server.py`.

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

## Use-Case Case Studies

The [`use_cases/`](use_cases/) directory contains case studies showing how
the data layer supports published MSR experimental work.  Each file maps a
specific paper to a data-layer capability with runnable code examples.

| Case study | Paper | Topic |
|---|---|---|
| [lucas_2025_316L_flinak_corrosion.md](use_cases/lucas_2025_316L_flinak_corrosion.md) | Lucas et al. (2025), *J. Nucl. Mater.* | 316L SS corrosion in FLiNaK / LiThF |
| [haubenreich_engel_1970_msre_operations.md](use_cases/haubenreich_engel_1970_msre_operations.md) | Haubenreich & Engel (1970) | MSRE full operating experience |
| [koger_1972_hastelloy_n_corrosion.md](use_cases/koger_1972_hastelloy_n_corrosion.md) | Koger (1972), ORNL-TM-4273 | Hastelloy N corrosion / mass transfer |
| [cantor_1968_fluoride_salt_properties.md](use_cases/cantor_1968_fluoride_salt_properties.md) | Cantor et al. (1968), ORNL-4229 | Physical properties of fluoride salts |
| [mccoy_1970_tellurium_embrittlement.md](use_cases/mccoy_1970_tellurium_embrittlement.md) | McCoy et al. (1970) | Tellurium embrittlement of Hastelloy N |
| [baes_1974_redox_chemistry.md](use_cases/baes_1974_redox_chemistry.md) | Baes (1974), *J. Nucl. Mater.* | Redox chemistry and UF₃/UF₄ control |

See [`use_cases/README.md`](use_cases/README.md) for instructions on adding
new case studies.

---

## Documentation

* [00_MCP_START_HERE.md](00_MCP_START_HERE.md) – five-minute quick start
* [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md) – full guide: architecture, all 7 tools, RAG pipeline, what the data layer does *not* do
* [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md) – deployment guide

---

## License

MIT
