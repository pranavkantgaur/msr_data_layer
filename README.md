# MSR Data Layer

A **Molten Salt Reactor (MSR) digital twin** exposed through the
[Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io),
enabling LLM agents (Claude, GitHub Copilot, custom agents) to query
live reactor sensor data, run thermal-hydraulic simulations, monitor
alarms, and retrieve answers from technical documents via an enhanced
multi-step Retrieval-Augmented Generation (RAG) pipeline.

The RAG implementation is inspired by the
[open-notebook](https://github.com/lfnovo/open-notebook) project,
adopting its multi-query decomposition, source-insight extraction, and
parallel-search approach.

---

## Repository Contents

| File | Description |
|---|---|
| `msr_mcp_server.py` | Core MCP server – tool handlers, JSON-RPC dispatcher |
| `msr_mcp_server_main.py` | Entry point – stdio transport server |
| `msr_digital_twin_client.py` | Python client for the MCP server |
| `msr_digital_twin_with_rag.py` | Enhanced multi-step RAG pipeline |
| `msr_kb_sources.py` | KB source loaders: msr-archive (static) + OpenAlex (dynamic) |
| `requirements_mcp.txt` | Python dependencies |
| `00_MCP_START_HERE.md` | Quick-start guide |
| `MSR_DIGITAL_TWIN_MCP_GUIDE.md` | Full architecture and tool reference |
| `MSR_MCP_DEPLOYMENT_GUIDE.md` | Deployment and production guide |
| `test_msr_mcp_server.py` | Unit tests for MCP server |
| `test_msr_rag.py` | Unit tests for enhanced RAG pipeline |
| `test_msr_kb_sources.py` | Unit tests for KB source loaders |

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
python msr_digital_twin_with_rag.py "Is the reactor operating within safe limits?"
```

### 4 – Connect Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "msr-digital-twin": {
      "command": "python",
      "args": ["/path/to/msr_mcp_server_main.py"]
    }
  }
}
```

---

## Enhanced RAG Pipeline

The RAG pipeline adopts the multi-step approach from
[open-notebook](https://github.com/lfnovo/open-notebook):

```
Ingestion:
  Document ──► sentence-aware chunking
           ──► dense embeddings (OpenAI API or random-projection fallback)
           ──► source insights (LLM: summary + topics + key_facts)
           ──► persistent KnowledgeBase (JSON + numpy)

Retrieval (per question):
  Question ──► [Step 1] QueryDecomposer (LLM) ──► up to 5 SubQueries
           ──► [Step 2] Parallel hybrid search (dense cosine + TF-IDF)
           ──► [Step 3] Sub-answer extraction (LLM per sub-query)
           ──► [Step 4] Synthesis (LLM: sub-answers + live reactor data)
```

### Embedding engines

| Condition | Engine used |
|---|---|
| `MSR_OPENAI_API_KEY` set | `OpenAIEmbeddingEngine` – real semantic embeddings |
| No API key | `RandomProjectionEmbeddingEngine` – numpy, no external deps |

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

Both loaders write state files to `MSR_KB_DIR` (default `./kb_store`):

```
kb_store/
  archive_state.json    ← URLs of ingested msr-archive OCR files
  openalex_state.json   ← IDs of ingested OpenAlex works
```

Re-running the updater only adds truly new content.

### Periodic updates (cron example)

```cron
# Update knowledge base daily at 02:00 UTC
0 2 * * *  cd /opt/msr && python msr_kb_sources.py --update-all >> logs/kb_update.log 2>&1
```

---

## Available MCP Tools

| Tool | Description |
|---|---|
| `get_reactor_status` | Current status, power, and core temperature |
| `get_sensor_reading` | Single named sensor value |
| `get_all_sensor_readings` | All sensors at once |
| `get_sensor_history` | Historical readings (up to last 100 samples) |
| `set_control_rod_position` | Adjust control rod depth (0–100 %) |
| `get_active_alarms` | List active alarms |
| `acknowledge_alarm` | Acknowledge an alarm by ID |
| `run_thermal_simulation` | Steady-state thermal-hydraulic simulation |

---

## Architecture

```
MCP Host (Claude / agent)
        │  stdin/stdout JSON-RPC 2.0
        ▼
msr_mcp_server_main.py
        │
        ▼
msr_mcp_server.py  ── tool handlers ── simulated reactor state
                                  └── alarm checker

msr_digital_twin_with_rag.py  (inspired by open-notebook)
  ├── RandomProjectionEmbeddingEngine (numpy, no external deps)
  ├── OpenAIEmbeddingEngine (when MSR_OPENAI_API_KEY is set)
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

53 unit tests covering the MCP server (19) and enhanced RAG pipeline (34), plus 34 tests for the KB source loaders (68 + 34 = 102 total).

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

---

## Documentation

* [00_MCP_START_HERE.md](00_MCP_START_HERE.md) – five-minute quick start
* [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md) – full guide with RAG pipeline details
* [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md) – deployment guide

---

## License

MIT
