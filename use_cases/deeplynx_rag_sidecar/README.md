# DeepLynx RAG Sidecar

A Python MCP sidecar that adds multi-step RAG, NL→SQL timeseries queries, and
academic-literature ingestion to a running [DeepLynx Nexus](https://github.com/idaholab/DeepLynx)
instance — without modifying DeepLynx's C# core.

## Motivation

See [`ANALYSIS.md`](ANALYSIS.md) for the full comparative analysis of DeepLynx vs.
MSR Data Layer and the rationale for this contribution.

In short: DeepLynx's DL-1045 MCP prototype annotates existing REST services as
tools. This sidecar extends that capability with:

1. **Multi-step RAG** — query decomposition → hybrid dense+sparse retrieval →
   per-result extraction → LLM synthesis.
2. **NL→SQL timeseries** — "What was the average core temp last Tuesday?"
   generates a safe `SELECT`, executes it against DeepLynx timeseries, and
   returns structured results.
3. **Academic literature ingestion** — fetches OpenAlex/arXiv papers for a
   topic and creates typed `Publication` nodes in DeepLynx.
4. **Air-gapped / zero-credential operation** — four-tier embedding fallback
   (OpenAI → GitHub Models → local GPU → random-projection) ensures the sidecar
   works in restricted environments.

---

## Quick start

```bash
# 1. Clone DeepLynx and start it (or point at an existing instance)
#    docker compose up  (from DeepLynx root)

# 2. Configure the sidecar
export DEEPLYNX_URL=http://localhost:5095
export DEEPLYNX_API_KEY=<your-api-key>
export DEEPLYNX_API_SECRET=<your-api-secret>
export GITHUB_TOKEN=<your-github-token>   # free via GitHub Copilot Pro

# 3. Run the sidecar (stdio MCP — for AI agents)
cd tools/deeplynx-rag-sidecar
python mcp_server.py

# 4. Or run as HTTP (for integration tests / humans)
python mcp_server.py --http 8001

# 5. Run tests (zero network / zero credentials required)
python -m pytest test_sidecar.py -v
```

---

## MCP tools exposed

| Tool | Description |
|------|-------------|
| `ingest_project_records` | Index a DeepLynx project's records into the local KB |
| `query_catalog` | Multi-step RAG answer over indexed records |
| `query_timeseries_nl` | Natural-language → SQL query over a timeseries data source |
| `get_record_context` | Return KB context for a specific record ID |
| `search_and_ingest_literature` | Fetch OpenAlex papers and create Publication nodes in DeepLynx |
| `get_sidecar_status` | Health / status including KB size and embedding mode |

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPLYNX_URL` | Recommended | Base URL of the DeepLynx Nexus API |
| `DEEPLYNX_API_KEY` | Recommended | DeepLynx API key |
| `DEEPLYNX_API_SECRET` | Recommended | DeepLynx API secret |
| `DEEPLYNX_OPENAI_API_KEY` | Optional | OpenAI-compatible key for LLM + embeddings |
| `DEEPLYNX_OPENAI_BASE_URL` | Optional | OpenAI-compatible base URL |
| `DEEPLYNX_OPENAI_MODEL` | Optional | Chat model (default: `gpt-4o-mini`) |
| `DEEPLYNX_EMBED_MODEL` | Optional | Embedding model (default: `text-embedding-3-small`) |
| `DEEPLYNX_GITHUB_TOKEN` | Optional | GitHub Models API token (fallback for LLM/embedding) |
| `DEEPLYNX_USE_LOCAL_GPU` | Optional | `true` to use `sentence-transformers` locally |

When none of the above are set, the sidecar operates fully offline using
random-projection embeddings and no LLM synthesis (returns raw context chunks).

---

## Architecture

```
DeepLynx Nexus (C# / .NET 10)
   REST API (:5095)
        │
        ▼
deeplynx_client.py          ← thin REST wrapper + stubs
        │
        ▼
rag_engine.py               ← chunking, embedding, hybrid retrieval,
        │                      multi-step RAG pipeline
        ▼
mcp_server.py               ← 6 MCP tools, stdio + HTTP transport
        │
        ▼
AI agents / LLM operators
(GitHub Copilot Chat, Claude Desktop, custom MCP clients)
```

---

## File structure

```
deeplynx_client.py   DeepLynx REST API wrapper
rag_engine.py        RAG engine (adapted from MSR Data Layer)
mcp_server.py        MCP server — 6 tools, stdio + HTTP
test_sidecar.py      Unit tests (zero network)
requirements.txt     Dependencies (stdlib only; pytest for tests)
ANALYSIS.md          Comparative analysis of DeepLynx vs. MSR Data Layer
README.md            This file
```

---

## Provenance

Patterns adapted from
[`pranavkantgaur/msr_data_layer`](https://github.com/pranavkantgaur/msr_data_layer):
- Sentence-aware chunking with overlap (`msr_digital_twin_with_rag.py`)
- Four-tier embedding fallback (`msr_digital_twin_with_rag.py`)
- Hybrid dense + TF-IDF retrieval (`msr_digital_twin_with_rag.py`)
- NL→SQL timeseries with safe-SELECT validation (`msr_kb_sources.py:TimeseriesStore`)
- Zero-dependency stub pattern (`msr_mcp_server.py`)

---

## Licence

Apache 2.0 — same as DeepLynx.
