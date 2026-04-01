# PR Draft — to be submitted to https://github.com/idaholab/DeepLynx

---

## Pull Request Title

**feat: add Python RAG sidecar — multi-step catalog retrieval, NL→SQL timeseries, and literature ingestion via MCP**

---

## Description

### Problem

DeepLynx Nexus's MCP prototype (DL-1045) annotates existing REST services as
tools. This is a solid foundation, but it leaves a gap: asking an AI agent
*"Which welds in Building-4 used a non-conforming filler material?"* returns a
list of record IDs — it cannot synthesise an answer across heterogeneous record
types, relate them to each other, or extract a concise finding.

A second gap: DeepLynx v0.4.0 added timeseries ingest and a UI viewer, but
there is no natural-language query interface. Operators must know sensor names
and write REST queries manually.

A third gap: project teams working in nuclear, advanced manufacturing, or
research-intensive domains want to connect peer-reviewed literature to their
project artefacts. Today there is no data source adapter for scholarly APIs.

### Solution

This PR adds a **Python MCP sidecar** (`tools/deeplynx-rag-sidecar/`) that
runs alongside a DeepLynx Nexus instance and closes all three gaps. It does
**not** modify DeepLynx's C# core, database schema, or Entity Framework models.

The patterns are adapted from
[`pranavkantgaur/msr_data_layer`](https://github.com/pranavkantgaur/msr_data_layer),
an Apache-2.0 open-source data layer for Molten Salt Reactor systems that has
been using these patterns in a production-like setting for nuclear operations.
The comparative analysis of both systems is in
[`tools/deeplynx-rag-sidecar/ANALYSIS.md`](tools/deeplynx-rag-sidecar/ANALYSIS.md).

### What this PR adds

| Component | File | Description |
|-----------|------|-------------|
| DeepLynx client wrapper | `deeplynx_client.py` | Thin REST wrapper (OAuth token exchange, record fetch, timeseries fetch, record create). Works offline via stubs when `DEEPLYNX_URL` is unset. |
| RAG engine | `rag_engine.py` | Multi-step pipeline: sentence-aware chunking → 4-tier embedding fallback → hybrid dense+TF-IDF retrieval → LLM query decomposition → sub-answer extraction → synthesis. |
| MCP server | `mcp_server.py` | 6 MCP tools via stdio transport (for AI agents) or HTTP (for integration). |
| Unit tests | `test_sidecar.py` | 30 tests, zero network calls, zero credentials required. |
| Dependencies | `requirements.txt` | stdlib only; `pytest` for tests. |
| Analysis | `ANALYSIS.md` | Full comparative analysis of DeepLynx vs. MSR Data Layer. |
| README | `README.md` | Setup guide, environment variables, architecture diagram. |

### MCP tools

| Tool | What it does |
|------|--------------|
| `ingest_project_records` | Fetch records from a DeepLynx container and index them into a local hybrid KB |
| `query_catalog` | Multi-step RAG over indexed records — returns a synthesised answer with source citations |
| `query_timeseries_nl` | Natural-language → LLM-generated SQL → DeepLynx timeseries query |
| `get_record_context` | Return KB context chunks for a specific record ID |
| `search_and_ingest_literature` | Fetch OpenAlex papers on a topic; create `Publication` nodes in DeepLynx |
| `get_sidecar_status` | Health check — KB size, embedding mode, DeepLynx connectivity |

### Air-gapped / restricted-network operation

Four-tier embedding fallback ensures operation in INL's restricted environments:

```
OpenAI API
    ↓ (if unavailable)
GitHub Models API (free with Copilot Pro)
    ↓ (if unavailable)
Local GPU (sentence-transformers/all-MiniLM-L6-v2)
    ↓ (if unavailable)
Random-projection (pure NumPy, zero external deps)
```

### Example interaction

```
User (via Copilot Chat): "Which inspection reports mention Hastelloy-N?"

→ ingest_project_records(container_id="MSRE-Phase2")
  Indexed 347 chunks from 48 records.

→ query_catalog(question="Which inspection reports mention Hastelloy-N?",
                container_id="MSRE-Phase2")
  Answer: "Three inspection reports reference Hastelloy-N alloy:
           [R-0112] Primary loop piping inspection (2024-03) — found minor
           oxidation on weld seams W-7 and W-12.
           [R-0156] Heat exchanger inspection (2024-07) — tube bundle nominal.
           [R-0201] Salt pump housing (2024-11) — no defects detected.
           Sources: deeplynx:MSRE-Phase2:R-0112, ...:R-0156, ...:R-0201."
```

```
User: "What was the average core temperature on 2024-01-15?"

→ query_timeseries_nl(
    question="average core temperature on 2024-01-15",
    container_id="MSRE-Phase2",
    datasource_id="ts-core-temps"
  )
  Generated SQL: SELECT AVG(value) FROM timeseries_data
                 WHERE sensor='core_temp_c'
                 AND timestamp BETWEEN '2024-01-15T00:00:00Z'
                 AND '2024-01-15T23:59:59Z'
  Result: 627.3 °C (24 readings)
```

---

### How to test

```bash
cd tools/deeplynx-rag-sidecar

# All tests pass with zero credentials / zero network
python -m pytest test_sidecar.py -v

# Run against a live DeepLynx instance
export DEEPLYNX_URL=http://localhost:5095
export DEEPLYNX_API_KEY=...
export DEEPLYNX_API_SECRET=...
export GITHUB_TOKEN=...   # for LLM + embeddings

python mcp_server.py --http 8001
# POST http://localhost:8001  {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_sidecar_status","arguments":{}}}
```

---

### Checklist

- [x] No changes to C# core, EF migrations, or PostgreSQL schema
- [x] All external I/O paths have working stubs
- [x] Unit tests pass with zero network / zero credentials
- [x] No hardcoded secrets, URLs, or credentials
- [x] Apache 2.0 licence preserved
- [x] `README.md` documents all environment variables
- [x] Comparative analysis in `ANALYSIS.md` explains scope and motivation
- [ ] Integration test against a running DeepLynx instance (pending access)

---

### Related issues / tickets

- DL-1045 (MCP Prototype) — this sidecar is complementary, not a replacement
- DL-330 (Event System) — events could be ingested by `ingest_project_records`
- DL-842 (App-to-App Integrations) — the sidecar uses the API key/secret pair

---

### Provenance and licence

All code in this PR is original work derived from patterns in
[`pranavkantgaur/msr_data_layer`](https://github.com/pranavkantgaur/msr_data_layer)
(Apache 2.0). No code was copied verbatim; the architecture and algorithms were
re-implemented to fit DeepLynx's data model and deployment conventions.

The OpenAlex API is used under its
[CC0 data licence](https://docs.openalex.org/additional-help/faq#how-is-openalex-licensed).
arXiv abstracts are used under the
[arXiv non-exclusive licence](https://arxiv.org/help/license).
