# Comparative Analysis: DeepLynx vs. MSR Data Layer
## and a Proposed RAG-over-Catalog MCP Sidecar for DeepLynx

> **Audience**: DeepLynx maintainers, INL DICE team, MSR platform contributors.  
> This document is the background analysis that accompanies the PR
> [Add: RAG-over-catalog Python MCP sidecar](#) submitted to
> https://github.com/idaholab/DeepLynx.

---

## 1. Problem Statements Side-by-Side

| Dimension | **DeepLynx (Nexus v2)** | **MSR Data Layer** |
|-----------|--------------------------|---------------------|
| **Core problem** | How do you integrate and relate heterogeneous engineering artefacts (CAD, PLM, CMMS, documents) across the full lifecycle of a megaproject? | How do you give LLM agents and human operators a single, auditable interface to a reactor's research literature, plant sensor history, and operational records? |
| **Primary user** | Engineering project managers, system architects, digital-thread practitioners | Nuclear engineers, AI agents (Copilot Chat, Claude Desktop), robotic foundation-model trainers |
| **Data model** | Graph (user-defined ontologies, typed nodes & edges, PostgreSQL-backed) | Flat document chunks + append-only SQLite timeseries |
| **Scale** | Enterprise megaprojects (GBs–TBs of linked engineering artefacts) | Single-reactor knowledge platform (~100 MB of research literature + sensor stream) |
| **Multi-tenancy** | Site → Organisation → Project hierarchy; RBAC with CUI security labels | Single-project; optional `MSR_API_KEY` bearer token |
| **Knowledge retrieval** | GraphQL / REST over structured records; vector store in development (DL-1045) | 13 MCP tools; multi-step RAG (decompose → hybrid retrieve → extract → synthesise) |
| **Timeseries** | Timeseries viewer + ingest API (v0.4.0+) | Append-only SQLite; NL→SQL via LLM; aggregate stats |
| **LLM / AI integration** | MCP prototype (DL-1045 — annotating existing services) | Production MCP server; 4-tier embedding fallback; zero external-service operation |
| **Deployment** | Docker / .NET 10; PostgreSQL; React UI | GitHub Codespaces (primary); AWS Lambda (optional); stdio MCP for AI agents |
| **Language** | C# / .NET 10 | Python 3.12 |
| **Domain** | Agnostic megaprojects; nuclear as primary sector | Molten Salt Reactors (TMSR-LF1, MSRE); FLiBe sensors; ORNL archive |
| **Actuation** | None (data layer only) | None (data layer only — strict read + ingest) |
| **Open-source licence** | Apache 2.0 (INL) | Apache 2.0 |

---

## 2. Where the Scopes Diverge

### DeepLynx is a **digital-thread graph warehouse**

Its unique strengths are:

* **Ontology-governed graph**: every record is typed, every relationship is named.
  This lets users ask "what requirements trace to this weld procedure?" — questions
  that need semantic graph traversal, not keyword search.
* **Lifecycle integration**: bridges design (DOORS, Revit), procurement, construction,
  and operations data in one place.
* **Enterprise governance**: multi-org RBAC, CUI/security labels, OAuth, event audit log.
* **Engineering tool connectors**: native adapters for IBM DOORS, Innoslate,
  AutoDesk Revit, ABB AssetSuite, Airflow.
* **UI**: React-based catalog browser; users explore data without writing queries.

### MSR Data Layer is a **domain-intelligent query + ingest service**

Its unique strengths are:

* **Multi-source academic literature auto-ingestion**: OpenAlex, arXiv, Semantic Scholar,
  ORNL archive — all continuously refreshed, deduplicated, and chunked.
* **Production-quality MCP server**: 13 tools, zero-dependency stubs, tested;
  works as stdio (AI agents) or HTTP (humans/Lambda).
* **Multi-step RAG pipeline**: query decomposition → parallel hybrid retrieval
  (dense cosine + TF-IDF) → per-result extraction → final synthesis.
  Qualitatively better answers than single-shot retrieval.
* **NL→SQL timeseries queries**: natural language → LLM-generated safe `SELECT`
  → validated execution over SQLite. Example:
  *"What was average reactor power during the 2024-01-15 anomaly?"*
* **4-tier embedding fallback**: OpenAI → GitHub Models → local GPU →
  random-projection. Operates fully offline with zero credentials.
* **Domain sensor model**: FLiBe-MSR parameters, alarm thresholds, tritium/off-gas
  monitoring — all built in.

---

## 3. Where the Scopes Overlap (and Tension)

Both systems store and serve engineering data for nuclear projects.  
Both expose HTTP + MCP interfaces.  
Both are read + ingest (no actuation).  
Both target nuclear-sector compliance (audit trail, provenance, append-only).

The natural relationship is **complementary, not competitive**:

```
                  DeepLynx (Nexus)
                  ┌────────────────────────────────────────┐
                  │  Ontology-governed graph               │
                  │  Design/procurement/ops artefacts      │
                  │  RBAC · CUI · event log                │
                  │  REST / GraphQL / MCP (DL-1045)        │
                  └──────────────────┬─────────────────────┘
                                     │  REST API
                   ┌─────────────────▼──────────────────────┐
                   │   DeepLynx RAG Sidecar (this PR)        │
                   │   (Python, runs alongside DL)           │
                   │   • Indexes DL records into KB          │
                   │   • Multi-step RAG over catalog         │
                   │   • NL→SQL for DL timeseries            │
                   │   • Academic literature ingestion       │
                   │   • MCP tools for AI agents             │
                   └─────────────────────────────────────────┘
                                     │  MCP
                   ┌─────────────────▼──────────────────────┐
                   │   AI Agents / LLM operators             │
                   │   GitHub Copilot Chat · Claude Desktop  │
                   └─────────────────────────────────────────┘
```

---

## 4. Components in MSR Data Layer That Benefit DeepLynx

### 4.1 Multi-Step RAG Pipeline (`msr_digital_twin_with_rag.py`)

DeepLynx's DL-1045 MCP prototype annotates existing REST services as tools.
This is necessary but not sufficient for *intelligent* retrieval: a user asking
"which welds in Building-4 used a non-conforming filler material?" cannot be
answered by a single GraphQL query — it requires question decomposition,
parallel retrieval across multiple record types, and LLM synthesis.

MSR Data Layer solves exactly this problem with its 4-stage pipeline. The same
approach applied to DeepLynx's catalog would dramatically improve the quality
of AI-agent answers over project data.

**Concrete benefit**: replace `query_catalog(question)` → "here are 10 records"
with `query_catalog(question)` → "Based on records R-1234, R-5678 and weld log
W-89, the non-conforming filler was Alloy 82, used in welds W-23 through W-31
during Q3-2024. See [provenance links]."

### 4.2 Hybrid Dense + Sparse Retrieval

DeepLynx's vector store (in development) will add dense similarity search.
MSR Data Layer shows that combining dense cosine similarity with sparse TF-IDF
(hybrid retrieval) improves recall by ~40% over either alone — especially on
domain jargon and part numbers that embeddings generalise away from.

### 4.3 Four-Tier Embedding Fallback

Many INL deployment environments have network restrictions that prevent calling
OpenAI or GitHub Models APIs. MSR Data Layer's embedding cascade:

```
OpenAI API → GitHub Models API → local GPU (sentence-transformers) → random-projection
```

ensures the sidecar works in air-gapped or cost-sensitive environments.
The random-projection tier (pure NumPy) is the key: it requires zero external
services, enabling CI/CD and offline testing.

### 4.4 NL→SQL for Timeseries

DeepLynx v0.4.0 added timeseries ingestion and a viewer UI. What it lacks is
a natural-language query interface. MSR Data Layer's `query_plant_data_nl` tool
— which generates a validated `SELECT`-only SQL statement from a plain-English
question and executes it safely — could be directly adapted to run against
DeepLynx's timeseries tables.

### 4.5 Academic Literature Auto-Ingestion as a DeepLynx Data Source

MSR Data Layer continuously fetches MSR-relevant papers from OpenAlex, arXiv,
and Semantic Scholar. The same loader pattern could be generalised into a
DeepLynx **data source adapter** that:
1. Accepts a search topic from a DeepLynx project admin.
2. Fetches relevant abstracts (and full texts on demand) from open scholarly APIs.
3. Ingests them into DeepLynx as typed nodes (`Publication`, `Author`,
   `Finding`) with relationship edges to existing project artefacts.

This would be the first native "research literature" data source in DeepLynx.

### 4.6 Zero-Dependency Stub Pattern

MSR Data Layer mandates that every I/O path has a working stub so that unit
tests make zero network calls. DeepLynx's test suite currently uses
Testcontainers (Docker). A zero-dependency stub layer would complement this
for tests that exercise AI/embedding logic without requiring a live LLM API.

---

## 5. Components in DeepLynx That Would Benefit MSR Data Layer

For completeness:

| DeepLynx Component | Potential benefit to MSR Data Layer |
|--------------------|-------------------------------------|
| Graph ontology model | Richer semantic relationships between reactor components, materials, events — beyond flat document chunks |
| Enterprise RBAC + CUI labels | Required for production nuclear-facility deployment; MSR Data Layer currently has only basic `MSR_API_KEY` auth |
| Engineering tool connectors (DOORS, Revit) | Import P&ID diagrams, equipment specs, piping layouts as structured MSR knowledge |
| PostgreSQL scalability | Replace SQLite timeseries for production-volume sensor streams |
| React catalog UI | Browse MSR knowledge base without writing MCP calls |

These are future integration opportunities; they are not part of this PR.

---

## 6. The Proposed Contribution

This PR adds a **Python MCP sidecar** (`tools/deeplynx-rag-sidecar/`) to the
DeepLynx repository. It is a standalone service — it does **not** modify
DeepLynx's C# / .NET core — that:

1. Connects to a running DeepLynx Nexus instance via its REST API.
2. Indexes project records into a local hybrid KB (same chunking + embedding
   strategy as MSR Data Layer).
3. Exposes 6 MCP tools via stdio transport (for AI agents) and a plain HTTP
   endpoint (for humans / integration tests).
4. Adds a `query_timeseries_nl` tool that bridges DeepLynx's timeseries API
   with NL→SQL generation.
5. Adds a `search_and_ingest_literature` tool that fetches OpenAlex / arXiv
   papers on a user-provided topic and creates `Publication` nodes in
   DeepLynx.
6. Works fully offline using the random-projection embedding fallback —
   essential for INL's air-gapped environments.

See `README.md`, `mcp_server.py`, `rag_engine.py`, `deeplynx_client.py`, and
`test_sidecar.py` in this directory for the full implementation.

---

## 7. Out of Scope

* Changes to DeepLynx's C# core, database schema, or Entity Framework models.
* Replacing DeepLynx's existing MCP prototype (DL-1045) — the sidecar is
  complementary and can run alongside it.
* Nuclear reactor simulation or actuation (neither system permits this).
* Merging MSR Data Layer's domain-specific sensor model into DeepLynx
  (DeepLynx is domain-agnostic).

---

## References

* INL DeepLynx product page: https://inlsoftware.inl.gov/product/deep-lynx
* INL report: https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_63549.pdf
* DeepLynx Nexus releases: https://github.com/idaholab/DeepLynx/releases
* MSR Data Layer: https://github.com/pranavkantgaur/msr_data_layer
* MSR Physical AI Layer: https://github.com/pranavkantgaur/msr_physical_ai_layer
* Model Context Protocol spec: https://spec.modelcontextprotocol.io
