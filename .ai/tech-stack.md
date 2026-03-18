# MSR Data Layer — Technology Stack

This is the **definitive technology stack** for `msr_data_layer`.
AI agents and contributors MUST NOT introduce packages, frameworks, or
services not listed here without opening a discussion first.

---

## Language & runtime

| Component | Version / constraint |
|---|---|
| Python | 3.12 (Lambda runtime: `python3.12`) |
| Node.js | Not used |
| Other runtimes | Not used |

---

## Core Python packages (`requirements_mcp.txt`)

| Package | Purpose |
|---|---|
| `mcp` | Model Context Protocol SDK — MCP server transport |
| `numpy` | Vector arithmetic, TF-IDF, random-projection embeddings |
| `openai` | OpenAI-compatible chat + embeddings API client (GitHub Models, OpenAI) |
| `requests` | HTTP client for SCADA/historian REST API and OpenAlex |

All other standard-library modules only.

---

## Lambda-only additions (`requirements_lambda.txt`)

| Package | Purpose |
|---|---|
| `boto3` | AWS SDK — S3, CloudWatch, Lambda invocation |

---

## GPU-only additions (`requirements_gpu.txt`)

| Package | Purpose |
|---|---|
| `torch` | PyTorch — GPU tensor computation |
| `sentence-transformers` | Local dense vector embeddings |
| `transformers` | HuggingFace model loading |

Default models:
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Chat: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

---

## Infrastructure

| Component | Technology |
|---|---|
| Serverless deployment | AWS SAM (CloudFormation + Lambda + API Gateway) |
| Container image | Amazon Linux 2023 (Lambda base) / CUDA 12 (GPU variant) |
| Object storage | AWS S3 (KB store and deployment artifacts) |
| Event scheduling | AWS EventBridge (periodic KB refresh) |
| Local dev server | SAM CLI `sam local start-api` |

---

## Embedding engines (priority order)

The RAG pipeline auto-selects the best available engine at startup:

| Priority | Engine | Trigger |
|---|---|---|
| 1 | OpenAI API | `MSR_OPENAI_API_KEY` is set |
| 2 | GitHub Models API | `MSR_GITHUB_TOKEN` is set (and no OpenAI key) |
| 3 | Local GPU (sentence-transformers) | `MSR_USE_LOCAL_GPU=true` |
| 4 | Random-projection (numpy) | No keys set — zero-dependency fallback |

---

## Literature discovery sources

The KB update pipeline spans four sources (inspired by
[AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw)'s
Phase B Literature Discovery):

| Source | Loader | CLI flag | Env vars |
|---|---|---|---|
| ORNL MSR Archive | `MSRArchiveLoader` | `--update-archive` | `MSR_ARCHIVE_REPO`, `MSR_ARCHIVE_BRANCH` |
| OpenAlex API | `OpenAlexLoader` | `--update-openalex` | `MSR_OPENALEX_MAX_RESULTS`, `MSR_OPENALEX_EMAIL` |
| arXiv Atom XML | `ArXivLoader` | `--update-arxiv` | `MSR_ARXIV_MAX_RESULTS` |
| Semantic Scholar | `SemanticScholarLoader` | `--update-semanticscholar` | `MSR_S2_API_KEY`, `MSR_S2_MAX_RESULTS` |

---

## External data sources

| Source | Protocol | Authentication |
|---|---|---|
| ORNL MSR Archive | HTTPS fetch (public) | None |
| OpenAlex API | REST/HTTPS | None (public, rate-limited) |
| arXiv Atom XML API | HTTPS (Atom/XML) | None (public; ≥3s between requests per ToS) |
| Semantic Scholar Graph API | REST/HTTPS | Optional `MSR_S2_API_KEY` (higher rate limits) |
| Plant SCADA/historian | REST/HTTPS (`MSR_PLANT_DATA_URL`) | Bearer token or mTLS (operator-configured) |
| GitHub Models API | REST/HTTPS | `MSR_GITHUB_TOKEN` |
| OpenAI-compatible API | REST/HTTPS | `MSR_OPENAI_API_KEY` |

---

## Testing

| Tool | Purpose |
|---|---|
| `pytest` | Unit test runner |
| `unittest.mock` | Mocking external APIs and file I/O |

No additional testing frameworks. All external calls must be mockable without
network access.

---

## CI/CD

| Tool | Purpose |
|---|---|
| GitHub Actions | Run `make test` on every push and pull request |
| AWS SAM | Build and deploy Lambda package |

---

## What is NOT in this stack

The following are explicitly **out of scope** and must not be added:

- Django, Flask, FastAPI (the HTTP layer is SAM/Lambda)
- SQLAlchemy or any SQL ORM (KB is file-based JSON)
- Redis, Kafka, or any message broker
- TensorFlow (use PyTorch/sentence-transformers only)
- Simulation codes (RELAP5, MCNP, OpenMC, RESTE-3D) — these live in separate repos
- Any nuclear data library (ENDF, JEFF) — physics is handled by simulation layer
