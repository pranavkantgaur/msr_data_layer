# msr_data_layer — AI Agent Entry Point

> **You are an expert molten-salt-reactor (MSR) data engineer.**
> This repository is the **data layer** for an MSR knowledge and operations
> platform. It is *not* a simulation or control system. Use this file to
> orient yourself before writing any code, then consult `.ai/architecture.md`
> and `.ai/requirements.md` for full context.

---

## What this repo does

`msr_data_layer` is a **read-and-ingest data service** for Molten Salt Reactor
systems (specifically the TMSR-LF1 / MSRE class). It exposes plant data,
historical ORNL documents, and operational logs through a
[Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io) server
so that LLM agents and human operators can query and ingest data without
direct database access.

It works alongside (but does not replace) a simulation/digital-twin tool and
the [MSR Physical AI Layer](https://github.com/pranavkantgaur/msr_physical_ai_layer)
robotic fleet.

---

## Build & run commands (one-liners)

```bash
# Install core dependencies
pip install -r requirements_mcp.txt

# Run all unit tests
make test

# Run the MCP server (stdio transport)
python msr_mcp_server_main.py

# Start local HTTP API (port 3000, requires SAM CLI)
make local-api

# Run demo client
python msr_digital_twin_client.py

# Build Lambda deployment package
make build

# Deploy to AWS (interactive first run)
make deploy-guided
```

No hardcoded secrets are permitted. Use environment variables (see below).

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `MSR_GITHUB_TOKEN` | Recommended | GitHub Models API for embeddings + chat (free with Copilot Pro) |
| `MSR_OPENAI_API_KEY` | Optional | OpenAI-compatible API key (overrides GITHUB_TOKEN) |
| `MSR_OPENAI_BASE_URL` | Optional | OpenAI-compatible base URL (default: `https://api.openai.com/v1`) |
| `MSR_OPENAI_MODEL` | Optional | Chat model (default: `gpt-4o-mini`) |
| `MSR_EMBED_MODEL` | Optional | Embedding model (default: `text-embedding-3-small`) |
| `MSR_PLANT_DATA_URL` | Optional | External SCADA/historian REST API URL; stub used when unset |
| `MSR_KB_DIR` | Optional | Persistent KB directory (default: `./kb_store`) |
| `MSR_DOCS_DIR` | Optional | Reference docs directory (default: `./docs`) |
| `MSR_USE_LOCAL_GPU` | Optional | `true` to use local GPU sentence-transformers |

Copy `samconfig.toml.example` → `samconfig.toml` (excluded from git) and
`DEPLOYMENT_CREDENTIALS.md` for deployment guidance — never commit real secrets.

---

## Repository layout

```
msr_mcp_server.py          MCP server — data tools + ingest_plant_data
msr_mcp_server_main.py     Entry point (stdio transport)
msr_digital_twin_with_rag.py  Multi-step RAG pipeline (vector + TF-IDF)
msr_kb_sources.py          KB loaders: MSR archive, OpenAlex, PlantDataLoader
lambda_function.py         AWS Lambda handler (HTTP + EventBridge + /data/ingest)
msr_digital_twin_client.py Python MCP client
template.yaml              AWS SAM template
Dockerfile.gpu             GPU container image
Makefile                   All build/test/deploy targets
requirements_mcp.txt       Core Python deps
requirements_lambda.txt    Lambda deps (adds boto3)
requirements_gpu.txt       GPU deps (torch, sentence-transformers)
use_cases/                 Worked examples per scientific paper + physical AI tasks
  physical_ai/             Foundation-model training use cases (12 robotic areas)
.ai/                       Agent grounding documents
  requirements.md          Goals, constraints, safety notes
  architecture.md          System overview + data flow
  tech-stack.md            Definitive technology stack
test_*.py                  Unit tests (pytest)
```

---

## Coding conventions

* **Language:** Python 3.12 only.
* **Style:** PEP 8, type hints on all public functions, Google-style docstrings.
* **Dependencies:** Only add packages already in `requirements_*.txt`. Never
  add new packages without updating the relevant requirements file.
* **No global state** — all configuration via environment variables or
  constructor parameters.
* **Tests:** All new code must have unit tests in the corresponding
  `test_*.py` file. Run `make test` before committing.
* **No hardcoded secrets, URLs, account IDs, or AWS resource names.**
* **Stubs first:** Every external I/O path must have a working stub so
  tests and local dev work with zero external services.

---

## Testing strategy

```bash
make test                          # run all 4 test files
python -m pytest test_msr_rag.py -v       # RAG pipeline
python -m pytest test_msr_mcp_server.py -v  # MCP server tools
python -m pytest test_msr_kb_sources.py -v  # KB loaders + PlantDataLoader
python -m pytest test_lambda_function.py -v # Lambda handler
```

Target: ≥70% coverage on core modules. CI runs `make test` on every push.

---

## What agents MUST NOT do

* Do **not** add control/actuation tools (e.g. `set_control_rod_position`,
  `acknowledge_alarm`, `run_thermal_simulation`). This is a DATA LAYER only.
* Do **not** commit `samconfig.toml`, `kb_store/`, `docs/`, `.env`, or any
  file containing credentials.
* Do **not** pin dependencies to exact versions unless there is a known
  incompatibility — prefer `>=` ranges.
* Do **not** modify `template.yaml` Lambda resources without updating
  `test_lambda_function.py`.

---

## Related repositories

| Repo | Role |
|---|---|
| [`msr_physical_ai_layer`](https://github.com/pranavkantgaur/msr_physical_ai_layer) | Robot fleet MCP server (12 operational areas) |
| [`msr-gstack`](https://github.com/pranavkantgaur/msr-gstack) | Multi-agent orchestration layer |

The physical AI training use cases in `use_cases/physical_ai/` document how
this data layer feeds foundation-model training for each of the 12 robotic
areas.
