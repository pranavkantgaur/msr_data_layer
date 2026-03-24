# scripts/

Utility scripts for the `msr_data_layer` repository.

---

## generate_architecture_diagrams.py

Dynamically regenerates every file in `docs/architecture/` by traversing the
live source code and calling the GitHub Models (Copilot Pro) or OpenAI-compatible
chat API.

The script reads the relevant source files for each architecture document,
includes the existing document as a reference template, and asks the LLM to
update all Mermaid diagram blocks so they accurately reflect the current
implementation.  When the source code has not changed, the LLM returns the
document unchanged (temperature=0 for maximum determinism).  When code evolves,
rerunning the script keeps every diagram in sync.

### Quick start

```bash
# Install core dependencies (if not already installed)
pip install -r requirements_mcp.txt

# Set your GitHub Copilot token
export GITHUB_TOKEN=<your-github-pat-with-copilot-pro>

# Regenerate all 7 architecture documents
python scripts/generate_architecture_diagrams.py

# Preview one document without writing to disk
python scripts/generate_architecture_diagrams.py --file 03_mcp_server.md --dry-run

# Regenerate a single file
python scripts/generate_architecture_diagrams.py --file 01_rag_pipeline.md
```

### Options

| Flag | Description |
|---|---|
| `--dry-run` | Print generated content to stdout; do not write files. |
| `--file FILENAME` | Regenerate only the specified diagram (e.g. `03_mcp_server.md`). |
| `--model MODEL` | Override the LLM model (default: `gpt-4o`). |
| `--repo-root PATH` | Override the repository root (default: parent of `scripts/`). |

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GITHUB_TOKEN` | Recommended | GitHub PAT with Copilot Pro / GitHub Models subscription. |
| `MSR_GITHUB_TOKEN` | Alternative | Same as `GITHUB_TOKEN` (explicit MSR-prefixed variant). |
| `MSR_OPENAI_API_KEY` | Optional | OpenAI-compatible key (takes precedence over GitHub token). |
| `MSR_OPENAI_BASE_URL` | Optional | Base URL override (auto-set when using GitHub token). |
| `MSR_OPENAI_MODEL` | Optional | Model override (same effect as `--model`). |

### How it works

1. **Source traversal** — for each architecture file the script collects the
   relevant Python, YAML, and Markdown source files.
2. **LLM call** — the source files + existing diagram are sent to the chat API
   with a system prompt instructing the model to preserve all prose and update
   only the Mermaid blocks that no longer match the code.
3. **Write-back** — the model response is written directly to
   `docs/architecture/<filename>`.

Diagrams are written in [Mermaid](https://mermaid.js.org/) and render
natively on GitHub.

### Source-file ↔ diagram mapping

| Architecture document | Source files analysed |
|---|---|
| `00_end_to_end.md` | all core modules + `template.yaml` + `Dockerfile.gpu` |
| `01_rag_pipeline.md` | `msr_digital_twin_with_rag.py` |
| `02_kb_sources.md` | `msr_kb_sources.py` |
| `03_mcp_server.md` | `msr_mcp_server.py`, `msr_mcp_server_main.py`, `server.py` |
| `04_lambda_deployment.md` | `lambda_function.py`, `template.yaml` |
| `05_embedding_engines.md` | `msr_digital_twin_with_rag.py` |
| `06_physical_ai_integration.md` | `msr_kb_sources.py`, `use_cases/physical_ai/*.md` |

### No new dependencies

The script uses only Python standard-library modules (`urllib`, `json`,
`argparse`, `pathlib`, `logging`) — no additional packages are required beyond
what is already in `requirements_mcp.txt`.
