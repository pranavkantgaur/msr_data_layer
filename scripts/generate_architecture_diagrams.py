#!/usr/bin/env python3
"""
generate_architecture_diagrams.py

Dynamically regenerate docs/architecture/*.md by traversing the live
repository source code and calling the GitHub Models (Copilot Pro) /
OpenAI-compatible chat-completions API.

For each architecture document the script:

  1. Reads the relevant source files for that diagram from the repo.
  2. Reads the existing architecture document as a template/reference.
  3. Sends both to the LLM with instructions to update the Mermaid diagram
     blocks to reflect the current source code, keeping all prose unchanged.
  4. Writes the LLM response back to docs/architecture/<filename>.

When the source code has not changed the LLM returns the document unchanged
(temperature=0 maximises determinism).  When code evolves, rerunning the
script keeps the diagrams in sync.

Usage
-----
    python scripts/generate_architecture_diagrams.py [OPTIONS]

Options
-------
    --dry-run              Print output to stdout; do not write files.
    --file FILENAME        Regenerate only one diagram (e.g. 03_mcp_server.md).
    --model MODEL          LLM model name (default: gpt-4o).
    --repo-root PATH       Repository root (default: parent of scripts/).

Environment variables
---------------------
    GITHUB_TOKEN / MSR_GITHUB_TOKEN
        GitHub personal access token with a Copilot Pro / GitHub Models
        subscription.  Used automatically when MSR_OPENAI_API_KEY is not set.
    MSR_OPENAI_API_KEY
        OpenAI-compatible API key (takes precedence over GitHub token).
    MSR_OPENAI_BASE_URL
        API base URL.  Defaults to models.inference.ai.azure.com when using a
        GitHub token, or api.openai.com/v1 when using an OpenAI key.
    MSR_OPENAI_MODEL
        Override the default model name (same effect as --model).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_GITHUB_MODELS_BASE_URL: str = "https://models.inference.ai.azure.com"
_DEFAULT_MODEL: str = "gpt-4o"
_MAX_FILE_CHARS: int = 40_000   # Per-file truncation limit (chars ≈ ~10 K tokens)
_MAX_OUTPUT_TOKENS: int = 8192  # Max tokens the LLM may emit per architecture file

# ── Diagram configurations ─────────────────────────────────────────────────────
# Each entry maps one docs/architecture/*.md file to the source files whose
# current implementation it should document.

_DIAGRAM_CONFIGS: list[dict] = [
    {
        "filename": "00_end_to_end.md",
        "description": (
            "Full end-to-end architecture showing every component and data flow: "
            "external data sources (ORNL archive, OpenAlex, arXiv, Semantic Scholar, "
            "SCADA/historian, sensor push) → KB source loaders → RAG pipeline → "
            "TimeseriesStore (SQLite plant_timeseries.db) → MCP server → "
            "transport variants (stdio and Lambda/HTTP) → AWS infrastructure → "
            "GPU container variant → consuming agents and operators. "
            "New timeseries layer: TimeseriesStore insert/query/NL→SQL, "
            "new MCP tools (query_sensor_timeseries, get_sensor_stats, query_plant_data_nl), "
            "new Lambda routes POST /timeseries/ingest and POST /timeseries/query."
        ),
        "source_files": [
            "msr_kb_sources.py",
            "msr_digital_twin_with_rag.py",
            "msr_mcp_server.py",
            "msr_mcp_server_main.py",
            "lambda_function.py",
            "template.yaml",
            "Dockerfile.gpu",
        ],
    },
    {
        "filename": "01_rag_pipeline.md",
        "description": (
            "RAG pipeline: ingestion pipeline (sentence-aware text chunking, dense "
            "vector embedding, source-insight extraction, KB persistence) and retrieval "
            "pipeline (LLM query decomposition into ≤5 sub-queries, hybrid dense+sparse "
            "search, multi-step synthesis).  Also shows embedding-engine priority."
        ),
        "source_files": [
            "msr_digital_twin_with_rag.py",
        ],
    },
    {
        "filename": "02_kb_sources.md",
        "description": (
            "Knowledge-base source loaders: MSRArchiveLoader, OpenAlexLoader, "
            "ArXivLoader, SemanticScholarLoader, PlantDataLoader, KBSourceManager. "
            "New: TimeseriesStore (SQLite sqlite3 stdlib) — plant sensor timeseries "
            "with time-range queries, aggregate statistics, and NL→SQL via "
            "execute_safe_select() + get_schema_description(); "
            "KBSourceManager.ingest_timeseries(), query_timeseries(), query_timeseries_nl(). "
            "State-file deduplication sequence.  CLI reference and environment variables."
        ),
        "source_files": [
            "msr_kb_sources.py",
        ],
    },
    {
        "filename": "03_mcp_server.md",
        "description": (
            "MCP server tool surface (class diagram showing MSRMCPServer, "
            "PlantDataLayer, RAGPipeline, PlantDataLoader, TimeseriesStore, "
            "KBSourceManager); read-tool data flow; "
            "write-tool (ingest_plant_data) data flow; "
            "new timeseries tools: query_sensor_timeseries, get_sensor_stats, "
            "query_plant_data_nl (NL→SQL via LLM + TimeseriesStore.execute_safe_select); "
            "stdio transport for local agents / Claude Desktop / Copilot; "
            "HTTP transport for Lambda / local-api with new routes "
            "POST /timeseries/ingest and POST /timeseries/query; "
            "sensor stub values."
        ),
        "source_files": [
            "msr_mcp_server.py",
            "msr_mcp_server_main.py",
            "lambda_function.py",
            "server.py",
        ],
    },
    {
        "filename": "04_lambda_deployment.md",
        "description": (
            "AWS Lambda deployment: CloudFormation resource topology (Lambda function, "
            "API Gateway HTTP API v2, S3 KB-persistence bucket, EventBridge daily "
            "schedule, IAM roles, CloudWatch/X-Ray); warm-Lambda request lifecycle; "
            "KB-update lifecycle (EventBridge or manual); SAM parameters; "
            "build and deploy commands."
        ),
        "source_files": [
            "lambda_function.py",
            "template.yaml",
        ],
    },
    {
        "filename": "05_embedding_engines.md",
        "description": (
            "Embedding engine selection flowchart (OpenAI API → GitHub Models API → "
            "local GPU sentence-transformers → random-projection numpy fallback); "
            "LLM selection (parallel to embedding selection); hybrid retrieval "
            "(dense cosine + sparse TF-IDF / BM25)."
        ),
        "source_files": [
            "msr_digital_twin_with_rag.py",
        ],
    },
    {
        "filename": "06_physical_ai_integration.md",
        "description": (
            "Three-stage training-data pipeline for the 12 robotic operational areas "
            "of msr_physical_ai_layer: Stage 1 – ORNL archive retrieval, Stage 2 – "
            "sensor-stream ingestion via PlantDataLoader, Stage 3 – labelled episode "
            "export via RAG.  Mindmap of data types ingested per robotic area.  "
            "API usage pattern from use_cases/physical_ai/."
        ),
        "source_files": [
            "msr_kb_sources.py",
            "use_cases/physical_ai/",   # directory – all .md files are collected
        ],
    },
]

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = """\
You are an expert software architect maintaining the architecture documentation \
for msr_data_layer — a read-only data service for Molten Salt Reactor (MSR) \
systems built on Python 3.12, AWS Lambda, and the Model Context Protocol (MCP).

Your task: given a set of source files from the repository and the existing \
architecture document, analyse the source code and return the updated document \
so that every Mermaid diagram block accurately reflects the current implementation.

Rules
-----
1. Return the COMPLETE updated markdown document and nothing else.
2. Do NOT add any prefix, suffix, explanation, or wrapper code-fences around \
   the output — return raw markdown that can be written directly to disk.
3. Keep all prose, headings, bullet lists, tables, and non-diagram sections \
   byte-for-byte identical unless a concrete code change requires updating them.
4. Update ```mermaid … ``` blocks so they accurately reflect the current code.
5. If the existing diagrams already match the current code, reproduce them \
   character-for-character.
6. Preserve the exact Mermaid diagram types used in the original document \
   (flowchart TD, sequenceDiagram, classDiagram, mindmap, flowchart LR, etc.).
7. All Mermaid syntax must be valid and renderable by GitHub's built-in renderer.
8. Do not add diagram blocks that do not exist in the original document.
"""

# ── API helpers ────────────────────────────────────────────────────────────────


def _resolve_api_config(model_override: str | None = None) -> tuple[str, str, str]:
    """Resolve (api_key, base_url, model) from environment variables.

    Args:
        model_override: CLI --model value, or None to fall back to env/default.

    Returns:
        Tuple of (api_key, base_url, model).

    Raises:
        SystemExit: When no API credential is available.
    """
    openai_key: str = os.environ.get("MSR_OPENAI_API_KEY", "")
    github_token: str = (
        os.environ.get("MSR_GITHUB_TOKEN", "")
        or os.environ.get("GITHUB_TOKEN", "")
    )
    model: str = model_override or os.environ.get("MSR_OPENAI_MODEL", _DEFAULT_MODEL)

    if openai_key:
        api_key = openai_key
        base_url = os.environ.get("MSR_OPENAI_BASE_URL", "https://api.openai.com/v1")
        log.info("Using OpenAI-compatible API at %s", base_url)
    elif github_token:
        api_key = github_token
        base_url = os.environ.get("MSR_OPENAI_BASE_URL", _GITHUB_MODELS_BASE_URL)
        log.info("Using GitHub Models API (Copilot Pro) at %s", base_url)
    else:
        log.error(
            "No API credential found.  "
            "Set GITHUB_TOKEN (GitHub Copilot Pro) or MSR_OPENAI_API_KEY."
        )
        sys.exit(1)

    return api_key, base_url, model


def _call_llm(
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = _MAX_OUTPUT_TOKENS,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        api_key: Bearer token for the API.
        base_url: Base URL of the endpoint (no trailing slash needed).
        model: Model identifier string.
        max_tokens: Maximum number of tokens the model may produce.

    Returns:
        The assistant reply as a plain string (stripped of leading/trailing
        whitespace).

    Raises:
        urllib.error.HTTPError: On a non-2xx HTTP response.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        log.error("LLM API error HTTP %s: %s", exc.code, body[:500])
        raise

    return data["choices"][0]["message"]["content"].strip()


# ── File-reading helpers ───────────────────────────────────────────────────────


def _read_source_file(path: Path, max_chars: int = _MAX_FILE_CHARS) -> str:
    """Read a source file, truncating if it exceeds *max_chars*.

    Args:
        path: Absolute path to the file.
        max_chars: Maximum number of characters to return.

    Returns:
        File contents as a string.  Returns an error sentinel string when the
        file does not exist or cannot be read.
    """
    if not path.exists():
        return f"[file not found: {path.name}]"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[error reading {path.name}: {exc}]"

    if len(content) > max_chars:
        content = (
            content[:max_chars]
            + f"\n\n... [truncated — file exceeds {max_chars} chars] ..."
        )
    return content


def _collect_source_files(
    repo_root: Path,
    source_paths: list[str],
) -> dict[str, str]:
    """Collect source-file contents for one diagram configuration.

    Args:
        repo_root: Absolute path to the repository root.
        source_paths: Relative file paths (or directory paths ending in ``/``).
            Directories are expanded: all ``*.md`` and ``*.py`` files inside are
            included individually.

    Returns:
        Ordered dict mapping relative-path string → file-content string.
    """
    result: dict[str, str] = {}

    for rel in source_paths:
        abs_path = repo_root / rel

        if rel.endswith("/"):
            # Directory — include all .md and .py files (sorted for stability)
            if abs_path.is_dir():
                all_files = sorted(abs_path.glob("*.md")) + sorted(
                    abs_path.glob("*.py")
                )
                for fpath in all_files:
                    key = str(fpath.relative_to(repo_root))
                    result[key] = _read_source_file(fpath)
            else:
                log.warning("Directory not found, skipping: %s", abs_path)
        else:
            result[rel] = _read_source_file(abs_path)

    return result


# ── Prompt builder ─────────────────────────────────────────────────────────────


def _build_user_prompt(
    description: str,
    source_files: dict[str, str],
    existing_content: str,
) -> str:
    """Assemble the user turn of the chat prompt.

    Args:
        description: Human-readable description of what this diagram covers.
        source_files: Map of relative path → file content.
        existing_content: Current content of the architecture markdown file.

    Returns:
        A single formatted string to use as the ``user`` message.
    """
    _LANG = {
        "py": "python",
        "yaml": "yaml",
        "yml": "yaml",
        "md": "markdown",
        "toml": "toml",
    }

    parts: list[str] = [
        f"## What this architecture document covers\n\n{description}\n",
        "---\n",
        "## Current repository source files\n",
    ]

    for rel_path, content in source_files.items():
        ext = Path(rel_path).suffix.lstrip(".")
        lang = _LANG.get(ext, "")
        parts.append(f"### `{rel_path}`\n\n```{lang}\n{content}\n```\n")

    parts.append("---\n")
    parts.append("## Existing architecture document (your input template)\n")
    parts.append(existing_content)
    parts.append(
        "\n\n---\n\n"
        "Analyse every source file shown above.  Return the complete updated "
        "architecture document.  Change ONLY the Mermaid diagram blocks that no "
        "longer accurately reflect the current code.  Leave everything else "
        "character-for-character identical."
    )

    return "\n".join(parts)


# ── Core generation function ───────────────────────────────────────────────────


def generate_diagram_doc(
    config: dict,
    repo_root: Path,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    """Generate or update a single architecture document using the LLM.

    Args:
        config: Diagram configuration (``filename``, ``source_files``,
            ``description`` keys).
        repo_root: Absolute path to the repository root.
        api_key: API bearer token.
        base_url: API base URL.
        model: Model identifier.

    Returns:
        Updated markdown content ready to write to disk (always ends with
        a newline).
    """
    arch_dir = repo_root / "docs" / "architecture"
    doc_path = arch_dir / config["filename"]

    if doc_path.exists():
        existing_content = doc_path.read_text(encoding="utf-8")
    else:
        log.warning("Document not found — will create from scratch: %s", doc_path)
        existing_content = (
            f"# {config['filename']}\n\n"
            "<!-- This file was not found; please review the generated content -->\n"
        )

    source_files = _collect_source_files(repo_root, config["source_files"])
    description = config.get("description", f"Architecture for {config['filename']}")

    user_prompt = _build_user_prompt(description, source_files, existing_content)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    log.info("  Sending request to %s (model=%s) …", base_url, model)
    result = _call_llm(messages, api_key, base_url, model)

    # Normalise trailing newline
    return result.rstrip("\n") + "\n"


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and regenerate architecture diagram documents."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated content to stdout; do not write any files.",
    )
    parser.add_argument(
        "--file",
        metavar="FILENAME",
        help="Regenerate only one diagram (e.g. 03_mcp_server.md).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        default=None,
        help=f"Chat model to use (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--repo-root",
        metavar="PATH",
        default=None,
        help="Repository root directory (default: parent of the scripts/ directory).",
    )
    args = parser.parse_args()

    # ── Resolve repository root ────────────────────────────────────────────────
    if args.repo_root:
        repo_root = Path(args.repo_root).resolve()
    else:
        repo_root = Path(__file__).parent.parent.resolve()

    sentinel = repo_root / "msr_mcp_server.py"
    if not sentinel.exists():
        log.error(
            "msr_mcp_server.py not found under %s.  "
            "Run from the repo root or pass --repo-root.",
            repo_root,
        )
        sys.exit(1)

    log.info("Repository root: %s", repo_root)

    # ── Resolve API credentials ────────────────────────────────────────────────
    api_key, base_url, model = _resolve_api_config(args.model)
    log.info("Model: %s", model)

    # ── Select diagram configs ─────────────────────────────────────────────────
    if args.file:
        configs = [c for c in _DIAGRAM_CONFIGS if c["filename"] == args.file]
        if not configs:
            log.error("No diagram config for '%s'.", args.file)
            log.error(
                "Available files: %s",
                [c["filename"] for c in _DIAGRAM_CONFIGS],
            )
            sys.exit(1)
    else:
        configs = _DIAGRAM_CONFIGS

    arch_dir = repo_root / "docs" / "architecture"
    arch_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each diagram ───────────────────────────────────────────────────
    success = 0
    errors = 0

    for config in configs:
        log.info("─" * 60)
        log.info("Processing: %s", config["filename"])
        try:
            updated = generate_diagram_doc(config, repo_root, api_key, base_url, model)
        except Exception as exc:  # noqa: BLE001
            log.error("  FAILED: %s", exc)
            errors += 1
            continue

        if args.dry_run:
            separator = "=" * 60
            print(f"\n{separator}")
            print(f"DRY RUN — docs/architecture/{config['filename']}")
            print(separator)
            print(updated)
        else:
            out_path = arch_dir / config["filename"]
            out_path.write_text(updated, encoding="utf-8")
            log.info("  Written: %s", out_path.relative_to(repo_root))

        success += 1

    log.info("─" * 60)
    log.info("Completed — %d OK, %d failed.", success, errors)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
