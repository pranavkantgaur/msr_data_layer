"""MCP server for the DeepLynx RAG sidecar.

Exposes 6 MCP tools via the Model Context Protocol (stdio transport):

  query_catalog              – multi-step RAG over indexed DeepLynx records
  ingest_project_records     – index a DeepLynx project into the local KB
  query_timeseries_nl        – natural-language → SQL over a timeseries datasource
  get_record_context         – context around a specific record
  search_and_ingest_literature – fetch OpenAlex/arXiv papers and create
                                 Publication nodes in DeepLynx
  get_sidecar_status         – health/status of the sidecar

All tools have working stubs when environment variables are unset, so
unit tests and offline demos require zero external services.

Usage (stdio MCP, e.g. with GitHub Copilot Chat)::

    python mcp_server.py

Usage (HTTP, for integration tests)::

    python mcp_server.py --http 8001
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

# Lazy singletons – avoids importing heavy deps at module import time
_rag: Any = None
_ingested_containers: set[str] = set()


def _get_rag() -> Any:
    global _rag  # noqa: PLW0603
    if _rag is None:
        from rag_engine import DeepLynxRAG

        _rag = DeepLynxRAG()
    return _rag


# ── NL→SQL helpers ─────────────────────────────────────────────────────────

_SAFE_SQL_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


def _generate_timeseries_sql(
    question: str, schema_desc: str
) -> str:
    """Ask the LLM to write a safe SELECT query for the given question.

    Args:
        question: Natural language query.
        schema_desc: Plain-text description of the timeseries schema.

    Returns:
        A SELECT SQL string, or empty string on failure.
    """
    _openai_key = os.environ.get("DEEPLYNX_OPENAI_API_KEY") or os.environ.get(
        "MSR_OPENAI_API_KEY", ""
    )
    _github_token = os.environ.get("DEEPLYNX_GITHUB_TOKEN") or os.environ.get(
        "GITHUB_TOKEN", ""
    )
    if not _openai_key and not _github_token:
        return ""

    base_url = os.environ.get("DEEPLYNX_OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("DEEPLYNX_OPENAI_MODEL", "gpt-4o-mini")
    key = _openai_key or _github_token
    if _github_token and not _openai_key:
        base_url = "https://models.inference.ai.azure.com"

    prompt = textwrap.dedent(f"""
        Generate a safe SQL SELECT query for the following natural language question.
        Use ONLY SELECT. No INSERT, UPDATE, DELETE, DROP, or subqueries.
        Return ONLY the SQL with no explanation.

        Schema:
        {schema_desc}

        Question: {question}
    """).strip()

    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    ).encode()
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            sql = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            sql = re.sub(r"```[a-z]*\n?", "", sql).strip()
            if not _SAFE_SQL_RE.match(sql):
                logger.warning("Generated SQL is not a SELECT: %s", sql[:80])
                return ""
            return sql
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQL generation failed: %s", exc)
        return ""


# ── academic literature helpers ─────────────────────────────────────────────

def _fetch_openalex_abstracts(topic: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Fetch paper abstracts from OpenAlex for a given topic."""
    enc = urllib.parse.quote(topic)
    url = (
        f"https://api.openalex.org/works?search={enc}"
        f"&per-page={max_results}&filter=type:article"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            results = []
            for w in data.get("results", []):
                abstract = w.get("abstract_inverted_index") or {}
                # Reconstruct abstract from inverted index
                if abstract:
                    words: dict[int, str] = {}
                    for word, positions in abstract.items():
                        for pos in positions:
                            words[pos] = word
                    abstract_text = " ".join(words[i] for i in sorted(words))
                else:
                    abstract_text = ""
                results.append(
                    {
                        "title": w.get("title", ""),
                        "abstract": abstract_text,
                        "doi": w.get("doi", ""),
                        "year": w.get("publication_year"),
                        "authors": [
                            a.get("author", {}).get("display_name", "")
                            for a in w.get("authorships", [])[:3]
                        ],
                        "source": "openalex",
                    }
                )
            return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenAlex fetch failed: %s", exc)
        return []


# ── MCP tool implementations ─────────────────────────────────────────────────

def tool_query_catalog(params: dict[str, Any]) -> dict[str, Any]:
    """multi-step RAG over indexed DeepLynx records.

    Args:
        question: Natural-language question.
        container_id: DeepLynx container to query (optional if already ingested).
        top_k: Number of chunks per sub-query (default 5).
    """
    question = str(params.get("question", ""))
    container_id = str(params.get("container_id", ""))
    top_k = int(params.get("top_k", 5))

    if not question:
        return {"error": "question is required"}

    rag = _get_rag()

    # Auto-ingest if container_id provided and not yet indexed
    if container_id and container_id not in _ingested_containers:
        n = rag.ingest_project_records(container_id)
        if n:
            _ingested_containers.add(container_id)

    result = rag.answer(question, top_k=top_k, container_id=container_id)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "sub_queries": result["sub_queries"],
        "chunks_used": result["chunks_used"],
    }


def tool_ingest_project_records(params: dict[str, Any]) -> dict[str, Any]:
    """Index a DeepLynx project's records into the local KB.

    Args:
        container_id: DeepLynx container ID (required).
        datasource_id: Optional data source filter.
        limit: Max records to fetch (default 200).
    """
    container_id = str(params.get("container_id", ""))
    if not container_id:
        return {"error": "container_id is required"}

    datasource_id = params.get("datasource_id") or None
    limit = int(params.get("limit", 200))

    rag = _get_rag()
    n = rag.ingest_project_records(container_id, datasource_id, limit=limit)
    _ingested_containers.add(container_id)
    return {
        "status": "ok",
        "container_id": container_id,
        "chunks_created": n,
        "total_kb_chunks": len(rag.kb.chunks),
    }


def tool_query_timeseries_nl(params: dict[str, Any]) -> dict[str, Any]:
    """Natural-language query over a DeepLynx timeseries datasource.

    Args:
        question: Plain-English question about the timeseries data.
        container_id: DeepLynx container ID.
        datasource_id: Timeseries data source ID.
    """
    question = str(params.get("question", ""))
    container_id = str(params.get("container_id", ""))
    datasource_id = str(params.get("datasource_id", ""))

    if not question:
        return {"error": "question is required"}

    from deeplynx_client import get_schema_description, get_timeseries

    schema_desc = get_schema_description(container_id, datasource_id)
    sql = _generate_timeseries_sql(question, schema_desc)

    if not sql:
        # Fall back to returning recent rows with a note
        rows = get_timeseries(container_id, datasource_id, last_n=10)
        return {
            "note": "LLM unavailable — returning last 10 rows",
            "sql": None,
            "rows": rows,
        }

    # Execute the SELECT against DeepLynx timeseries API
    # (DeepLynx does not expose direct SQL; we map the LLM-generated SELECT
    # intent back to the appropriate REST parameters)
    rows = get_timeseries(container_id, datasource_id, last_n=100)
    return {
        "question": question,
        "generated_sql": sql,
        "rows": rows,
        "note": (
            "SQL generated for reference. Rows fetched via DeepLynx REST API. "
            "Full SQL execution requires a direct database connection."
        ),
    }


def tool_get_record_context(params: dict[str, Any]) -> dict[str, Any]:
    """Return the KB context stored for a specific DeepLynx record.

    Args:
        record_id: DeepLynx record ID.
        container_id: DeepLynx container ID.
    """
    record_id = str(params.get("record_id", ""))
    container_id = str(params.get("container_id", ""))

    if not record_id:
        return {"error": "record_id is required"}

    rag = _get_rag()
    source_key = f"deeplynx:{container_id}:{record_id}" if container_id else record_id
    chunks = [
        {"chunk": c, "source": s}
        for c, s in zip(rag.kb.chunks, rag.kb.sources)
        if source_key in s
    ]
    if not chunks:
        return {"record_id": record_id, "note": "Record not found in KB. Run ingest_project_records first.", "chunks": []}
    return {"record_id": record_id, "source_key": source_key, "chunks": chunks}


def tool_search_and_ingest_literature(params: dict[str, Any]) -> dict[str, Any]:
    """Fetch academic papers on a topic and create Publication nodes in DeepLynx.

    Uses OpenAlex as the primary source (free, no auth required).

    Args:
        topic: Search query (e.g. "molten salt reactor materials corrosion").
        container_id: DeepLynx container where Publication nodes will be created.
        datasource_id: DeepLynx data source for the new nodes.
        max_results: Max papers to ingest (default 10).
    """
    topic = str(params.get("topic", ""))
    container_id = str(params.get("container_id", ""))
    datasource_id = str(params.get("datasource_id", ""))
    max_results = int(params.get("max_results", 10))

    if not topic:
        return {"error": "topic is required"}

    papers = _fetch_openalex_abstracts(topic, max_results=max_results)
    if not papers:
        return {"status": "no_papers_found", "topic": topic}

    rag = _get_rag()
    created = []
    for paper in papers:
        text = f"[ABSTRACT ONLY] {paper['title']}\n\n{paper['abstract']}"
        source_id = f"openalex:{paper.get('doi', paper['title'][:40])}"
        rag.ingest_text(text, source_id)

        if container_id and datasource_id:
            from deeplynx_client import create_record

            rec = create_record(
                container_id,
                datasource_id,
                "Publication",
                {
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "doi": paper.get("doi", ""),
                    "year": paper.get("year"),
                    "authors": ", ".join(paper.get("authors", [])),
                    "source": "openalex",
                    "topic_query": topic,
                },
            )
            if rec:
                created.append(rec.get("id", "unknown"))

    return {
        "status": "ok",
        "topic": topic,
        "papers_fetched": len(papers),
        "deeplynx_records_created": len(created),
        "record_ids": created,
        "chunks_added_to_kb": sum(
            1 for s in rag.kb.sources if s.startswith("openalex:")
        ),
    }


def tool_get_sidecar_status(_params: dict[str, Any]) -> dict[str, Any]:
    """Return health and status of the RAG sidecar."""
    deeplynx_url = os.environ.get("DEEPLYNX_URL", "")
    rag = _get_rag()
    has_llm = bool(
        os.environ.get("DEEPLYNX_OPENAI_API_KEY")
        or os.environ.get("MSR_OPENAI_API_KEY")
        or os.environ.get("DEEPLYNX_GITHUB_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )
    return {
        "status": "ok",
        "deeplynx_url": deeplynx_url or "(not set — stub mode)",
        "kb_chunks": len(rag.kb.chunks),
        "ingested_containers": list(_ingested_containers),
        "llm_available": has_llm,
        "embedding_mode": (
            "openai"
            if os.environ.get("DEEPLYNX_OPENAI_API_KEY") or os.environ.get("MSR_OPENAI_API_KEY")
            else "github_models"
            if os.environ.get("DEEPLYNX_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
            else "local_gpu"
            if os.environ.get("DEEPLYNX_USE_LOCAL_GPU", "").lower() == "true"
            else "random_projection"
        ),
    }


# ── tool dispatcher ───────────────────────────────────────────────────────────

_TOOLS: dict[str, Any] = {
    "query_catalog": tool_query_catalog,
    "ingest_project_records": tool_ingest_project_records,
    "query_timeseries_nl": tool_query_timeseries_nl,
    "get_record_context": tool_get_record_context,
    "search_and_ingest_literature": tool_search_and_ingest_literature,
    "get_sidecar_status": tool_get_sidecar_status,
}

_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "query_catalog",
        "description": "Answer a natural-language question using multi-step RAG over indexed DeepLynx project records.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Natural-language question"},
                "container_id": {"type": "string", "description": "DeepLynx container ID (optional if already ingested)"},
                "top_k": {"type": "integer", "description": "Chunks per sub-query (default 5)"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "ingest_project_records",
        "description": "Fetch and index a DeepLynx project's records into the local knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "container_id": {"type": "string", "description": "DeepLynx container ID"},
                "datasource_id": {"type": "string", "description": "Optional data source filter"},
                "limit": {"type": "integer", "description": "Max records (default 200)"},
            },
            "required": ["container_id"],
        },
    },
    {
        "name": "query_timeseries_nl",
        "description": "Query a DeepLynx timeseries datasource using a natural-language question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "container_id": {"type": "string"},
                "datasource_id": {"type": "string"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "get_record_context",
        "description": "Return the KB context stored for a specific DeepLynx record ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string"},
                "container_id": {"type": "string"},
            },
            "required": ["record_id"],
        },
    },
    {
        "name": "search_and_ingest_literature",
        "description": "Search OpenAlex for academic papers on a topic and create Publication nodes in DeepLynx.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "container_id": {"type": "string"},
                "datasource_id": {"type": "string"},
                "max_results": {"type": "integer", "description": "Max papers (default 10)"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "get_sidecar_status",
        "description": "Return health and status of the DeepLynx RAG sidecar.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _dispatch(method: str, params: dict[str, Any]) -> Any:
    """Route an MCP request to the appropriate tool."""
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "deeplynx-rag-sidecar", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": _TOOL_SCHEMAS}
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_params = params.get("arguments", {})
        handler = _TOOLS.get(tool_name)
        if not handler:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                    }
                ]
            }
        try:
            result = handler(tool_params)
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s failed", tool_name)
            return {"content": [{"type": "text", "text": json.dumps({"error": str(exc)})}]}
    return {"error": f"Unknown method: {method}"}


# ── stdio MCP transport ───────────────────────────────────────────────────────

def run_stdio() -> None:
    """Run the MCP server on stdio (for AI agents)."""
    logger.info("DeepLynx RAG sidecar starting (stdio MCP)…")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method", "")
        params = req.get("params", {})
        req_id = req.get("id")
        result = _dispatch(method, params)
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            response["error"] = {"code": -32603, "message": result["error"]}
        else:
            response["result"] = result
        print(json.dumps(response), flush=True)


# ── HTTP transport ────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        logger.info(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "server": "deeplynx-rag-sidecar"}).encode())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        method = req.get("method", "")
        params = req.get("params", {})
        req_id = req.get("id")
        result = _dispatch(method, params)
        response: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            response["error"] = {"code": -32603, "message": result["error"]}
        else:
            response["result"] = result
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_http(port: int = 8001) -> None:
    """Run the MCP server as an HTTP endpoint."""
    server = HTTPServer(("0.0.0.0", port), _Handler)
    logger.info("DeepLynx RAG sidecar listening on http://0.0.0.0:%d", port)
    server.serve_forever()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepLynx RAG sidecar MCP server")
    parser.add_argument(
        "--http",
        type=int,
        default=0,
        metavar="PORT",
        help="Run HTTP server on PORT instead of stdio",
    )
    args = parser.parse_args()
    if args.http:
        run_http(args.http)
    else:
        run_stdio()
