"""
MSR Knowledge Base Service – AWS Lambda Handler

Exposes the MSR data layer MCP server and RAG knowledge base as an HTTPS
service via AWS Lambda + API Gateway (HTTP API v2).  A single Lambda function
handles all four concerns:

1. **MCP endpoint** (``POST /mcp``) – full JSON-RPC 2.0 / MCP protocol so any
   MCP-capable host (Claude, GitHub Copilot, custom agent) can talk to the
   data layer tools over HTTPS instead of stdio.

2. **Query endpoint** (``POST /query``) – simple REST wrapper around the RAG
   pipeline so agents or operators can send a plain-text question and receive
   a synthesised answer without implementing MCP.

3. **Deep research endpoint** (``POST /research/deep``) – expanded RAG query
   designed for deep research agents.  Retrieves a larger document window
   (default top_k = 15), collects all distinct source identifiers cited across
   sub-queries, and produces a comprehensive long-form research report with
   numbered citations and an open-questions section.

4. **Knowledge-base update endpoint** (``POST /kb/update``) – triggers
   ingestion from the static msr-archive and/or the dynamic OpenAlex source.
   Can also be invoked on a schedule via Amazon EventBridge (the same function
   is the target; see ``template.yaml``).

5. **Plant data ingestion endpoint** (``POST /data/ingest``) – accepts plant
   operational data (sensor snapshots, event logs, maintenance reports) from
   operators or agents and ingests it into the RAG knowledge base.

6. **Timeseries query endpoint** (``POST /timeseries/query``) – accepts either
   a structured ``{sensor_name, start_time, end_time, last_n, aggregation}``
   payload or a natural-language ``{question}`` payload and returns sensor
   readings or statistics from the SQLite timeseries store.

7. **Timeseries ingest endpoint** (``POST /timeseries/ingest``) – accepts
   ``{readings: [...], source_id, data_type}`` and inserts timestamped sensor
   readings into the SQLite timeseries store (and optionally the RAG KB).

8. **Health endpoint** (``GET /health``) – returns 200 with service metadata
   for load-balancer checks and uptime monitoring.

Knowledge Base Persistence
--------------------------
Lambda execution environments are ephemeral.  The KB files (``chunks.json``,
``embeddings.npy``, ``insights.json``, ``tfidf.json``, plus loader state) are
synced to an **S3 bucket** so they survive across invocations:

* On cold start the function downloads all KB files from S3 to ``/tmp/kb_store``.
* After every mutating operation (``/kb/update``, ``/data/ingest``, or
  scheduled refresh) the updated files are uploaded back to S3.

S3 sync is **optional** – if ``MSR_KB_S3_BUCKET`` is not set the function
works entirely in ``/tmp`` (KB is rebuilt on every cold start from ``MSR_DOCS_DIR``
or from scratch).

In-memory warm-Lambda caching
------------------------------
The ``MSRDigitalTwinRAG`` instance is kept in a module-level variable and
reused across requests served by the same warm Lambda container, avoiding
repeated cold-start KB loading.

Authentication
--------------
Set the ``MSR_API_KEY`` environment variable to enable simple bearer-token
auth.  Requests must then include an ``X-Api-Key`` header matching that value.
Leave the variable unset to disable authentication (useful for testing).

Environment Variables
---------------------
MSR_KB_S3_BUCKET       S3 bucket name for KB persistence (no sync if unset)
MSR_KB_S3_PREFIX       S3 key prefix (default: ``kb/``)
MSR_API_KEY            Shared API key for request authentication (optional)
MSR_PLANT_DATA_URL     URL of external plant data REST API (optional; when
                       unset, the development stub is used for sensor reads)
MSR_GITHUB_TOKEN       GitHub personal access token – used with the GitHub
                       Models API (https://models.inference.ai.azure.com) for
                       LLM and embeddings.  Set automatically to ``GITHUB_TOKEN``
                       inside GitHub Codespaces (see ``.devcontainer/``).
                       Takes effect only when ``MSR_OPENAI_API_KEY`` is unset.
MSR_OPENAI_API_KEY     OpenAI key for LLM + embeddings (takes precedence over
                       ``MSR_GITHUB_TOKEN`` when both are set)
MSR_OPENAI_BASE_URL    OpenAI-compatible API base URL
MSR_OPENAI_MODEL       Chat model (default: gpt-4o-mini)
MSR_EMBED_MODEL        Embedding model (default: text-embedding-3-small)
MSR_DOCS_DIR           Local documents directory (default: /tmp/docs)
MSR_USE_LOCAL_GPU      Set to ``true`` to use local GPU models for embeddings
                       and response generation instead of the OpenAI API.
                       Requires the GPU container image (see ``Dockerfile.gpu``).
MSR_LOCAL_EMBED_MODEL  HuggingFace embedding model (default:
                       ``sentence-transformers/all-MiniLM-L6-v2``)
MSR_LOCAL_LLM_MODEL    HuggingFace generation model (default:
                       ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``)
MSR_HF_CACHE_DIR       HuggingFace model cache (default: ``/tmp/hf_cache``)

Deployment
----------
See ``template.yaml`` (AWS SAM), ``Dockerfile.gpu`` (GPU container), and
``Makefile`` for build/deploy commands.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Optional boto3 (not available in local unit-test runs unless installed)
# ---------------------------------------------------------------------------

try:
    import boto3  # type: ignore[import-untyped]
    from botocore.exceptions import BotoCoreError, ClientError as S3ClientError  # type: ignore[import-untyped]
    _BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BOTO3_AVAILABLE = False
    BotoCoreError = Exception  # type: ignore[misc, assignment]
    S3ClientError = Exception  # type: ignore[misc, assignment]

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

_KB_LOCAL_DIR = "/tmp/kb_store"
_S3_BUCKET = os.environ.get("MSR_KB_S3_BUCKET", "")
_S3_PREFIX = os.environ.get("MSR_KB_S3_PREFIX", "kb/").rstrip("/") + "/"
_API_KEY = os.environ.get("MSR_API_KEY", "")

# KB files that should be synced to/from S3
_KB_FILES = [
    "chunks.json",
    "embeddings.npy",
    "insights.json",
    "tfidf.json",
    "archive_state.json",
    "openalex_state.json",
    "plant_data_state.json",
]

# ---------------------------------------------------------------------------
# Module-level warm-Lambda cache
# ---------------------------------------------------------------------------

_rag_cache: Any = None   # MSRDigitalTwinRAG instance


# ---------------------------------------------------------------------------
# S3 sync helpers
# ---------------------------------------------------------------------------

def _s3_client() -> Any:
    return boto3.client("s3")  # type: ignore[attr-defined]


def sync_kb_from_s3() -> None:
    """
    Download KB files from S3 to ``/tmp/kb_store``.

    No-op when ``MSR_KB_S3_BUCKET`` is unset or boto3 is unavailable.
    """
    if not _S3_BUCKET or not _BOTO3_AVAILABLE:
        return
    kb_path = Path(_KB_LOCAL_DIR)
    kb_path.mkdir(parents=True, exist_ok=True)
    s3 = _s3_client()
    for filename in _KB_FILES:
        s3_key = f"{_S3_PREFIX}{filename}"
        local_path = kb_path / filename
        try:
            s3.download_file(_S3_BUCKET, s3_key, str(local_path))
            logger.info("KB sync: downloaded s3://%s/%s", _S3_BUCKET, s3_key)
        except (S3ClientError, BotoCoreError) as exc:
            # File may not exist yet (first run) – that's fine
            logger.debug("KB sync: %s not in S3 yet (%s)", filename, exc)


def sync_kb_to_s3() -> None:
    """
    Upload KB files from ``/tmp/kb_store`` to S3.

    No-op when ``MSR_KB_S3_BUCKET`` is unset or boto3 is unavailable.
    """
    if not _S3_BUCKET or not _BOTO3_AVAILABLE:
        return
    kb_path = Path(_KB_LOCAL_DIR)
    if not kb_path.is_dir():
        return
    s3 = _s3_client()
    for filename in _KB_FILES:
        local_path = kb_path / filename
        if not local_path.exists():
            continue
        s3_key = f"{_S3_PREFIX}{filename}"
        try:
            s3.upload_file(str(local_path), _S3_BUCKET, s3_key)
            logger.info("KB sync: uploaded s3://%s/%s", _S3_BUCKET, s3_key)
        except (S3ClientError, BotoCoreError) as exc:
            logger.warning("KB sync: failed to upload %s: %s", filename, exc)


# ---------------------------------------------------------------------------
# RAG instance (warm-Lambda cache)
# ---------------------------------------------------------------------------

def _get_rag() -> Any:
    """
    Return the module-level cached RAG instance, initialising it on first call.

    Sets ``MSR_KB_DIR`` to ``/tmp/kb_store`` before construction so the KB
    files end up in Lambda's ephemeral storage.
    """
    global _rag_cache  # noqa: PLW0603
    if _rag_cache is None:
        os.environ.setdefault("MSR_KB_DIR", _KB_LOCAL_DIR)
        os.environ.setdefault("MSR_DOCS_DIR", "/tmp/docs")
        sync_kb_from_s3()
        from msr_digital_twin_with_rag import MSRDigitalTwinRAG  # noqa: PLC0415
        _rag_cache = MSRDigitalTwinRAG()
        logger.info("RAG instance initialised (warm cache).")
    return _rag_cache


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _response(
    status: int,
    body: Any,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    return {
        "statusCode": status,
        "headers": headers,
        "body": json.dumps(body, indent=2),
    }


def _error(status: int, message: str) -> dict[str, Any]:
    return _response(status, {"error": message})


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _is_authenticated(event: dict[str, Any]) -> bool:
    """Return True if the request carries a valid API key (or auth is disabled)."""
    if not _API_KEY:
        return True  # Auth disabled
    headers = event.get("headers") or {}
    # API Gateway v2 lower-cases header names
    provided = (
        headers.get("x-api-key")
        or headers.get("X-Api-Key")
        or headers.get("Authorization", "").removeprefix("Bearer ").strip()
    )
    return provided == _API_KEY


# ---------------------------------------------------------------------------
# Request body parsing (supports base64-encoded bodies from API Gateway)
# ---------------------------------------------------------------------------

def _parse_body(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Return *(raw_str, parsed_dict)* from the Lambda event body.

    Handles base64-encoded bodies (binary payloads via API Gateway).
    """
    raw = event.get("body") or ""
    if event.get("isBase64Encoded") and raw:
        try:
            raw = base64.b64decode(raw).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            raw = ""
    parsed: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw, parsed


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def _handle_health() -> dict[str, Any]:
    """``GET /health`` – service liveness check."""
    from msr_mcp_server import _get_current_state, get_data_source_info  # noqa: PLC0415
    from msr_digital_twin_with_rag import _gpu_device, _TORCH_AVAILABLE  # noqa: PLC0415
    state = _get_current_state()
    ds_info = get_data_source_info()
    # Include current plant status in the data_source info block
    ds_info["plant_status"] = state.get("status", "UNKNOWN")
    use_local_gpu = os.environ.get("MSR_USE_LOCAL_GPU", "").lower() in ("1", "true", "yes")
    return _response(200, {
        "service": "msr-knowledge-base",
        "version": "1.0.0",
        "status": "healthy",
        "data_source": ds_info,
        "kb_dir": _KB_LOCAL_DIR,
        "s3_bucket": _S3_BUCKET or "(not configured)",
        "gpu": {
            "torch_available": _TORCH_AVAILABLE,
            "device": _gpu_device() if _TORCH_AVAILABLE else "cpu",
            "local_gpu_mode": use_local_gpu,
            "embed_model": os.environ.get(
                "MSR_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ) if use_local_gpu else None,
            "llm_model": os.environ.get(
                "MSR_LOCAL_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ) if use_local_gpu else None,
        },
    })


def _handle_mcp(raw_body: str) -> dict[str, Any]:
    """
    ``POST /mcp`` – MCP JSON-RPC 2.0 endpoint.

    Accepts a single MCP message (JSON-RPC 2.0 object) and returns the
    server's response.  MCP-capable hosts (Claude Desktop, VS Code Copilot,
    custom agents) can point to this URL instead of the stdio transport.
    """
    if not raw_body.strip():
        return _error(400, "Request body must be a JSON-RPC 2.0 message.")
    from msr_mcp_server import handle_message  # noqa: PLC0415
    response_str = handle_message(raw_body)
    if not response_str:
        return _response(204, {})
    try:
        return _response(200, json.loads(response_str))
    except json.JSONDecodeError:
        return _response(200, {"raw": response_str})


def _handle_query(body: dict[str, Any], request_id: str = "") -> dict[str, Any]:
    """
    ``POST /query`` – RAG query endpoint.

    Request body::

        {
          "question": "What is the thermal efficiency of TMSR-LF1?",
          "top_k": 5          // optional, default 5
        }

    Response::

        {
          "question": "...",
          "answer": "...",
          "top_k": 5
        }
    """
    req_id = request_id or uuid.uuid4().hex[:12]
    question = body.get("question", "").strip()
    if not question:
        return _error(400, "Request body must include a non-empty 'question' field.")
    top_k = int(body.get("top_k", 5))
    top_k = max(1, min(top_k, 20))
    include_diagnostics = bool(body.get("diagnostics") or body.get("debug"))

    timeout_s = float(os.environ.get("MSR_QUERY_TIMEOUT_SECONDS", "55"))
    timeout_s = max(5.0, timeout_s)

    logger.info(
        "[query:%s] start top_k=%d timeout_s=%.1f question_chars=%d",
        req_id,
        top_k,
        timeout_s,
        len(question),
    )

    rag = _get_rag()
    started = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rag.answer, question, top_k, req_id)
            answer = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.error(
            "[query:%s] timeout after %.1fms (gateway likely timed out first if >~60s)",
            req_id,
            elapsed,
        )
        payload: dict[str, Any] = {
            "error": f"Query timed out after {timeout_s:.1f}s",
            "request_id": req_id,
            "elapsed_ms": round(elapsed, 1),
            "hint": (
                "RAG query exceeded the configured server timeout. "
                "Check stage-level logs for this request_id."
            ),
        }
        return _response(504, payload)
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.exception("[query:%s] RAG query failed after %.1fms", req_id, elapsed)
        payload = {
            "error": f"RAG query failed: {exc}",
            "request_id": req_id,
            "elapsed_ms": round(elapsed, 1),
        }
        return _response(500, payload)

    elapsed = (time.perf_counter() - started) * 1000.0
    logger.info("[query:%s] success elapsed_ms=%.1f", req_id, elapsed)

    response_body: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "top_k": top_k,
    }
    if include_diagnostics:
        response_body["diagnostics"] = {
            "request_id": req_id,
            "elapsed_ms": round(elapsed, 1),
            "timeout_seconds": timeout_s,
            "question_chars": len(question),
            "top_k": top_k,
        }
    return _response(200, response_body)


def _handle_deep_research(body: dict[str, Any], request_id: str = "") -> dict[str, Any]:
    """
    ``POST /research/deep`` – deep research endpoint for AI research agents.

    Runs an expanded multi-step RAG pipeline that retrieves more documents
    per sub-query (default *top_k* = 15), collects all distinct source
    identifiers referenced, and produces a comprehensive long-form research
    report with numbered citations.

    Request body::

        {
          "question": "What corrosion mechanisms affect 316L SS in FLiNaK?",
          "top_k": 15          // optional, default 15, clamped to [1, 50]
        }

    Response::

        {
          "question": "...",
          "report": "...",
          "sources": ["source_id_1", "source_id_2", ...],
          "source_count": 7,
          "top_k": 15
        }
    """
    req_id = request_id or uuid.uuid4().hex[:12]
    question = body.get("question", "").strip()
    if not question:
        return _error(400, "Request body must include a non-empty 'question' field.")
    top_k = int(body.get("top_k", 15))
    top_k = max(1, min(top_k, 50))
    include_diagnostics = bool(body.get("diagnostics") or body.get("debug"))

    timeout_s = float(os.environ.get("MSR_DEEP_RESEARCH_TIMEOUT_SECONDS", "115"))
    timeout_s = max(10.0, timeout_s)

    logger.info(
        "[deep_research:%s] start top_k=%d timeout_s=%.1f question_chars=%d",
        req_id,
        top_k,
        timeout_s,
        len(question),
    )

    rag = _get_rag()
    started = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rag.deep_research, question, top_k, req_id)
            result = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.error(
            "[deep_research:%s] timeout after %.1fms",
            req_id,
            elapsed,
        )
        payload: dict[str, Any] = {
            "error": f"Deep research timed out after {timeout_s:.1f}s",
            "request_id": req_id,
            "elapsed_ms": round(elapsed, 1),
            "hint": (
                "Deep research exceeded the configured timeout.  "
                "Try a more specific question or reduce top_k."
            ),
        }
        return _response(504, payload)
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.exception(
            "[deep_research:%s] deep research failed after %.1fms", req_id, elapsed
        )
        payload = {
            "error": f"Deep research failed: {exc}",
            "request_id": req_id,
            "elapsed_ms": round(elapsed, 1),
        }
        return _response(500, payload)

    elapsed = (time.perf_counter() - started) * 1000.0
    logger.info("[deep_research:%s] success elapsed_ms=%.1f", req_id, elapsed)

    response_body: dict[str, Any] = {
        "question": question,
        "report": result["report"],
        "sources": result["sources"],
        "source_count": result["source_count"],
        "top_k": top_k,
    }
    if include_diagnostics:
        response_body["diagnostics"] = {
            "request_id": req_id,
            "elapsed_ms": round(elapsed, 1),
            "timeout_seconds": timeout_s,
            "question_chars": len(question),
            "top_k": top_k,
        }
    return _response(200, response_body)


def _handle_plant_data_ingest(body: dict[str, Any]) -> dict[str, Any]:
    """
    ``POST /data/ingest`` – ingest plant operational data into the KB.

    Request body::

        {
          "content":   "Core temp 702°C at 14:32 UTC, flow 248 kg/s",
          "data_type": "sensor_snapshot",  // optional, default "operational_data"
          "source_id": "shift-log-2024-01-15-1432"  // optional, auto-generated if absent
        }

    ``content`` can also be a JSON-encoded sensor snapshot or event log.

    Response::

        {
          "source_id":    "shift-log-2024-01-15-1432",
          "data_type":    "sensor_snapshot",
          "chunks_added": 2
        }
    """
    content = (body.get("content") or "").strip()
    if not content:
        return _error(400, "Request body must include a non-empty 'content' field.")

    data_type = (body.get("data_type") or "operational_data").strip()
    valid_types = {"sensor_snapshot", "event_log", "maintenance_report", "operational_data"}
    if data_type not in valid_types:
        return _error(
            400,
            f"data_type must be one of: {', '.join(sorted(valid_types))}."
        )

    source_id = (body.get("source_id") or "").strip()
    if not source_id:
        source_id = f"{data_type}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"

    rag = _get_rag()
    try:
        from msr_kb_sources import PlantDataLoader  # noqa: PLC0415
        loader = PlantDataLoader()
        chunks_added = loader.ingest_text(rag, content, source_id, data_type=data_type)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Plant data ingestion failed")
        return _error(500, f"Ingestion failed: {exc}")

    # Persist updated KB back to S3
    sync_kb_to_s3()

    return _response(200, {
        "source_id": source_id,
        "data_type": data_type,
        "chunks_added": chunks_added,
    })


def _handle_kb_update(body: dict[str, Any]) -> dict[str, Any]:
    """
    ``POST /kb/update`` – trigger KB ingestion from one or more sources.

    Request body::

        {
          "source": "archive" | "openalex" | "arxiv" | "semanticscholar" | "all",
          "max_docs": 20   // optional
        }

    Response::

        {
          "source": "all",
          "added": {"archive": 5, "openalex": 3, "arxiv": 2, "semanticscholar": 1}
        }
    """
    source = body.get("source", "all").lower().strip()
    if source not in ("archive", "openalex", "arxiv", "semanticscholar", "all"):
        return _error(400, "source must be 'archive', 'openalex', 'arxiv', 'semanticscholar', or 'all'.")
    max_docs_raw = body.get("max_docs", 0)
    try:
        max_docs = int(max_docs_raw)
    except (TypeError, ValueError):
        max_docs = 0

    rag = _get_rag()
    added: dict[str, int] = {}
    try:
        if source in ("archive", "all"):
            added["archive"] = rag.load_msr_archive(max_docs=max_docs)
        if source in ("openalex", "all"):
            added["openalex"] = rag.update_openalex(
                max_docs=max_docs or None
            )
        if source in ("arxiv", "all"):
            added["arxiv"] = rag.update_arxiv(
                max_docs=max_docs or None
            )
        if source in ("semanticscholar", "all"):
            added["semanticscholar"] = rag.update_semanticscholar(
                max_docs=max_docs or None
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("KB update failed")
        return _error(500, f"KB update failed: {exc}")

    # Persist updated KB back to S3
    sync_kb_to_s3()

    total = sum(added.values())
    return _response(200, {
        "source": source,
        "added": added,
        "total_new_documents": total,
    })


def _handle_scheduled_kb_update(event: dict[str, Any]) -> dict[str, Any]:
    """
    Handle an Amazon EventBridge scheduled invocation.

    Triggers a full KB update (both archive + OpenAlex) and persists to S3.
    The function returns a dict that EventBridge ignores but is visible in
    CloudWatch Logs.
    """
    logger.info("Scheduled KB update triggered by EventBridge.")
    rag = _get_rag()
    added: dict[str, int] = {}
    try:
        added["archive"] = rag.load_msr_archive()
        added["openalex"] = rag.update_openalex()
        added["arxiv"] = rag.update_arxiv()
        added["semanticscholar"] = rag.update_semanticscholar()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scheduled KB update failed")
        return {"success": False, "error": str(exc)}
    sync_kb_to_s3()
    logger.info("Scheduled KB update complete: %s", added)
    return {"success": True, "added": added}


def _handle_timeseries_ingest(body: dict[str, Any]) -> dict[str, Any]:
    """
    ``POST /timeseries/ingest`` – ingest timestamped sensor readings into the
    SQLite timeseries store (and optionally into the RAG KB as text).

    Expected body::

        {
            "readings": [
                {"sensor_name": "reactor_power_mw", "value": 99.8,
                 "unit": "MW", "timestamp": "2024-01-15T14:00:00Z"},
                ...
            ],
            "source_id": "scada-20240115T1400Z",
            "data_type": "sensor_snapshot",   // optional
            "also_ingest_text": true           // optional (default true)
        }
    """
    readings = body.get("readings")
    if not readings or not isinstance(readings, list):
        return _error(400, "'readings' must be a non-empty list.")

    source_id: str = body.get("source_id", "")
    if not source_id:
        source_id = f"timeseries-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    data_type: str = body.get("data_type", "sensor_snapshot")
    also_ingest_text: bool = bool(body.get("also_ingest_text", True))

    try:
        from msr_kb_sources import KBSourceManager  # noqa: PLC0415

        rag = _get_rag()
        mgr = KBSourceManager(rag)
        result = mgr.ingest_timeseries(
            readings,
            source_id=source_id,
            data_type=data_type,
            also_ingest_text=also_ingest_text,
        )
        if result["timeseries_rows"] > 0:
            sync_kb_to_s3()
        return _response(200, {
            "success": True,
            "source_id": source_id,
            "timeseries_rows": result["timeseries_rows"],
            "rag_chunks": result["rag_chunks"],
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("Timeseries ingest failed")
        return _error(500, f"Timeseries ingest failed: {exc}")


def _handle_timeseries_query(body: dict[str, Any]) -> dict[str, Any]:
    """
    ``POST /timeseries/query`` – query the SQLite timeseries store.

    Supports two query modes:

    **Structured mode** – provide ``sensor_name`` and optional bounds::

        {
            "sensor_name": "reactor_power_mw",
            "start_time": "2024-01-15T00:00:00Z",   // optional
            "end_time":   "2024-01-15T23:59:59Z",    // optional
            "last_n": 10,                             // optional
            "aggregation": "avg"                      // optional: avg/min/max/count
        }

    **Natural-language mode** – provide ``question``::

        {
            "question": "What was the average reactor power last week?"
        }
    """
    question: str = body.get("question", "").strip()
    if question:
        # NL→SQL mode
        try:
            from msr_kb_sources import KBSourceManager  # noqa: PLC0415

            rag = _get_rag()
            mgr = KBSourceManager(rag)
            result = mgr.query_timeseries_nl(question)
            return _response(200, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("NL timeseries query failed")
            return _error(500, f"Query failed: {exc}")

    # Structured mode
    sensor_name: str = body.get("sensor_name", "").strip()
    if not sensor_name:
        return _error(400, "Provide 'sensor_name' for structured queries or 'question' for NL queries.")

    try:
        from msr_kb_sources import KBSourceManager  # noqa: PLC0415

        rag = _get_rag()
        mgr = KBSourceManager(rag)
        result = mgr.query_timeseries(
            sensor_name,
            start=body.get("start_time") or None,
            end=body.get("end_time") or None,
            last_n=int(body.get("last_n", 0)) or None,
            aggregation=body.get("aggregation") or None,
        )
        return _response(200, result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Structured timeseries query failed")
        return _error(500, f"Query failed: {exc}")


# ---------------------------------------------------------------------------
# API Gateway event router
# ---------------------------------------------------------------------------

def _route_http(event: dict[str, Any]) -> dict[str, Any]:
    """Route an API Gateway HTTP API v2 (or v1) event to the right handler."""
    # API Gateway v2 uses requestContext.http; v1 uses httpMethod + path
    rc = event.get("requestContext", {})
    http_ctx = rc.get("http", {})
    method = (http_ctx.get("method") or event.get("httpMethod") or "GET").upper()
    path = event.get("rawPath") or event.get("path") or "/"

    # Strip trailing slash for normalisation (keep root "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    if not _is_authenticated(event):
        return _error(403, "Forbidden: invalid or missing X-Api-Key header.")

    raw_body, parsed_body = _parse_body(event)
    headers = event.get("headers") or {}
    request_id = (
        headers.get("x-request-id")
        or headers.get("X-Request-Id")
        or headers.get("x-correlation-id")
        or headers.get("X-Correlation-Id")
        or uuid.uuid4().hex[:12]
    )

    # Health / root
    if path in ("/health", "/") and method == "GET":
        return _handle_health()

    # MCP JSON-RPC endpoint
    if path == "/mcp" and method == "POST":
        return _handle_mcp(raw_body)

    # RAG query endpoint
    if path == "/query" and method == "POST":
        return _handle_query(parsed_body, request_id=request_id)

    # Deep research endpoint
    if path == "/research/deep" and method == "POST":
        return _handle_deep_research(parsed_body, request_id=request_id)

    # KB update endpoint
    if path == "/kb/update" and method == "POST":
        return _handle_kb_update(parsed_body)

    # Plant data ingestion endpoint
    if path == "/data/ingest" and method == "POST":
        return _handle_plant_data_ingest(parsed_body)

    # Timeseries ingest endpoint
    if path == "/timeseries/ingest" and method == "POST":
        return _handle_timeseries_ingest(parsed_body)

    # Timeseries query endpoint
    if path == "/timeseries/query" and method == "POST":
        return _handle_timeseries_query(parsed_body)

    return _error(404, f"Not found: {method} {path}")


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda entry point.

    Handles two invocation types:

    * **HTTP (API Gateway)** – routes by path/method (see :func:`_route_http`).
    * **Scheduled (EventBridge)** – triggers a full KB update
      (``event["source"] == "aws.events"``).
    """
    # EventBridge scheduled trigger
    if event.get("source") == "aws.events" or event.get("detail-type") == "Scheduled Event":
        return _handle_scheduled_kb_update(event)

    # HTTP API Gateway (v2 or v1)
    if "requestContext" in event or "httpMethod" in event:
        return _route_http(event)

    # Direct Lambda invocation with a plain dict (e.g. from another Lambda)
    return _route_http({
        "requestContext": {"http": {"method": event.get("method", "POST")}},
        "rawPath": event.get("path", "/query"),
        "body": json.dumps(event),
        "isBase64Encoded": False,
        "headers": event.get("headers", {}),
    })
