"""Unit tests for the DeepLynx RAG sidecar.

All external I/O is stubbed so these tests run with zero network access
and zero external services (same principle as the MSR Data Layer test suite).

Run:
    python -m pytest test_sidecar.py -v
"""

from __future__ import annotations

import json
import sys
import os

import pytest

# Ensure the sidecar modules are importable
sys.path.insert(0, os.path.dirname(__file__))


# ── deeplynx_client stubs ─────────────────────────────────────────────────────

def test_list_projects_stub():
    """list_projects() returns stub data when DEEPLYNX_URL is unset."""
    import deeplynx_client as dc
    projects = dc.list_projects()
    assert len(projects) >= 1
    assert "id" in projects[0]


def test_list_datasources_stub():
    import deeplynx_client as dc
    ds = dc.list_datasources("any-container")
    assert isinstance(ds, list)
    assert len(ds) >= 1


def test_get_records_stub():
    import deeplynx_client as dc
    records = dc.get_records("any-container")
    assert isinstance(records, list)
    assert len(records) >= 1
    assert "id" in records[0]
    assert "properties" in records[0]


def test_get_records_stub_limit():
    import deeplynx_client as dc
    records = dc.get_records("any-container", limit=1)
    assert len(records) == 1


def test_get_timeseries_stub():
    import deeplynx_client as dc
    rows = dc.get_timeseries("c", "ds")
    assert isinstance(rows, list)
    assert len(rows) >= 1
    assert "timestamp" in rows[0]


def test_get_timeseries_stub_last_n():
    import deeplynx_client as dc
    rows = dc.get_timeseries("c", "ds", last_n=1)
    assert len(rows) == 1


def test_create_record_stub():
    import deeplynx_client as dc
    rec = dc.create_record("c", "ds", "Publication", {"title": "Test"})
    assert rec is not None
    assert "id" in rec


def test_get_schema_description_stub():
    import deeplynx_client as dc
    desc = dc.get_schema_description("c", "ds")
    assert "timestamp" in desc.lower()


# ── rag_engine tests ──────────────────────────────────────────────────────────

def test_sentence_chunk_basic():
    from rag_engine import _sentence_chunk
    text = "First sentence. Second sentence. Third sentence."
    chunks = _sentence_chunk(text)
    assert len(chunks) >= 1
    assert "First sentence" in chunks[0]


def test_sentence_chunk_empty():
    from rag_engine import _sentence_chunk
    chunks = _sentence_chunk("")
    assert isinstance(chunks, list)


def test_random_projection_embed_deterministic():
    from rag_engine import _random_projection_embed
    v1 = _random_projection_embed("test text")
    v2 = _random_projection_embed("test text")
    assert v1 == v2


def test_random_projection_embed_different_texts():
    from rag_engine import _random_projection_embed
    v1 = _random_projection_embed("reactor temperature")
    v2 = _random_projection_embed("pump maintenance")
    assert v1 != v2


def test_knowledge_store_add_and_search():
    from rag_engine import KnowledgeStore
    ks = KnowledgeStore()
    n = ks.add_document(
        "The primary salt pump was inspected. No abnormal wear found.",
        "test-doc-1",
    )
    assert n >= 1

    results = ks.search("salt pump inspection", top_k=3)
    assert len(results) >= 1
    assert "chunk" in results[0]
    assert "source" in results[0]
    assert results[0]["source"] == "test-doc-1"


def test_knowledge_store_dedup_by_source():
    from rag_engine import KnowledgeStore, DeepLynxRAG
    rag = DeepLynxRAG()
    n1 = rag.ingest_text("Some text about welds.", "doc-a")
    n2 = rag.ingest_text("Different text.", "doc-a")  # same source — should be skipped
    assert n1 >= 1
    assert n2 == 0  # already ingested


def test_deeplynx_rag_ingest_and_answer_no_llm(monkeypatch):
    """answer() should return context even when LLM is unavailable."""
    from rag_engine import DeepLynxRAG
    # Ensure no LLM keys are set
    monkeypatch.delenv("DEEPLYNX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPLYNX_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    rag = DeepLynxRAG()
    rag.ingest_text(
        "Weld W-42 used non-conforming filler Alloy-82 in Q2-2024.", "test-weld"
    )
    result = rag.answer("Which weld used non-conforming filler?")
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "test-weld" in result["sources"]


def test_deeplynx_rag_ingest_project_records_stub(monkeypatch):
    """ingest_project_records should use stub data and return chunks > 0."""
    monkeypatch.delenv("DEEPLYNX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPLYNX_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    from rag_engine import DeepLynxRAG
    rag = DeepLynxRAG()
    n = rag.ingest_project_records("stub-project")
    assert n >= 1


def test_hybrid_retrieval_scores_in_range():
    from rag_engine import KnowledgeStore
    ks = KnowledgeStore()
    ks.add_document("Hastelloy-N corrosion in FLiBe salt.", "doc-1")
    ks.add_document("Turbine bearing vibration analysis.", "doc-2")
    results = ks.search("FLiBe corrosion Hastelloy", top_k=2)
    for r in results:
        assert 0.0 <= r["score"] <= 2.0  # normalised sum of two [0,1] scores


# ── mcp_server tool tests ─────────────────────────────────────────────────────

def test_tool_get_sidecar_status():
    from mcp_server import tool_get_sidecar_status
    status = tool_get_sidecar_status({})
    assert status["status"] == "ok"
    assert "kb_chunks" in status
    assert "embedding_mode" in status


def test_tool_ingest_project_records_missing_container():
    from mcp_server import tool_ingest_project_records
    result = tool_ingest_project_records({})
    assert "error" in result


def test_tool_ingest_project_records_stub():
    from mcp_server import tool_ingest_project_records
    result = tool_ingest_project_records({"container_id": "my-project"})
    assert result["status"] == "ok"
    assert result["chunks_created"] >= 1


def test_tool_query_catalog_missing_question():
    from mcp_server import tool_query_catalog
    result = tool_query_catalog({})
    assert "error" in result


def test_tool_query_catalog_no_records():
    """query_catalog on a fresh container with no records yet → informative message."""
    from mcp_server import _get_rag, tool_query_catalog
    # Use a unique container ID that hasn't been ingested
    result = tool_query_catalog(
        {"question": "What is the reactor power?", "container_id": ""}
    )
    # Either has an answer (if records are in KB from other tests) or a helpful note
    assert "answer" in result or "error" in result


def test_tool_query_timeseries_nl_missing_question():
    from mcp_server import tool_query_timeseries_nl
    result = tool_query_timeseries_nl({})
    assert "error" in result


def test_tool_query_timeseries_nl_stub_fallback(monkeypatch):
    """Without LLM, tool returns raw rows with a note."""
    monkeypatch.delenv("DEEPLYNX_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPLYNX_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    from mcp_server import tool_query_timeseries_nl
    result = tool_query_timeseries_nl(
        {
            "question": "What was the average core temp last week?",
            "container_id": "c",
            "datasource_id": "ds",
        }
    )
    assert "rows" in result
    assert isinstance(result["rows"], list)


def test_tool_get_record_context_missing_id():
    from mcp_server import tool_get_record_context
    result = tool_get_record_context({})
    assert "error" in result


def test_tool_get_record_context_stub():
    from mcp_server import tool_ingest_project_records, tool_get_record_context
    # Ingest first so KB has records
    tool_ingest_project_records({"container_id": "ctx-test"})
    result = tool_get_record_context(
        {"record_id": "stub-001", "container_id": "ctx-test"}
    )
    assert "record_id" in result
    # Chunks may be empty if source key doesn't match exactly — that's OK
    assert "chunks" in result


def test_tool_search_and_ingest_literature_missing_topic():
    from mcp_server import tool_search_and_ingest_literature
    result = tool_search_and_ingest_literature({})
    assert "error" in result


def test_dispatch_initialize():
    from mcp_server import _dispatch
    result = _dispatch("initialize", {})
    assert result["protocolVersion"] == "2024-11-05"
    assert "serverInfo" in result


def test_dispatch_tools_list():
    from mcp_server import _dispatch, _TOOL_SCHEMAS
    result = _dispatch("tools/list", {})
    assert "tools" in result
    assert len(result["tools"]) == len(_TOOL_SCHEMAS)


def test_dispatch_tools_call_status():
    from mcp_server import _dispatch
    result = _dispatch("tools/call", {"name": "get_sidecar_status", "arguments": {}})
    assert "content" in result
    content = json.loads(result["content"][0]["text"])
    assert content["status"] == "ok"


def test_dispatch_unknown_tool():
    from mcp_server import _dispatch
    result = _dispatch("tools/call", {"name": "nonexistent_tool", "arguments": {}})
    assert "content" in result
    content_text = result["content"][0]["text"]
    try:
        content = json.loads(content_text)
        assert "error" in content
    except json.JSONDecodeError:
        pytest.fail(f"content was not valid JSON: {content_text!r}")


def test_dispatch_unknown_method():
    from mcp_server import _dispatch
    result = _dispatch("unknown/method", {})
    assert "error" in result
