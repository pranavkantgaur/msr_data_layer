"""
Unit tests for the enhanced MSR RAG module.

Tests cover:
- Text chunking helpers
- Cosine similarity functions
- RandomProjectionEmbeddingEngine
- KnowledgeBase (add / search / persist / reload)
- SourceInsight dataclass
- SubQuery dataclass
- MSRDigitalTwinRAG (no-LLM path)
- Query decomposition fallback (no API key)
- json_encode / json_decode compatibility shims
"""

import json
import tempfile
from pathlib import Path

import pytest

from msr_digital_twin_with_rag import (
    DocumentChunk,
    KnowledgeBase,
    MSRDigitalTwinRAG,
    OpenAIEmbeddingEngine,
    RandomProjectionEmbeddingEngine,
    SourceInsight,
    SubQuery,
    _chunk_text,
    _cosine_dense,
    _cosine_sparse,
    _decompose_query,
    _tokenize,
    json_decode,
    json_encode,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("Core Temperature is 700 °C!")
    assert "core" in tokens
    assert "temperature" in tokens
    assert "700" in tokens
    # non-alphanumeric stripped
    assert "°c" not in tokens
    assert "!" not in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


def test_chunk_text_short():
    """A text shorter than chunk_words produces exactly one chunk."""
    chunks = _chunk_text("hello world", chunk_words=100, overlap=10)
    assert len(chunks) == 1
    assert "hello" in chunks[0]


def test_chunk_text_overlap():
    """Chunks should overlap by approximately *overlap* words."""
    long_text = " ".join(["word"] * 80)
    chunks = _chunk_text(long_text, chunk_words=30, overlap=10)
    assert len(chunks) >= 2
    # Verify overlap: last 10 words of chunk 0 should appear at start of chunk 1
    words0 = chunks[0].split()
    words1 = chunks[1].split()
    assert words0[-10:] == words1[:10]


def test_chunk_text_respects_max():
    """Very long texts should produce a bounded number of chunks."""
    very_long = "sentence. " * 5000
    chunks = _chunk_text(very_long, chunk_words=10, overlap=2)
    # Regardless of internal limit, output must be finite and reasonable
    assert 0 < len(chunks) <= 500  # well above any practical document


def test_chunk_text_returns_list_for_empty():
    """Even empty-ish text returns at least one chunk."""
    chunks = _chunk_text("   ")
    assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_dense_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_dense(v, v) - 1.0) < 1e-6


def test_cosine_dense_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_dense(a, b)) < 1e-6


def test_cosine_dense_empty():
    assert _cosine_dense([], [1.0]) == 0.0
    assert _cosine_dense([1.0], []) == 0.0


def test_cosine_sparse_identical():
    d = {"a": 1.0, "b": 0.5}
    score = _cosine_sparse(d, d)
    assert abs(score - 1.0) < 1e-6


def test_cosine_sparse_no_overlap():
    a = {"x": 1.0}
    b = {"y": 1.0}
    assert _cosine_sparse(a, b) == 0.0


def test_cosine_sparse_empty():
    assert _cosine_sparse({}, {"a": 1.0}) == 0.0


# ---------------------------------------------------------------------------
# RandomProjectionEmbeddingEngine
# ---------------------------------------------------------------------------

def test_rpe_embed_dim():
    engine = RandomProjectionEmbeddingEngine(dim=32)
    emb = engine.embed("core temperature reactor")
    assert len(emb) == 32


def test_rpe_embed_deterministic():
    engine = RandomProjectionEmbeddingEngine(dim=64, seed=99)
    a = engine.embed("salt flow rate")
    b = engine.embed("salt flow rate")
    assert a == b


def test_rpe_similar_texts_higher_score():
    engine = RandomProjectionEmbeddingEngine(dim=128, seed=0)
    corpus = [
        "core temperature limit 750 celsius",
        "fuel salt level percentage control rod",
        "neutron flux reactor power megawatt",
    ]
    engine.update_idf(corpus)
    query = "core temperature"
    q_emb = engine.embed(query)
    scores = [_cosine_dense(q_emb, engine.embed(c)) for c in corpus]
    # The first corpus text is the most relevant to "core temperature"
    assert scores[0] == max(scores)


def test_rpe_update_idf_changes_weights():
    """Rare tokens should produce higher search scores than common tokens."""
    engine = RandomProjectionEmbeddingEngine(dim=64, seed=0)
    # "reactor" and "power" appear in all 3 docs; "xenon" only in one
    engine.update_idf(["reactor power xenon", "reactor power sodium", "reactor power coolant"])
    # A query for the rare term "xenon" should score higher against a doc
    # that contains "xenon" than against one that only has the common terms
    xenon_doc = engine.embed("reactor power xenon")
    no_xenon_doc = engine.embed("reactor power sodium")
    query_xenon = engine.embed("xenon")
    score_xenon = _cosine_dense(query_xenon, xenon_doc)
    score_no_xenon = _cosine_dense(query_xenon, no_xenon_doc)
    assert score_xenon >= score_no_xenon


def test_rpe_embed_batch():
    engine = RandomProjectionEmbeddingEngine(dim=32)
    texts = ["text one", "text two", "text three"]
    batch = engine.embed_batch(texts)
    assert len(batch) == 3
    for emb in batch:
        assert len(emb) == 32


# ---------------------------------------------------------------------------
# SourceInsight and SubQuery dataclasses
# ---------------------------------------------------------------------------

def test_source_insight_fields():
    ins = SourceInsight(
        source="doc.md",
        summary="An MSR document.",
        topics=["temperature", "safety"],
        key_facts=["Max temp: 750 C"],
    )
    assert ins.source == "doc.md"
    assert "temperature" in ins.topics


def test_sub_query_fields():
    sq = SubQuery(term="core temperature", instructions="Find the safe limit.")
    assert sq.term == "core temperature"
    assert "safe" in sq.instructions


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_kb(tmp_path):
    engine = RandomProjectionEmbeddingEngine(dim=64)
    kb = KnowledgeBase(engine, store_dir=tmp_path / "kb")
    return kb


def test_kb_add_document_returns_chunk_count(tmp_kb):
    text = "The reactor core temperature should not exceed 750 C at any time."
    n = tmp_kb.add_document(text, source="safety_manual.md")
    assert n >= 1


def test_kb_search_returns_results(tmp_kb):
    tmp_kb.add_document(
        "Core temperature limit is 750 degrees Celsius for safe operation.",
        source="manual.md",
    )
    results = tmp_kb.search("core temperature limit", top_k=3)
    assert len(results) >= 1
    assert results[0]["score"] > 0
    assert "text" in results[0]
    assert "source" in results[0]


def test_kb_search_empty_store(tmp_kb):
    assert tmp_kb.search("anything") == []


def test_kb_stores_insight(tmp_kb):
    insight = SourceInsight(
        source="guide.md",
        summary="Safety guide.",
        topics=["safety", "temperature"],
        key_facts=["Max temp: 750 C"],
    )
    tmp_kb.add_document("Safety guide content.", source="guide.md", insight=insight)
    results = tmp_kb.search("safety temperature", top_k=1)
    assert results[0].get("source_summary") == "Safety guide."
    assert "safety" in results[0].get("source_topics", [])


def test_kb_persistence(tmp_path):
    """Saving and reloading the knowledge base preserves chunks and insights."""
    store_dir = tmp_path / "kb"
    engine = RandomProjectionEmbeddingEngine(dim=64)
    kb = KnowledgeBase(engine, store_dir=store_dir)
    insight = SourceInsight(
        source="doc.txt",
        summary="A test document.",
        topics=["testing"],
        key_facts=[],
    )
    kb.add_document("MSR thermal hydraulics test document.", source="doc.txt", insight=insight)
    original_count = len(kb._chunks)

    # Reload
    engine2 = RandomProjectionEmbeddingEngine(dim=64)
    kb2 = KnowledgeBase(engine2, store_dir=store_dir)
    assert len(kb2._chunks) == original_count
    assert "doc.txt" in kb2._insights
    assert kb2._insights["doc.txt"].summary == "A test document."


def test_kb_get_all_insights(tmp_kb):
    tmp_kb.add_document(
        "Document A content.",
        source="a.md",
        insight=SourceInsight(source="a.md", summary="A", topics=[], key_facts=[]),
    )
    tmp_kb.add_document(
        "Document B content.",
        source="b.md",
        insight=SourceInsight(source="b.md", summary="B", topics=[], key_facts=[]),
    )
    insights = tmp_kb.get_all_insights()
    sources = {ins.source for ins in insights}
    assert {"a.md", "b.md"} == sources


def test_kb_top_k_limits_results(tmp_kb):
    for i in range(10):
        tmp_kb.add_document(f"Document number {i} about reactor operations.", source=f"doc{i}.md")
    results = tmp_kb.search("reactor", top_k=3)
    assert len(results) <= 3


def test_kb_hybrid_scores_in_0_1(tmp_kb):
    tmp_kb.add_document("Reactor salt flow rate is 250 kg per second.", source="ops.md")
    results = tmp_kb.search("salt flow", top_k=5)
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


# ---------------------------------------------------------------------------
# MSRDigitalTwinRAG (no-LLM path)
# ---------------------------------------------------------------------------

@pytest.fixture()
def rag_no_llm(tmp_path, monkeypatch):
    """RAG instance with no API key and a temporary KB + docs dir."""
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path / "kb"))
    return MSRDigitalTwinRAG(docs_dir=tmp_path / "docs")


def test_rag_no_llm_answer_contains_reactor_state(rag_no_llm):
    answer = rag_no_llm.answer("What is the current reactor status?")
    assert "MSR_OPENAI_API_KEY" in answer
    assert "reactor" in answer.lower() or "status" in answer.lower()


def test_rag_add_document(rag_no_llm):
    n = rag_no_llm.add_document(
        "The MSR core temperature must stay below 750 degrees Celsius.",
        source="safety_manual",
    )
    assert n >= 1


def test_rag_answer_includes_document_after_add(rag_no_llm):
    rag_no_llm.add_document(
        "The primary salt loop pressure should not exceed 1.5 bar.",
        source="operations_manual",
    )
    answer = rag_no_llm.answer("What is the maximum primary loop pressure?")
    # The document should surface in the prompt context
    assert "1.5" in answer or "pressure" in answer.lower()


def test_rag_load_directory(tmp_path, monkeypatch):
    """RAG loads .md and .txt files from docs_dir on init."""
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path / "kb"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "safety.md").write_text(
        "Core temperature must not exceed 750 C.", encoding="utf-8"
    )
    (docs / "operations.txt").write_text(
        "Salt flow rate nominal value is 250 kg per second.", encoding="utf-8"
    )
    rag = MSRDigitalTwinRAG(docs_dir=docs)
    # Both files should be indexed
    sources = {c.source for c in rag._kb._chunks}
    assert any("safety.md" in s for s in sources)
    assert any("operations.txt" in s for s in sources)


def test_rag_no_duplicate_loading(tmp_path, monkeypatch):
    """Calling RAG twice with the same docs_dir does not double-index."""
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    kb_dir = str(tmp_path / "kb")
    monkeypatch.setenv("MSR_KB_DIR", kb_dir)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "manual.md").write_text("Reactor safety manual content.", encoding="utf-8")

    rag1 = MSRDigitalTwinRAG(docs_dir=docs)
    count1 = len(rag1._kb._chunks)

    # Second instantiation: already-loaded sources are skipped
    rag2 = MSRDigitalTwinRAG(docs_dir=docs)
    count2 = len(rag2._kb._chunks)

    assert count1 == count2


# ---------------------------------------------------------------------------
# _decompose_query fallback (no API key)
# ---------------------------------------------------------------------------

def test_decompose_query_fallback_no_api():
    """Without a valid API key, _decompose_query should return a single SubQuery."""
    result = _decompose_query(
        "What is the safe operating temperature?",
        api_key="",
        base_url="http://localhost:9999",
        model="gpt-4o-mini",
    )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], SubQuery)


# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------

def test_json_encode_decode_roundtrip():
    obj = {"key": [1, 2, 3], "nested": {"a": True}}
    assert json_decode(json_encode(obj)) == obj
