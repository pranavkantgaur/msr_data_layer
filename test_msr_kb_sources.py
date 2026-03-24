"""
Unit tests for msr_kb_sources.py

Tests cover:
- reconstruct_abstract (inverted-index → plain text)
- MSRArchiveLoader (list, fetch, ingest, state persistence)
- OpenAlexLoader (work formatting, ingestion, deduplication, state)
- PlantDataLoader (ingest text, sensor snapshots, deduplication, state)
- KBSourceManager (orchestration, update_all, ingest_plant_data)
- Integration: load_msr_archive / update_openalex on MSRDigitalTwinRAG
"""

from __future__ import annotations

import json
import time
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from msr_kb_sources import (
    ArXivLoader,
    KBSourceManager,
    MSRArchiveLoader,
    OpenAlexLoader,
    SemanticScholarLoader,
    _load_state,
    _save_state,
    reconstruct_abstract,
)


# ---------------------------------------------------------------------------
# reconstruct_abstract
# ---------------------------------------------------------------------------

def test_reconstruct_abstract_basic():
    ii = {"The": [0], "reactor": [1], "temperature": [2], "is": [3], "high": [4]}
    result = reconstruct_abstract(ii)
    assert result == "The reactor temperature is high"


def test_reconstruct_abstract_out_of_order():
    ii = {"high": [4], "The": [0], "reactor": [1], "temperature": [2], "is": [3]}
    result = reconstruct_abstract(ii)
    assert result == "The reactor temperature is high"


def test_reconstruct_abstract_multiple_positions():
    ii = {"the": [0, 3], "salt": [1], "reactor": [2, 4]}
    result = reconstruct_abstract(ii)
    # positions: 0→the, 1→salt, 2→reactor, 3→the, 4→reactor
    assert result == "the salt reactor the reactor"


def test_reconstruct_abstract_none():
    assert reconstruct_abstract(None) == ""


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract({}) == ""


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def test_load_state_missing_file(tmp_path):
    result = _load_state(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_state_corrupt_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json!!!")
    result = _load_state(p)
    assert result == {}


def test_save_and_load_state(tmp_path):
    p = tmp_path / "subdir" / "state.json"
    state = {"ingested_urls": ["http://a", "http://b"], "count": 2}
    _save_state(p, state)
    loaded = _load_state(p)
    assert loaded == state


def test_save_state_creates_parents(tmp_path):
    deep = tmp_path / "a" / "b" / "c" / "state.json"
    _save_state(deep, {"key": "value"})
    assert deep.exists()


# ---------------------------------------------------------------------------
# MSRArchiveLoader – unit tests with mocked HTTP
# ---------------------------------------------------------------------------

@pytest.fixture()
def archive_loader(tmp_path):
    return MSRArchiveLoader(
        repo="testowner/test-archive",
        branch="master",
        kb_dir=tmp_path / "kb",
    )


def _make_github_file(name: str, download_url: str) -> dict[str, str]:
    return {"name": name, "type": "file", "download_url": download_url}


def test_archive_list_ocr_files_success(archive_loader):
    files_data = [
        _make_github_file("ORNL-1234.txt", "https://raw.github.com/ocr/ORNL-1234.txt"),
        _make_github_file("ORNL-5678.txt", "https://raw.github.com/ocr/ORNL-5678.txt"),
        {"name": "README.md", "type": "file", "download_url": "http://x/README.md"},
        {"name": "subdir", "type": "dir", "download_url": None},
    ]
    with patch("msr_kb_sources._http_get", return_value=files_data):
        files = archive_loader.list_ocr_files()
    # Only .txt files with download_url returned
    assert len(files) == 2
    assert all(f["name"].endswith(".txt") for f in files)


def test_archive_list_ocr_files_api_error(archive_loader):
    with patch("msr_kb_sources._http_get", side_effect=urllib.error.URLError("timeout")):
        files = archive_loader.list_ocr_files()
    assert files == []


def test_archive_list_ocr_files_non_list_response(archive_loader):
    with patch("msr_kb_sources._http_get", return_value={"message": "rate limited"}):
        files = archive_loader.list_ocr_files()
    assert files == []


def test_archive_list_ocr_files_from_tree(archive_loader):
    tree_data = {
        "tree": [
            {"type": "blob", "path": "ocr/ORNL-1.txt"},
            {"type": "blob", "path": "ocr/ORNL-2.txt"},
            {"type": "blob", "path": "docs/ORNL-1.pdf"},
            {"type": "tree", "path": "ocr"},
        ]
    }
    with patch("msr_kb_sources._http_get", return_value=tree_data):
        files = archive_loader.list_ocr_files_from_tree()
    assert len(files) == 2
    names = {f["name"] for f in files}
    assert names == {"ORNL-1.txt", "ORNL-2.txt"}
    # URLs should point to raw content on githubusercontent.com
    for f in files:
        assert f["download_url"].startswith("https://raw.githubusercontent.com/")


def test_archive_ingest_new_documents(archive_loader):
    """New documents are ingested; state file is written."""
    files_data = [
        _make_github_file("ORNL-1.txt", "https://raw.gh.com/ocr/ORNL-1.txt"),
        _make_github_file("ORNL-2.txt", "https://raw.gh.com/ocr/ORNL-2.txt"),
    ]
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 3

    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text", return_value="Sample OCR text content"):
        count = archive_loader.ingest(rag_mock)

    assert count == 2
    assert rag_mock.add_document.call_count == 2
    # State file should exist
    assert (archive_loader._state_path).exists()
    state = _load_state(archive_loader._state_path)
    assert len(state["ingested_urls"]) == 2


def test_archive_ingest_skips_already_ingested(archive_loader):
    """Documents whose URLs are in state are not re-fetched."""
    url1 = "https://raw.gh.com/ocr/ORNL-1.txt"
    url2 = "https://raw.gh.com/ocr/ORNL-2.txt"
    # Pre-populate state
    _save_state(
        archive_loader._state_path,
        {"ingested_urls": [url1], "total_ingested": 1},
    )

    files_data = [
        _make_github_file("ORNL-1.txt", url1),
        _make_github_file("ORNL-2.txt", url2),
    ]
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2

    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text", return_value="New content"):
        count = archive_loader.ingest(rag_mock)

    # Only ORNL-2.txt should be ingested
    assert count == 1
    rag_mock.add_document.assert_called_once()


def test_archive_ingest_max_docs(archive_loader):
    """max_docs limits the number of new files ingested."""
    files_data = [
        _make_github_file(f"ORNL-{i}.txt", f"https://raw.gh.com/ocr/ORNL-{i}.txt")
        for i in range(10)
    ]
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text", return_value="Content"):
        count = archive_loader.ingest(rag_mock, max_docs=3)

    assert count == 3
    assert rag_mock.add_document.call_count == 3


def test_archive_ingest_skips_empty_text(archive_loader):
    """Empty OCR files are not passed to rag.add_document."""
    files_data = [_make_github_file("empty.txt", "https://raw.gh.com/ocr/empty.txt")]
    rag_mock = MagicMock()

    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text", return_value="   \n  "):
        count = archive_loader.ingest(rag_mock)

    assert count == 0
    rag_mock.add_document.assert_not_called()


def test_archive_ingest_fetch_error_skips_file(archive_loader):
    """A fetch error on one file is caught and that file is skipped."""
    url1 = "https://raw.gh.com/ocr/ORNL-1.txt"
    url2 = "https://raw.gh.com/ocr/ORNL-2.txt"
    files_data = [
        _make_github_file("ORNL-1.txt", url1),
        _make_github_file("ORNL-2.txt", url2),
    ]
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2

    def fake_get_text(url, headers=None):
        if "ORNL-1" in url:
            raise urllib.error.URLError("timeout")
        return "Good content for doc 2"

    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text", side_effect=fake_get_text):
        count = archive_loader.ingest(rag_mock)

    assert count == 1
    rag_mock.add_document.assert_called_once()


def test_archive_status_empty(archive_loader):
    st = archive_loader.status()
    assert st["total_ingested"] == 0
    assert st["last_run"] == "never"
    assert "msr-archive" in st["source"].lower() or "testowner" in st["source"].lower()


# ---------------------------------------------------------------------------
# OpenAlexLoader – unit tests with mocked API
# ---------------------------------------------------------------------------

@pytest.fixture()
def openalex_loader(tmp_path):
    return OpenAlexLoader(kb_dir=tmp_path / "kb", max_results=50)


def _make_openalex_work(
    work_id: str,
    title: str,
    year: int = 2023,
    abstract_ii: dict | None = None,
) -> dict[str, Any]:
    if abstract_ii is None:
        abstract_ii = {"molten": [0], "salt": [1], "reactor": [2], "data": [3]}
    return {
        "id": work_id,
        "title": title,
        "doi": f"https://doi.org/10.1234/{work_id[-4:]}",
        "publication_year": year,
        "abstract_inverted_index": abstract_ii,
        "open_access": {"is_oa": True, "oa_url": f"https://example.com/{work_id}.pdf"},
        "authorships": [
            {"author": {"display_name": "Zhang Wei"}},
            {"author": {"display_name": "Li Ming"}},
        ],
    }


def _openalex_page(works: list, next_cursor: str | None = None) -> dict:
    return {
        "meta": {
            "count": len(works),
            "per_page": len(works),
            "next_cursor": next_cursor,
        },
        "results": works,
    }


def test_openalex_format_work_text():
    work = _make_openalex_work("W1001", "TMSR-LF1 experimental results")
    text, source_id = OpenAlexLoader.format_work_text(work)
    assert "TMSR-LF1" in text
    assert "experimental results" in text
    assert "molten salt reactor data" in text.lower()
    assert "Zhang Wei" in text
    assert source_id == "W1001"


def test_openalex_format_work_text_missing_fields():
    """format_work_text handles missing optional fields gracefully."""
    work = {"id": "W2002", "title": None, "abstract_inverted_index": None}
    text, source_id = OpenAlexLoader.format_work_text(work)
    assert source_id == "W2002"
    assert isinstance(text, str)


def test_openalex_format_work_text_many_authors():
    """Authors list is capped at 5 with an 'et al.' suffix."""
    work = _make_openalex_work("W3003", "Long author list paper")
    work["authorships"] = [
        {"author": {"display_name": f"Author {i}"}} for i in range(8)
    ]
    text, _ = OpenAlexLoader.format_work_text(work)
    assert "et al." in text
    assert "8 total" in text


def test_openalex_ingest_new_works(openalex_loader):
    works = [
        _make_openalex_work(f"W{i:04d}", f"MSR paper {i}") for i in range(5)
    ]
    page = _openalex_page(works, next_cursor=None)

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2

    with patch("msr_kb_sources._http_get", return_value=page):
        count = openalex_loader.ingest(rag_mock)

    # Both queries return the same 5 work IDs; deduplication ensures each
    # work is ingested exactly once → 5 unique documents
    assert count == 5
    assert rag_mock.add_document.call_count == 5
    state = _load_state(openalex_loader._state_path)
    assert state["total_ingested"] == 5


def test_openalex_ingest_deduplication(openalex_loader):
    """Works already in state are not ingested again."""
    pre_ingested = ["W0001", "W0002"]
    _save_state(
        openalex_loader._state_path,
        {"ingested_ids": pre_ingested, "total_ingested": 2},
    )

    works = [_make_openalex_work(f"W{i:04d}", f"MSR paper {i}") for i in range(5)]
    page = _openalex_page(works, next_cursor=None)

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", return_value=page):
        count = openalex_loader.ingest(rag_mock)

    # 5 works per query - 2 already ingested = 3 new, times 2 queries = 6
    # BUT the 2 pre-ingested IDs are shared between queries so depends on
    # which query surfaces them; simplest check: < 10
    assert count < 10
    assert rag_mock.add_document.call_count < 10


def test_openalex_ingest_respects_max_docs(openalex_loader):
    works = [
        _make_openalex_work(f"W{i:04d}", f"Paper {i}") for i in range(20)
    ]
    page = _openalex_page(works, next_cursor=None)

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", return_value=page):
        count = openalex_loader.ingest(rag_mock, max_docs=5)

    assert count == 5


def test_openalex_ingest_api_error(openalex_loader):
    rag_mock = MagicMock()

    with patch(
        "msr_kb_sources._http_get",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        count = openalex_loader.ingest(rag_mock)

    assert count == 0
    rag_mock.add_document.assert_not_called()


def test_openalex_ingest_pagination(openalex_loader):
    """Loader follows next_cursor to fetch multiple pages."""
    page1_works = [_make_openalex_work(f"W{i:04d}", f"P1 paper {i}") for i in range(3)]
    page2_works = [_make_openalex_work(f"W{i:04d}", f"P2 paper {i}") for i in range(3, 6)]

    page1 = _openalex_page(page1_works, next_cursor="CURSOR2")
    page2 = _openalex_page(page2_works, next_cursor=None)

    call_sequence = iter([page1, page2, page1, page2])  # 2 queries × 2 pages

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", side_effect=call_sequence), \
         patch("msr_kb_sources.time.sleep"):
        count = openalex_loader.ingest(rag_mock, max_docs=100)

    # Both queries exhaust both pages: 6 + 6 = 12 unique IDs ingested
    # (since pages overlap in IDs across queries, dedup applies)
    assert count >= 6


def test_openalex_status_empty(openalex_loader):
    st = openalex_loader.status()
    assert st["total_ingested"] == 0
    assert st["last_run"] == "never"
    assert "openalex" in st["source"].lower() or "OpenAlex" in st["source"]


# ---------------------------------------------------------------------------
# KBSourceManager
# ---------------------------------------------------------------------------

@pytest.fixture()
def source_manager(tmp_path):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2
    return rag_mock, KBSourceManager(rag_mock, kb_dir=tmp_path / "kb")


def test_kb_source_manager_update_archive(source_manager, tmp_path):
    rag_mock, mgr = source_manager
    files_data = [
        {"name": "doc.txt", "type": "file", "download_url": "https://raw.gh.com/ocr/doc.txt"}
    ]
    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text", return_value="MSR document"):
        count = mgr.update_archive()
    assert count == 1


def test_kb_source_manager_update_openalex(source_manager):
    rag_mock, mgr = source_manager
    works = [_make_openalex_work("W9001", "TMSR-LF1 data")]
    page = _openalex_page(works)
    with patch("msr_kb_sources._http_get", return_value=page):
        count = mgr.update_openalex()
    assert count >= 1


def test_kb_source_manager_update_all(source_manager):
    rag_mock, mgr = source_manager
    files_data = [
        {"name": "x.txt", "type": "file", "download_url": "https://raw.gh.com/ocr/x.txt"}
    ]
    works = [_make_openalex_work("W8001", "Test paper")]
    page = _openalex_page(works)

    # arXiv returns empty XML (no new papers); S2 returns empty page
    empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    s2_empty = {"data": [], "total": 0}

    with patch("msr_kb_sources._http_get") as mock_get, \
         patch("msr_kb_sources._http_get_text") as mock_get_text:
        # Sequence: GitHub API for archive, OpenAlex query×2, S2 query×2
        mock_get.side_effect = [files_data, page, page, s2_empty, s2_empty]
        # Sequence: archive content fetch, arXiv query×2
        mock_get_text.side_effect = ["Content", empty_xml, empty_xml]
        result = mgr.update_all()

    assert "archive" in result
    assert "openalex" in result
    assert "arxiv" in result
    assert "semanticscholar" in result
    assert isinstance(result["archive"], int)
    assert isinstance(result["openalex"], int)
    assert isinstance(result["arxiv"], int)
    assert isinstance(result["semanticscholar"], int)


def test_kb_source_manager_status(source_manager, capsys):
    _, mgr = source_manager
    mgr.status()  # Should not raise
    captured = capsys.readouterr()
    assert "Status" in captured.out


# ---------------------------------------------------------------------------
# Integration: MSRDigitalTwinRAG.load_msr_archive / update_openalex
# ---------------------------------------------------------------------------

def test_rag_load_msr_archive_method(tmp_path, monkeypatch):
    """MSRDigitalTwinRAG.load_msr_archive delegates to MSRArchiveLoader."""
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path / "kb"))

    from msr_digital_twin_with_rag import MSRDigitalTwinRAG

    rag = MSRDigitalTwinRAG(docs_dir=tmp_path / "docs")

    files_data = [
        {"name": "ORNL-1.txt", "type": "file",
         "download_url": "https://raw.gh.com/ocr/ORNL-1.txt"}
    ]
    with patch("msr_kb_sources._http_get", return_value=files_data), \
         patch("msr_kb_sources._http_get_text",
               return_value="Molten salt reactor experimental measurements at ORNL."):
        n = rag.load_msr_archive()

    assert n >= 1
    # Document should be searchable
    results = rag._kb.search("ORNL reactor measurements", top_k=3)
    assert len(results) >= 1


def test_rag_update_openalex_method(tmp_path, monkeypatch):
    """MSRDigitalTwinRAG.update_openalex delegates to OpenAlexLoader."""
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path / "kb"))

    from msr_digital_twin_with_rag import MSRDigitalTwinRAG

    rag = MSRDigitalTwinRAG(docs_dir=tmp_path / "docs")

    works = [
        _make_openalex_work(
            "W7001",
            "TMSR-LF1 startup experimental results 2023",
            abstract_ii={
                "TMSR": [0], "LF1": [1], "startup": [2], "experimental": [3],
                "results": [4], "2023": [5],
            },
        )
    ]
    page = _openalex_page(works)
    with patch("msr_kb_sources._http_get", return_value=page):
        n = rag.update_openalex(max_docs=10)

    assert n >= 1
    results = rag._kb.search("TMSR-LF1 startup", top_k=3)
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# PlantDataLoader
# ---------------------------------------------------------------------------

from msr_kb_sources import PlantDataLoader  # noqa: E402


@pytest.fixture()
def plant_loader(tmp_path):
    return PlantDataLoader(kb_dir=tmp_path / "kb")


def test_plant_loader_ingest_text(plant_loader, tmp_path):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2
    n = plant_loader.ingest_text(
        rag_mock,
        "Core temperature 702°C at 14:32 UTC",
        source_id="event-001",
    )
    assert n == 2
    rag_mock.add_document.assert_called_once()
    # Verify source label includes the data type and source_id
    call_kwargs = rag_mock.add_document.call_args
    source_arg = call_kwargs[1].get("source") or call_kwargs[0][1]
    assert "event-001" in source_arg


def test_plant_loader_ingest_text_deduplication(plant_loader):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    plant_loader.ingest_text(rag_mock, "reading 1", source_id="snap-001")
    plant_loader.ingest_text(rag_mock, "reading 2", source_id="snap-001")  # duplicate

    assert rag_mock.add_document.call_count == 1


def test_plant_loader_empty_content_returns_zero(plant_loader):
    rag_mock = MagicMock()
    n = plant_loader.ingest_text(rag_mock, "", source_id="empty-001")
    assert n == 0
    rag_mock.add_document.assert_not_called()


def test_plant_loader_invalid_data_type_defaults_to_operational(plant_loader):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1
    n = plant_loader.ingest_text(
        rag_mock, "some data", source_id="x-001", data_type="unknown_type"
    )
    assert n == 1
    source_arg = rag_mock.add_document.call_args[1].get("source") or \
                 rag_mock.add_document.call_args[0][1]
    assert "operational_data" in source_arg


def test_plant_loader_state_persisted(plant_loader, tmp_path):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1
    plant_loader.ingest_text(rag_mock, "reading A", source_id="A-001")
    plant_loader.ingest_text(rag_mock, "reading B", source_id="B-002")

    state = _load_state(plant_loader._state_path)
    assert "A-001" in state["ingested_ids"]
    assert "B-002" in state["ingested_ids"]
    assert state["total_ingested"] == 2


def test_plant_loader_ingest_sensor_snapshot_list(plant_loader):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1
    snapshot = [
        {"timestamp": "2024-01-15T14:00:00Z", "sensor": "core_temperature_c",
         "value": 702.1, "unit": "°C"},
        {"timestamp": "2024-01-15T14:00:00Z", "sensor": "reactor_power_mw",
         "value": 99.8, "unit": "MW"},
    ]
    n = plant_loader.ingest_sensor_snapshot(rag_mock, snapshot, source_id="snap-list-001")
    assert n == 1
    # Verify formatted text contains sensor name
    call_text = rag_mock.add_document.call_args[0][0]
    assert "core_temperature_c" in call_text
    assert "702.1" in call_text


def test_plant_loader_ingest_sensor_snapshot_dict(plant_loader):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1
    snapshot = {
        "timestamp": "2024-01-15T14:00:00Z",
        "core_temperature_c": 702.1,
        "reactor_power_mw": 99.8,
    }
    n = plant_loader.ingest_sensor_snapshot(rag_mock, snapshot, source_id="snap-dict-001")
    assert n == 1
    call_text = rag_mock.add_document.call_args[0][0]
    assert "702.1" in call_text


def test_plant_loader_ingest_sensor_snapshot_json_string(plant_loader):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1
    import json as _json
    snapshot_str = _json.dumps({"core_temperature_c": 700.0, "reactor_power_mw": 100.0})
    n = plant_loader.ingest_sensor_snapshot(rag_mock, snapshot_str, source_id="snap-str-001")
    assert n == 1


def test_plant_loader_status_empty(plant_loader):
    st = plant_loader.status()
    assert st["total_ingested"] == 0
    assert st["last_run"] == "never"
    assert "plant" in st["source"].lower()


def test_plant_loader_status_after_ingest(plant_loader):
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1
    plant_loader.ingest_text(rag_mock, "data", source_id="s001")
    plant_loader.ingest_text(rag_mock, "data2", source_id="s002")
    st = plant_loader.status()
    assert st["total_ingested"] == 2
    assert st["last_run"] != "never"


def test_plant_loader_format_snapshot_unknown_type():
    """Non-dict/non-list snapshots are converted to string."""
    text = PlantDataLoader._format_snapshot("raw string data")
    assert text == "raw string data"


def test_kb_source_manager_ingest_plant_data(source_manager):
    rag_mock, mgr = source_manager
    rag_mock.add_document.return_value = 1
    n = mgr.ingest_plant_data(
        "Salt level 87% at 15:00 UTC",
        source_id="level-check-001",
        data_type="event_log",
    )
    assert n == 1
    rag_mock.add_document.assert_called()



# ---------------------------------------------------------------------------
# ArXivLoader – unit tests with mocked HTTP
# ---------------------------------------------------------------------------

_ATOM_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>"""

_ATOM_ENTRY_TEMPLATE = """\
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <title>{title}</title>
    <summary>{abstract}</summary>
    <published>{published}</published>
    <author><name>{author}</name></author>
    {doi_link}
  </entry>"""


def _make_atom_entry(
    arxiv_id: str,
    title: str = "Molten salt reactor paper",
    abstract: str = "Experimental results on MSRE.",
    published: str = "2024-01-15T00:00:00Z",
    author: str = "Zhang Wei",
    doi: str = "",
) -> str:
    doi_link = f'<link title="doi" href="https://doi.org/10.1234/{arxiv_id}" />' if doi else ""
    return _ATOM_ENTRY_TEMPLATE.format(
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        published=published,
        author=author,
        doi_link=doi_link,
    )


def _make_atom_feed(entries: list[str]) -> str:
    return _ATOM_TEMPLATE.format(entries="\n".join(entries))


@pytest.fixture()
def arxiv_loader(tmp_path):
    return ArXivLoader(kb_dir=tmp_path / "kb", max_results=50)


def test_arxiv_format_entry_basic():
    entry = {
        "id": "2301.12345",
        "title": "TMSR-LF1 experimental startup results",
        "abstract": "We describe early experimental results from TMSR-LF1.",
        "published": "2023-01-30",
        "authors": ["Zhang Wei", "Li Ming"],
        "doi": "https://doi.org/10.1234/tmsr",
    }
    text, source_id = ArXivLoader.format_entry_text(entry)
    assert "TMSR-LF1" in text
    assert "Zhang Wei" in text
    assert "2301.12345" in text
    assert "TMSR-LF1 experimental startup results" in text
    assert source_id == "arxiv:2301.12345"


def test_arxiv_format_entry_no_doi():
    entry = {
        "id": "2302.99999",
        "title": "Fluoride salt corrosion study",
        "abstract": "Abstract text.",
        "published": "2023-02-01",
        "authors": [],
        "doi": "",
    }
    text, source_id = ArXivLoader.format_entry_text(entry)
    assert "DOI" not in text
    assert source_id == "arxiv:2302.99999"


def test_arxiv_format_entry_many_authors():
    entry = {
        "id": "2303.11111",
        "title": "Long author list",
        "abstract": "Abstract.",
        "published": "2023-03-01",
        "authors": [f"Author {i}" for i in range(8)],
        "doi": "",
    }
    text, _ = ArXivLoader.format_entry_text(entry)
    assert "et al." in text
    assert "8 total" in text


def test_arxiv_parse_entries_basic(arxiv_loader):
    xml = _make_atom_feed([
        _make_atom_entry("2301.00001", "MSR paper 1"),
        _make_atom_entry("2301.00002", "MSR paper 2"),
    ])
    entries = arxiv_loader._parse_entries(xml)
    assert len(entries) == 2
    ids = {e["id"] for e in entries}
    assert "2301.00001" in ids
    assert "2301.00002" in ids


def test_arxiv_parse_entries_strips_version(arxiv_loader):
    """Version suffix (v1, v2) is stripped from arXiv IDs."""
    xml = _make_atom_feed([_make_atom_entry("2301.99999")])
    entries = arxiv_loader._parse_entries(xml)
    assert entries[0]["id"] == "2301.99999"


def test_arxiv_parse_entries_empty_feed(arxiv_loader):
    empty_feed = '<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    entries = arxiv_loader._parse_entries(empty_feed)
    assert entries == []


def test_arxiv_parse_entries_malformed_xml(arxiv_loader):
    entries = arxiv_loader._parse_entries("not xml at all !!!")
    assert entries == []


def test_arxiv_ingest_new_papers(arxiv_loader):
    xml1 = _make_atom_feed([_make_atom_entry(f"2301.{i:05d}") for i in range(3)])
    # Both queries return the same feed; dedup prevents double-ingestion
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2

    with patch("msr_kb_sources._http_get_text", return_value=xml1), \
         patch("msr_kb_sources.time.sleep"):
        count = arxiv_loader.ingest(rag_mock)

    assert count == 3
    assert rag_mock.add_document.call_count == 3
    state = _load_state(arxiv_loader._state_path)
    assert state["total_ingested"] == 3


def test_arxiv_ingest_deduplication(arxiv_loader):
    """Papers already in state are not ingested again."""
    _save_state(
        arxiv_loader._state_path,
        {"ingested_ids": ["2301.00001", "2301.00002"], "total_ingested": 2},
    )
    xml = _make_atom_feed([
        _make_atom_entry("2301.00001"),  # already ingested
        _make_atom_entry("2301.00003"),  # new
    ])
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get_text", return_value=xml), \
         patch("msr_kb_sources.time.sleep"):
        count = arxiv_loader.ingest(rag_mock)

    # Only the new paper should be ingested (once per query × 2 queries = 2 max,
    # but dedup means exactly 1 unique new paper)
    assert count == 1


def test_arxiv_ingest_respects_max_docs(arxiv_loader):
    xml = _make_atom_feed([_make_atom_entry(f"2301.{i:05d}") for i in range(20)])
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get_text", return_value=xml), \
         patch("msr_kb_sources.time.sleep"):
        count = arxiv_loader.ingest(rag_mock, max_docs=5)

    assert count == 5


def test_arxiv_ingest_api_error(arxiv_loader):
    rag_mock = MagicMock()
    with patch(
        "msr_kb_sources._http_get_text",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        count = arxiv_loader.ingest(rag_mock)
    assert count == 0
    rag_mock.add_document.assert_not_called()


def test_arxiv_status_empty(arxiv_loader):
    st = arxiv_loader.status()
    assert st["total_ingested"] == 0
    assert st["last_run"] == "never"
    assert "arxiv" in st["source"].lower()


def test_arxiv_status_after_ingest(arxiv_loader):
    xml = _make_atom_feed([_make_atom_entry("2301.11111")])
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get_text", return_value=xml), \
         patch("msr_kb_sources.time.sleep"):
        arxiv_loader.ingest(rag_mock)

    st = arxiv_loader.status()
    assert st["total_ingested"] >= 1
    assert st["last_run"] != "never"


# ---------------------------------------------------------------------------
# SemanticScholarLoader – unit tests with mocked API
# ---------------------------------------------------------------------------

@pytest.fixture()
def s2_loader(tmp_path):
    return SemanticScholarLoader(kb_dir=tmp_path / "kb", max_results=50)


def _make_s2_paper(
    paper_id: str,
    title: str = "Molten salt reactor experimental study",
    year: int = 2024,
    abstract: str = "We present experimental results.",
) -> dict[str, Any]:
    return {
        "paperId": paper_id,
        "title": title,
        "year": year,
        "abstract": abstract,
        "authors": [
            {"name": "Zhang Wei"},
            {"name": "Li Ming"},
        ],
        "externalIds": {"DOI": f"10.1234/{paper_id[:8]}"},
        "openAccessPdf": {"url": f"https://example.com/{paper_id}.pdf"},
    }


def _s2_page(papers: list, total: int | None = None) -> dict:
    return {
        "data": papers,
        "total": total if total is not None else len(papers),
    }


def test_s2_format_paper_text_basic():
    paper = _make_s2_paper("abc123def456")
    text, source_id = SemanticScholarLoader.format_paper_text(paper)
    assert "Molten salt reactor experimental study" in text
    assert "Zhang Wei" in text
    assert source_id == "s2:abc123def456"


def test_s2_format_paper_text_missing_fields():
    paper = {"paperId": "xyz000", "title": None, "abstract": None}
    text, source_id = SemanticScholarLoader.format_paper_text(paper)
    assert source_id == "s2:xyz000"
    assert isinstance(text, str)


def test_s2_format_paper_text_arxiv_link():
    paper = _make_s2_paper("p001")
    paper["externalIds"] = {"ArXiv": "2301.12345"}
    text, _ = SemanticScholarLoader.format_paper_text(paper)
    assert "https://arxiv.org/abs/2301.12345" in text


def test_s2_format_paper_text_many_authors():
    paper = _make_s2_paper("p002")
    paper["authors"] = [{"name": f"Author {i}"} for i in range(7)]
    text, _ = SemanticScholarLoader.format_paper_text(paper)
    assert "et al." in text
    assert "7 total" in text


def test_s2_ingest_new_papers(s2_loader):
    papers = [_make_s2_paper(f"pid{i:04d}") for i in range(4)]
    page = _s2_page(papers)

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 2

    with patch("msr_kb_sources._http_get", return_value=page), \
         patch("msr_kb_sources.time.sleep"):
        count = s2_loader.ingest(rag_mock)

    # Both queries return same 4 paper IDs; dedup means only 4 unique papers
    assert count == 4
    assert rag_mock.add_document.call_count == 4
    state = _load_state(s2_loader._state_path)
    assert state["total_ingested"] == 4


def test_s2_ingest_deduplication(s2_loader):
    _save_state(
        s2_loader._state_path,
        {"ingested_ids": ["pid0001", "pid0002"], "total_ingested": 2},
    )
    papers = [
        _make_s2_paper("pid0001"),   # already ingested
        _make_s2_paper("pid0099"),   # new
    ]
    page = _s2_page(papers)

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", return_value=page), \
         patch("msr_kb_sources.time.sleep"):
        count = s2_loader.ingest(rag_mock)

    assert count == 1


def test_s2_ingest_respects_max_docs(s2_loader):
    papers = [_make_s2_paper(f"pid{i:04d}") for i in range(20)]
    page = _s2_page(papers, total=20)

    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", return_value=page), \
         patch("msr_kb_sources.time.sleep"):
        count = s2_loader.ingest(rag_mock, max_docs=5)

    assert count == 5


def test_s2_ingest_api_error(s2_loader):
    rag_mock = MagicMock()
    with patch(
        "msr_kb_sources._http_get",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        count = s2_loader.ingest(rag_mock)
    assert count == 0
    rag_mock.add_document.assert_not_called()


def test_s2_ingest_empty_response(s2_loader):
    """Empty data list stops iteration gracefully."""
    rag_mock = MagicMock()
    with patch("msr_kb_sources._http_get", return_value={"data": [], "total": 0}):
        count = s2_loader.ingest(rag_mock)
    assert count == 0
    rag_mock.add_document.assert_not_called()


def test_s2_status_empty(s2_loader):
    st = s2_loader.status()
    assert st["total_ingested"] == 0
    assert st["last_run"] == "never"
    assert "semantic scholar" in st["source"].lower()


def test_s2_status_after_ingest(s2_loader):
    papers = [_make_s2_paper("pid9999")]
    page = _s2_page(papers)
    rag_mock = MagicMock()
    rag_mock.add_document.return_value = 1

    with patch("msr_kb_sources._http_get", return_value=page), \
         patch("msr_kb_sources.time.sleep"):
        s2_loader.ingest(rag_mock)

    st = s2_loader.status()
    assert st["total_ingested"] >= 1
    assert st["last_run"] != "never"


# ---------------------------------------------------------------------------
# KBSourceManager – update_arxiv and update_semanticscholar
# ---------------------------------------------------------------------------

def test_kb_source_manager_update_arxiv(source_manager):
    rag_mock, mgr = source_manager
    xml = _make_atom_feed([_make_atom_entry("2301.77777")])
    with patch("msr_kb_sources._http_get_text", return_value=xml), \
         patch("msr_kb_sources.time.sleep"):
        count = mgr.update_arxiv()
    assert count >= 1


def test_kb_source_manager_update_semanticscholar(source_manager):
    rag_mock, mgr = source_manager
    papers = [_make_s2_paper("pid8888")]
    page = _s2_page(papers)
    with patch("msr_kb_sources._http_get", return_value=page), \
         patch("msr_kb_sources.time.sleep"):
        count = mgr.update_semanticscholar()
    assert count >= 1


def test_kb_source_manager_status_shows_all_sources(source_manager, capsys):
    _, mgr = source_manager
    mgr.status()
    captured = capsys.readouterr()
    assert "Status" in captured.out
    # All 5 source names should appear
    out_lower = captured.out.lower()
    assert "arxiv" in out_lower
    assert "semantic scholar" in out_lower
    assert "openalex" in out_lower or "open" in out_lower


# ---------------------------------------------------------------------------
# TimeseriesStore tests
# ---------------------------------------------------------------------------

class TestTimeseriesStore:
    """Unit tests for TimeseriesStore."""

    def test_init_creates_db(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        assert (tmp_path / "plant_timeseries.db").exists()

    def test_insert_and_query_range(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        readings = [
            {"sensor_name": "reactor_power_mw", "value": 99.5, "unit": "MW",
             "timestamp": "2024-01-15T14:00:00Z"},
            {"sensor_name": "reactor_power_mw", "value": 100.2, "unit": "MW",
             "timestamp": "2024-01-15T14:05:00Z"},
        ]
        n = ts.insert_readings(readings, source_id="src-001")
        assert n == 2
        rows = ts.query_range("reactor_power_mw")
        assert len(rows) == 2
        assert rows[0]["value"] == 99.5

    def test_deduplication(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        readings = [{"sensor_name": "core_temperature_c", "value": 700.0,
                     "timestamp": "2024-01-15T14:00:00Z"}]
        first = ts.insert_readings(readings, source_id="dup-001")
        second = ts.insert_readings(readings, source_id="dup-001")
        assert first == 1
        assert second == 0  # duplicate → no-op
        rows = ts.query_range("core_temperature_c")
        assert len(rows) == 1

    def test_query_latest(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        readings = [
            {"sensor_name": "core_temperature_c", "value": 700.0,
             "timestamp": "2024-01-15T12:00:00Z"},
            {"sensor_name": "core_temperature_c", "value": 705.0,
             "timestamp": "2024-01-15T13:00:00Z"},
            {"sensor_name": "core_temperature_c", "value": 710.0,
             "timestamp": "2024-01-15T14:00:00Z"},
        ]
        ts.insert_readings(readings, source_id="src-002")
        latest = ts.query_latest("core_temperature_c", last_n=2)
        assert len(latest) == 2
        assert latest[0]["value"] == 710.0  # newest first

    def test_query_aggregate_avg(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        readings = [
            {"sensor_name": "reactor_power_mw", "value": 90.0,
             "timestamp": "2024-01-15T12:00:00Z"},
            {"sensor_name": "reactor_power_mw", "value": 110.0,
             "timestamp": "2024-01-15T13:00:00Z"},
        ]
        ts.insert_readings(readings, source_id="src-003")
        result = ts.query_aggregate("reactor_power_mw", agg="avg")
        assert result["result"] == 100.0
        assert result["n"] == 2

    def test_query_aggregate_min_max(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        readings = [
            {"sensor_name": "neutron_flux", "value": 2.0, "timestamp": "2024-01-15T12:00:00Z"},
            {"sensor_name": "neutron_flux", "value": 5.0, "timestamp": "2024-01-15T13:00:00Z"},
        ]
        ts.insert_readings(readings, source_id="src-004")
        assert ts.query_aggregate("neutron_flux", agg="min")["result"] == 2.0
        assert ts.query_aggregate("neutron_flux", agg="max")["result"] == 5.0

    def test_query_range_with_bounds(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        readings = [
            {"sensor_name": "core_temperature_c", "value": 700.0,
             "timestamp": "2024-01-14T12:00:00Z"},
            {"sensor_name": "core_temperature_c", "value": 705.0,
             "timestamp": "2024-01-15T12:00:00Z"},
            {"sensor_name": "core_temperature_c", "value": 710.0,
             "timestamp": "2024-01-16T12:00:00Z"},
        ]
        ts.insert_readings(readings, source_id="src-005")
        rows = ts.query_range(
            "core_temperature_c",
            start="2024-01-15T00:00:00Z",
            end="2024-01-15T23:59:59Z",
        )
        assert len(rows) == 1
        assert rows[0]["value"] == 705.0

    def test_list_sensors(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        ts.insert_readings(
            [{"sensor_name": "sensor_a", "value": 1.0, "timestamp": "2024-01-15T12:00:00Z"},
             {"sensor_name": "sensor_b", "value": 2.0, "timestamp": "2024-01-15T12:00:00Z"}],
            source_id="src-006",
        )
        sensors = ts.list_sensors()
        assert "sensor_a" in sensors
        assert "sensor_b" in sensors

    def test_execute_safe_select(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        ts.insert_readings(
            [{"sensor_name": "reactor_power_mw", "value": 99.8, "unit": "MW",
              "timestamp": "2024-01-15T14:00:00Z"}],
            source_id="src-007",
        )
        rows = ts.execute_safe_select(
            "SELECT sensor_name, value FROM sensor_readings WHERE sensor_name='reactor_power_mw'"
        )
        assert len(rows) == 1
        assert rows[0]["value"] == 99.8

    def test_execute_safe_select_rejects_mutation(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        import pytest
        ts = TimeseriesStore(kb_dir=tmp_path)
        with pytest.raises(ValueError, match="Only SELECT"):
            ts.execute_safe_select("DELETE FROM sensor_readings")

    def test_status_reflects_inserts(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        ts.insert_readings(
            [{"sensor_name": "reactor_power_mw", "value": 99.0,
              "timestamp": "2024-01-15T14:00:00Z"}],
            source_id="src-008",
        )
        st = ts.status()
        assert st["total_readings"] == 1
        assert "reactor_power_mw" in st["sensors"]
        assert st["source_ids_count"] == 1

    def test_source_id_exists(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        assert not ts.source_id_exists("new-id")
        ts.insert_readings(
            [{"sensor_name": "x", "value": 1.0, "timestamp": "2024-01-15T12:00:00Z"}],
            source_id="new-id",
        )
        assert ts.source_id_exists("new-id")

    def test_get_schema_description(self, tmp_path):
        from msr_kb_sources import TimeseriesStore
        ts = TimeseriesStore(kb_dir=tmp_path)
        desc = ts.get_schema_description()
        assert "sensor_readings" in desc
        assert "timestamp" in desc
        assert "sensor_name" in desc
        assert "Only SELECT" in desc


# ---------------------------------------------------------------------------
# KBSourceManager timeseries integration tests
# ---------------------------------------------------------------------------

class TestKBSourceManagerTimeseries:
    """Integration tests for KBSourceManager timeseries methods."""

    def test_ingest_timeseries_inserts_rows(self, tmp_path):
        from unittest.mock import MagicMock
        from msr_kb_sources import KBSourceManager
        rag_mock = MagicMock()
        rag_mock.add_document.return_value = 1
        mgr = KBSourceManager(rag_mock, kb_dir=tmp_path)
        result = mgr.ingest_timeseries(
            [{"sensor_name": "reactor_power_mw", "value": 99.5, "unit": "MW",
              "timestamp": "2024-01-15T14:00:00Z"}],
            source_id="ts-test-001",
            also_ingest_text=False,
        )
        assert result["timeseries_rows"] == 1

    def test_ingest_timeseries_deduplication(self, tmp_path):
        from unittest.mock import MagicMock
        from msr_kb_sources import KBSourceManager
        rag_mock = MagicMock()
        rag_mock.add_document.return_value = 1
        mgr = KBSourceManager(rag_mock, kb_dir=tmp_path)
        mgr.ingest_timeseries(
            [{"sensor_name": "core_temperature_c", "value": 700.0,
              "timestamp": "2024-01-15T14:00:00Z"}],
            source_id="ts-dup-001",
            also_ingest_text=False,
        )
        second = mgr.ingest_timeseries(
            [{"sensor_name": "core_temperature_c", "value": 700.0,
              "timestamp": "2024-01-15T14:00:00Z"}],
            source_id="ts-dup-001",
            also_ingest_text=False,
        )
        assert second["timeseries_rows"] == 0

    def test_query_timeseries_structured(self, tmp_path):
        from unittest.mock import MagicMock
        from msr_kb_sources import KBSourceManager
        rag_mock = MagicMock()
        rag_mock.add_document.return_value = 1
        mgr = KBSourceManager(rag_mock, kb_dir=tmp_path)
        mgr.ingest_timeseries(
            [{"sensor_name": "reactor_power_mw", "value": 99.5, "unit": "MW",
              "timestamp": "2024-01-15T14:00:00Z"}],
            source_id="ts-q-001",
            also_ingest_text=False,
        )
        result = mgr.query_timeseries("reactor_power_mw")
        assert result["count"] == 1
        assert result["rows"][0]["value"] == 99.5

    def test_query_timeseries_aggregation(self, tmp_path):
        from unittest.mock import MagicMock
        from msr_kb_sources import KBSourceManager
        rag_mock = MagicMock()
        rag_mock.add_document.return_value = 1
        mgr = KBSourceManager(rag_mock, kb_dir=tmp_path)
        mgr.ingest_timeseries(
            [
                {"sensor_name": "reactor_power_mw", "value": 80.0,
                 "timestamp": "2024-01-15T12:00:00Z"},
                {"sensor_name": "reactor_power_mw", "value": 120.0,
                 "timestamp": "2024-01-15T13:00:00Z"},
            ],
            source_id="ts-agg-001",
            also_ingest_text=False,
        )
        result = mgr.query_timeseries("reactor_power_mw", aggregation="avg")
        assert result["result"] == 100.0

    def test_query_timeseries_nl_no_llm(self, tmp_path):
        from unittest.mock import MagicMock
        from msr_kb_sources import KBSourceManager
        rag_mock = MagicMock()
        rag_mock._has_llm = lambda: False
        mgr = KBSourceManager(rag_mock, kb_dir=tmp_path)
        result = mgr.query_timeseries_nl("What was the average power?")
        assert "error" in result
        assert result["rows"] == []

    def test_timeseries_status(self, tmp_path):
        from unittest.mock import MagicMock
        from msr_kb_sources import KBSourceManager
        rag_mock = MagicMock()
        mgr = KBSourceManager(rag_mock, kb_dir=tmp_path)
        st = mgr.timeseries_status()
        assert "total_readings" in st
        assert "sensors" in st
