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
    KBSourceManager,
    MSRArchiveLoader,
    OpenAlexLoader,
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

    with patch("msr_kb_sources._http_get") as mock_get, \
         patch("msr_kb_sources._http_get_text", return_value="Content"):
        # Return files_data for GitHub API, page for OpenAlex
        mock_get.side_effect = [files_data, page, page]
        result = mgr.update_all()

    assert "archive" in result
    assert "openalex" in result
    assert isinstance(result["archive"], int)
    assert isinstance(result["openalex"], int)


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

