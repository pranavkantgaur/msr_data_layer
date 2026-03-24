"""Unit tests for the MSR Data Layer MCP Server."""

import json
import pytest
from msr_mcp_server import (
    handle_message,
    get_reactor_status,
    get_sensor_reading,
    get_all_sensor_readings,
    get_sensor_history,
    get_active_alarms,
    get_data_source_info,
    ingest_plant_data,
    TOOLS,
    TOOL_MAP,
)


# ---------------------------------------------------------------------------
# handle_message – JSON-RPC dispatcher
# ---------------------------------------------------------------------------

def _call(method, params=None, req_id=1):
    raw = json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
    return json.loads(handle_message(raw))


def test_initialize():
    resp = _call("initialize")
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "msr-data-layer"


def test_tools_list():
    resp = _call("tools/list")
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    expected = {
        "get_reactor_status",
        "get_sensor_reading",
        "get_all_sensor_readings",
        "get_sensor_history",
        "get_active_alarms",
        "get_data_source_info",
        "ingest_plant_data",
        "query_sensor_timeseries",
        "get_sensor_stats",
        "query_plant_data_nl",
        "ingest_full_paper_text",
    }
    assert expected == names


def test_tools_list_does_not_include_simulation():
    """Simulation and actuation tools must not be exposed."""
    resp = _call("tools/list")
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "run_thermal_simulation" not in names
    assert "set_control_rod_position" not in names
    assert "acknowledge_alarm" not in names


def test_tools_call_get_reactor_status():
    resp = _call("tools/call", {"name": "get_reactor_status", "arguments": {}})
    content = json.loads(resp["result"]["content"][0]["text"])
    assert "status" in content
    assert "reactor_power_mw" in content


def test_tools_call_unknown_tool():
    resp = _call("tools/call", {"name": "nonexistent", "arguments": {}})
    assert "error" in resp


def test_unknown_method_returns_error():
    resp = _call("rpc.discover")
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_parse_error():
    resp = json.loads(handle_message("not json {{{"))
    assert "error" in resp
    assert resp["error"]["code"] == -32700


def test_notification_returns_empty():
    raw = json.dumps({"jsonrpc": "2.0", "method": "some/notification", "params": {}})
    assert handle_message(raw) == ""


# ---------------------------------------------------------------------------
# Tool handler functions – read tools
# ---------------------------------------------------------------------------

def test_get_reactor_status_keys():
    status = get_reactor_status()
    assert {"status", "reactor_power_mw", "core_temperature_c", "last_updated", "data_source"} <= set(status)


def test_get_reactor_status_data_source_stub(monkeypatch):
    monkeypatch.delenv("MSR_PLANT_DATA_URL", raising=False)
    status = get_reactor_status()
    assert status["data_source"] == "development-stub"


def test_get_sensor_reading_valid():
    result = get_sensor_reading("core_temperature_c")
    assert result["sensor"] == "core_temperature_c"
    assert isinstance(result["value"], float)
    assert result["unit"] == "°C"
    assert "data_source" in result


def test_get_sensor_reading_invalid():
    result = get_sensor_reading("does_not_exist")
    assert "error" in result
    assert "available_sensors" in result


def test_get_all_sensor_readings():
    result = get_all_sensor_readings()
    assert "readings" in result
    assert "core_temperature_c" in result["readings"]
    assert "data_source" in result


def test_get_sensor_history_empty():
    """Sensor history is empty until sensors are polled; should still return valid structure."""
    result = get_sensor_history("reactor_power_mw", last_n=5)
    assert "values" in result


def test_get_sensor_history_unknown():
    result = get_sensor_history("unknown_sensor")
    assert "error" in result


def test_get_active_alarms_structure():
    result = get_active_alarms()
    assert "alarm_count" in result
    assert "alarms" in result


# ---------------------------------------------------------------------------
# get_data_source_info
# ---------------------------------------------------------------------------

def test_get_data_source_info_stub(monkeypatch):
    monkeypatch.delenv("MSR_PLANT_DATA_URL", raising=False)
    info = get_data_source_info()
    assert info["mode"] == "development-stub"
    assert info["plant_data_url"] is None
    assert info["connected"] is True
    assert "development stub" in info["message"].lower()


def test_get_data_source_info_external_url_unreachable(monkeypatch):
    monkeypatch.setenv("MSR_PLANT_DATA_URL", "http://127.0.0.1:19999/api/plant")
    info = get_data_source_info()
    assert info["mode"] == "external"
    assert info["connected"] is False
    assert "unreachable" in info["message"].lower()


def test_get_data_source_info_external_url_reachable(monkeypatch):
    """When the URL responds with JSON, connected should be True."""
    import urllib.request
    from unittest.mock import patch, MagicMock
    monkeypatch.setenv("MSR_PLANT_DATA_URL", "http://mock-plant.example.com/api")

    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = b'{"core_temperature_c": 700.0}'

    with patch("urllib.request.urlopen", return_value=mock_resp):
        info = get_data_source_info()
    assert info["mode"] == "external"
    assert info["connected"] is True


# ---------------------------------------------------------------------------
# ingest_plant_data (MCP tool – requires PlantDataLoader / RAG)
# ---------------------------------------------------------------------------

def test_ingest_plant_data_empty_content():
    result = ingest_plant_data(content="")
    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_ingest_plant_data_import_error(monkeypatch):
    """When the KB is unavailable, ingest returns success=False."""
    import sys
    # Block import of msr_kb_sources to simulate missing KB
    original = sys.modules.get("msr_kb_sources")
    sys.modules["msr_kb_sources"] = None  # type: ignore[assignment]
    try:
        result = ingest_plant_data(content="Core temperature reading: 702°C")
        assert result["success"] is False
    finally:
        if original is not None:
            sys.modules["msr_kb_sources"] = original
        else:
            sys.modules.pop("msr_kb_sources", None)


# ---------------------------------------------------------------------------
# Tool registry consistency
# ---------------------------------------------------------------------------

def test_all_tools_have_handler():
    for tool in TOOLS:
        assert callable(tool["handler"]), f"Tool '{tool['name']}' has no callable handler"


# ---------------------------------------------------------------------------
# Timeseries tool tests
# ---------------------------------------------------------------------------

def test_query_sensor_timeseries_empty_store(tmp_path, monkeypatch):
    """query_sensor_timeseries returns empty rows when store has no data."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_mcp_server import query_sensor_timeseries
    result = query_sensor_timeseries("reactor_power_mw")
    assert result["sensor_name"] == "reactor_power_mw"
    assert result["rows"] == []
    assert result["count"] == 0


def test_query_sensor_timeseries_with_data(tmp_path, monkeypatch):
    """query_sensor_timeseries returns inserted rows."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_kb_sources import TimeseriesStore
    ts = TimeseriesStore(kb_dir=tmp_path)
    ts.insert_readings(
        [{"sensor_name": "reactor_power_mw", "value": 99.5, "unit": "MW",
          "timestamp": "2024-01-15T14:00:00Z"}],
        source_id="test-001",
    )
    from msr_mcp_server import query_sensor_timeseries
    result = query_sensor_timeseries("reactor_power_mw")
    assert result["count"] == 1
    assert result["rows"][0]["value"] == 99.5


def test_query_sensor_timeseries_last_n(tmp_path, monkeypatch):
    """query_sensor_timeseries last_n=1 returns only the most recent reading."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_kb_sources import TimeseriesStore
    ts = TimeseriesStore(kb_dir=tmp_path)
    ts.insert_readings(
        [
            {"sensor_name": "core_temperature_c", "value": 700.0,
             "timestamp": "2024-01-15T13:00:00Z"},
            {"sensor_name": "core_temperature_c", "value": 702.0,
             "timestamp": "2024-01-15T14:00:00Z"},
        ],
        source_id="test-002",
    )
    from msr_mcp_server import query_sensor_timeseries
    result = query_sensor_timeseries("core_temperature_c", last_n=1)
    assert result["count"] == 1
    assert result["rows"][0]["value"] == 702.0


def test_get_sensor_stats_empty(tmp_path, monkeypatch):
    """get_sensor_stats returns result=None when store is empty."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_mcp_server import get_sensor_stats
    result = get_sensor_stats("reactor_power_mw", aggregation="avg")
    assert result["sensor_name"] == "reactor_power_mw"
    assert result["result"] is None


def test_get_sensor_stats_with_data(tmp_path, monkeypatch):
    """get_sensor_stats returns correct avg / min / max."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_kb_sources import TimeseriesStore
    ts = TimeseriesStore(kb_dir=tmp_path)
    ts.insert_readings(
        [
            {"sensor_name": "reactor_power_mw", "value": 90.0,
             "timestamp": "2024-01-15T12:00:00Z"},
            {"sensor_name": "reactor_power_mw", "value": 110.0,
             "timestamp": "2024-01-15T13:00:00Z"},
        ],
        source_id="test-003",
    )
    from msr_mcp_server import get_sensor_stats
    avg_result = get_sensor_stats("reactor_power_mw", aggregation="avg")
    assert avg_result["result"] == 100.0

    min_result = get_sensor_stats("reactor_power_mw", aggregation="min")
    assert min_result["result"] == 90.0

    max_result = get_sensor_stats("reactor_power_mw", aggregation="max")
    assert max_result["result"] == 110.0


def test_query_plant_data_nl_no_llm(tmp_path, monkeypatch):
    """query_plant_data_nl returns graceful error when no LLM is configured."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    monkeypatch.delenv("MSR_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MSR_USE_LOCAL_GPU", raising=False)
    from msr_mcp_server import query_plant_data_nl
    result = query_plant_data_nl("What was the average power?")
    assert result["question"] == "What was the average power?"
    # Should return error or empty rows — not raise
    assert "rows" in result
    assert isinstance(result["rows"], list)


# ---------------------------------------------------------------------------
# ingest_full_paper_text tool tests
# ---------------------------------------------------------------------------

def test_ingest_full_paper_text_empty_text(tmp_path, monkeypatch):
    """ingest_full_paper_text returns error when text is empty."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_mcp_server import ingest_full_paper_text
    result = ingest_full_paper_text("arxiv:2401.12345", "")
    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_ingest_full_paper_text_empty_source_id(tmp_path, monkeypatch):
    """ingest_full_paper_text returns error when source_id is empty."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    from msr_mcp_server import ingest_full_paper_text
    result = ingest_full_paper_text("", "Full paper text here.")
    assert result["success"] is False
    assert "empty" in result["error"].lower()


def test_ingest_full_paper_text_success(tmp_path, monkeypatch):
    """ingest_full_paper_text stores full text under full:<source_id>."""
    monkeypatch.setenv("MSR_KB_DIR", str(tmp_path))
    monkeypatch.delenv("MSR_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("MSR_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)  # prevent GitHub Models API engine
    from msr_mcp_server import ingest_full_paper_text
    result = ingest_full_paper_text(
        "arxiv:2401.12345",
        "# Full paper text\n\nThis is the complete text of the paper about "
        "molten salt reactors. The key finding is that FLiNaK is stable...",
    )
    assert result["success"] is True
    assert result["source_id"] == "arxiv:2401.12345"
    assert result["full_source_id"] == "full:arxiv:2401.12345"
    assert result["chunks_added"] >= 1
