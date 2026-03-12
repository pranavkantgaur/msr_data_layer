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
