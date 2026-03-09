"""Unit tests for the MSR MCP server."""

import json
import pytest
from msr_mcp_server import (
    handle_message,
    get_reactor_status,
    get_sensor_reading,
    get_all_sensor_readings,
    get_sensor_history,
    set_control_rod_position,
    get_active_alarms,
    acknowledge_alarm,
    run_thermal_simulation,
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
        "get_reactor_status", "get_sensor_reading", "get_all_sensor_readings",
        "get_sensor_history", "set_control_rod_position", "get_active_alarms",
        "acknowledge_alarm", "run_thermal_simulation",
    }
    assert expected == names


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
# Tool handler functions
# ---------------------------------------------------------------------------

def test_get_reactor_status_keys():
    status = get_reactor_status()
    assert {"status", "reactor_power_mw", "core_temperature_c", "last_updated"} == set(status)


def test_get_sensor_reading_valid():
    result = get_sensor_reading("core_temperature_c")
    assert result["sensor"] == "core_temperature_c"
    assert isinstance(result["value"], float)
    assert result["unit"] == "°C"


def test_get_sensor_reading_invalid():
    result = get_sensor_reading("does_not_exist")
    assert "error" in result
    assert "available_sensors" in result


def test_get_all_sensor_readings():
    result = get_all_sensor_readings()
    assert "readings" in result
    assert "core_temperature_c" in result["readings"]


def test_get_sensor_history_empty():
    # History may be empty on a fresh import; should still return valid structure
    result = get_sensor_history("reactor_power_mw", last_n=5)
    assert "values" in result


def test_get_sensor_history_unknown():
    result = get_sensor_history("unknown_sensor")
    assert "error" in result


def test_set_control_rod_position_valid():
    result = set_control_rod_position(80.0)
    assert result["success"] is True
    assert result["control_rod_position_pct"] == 80.0


def test_set_control_rod_position_invalid():
    result = set_control_rod_position(150.0)
    assert "error" in result


def test_get_active_alarms_structure():
    result = get_active_alarms()
    assert "alarm_count" in result
    assert "alarms" in result


def test_acknowledge_alarm_not_found():
    result = acknowledge_alarm("NONEXISTENT_ALARM")
    assert result["success"] is False


def test_run_thermal_simulation():
    result = run_thermal_simulation(100.0, 650.0, 250.0)
    assert result["inlet_temp_c"] == 650.0
    assert result["outlet_temp_c"] > result["inlet_temp_c"]
    assert 0 < result["estimated_efficiency"] < 1
    assert result["estimated_electrical_output_mwe"] > 0


def test_all_tools_have_handler():
    for tool in TOOLS:
        assert callable(tool["handler"]), f"Tool '{tool['name']}' has no callable handler"
