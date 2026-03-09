"""
MSR Data Layer – MCP Server

Exposes MSR (Molten Salt Reactor) plant data through the Model Context
Protocol (MCP), allowing LLM agents to read live sensor data, monitor
alarms, and ingest operational data into the knowledge base.

This module is a **data layer** – it reads from an external plant data
source (SCADA, historian, or digital twin) and provides a clean MCP
interface for agents and operators.  It does **not** contain a built-in
reactor simulation or accept control commands.

Data Source
-----------
Set ``MSR_PLANT_DATA_URL`` to the base URL of an external REST API that
returns plant sensor data as a JSON object.  When unset, a development
stub with representative FLiBe-MSR parameters is used so the service
can be exercised without a live data connection.

The external API is expected to respond to ``GET <MSR_PLANT_DATA_URL>``
with a JSON object whose keys are sensor names (e.g.
``"core_temperature_c": 700.5``).

MCP Tools provided
------------------
Read tools (always present):
- ``get_reactor_status``   – summary of key operational parameters
- ``get_sensor_reading``   – latest value of a named sensor
- ``get_all_sensor_readings`` – all sensor values at once
- ``get_sensor_history``   – recent history for a sensor
- ``get_active_alarms``    – active safety/operational alarms
- ``get_data_source_info`` – data source connectivity and configuration

Write tools (data ingestion):
- ``ingest_plant_data``    – ingest operational data into the knowledge base

Environment Variables
---------------------
MSR_PLANT_DATA_URL   URL of external plant data REST API (optional)
                     When unset, the development stub is used.
"""

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Development stub – representative FLiBe-MSR parameters
# (used only when MSR_PLANT_DATA_URL is not set)
# ---------------------------------------------------------------------------

_STUB_STATE: dict[str, Any] = {
    "reactor_power_mw": 100.0,
    "core_temperature_c": 700.0,
    "salt_flow_rate_kg_s": 250.0,
    "fuel_salt_level_pct": 87.5,
    "coolant_salt_level_pct": 91.2,
    "neutron_flux_n_cm2_s": 2.5e13,
    "primary_loop_pressure_bar": 1.1,
    "heat_exchanger_outlet_c": 565.0,
    "turbine_inlet_temp_c": 540.0,
    "turbine_output_mwe": 42.0,
    "tritium_production_rate_g_day": 0.12,
    "off_gas_activity_bq_m3": 3.8e6,
    "status": "NOMINAL",
    "last_updated": datetime.now(timezone.utc).isoformat(),
}

_ALARMS: list[dict[str, Any]] = []

_SENSOR_HISTORY: dict[str, list[float]] = {
    key: [] for key in _STUB_STATE if isinstance(_STUB_STATE[key], (int, float))
}


def _get_current_state() -> dict[str, Any]:
    """
    Return the current plant sensor state.

    Fetches from ``MSR_PLANT_DATA_URL`` when configured.
    Falls back to the development stub when the URL is unset or unreachable.
    """
    external_url = os.environ.get("MSR_PLANT_DATA_URL", "").strip()
    if external_url:
        try:
            req = urllib.request.Request(
                external_url,
                headers={"Accept": "application/json", "User-Agent": "msr-data-layer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict):
                data.setdefault("last_updated", datetime.now(timezone.utc).isoformat())
                return data
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                OSError, TimeoutError, ValueError) as exc:
            import sys
            print(
                f"[DataLayer] Could not fetch from MSR_PLANT_DATA_URL ({exc!r}); "
                "using development stub.",
                file=sys.stderr,
            )

    # Development stub – return copy with timestamp
    state = dict(_STUB_STATE)
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    return state


def _record_history(state: dict[str, Any]) -> None:
    for key in _SENSOR_HISTORY:
        if key in state:
            _SENSOR_HISTORY[key].append(state[key])
            if len(_SENSOR_HISTORY[key]) > 1000:
                _SENSOR_HISTORY[key] = _SENSOR_HISTORY[key][-1000:]


# ---------------------------------------------------------------------------
# MCP tool handler functions
# ---------------------------------------------------------------------------

def get_reactor_status() -> dict[str, Any]:
    """Return the current operational status of the MSR plant."""
    state = _get_current_state()
    _record_history(state)
    return {
        "status": state.get("status", "UNKNOWN"),
        "reactor_power_mw": state.get("reactor_power_mw"),
        "core_temperature_c": state.get("core_temperature_c"),
        "last_updated": state["last_updated"],
        "data_source": os.environ.get("MSR_PLANT_DATA_URL", "development-stub"),
    }


def get_sensor_reading(sensor_name: str) -> dict[str, Any]:
    """
    Return the latest reading for a named sensor.

    Parameters
    ----------
    sensor_name : str
        One of the keys in the plant sensor state dictionary, e.g.
        ``"core_temperature_c"`` or ``"reactor_power_mw"``.
    """
    state = _get_current_state()
    if sensor_name not in state:
        available = [k for k in state if isinstance(state[k], (int, float))]
        return {
            "error": f"Unknown sensor '{sensor_name}'.",
            "available_sensors": available,
        }
    return {
        "sensor": sensor_name,
        "value": state[sensor_name],
        "unit": _sensor_unit(sensor_name),
        "timestamp": state["last_updated"],
        "data_source": os.environ.get("MSR_PLANT_DATA_URL", "development-stub"),
    }


def get_all_sensor_readings() -> dict[str, Any]:
    """Return readings for every sensor in the current plant state."""
    state = _get_current_state()
    _record_history(state)
    readings = {
        k: {"value": v, "unit": _sensor_unit(k)}
        for k, v in state.items()
        if isinstance(v, (int, float))
    }
    return {
        "readings": readings,
        "timestamp": state["last_updated"],
        "data_source": os.environ.get("MSR_PLANT_DATA_URL", "development-stub"),
    }


def get_sensor_history(sensor_name: str, last_n: int = 10) -> dict[str, Any]:
    """
    Return the last *n* recorded values for a sensor.

    Parameters
    ----------
    sensor_name : str
        Sensor key (same namespace as :func:`get_sensor_reading`).
    last_n : int
        Number of historical data points to return (max 100).
    """
    if sensor_name not in _SENSOR_HISTORY:
        return {"error": f"No history for sensor '{sensor_name}'."}
    last_n = min(max(1, last_n), 100)
    history = _SENSOR_HISTORY[sensor_name][-last_n:]
    return {
        "sensor": sensor_name,
        "unit": _sensor_unit(sensor_name),
        "count": len(history),
        "values": history,
    }


def get_active_alarms() -> dict[str, Any]:
    """Return the list of currently active alarms."""
    _check_alarms()
    return {"alarm_count": len(_ALARMS), "alarms": list(_ALARMS)}


def get_data_source_info() -> dict[str, Any]:
    """
    Return information about the configured plant data source.

    Shows whether the service is connected to a live external data source
    or using the development stub.
    """
    external_url = os.environ.get("MSR_PLANT_DATA_URL", "").strip()
    info: dict[str, Any] = {
        "mode": "external" if external_url else "development-stub",
        "plant_data_url": external_url or None,
    }
    if external_url:
        # Test connectivity
        try:
            req = urllib.request.Request(
                external_url,
                headers={"Accept": "application/json", "User-Agent": "msr-data-layer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
            info["connected"] = True
            info["message"] = "External data source reachable."
        except (urllib.error.URLError, urllib.error.HTTPError,
                OSError, TimeoutError) as exc:
            info["connected"] = False
            info["message"] = f"External data source unreachable: {exc}"
    else:
        info["connected"] = True
        info["message"] = (
            "Using development stub data. "
            "Set MSR_PLANT_DATA_URL to connect to a live plant data source."
        )
    return info


def ingest_plant_data(
    content: str,
    data_type: str = "operational_data",
    source_id: str = "",
) -> dict[str, Any]:
    """
    Ingest plant operational data into the knowledge base.

    Use this tool to push sensor snapshots, event logs, maintenance reports,
    or any plant operational records into the RAG knowledge base so they
    become searchable by future queries.

    Parameters
    ----------
    content : str
        The plant data to ingest.  Can be plain text (e.g. a maintenance
        report) or a JSON-encoded sensor snapshot / event log.
    data_type : str
        Category of the data: ``"sensor_snapshot"``, ``"event_log"``,
        ``"maintenance_report"``, or ``"operational_data"`` (default).
    source_id : str
        Optional unique identifier for the data record.  When omitted, a
        timestamp-based ID is generated.  Duplicate source IDs are skipped.
    """
    if not content or not content.strip():
        return {"success": False, "error": "content must not be empty."}
    if not source_id:
        source_id = f"{data_type}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    try:
        from msr_kb_sources import PlantDataLoader  # noqa: PLC0415
        from msr_digital_twin_with_rag import MSRDigitalTwinRAG  # noqa: PLC0415

        rag = MSRDigitalTwinRAG()
        loader = PlantDataLoader()
        chunks_added = loader.ingest_text(rag, content, source_id, data_type=data_type)
        return {
            "success": True,
            "source_id": source_id,
            "data_type": data_type,
            "chunks_added": chunks_added,
        }
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Ingestion failed: {exc}"}


# ---------------------------------------------------------------------------
# MCP tool registry
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_reactor_status",
        "description": "Get the current operational status and key parameters of the MSR plant.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": get_reactor_status,
    },
    {
        "name": "get_sensor_reading",
        "description": "Read the latest value of a specific MSR plant sensor.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sensor_name": {
                    "type": "string",
                    "description": "Name of the sensor, e.g. 'core_temperature_c'.",
                }
            },
            "required": ["sensor_name"],
        },
        "handler": get_sensor_reading,
    },
    {
        "name": "get_all_sensor_readings",
        "description": "Get the latest readings from all MSR plant sensors at once.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": get_all_sensor_readings,
    },
    {
        "name": "get_sensor_history",
        "description": "Retrieve historical readings for a sensor.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sensor_name": {"type": "string", "description": "Sensor key."},
                "last_n": {
                    "type": "integer",
                    "description": "Number of historical samples (1-100).",
                    "default": 10,
                },
            },
            "required": ["sensor_name"],
        },
        "handler": get_sensor_history,
    },
    {
        "name": "get_active_alarms",
        "description": "List all currently active alarms in the MSR plant system.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": get_active_alarms,
    },
    {
        "name": "get_data_source_info",
        "description": (
            "Return information about the configured plant data source, "
            "including connectivity status and whether a live SCADA/historian URL is configured."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": get_data_source_info,
    },
    {
        "name": "ingest_plant_data",
        "description": (
            "Ingest plant operational data (sensor snapshots, event logs, "
            "maintenance reports) into the RAG knowledge base for future queries."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Plant data text or JSON-encoded record to ingest.",
                },
                "data_type": {
                    "type": "string",
                    "description": "Category: sensor_snapshot, event_log, maintenance_report, operational_data.",
                    "default": "operational_data",
                },
                "source_id": {
                    "type": "string",
                    "description": "Optional unique ID for the record (auto-generated if omitted).",
                    "default": "",
                },
            },
            "required": ["content"],
        },
        "handler": ingest_plant_data,
    },
]

TOOL_MAP: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOLS}


# ---------------------------------------------------------------------------
# JSON-RPC request / response helpers
# ---------------------------------------------------------------------------

def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# MCP message dispatcher
# ---------------------------------------------------------------------------

def handle_message(raw: str) -> str:
    """
    Process a single JSON-RPC 2.0 message and return the serialised response.

    Implements the minimal MCP 2024-11 surface:
    - ``initialize``
    - ``tools/list``
    - ``tools/call``
    """
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as exc:
        return json.dumps(_jsonrpc_error(None, -32700, f"Parse error: {exc}"))

    req_id = msg.get("id")
    method = msg.get("method", "")
    params = msg.get("params", {})

    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "msr-data-layer", "version": "1.0.0"},
        }
        return json.dumps(_jsonrpc_result(req_id, result))

    if method == "tools/list":
        tool_list = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in TOOLS
        ]
        return json.dumps(_jsonrpc_result(req_id, {"tools": tool_list}))

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if tool_name not in TOOL_MAP:
            return json.dumps(
                _jsonrpc_error(req_id, -32601, f"Unknown tool: '{tool_name}'")
            )
        try:
            output = TOOL_MAP[tool_name]["handler"](**arguments)
        except TypeError as exc:
            return json.dumps(_jsonrpc_error(req_id, -32602, f"Invalid params: {exc}"))
        except Exception as exc:  # noqa: BLE001
            return json.dumps(_jsonrpc_error(req_id, -32603, f"Internal error: {exc}"))
        return json.dumps(
            _jsonrpc_result(
                req_id,
                {"content": [{"type": "text", "text": json.dumps(output, indent=2)}]},
            )
        )

    # Notifications (no id) are silently ignored per MCP spec
    if req_id is None:
        return ""

    return json.dumps(_jsonrpc_error(req_id, -32601, f"Method not found: '{method}'"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sensor_unit(sensor_name: str) -> str:
    units = {
        "reactor_power_mw": "MW",
        "core_temperature_c": "°C",
        "salt_flow_rate_kg_s": "kg/s",
        "fuel_salt_level_pct": "%",
        "coolant_salt_level_pct": "%",
        "neutron_flux_n_cm2_s": "n/cm²/s",
        "primary_loop_pressure_bar": "bar",
        "heat_exchanger_outlet_c": "°C",
        "turbine_inlet_temp_c": "°C",
        "turbine_output_mwe": "MWe",
        "tritium_production_rate_g_day": "g/day",
        "off_gas_activity_bq_m3": "Bq/m³",
    }
    return units.get(sensor_name, "")


def _check_alarms() -> None:
    state = _get_current_state()
    thresholds = {
        "core_temperature_c": (None, 750.0, "CORE_TEMP_HIGH"),
        "reactor_power_mw": (None, 110.0, "POWER_HIGH"),
        "fuel_salt_level_pct": (70.0, None, "FUEL_LEVEL_LOW"),
        "primary_loop_pressure_bar": (None, 1.5, "PRIMARY_PRESSURE_HIGH"),
    }
    existing_ids = {a["alarm_id"] for a in _ALARMS}
    for sensor, (low, high, alarm_id) in thresholds.items():
        value = state.get(sensor, 0)
        triggered = (high is not None and value > high) or (low is not None and value < low)
        if triggered and alarm_id not in existing_ids:
            _ALARMS.append(
                {
                    "alarm_id": alarm_id,
                    "sensor": sensor,
                    "value": value,
                    "threshold_high": high,
                    "threshold_low": low,
                    "severity": "WARNING",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
