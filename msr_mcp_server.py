"""
MSR Data Layer - MCP Server

Exposes MSR (Molten Salt Reactor) digital twin data through the
Model Context Protocol (MCP), allowing LLM agents to query and
interact with reactor state, sensor readings, and simulation results.
"""

import json
import math
import random
import time
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Simulated MSR state (in a real deployment this would connect to a database
# or a live SCADA / historian system)
# ---------------------------------------------------------------------------

_BASE_STATE: dict[str, Any] = {
    "reactor_power_mw": 100.0,
    "core_temperature_c": 700.0,
    "salt_flow_rate_kg_s": 250.0,
    "fuel_salt_level_pct": 87.5,
    "coolant_salt_level_pct": 91.2,
    "control_rod_position_pct": 45.0,
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

_ALARMS: list[dict[str, str]] = []

_SENSOR_HISTORY: dict[str, list[float]] = {
    key: [] for key in _BASE_STATE if isinstance(_BASE_STATE[key], (int, float))
}


def _get_current_state() -> dict[str, Any]:
    """Return a snapshot of the current reactor state with small noise."""
    noise_factor = 0.002  # 0.2 % random walk
    state = dict(_BASE_STATE)
    for key, value in state.items():
        if isinstance(value, float):
            state[key] = round(value * (1 + random.uniform(-noise_factor, noise_factor)), 4)
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
    """Return the current operational status of the MSR."""
    state = _get_current_state()
    _record_history(state)
    return {
        "status": state["status"],
        "reactor_power_mw": state["reactor_power_mw"],
        "core_temperature_c": state["core_temperature_c"],
        "last_updated": state["last_updated"],
    }


def get_sensor_reading(sensor_name: str) -> dict[str, Any]:
    """
    Return the latest reading for a named sensor.

    Parameters
    ----------
    sensor_name : str
        One of the keys in the reactor state dictionary, e.g.
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
    }


def get_all_sensor_readings() -> dict[str, Any]:
    """Return readings for every sensor in the reactor model."""
    state = _get_current_state()
    _record_history(state)
    readings = {
        k: {"value": v, "unit": _sensor_unit(k)}
        for k, v in state.items()
        if isinstance(v, (int, float))
    }
    return {"readings": readings, "timestamp": state["last_updated"]}


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


def set_control_rod_position(position_pct: float) -> dict[str, Any]:
    """
    Adjust the control rod insertion depth.

    Parameters
    ----------
    position_pct : float
        Desired position as a percentage (0 = fully inserted / shutdown,
        100 = fully withdrawn / maximum reactivity).
    """
    if not 0.0 <= position_pct <= 100.0:
        return {"error": "position_pct must be between 0 and 100."}
    _BASE_STATE["control_rod_position_pct"] = round(position_pct, 2)
    # Proportionally adjust power and neutron flux
    scale = position_pct / 100.0
    _BASE_STATE["reactor_power_mw"] = round(100.0 * scale, 2)
    _BASE_STATE["neutron_flux_n_cm2_s"] = round(2.5e13 * scale, 2)
    return {
        "success": True,
        "control_rod_position_pct": _BASE_STATE["control_rod_position_pct"],
        "reactor_power_mw": _BASE_STATE["reactor_power_mw"],
        "message": "Control rod position updated in simulation.",
    }


def get_active_alarms() -> dict[str, Any]:
    """Return the list of currently active alarms."""
    _check_alarms()
    return {"alarm_count": len(_ALARMS), "alarms": list(_ALARMS)}


def acknowledge_alarm(alarm_id: str) -> dict[str, Any]:
    """
    Acknowledge an active alarm by its ID.

    Parameters
    ----------
    alarm_id : str
        The ``alarm_id`` field from a record returned by :func:`get_active_alarms`.
    """
    global _ALARMS  # noqa: PLW0603
    before = len(_ALARMS)
    _ALARMS = [a for a in _ALARMS if a["alarm_id"] != alarm_id]
    if len(_ALARMS) < before:
        return {"success": True, "message": f"Alarm '{alarm_id}' acknowledged."}
    return {"success": False, "message": f"Alarm '{alarm_id}' not found."}


def run_thermal_simulation(
    power_mw: float,
    inlet_temp_c: float,
    flow_rate_kg_s: float,
) -> dict[str, Any]:
    """
    Run a simplified steady-state thermal-hydraulic simulation.

    Parameters
    ----------
    power_mw : float
        Reactor thermal power in MW.
    inlet_temp_c : float
        Primary salt inlet temperature in °C.
    flow_rate_kg_s : float
        Primary salt mass flow rate in kg/s.
    """
    # Specific heat capacity of FLiBe salt ≈ 2415 J/(kg·K)
    cp_flibe = 2415.0
    delta_t = (power_mw * 1e6) / (flow_rate_kg_s * cp_flibe)
    outlet_temp_c = inlet_temp_c + delta_t
    # Thermal efficiency estimate (Carnot-like, T_hot in K vs T_cold = 300 K)
    t_hot_k = (outlet_temp_c + 273.15)
    t_cold_k = 300.0
    efficiency = 1 - t_cold_k / t_hot_k
    electrical_output_mwe = power_mw * efficiency
    return {
        "power_mw": power_mw,
        "inlet_temp_c": round(inlet_temp_c, 2),
        "outlet_temp_c": round(outlet_temp_c, 2),
        "delta_t_c": round(delta_t, 2),
        "estimated_efficiency": round(efficiency, 4),
        "estimated_electrical_output_mwe": round(electrical_output_mwe, 2),
        "note": "Simplified steady-state model using FLiBe heat capacity.",
    }


# ---------------------------------------------------------------------------
# MCP tool registry
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_reactor_status",
        "description": "Get the current operational status and key parameters of the MSR.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": get_reactor_status,
    },
    {
        "name": "get_sensor_reading",
        "description": "Read the latest value of a specific MSR sensor.",
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
        "description": "Get the latest readings from all MSR sensors at once.",
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
        "name": "set_control_rod_position",
        "description": "Adjust the control rod position to change reactor power.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "position_pct": {
                    "type": "number",
                    "description": "Control rod position 0-100 %.",
                }
            },
            "required": ["position_pct"],
        },
        "handler": set_control_rod_position,
    },
    {
        "name": "get_active_alarms",
        "description": "List all currently active alarms in the MSR system.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "handler": get_active_alarms,
    },
    {
        "name": "acknowledge_alarm",
        "description": "Acknowledge an active alarm by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "alarm_id": {
                    "type": "string",
                    "description": "The alarm_id to acknowledge.",
                }
            },
            "required": ["alarm_id"],
        },
        "handler": acknowledge_alarm,
    },
    {
        "name": "run_thermal_simulation",
        "description": (
            "Run a simplified steady-state thermal-hydraulic simulation "
            "for the MSR primary loop."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "power_mw": {"type": "number", "description": "Reactor power in MW."},
                "inlet_temp_c": {"type": "number", "description": "Salt inlet temp in °C."},
                "flow_rate_kg_s": {"type": "number", "description": "Flow rate in kg/s."},
            },
            "required": ["power_mw", "inlet_temp_c", "flow_rate_kg_s"],
        },
        "handler": run_thermal_simulation,
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
        "control_rod_position_pct": "%",
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
