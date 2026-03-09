# MSR Digital Twin MCP Interface – Full Guide

## Overview

The MSR (Molten Salt Reactor) data layer exposes a physics-based digital
twin through the [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io).
This allows large language model (LLM) agents to:

* Query real-time (simulated) sensor readings
* Run thermal-hydraulic simulations
* Monitor and acknowledge alarms
* Control reactor parameters in a sandboxed environment

---

## Architecture

```
┌─────────────────────────┐        stdin / stdout
│   MCP Host              │◄──────────────────────►┌───────────────────────────┐
│  (Claude Desktop,       │    JSON-RPC 2.0         │  msr_mcp_server_main.py  │
│   VS Code, agent, …)    │                         │  (stdio transport)       │
└─────────────────────────┘                         │                          │
                                                    │  msr_mcp_server.py       │
                                                    │  ┌──────────────────┐    │
                                                    │  │ Tool handlers    │    │
                                                    │  │  • sensors       │    │
                                                    │  │  • simulation    │    │
                                                    │  │  • alarms        │    │
                                                    │  └──────────────────┘    │
                                                    └───────────────────────────┘

                                                    ┌───────────────────────────┐
                                                    │ msr_digital_twin_with_rag │
                                                    │  DocumentStore (TF-IDF)   │
                                                    │  + Live reactor context   │
                                                    │  + OpenAI-compatible LLM  │
                                                    └───────────────────────────┘
```

---

## Module Reference

### `msr_mcp_server.py`

Core library.  Contains:

* **`_BASE_STATE`** – simulated MSR operating point
* **Tool handler functions** – one Python function per MCP tool
* **`TOOLS`** – list of MCP tool descriptors with JSON schemas
* **`handle_message(raw)`** – JSON-RPC 2.0 dispatcher

#### Simulated Sensors

| Sensor Key | Unit | Nominal Value |
|---|---|---|
| `reactor_power_mw` | MW | 100 |
| `core_temperature_c` | °C | 700 |
| `salt_flow_rate_kg_s` | kg/s | 250 |
| `fuel_salt_level_pct` | % | 87.5 |
| `coolant_salt_level_pct` | % | 91.2 |
| `control_rod_position_pct` | % | 45 |
| `neutron_flux_n_cm2_s` | n/cm²/s | 2.5 × 10¹³ |
| `primary_loop_pressure_bar` | bar | 1.1 |
| `heat_exchanger_outlet_c` | °C | 565 |
| `turbine_inlet_temp_c` | °C | 540 |
| `turbine_output_mwe` | MWe | 42 |
| `tritium_production_rate_g_day` | g/day | 0.12 |
| `off_gas_activity_bq_m3` | Bq/m³ | 3.8 × 10⁶ |

### `msr_mcp_server_main.py`

Entry point.  Reads newline-delimited JSON-RPC messages from **stdin**
and writes responses to **stdout**.  Log messages go to **stderr**.

### `msr_digital_twin_client.py`

Python client class `MSRDigitalTwinClient`.  Spawns the server as a
subprocess and wraps each MCP tool in a typed Python method.

### `msr_digital_twin_with_rag.py`

RAG pipeline:

1. `DocumentStore` – TF-IDF indexing and cosine-similarity retrieval
2. `MSRDigitalTwinRAG` – orchestrates document retrieval + live reactor
   context + LLM call

---

## Tool Reference

### `get_reactor_status`

Returns the current operational status, power level, and core
temperature.

**Input:** _(none)_

**Output example:**
```json
{
  "status": "NOMINAL",
  "reactor_power_mw": 100.2,
  "core_temperature_c": 701.4,
  "last_updated": "2026-03-09T06:00:00+00:00"
}
```

---

### `get_sensor_reading`

**Input:**
```json
{ "sensor_name": "core_temperature_c" }
```

**Output example:**
```json
{ "sensor": "core_temperature_c", "value": 701.4, "unit": "°C", "timestamp": "…" }
```

---

### `get_all_sensor_readings`

Returns a map of every sensor with its value and unit.

---

### `get_sensor_history`

**Input:**
```json
{ "sensor_name": "reactor_power_mw", "last_n": 20 }
```

---

### `set_control_rod_position`

Adjusts the control rod insertion depth.  Power and neutron flux are
scaled proportionally.

**Input:**
```json
{ "position_pct": 60.0 }
```

---

### `get_active_alarms`

Returns all currently active alarms.

**Alarm thresholds (automatic):**

| Sensor | Condition | Alarm ID |
|---|---|---|
| `core_temperature_c` | > 750 °C | `CORE_TEMP_HIGH` |
| `reactor_power_mw` | > 110 MW | `POWER_HIGH` |
| `fuel_salt_level_pct` | < 70 % | `FUEL_LEVEL_LOW` |
| `primary_loop_pressure_bar` | > 1.5 bar | `PRIMARY_PRESSURE_HIGH` |

---

### `acknowledge_alarm`

**Input:**
```json
{ "alarm_id": "CORE_TEMP_HIGH" }
```

---

### `run_thermal_simulation`

Steady-state thermal-hydraulic model using FLiBe salt properties.
Estimates outlet temperature, thermal efficiency, and electrical output.

**Input:**
```json
{ "power_mw": 100, "inlet_temp_c": 650, "flow_rate_kg_s": 250 }
```

**Output example:**
```json
{
  "power_mw": 100,
  "inlet_temp_c": 650.0,
  "outlet_temp_c": 815.15,
  "delta_t_c": 165.15,
  "estimated_efficiency": 0.3379,
  "estimated_electrical_output_mwe": 33.79
}
```

---

## Extending the Server

### Adding a new tool

1. Write a handler function in `msr_mcp_server.py`.
2. Add an entry to the `TOOLS` list with a JSON schema.
3. The `TOOL_MAP` and `handle_message` dispatcher pick it up automatically.

### Connecting to a real data source

Replace the `_BASE_STATE` dictionary and the `_get_current_state()`
function with calls to your SCADA historian, OPC-UA server, or database.

---

See also → [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md)
