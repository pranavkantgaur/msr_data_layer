# MSR Data Layer

A **Molten Salt Reactor (MSR) digital twin** exposed through the
[Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io),
enabling LLM agents (Claude, GitHub Copilot, custom agents) to query
live reactor sensor data, run thermal-hydraulic simulations, monitor
alarms, and retrieve answers from technical documents via
Retrieval-Augmented Generation (RAG).

---

## Repository Contents

| File | Description |
|---|---|
| `msr_mcp_server.py` | Core MCP server – tool handlers, JSON-RPC dispatcher |
| `msr_mcp_server_main.py` | Entry point – stdio transport server |
| `msr_digital_twin_client.py` | Python client for the MCP server |
| `msr_digital_twin_with_rag.py` | RAG pipeline over documents + live reactor data |
| `requirements_mcp.txt` | Python dependencies |
| `00_MCP_START_HERE.md` | Quick-start guide |
| `MSR_DIGITAL_TWIN_MCP_GUIDE.md` | Full architecture and tool reference |
| `MSR_MCP_DEPLOYMENT_GUIDE.md` | Deployment and production guide |

---

## Quick Start

### 1 – Install dependencies

```bash
pip install -r requirements_mcp.txt
```

### 2 – Run the demo client

```bash
python msr_digital_twin_client.py
```

Output:
```
MSR Digital Twin Client – quick demo

Available tools:
  • get_reactor_status – Get the current operational status …
  • get_sensor_reading – Read the latest value of a specific MSR sensor.
  • …

Reactor Status:
  status: NOMINAL
  reactor_power_mw: 100.2
  core_temperature_c: 701.4
  last_updated: 2026-03-09T06:00:00+00:00

Core temperature reading:
  core_temperature_c: 701.4 °C

Thermal simulation (100 MW, 650 °C inlet, 250 kg/s):
  outlet_temp_c: 815.15
  estimated_efficiency: 0.3379
  …
```

### 3 – Connect Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "msr-digital-twin": {
      "command": "python",
      "args": ["/path/to/msr_mcp_server_main.py"]
    }
  }
}
```

### 4 – Ask questions with RAG

```bash
# Without LLM – returns enriched context
python msr_digital_twin_with_rag.py "What is the current core temperature?"

# With LLM (OpenAI-compatible)
export MSR_OPENAI_API_KEY=sk-...
python msr_digital_twin_with_rag.py "Is the reactor operating within safe limits?"
```

---

## Available MCP Tools

| Tool | Description |
|---|---|
| `get_reactor_status` | Current status, power, and core temperature |
| `get_sensor_reading` | Single named sensor value |
| `get_all_sensor_readings` | All sensors at once |
| `get_sensor_history` | Historical readings (up to last 100 samples) |
| `set_control_rod_position` | Adjust control rod depth (0–100 %) |
| `get_active_alarms` | List active alarms |
| `acknowledge_alarm` | Acknowledge an alarm by ID |
| `run_thermal_simulation` | Steady-state thermal-hydraulic simulation |

---

## Architecture

```
MCP Host (Claude / agent)
        │  stdin/stdout JSON-RPC 2.0
        ▼
msr_mcp_server_main.py
        │
        ▼
msr_mcp_server.py  ── tool handlers ── simulated reactor state
                                  └── alarm checker

msr_digital_twin_with_rag.py
  ├── DocumentStore (TF-IDF retrieval)
  ├── MSRDigitalTwinClient (live data)
  └── LLM call (OpenAI-compatible, optional)
```

---

## Testing

```bash
pytest -v
```

---

## Documentation

* [00_MCP_START_HERE.md](00_MCP_START_HERE.md) – five-minute quick start
* [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md) – full guide
* [MSR_MCP_DEPLOYMENT_GUIDE.md](MSR_MCP_DEPLOYMENT_GUIDE.md) – deployment guide

---

## License

MIT
