# MSR Digital Twin – MCP Quick Start

Welcome!  This guide gets you up and running with the MSR digital twin
MCP interface in under five minutes.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 + |
| Operating system | Linux / macOS / Windows (WSL) |

---

## 1 – Install dependencies

```bash
pip install -r requirements_mcp.txt
```

---

## 2 – Start the MCP server

```bash
python msr_mcp_server_main.py
```

The server listens on **stdin** and writes JSON-RPC 2.0 responses to
**stdout**.  You can test it manually:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python msr_mcp_server_main.py
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | python msr_mcp_server_main.py
```

---

## 3 – Use the Python client

```python
from msr_digital_twin_client import MSRDigitalTwinClient

with MSRDigitalTwinClient() as client:
    print(client.get_reactor_status())
    print(client.get_all_sensor_readings())
    print(client.run_thermal_simulation(100.0, 650.0, 250.0))
```

---

## 4 – Connect Claude Desktop (or another MCP host)

Add the following block to your Claude Desktop configuration file
(`~/Library/Application Support/Claude/claude_desktop_config.json` on
macOS):

```json
{
  "mcpServers": {
    "msr-digital-twin": {
      "command": "python",
      "args": ["/absolute/path/to/msr_mcp_server_main.py"]
    }
  }
}
```

Restart Claude Desktop.  The MSR tools will appear in the tool palette.

---

## 5 – Try the RAG interface

Place reference documents (Markdown or plain text) in a `docs/` folder
and run:

```bash
python msr_digital_twin_with_rag.py "What is the current core temperature?"
```

Set `MSR_OPENAI_API_KEY` to enable LLM-powered answers:

```bash
export MSR_OPENAI_API_KEY=sk-...
python msr_digital_twin_with_rag.py "Is the reactor operating within safe limits?"
```

---

## Available Tools

| Tool | Description |
|---|---|
| `get_reactor_status` | Overall status and key parameters |
| `get_sensor_reading` | Single sensor value |
| `get_all_sensor_readings` | All sensor values at once |
| `get_sensor_history` | Historical readings for a sensor |
| `set_control_rod_position` | Adjust reactor power via control rods |
| `get_active_alarms` | List active alarms |
| `acknowledge_alarm` | Acknowledge an alarm by ID |
| `run_thermal_simulation` | Steady-state thermal-hydraulic simulation |

---

Next steps → see [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md)
