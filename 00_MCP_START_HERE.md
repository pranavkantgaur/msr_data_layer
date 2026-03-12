# MSR Data Layer – MCP Quick Start

Welcome!  This guide gets you up and running with the MSR data layer MCP
interface in under five minutes.

> **What is this?**  A read-only data layer that exposes MSR plant sensor
> readings and a RAG knowledge base through the Model Context Protocol.  It
> is **not** a reactor simulator or control system.

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
from msr_digital_twin_client import MSRDataLayerClient

with MSRDataLayerClient() as client:
    # List all available tools
    for tool in client.list_tools():
        print(f"  • {tool['name']} – {tool['description']}")

    # Read live (or stubbed) plant data
    print(client.get_reactor_status())
    print(client.get_all_sensor_readings())
    print(client.get_active_alarms())
```

---

## 4 – Connect to a live plant data source

The data layer reads sensor values from an external REST API when
`MSR_PLANT_DATA_URL` is set:

```bash
export MSR_PLANT_DATA_URL=https://your-scada.example.com/api/plant/state
python msr_mcp_server_main.py
```

When this variable is unset, a built-in development stub with representative
FLiBe-MSR parameters is used so the service can be exercised without a live
connection.

---

## 5 – Connect Claude Desktop (or another MCP host)

Add to `claude_desktop_config.json`
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "msr-data-layer": {
      "command": "python",
      "args": ["/absolute/path/to/msr_mcp_server_main.py"]
    }
  }
}
```

Restart Claude Desktop.  The MSR data layer tools will appear in the tool palette.

---

## 6 – Try the enhanced RAG interface

Place reference documents (Markdown or plain text) in a `docs/` folder
and run:

```bash
python msr_digital_twin_with_rag.py "What are the safe temperature limits?"
```

Set `MSR_OPENAI_API_KEY` to enable LLM-powered multi-step answers:

```bash
export MSR_OPENAI_API_KEY=sk-...
export MSR_EMBED_MODEL=text-embedding-3-small
export MSR_KB_DIR=./kb_store

python msr_digital_twin_with_rag.py "What does the ORNL archive say about FLiNaK corrosion rates?"
```

With an API key, the RAG pipeline uses:
1. Query decomposition → ≤5 targeted sub-queries
2. Parallel hybrid search (dense embedding + TF-IDF)
3. Sub-answer extraction per search result
4. Final synthesis combining all findings + live plant data

Without an API key, the pipeline uses the random-projection embedding
engine (numpy, no external deps) and returns an enriched context prompt.

---

## Available Tools

| Tool | Type | Description |
|---|---|---|
| `get_reactor_status` | Read | Overall plant status and key parameters |
| `get_sensor_reading` | Read | Single named sensor value |
| `get_all_sensor_readings` | Read | All sensor values at once |
| `get_sensor_history` | Read | Historical readings for a sensor (session buffer) |
| `get_active_alarms` | Read | List currently active alarms |
| `get_data_source_info` | Read | Data source mode, URL, and connectivity status |
| `ingest_plant_data` | Write | Push operational data into the RAG knowledge base |

---

Next steps → see [MSR_DIGITAL_TWIN_MCP_GUIDE.md](MSR_DIGITAL_TWIN_MCP_GUIDE.md)
