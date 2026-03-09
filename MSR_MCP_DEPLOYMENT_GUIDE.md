# MSR MCP Server – Deployment Guide

## Local Development

```bash
# Clone and enter the repository
git clone https://github.com/pranavkantgaur/msr_data_layer.git
cd msr_data_layer

# Install dependencies
pip install -r requirements_mcp.txt

# Run the server (interactive test)
python msr_mcp_server_main.py
```

---

## Connecting an MCP Host

### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

### VS Code (GitHub Copilot / Continue)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "msr-digital-twin": {
      "type": "stdio",
      "command": "python",
      "args": ["${workspaceFolder}/msr_mcp_server_main.py"]
    }
  }
}
```

### Custom Python Agent

```python
from msr_digital_twin_client import MSRDigitalTwinClient

with MSRDigitalTwinClient() as client:
    tools = client.list_tools()
    status = client.get_reactor_status()
```

---

## Running with Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements_mcp.txt
CMD ["python", "msr_mcp_server_main.py"]
```

```bash
docker build -t msr-mcp-server .
docker run -i msr-mcp-server   # -i keeps stdin open for the MCP host
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MSR_OPENAI_API_KEY` | _(unset)_ | API key for LLM + embeddings (RAG module) |
| `MSR_OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `MSR_OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `MSR_EMBED_MODEL` | `text-embedding-3-small` | Embedding model (OpenAI engine) |
| `MSR_DOCS_DIR` | `./docs` | Reference documents directory |
| `MSR_KB_DIR` | `./kb_store` | Persistent knowledge-base directory |

---

## Production Considerations

| Topic | Recommendation |
|---|---|
| **Persistence** | Replace the in-memory `_BASE_STATE` with a time-series database (InfluxDB, TimescaleDB) |
| **Authentication** | Add an API-key check in `handle_message` before dispatching tool calls |
| **Transport** | For multi-client access wrap the server with SSE or WebSocket transport |
| **Scaling** | The RAG document store is in-memory; migrate to ChromaDB or Qdrant for large corpora |
| **Monitoring** | Expose Prometheus metrics from the tool handlers |
| **Testing** | Run `pytest` to execute the unit test suite |

---

## Running Tests

```bash
pytest -v
```

---

## Updating the Simulation Model

The baseline operating point is defined in `msr_mcp_server.py`:

```python
_BASE_STATE: dict[str, Any] = {
    "reactor_power_mw": 100.0,
    "core_temperature_c": 700.0,
    ...
}
```

Adjust values to match your target design point.  Alarm thresholds are
defined in `_check_alarms()` and can be edited independently.
