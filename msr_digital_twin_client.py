"""
MSR Data Layer Client

A lightweight Python client that communicates with the MSR Data Layer MCP
server over stdio (subprocess) and exposes a clean API for reading plant
sensor data and ingesting operational data into the knowledge base.

This client connects to the data layer, which reads from an external plant
data source (SCADA, historian, or digital twin API).  It does not include
simulation or control-actuation capabilities.

Example
-------
    from msr_digital_twin_client import MSRDataLayerClient

    with MSRDataLayerClient() as client:
        status = client.get_reactor_status()
        print(status)
        readings = client.get_all_sensor_readings()
        print(readings)
"""

import json
import subprocess
import sys
import threading
from typing import Any


class MCPError(Exception):
    """Raised when the MCP server returns an error response."""


# Backward-compatible alias
MSRDigitalTwinClient = None  # set at bottom of module


class MSRDataLayerClient:
    """
    Subprocess-based MCP client for the MSR Data Layer.

    Spawns ``msr_mcp_server_main.py`` as a child process and communicates
    over its stdin/stdout using the JSON-RPC 2.0 framing defined by the
    Model Context Protocol.
    """

    def __init__(self, server_script: str = "msr_mcp_server_main.py") -> None:
        self._server_script = server_script
        self._proc: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MSRDataLayerClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start the MCP server subprocess and perform the MCP handshake."""
        self._proc = subprocess.Popen(
            [sys.executable, self._server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # MCP initialize handshake
        self._call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})

    def disconnect(self) -> None:
        """Terminate the MCP server subprocess."""
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close()  # type: ignore[union-attr]
            self._proc.terminate()
            self._proc.wait(timeout=5)
        self._proc = None

    # ------------------------------------------------------------------
    # High-level API methods
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tools exposed by the MCP server."""
        result = self._call("tools/list")
        return result.get("tools", [])

    def get_reactor_status(self) -> dict[str, Any]:
        """Fetch the current plant operational status."""
        return self._invoke_tool("get_reactor_status")

    def get_sensor_reading(self, sensor_name: str) -> dict[str, Any]:
        """Read a single sensor value."""
        return self._invoke_tool("get_sensor_reading", {"sensor_name": sensor_name})

    def get_all_sensor_readings(self) -> dict[str, Any]:
        """Read all sensor values at once."""
        return self._invoke_tool("get_all_sensor_readings")

    def get_sensor_history(self, sensor_name: str, last_n: int = 10) -> dict[str, Any]:
        """Retrieve recent history for a sensor."""
        return self._invoke_tool(
            "get_sensor_history", {"sensor_name": sensor_name, "last_n": last_n}
        )

    def get_active_alarms(self) -> dict[str, Any]:
        """Return all active alarms."""
        return self._invoke_tool("get_active_alarms")

    def get_data_source_info(self) -> dict[str, Any]:
        """Return data source configuration and connectivity status."""
        return self._invoke_tool("get_data_source_info")

    def ingest_plant_data(
        self,
        content: str,
        data_type: str = "operational_data",
        source_id: str = "",
    ) -> dict[str, Any]:
        """
        Ingest plant operational data into the knowledge base.

        Parameters
        ----------
        content : str
            Plain text or JSON-encoded plant data record.
        data_type : str
            Category: ``"sensor_snapshot"``, ``"event_log"``,
            ``"maintenance_report"``, or ``"operational_data"``.
        source_id : str
            Optional unique identifier for de-duplication.
        """
        return self._invoke_tool(
            "ingest_plant_data",
            {"content": content, "data_type": data_type, "source_id": source_id},
        )

    # ------------------------------------------------------------------
    # Low-level JSON-RPC helpers
    # ------------------------------------------------------------------

    def _invoke_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a tool through MCP and return the parsed JSON result."""
        result = self._call(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
        )
        content = result.get("content", [])
        if content and content[0].get("type") == "text":
            return json.loads(content[0]["text"])
        return result

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the ``result`` field."""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("Client is not connected. Call connect() first.")
        with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params or {},
            }
            self._proc.stdin.write(json.dumps(request) + "\n")  # type: ignore[union-attr]
            self._proc.stdin.flush()  # type: ignore[union-attr]
            raw = self._proc.stdout.readline()  # type: ignore[union-attr]

        if not raw:
            raise MCPError("Server closed the connection unexpectedly.")

        response = json.loads(raw)
        if "error" in response:
            err = response["error"]
            raise MCPError(f"[{err.get('code')}] {err.get('message')}")
        return response.get("result", {})


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# Code written against the old "MSRDigitalTwinClient" name continues to work.
MSRDigitalTwinClient = MSRDataLayerClient  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("MSR Data Layer Client – quick demo\n")
    with MSRDataLayerClient() as client:
        print("Available tools:")
        for tool in client.list_tools():
            print(f"  • {tool['name']} – {tool['description']}")

        print("\nPlant Status:")
        status = client.get_reactor_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        print("\nCore temperature reading:")
        reading = client.get_sensor_reading("core_temperature_c")
        print(f"  {reading['sensor']}: {reading['value']} {reading['unit']}")

        print("\nData source info:")
        info = client.get_data_source_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
