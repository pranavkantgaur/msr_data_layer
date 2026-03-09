"""
MSR Digital Twin Client

A lightweight Python client that communicates with the MSR MCP server
over stdio (subprocess) and exposes a clean API for querying and
controlling the digital twin.

Example
-------
    from msr_digital_twin_client import MSRDigitalTwinClient

    with MSRDigitalTwinClient() as client:
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


class MSRDigitalTwinClient:
    """
    Subprocess-based MCP client for the MSR digital twin.

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

    def __enter__(self) -> "MSRDigitalTwinClient":
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
        """Fetch the current reactor status."""
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

    def set_control_rod_position(self, position_pct: float) -> dict[str, Any]:
        """Adjust control rod insertion depth (0-100 %)."""
        return self._invoke_tool(
            "set_control_rod_position", {"position_pct": position_pct}
        )

    def get_active_alarms(self) -> dict[str, Any]:
        """Return all active alarms."""
        return self._invoke_tool("get_active_alarms")

    def acknowledge_alarm(self, alarm_id: str) -> dict[str, Any]:
        """Acknowledge an alarm by ID."""
        return self._invoke_tool("acknowledge_alarm", {"alarm_id": alarm_id})

    def run_thermal_simulation(
        self,
        power_mw: float,
        inlet_temp_c: float,
        flow_rate_kg_s: float,
    ) -> dict[str, Any]:
        """Run the steady-state thermal-hydraulic simulation."""
        return self._invoke_tool(
            "run_thermal_simulation",
            {
                "power_mw": power_mw,
                "inlet_temp_c": inlet_temp_c,
                "flow_rate_kg_s": flow_rate_kg_s,
            },
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
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("MSR Digital Twin Client – quick demo\n")
    with MSRDigitalTwinClient() as client:
        print("Available tools:")
        for tool in client.list_tools():
            print(f"  • {tool['name']} – {tool['description']}")

        print("\nReactor Status:")
        status = client.get_reactor_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        print("\nCore temperature reading:")
        reading = client.get_sensor_reading("core_temperature_c")
        print(f"  {reading['sensor']}: {reading['value']} {reading['unit']}")

        print("\nThermal simulation (100 MW, 650 °C inlet, 250 kg/s):")
        sim = client.run_thermal_simulation(100.0, 650.0, 250.0)
        for key, value in sim.items():
            print(f"  {key}: {value}")
