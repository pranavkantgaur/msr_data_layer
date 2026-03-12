"""
MSR Data Layer - MCP Server Entry Point

Runs the MSR MCP server in stdio mode so that any MCP-compatible client
(e.g. Claude Desktop, VS Code Copilot, custom agent) can connect and use
the MSR digital twin tools.

Usage
-----
    python msr_mcp_server_main.py

The server reads JSON-RPC 2.0 messages from stdin (one per line) and
writes responses to stdout.  Diagnostic / log output goes to stderr.
"""

import sys
import logging

from msr_mcp_server import handle_message

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("msr_mcp_server_main")


def main() -> None:
    logger.info("MSR MCP Server starting (stdio transport).")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            logger.debug("← %s", line)
            response = handle_message(line)
            if response:
                print(response, flush=True)
                logger.debug("→ %s", response)
    except KeyboardInterrupt:
        logger.info("MSR MCP Server stopped by user.")
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error – server shutting down.")
        sys.exit(1)


if __name__ == "__main__":
    main()
