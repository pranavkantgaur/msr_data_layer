"""
MSR Data Layer – Standalone HTTP Server (Primary Deployment)

This is the **primary HTTP server** for the MSR Data Layer.  It wraps
``lambda_function.lambda_handler`` as a plain HTTP server so the service
can run inside GitHub Codespaces (or any host) without AWS infrastructure.

Architecture overview
---------------------
The MSR Data Layer exposes two independent interfaces:

1. **This HTTP server** (``server.py``) – for human operators, demo notebooks,
   and any client that speaks plain HTTP.  Start with ``python server.py`` or
   ``make serve``.  In a GitHub Codespace the port is automatically made public
   so you get a shareable ``https://<codespace>-8000.app.github.dev`` URL.

2. **stdio MCP server** (``msr_mcp_server_main.py``) – for AI agents such as
   GitHub Copilot Chat, Claude, and custom MCP-compatible agents.  Start with
   ``python msr_mcp_server_main.py``.  No port needed; communication is over
   stdin/stdout.

Both interfaces share the same underlying KB and timeseries store.

Primary deployment: GitHub Codespaces
--------------------------------------
1. Open the repository in a GitHub Codespace.
2. The ``.devcontainer/devcontainer.json`` installs dependencies and starts
   this server automatically on port 8000 with *public* visibility.
3. The public URL (e.g. ``https://<codespace-name>-8000.app.github.dev``) is
   printed in the Codespace's *Ports* panel and in ``/tmp/msr_server.log``.
4. Paste that URL into the demo notebook or share it with reviewers.

The ``GITHUB_TOKEN`` injected automatically by Codespaces is forwarded to
the RAG pipeline as ``MSR_GITHUB_TOKEN``, enabling GitHub Models API
(gpt-4o-mini / text-embedding-3-small) **at no extra cost** with a Copilot
Pro subscription.

Local usage
-----------
    python server.py                    # default: 0.0.0.0:8000
    python server.py --port 9000        # custom port
    python server.py --host 127.0.0.1   # localhost only

    # or via Makefile:
    make serve                          # same as python server.py

All environment variables understood by ``lambda_function.py`` are respected
(``MSR_GITHUB_TOKEN``, ``MSR_API_KEY``, ``MSR_PLANT_DATA_URL``, etc.).

Endpoints
---------
GET  /health                – liveness check
POST /mcp                   – JSON-RPC 2.0 / MCP protocol
POST /query                 – plain-text RAG query
POST /research/deep         – deep research agent endpoint (expanded RAG + citations)
POST /kb/update             – trigger KB ingestion (archive + OpenAlex + arXiv + S2)
POST /data/ingest           – push plant operational data (text/event logs)
POST /timeseries/ingest     – push timestamped sensor readings (structured)
POST /timeseries/query      – query sensor timeseries (structured or natural language)

Optional: AWS deployment
------------------------
``lambda_function.py`` also serves as an AWS Lambda handler for those who
need cloud-hosted deployment.  See ``template.yaml`` and ``make deploy-guided``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: forward GITHUB_TOKEN → MSR_GITHUB_TOKEN when running in
# GitHub Codespaces (or any environment where GITHUB_TOKEN is set).
# ---------------------------------------------------------------------------

if os.environ.get("GITHUB_TOKEN") and not os.environ.get("MSR_GITHUB_TOKEN"):
    os.environ["MSR_GITHUB_TOKEN"] = os.environ["GITHUB_TOKEN"]
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Forwarded GITHUB_TOKEN → MSR_GITHUB_TOKEN (GitHub Models API)."
    )

# Import the Lambda handler *after* the env-var bootstrap so that the RAG
# module picks up MSR_GITHUB_TOKEN during its module-level initialisation.
from lambda_function import lambda_handler  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS helper
# ---------------------------------------------------------------------------

_CORS_HEADERS: dict[str, str] = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Api-Key, Authorization",
}


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _MSRHandler(BaseHTTPRequestHandler):
    """Translate raw HTTP requests into Lambda events and write the response."""

    # Silence the default access-log line (Lambda handler logs its own info)
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: ANN401
        logger.info(fmt, *args)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _build_lambda_event(self, method: str, path: str, body: bytes) -> dict[str, Any]:
        """Convert an HTTP request into an API-Gateway v2 style Lambda event."""
        # Collect all request headers
        headers: dict[str, str] = {
            k.lower(): v for k, v in self.headers.items()
        }
        return {
            "requestContext": {"http": {"method": method}},
            "rawPath": path,
            "headers": headers,
            "body": body.decode("utf-8", errors="replace") if body else None,
            "isBase64Encoded": False,
        }

    def _send_lambda_response(self, result: dict[str, Any]) -> None:
        status = int(result.get("statusCode", 200))
        body: str = result.get("body", "")

        self.send_response(status)

        # Content-Type from Lambda response (fallback to JSON)
        lam_headers: dict[str, str] = result.get("headers", {})
        content_type = lam_headers.get(
            "Content-Type", lam_headers.get("content-type", "application/json")
        )
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))

        # CORS headers
        for k, v in _CORS_HEADERS.items():
            self.send_header(k, v)

        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle preflight CORS requests."""
        self.send_response(204)
        for k, v in _CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def _handle_request(self, method: str) -> None:
        path = self.path.split("?")[0]  # strip query-string
        body = self._read_body()
        event = self._build_lambda_event(method, path, body)
        try:
            result = lambda_handler(event, None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in lambda_handler")
            result = {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(exc)}),
            }
        self._send_lambda_response(result)

    def do_GET(self) -> None:  # noqa: N802
        self._handle_request("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle_request("POST")


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MSR Data Layer – standalone HTTP server"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MSR_SERVER_HOST", "0.0.0.0"),
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MSR_SERVER_PORT", "8000")),
        help="TCP port to listen on (default: 8000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _parse_args(argv)
    server = HTTPServer((args.host, args.port), _MSRHandler)
    logger.info(
        "MSR Data Layer HTTP server listening on http://%s:%d",
        args.host,
        args.port,
    )
    logger.info(
        "Endpoints: GET /health  POST /mcp  POST /query  POST /research/deep  "
        "POST /kb/update  POST /data/ingest  "
        "POST /timeseries/ingest  POST /timeseries/query"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
