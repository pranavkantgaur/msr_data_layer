#!/usr/bin/env python3
"""
health_check.py

Test the MSR Data Layer HTTP API endpoints and log results to
``logs/health.log``.  Exits non-zero on failure so GitHub Actions can
trigger the ``fix_agent`` step.

Endpoints checked
-----------------
GET  /health         Service liveness check (expect 200)
POST /query          RAG query smoke-test (expect 200)
POST /data/ingest    Plant data ingest smoke-test (expect 200)

Environment variables
---------------------
MSR_BASE_URL        Base URL of the deployed server
                    (default: ``http://localhost:8000``).
MSR_API_KEY         Optional API key forwarded in the ``X-Api-Key`` header.
HC_RETRIES          Per-endpoint retry count (default: 3).
HC_TIMEOUT          Request timeout in seconds (default: 15).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "logs" / "health.log"

BASE_URL = os.environ.get("MSR_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("MSR_API_KEY", "")
RETRIES = int(os.environ.get("HC_RETRIES", "3"))
TIMEOUT = int(os.environ.get("HC_TIMEOUT", "15"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(msg)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}
    if API_KEY:
        h["X-Api-Key"] = API_KEY
    return h


def _request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """
    Perform a single HTTP request and return *(status_code, response_body)*.

    Uses the stdlib ``urllib.request`` so no extra dependencies are needed.
    """
    url = BASE_URL + path
    data: bytes | None = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body_text = exc.read().decode("utf-8")
        except Exception:  # noqa: BLE001
            body_text = ""
        return exc.code, {"error": body_text}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ConnectionError(str(exc)) from exc


def check_endpoint(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    label: str = "",
    expect_status: int = 200,
) -> bool:
    """
    Check a single endpoint with retry + exponential back-off for rate limits.

    Returns ``True`` on success, ``False`` on failure.
    """
    label = label or f"{method} {path}"
    for attempt in range(RETRIES):
        try:
            status, resp = _request(method, path, body)
            if status == 200:
                _log(f"  ✓ {label} → {status}")
                return True
            if status == 429:
                wait = 2 ** attempt
                _log(f"  ⚠ {label} → 429 rate-limited; retrying in {wait}s …")
                time.sleep(wait)
            else:
                _log(f"  ✗ {label} → unexpected status {status}")
                return False
        except ConnectionError as exc:
            wait = 2 ** attempt
            _log(f"  ✗ {label} → connection error: {exc}; retrying in {wait}s …")
            time.sleep(wait)
    _log(f"  ✗ {label} → failed after {RETRIES} attempts")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("", encoding="utf-8")  # clear previous log

    _log(f"[health_check] Target: {BASE_URL}")
    _log(f"[health_check] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    _log("")

    checks = [
        ("GET", "/health", None, "GET /health"),
        (
            "POST", "/query",
            {"question": "What is the primary loop salt composition in MSRE?"},
            "POST /query (smoke)",
        ),
        (
            "POST", "/data/ingest",
            {
                "content": "health-check probe: reactor nominal at 2024-01-15T00:00:00Z",
                "data_type": "operational_data",
                "source_id": "health-probe-auto",
            },
            "POST /data/ingest (smoke)",
        ),
    ]

    results = []
    for method, path, body, label in checks:
        ok = check_endpoint(method, path, body, label=label)
        results.append(ok)

    _log("")
    passed = sum(results)
    total = len(results)
    _log(f"[health_check] Result: {passed}/{total} checks passed")

    if not all(results):
        _log("[health_check] FAILED – one or more endpoints are unhealthy")
        sys.exit(1)

    _log("[health_check] All endpoints healthy ✓")


if __name__ == "__main__":
    main()
