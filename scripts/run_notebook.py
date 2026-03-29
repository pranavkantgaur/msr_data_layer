#!/usr/bin/env python3
"""
run_notebook.py

Optionally start the MSR Data Layer HTTP server (``server.py``), then
execute ``use_cases/lucas_et_al_2025_demo.ipynb`` using ``nbconvert``,
capturing all output to ``logs/notebook.log``.

Exits with a non-zero code on failure so the GitHub Actions step is
marked failed, triggering the ``fix_agent`` step.

Environment variables
---------------------
MSR_BASE_URL        Base URL of the MSR Data Layer HTTP server.
                    When set to a non-localhost URL (e.g. a Codespace URL)
                    the server is assumed to be already running.
                    When set to ``http://localhost:8000`` (default) or not
                    set, this script starts ``server.py`` automatically and
                    stops it after notebook execution.
MSR_AUTOSTART_SERVER
                    Set to ``false`` to skip the auto-start even on localhost
                    (useful when you have started the server separately).
MSR_GITHUB_TOKEN    GitHub token forwarded to the server for LLM-backed RAG
                    (GitHub Models API).
MSR_KB_DIR          Persistent KB directory injected into the server process.
NOTEBOOK_TIMEOUT    Per-cell execution timeout in seconds (default: 300).
NOTEBOOK_RETRIES    Number of times to retry a failed execution (default: 2).
SERVER_READY_TIMEOUT
                    Seconds to wait for the server to become healthy (default: 60).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "use_cases" / "lucas_et_al_2025_demo.ipynb"
OUTPUT_PATH = REPO_ROOT / "logs" / "executed_notebook.ipynb"
LOG_FILE = REPO_ROOT / "logs" / "notebook.log"
SERVER_LOG = REPO_ROOT / "logs" / "server.log"

BASE_URL = os.environ.get("MSR_BASE_URL", "http://localhost:8000")
TIMEOUT = int(os.environ.get("NOTEBOOK_TIMEOUT", "300"))
RETRIES = int(os.environ.get("NOTEBOOK_RETRIES", "2"))
SERVER_READY_TIMEOUT = int(os.environ.get("SERVER_READY_TIMEOUT", "60"))

# Auto-start the server only when targeting localhost
_autostart_default = BASE_URL.startswith("http://localhost") or BASE_URL.startswith("http://127.0.0.1")
AUTOSTART_SERVER = os.environ.get("MSR_AUTOSTART_SERVER", "true" if _autostart_default else "false").lower() not in (
    "false", "0", "no"
)


# ---------------------------------------------------------------------------
# Server management helpers
# ---------------------------------------------------------------------------

def _build_server_env() -> dict[str, str]:
    """Build the environment dict for the server subprocess."""
    env = dict(os.environ)
    # Forward the GitHub token for LLM-backed RAG
    for var in ("MSR_GITHUB_TOKEN", "GITHUB_TOKEN"):
        if env.get(var):
            env.setdefault("MSR_GITHUB_TOKEN", env[var])
            break
    # Use an ephemeral KB dir in CI if not already set
    env.setdefault("MSR_KB_DIR", str(REPO_ROOT / "logs" / "_kb_store"))
    return env


def _wait_for_server(url: str, timeout: int) -> bool:
    """
    Poll ``GET /health`` until it returns 200 or *timeout* seconds pass.

    Returns ``True`` on success.
    """
    health_url = url.rstrip("/") + "/health"
    deadline = time.monotonic() + timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[run_notebook] ✓ Server ready after {attempt} poll(s) on {health_url}")
                    return True
        except (urllib.error.URLError, OSError, ConnectionRefusedError):
            pass
        time.sleep(1)
    print(f"[run_notebook] ✗ Server not ready after {timeout}s on {health_url}", file=sys.stderr)
    return False


def start_server() -> "subprocess.Popen[bytes] | None":
    """
    Start ``server.py`` in the background.

    Returns the ``Popen`` object on success, or ``None`` if ``server.py``
    is not found or the server doesn't become healthy in time.
    """
    server_script = REPO_ROOT / "server.py"
    if not server_script.exists():
        print(f"[run_notebook] ⚠ {server_script} not found — cannot auto-start server.", file=sys.stderr)
        return None

    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = _build_server_env()
    print(f"[run_notebook] Starting server (log → {SERVER_LOG}) …")
    with SERVER_LOG.open("wb") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            stdout=log_fh, stderr=log_fh,
            env=env, cwd=str(REPO_ROOT),
        )

    if not _wait_for_server(BASE_URL, SERVER_READY_TIMEOUT):
        proc.terminate()
        return None
    return proc


def stop_server(proc: "subprocess.Popen[bytes]") -> None:
    """Gracefully stop the server process."""
    try:
        proc.terminate()
        proc.wait(timeout=10)
        print("[run_notebook] Server stopped.")
    except Exception as exc:  # noqa: BLE001
        print(f"[run_notebook] ⚠ Error stopping server: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def _retry_call(fn, retries: int = RETRIES) -> None:
    """Call *fn()* up to *retries* times, backing off on rate-limit errors."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            fn()
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg:
                wait = 2 ** attempt
                print(
                    f"[run_notebook] Rate limit hit on attempt {attempt + 1}; "
                    f"sleeping {wait}s …",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                raise
    raise last_exc  # type: ignore[misc]


def run_once() -> None:
    """Execute the notebook once with nbconvert."""
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=" + str(TIMEOUT),
        "--output", str(OUTPUT_PATH),
        str(NOTEBOOK_PATH),
    ]
    env = dict(os.environ)
    env["MSR_BASE_URL"] = BASE_URL

    with LOG_FILE.open("w", encoding="utf-8") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=fh, env=env)

    if result.returncode != 0:
        raise RuntimeError(
            f"nbconvert exited with code {result.returncode}. "
            f"See {LOG_FILE} for details."
        )


def main() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not NOTEBOOK_PATH.exists():
        print(
            f"[run_notebook] ERROR: notebook not found: {NOTEBOOK_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Auto-start server if targeting localhost ──────────────────────────
    server_proc = None
    if AUTOSTART_SERVER:
        server_proc = start_server()
        if server_proc is None:
            print("[run_notebook] ✗ Server failed to start.", file=sys.stderr)
            sys.exit(1)

    try:
        print(f"[run_notebook] Executing {NOTEBOOK_PATH} (base_url={BASE_URL}) …")
        try:
            _retry_call(run_once, retries=RETRIES)
            print(f"[run_notebook] ✓ Notebook executed successfully → {OUTPUT_PATH}")
        except Exception as exc:  # noqa: BLE001
            print(f"[run_notebook] ✗ Notebook execution failed: {exc}", file=sys.stderr)
            sys.exit(1)
    finally:
        if server_proc is not None:
            stop_server(server_proc)


if __name__ == "__main__":
    main()
