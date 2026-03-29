#!/usr/bin/env python3
"""
run_notebook.py

Execute ``use_cases/lucas_et_al_2025_demo.ipynb`` using ``nbconvert``,
capturing all output to ``logs/notebook.log``.

Exits with a non-zero code on failure so the GitHub Actions step is
marked failed, triggering the ``fix_agent`` step.

Environment variables
---------------------
MSR_BASE_URL        Base URL of the deployed MSR data layer HTTP server
                    (e.g. ``https://<codespace>-8000.app.github.dev``).
                    Injected into the notebook environment so every cell
                    that constructs an endpoint URL uses the right host.
                    Defaults to ``http://localhost:8000`` for local dev.
NOTEBOOK_TIMEOUT    Per-cell execution timeout in seconds (default: 120).
NOTEBOOK_RETRIES    Number of times to retry a failed execution (default: 2).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "use_cases" / "lucas_et_al_2025_demo.ipynb"
OUTPUT_PATH = REPO_ROOT / "logs" / "executed_notebook.ipynb"
LOG_FILE = REPO_ROOT / "logs" / "notebook.log"

BASE_URL = os.environ.get("MSR_BASE_URL", "http://localhost:8000")
TIMEOUT = int(os.environ.get("NOTEBOOK_TIMEOUT", "120"))
RETRIES = int(os.environ.get("NOTEBOOK_RETRIES", "2"))


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

    print(f"[run_notebook] Executing {NOTEBOOK_PATH} (base_url={BASE_URL}) …")
    try:
        _retry_call(run_once, retries=RETRIES)
        print(f"[run_notebook] ✓ Notebook executed successfully → {OUTPUT_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"[run_notebook] ✗ Notebook execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
