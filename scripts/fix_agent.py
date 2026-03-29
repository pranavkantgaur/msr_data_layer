#!/usr/bin/env python3
"""
fix_agent.py  (safe PR mode)

Reads the captured log files, calls an LLM to propose a minimal fix, and
opens a GitHub pull request with the patch applied.  **Never commits
directly to ``main``** – always creates a fresh ``auto-fix/<timestamp>``
branch.

Safety constraints (matching the conversation design)
------------------------------------------------------
- Only modifies files outside ``auth/``, ``secrets/``, and infrastructure
  configs (``template.yaml``, ``samconfig.toml``, ``Dockerfile*``).
- Requires ``GH_TOKEN`` (GitHub Actions ``GITHUB_TOKEN``) to open the PR
  via the GitHub CLI (``gh``).
- If no LLM key is available the agent logs a diagnostic and exits 0 so
  the notification step still runs.

Environment variables
---------------------
OPENAI_API_KEY / MSR_OPENAI_API_KEY / MSR_GITHUB_TOKEN
    LLM credentials.  ``OPENAI_API_KEY`` / ``MSR_OPENAI_API_KEY`` take
    precedence; ``MSR_GITHUB_TOKEN`` falls back to GitHub Models API.
MSR_OPENAI_BASE_URL
    LLM base URL (default: ``https://api.openai.com/v1`` when using
    OpenAI key, ``https://models.inference.ai.azure.com`` for GitHub
    Models).
MSR_OPENAI_MODEL
    Model override (default: ``gpt-4o-mini``).
GH_TOKEN
    GitHub token with ``repo`` + ``pull-requests: write`` scope.
    Automatically set inside GitHub Actions via ``${{ secrets.GITHUB_TOKEN }}``.
GITHUB_REPOSITORY
    ``owner/repo`` string injected by GitHub Actions (e.g.
    ``pranavkantgaur/msr_data_layer``).
"""

from __future__ import annotations

import json
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
LOG_FILES = [
    REPO_ROOT / "logs" / "notebook.log",
    REPO_ROOT / "logs" / "health.log",
]
MAX_LOG_CHARS = 8_000  # truncate to avoid huge prompts

# LLM configuration – mirror the same priority order as the RAG pipeline
_OPENAI_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("MSR_OPENAI_API_KEY")
    or ""
)
_GITHUB_TOKEN = os.environ.get("MSR_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""

if _OPENAI_KEY:
    _LLM_KEY = _OPENAI_KEY
    _LLM_BASE_URL = os.environ.get(
        "MSR_OPENAI_BASE_URL", "https://api.openai.com/v1"
    )
else:
    _LLM_KEY = _GITHUB_TOKEN
    _LLM_BASE_URL = os.environ.get(
        "MSR_OPENAI_BASE_URL", "https://models.inference.ai.azure.com"
    )

_LLM_MODEL = os.environ.get("MSR_OPENAI_MODEL", "gpt-4o-mini")
_GH_TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
_GH_REPO = os.environ.get("GITHUB_REPOSITORY", "pranavkantgaur/msr_data_layer")

# Files the agent is NOT allowed to patch (safety constraint)
_PROTECTED_PATTERNS = (
    "samconfig.toml",
    "Dockerfile",
    "template.yaml",
    ".env",
    "secrets",
    "auth/",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_logs() -> str:
    """Return the concatenated content of all log files (truncated)."""
    parts: list[str] = []
    for path in LOG_FILES:
        if path.exists():
            parts.append(f"=== {path.name} ===\n{path.read_text(encoding='utf-8', errors='replace')}")
    combined = "\n".join(parts)
    return combined[:MAX_LOG_CHARS]


def _llm_chat(messages: list[dict], max_tokens: int = 1024) -> str:
    """
    Call the chat-completions API and return the assistant content.

    Uses ``urllib.request`` (stdlib) to avoid requiring the ``openai``
    package in CI when it is not installed.
    """
    payload = {
        "model": _LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    url = _LLM_BASE_URL.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_LLM_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"].strip()


def propose_fix(logs: str) -> str:
    """Ask the LLM for a minimal fix and return its response."""
    system = (
        "You are an expert Python developer working on the MSR Data Layer "
        "(https://github.com/pranavkantgaur/msr_data_layer).\n"
        "The repository is a READ-ONLY data layer for Molten Salt Reactors — "
        "it has NO simulation, control, or actuation code.\n\n"
        "Your task: analyse the failure logs and propose a MINIMAL patch in "
        "unified diff (git diff) format to fix the root cause.\n\n"
        "Constraints:\n"
        "- Only modify Python source files in the repo root or scripts/.\n"
        "- Do NOT touch auth, secrets, infrastructure configs, or Dockerfiles.\n"
        "- Do NOT add control/actuation tools to the MCP server.\n"
        "- Output ONLY the git diff patch — no prose explanation."
    )
    user = (
        "The CI pipeline failed.  Logs:\n\n"
        f"{logs}\n\n"
        "Produce a minimal unified diff that fixes the root cause."
    )
    return _llm_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=1024,
    )


def _is_safe_patch(patch: str) -> bool:
    """Return ``False`` if the patch touches protected paths."""
    for pattern in _PROTECTED_PATTERNS:
        if pattern in patch:
            print(
                f"[fix_agent] ⚠ Patch touches protected path '{pattern}' — refusing.",
                file=sys.stderr,
            )
            return False
    return True


def create_pr(branch: str, patch: str, logs_summary: str) -> str:
    """
    Apply the patch, push to a new branch, and open a PR via ``gh``.

    Returns the PR URL on success or an error message on failure.
    """
    # Write the patch to a temp file
    patch_file = REPO_ROOT / "fix.patch"
    patch_file.write_text(patch, encoding="utf-8")

    try:
        # Create the branch
        subprocess.run(
            ["git", "checkout", "-b", branch],
            check=True, cwd=REPO_ROOT,
        )

        # Try to apply the patch (best-effort — may not apply cleanly)
        apply = subprocess.run(
            ["git", "apply", "--ignore-whitespace", str(patch_file)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if apply.returncode != 0:
            print(
                f"[fix_agent] ⚠ git apply failed:\n{apply.stderr}",
                file=sys.stderr,
            )

        # Stage all changes (including patch file removal)
        patch_file.unlink(missing_ok=True)
        subprocess.run(["git", "add", "--all"], check=True, cwd=REPO_ROOT)

        # Only commit if there are staged changes
        diff = subprocess.run(
            ["git", "diff", "--cached", "--stat"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if not diff.stdout.strip():
            print("[fix_agent] ⚠ No changes to commit after applying patch.")
            return "no-changes"

        subprocess.run(
            ["git", "commit", "-m", f"auto-fix: agent patch for CI failure\n\n{logs_summary[:500]}"],
            check=True, cwd=REPO_ROOT,
        )
        subprocess.run(
            ["git", "push", "origin", branch],
            check=True, cwd=REPO_ROOT,
        )

        # Open the PR via GitHub CLI
        pr_result = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", "auto-fix: agent-proposed patch for CI failure",
                "--body", (
                    "## Autonomous Agent Fix\n\n"
                    "This PR was created automatically by `scripts/fix_agent.py` "
                    "after a CI failure.\n\n"
                    "**Logs summary:**\n```\n"
                    + logs_summary[:2000]
                    + "\n```\n\n"
                    "**Proposed patch:**\n```diff\n"
                    + patch[:3000]
                    + "\n```\n\n"
                    "> ⚠️ Review carefully before merging."
                ),
                "--base", "main",
                "--head", branch,
                "--repo", _GH_REPO,
            ],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if pr_result.returncode != 0:
            print(f"[fix_agent] ✗ gh pr create failed:\n{pr_result.stderr}", file=sys.stderr)
            return f"pr-failed: {pr_result.stderr[:200]}"

        pr_url = pr_result.stdout.strip()
        print(f"[fix_agent] ✓ PR created: {pr_url}")
        return pr_url

    finally:
        patch_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not _LLM_KEY:
        print(
            "[fix_agent] No LLM key available "
            "(set OPENAI_API_KEY, MSR_OPENAI_API_KEY, or MSR_GITHUB_TOKEN). "
            "Skipping fix agent.",
            file=sys.stderr,
        )
        # Exit 0 so the notification step still runs
        sys.exit(0)

    print("[fix_agent] Reading logs …")
    logs = read_logs()
    if not logs.strip():
        print("[fix_agent] No logs found — nothing to fix.")
        sys.exit(0)

    print(f"[fix_agent] Asking {_LLM_MODEL} for a fix …")
    try:
        patch = propose_fix(logs)
    except Exception as exc:  # noqa: BLE001
        print(f"[fix_agent] ✗ LLM call failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not patch.strip():
        print("[fix_agent] LLM returned an empty patch — no action taken.")
        sys.exit(0)

    print(f"[fix_agent] Received patch ({len(patch)} chars).")

    if not _is_safe_patch(patch):
        print("[fix_agent] Patch rejected (touches protected paths).", file=sys.stderr)
        sys.exit(1)

    if not _GH_TOKEN:
        print("[fix_agent] GH_TOKEN not set — cannot create PR.", file=sys.stderr)
        # Write the patch to disk so a human can apply it
        patch_out = REPO_ROOT / "logs" / "proposed_fix.patch"
        patch_out.parent.mkdir(parents=True, exist_ok=True)
        patch_out.write_text(patch, encoding="utf-8")
        print(f"[fix_agent] Patch saved to {patch_out}")
        sys.exit(0)

    branch = f"auto-fix/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    logs_summary = logs[:500]
    create_pr(branch, patch, logs_summary)


if __name__ == "__main__":
    main()
