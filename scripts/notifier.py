#!/usr/bin/env python3
"""
notifier.py

Send a status notification after each agent-loop run via:
- Email (``smtplib``, Gmail-compatible STARTTLS)
- WhatsApp (Twilio API)

Both channels are optional — if the relevant environment variables are not
set the channel is silently skipped.

Environment variables
---------------------
EMAIL_USER          Gmail sender address
EMAIL_PASS          Gmail App Password (not your account password; see
                    https://support.google.com/accounts/answer/185833)
EMAIL_TO            Recipient address(es), comma-separated
TWILIO_SID          Twilio Account SID
TWILIO_TOKEN        Twilio Auth Token
WHATSAPP_TO         Destination WhatsApp number in Twilio format,
                    e.g. ``whatsapp:+91XXXXXXXXXX``
GITHUB_JOB_STATUS   Set by GitHub Actions to ``success`` or ``failure``
                    (populated via ``env: GITHUB_JOB_STATUS: ${{ job.status }}``)
GITHUB_RUN_ID       GitHub Actions run ID (included in the message)
GITHUB_REPOSITORY   ``owner/repo`` (included in the message)
"""

from __future__ import annotations

import os
import smtplib
import sys
from email.mime.text import MIMEText
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILES = [
    REPO_ROOT / "logs" / "notebook.log",
    REPO_ROOT / "logs" / "health.log",
]
MAX_LOG_PREVIEW = 2_000

_JOB_STATUS = os.environ.get("GITHUB_JOB_STATUS", "unknown")
_RUN_ID = os.environ.get("GITHUB_RUN_ID", "")
_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "pranavkantgaur/msr_data_layer")

_EMAIL_USER = os.environ.get("EMAIL_USER", "")
_EMAIL_PASS = os.environ.get("EMAIL_PASS", "")
_EMAIL_TO = os.environ.get("EMAIL_TO", "")
_TWILIO_SID = os.environ.get("TWILIO_SID", "")
_TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
_WHATSAPP_TO = os.environ.get("WHATSAPP_TO", "")

# Twilio sandbox number — standard for WhatsApp sandbox
_TWILIO_FROM = "whatsapp:+14155238886"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_emoji() -> str:
    return "✅" if _JOB_STATUS == "success" else "❌"


def _build_subject() -> str:
    return f"MSR Agent Loop {_status_emoji()} {_JOB_STATUS.upper()} – run {_RUN_ID}"


def _build_body() -> str:
    lines = [
        f"MSR Autonomous Agent Loop — {_status_emoji()} {_JOB_STATUS.upper()}",
        "",
        f"Repository : {_REPOSITORY}",
        f"Run ID     : {_RUN_ID}",
        f"Status     : {_JOB_STATUS}",
        "",
    ]
    if _RUN_ID:
        lines.append(
            f"Details    : https://github.com/{_REPOSITORY}/actions/runs/{_RUN_ID}"
        )
        lines.append("")

    for log_path in LOG_FILES:
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace")
            preview = content[:MAX_LOG_PREVIEW]
            lines += [
                f"--- {log_path.name} ---",
                preview,
                "",
            ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Email via smtplib
# ---------------------------------------------------------------------------

def send_email() -> None:
    """Send a status email via Gmail SMTP (STARTTLS)."""
    if not (_EMAIL_USER and _EMAIL_PASS and _EMAIL_TO):
        print("[notifier] Email not configured — skipping (set EMAIL_USER, EMAIL_PASS, EMAIL_TO).")
        return

    subject = _build_subject()
    body = _build_body()
    recipients = [r.strip() for r in _EMAIL_TO.split(",") if r.strip()]

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = _EMAIL_USER
    msg["To"] = ", ".join(recipients)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(_EMAIL_USER, _EMAIL_PASS)
            server.sendmail(_EMAIL_USER, recipients, msg.as_string())
        print(f"[notifier] ✓ Email sent to {_EMAIL_TO}")
    except Exception as exc:  # noqa: BLE001
        print(f"[notifier] ✗ Email failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# WhatsApp via Twilio
# ---------------------------------------------------------------------------

def send_whatsapp() -> None:
    """Send a status message via the Twilio WhatsApp API."""
    if not (_TWILIO_SID and _TWILIO_TOKEN and _WHATSAPP_TO):
        print(
            "[notifier] WhatsApp not configured — skipping "
            "(set TWILIO_SID, TWILIO_TOKEN, WHATSAPP_TO)."
        )
        return

    try:
        from twilio.rest import Client  # noqa: PLC0415

        body = f"MSR Agent {_status_emoji()} {_JOB_STATUS.upper()}"
        if _RUN_ID:
            body += f"\nRun: https://github.com/{_REPOSITORY}/actions/runs/{_RUN_ID}"

        client = Client(_TWILIO_SID, _TWILIO_TOKEN)
        message = client.messages.create(
            body=body,
            from_=_TWILIO_FROM,
            to=_WHATSAPP_TO,
        )
        print(f"[notifier] ✓ WhatsApp sent (SID: {message.sid})")
    except ImportError:
        print(
            "[notifier] ⚠ Twilio library not installed — install with "
            "'pip install twilio' or add to requirements_agent.txt.",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[notifier] ✗ WhatsApp failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[notifier] Status: {_JOB_STATUS}")
    send_email()
    send_whatsapp()
    print("[notifier] Done.")


if __name__ == "__main__":
    main()
