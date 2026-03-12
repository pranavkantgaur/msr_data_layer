#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MSR Data Layer – Codespaces server startup script
#
# Called by devcontainer.json postStartCommand every time the Codespace
# starts or resumes.
#
# Uses nohup so the server process is immune to SIGHUP when the
# postStartCommand shell exits (plain "&" would kill the child).
#
# Logs  → /tmp/msr_server.log
# PID   → /tmp/msr_server.pid
# ---------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE=/tmp/msr_server.log
PID_FILE=/tmp/msr_server.pid

# Stop any previously running instance (e.g. after a Codespace resume)
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[MSR] Stopping previous server (PID=$OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

cd "$WORKDIR"

# nohup detaches the process from the shell so it keeps running after
# postStartCommand's shell exits.  Both stdout and stderr go to the log.
nohup python server.py --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "[MSR] Data Layer server started (PID=$(cat $PID_FILE))"
echo "[MSR] Log: $LOG_FILE"
echo "[MSR] To check status: curl http://localhost:8000/health"
