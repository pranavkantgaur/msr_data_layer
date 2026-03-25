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

report_failure() {
    echo "[MSR] Server failed to start." >&2
    if [ -s "$LOG_FILE" ]; then
        echo "[MSR] Startup log:" >&2
        sed -n '1,160p' "$LOG_FILE" >&2
    else
        echo "[MSR] Startup log is empty." >&2
    fi
}

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

rm -f "$LOG_FILE"

# nohup detaches the process from the shell so it keeps running after
# postStartCommand's shell exits.  Both stdout and stderr go to the log.
nohup python server.py --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

SERVER_PID=$(cat "$PID_FILE")

# Fail fast if the child process exits immediately after launch.
sleep 1
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    rm -f "$PID_FILE"
    report_failure
    exit 1
fi

# Confirm the listener is actually serving requests before reporting success.
for _ in $(seq 1 20); do
    if curl --silent --show-error --fail http://localhost:8000/health >/dev/null 2>&1; then
        echo "[MSR] Data Layer server started (PID=$SERVER_PID)"
        echo "[MSR] Log: $LOG_FILE"
        echo "[MSR] To check status: curl http://localhost:8000/health"
        exit 0
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        report_failure
        exit 1
    fi

    sleep 0.5
done

kill "$SERVER_PID" 2>/dev/null || true
rm -f "$PID_FILE"
report_failure
echo "[MSR] Health check did not succeed within 10 seconds." >&2
exit 1
