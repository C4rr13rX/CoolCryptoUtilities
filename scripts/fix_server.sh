#!/usr/bin/env bash
# Attempt to heal common Django/PostgreSQL issues automatically.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/runtime}"
LOG_FILE="$LOG_DIR/fix_server_$(date +'%Y-%m-%d_%H-%M-%S').log"
PID_FILE="$REPO_ROOT/runtime/server.pid"
RUN_LOG="$REPO_ROOT/runtime/server.log"
SERVER_PORT="${SERVER_PORT:-9000}"

mkdir -p "$LOG_DIR"

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

header() {
  log ""
  log "============================================================"
  log "$@"
  log "============================================================"
}

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "[ERROR] Missing command: $1"
    exit 1
  fi
}

cd "$REPO_ROOT"
require python3
require bash

if [ ! -d "$REPO_ROOT/bin" ]; then
  log "[ERROR] Virtualenv not found at $REPO_ROOT/bin. Activate your environment first."
  exit 1
fi

header "[0] Loading environment"
set +u
# shellcheck disable=SC1091
source "$REPO_ROOT/bin/activate"
set -u
if [ -f "$REPO_ROOT/.env.postgres" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env.postgres"
  set +a
fi

header "[1] Authenticating sudo"
SUDO_AVAILABLE=0
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    SUDO_AVAILABLE=1
    log "sudo access confirmed."
  elif sudo -v; then
    SUDO_AVAILABLE=1
    log "sudo session refreshed."
  else
    log "[WARN] sudo unavailable; continuing without service management."
  fi
else
  log "[WARN] sudo command not present."
fi

header "[2] Ensuring PostgreSQL is running"
if (( SUDO_AVAILABLE )); then
  sudo service postgresql start | tee -a "$LOG_FILE"
else
  log "[WARN] Skipping 'service postgresql start' (sudo unavailable)."
fi

header "[3] Applying migrations"
python3 web/manage.py migrate --noinput | tee -a "$LOG_FILE"

header "[4] Django system check"
python3 web/manage.py check | tee -a "$LOG_FILE"

header "[5] Restarting Django runserver"
# attempt to free server port if another process is bound
EXISTING_PID=$(pgrep -f "runserver .*:$SERVER_PORT" || true)
if [ -n "$EXISTING_PID" ]; then
  log "Port $SERVER_PORT currently used by runserver PID(s): $EXISTING_PID (terminating)."
  kill $EXISTING_PID || true
  sleep 1
fi
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  log "Stopping previous runserver (PID $(cat "$PID_FILE"))."
  kill "$(cat "$PID_FILE")" || true
  sleep 1
fi
if pgrep -f "web/manage.py runserver" >/dev/null; then
  log "Killing stray runserver processes."
  pkill -f "web/manage.py runserver" || true
  sleep 1
fi
log "Starting new runserver on port $SERVER_PORT (output -> $RUN_LOG)."
nohup python3 web/manage.py runserver 0.0.0.0:$SERVER_PORT --guardian-off >"$RUN_LOG" 2>&1 &
RUN_PID=$!
echo "$RUN_PID" > "$PID_FILE"
log "runserver PID: $RUN_PID"

header "[6] Running diagnostics"
SERVER_PORT="$SERVER_PORT" bash scripts/diagnose_server.sh | tee -a "$LOG_FILE"

if ! curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:$SERVER_PORT/ | grep -q "200\|302"; then
  log "[WARN] HTTP probe still failing; showing last 20 lines of runserver log"
  tail -n 20 "$RUN_LOG" | tee -a "$LOG_FILE"
fi

header "[7] Completed"
log "Fix script finished. Tail $RUN_LOG for live server output."
