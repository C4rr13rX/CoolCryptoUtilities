#!/usr/bin/env bash
# Diagnose Django/PostgreSQL connectivity issues and emit a report.

set -euo pipefail

REPORT_DIR="${REPORT_DIR:-./runtime/reports}"
SERVER_PORT="${SERVER_PORT:-8000}"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
REPORT_PATH="$REPORT_DIR/server_diag_$TIMESTAMP.log"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="$REPO_ROOT/bin"

mkdir -p "$REPORT_DIR"

if [ -f "$REPO_ROOT/.env.postgres" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env.postgres"
  set +a
fi

PORT_STATUS="unknown"
HTTP_STATUS="unknown"
PG_STATUS="unknown"
PG_MESSAGE=""
SOCKET_STATUS="unknown"
RECOMMENDATIONS=()

add_recommendation() {
  RECOMMENDATIONS+=("$1")
}

log() {
  echo "$@" | tee -a "$REPORT_PATH"
}

header() {
  log ""
  log "============================================================"
  log "$@"
  log "============================================================"
}


require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "[ERROR] missing dependency: $1"
    exit 1
  fi
}

require_tool "python3"
require_tool "curl"

header "[0] Capturing environment snapshot..."
log "Report: $REPORT_PATH"
log "Generated: $(date)"
log "Host: $(hostname)"
log "Kernel: $(uname -a)"
log "User: $USER"

SUDO_AVAILABLE=0
header "[1] Checking sudo availability..."
if command -v sudo >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null; then
    SUDO_AVAILABLE=1
    log "sudo access confirmed."
  elif sudo -v; then
    SUDO_AVAILABLE=1
    log "sudo session refreshed."
  else
    log "[WARN] sudo unavailable in this environment (no-new-privileges?). Continuing without elevated checks."
  fi
else
  log "[WARN] sudo command not present; skipping privileged checks."
fi

header "[2] Checking system services"
if (( SUDO_AVAILABLE )); then
  sudo systemctl status postgresql --no-pager | tee -a "$REPORT_PATH" || true
  sudo systemctl status nginx --no-pager | tee -a "$REPORT_PATH" || true
else
  log "[WARN] Skipping service status (sudo unavailable)."
fi

header "[3] Inspecting listening ports"
PORT_OUTPUT=""
if command -v ss >/dev/null 2>&1; then
  PORT_OUTPUT="$(ss -ltnp 2>&1)"
else
  PORT_OUTPUT="$(netstat -ltnp 2>&1)"
fi
echo "$PORT_OUTPUT" | tee -a "$REPORT_PATH"
if echo "$PORT_OUTPUT" | grep -q ":$SERVER_PORT"; then
  PORT_STATUS="listening"
else
  PORT_STATUS="not-listening"
  add_recommendation "Port $SERVER_PORT is closed. Start Django via 'SERVER_PORT=$SERVER_PORT python3 web/manage.py runserver 0.0.0.0:$SERVER_PORT --guardian-off'."
fi

header "[4] Verifying PostgreSQL socket"
PSQL_EXIT=0
if (( SUDO_AVAILABLE )); then
  if ! sudo -u postgres psql -Atc "SELECT 'psql_ok';" >/dev/null 2>&1; then
    log "[WARN] Unable to run psql as postgres."
  else
    sudo -u postgres psql -Atc "SELECT datname FROM pg_database;" | tee -a "$REPORT_PATH"
  fi
else
  if [ -n "${POSTGRES_PASSWORD:-}" ]; then
    set +e
    PGPASSWORD="$POSTGRES_PASSWORD" psql -h "${POSTGRES_HOST:-127.0.0.1}" -U "${POSTGRES_USER:-coolcrypto}" -d "${POSTGRES_DB:-coolcrypto}" -c "SELECT current_database();" 2>&1 | tee -a "$REPORT_PATH"
    PSQL_EXIT=$?
    set -e
    if [ $PSQL_EXIT -ne 0 ]; then
      log "[WARN] direct psql probe failed (exit $PSQL_EXIT)."
    fi
  else
    log "[WARN] Skipping psql probe (sudo unavailable and POSTGRES_PASSWORD unset)."
  fi
fi

header "[4b] Checking raw socket connectivity"
SOCKET_TEST_JSON=$("$VENV_BIN/python3" - <<'PYTHON'
import json
import os
import socket

host = os.getenv("POSTGRES_HOST", "127.0.0.1")
port = int(os.getenv("POSTGRES_PORT", "5432"))
result = {"host": host, "port": port}

def print_json(payload):
    print(json.dumps(payload))

if host.startswith("/"):
    target = host if host.endswith(".s.PGSQL.5432") else f"{host}/.s.PGSQL.{port}"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(target)
    except OSError as exc:
        result.update({"status": "FAIL", "errno": exc.errno, "message": str(exc)})
    else:
        result["status"] = "OK"
    finally:
        sock.close()
else:
    try:
        socket.create_connection((host, port), timeout=5).close()
    except OSError as exc:
        result.update({"status": "FAIL", "errno": exc.errno, "message": str(exc)})
    else:
        result["status"] = "OK"

print_json(result)
PYTHON
)
echo "$SOCKET_TEST_JSON" | tee -a "$REPORT_PATH"
SOCKET_STATUS=$(python3 -c 'import json,sys; data=json.loads(sys.argv[1]); print(data.get("status","unknown"))' "$SOCKET_TEST_JSON")
if [ "$SOCKET_STATUS" != "OK" ]; then
  SOCKET_ERR=$(python3 -c 'import json,sys; data=json.loads(sys.argv[1]); print(data.get("message",""))' "$SOCKET_TEST_JSON")
  add_recommendation "Low-level socket probe failed (${SOCKET_ERR}). Check firewall/sandbox restrictions that may block localhost connections."
fi

header "[5] Validating Django settings"
pushd "$REPO_ROOT" >/dev/null
if [ -f ".env.postgres" ]; then
  log ".env.postgres contents:"
  cat .env.postgres | tee -a "$REPORT_PATH"
else
  log "[WARN] .env.postgres not found."
fi
popd >/dev/null

header "[6] Testing Python-level connectivity"
export DJANGO_PROJECT_ROOT="$REPO_ROOT"
PY_RESULTS=$("$VENV_BIN/python3" - <<'PYTHON'
import os
from pathlib import Path
from dotenv import load_dotenv

repo_root = Path(os.environ.get("DJANGO_PROJECT_ROOT", ".")).resolve()
env_path = repo_root / ".env.postgres"
if env_path.exists():
    load_dotenv(env_path, override=True)

from psycopg import connect
keys = ["POSTGRES_HOST","POSTGRES_PORT","POSTGRES_DB","POSTGRES_USER","POSTGRES_PASSWORD","POSTGRES_SSLMODE"]
print("Environment snapshot:")
for key in keys:
    print(f"  {key}={os.getenv(key)!r}")

status = "OK"
message = ""
try:
    conn = connect(
        host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "coolcrypto"),
        user=os.getenv("POSTGRES_USER", "coolcrypto"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
        connect_timeout=5,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user;")
        print(f"psycopg connect: OK ({cur.fetchone()})")
    conn.close()
except Exception as exc:
    status = "FAIL"
    message = str(exc)
    print(f"psycopg connect: FAILED -> {exc}")

print(f"PY_STATUS={status}")
print(f"PY_ERROR={message}")
PYTHON
)
echo "$PY_RESULTS" >>"$REPORT_PATH"
PG_STATUS=$(echo "$PY_RESULTS" | awk -F= '/^PY_STATUS=/{print $2}' | tail -n1)
PG_MESSAGE=$(echo "$PY_RESULTS" | awk -F= '/^PY_ERROR=/{print substr($0,10)}' | tail -n1)
if [ "$PG_STATUS" != "OK" ]; then
  PG_STATUS="FAILED"
  add_recommendation "Django could not connect to PostgreSQL (${PG_MESSAGE:-no details}). Ensure postgres is up and credentials in .env.postgres are correct."
else
  PG_STATUS="OK"
fi

header "[7] Probing Django runserver status"
RUNSERVER_PID=$(pgrep -f "web/manage.py runserver" || true)
if [ -n "$RUNSERVER_PID" ]; then
  log "runserver process detected: PID(s) $RUNSERVER_PID"
  ps -fp $RUNSERVER_PID | tee -a "$REPORT_PATH"
else
  log "[WARN] No runserver process detected."
fi

header "[8] Attempting local HTTP request"
if command -v curl >/dev/null 2>&1; then
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:$SERVER_PORT/ || echo "000")
  HTTP_STATUS="$HTTP_CODE"
  log "HTTP status from 127.0.0.1:$SERVER_PORT -> $HTTP_CODE"
  if [ "$HTTP_CODE" = "000" ]; then
    add_recommendation "HTTP probe to 127.0.0.1:$SERVER_PORT timed out. Verify runserver is bound to that interface or forward the port."
  elif [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "302" ]; then
    add_recommendation "Server responded with HTTP $HTTP_CODE. Inspect Django logs for errors."
  fi
else
  HTTP_STATUS="curl-missing"
  log "[WARN] curl not found; skipping HTTP probe."
fi

header "[9] Summary"
log "PostgreSQL connectivity: $PG_STATUS"
if [ -n "$PG_MESSAGE" ]; then
  log "  Details: $PG_MESSAGE"
fi
log "Raw socket probe: $SOCKET_STATUS"
log "Port $SERVER_PORT state: $PORT_STATUS"
log "HTTP probe code: $HTTP_STATUS"
log "Report saved to: $REPORT_PATH"

if [ ${#RECOMMENDATIONS[@]} -gt 0 ]; then
  log ""
  log "Next steps:"
  for rec in "${RECOMMENDATIONS[@]}"; do
    log " - $rec"
  done
else
  log "No obvious issues detected."
fi
