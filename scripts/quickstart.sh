#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Python not found. Install Python 3.11+ first." >&2
  exit 1
fi

$PY - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 8):
    print("Python 3.8+ is required. Install a newer Python (pyenv/brew/apt).", file=sys.stderr)
    raise SystemExit(1)
PY

if [ ! -d .venv ]; then
  echo "[quickstart] Creating venv"
  $PY -m venv .venv
fi

VENV_PY=".venv/bin/python"
$VENV_PY -m pip install -U pip
$VENV_PY -m pip install -r requirements.txt
if [ -f requirements_textbooks.txt ]; then
  $VENV_PY -m pip install -r requirements_textbooks.txt
fi

export DJANGO_DB_VENDOR=sqlite
export DJANGO_PREFER_SQLITE_FALLBACK=1

echo "[quickstart] Running migrations"
$VENV_PY web/manage.py migrate

BIN_PATH="$REPO_ROOT/bin"
export PATH="$BIN_PATH:$PATH"

MARKER="CoolCryptoUtilities CLI"
UPDATED=0
for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" "$HOME/.zshrc"; do
  if [ -f "$rc" ] && ! grep -q "$MARKER" "$rc"; then
    {
      echo ""
      echo "# $MARKER"
      echo "export PATH=\"$BIN_PATH:\$PATH\""
    } >>"$rc"
    UPDATED=1
  fi
done
if [ "$UPDATED" -eq 0 ]; then
  rc="$HOME/.profile"
  {
    echo ""
    echo "# $MARKER"
    echo "export PATH=\"$BIN_PATH:\$PATH\""
  } >>"$rc"
fi

echo "[quickstart] Starting Django at http://$HOST:$PORT"
exec $VENV_PY web/manage.py runserver "$HOST:$PORT"
