#!/usr/bin/env bash
set -euo pipefail

POSTGRES_DB=${POSTGRES_DB:-coolcrypto}
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-postgres}
POSTGRES_HOST=${POSTGRES_HOST:-127.0.0.1}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

printf "[setup] CoolCryptoUtilities full setup\n"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 not found. Install Python 3.11+ first." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "[setup] Creating venv"
  python3 -m venv .venv
fi

PY=.venv/bin/python
$PY -m pip install -U pip

echo "[setup] Installing Python deps"
$PY -m pip install -r requirements.txt
$PY -m pip install django "psycopg[binary]"
if [ -f requirements_textbooks.txt ]; then
  $PY -m pip install -r requirements_textbooks.txt
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "[setup] PostgreSQL not found; attempting install"
  if command -v brew >/dev/null 2>&1; then
    brew install postgresql@16
    brew services start postgresql@16
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y postgresql
    sudo service postgresql start
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y postgresql-server postgresql-contrib
    sudo postgresql-setup --initdb
    sudo systemctl enable --now postgresql
  else
    echo "Install PostgreSQL manually for your distro." >&2
  fi
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "[setup] Tesseract not found; attempting install"
  if command -v brew >/dev/null 2>&1; then
    brew install tesseract
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y tesseract
  else
    echo "Install Tesseract manually for your distro." >&2
  fi
fi

if command -v createdb >/dev/null 2>&1; then
  echo "[setup] Initializing database"
  export PGHOST="$POSTGRES_HOST" PGPORT="$POSTGRES_PORT" PGUSER="$POSTGRES_USER" PGPASSWORD="$POSTGRES_PASSWORD"
  createdb "$POSTGRES_DB" || echo "createdb skipped (already exists or permission)"
fi

if [ -f web/frontend/package.json ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[setup] Installing frontend deps"
    (cd web/frontend && npm install)
  else
    echo "npm not found. Install Node.js to build frontend." >&2
  fi
fi

echo "[setup] Running migrations"
export DJANGO_DB_VENDOR=postgres
$PY web/manage.py migrate

echo "[setup] Done."
