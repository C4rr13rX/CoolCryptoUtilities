#!/usr/bin/env python3
"""
Utility to migrate the existing SQLite database into PostgreSQL.

Usage example:

    python3 scripts/sqlite_to_postgres.py \
        --sqlite-path storage/trading_cache.db \
        --pg-host 127.0.0.1 --pg-port 5432 \
        --pg-db coolcrypto --pg-user postgres --pg-password postgres

The script will:
  1. Dump the current SQLite contents via `dumpdata`.
  2. Run `migrate` against PostgreSQL.
  3. Load the dumped data into PostgreSQL.

Make sure PostgreSQL is running and the target database exists before executing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
MANAGE_PY = REPO_ROOT / "web" / "manage.py"


def run_manage_command(args, extra_env):
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [sys.executable, str(MANAGE_PY), *args]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate SQLite data into PostgreSQL.")
    parser.add_argument(
        "--sqlite-path",
        default=str(REPO_ROOT / "storage" / "trading_cache.db"),
        help="Path to the existing SQLite database.",
    )
    parser.add_argument("--pg-host", default=os.getenv("POSTGRES_HOST", "127.0.0.1"))
    parser.add_argument("--pg-port", default=os.getenv("POSTGRES_PORT", "5432"))
    parser.add_argument("--pg-db", default=os.getenv("POSTGRES_DB", "coolcrypto"))
    parser.add_argument("--pg-user", default=os.getenv("POSTGRES_USER", "coolcrypto"))
    parser.add_argument("--pg-password", default=os.getenv("POSTGRES_PASSWORD", "coolcrypto"))
    parser.add_argument("--pg-sslmode", default=os.getenv("POSTGRES_SSLMODE", "prefer"))
    args = parser.parse_args()

    sqlite_env = {
        "DJANGO_DB_VENDOR": "sqlite",
        "DJANGO_SQLITE_PATH": str(Path(args.sqlite_path).resolve()),
    }
    postgres_env = {
        "DJANGO_DB_VENDOR": "postgres",
        "POSTGRES_HOST": args.pg_host,
        "POSTGRES_PORT": str(args.pg_port),
        "POSTGRES_DB": args.pg_db,
        "POSTGRES_USER": args.pg_user,
        "POSTGRES_PASSWORD": args.pg_password,
        "POSTGRES_SSLMODE": args.pg_sslmode,
    }

    print("🔄 Dumping data from SQLite…")
    with tempfile.NamedTemporaryFile(prefix="sqlite-dump-", suffix=".json", delete=False) as dump_file:
        dump_path = dump_file.name

    try:
        run_manage_command(
            [
                "dumpdata",
                "--natural-foreign",
                "--natural-primary",
                "--indent",
                "2",
                "--output",
                dump_path,
            ],
            sqlite_env,
        )

        print("🧱 Applying migrations to PostgreSQL…")
        run_manage_command(["migrate", "--noinput"], postgres_env)
        ensure_unmanaged_tables(postgres_env)

        print("📦 Loading data into PostgreSQL…")
        run_manage_command(["loaddata", dump_path], postgres_env)

        print("✅ Migration complete. Update your environment to use DJANGO_DB_VENDOR=postgres.")
    finally:
        try:
            Path(dump_path).unlink(missing_ok=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def ensure_unmanaged_tables(pg_env: dict[str, str]) -> None:
    """Create unmanaged tables (metrics, market_stream, etc.) if they are missing."""
    connection = psycopg.connect(
        host=pg_env["POSTGRES_HOST"],
        port=pg_env["POSTGRES_PORT"],
        dbname=pg_env["POSTGRES_DB"],
        user=pg_env["POSTGRES_USER"],
        password=pg_env["POSTGRES_PASSWORD"],
        sslmode=pg_env.get("POSTGRES_SSLMODE", "prefer"),
    )
    connection.autocommit = True
    stmts = [
        """
        CREATE TABLE IF NOT EXISTS market_stream (
            id BIGSERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            chain VARCHAR(64),
            symbol VARCHAR(64),
            price DOUBLE PRECISION,
            volume DOUBLE PRECISION,
            raw JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id BIGSERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            stage VARCHAR(64),
            category VARCHAR(128),
            name VARCHAR(128),
            value DOUBLE PRECISION,
            meta JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id BIGSERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            source VARCHAR(128),
            severity VARCHAR(32),
            label VARCHAR(128),
            details JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS trading_ops (
            id BIGSERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            wallet VARCHAR(128),
            chain VARCHAR(64),
            symbol VARCHAR(64),
            action VARCHAR(32),
            status VARCHAR(64),
            details JSONB
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS advisories (
            id BIGSERIAL PRIMARY KEY,
            ts DOUBLE PRECISION,
            scope VARCHAR(128),
            topic VARCHAR(128),
            severity VARCHAR(32),
            message TEXT,
            recommendation TEXT,
            meta JSONB,
            resolved BOOLEAN DEFAULT FALSE,
            resolved_ts DOUBLE PRECISION
        );
        """,
    ]
    with connection.cursor() as cursor:
        for stmt in stmts:
            cursor.execute(stmt)
    connection.close()


if __name__ == "__main__":
    main()
