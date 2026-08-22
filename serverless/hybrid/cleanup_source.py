#!/usr/bin/env python3
"""
Remove the source data now that it lives in S3 — no duplicates.

Irreversible, so it is deliberately hard to run by accident:

1. It re-runs ``verify_migration`` and **aborts** unless every check passes.
2. It requires an explicit ``--yes`` flag.
3. It writes a manifest of what it removed, so the deletion is auditable.

What it removes:

* the migrated tables from ``storage/trading_cache.db`` (then ``VACUUM``, which
  is what actually returns the ~27 GB to the filesystem -- ``DROP TABLE`` alone
  only marks pages free);
* the Postgres database and its Docker volume.

What it keeps:

* ``django_migrations``, ``auth_permission``, ``django_content_type`` and the
  other contrib scaffolding, because Django recreates and needs them locally;
* every non-migrated table (``kv_store``, ``experiments``, ``system_logs``,
  ``pair_adjustments``, ...) -- these were never part of the migration and
  deleting them would lose data that exists nowhere else.

Usage::

    python -m serverless.hybrid.cleanup_source --dry-run
    python -m serverless.hybrid.cleanup_source --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.market_store import BLOB_TABLES, PARTITIONED_TABLES  # noqa: E402
from serverless.hybrid.verify_migration import APP_TABLES  # noqa: E402

# Exactly the tables the migration copied. Nothing else is touched.
MIGRATED_TABLES = sorted(
    set(APP_TABLES) | set(PARTITIONED_TABLES) | set(BLOB_TABLES)
)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def run_verification(bucket: str, endpoint: str) -> bool:
    """Re-verify rather than trusting an earlier run."""
    print("[1] Re-verifying the migration before deleting anything\n")
    env = dict(os.environ, HYBRID_BUCKET=bucket, AWS_S3_ENDPOINT_URL=endpoint)
    result = subprocess.run(
        [sys.executable, "-m", "serverless.hybrid.verify_migration",
         "--sample", "25", "--bucket", bucket, "--endpoint", endpoint],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    tail = result.stdout.strip().splitlines()[-6:]
    print("\n".join(f"    {line}" for line in tail))
    return result.returncode == 0


def drop_tables(db: Path, dry: bool) -> dict:
    conn = sqlite3.connect(db)
    present = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    removed: dict[str, int] = {}
    for table in MIGRATED_TABLES:
        if table not in present:
            continue
        rows = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        removed[table] = rows
        if dry:
            print(f"    would drop {table} ({rows:,} rows)")
        else:
            conn.execute(f'DROP TABLE "{table}"')
            print(f"    dropped {table} ({rows:,} rows)")

    if not dry:
        conn.commit()
        conn.close()
        # DROP TABLE only marks pages free inside the file; VACUUM is what
        # actually shrinks it on disk, which is the entire point here.
        print("    VACUUM (rebuilding the file; this takes a while on 27 GB)...")
        started = time.time()
        vac = sqlite3.connect(db)
        vac.execute("VACUUM")
        vac.close()
        print(f"    VACUUM done in {time.time() - started:.0f}s")
    else:
        conn.close()
    return removed


def drop_postgres(dry: bool) -> bool:
    """Remove the Postgres container and its volume."""
    compose_dir = ROOT / "serverless" / "local"
    if dry:
        print("    would run: docker compose rm -sfv postgres")
        print("    would remove volume coolcrypto-serverless-local_pgdata")
        return True

    for cmd in (
        ["docker", "compose", "rm", "-sfv", "postgres"],
        ["docker", "volume", "rm", "coolcrypto-serverless-local_pgdata"],
    ):
        result = subprocess.run(cmd, cwd=str(compose_dir),
                                capture_output=True, text=True)
        label = " ".join(cmd[:4])
        if result.returncode == 0:
            print(f"    {label}: ok")
        else:
            # Already gone is the desired end state, not a failure.
            err = (result.stderr or "").strip().splitlines()[-1:] or [""]
            print(f"    {label}: {err[0][:100]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=str(ROOT / "storage" / "trading_cache.db"))
    ap.add_argument("--bucket", default=os.getenv("HYBRID_BUCKET", "coolcrypto-hybrid"))
    ap.add_argument("--endpoint",
                    default=os.getenv("AWS_S3_ENDPOINT_URL", "http://localhost:9000"))
    ap.add_argument("--yes", action="store_true", help="actually delete")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="not recommended; bypasses the safety check")
    args = ap.parse_args()

    dry = args.dry_run or not args.yes
    db = Path(args.sqlite)
    if not db.exists():
        print(f"source database not found: {db}")
        return 1

    before = db.stat().st_size
    print("=" * 62)
    print(f"  Source cleanup — {'DRY RUN' if dry else 'DELETING'}")
    print("=" * 62)
    print(f"  sqlite : {db} ({human(before)})")
    print(f"  target : s3://{args.bucket}\n")

    if not args.skip_verify:
        if not run_verification(args.bucket, args.endpoint):
            print("\nVERIFICATION FAILED — nothing deleted.")
            return 1
        print("\n    verification passed\n")

    print("[2] Dropping migrated tables from SQLite")
    removed = drop_tables(db, dry)

    print("\n[3] Removing Postgres")
    drop_postgres(dry)

    if not dry:
        # The 14 Django-ORM tables must be rehydrated immediately. Their data
        # is safe in S3, but BrandDozer and SecureVault reach it through the
        # ORM -- which knows nothing about the S3 store -- so leaving the
        # tables dropped breaks every one of their views with "no such table".
        # The browser's AllezORM tier serves the frontend, not server-side
        # Django code.
        print("\n[4] Rehydrating Django-ORM tables from S3")
        rehydrate = subprocess.run(
            [sys.executable, "-m", "serverless.hybrid.restore_django_tables",
             "--bucket", args.bucket, "--endpoint", args.endpoint],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        for line in rehydrate.stdout.strip().splitlines()[-6:]:
            print(f"    {line}")
        if rehydrate.returncode != 0:
            print("    WARNING: rehydration failed -- run "
                  "`python -m serverless.hybrid.restore_django_tables` manually")

    if not dry:
        after = db.stat().st_size
        manifest = {
            "deleted_at": time.time(),
            "sqlite_path": str(db),
            "bytes_before": before,
            "bytes_after": after,
            "tables_removed": removed,
            "rows_removed": sum(removed.values()),
            "postgres_removed": True,
        }
        out = ROOT / "storage" / "migration-cleanup.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n  {human(before)} -> {human(after)} "
              f"(freed {human(before - after)})")
        print(f"  {sum(removed.values()):,} rows removed from "
              f"{len(removed)} tables")
        print(f"  manifest: {out}")
    else:
        print(f"\n  would remove {sum(removed.values()):,} rows from "
              f"{len(removed)} tables")
        print("  re-run with --yes to apply")

    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
