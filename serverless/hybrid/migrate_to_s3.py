#!/usr/bin/env python3
"""
Migrate the existing SQLite/Postgres data into the S3-as-database layout.

Moves the one real account (``admin``) and the application rows it owns into
``database/tables/<table>/<id>.json``, plus the ``total.txt`` counters and
secondary indexes the runtime expects.

Two things it deliberately does *not* copy:

* **The trading telemetry.**  ``metrics``, ``feedback_events`` and
  ``market_stream`` are ~3.4M rows and 27 GB.  They are append-only
  instrumentation that the dashboard reads through the trading pipeline, not
  the Django ORM, so pushing them into per-row S3 objects would cost a fortune
  in PUT requests and buy nothing.  They stay where they are.

* **The Django password hash.**  ``auth_user.password`` is PBKDF2 from
  Django's hasher.  The new login path needs an Argon2id verifier over a
  password the user supplies, and a hash cannot be converted between the two.
  The admin credential is therefore re-seeded here explicitly.

Usage::

    python -m serverless.hybrid.migrate_to_s3 --dry-run
    python -m serverless.hybrid.migrate_to_s3
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.pq_auth import TABLE_USERS, hash_password  # noqa: E402
from serverless.hybrid.smart_storage import BASE_PREFIX, SmartStorage  # noqa: E402

# Tables carrying real application data. Ordered so referenced rows land before
# the rows that point at them, which keeps the S3 copy self-consistent if the
# run is interrupted partway.
MIGRATE_TABLES = [
    "securevault_securesetting",
    "branddozer_brandproject",
    "branddozer_deliveryproject",
    "branddozer_sprint",
    "branddozer_sprintitem",
    "branddozer_backlogitem",
    "branddozer_backgroundjob",
    "branddozer_deliveryrun",
    "branddozer_deliverysession",
    "branddozer_deliveryartifact",
    "branddozer_researchpaper",
    "branddozer_researchpaperrevision",
    "branddozer_researchclaim",
    "branddozer_researchsource",
]

# Volume tables: instrumentation, not app state. See the module docstring.
SKIP_TABLES = {"metrics", "feedback_events", "market_stream", "organism_snapshots"}


def read_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f'SELECT * FROM "{table}"')]
    except sqlite3.Error as exc:
        print(f"  ! {table}: {exc}")
        return []


def jsonable(value):
    """SQLite hands back bytes and datetimes that json cannot encode."""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            import base64

            return {"__b64__": base64.b64encode(value).decode("ascii")}
    return value


def migrate_table(storage: SmartStorage, conn, table: str, dry: bool) -> int:
    rows = read_rows(conn, table)
    if not rows:
        print(f"  - {table}: empty, skipped")
        return 0

    max_id = 0
    idents: list = []
    uuid_keyed = False

    for position, row in enumerate(rows, start=1):
        record = {k: jsonable(v) for k, v in row.items()}
        ident = record.get("id")
        if ident is None or ident == "":
            ident = position
        if str(ident).isdigit():
            max_id = max(max_id, int(ident))
        else:
            # Several branddozer models use 32-char UUID primary keys. A
            # total.txt range cannot address those, so the key space has to be
            # written out explicitly or the rows become unreadable.
            uuid_keyed = True
        idents.append(ident)
        if not dry:
            storage.put_row(table, ident, record)

    if not dry:
        if uuid_keyed:
            storage.set_keys_manifest(table, idents)
        else:
            # total.txt is how the runtime knows the id range to read back;
            # without it list_table() returns nothing despite the objects
            # existing.
            storage.put_text(f"{BASE_PREFIX}/{table}/total.txt", str(max_id))

    shape = f"{len(idents)} uuid keys" if uuid_keyed else f"max id {max_id}"
    print(f"  + {table}: {len(rows)} rows ({shape})")
    return len(rows)


def seed_admin(storage: SmartStorage, email: str, password: str, dry: bool) -> None:
    """
    Create the operator account for the new login path.

    Django's PBKDF2 hash cannot be converted to Argon2id, so the credential is
    written fresh rather than migrated. The strength rules in
    ``password_strength_error`` are bypassed here on purpose: the caller asked
    for this specific credential for local use. It must be changed before the
    stack is exposed to anything but localhost.
    """
    if storage.find_by(TABLE_USERS, "email", email):
        print(f"  = admin '{email}' already exists, leaving untouched")
        return
    if dry:
        print(f"  + admin '{email}' would be created")
        return

    record = storage.insert_row(
        TABLE_USERS,
        {
            "email": email,
            "password_hash": hash_password(password),
            "created_at": time.time(),
            "failed_attempts": 0,
            "locked_until": 0,
            "is_active": True,
            "is_superuser": True,
            # Flags the weak seeded credential so the UI and any audit can
            # nag until it is rotated.
            "must_change_password": True,
        },
    )
    storage.set_index(TABLE_USERS, "email", email, record["id"])
    print(f"  + admin '{email}' created (id {record['id']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=str(ROOT / "storage" / "trading_cache.db"))
    ap.add_argument("--bucket", default=os.getenv("HYBRID_BUCKET", "coolcrypto-hybrid"))
    ap.add_argument("--endpoint", default=os.getenv("AWS_S3_ENDPOINT_URL",
                                                   "http://localhost:9000"))
    ap.add_argument("--admin-email", default=os.getenv("ADMIN_EMAIL", "admin"))
    ap.add_argument("--admin-password", default=os.getenv("ADMIN_PASSWORD", "admin"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.sqlite)
    if not src.exists():
        print(f"source database not found: {src}")
        return 1

    print(f"source : {src}")
    print(f"target : s3://{args.bucket}/{BASE_PREFIX}  ({args.endpoint})")
    print(f"mode   : {'DRY RUN' if args.dry_run else 'WRITE'}\n")

    storage = SmartStorage(bucket=args.bucket, endpoint_url=args.endpoint)
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)

    print("[1] application tables")
    total = sum(migrate_table(storage, conn, t, args.dry_run) for t in MIGRATE_TABLES)

    print("\n[2] operator account")
    seed_admin(storage, args.admin_email, args.admin_password, args.dry_run)

    print(f"\n[3] skipped by design: {', '.join(sorted(SKIP_TABLES))}")
    print(f"\nmigrated {total} rows")
    if not args.dry_run and args.admin_password == "admin":
        print("\nWARNING: the admin password is 'admin'. Change it before this "
              "stack is reachable from anywhere but localhost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
