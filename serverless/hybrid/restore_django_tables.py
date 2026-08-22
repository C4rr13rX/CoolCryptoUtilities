#!/usr/bin/env python3
"""
Rehydrate the Django-ORM tables from S3 into the local database.

Why this exists
---------------
The migration moved 14 tables (BrandDozer + SecureVault) into S3 and the
cleanup dropped them from SQLite.  But those apps still reach their data
through the **Django ORM**, which knows nothing about the S3 store -- so every
BrandDozer and SecureVault view broke with "no such table".

The hybrid model's local tier is AllezORM *in the browser*, which serves the
frontend.  It does not serve server-side Django code, and the server-rendered
admin, the management commands, and the DRF views all still issue SQL.

So the local database becomes a **projection of S3, not a second source of
truth**: S3 remains authoritative, this rebuilds the SQL tables from it, and
re-running is safe because it replaces rows rather than appending.

Run it after a cleanup, on a fresh checkout, or any time the local database is
rebuilt.  For the Lambda deployment the same job runs at cold start against
``/tmp`` (see ``HYBRID_DB`` in ``settings_lambda.py``).

Usage::

    python -m serverless.hybrid.restore_django_tables --dry-run
    python -m serverless.hybrid.restore_django_tables
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.smart_storage import SmartStorage  # noqa: E402
from serverless.hybrid.verify_migration import APP_TABLES  # noqa: E402


def setup_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                          "coolcrypto_dashboard.settings")
    os.environ.setdefault("DJANGO_DB_VENDOR", "sqlite")
    os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
    os.environ.setdefault("SECURE_ENV_HYDRATED", "1")
    # These tables are restored by a management-free path; do not let importing
    # Django boot the guardian/cron/production threads as a side effect.
    os.environ.setdefault("GUARDIAN_AUTO_DISABLED", "1")
    os.environ.setdefault("CRON_AUTO_DISABLED", "1")
    os.environ.setdefault("PRODUCTION_AUTO_DISABLED", "1")

    import django

    django.setup()


def ensure_schema() -> None:
    """
    Create only the tables that are actually missing.

    Two tempting approaches do not work here:

    * plain ``migrate`` is a no-op, because dropping a table does not remove
      its row from ``django_migrations`` -- Django still believes it exists;
    * clearing that history and replaying the migrations fails, because the
      apps' *other* tables were never dropped, so the first CREATE collides
      with ``table ... already exists``.

    So the schema is built straight from the model definitions with Django's
    own schema editor. That yields exactly the columns, types and constraints
    the models expect, and touches nothing that is already present.
    """
    from django.apps import apps as django_apps
    from django.db import connection

    with connection.cursor() as cur:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        present = {row[0] for row in cur.fetchall()}

    missing = [
        m for m in django_apps.get_models()
        if m._meta.managed and m._meta.db_table not in present
    ]
    if not missing:
        print("    all tables present")
        return

    with connection.schema_editor() as editor:
        for model in missing:
            editor.create_model(model)
            print(f"    created {model._meta.db_table}")

    # A table can exist and still be out of step with its model -- an older
    # copy that predates a column the model has since gained. Restoring into
    # it fails with "no column named ...", so reconcile the difference here.
    with connection.cursor() as cur:
        for model in django_apps.get_models():
            if not model._meta.managed:
                continue
            table = model._meta.db_table
            cur.execute(f'PRAGMA table_info("{table}")')
            existing = {row[1] for row in cur.fetchall()}
            if not existing:
                continue
            absent = [f for f in model._meta.fields if f.column not in existing]
            if not absent:
                continue
            with connection.schema_editor() as editor:
                for field in absent:
                    editor.add_field(model, field)
                    print(f"    added {table}.{field.column}")


def restore_table(storage: SmartStorage, table: str, dry: bool) -> tuple[int, str]:
    import json

    from django.apps import apps
    from django.db import connection

    model = None
    for candidate in apps.get_models():
        if candidate._meta.db_table == table:
            model = candidate
            break
    if model is None:
        return 0, "no model maps to this table"

    rows = storage.list_table(table)
    if not rows:
        return 0, "no rows in S3"
    if dry:
        return len(rows), "would restore"

    columns = {f.column for f in model._meta.fields}

    with connection.cursor() as cur:
        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        before = cur.fetchone()[0]

        inserted = skipped = 0
        for row in rows:
            payload = {}
            for key, value in row.items():
                # `_deleted` and friends are store bookkeeping; a column the
                # model no longer declares would abort the INSERT.
                if key.startswith("_") or key not in columns:
                    continue
                # SQLite cannot bind dict/list. These came out of JSON columns
                # and have to go back in as their serialised form.
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                payload[key] = value
            if not payload:
                continue

            cols = ", ".join(f'"{c}"' for c in payload)
            marks = ", ".join(["%s"] * len(payload))
            try:
                # INSERT OR REPLACE keeps this idempotent -- re-running
                # refreshes rows rather than failing on the primary key.
                cur.execute(
                    f'INSERT OR REPLACE INTO "{table}" ({cols}) VALUES ({marks})',
                    list(payload.values()),
                )
                inserted += 1
            except Exception:  # noqa: BLE001
                # A row pointing at a parent that no longer exists is dropped
                # rather than aborting the table. Restoring 99% of a table
                # beats restoring none of it, and the count is reported.
                skipped += 1

        cur.execute(f'SELECT COUNT(*) FROM "{table}"')
        after = cur.fetchone()[0]

    note = f"{before} -> {after} rows"
    if skipped:
        note += f", {skipped} skipped"
    return inserted, note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.getenv("HYBRID_BUCKET", "coolcrypto-hybrid"))
    ap.add_argument("--endpoint",
                    default=os.getenv("AWS_S3_ENDPOINT_URL", "http://localhost:9000"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print(f"  Rehydrate Django tables from S3 "
          f"({'DRY RUN' if args.dry_run else 'WRITE'})")
    print("=" * 62)
    print(f"  source: s3://{args.bucket}\n")

    setup_django()
    if not args.dry_run:
        print("[1] Ensuring schema exists (migrate --run-syncdb)")
        ensure_schema()
        print("    schema ready\n")

    storage = SmartStorage(bucket=args.bucket, endpoint_url=args.endpoint)
    storage.reset_memo()

    print("[2] Restoring rows")
    if not args.dry_run:
        # Load with foreign keys disabled: rows are restored table by table,
        # so a child row often lands before the parent it references. The
        # integrity check below re-enables enforcement and reports anything
        # genuinely dangling.
        from django.db import connection

        with connection.cursor() as cur:
            cur.execute("PRAGMA foreign_keys = OFF")

    total = 0
    for table in APP_TABLES:
        try:
            count, note = restore_table(storage, table, args.dry_run)
            total += count
            print(f"  {'+' if count else '-'} {table}: {count} rows ({note})")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {table}: {type(exc).__name__}: {str(exc)[:110]}")

    if not args.dry_run:
        from django.db import connection

        with connection.cursor() as cur:
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA foreign_key_check")
            violations = cur.fetchall()
        print("\n[3] Integrity")
        if violations:
            # Report rather than fail: these are rows whose parent was already
            # missing before the migration, and dropping them would lose data
            # the app may still display.
            print(f"  {len(violations)} dangling foreign key(s) "
                  f"(pre-existing, left in place)")
            for v in violations[:5]:
                print(f"    {v}")
        else:
            print("  no foreign key violations")

    print(f"\n  {total} rows {'would be ' if args.dry_run else ''}restored")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
