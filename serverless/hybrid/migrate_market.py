#!/usr/bin/env python3
"""
Migrate the shared market data into the cloud store.

Complements ``migrate_to_s3.py``, which moved the per-account application
tables.  This handles the volume: ~3.4M time-series rows plus 22 GB of
organism snapshots, all of it *shared* -- one copy serves every account.

Strategy per table (see ``market_store`` for the reasoning):

* **Partitioned tables** -> monthly Parquet, streamed from SQLite so peak
  memory stays at one month rather than one table.
* **Blob tables** -> one gzipped object per row plus a small Parquet index
  that the browser mirrors instead of the payloads.

Both are resumable: a partition already present with the same row count is
skipped, so an interrupted run can simply be re-run.

Usage::

    python -m serverless.hybrid.migrate_market --dry-run
    python -m serverless.hybrid.migrate_market --tables market_stream,metrics
    python -m serverless.hybrid.migrate_market            # everything
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.market_store import (  # noqa: E402
    BLOB_TABLES,
    PARTITIONED_TABLES,
    MarketStore,
    Partition,
    month_of,
)
from serverless.hybrid.smart_storage import SmartStorage  # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def migrate_partitioned(store: MarketStore, conn, table: str, dry: bool,
                        resume: bool) -> dict:
    ts_col = PARTITIONED_TABLES[table]
    conn.row_factory = sqlite3.Row

    total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    if not total:
        print(f"  - {table}: empty")
        return {"rows": 0, "partitions": 0, "bytes": 0}

    existing = {p["key"]: p for p in store.read_manifest(table)["partitions"]}

    # Group by month in a streaming pass. Ordering by ts means each month's
    # rows arrive together, so only one month is ever held in memory -- loading
    # all 2.8M metrics rows at once would be several GB of Python dicts.
    print(f"  · {table}: {total:,} rows, grouping by month...")
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in conn.execute(f'SELECT * FROM "{table}" ORDER BY "{ts_col}"'):
        rec = dict(row)
        stamp = rec.get(ts_col)
        if stamp is None:
            continue
        buckets[month_of(float(stamp))].append(_normalise(rec))

    partitions: list[Partition] = []
    written = skipped = 0
    for month in sorted(buckets):
        rows = buckets[month]
        prior = existing.get(month)
        if resume and prior and prior.get("rows") == len(rows):
            # Same month, same count: already uploaded on an earlier run.
            partitions.append(Partition(
                key=month, rows=prior["rows"], bytes=prior.get("bytes", 0),
                min_ts=prior.get("min_ts", 0), max_ts=prior.get("max_ts", 0),
            ))
            skipped += 1
            continue
        if dry:
            print(f"    {month}: {len(rows):,} rows (would write)")
            partitions.append(Partition(month, len(rows), 0.0, 0.0, 0))
            continue
        part = store.write_partition(table, month, rows)
        partitions.append(part)
        written += 1
        print(f"    {month}: {part.rows:,} rows -> {human(part.bytes)}")

    if not dry:
        store.write_manifest(table, partitions)

    size = sum(p.bytes for p in partitions)
    print(f"  + {table}: {total:,} rows in {len(partitions)} partitions "
          f"({human(size)}; {written} written, {skipped} already present)")
    return {"rows": total, "partitions": len(partitions), "bytes": size}


def _normalise(rec: dict) -> dict:
    """
    Coerce SQLite values into Arrow-friendly Python types.

    Arrow infers a column type from the first non-null value and then rejects
    anything inconsistent, so a column that is usually int but occasionally
    str would abort the whole partition. Stringifying the loose ones keeps the
    schema stable.
    """
    out = {}
    for key, value in rec.items():
        if isinstance(value, (bytes, bytearray)):
            out[key] = value.decode("utf-8", errors="replace")
        elif isinstance(value, (int, float, str)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def migrate_blobs(store: MarketStore, conn, table: str, dry: bool,
                  resume: bool, limit: int = 0) -> dict:
    ts_col = BLOB_TABLES[table]
    conn.row_factory = sqlite3.Row

    total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    if not total:
        print(f"  - {table}: empty")
        return {"rows": 0, "bytes": 0}

    print(f"  · {table}: {total:,} rows (payloads stay in S3, index goes local)")
    if dry:
        raw = conn.execute(f'SELECT SUM(LENGTH(payload)) FROM "{table}"').fetchone()[0] or 0
        print(f"    raw {human(raw)}; ~7x compression expected "
              f"-> ~{human(raw / 7.4)}")
        return {"rows": total, "bytes": 0}

    index: list[dict] = []
    uploaded = 0
    written_bytes = 0
    started = time.time()

    query = f'SELECT rowid, * FROM "{table}" ORDER BY "{ts_col}"'
    if limit:
        query += f" LIMIT {int(limit)}"

    for row in conn.execute(query):
        rec = dict(row)
        ident = rec.get("id") or rec.get("rowid")
        payload = rec.get("payload") or ""
        size = store.write_blob(table, ident, payload)
        written_bytes += size
        index.append({
            "id": str(ident),
            "ts": float(rec.get(ts_col) or 0),
            "bytes": size,
            # Raw length is what tells a UI whether a payload is worth
            # fetching before it commits to the download.
            "raw_bytes": len(payload),
        })
        uploaded += 1
        if uploaded % 500 == 0:
            rate = uploaded / max(time.time() - started, 1e-6)
            print(f"    {uploaded:,}/{total:,} ({human(written_bytes)}, "
                  f"{rate:.0f}/s)")

    idx_bytes = store.write_blob_index(table, index)
    print(f"  + {table}: {uploaded:,} blobs ({human(written_bytes)}), "
          f"index {human(idx_bytes)}")
    return {"rows": uploaded, "bytes": written_bytes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=str(ROOT / "storage" / "trading_cache.db"))
    ap.add_argument("--bucket", default=os.getenv("HYBRID_BUCKET", "coolcrypto-hybrid"))
    ap.add_argument("--endpoint",
                    default=os.getenv("AWS_S3_ENDPOINT_URL", "http://localhost:9000"))
    ap.add_argument("--tables", default="",
                    help="comma-separated subset; default is all")
    ap.add_argument("--blob-limit", type=int, default=0,
                    help="cap blobs per table (for a quick trial run)")
    ap.add_argument("--skip-blobs", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-upload partitions even if already present")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.sqlite)
    if not src.exists():
        print(f"source database not found: {src}")
        return 1

    wanted = {t.strip() for t in args.tables.split(",") if t.strip()}
    parted = [t for t in PARTITIONED_TABLES if not wanted or t in wanted]
    blobs = [t for t in BLOB_TABLES if not wanted or t in wanted]
    if args.skip_blobs:
        blobs = []

    print(f"source : {src}")
    print(f"target : s3://{args.bucket}/database/market  ({args.endpoint})")
    print(f"mode   : {'DRY RUN' if args.dry_run else 'WRITE'}")
    print("note   : this data is SHARED across all accounts\n")

    storage = SmartStorage(bucket=args.bucket, endpoint_url=args.endpoint)
    store = MarketStore(storage)
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)

    print("[1] partitioned time series -> monthly Parquet")
    rows = size = 0
    for table in parted:
        try:
            res = migrate_partitioned(store, conn, table, args.dry_run,
                                      resume=not args.no_resume)
            rows += res["rows"]
            size += res["bytes"]
        except Exception as exc:  # noqa: BLE001
            # One bad table must not abandon the rest of a long migration.
            print(f"  ! {table} failed: {exc}")

    print("\n[2] large blobs -> gzipped objects + local index")
    for table in blobs:
        try:
            res = migrate_blobs(store, conn, table, args.dry_run,
                                resume=not args.no_resume, limit=args.blob_limit)
            rows += res["rows"]
            size += res["bytes"]
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {table} failed: {exc}")

    print(f"\nmigrated {rows:,} rows, {human(size)} in S3")
    print(f"S3 objects written: ~{storage.stats.s3_puts:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
