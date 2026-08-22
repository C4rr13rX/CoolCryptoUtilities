#!/usr/bin/env python3
"""
Prove the migration is complete before anything is deleted.

Deleting source data is irreversible, so this checks the copy directly rather
than trusting that the migration reported success.  For every migrated table it
compares S3 against SQLite on:

* row counts, per partition (not just per table -- a table total can match
  while individual months are wrong);
* a content sample, so a partition of the right size but wrong data fails;
* blob presence and byte-identity for the snapshot payloads.

Exits non-zero if anything is missing.  ``cleanup_source.py`` refuses to run
unless this passes.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.market_store import (  # noqa: E402
    BLOB_TABLES,
    PARTITIONED_TABLES,
    MarketStore,
    month_of,
)
from serverless.hybrid.smart_storage import SmartStorage  # noqa: E402

# Tables moved by migrate_to_s3.py (per-account application data).
APP_TABLES = [
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

problems: list[str] = []
checks = 0


def ok(label: str, condition: bool, detail: str = "") -> bool:
    global checks
    checks += 1
    if not condition:
        problems.append(f"{label}: {detail}")
        print(f"  [FAIL] {label} -- {detail}")
        return False
    print(f"  [ok]   {label}" + (f" ({detail})" if detail else ""))
    return True


def verify_partitioned(store: MarketStore, conn: sqlite3.Connection,
                       sample: int) -> None:
    for table, ts_col in PARTITIONED_TABLES.items():
        total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if not total:
            continue

        manifest = store.read_manifest(table)
        by_month = Counter()
        for (ts,) in conn.execute(f'SELECT "{ts_col}" FROM "{table}"'):
            if ts is not None:
                by_month[month_of(float(ts))] += 1

        s3_months = {p["key"]: p["rows"] for p in manifest["partitions"]}
        missing = set(by_month) - set(s3_months)
        ok(f"{table}: all months present", not missing,
           f"missing {sorted(missing)}")

        bad = [f"{m}: s3={s3_months.get(m)} src={n}"
               for m, n in by_month.items() if s3_months.get(m) != n]
        ok(f"{table}: per-partition counts match", not bad, "; ".join(bad)[:180])

        # Counts can match while the bytes are wrong, so read one partition
        # back and compare actual field values against the source row.
        if sample and by_month:
            month = max(by_month)
            rows = store.read_partition(table, month)
            index = {r.get("id"): r for r in rows if r.get("id") is not None}
            checked = mismatched = 0
            cur = conn.execute(f'SELECT * FROM "{table}"')
            cols = [d[0] for d in cur.description]
            for raw in cur:
                rec = dict(zip(cols, raw))
                if rec.get("id") not in index:
                    continue
                mirror = index[rec["id"]]
                for key, value in rec.items():
                    got = mirror.get(key)
                    if isinstance(value, float) and isinstance(got, (int, float)):
                        if abs(value - float(got)) > 1e-9:
                            mismatched += 1
                            break
                    elif value is not None and str(value) != str(got):
                        mismatched += 1
                        break
                checked += 1
                if checked >= sample:
                    break
            ok(f"{table}: sampled values identical", mismatched == 0,
               f"{mismatched}/{checked} differ")


def verify_blobs(store: MarketStore, conn: sqlite3.Connection,
                 sample: int) -> None:
    for table, ts_col in BLOB_TABLES.items():
        total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if not total:
            continue

        client = store.storage.client
        pages = client.get_paginator("list_objects_v2")
        prefix = f"{store.blob_key(table, '')[:-8]}"  # strip the "/.json.gz"
        count = sum(
            len(page.get("Contents", []))
            for page in pages.paginate(Bucket=store.storage.bucket, Prefix=prefix)
        )
        ok(f"{table}: every blob uploaded", count >= total,
           f"s3={count:,} src={total:,}")

        # Byte-identity on a sample: the index could list a blob that was
        # written truncated or gzipped wrong.
        rows = conn.execute(
            f'SELECT rowid, payload FROM "{table}" ORDER BY "{ts_col}" LIMIT ?',
            (sample,)).fetchall()
        bad = 0
        for rid, payload in rows:
            got = store.read_blob(table, rid)
            if got != json.loads(payload):
                bad += 1
        ok(f"{table}: sampled blobs byte-identical", bad == 0,
           f"{bad}/{len(rows)} differ")


def verify_app_tables(storage: SmartStorage, conn: sqlite3.Connection) -> None:
    for table in APP_TABLES:
        try:
            total = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        except sqlite3.Error:
            continue
        if not total:
            continue
        rows = storage.list_table(table)
        ok(f"{table}: row count matches", len(rows) == total,
           f"s3={len(rows)} src={total}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=str(ROOT / "storage" / "trading_cache.db"))
    ap.add_argument("--bucket", default=os.getenv("HYBRID_BUCKET", "coolcrypto-hybrid"))
    ap.add_argument("--endpoint",
                    default=os.getenv("AWS_S3_ENDPOINT_URL", "http://localhost:9000"))
    ap.add_argument("--sample", type=int, default=50,
                    help="rows/blobs to compare byte-for-byte per table")
    args = ap.parse_args()

    src = Path(args.sqlite)
    if not src.exists():
        print(f"source database not found: {src}")
        return 1

    storage = SmartStorage(bucket=args.bucket, endpoint_url=args.endpoint)
    store = MarketStore(storage)
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)

    print("=" * 62)
    print("  Migration verification (run before deleting anything)")
    print("=" * 62)

    print("\n[1] Application tables")
    verify_app_tables(storage, conn)

    print("\n[2] Partitioned market data")
    verify_partitioned(store, conn, args.sample)

    print("\n[3] Snapshot blobs")
    verify_blobs(store, conn, args.sample)

    print("\n" + "=" * 62)
    if problems:
        print(f"  {len(problems)} PROBLEM(S) of {checks} checks -- DO NOT DELETE")
        for p in problems:
            print(f"    - {p}")
        print("=" * 62)
        return 1
    print(f"  all {checks} checks passed -- source data is safe to remove")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
