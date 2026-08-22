#!/usr/bin/env python3
"""
Integration tests for the shared market-data layer.

Covers the two things that matter about this data: that the Parquet
round-trips without losing rows, and that the API serves partitions the way
the browser expects (presigned redirects, not proxied bodies).

Run:  python serverless/local/test_market_stack.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AWS_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
os.environ.setdefault("HYBRID_BUCKET", "coolcrypto-hybrid")
# Must match what deploy_local.sh gives the Lambdas, or a token minted here
# will not verify there.
os.environ.setdefault("PQ_SESSION_SECRET",
                      "local-dev-session-secret-32-chars-min!")

from serverless.hybrid.market_store import MarketStore, month_of  # noqa: E402
from serverless.hybrid.pq_auth import AuthService, seal_password  # noqa: E402
from serverless.hybrid.smart_storage import SmartStorage  # noqa: E402

ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")

lam = boto3.client(
    "lambda", endpoint_url=ENDPOINT, region_name="us-east-1",
    aws_access_key_id="test", aws_secret_access_key="test",
    config=Config(retries={"max_attempts": 3}, read_timeout=310),
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def http_event(path: str, token: str | None = None) -> dict:
    headers = {"host": "localhost"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return {
        "version": "2.0", "rawPath": path, "headers": headers,
        "requestContext": {"http": {"method": "GET", "path": path},
                           "stage": "prod", "domainName": "localhost"},
        "body": "{}", "isBase64Encoded": False,
    }


def invoke(fn: str, event: dict) -> dict:
    resp = lam.invoke(FunctionName=fn, Payload=json.dumps(event).encode())
    raw = resp["Payload"].read().decode()
    if resp.get("FunctionError"):
        return {"__error__": resp["FunctionError"], "raw": raw[:400]}
    out = json.loads(raw)
    if isinstance(out, dict) and out.get("body"):
        try:
            out["parsed"] = json.loads(out["body"])
        except (json.JSONDecodeError, TypeError):
            out["parsed"] = {}
    return out


def login_token() -> str:
    """Authenticate as the migrated admin so the API accepts our reads."""
    storage = SmartStorage(bucket="coolcrypto-hybrid",
                           endpoint_url="http://localhost:9000")
    storage.reset_memo()
    svc = AuthService(storage=storage)
    ch = svc.begin_login()
    tr = AuthService.transcript_for(ch["challenge_id"], ch["server_key"])
    env = seal_password("admin", ch["server_key"], tr)
    return svc.complete_login(ch["challenge_id"], "admin", env)["token"]


def main() -> int:
    print("=" * 62)
    print("  Shared market data (Parquet in S3, AllezORM locally)")
    print("=" * 62)

    storage = SmartStorage(bucket="coolcrypto-hybrid",
                           endpoint_url="http://localhost:9000")
    store = MarketStore(storage)

    # -- 1. manifests ---------------------------------------------------
    print("\n[1] Manifests")
    for table, minimum in (("market_stream", 100_000), ("metrics", 2_800_000)):
        m = store.read_manifest(table)
        check(f"{table} manifest present", bool(m["partitions"]),
              f"{len(m['partitions'])} partitions")
        check(f"{table} holds the migrated rows", m["rows"] >= minimum,
              f"manifest={m['rows']:,} expected>={minimum:,}")
        check(f"{table} marked shared", m.get("shared") is True)

    # -- 2. partitions are internally consistent -----------------------
    # The source tables were deleted after verification (S3 is the only copy
    # now), so these check the store against its own manifest rather than
    # diffing SQLite. `verify_migration.py` did the source comparison while
    # the source still existed.
    print("\n[2] Partition integrity")
    for table in ("market_stream", "trade_fills", "advisories"):
        manifest = store.read_manifest(table)
        bad = []
        for part in manifest["partitions"]:
            rows = store.read_partition(table, part["key"])
            if len(rows) != part["rows"]:
                bad.append(f"{part['key']}: read={len(rows)} manifest={part['rows']}")
        check(f"{table} partitions match manifest", not bad,
              "; ".join(bad)[:200])

    # -- 3. column fidelity --------------------------------------------
    print("\n[3] Column fidelity")
    rows = store.read_partition("market_stream", "2026-08")
    check("partition non-empty", bool(rows), f"{len(rows)} rows")
    if rows:
        sample = rows[0]
        for col in ("id", "ts", "chain", "symbol", "price", "raw"):
            if col not in sample:
                check(f"column {col} preserved", False, "missing")
                break
        else:
            check("all columns preserved", True, ", ".join(sample.keys()))
        check("timestamps are real epochs",
              all(1.6e9 < float(r["ts"]) < 2.1e9 for r in rows[:200]))

    # -- 4. blob store --------------------------------------------------
    print("\n[4] Organism snapshots (blobs)")
    index_manifest = store.storage.get_json(
        f"database/market/organism_snapshots/_manifest.json")
    blob_id = "1"
    blob = store.read_blob("organism_snapshots", blob_id)
    check("blob fetches and parses", isinstance(blob, dict),
          type(blob).__name__)

    # -- 5. API ---------------------------------------------------------
    print("\n[5] Market API")
    r = invoke("coolcrypto-market", http_event("/market/metrics/manifest"))
    check("unauthenticated blocked", r.get("statusCode") == 401,
          f"status={r.get('statusCode')}")

    token = login_token()
    r = invoke("coolcrypto-market", http_event("/market/metrics/manifest", token))
    check("manifest served", r.get("statusCode") == 200
          and r.get("parsed", {}).get("rows", 0) > 0,
          json.dumps(r.get("parsed", {}))[:160])

    r = invoke("coolcrypto-market",
               http_event("/market/market_stream/partition/2026-08", token))
    loc = (r.get("headers") or {}).get("Location", "")
    # A 302 to S3, not a proxied body: a month of metrics exceeds API
    # Gateway's 10 MB response cap and would bill for double transfer.
    # Accept either signature scheme -- MinIO signs with SigV2 (`Signature=`),
    # real S3 with SigV4 (`X-Amz-Signature=`).
    signed = "X-Amz-Signature=" in loc or "Signature=" in loc
    check("partition presigned, not proxied",
          r.get("statusCode") == 302 and signed,
          f"status={r.get('statusCode')} loc={loc[:80]}")

    r = invoke("coolcrypto-market",
               http_event("/market/market_stream/partition/not-a-month", token))
    check("bad partition key rejected", r.get("statusCode") == 400)

    r = invoke("coolcrypto-market",
               http_event("/market/auth_users/manifest", token))
    check("non-market table rejected", r.get("statusCode") == 404)

    r = invoke("coolcrypto-market",
               http_event("/market/organism_snapshots/index", token))
    check("blob index presigned", r.get("statusCode") == 302)

    r = invoke("coolcrypto-market",
               http_event("/market/organism_snapshots/blob/1", token))
    check("blob served via API", r.get("statusCode") == 200
          and isinstance(r.get("parsed"), dict),
          f"status={r.get('statusCode')}")

    # -- 6. object economics -------------------------------------------
    print("\n[6] Object count")
    s3 = boto3.client("s3", endpoint_url="http://localhost:9000",
                      aws_access_key_id="minioadmin",
                      aws_secret_access_key="minioadmin",
                      region_name="us-east-1")
    pages = s3.get_paginator("list_objects_v2")
    parquet = sum(
        1
        for page in pages.paginate(Bucket="coolcrypto-hybrid",
                                   Prefix="database/market/")
        for o in page.get("Contents", [])
        if o["Key"].endswith(".parquet")
    )
    # 3.4M rows must not become 3.4M objects -- that is the whole point of
    # partitioning, and the difference between cents and hundreds of dollars.
    check("3.4M rows fit in <100 parquet objects", parquet < 100,
          f"{parquet} objects")

    print("\n" + "=" * 62)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"    FAILED: {f}")
    print("=" * 62)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
