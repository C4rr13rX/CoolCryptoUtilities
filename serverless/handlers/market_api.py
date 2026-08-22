"""
Shared market-data API (API Gateway -> Lambda -> S3).

Serves the Parquet partitions the browser mirrors into AllezORM.  Like the
other hybrid handlers it imports no Django, so it cold-starts in ~0.3s.

    GET /market/<table>/manifest            partitions, row counts, time ranges
    GET /market/<table>/partition/<YYYY-MM>  one Parquet month
    GET /market/organism_snapshots/index     the lightweight blob index
    GET /market/organism_snapshots/blob/<id> one snapshot payload

Everything here is **shared across accounts** -- there is no per-user data, so
no ownership filtering. A session is still required: this is not public data,
and an open endpoint would let a stranger run up the S3 bill.

Partitions are served as a redirect to a presigned URL rather than proxied
through Lambda. A month of metrics is several MB, and API Gateway caps a
response at 10 MB (6 MB for the Lambda payload) -- proxying would both hit that
ceiling and bill for transferring every byte twice.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from serverless.hybrid.market_store import (  # noqa: E402
    BLOB_TABLES,
    PARTITIONED_TABLES,
    MarketStore,
)
from serverless.hybrid.pq_auth import AuthError, AuthService  # noqa: E402
from serverless.hybrid.smart_storage import get_storage  # noqa: E402

logger = logging.getLogger("serverless.market_api")
logging.getLogger().setLevel(os.getenv("DJANGO_LOG_LEVEL", "INFO"))

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# How long a presigned partition URL stays valid. Long enough for a slow
# connection to finish a multi-MB download, short enough that a leaked URL
# stops working quickly.
PRESIGN_TTL = int(os.getenv("MARKET_PRESIGN_TTL", "900"))


def _headers(extra: dict | None = None) -> dict:
    out = {"Content-Type": "application/json"}
    origin = os.getenv("PQ_ALLOWED_ORIGIN", "")
    if origin:
        out["Access-Control-Allow-Origin"] = origin
        out["Access-Control-Allow-Credentials"] = "true"
        out["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        out["Access-Control-Allow-Methods"] = "GET,OPTIONS"
    out.update(extra or {})
    return out


def _respond(status: int, body, extra_headers: dict | None = None) -> dict:
    return {
        "statusCode": status,
        "headers": _headers(extra_headers),
        "body": json.dumps(body, default=str),
    }


def _path(event: dict) -> str:
    ctx = (event.get("requestContext") or {}).get("http") or {}
    return ctx.get("path") or event.get("rawPath") or event.get("path") or ""


def _bearer(event: dict) -> str:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    auth = headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def lambda_handler(event, context):
    storage = get_storage()
    storage.reset_memo()

    ctx = (event.get("requestContext") or {}).get("http") or {}
    method = (ctx.get("method") or event.get("httpMethod") or "GET").upper()
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": _headers(), "body": ""}
    if method != "GET":
        return _respond(405, {"error": "read-only endpoint"})

    try:
        AuthService(storage=storage).check_session(_bearer(event))
    except AuthError as exc:
        return _respond(401, {"error": str(exc), "code": exc.code})

    parts = [p for p in _path(event).split("/") if p]
    if "market" in parts:
        parts = parts[parts.index("market") + 1:]
    if len(parts) < 2:
        return _respond(404, {"error": "expected /market/<table>/<action>"})

    table, action, rest = parts[0], parts[1], parts[2:]
    known = set(PARTITIONED_TABLES) | set(BLOB_TABLES)
    if table not in known:
        # One message whether the table is unknown or merely not shared, so
        # this cannot be used to enumerate internal table names.
        return _respond(404, {"error": "unknown table"})

    store = MarketStore(storage)
    started = time.time()
    try:
        if action == "manifest":
            manifest = store.read_manifest(table)
            return _respond(200, manifest, {
                # Closed partitions never change; the client revalidates the
                # current month via the manifest itself.
                "Cache-Control": "public, max-age=60",
            })

        if action == "partition":
            if not rest or not _MONTH_RE.match(rest[0]):
                return _respond(400, {"error": "expected partition/YYYY-MM"})
            url = storage.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": storage.bucket,
                        "Key": store.partition_key(table, rest[0])},
                ExpiresIn=PRESIGN_TTL,
            )
            # 302 to S3: the browser streams the Parquet straight from storage
            # instead of pushing multi-MB bodies through Lambda.
            return {"statusCode": 302,
                    "headers": _headers({"Location": url}), "body": ""}

        if action == "index" and table in BLOB_TABLES:
            url = storage.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": storage.bucket,
                        "Key": store.blob_index_key(table)},
                ExpiresIn=PRESIGN_TTL,
            )
            return {"statusCode": 302,
                    "headers": _headers({"Location": url}), "body": ""}

        if action == "blob" and table in BLOB_TABLES:
            if not rest or not _ID_RE.match(rest[0]):
                return _respond(400, {"error": "invalid blob id"})
            payload = store.read_blob(table, rest[0])
            if payload is None:
                return _respond(404, {"error": "not found"})
            return _respond(200, payload)

        return _respond(404, {"error": f"unknown action {action}"})

    except Exception:  # noqa: BLE001
        logger.exception("market api error table=%s action=%s", table, action)
        return _respond(500, {"error": "internal error"})
    finally:
        logger.info("market %s/%s ms=%d gets=%d", table, action,
                    int((time.time() - started) * 1000), storage.stats.s3_gets)
