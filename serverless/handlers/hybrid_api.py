"""
Hybrid data API (API Gateway -> Lambda -> S3).

The server half of the model: a thin, stateless REST facade over the
S3-as-database layer.  The browser's AllezORM mirror is the fast path, so this
function only runs on first load, on writes, and when the change feed says
something moved.

    GET    /hybrid/<table>            list rows
    GET    /hybrid/<table>/_change    current change sequence (cheap poll)
    GET    /hybrid/<table>/<id>       one row
    POST   /hybrid/<table>            insert (server allocates the id)
    PUT    /hybrid/<table>/<id>       update
    DELETE /hybrid/<table>/<id>       soft delete

Like ``auth.py`` this imports no Django: a data read should not pay for the
app registry, and keeping the function small is what keeps it in the free tier.

Every route requires a valid session token.  There is no anonymous read path --
the tables here are operator data, and an open endpoint is both a disclosure
risk and a way for a stranger to run up an S3 bill.
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

from serverless.hybrid.pq_auth import AuthError, AuthService  # noqa: E402
from serverless.hybrid.smart_storage import get_storage  # noqa: E402

logger = logging.getLogger("serverless.hybrid_api")
logging.getLogger().setLevel(os.getenv("DJANGO_LOG_LEVEL", "INFO"))

# Only these tables are reachable. An allowlist rather than a pattern: without
# it, a caller could address `auth_users` and read password hashes, or invent
# a table name and write arbitrary objects into the bucket.
ALLOWED_TABLES = {
    t.strip()
    for t in os.getenv(
        "HYBRID_TABLES",
        "watchlists,notes,dashboards,alerts,annotations,preferences",
    ).split(",")
    if t.strip()
}

# Tables that hold credentials, never routable regardless of ALLOWED_TABLES.
BLOCKED_TABLES = {"auth_users", "auth_challenges", "auth_sessions"}

_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_BODY_BYTES = int(os.getenv("HYBRID_MAX_BODY", "262144"))  # 256 KiB


def _headers() -> dict:
    out = {"Content-Type": "application/json", "Cache-Control": "no-store"}
    origin = os.getenv("PQ_ALLOWED_ORIGIN", "")
    if origin:
        out["Access-Control-Allow-Origin"] = origin
        out["Access-Control-Allow-Credentials"] = "true"
        out["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        out["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    return out


def _respond(status: int, body) -> dict:
    return {"statusCode": status, "headers": _headers(),
            "body": json.dumps(body, default=str)}


def _method(event: dict) -> str:
    ctx = (event.get("requestContext") or {}).get("http") or {}
    return (ctx.get("method") or event.get("httpMethod") or "GET").upper()


def _path(event: dict) -> str:
    ctx = (event.get("requestContext") or {}).get("http") or {}
    return ctx.get("path") or event.get("rawPath") or event.get("path") or ""


def _body(event: dict) -> dict:
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    if len(raw) > MAX_BODY_BYTES:
        raise ValueError("request body too large")
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON body") from exc
    return parsed if isinstance(parsed, dict) else {}


def _bearer(event: dict) -> str:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    auth = headers.get("authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else ""


def _sanitize(row: dict) -> dict:
    """Strip internal bookkeeping before a row leaves the server."""
    return {k: v for k, v in row.items() if not k.startswith("_")}


def lambda_handler(event, context):
    storage = get_storage()
    # A warm sandbox must not serve rows memoised for a different user.
    storage.reset_memo()

    method = _method(event)
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": _headers(), "body": ""}

    # --- authentication (before any parsing or S3 work) ---
    try:
        AuthService(storage=storage).check_session(_bearer(event))
    except AuthError as exc:
        return _respond(401, {"error": str(exc), "code": exc.code})

    # --- routing ---
    path = _path(event)
    parts = [p for p in path.split("/") if p]
    if "hybrid" in parts:
        parts = parts[parts.index("hybrid") + 1:]
    if not parts:
        return _respond(404, {"error": "no table in path"})

    table, rest = parts[0], parts[1:]
    if not _TABLE_RE.match(table):
        return _respond(400, {"error": "invalid table name"})
    if table in BLOCKED_TABLES or table not in ALLOWED_TABLES:
        # One message for both cases: distinguishing "blocked" from "unknown"
        # confirms which internal tables exist.
        return _respond(404, {"error": "unknown table"})

    started = time.time()
    try:
        # change-feed poll: one small GET the browser uses to skip a full fetch
        if rest and rest[0] == "_change":
            return _respond(200, {"seq": storage.change_seq(table)})

        if method == "GET" and not rest:
            qs = event.get("queryStringParameters") or {}
            limit = min(int(qs.get("limit") or 0) or 500, 500)
            offset = max(int(qs.get("offset") or 0), 0)
            rows = storage.list_table(table, limit=limit, offset=offset)
            return _respond(200, {
                "items": [_sanitize(r) for r in rows],
                "total": len(rows),
                "seq": storage.change_seq(table),
            })

        if method == "GET" and rest:
            row = storage.get_row(table, rest[0])
            return (_respond(200, _sanitize(row)) if row
                    else _respond(404, {"error": "not found"}))

        if method == "POST" and not rest:
            payload = _body(event)
            # `id` is the server's to assign; accepting one would let a client
            # overwrite an unrelated row.
            payload.pop("id", None)
            created = storage.insert_row(table, payload)
            return _respond(201, _sanitize(created))

        if method == "PUT" and rest:
            ident = rest[0]
            existing = storage.get_row(table, ident)
            if not existing:
                return _respond(404, {"error": "not found"})
            payload = _body(event)
            payload.pop("id", None)
            merged = {**existing, **payload, "id": existing.get("id", ident)}
            storage.put_row(table, ident, merged)
            storage.record_change(table, {"op": "update", "id": ident})
            return _respond(200, _sanitize(merged))

        if method == "DELETE" and rest:
            storage.delete_row(table, rest[0])
            return _respond(204, {})

        return _respond(405, {"error": f"{method} not allowed here"})

    except ValueError as exc:
        return _respond(400, {"error": str(exc)})
    except Exception:  # noqa: BLE001
        logger.exception("hybrid api error table=%s method=%s", table, method)
        return _respond(500, {"error": "internal error"})
    finally:
        logger.info(
            "hybrid table=%s %s ms=%d gets=%d puts=%d",
            table, method, int((time.time() - started) * 1000),
            storage.stats.s3_gets, storage.stats.s3_puts,
        )
