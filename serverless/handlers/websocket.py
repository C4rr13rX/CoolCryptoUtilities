"""
API Gateway WebSocket API -> Django handler.

The three consumers in ``opsconsole/consumers.py`` are built on a model Lambda
cannot host: ``connect()`` accepts the socket, then spawns an asyncio task that
loops forever, sleeps, and pushes a fresh snapshot.  That requires a process
that stays alive between frames.  Lambda freezes the sandbox the moment a
handler returns.

So the responsibility is inverted:

* **API Gateway owns the connection.**  It stays open independently of any
  Lambda and invokes us on $connect / $disconnect / message routes only.
* **Connection IDs are persisted in S3** (via the hybrid store) so a later
  invocation -- in a different sandbox -- can find who to talk to. They cannot
  live in Django's database: under the hybrid model that is a scratch SQLite
  file in /tmp, private to one sandbox and empty on every cold start.
* **The push loop becomes a scheduled Lambda.**  ``broadcast_handler`` is what
  the consumers' ``_stream()`` used to be -- EventBridge fires it on an
  interval, it builds the same payloads and posts them to every live
  connection through the management API.

The payload builders below are the consumers' ``_payload``/``_push_snapshot``
logic, reused verbatim so the wire format the frontend already parses does not
change.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings_lambda"
)

import boto3  # noqa: E402
from serverless.hybrid.smart_storage import get_storage  # noqa: E402

# Table in the S3 store holding live API Gateway connection ids.
WS_TABLE = "ws_connections"
# Redirect the bundle's write targets to /tmp before ANY services.* module
# is imported -- several create directories at import time and /var/task is
# read-only. Must run before django.setup() pulls in the app registry.
from serverless.bootstrap import prepare_writable_dirs  # noqa: E402

prepare_writable_dirs(ROOT)

import django  # noqa: E402

django.setup()

logger = logging.getLogger("serverless.websocket")

# Route key -> the consumer channel it stands in for. Mirrors the paths in
# opsconsole/routing.py so the frontend can keep using one socket per stream.
CHANNELS = {
    "console.logs": "ws/console/logs/",
    "wallet.state": "ws/wallet/state/",
    "app.state": "ws/app/state/",
}


def _management_client(event):
    """Build the client used to push frames back into an open connection."""
    endpoint = os.getenv("WEBSOCKET_MANAGEMENT_ENDPOINT")
    if not endpoint:
        ctx = (event or {}).get("requestContext", {})
        domain, stage = ctx.get("domainName"), ctx.get("stage")
        endpoint = f"https://{domain}/{stage}" if domain and stage else None
    return boto3.client(
        "apigatewaymanagementapi",
        endpoint_url=endpoint,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


# ---------------------------------------------------------------------------
# Payload builders -- the synchronous equivalents of the consumers' _payload().
# ---------------------------------------------------------------------------
def _console_payload() -> dict:
    from opsconsole.manager import manager

    return {
        "timestamp": time.time(),
        "lines": manager.tail(200),
        "status": manager.status(),
    }


def _wallet_payload() -> dict:
    from services.wallet_reconciliation import (
        reconciled_wallet_snapshot,
        request_wallet_refresh,
    )
    from services.wallet_state import load_wallet_state

    reconciliation = reconciled_wallet_snapshot("guardian")
    if not reconciliation.get("fresh"):
        request_wallet_refresh()
    snapshot = dict(load_wallet_state() or {})
    snapshot["reconciliation"] = reconciliation
    snapshot["totals"] = dict(snapshot.get("totals") or {})
    snapshot["totals"]["cached_usd"] = reconciliation.get("cached_total_usd")
    snapshot["totals"]["usd"] = reconciliation.get("total_usd")
    revision = (
        f"{reconciliation.get('updated_epoch') or 0}:{reconciliation.get('status')}"
    )
    return {
        "type": "wallet.snapshot",
        "timestamp": time.time(),
        "revision": revision,
        "snapshot": snapshot,
        "reconciliation": reconciliation,
    }


PAYLOAD_BUILDERS = {
    "console.logs": _console_payload,
    "wallet.state": _wallet_payload,
    "app.state": _wallet_payload,
}


# ---------------------------------------------------------------------------
# $connect / $disconnect / default
# ---------------------------------------------------------------------------
def _connections(storage) -> list[dict]:
    """Every registered connection. Keyed by connection id, not a counter."""
    manifest = storage.keys_manifest(WS_TABLE) or []
    rows = []
    for ident in manifest:
        rec = storage.get_row(WS_TABLE, ident)
        if rec:
            rows.append(rec)
    return rows


def _register(storage, connection_id: str, channel: str) -> None:
    storage.put_row(WS_TABLE, connection_id, {
        "id": connection_id, "channel": channel, "connected_at": time.time(),
    })
    manifest = storage.keys_manifest(WS_TABLE) or []
    if connection_id not in manifest:
        manifest.append(connection_id)
        storage.set_keys_manifest(WS_TABLE, manifest)


def _unregister(storage, connection_id: str) -> None:
    manifest = [k for k in (storage.keys_manifest(WS_TABLE) or [])
                if k != connection_id]
    storage.set_keys_manifest(WS_TABLE, manifest)
    storage.delete_row(WS_TABLE, connection_id)


def lambda_handler(event, context):
    storage = get_storage()
    storage.reset_memo()

    ctx = (event or {}).get("requestContext", {})
    route = ctx.get("routeKey")
    connection_id = ctx.get("connectionId")

    if route == "$connect":
        # The channel travels as a query string param because API Gateway
        # WebSocket APIs route on a key, not a path like Channels does.
        params = (event or {}).get("queryStringParameters") or {}
        channel = params.get("channel", "console.logs")
        if channel not in CHANNELS:
            return {"statusCode": 400, "body": f"unknown channel {channel}"}
        _register(storage, connection_id, channel)
        logger.info("ws connect id=%s channel=%s", connection_id, channel)
        return {"statusCode": 200, "body": "connected"}

    if route == "$disconnect":
        _unregister(storage, connection_id)
        logger.info("ws disconnect id=%s", connection_id)
        return {"statusCode": 200, "body": "disconnected"}

    # Default route: a client asking for an immediate snapshot rather than
    # waiting for the next scheduled broadcast.
    row = storage.get_row(WS_TABLE, connection_id)
    if row is None:
        return {"statusCode": 410, "body": "unknown connection"}

    builder = PAYLOAD_BUILDERS.get(row.get("channel"))
    if builder is None:
        return {"statusCode": 400, "body": "no builder"}

    client = _management_client(event)
    try:
        client.post_to_connection(
            ConnectionId=connection_id,
            Data=json.dumps(builder(), default=str).encode(),
        )
    except client.exceptions.GoneException:
        _unregister(storage, connection_id)
    return {"statusCode": 200, "body": "ok"}


# ---------------------------------------------------------------------------
# Scheduled fan-out -- the replacement for the consumers' _stream() loop.
# ---------------------------------------------------------------------------
def broadcast_handler(event, context):
    storage = get_storage()
    storage.reset_memo()

    client = _management_client(event)
    sent = stale = 0
    all_rows = _connections(storage)

    for channel, builder in PAYLOAD_BUILDERS.items():
        rows = [r for r in all_rows if r.get("channel") == channel]
        if not rows:
            continue
        # Build once per channel, not once per connection: the payload is
        # identical for every subscriber and _wallet_payload() hits the DB and
        # the reconciliation service.
        try:
            data = json.dumps(builder(), default=str).encode()
        except Exception:  # noqa: BLE001
            logger.exception("payload build failed for channel=%s", channel)
            continue

        for row in rows:
            cid = row.get("id")
            try:
                client.post_to_connection(ConnectionId=cid, Data=data)
                sent += 1
            except client.exceptions.GoneException:
                # Client vanished without a $disconnect (network drop, tab
                # close during a deploy). Reap it so the table does not grow
                # unbounded and we stop paying for dead pushes.
                _unregister(storage, cid)
                stale += 1
            except Exception:  # noqa: BLE001
                logger.exception("push failed id=%s", cid)

    logger.info("broadcast sent=%d stale=%d", sent, stale)
    return {"status": "ok", "sent": sent, "stale": stale}
