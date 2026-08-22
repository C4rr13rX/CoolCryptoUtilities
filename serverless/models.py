from __future__ import annotations

from django.db import models


class WebSocketConnection(models.Model):
    """
    Legacy: live API Gateway WebSocket connections.

    **No longer used by the runtime.** Connection ids now live in the S3
    hybrid store (``ws_connections``), because under the hybrid model Django's
    database is a scratch SQLite file in ``/tmp`` -- private to one sandbox and
    empty on every cold start. The ``$connect`` invocation and the scheduled
    broadcaster run in different sandboxes, so a per-sandbox table could never
    let one find the connections registered by the other.

    Kept so the existing migration still applies cleanly on the Postgres
    fallback path (``HYBRID_DB=0``). See ``serverless/handlers/websocket.py``
    for the store that replaced it.
    """

    connection_id = models.CharField(max_length=128, unique=True, db_index=True)
    channel = models.CharField(max_length=64, db_index=True)
    connected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["channel", "connected_at"])]

    def __str__(self) -> str:
        return f"{self.channel}:{self.connection_id}"
