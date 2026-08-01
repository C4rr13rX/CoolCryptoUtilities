from __future__ import annotations

import asyncio
import contextlib
import time

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from .manager import manager


class ConsoleLogConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        await self.accept()
        self._active = True
        self._last_payload: list[str] = []
        await self._push_snapshot()
        self._worker = asyncio.create_task(self._stream())

    async def disconnect(self, code):
        self._active = False
        if hasattr(self, "_worker"):
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    async def _stream(self):
        while self._active:
            await asyncio.sleep(2.0)
            await self._push_snapshot()

    async def _push_snapshot(self):
        lines = manager.tail(200)
        if lines != self._last_payload:
            self._last_payload = lines
            await self.send_json({
                "timestamp": time.time(),
                "lines": lines,
                "status": manager.status(),
            })


class WalletStateConsumer(AsyncJsonWebsocketConsumer):
    """Broadcast complete wallet revisions; emit heartbeat frames otherwise."""

    async def connect(self):
        user = self.scope.get("user")
        if user is not None and getattr(user, "is_anonymous", False):
            await self.close(code=4401)
            return
        await self.accept()
        self._active = True
        self._last_revision = ""
        self._last_heartbeat = 0.0
        await self._push(force=True)
        self._worker = asyncio.create_task(self._stream())

    async def disconnect(self, code):
        self._active = False
        if hasattr(self, "_worker"):
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker

    async def _stream(self):
        while self._active:
            await asyncio.sleep(1.0)
            await self._push()

    async def _payload(self):
        from services.wallet_reconciliation import (
            reconciled_wallet_snapshot,
            request_wallet_refresh,
        )
        from services.wallet_state import load_wallet_state

        reconciliation = await sync_to_async(reconciled_wallet_snapshot)("guardian")
        if not reconciliation.get("fresh"):
            await sync_to_async(request_wallet_refresh)()
        snapshot = await sync_to_async(load_wallet_state)()
        snapshot = dict(snapshot or {})
        snapshot["reconciliation"] = reconciliation
        snapshot["totals"] = dict(snapshot.get("totals") or {})
        snapshot["totals"]["cached_usd"] = reconciliation.get("cached_total_usd")
        snapshot["totals"]["usd"] = reconciliation.get("total_usd")
        revision = f"{reconciliation.get('updated_epoch') or 0}:{reconciliation.get('status')}"
        return revision, snapshot, reconciliation

    async def _push(self, force=False):
        revision, snapshot, reconciliation = await self._payload()
        now = time.time()
        changed = revision != self._last_revision
        heartbeat_due = now - self._last_heartbeat >= 15.0
        if not (force or changed or heartbeat_due):
            return
        self._last_revision = revision
        self._last_heartbeat = now
        await self.send_json({
            "type": "wallet.snapshot" if changed or force else "wallet.heartbeat",
            "timestamp": now,
            "revision": revision,
            "snapshot": snapshot,
            "reconciliation": reconciliation,
        })
