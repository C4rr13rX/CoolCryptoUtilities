from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
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


class AppStateConsumer(AsyncJsonWebsocketConsumer):
    """Push coherent dashboard + wallet revisions, with liveness heartbeats."""

    async def connect(self):
        user = self.scope.get("user")
        if user is None or getattr(user, "is_anonymous", True):
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

    @staticmethod
    def _build_payload():
        from services.wallet_state import load_wallet_state
        from telemetry.views import build_dashboard_summary

        summary = build_dashboard_summary()
        wallet_snapshot = dict(load_wallet_state() or {})
        reconciliation = dict(summary.get("wallet") or {})
        wallet_snapshot["reconciliation"] = reconciliation
        wallet_snapshot["totals"] = dict(wallet_snapshot.get("totals") or {})
        wallet_snapshot["totals"]["cached_usd"] = reconciliation.get("cached_total_usd")
        wallet_snapshot["totals"]["usd"] = reconciliation.get("total_usd")

        # Volatile clock fields are not state revisions. Excluding them keeps
        # heartbeat frames lightweight while every substantive change still
        # produces an immediate full snapshot.
        revision_material = dict(summary)
        operational = dict(revision_material.get("operational_state") or {})
        operational.pop("generated_at", None)
        if isinstance(operational.get("wallet"), dict):
            operational["wallet"] = dict(operational["wallet"])
            operational["wallet"].pop("age_seconds", None)
        revision_material["operational_state"] = operational
        if isinstance(revision_material.get("wallet"), dict):
            revision_material["wallet"] = dict(revision_material["wallet"])
            revision_material["wallet"].pop("age_seconds", None)
        encoded = json.dumps(revision_material, sort_keys=True, default=str, separators=(",", ":"))
        revision = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
        return revision, summary, wallet_snapshot, reconciliation

    async def _push(self, force=False):
        try:
            revision, summary, wallet_snapshot, reconciliation = await sync_to_async(
                self._build_payload,
                thread_sensitive=True,
            )()
        except Exception as exc:
            now = time.time()
            if force or now - self._last_heartbeat >= 15.0:
                self._last_heartbeat = now
                await self.send_json({
                    "type": "app.heartbeat",
                    "timestamp": now,
                    "healthy": False,
                    "detail": type(exc).__name__,
                })
            return

        now = time.time()
        changed = revision != self._last_revision
        heartbeat_due = now - self._last_heartbeat >= 15.0
        if not (force or changed or heartbeat_due):
            return
        self._last_revision = revision
        self._last_heartbeat = now
        payload = {
            "type": "app.snapshot" if changed or force else "app.heartbeat",
            "timestamp": now,
            "healthy": True,
            "revision": revision,
        }
        if changed or force:
            payload.update({
                "summary": summary,
                "wallet_snapshot": wallet_snapshot,
                "reconciliation": reconciliation,
            })
        await self.send_json(payload)
