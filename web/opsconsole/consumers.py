from __future__ import annotations

import asyncio
import contextlib
import time

from channels.generic.websocket import AsyncWebsocketConsumer

from .manager import manager


class ConsoleLogConsumer(AsyncWebsocketConsumer):
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
