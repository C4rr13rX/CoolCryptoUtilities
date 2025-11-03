from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import aiohttp

from db import get_db


CallbackType = Union[Callable[[Dict[str, Any]], Awaitable[None]], Callable[[Dict[str, Any]], None]]


class MarketDataStream:
    """
    Handles live websocket subscriptions (when available) or falls back to a
    lightweight simulation so downstream systems always receive data.
    Each sample is persisted to the shared TradingDatabase for later training.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        symbol: str = "ETH-USD",
        chain: str = "ethereum",
        simulation_interval: float = 2.0,
    ) -> None:
        self.url = url or os.getenv("MARKET_WS_URL")
        self.symbol = symbol
        self.chain = chain
        self.simulation_interval = simulation_interval
        self._callbacks: List[CallbackType] = []
        self._db = get_db()
        self._session: Optional[aiohttp.ClientSession] = None
        self._stop_event = asyncio.Event()

    def register(self, callback: CallbackType) -> None:
        self._callbacks.append(callback)

    async def start(self) -> None:
        self._stop_event.clear()
        if self.url:
            try:
                await self._consume_ws()
                return
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"[market-stream] websocket failed ({exc}); falling back to simulation.")
        await self._simulate()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._session:
            await self._session.close()
            self._session = None

    async def _consume_ws(self) -> None:  # pragma: no cover - requires network
        async with aiohttp.ClientSession() as session:
            self._session = session
            async with session.ws_connect(self.url) as ws:
                subscribe_msg = os.getenv("MARKET_WS_SUBSCRIBE")
                if subscribe_msg:
                    await ws.send_str(subscribe_msg)
                while not self._stop_event.is_set():
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        sample = self._normalize_payload(data)
                        await self._dispatch(sample)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

    async def _simulate(self) -> None:
        price = random.uniform(1500.0, 2500.0)
        volatility = 0.015
        while not self._stop_event.is_set():
            drift = random.uniform(-volatility, volatility) * price
            price = max(price + drift, 0.01)
            volume = random.uniform(5.0, 50.0)
            sample = {
                "ts": time.time(),
                "symbol": self.symbol,
                "chain": self.chain,
                "price": price,
                "volume": volume,
                "simulated": True,
            }
            await self._dispatch(sample)
            await asyncio.sleep(self.simulation_interval)

    async def _dispatch(self, sample: Dict[str, Any]) -> None:
        self._db.insert_market_sample(
            chain=sample.get("chain", self.chain),
            symbol=sample.get("symbol", self.symbol),
            price=float(sample.get("price") or 0),
            volume=float(sample.get("volume") or 0),
            raw=sample,
        )
        for callback in list(self._callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):  # type: ignore[arg-type]
                    await callback(sample)  # type: ignore[misc]
                else:
                    callback(sample)  # type: ignore[misc]
            except Exception as exc:
                print(f"[market-stream] callback error: {exc}")

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        price = float(payload.get("price") or payload.get("p") or 0)
        volume = float(payload.get("volume") or payload.get("v") or 0)
        symbol = payload.get("symbol") or payload.get("s") or self.symbol
        ts = float(payload.get("ts") or payload.get("time") or time.time())
        return {
            "ts": ts,
            "symbol": symbol,
            "chain": self.chain,
            "price": price,
            "volume": volume,
            "raw": payload,
        }

