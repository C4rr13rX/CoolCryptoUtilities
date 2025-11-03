from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from db import TradingDatabase, get_db
from trading.data_stream import MarketDataStream
from trading.pipeline import TrainingPipeline


class TradingBot:
    """
    High-level orchestrator that ties together the market stream, the training
    pipeline, and queue-based trade execution. Real sending of transactions is
    intentionally outside the scope to keep this module simulation-friendly.
    """

    def __init__(
        self,
        *,
        db: Optional[TradingDatabase] = None,
        stream: Optional[MarketDataStream] = None,
        pipeline: Optional[TrainingPipeline] = None,
        window_size: int = 60,
    ) -> None:
        self.db = db or get_db()
        self.pipeline = pipeline or TrainingPipeline(db=self.db)
        self.stream = stream or MarketDataStream()
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self.queue: List[Dict[str, Any]] = []
        self._bg_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.stream.register(self._handle_sample)
        await asyncio.gather(
            self.stream.start(),
            self._start_background_refinement(),
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._bg_task:
            self._bg_task.cancel()
        await self.stream.stop()

    async def _handle_sample(self, sample: Dict[str, Any]) -> None:
        self._buffer.append(sample)
        if len(self._buffer) < self.window_size:
            return  # wait until window filled

        model = self.pipeline.ensure_active_model()
        inputs = self._prepare_inputs(list(self._buffer))
        try:
            preds = model.predict(inputs, verbose=0)
        except Exception as exc:
            print(f"[trading-bot] prediction failed: {exc}")
            return
        decision = self._interpret_predictions(preds, sample)
        self.queue.append(decision)
        self.db.log_trade(
            wallet=decision.get("wallet", "ghost"),
            chain=decision.get("chain", sample.get("chain", "ethereum")),
            symbol=decision.get("symbol", sample.get("symbol", "asset")),
            action="queue",
            status=decision.get("status", "ghost"),
            details=decision,
        )

    def _prepare_inputs(self, window: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        prices = np.array([float(row.get("price", 0.0)) for row in window], dtype=np.float32)
        volumes = np.array([float(row.get("volume", 0.0)) for row in window], dtype=np.float32)
        price_vol = np.stack([prices, volumes], axis=-1).reshape(1, self.window_size, 2)

        sentiment = np.zeros((1, self.pipeline.sent_seq_len, 1), dtype=np.float32)
        tech = np.zeros((1, self.pipeline.tech_count), dtype=np.float32)
        hour = np.array([[int(time.gmtime(row.get("ts", time.time())).tm_hour) for row in window][-1]], dtype=np.int32)
        hour = hour.reshape(1, 1)
        dow = np.array([[int(time.gmtime(row.get("ts", time.time())).tm_wday) for row in window][-1]], dtype=np.int32)
        dow = dow.reshape(1, 1)
        gas = np.full((1, 1), 0.0015, dtype=np.float32)
        tax = np.full((1, 1), 0.005, dtype=np.float32)

        headline = f"{window[-1].get('symbol', 'asset')} price {window[-1].get('price', 0)}"
        full_text = str(window[-1].get("raw", ""))[:512]

        inputs = {
            "price_vol_input": price_vol,
            "sentiment_seq": sentiment,
            "headline_text": np.array([[headline]], dtype=object),
            "full_text": np.array([[full_text]], dtype=object),
            "tech_input": tech,
            "hour_input": hour,
            "dow_input": dow,
            "gas_fee_input": gas,
            "tax_rate_input": tax,
        }
        return inputs

    def _interpret_predictions(self, preds, sample: Dict[str, Any]) -> Dict[str, Any]:
        exit_conf, price_mu, price_log_var, price_dir, net_margin, net_pnl, tech_recon = preds
        exit_conf_val = float(exit_conf[0][0])
        direction_prob = float(price_dir[0][0])
        delta = float(price_mu[0][0])
        margin = float(net_margin[0][0])
        pnl = float(net_pnl[0][0])

        decision = {
            "timestamp": sample.get("ts", time.time()),
            "symbol": sample.get("symbol", "asset"),
            "chain": sample.get("chain", "ethereum"),
            "exit_confidence": exit_conf_val,
            "direction_prob": direction_prob,
            "expected_delta": delta,
            "net_margin": margin,
            "net_pnl": pnl,
            "status": "ghost" if exit_conf_val < 0.8 else "candidate",
        }
        return decision

    async def _start_background_refinement(self, cadence: float = 900.0) -> None:
        async def _loop():
            while self._running:
                await asyncio.sleep(cadence)
                try:
                    await asyncio.to_thread(self.pipeline.train_candidate)
                except Exception as exc:
                    print(f"[trading-bot] background refinement error: {exc}")

        self._bg_task = asyncio.create_task(_loop())
        await self._bg_task

    def dequeue(self) -> Optional[Dict[str, Any]]:
        if not self.queue:
            return None
        return self.queue.pop(0)

    def pending_trades(self) -> List[Dict[str, Any]]:
        return list(self.queue)

