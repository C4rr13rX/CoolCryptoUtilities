from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

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
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.stable_bank: float = 0.0
        self.total_profit: float = 0.0
        self.realized_profit: float = 0.0
        self.total_trades: int = 0
        self.wins: int = 0
        self.max_trade_share: float = 0.15
        self.stable_checkpoint_ratio: float = 0.15
        self.bus_routes: Dict[str, List[str]] = {}
        self.stable_tokens = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD"}
        self._load_state()

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
        if decision and decision.get("action") != "hold":
            self.queue.append(decision)
            self.db.log_trade(
                wallet=decision.get("wallet", "ghost"),
                chain=decision.get("chain", sample.get("chain", "ethereum")),
                symbol=decision.get("symbol", sample.get("symbol", "asset")),
                action=decision.get("action", "queue"),
                status=decision.get("status", "ghost"),
                details=decision,
            )
            self._save_state()

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
        try:
            asset_id = int(self.pipeline.data_loader._get_asset_id(window[-1].get("symbol", "SIM")))  # type: ignore[attr-defined]
        except Exception:
            asset_id = 0
        asset = np.array([[asset_id]], dtype=np.int32)

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
            "asset_id_input": asset,
        }
        return inputs

    def _interpret_predictions(self, preds, sample: Dict[str, Any]) -> Dict[str, Any]:
        exit_conf, price_mu, price_log_var, price_dir, net_margin, net_pnl, tech_recon = preds
        exit_conf_val = float(exit_conf[0][0])
        direction_prob = float(price_dir[0][0])
        delta = float(price_mu[0][0])
        margin = float(net_margin[0][0])
        pnl = float(net_pnl[0][0])

        symbol = sample.get("symbol", "asset")
        price = float(sample.get("price", 0.0))
        volume = float(sample.get("volume", 0.0))
        route = self.bus_routes.get(symbol) or symbol.split("-")
        route = [t.upper() for t in route if t]
        if not route:
            route = [symbol.upper()]
        stable_target = next((tok for tok in route if tok.upper() in self.stable_tokens), "USDC")
        pos = self.positions.get(symbol)
        fees = 0.0015 + 0.005

        decision: Dict[str, Any] = {
            "timestamp": sample.get("ts", time.time()),
            "symbol": symbol,
            "chain": sample.get("chain", "ethereum"),
            "exit_confidence": exit_conf_val,
            "direction_prob": direction_prob,
            "expected_delta": delta,
            "net_margin": margin,
            "net_pnl": pnl,
            "status": "ghost",
            "action": "hold",
            "wallet": "ghost",
        }

        trade_size = max(min(volume * self.max_trade_share, volume), 0.0)
        trade_size = min(trade_size, 100.0)
        if trade_size <= 0.0:
            return decision

        if pos is None:
            if direction_prob > 0.58 and exit_conf_val >= 0.6 and margin > fees * 1.1:
                self.positions[symbol] = {
                    "entry_price": price,
                    "size": trade_size,
                    "ts": sample.get("ts", time.time()),
                    "route": route,
                    "bus_index": 0,
                }
                decision.update(
                    {
                        "action": "enter",
                        "status": "ghost-entry",
                        "size": trade_size,
                        "entry_price": price,
                        "route": route,
                    }
                )
                print(
                    "[ghost] enter %s size=%.4f price=%.4f dir=%.3f margin=%.6f"
                    % (symbol, trade_size, price, direction_prob, margin)
                )
        else:
            should_exit = False
            reason = "signal"
            if direction_prob < 0.45 or exit_conf_val < 0.5:
                should_exit = True
                reason = "confidence_drop"
            elif margin <= fees:
                should_exit = True
                reason = "negative_margin"

            if should_exit:
                profit = (price - pos["entry_price"]) * pos["size"]
                self.total_trades += 1
                if profit > 0:
                    self.wins += 1
                checkpoint = 0.0
                next_stop: Optional[str] = None
                if pos.get("route"):
                    next_stop = pos["route"][(pos.get("bus_index", 0) + 1) % len(pos["route"])]
                if profit > 0:
                    checkpoint = profit * self.stable_checkpoint_ratio
                    self.stable_bank += checkpoint
                    profit -= checkpoint
                self.total_profit += profit
                self.realized_profit += profit
                del self.positions[symbol]
                decision.update(
                    {
                        "action": "exit",
                        "status": "ghost-exit",
                        "exit_reason": reason,
                        "size": pos["size"],
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "profit": profit,
                        "checkpoint": checkpoint,
                        "stable_token": stable_target,
                        "bank_balance": self.stable_bank,
                        "total_profit": self.total_profit,
                        "win_rate": (self.wins / self.total_trades) if self.total_trades else 0.0,
                        "next_bus": next_stop,
                    }
                )
                print(
                    "[ghost] exit %s size=%.4f profit=%.6f checkpoint=%.6f bank=%.6f reason=%s"
                    % (symbol, decision["size"], profit, checkpoint, self.stable_bank, reason)
                )
            else:
                decision.update(
                    {
                        "action": "hold",
                        "status": "ghost-hold",
                        "size": pos["size"],
                        "entry_price": pos["entry_price"],
                        "unrealized": (price - pos["entry_price"]) * pos["size"],
                    }
                )
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

    def configure_route(self, symbol: str, tokens: List[str]) -> None:
        self.bus_routes[symbol] = tokens

    def record_fill(
        self,
        *,
        symbol: str,
        chain: str,
        expected_amount: float,
        executed_amount: float,
        expected_price: float,
        executed_price: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store live execution feedback for adaptive scheduling."""
        slip_details = {
            "expected_amount": expected_amount,
            "executed_amount": executed_amount,
            "expected_price": expected_price,
            "executed_price": executed_price,
        }
        if extra:
            slip_details.update(extra)
        self.db.record_trade_fill(
            chain=chain,
            symbol=symbol,
            expected_amount=expected_amount,
            executed_amount=executed_amount,
            expected_price=expected_price,
            executed_price=executed_price,
            details=slip_details,
        )
        self._save_state()

    def _load_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        ghost = state.get("ghost_trading") if isinstance(state, dict) else None
        if not isinstance(ghost, dict):
            return
        self.stable_bank = float(ghost.get("stable_bank", self.stable_bank))
        self.total_profit = float(ghost.get("total_profit", self.total_profit))
        self.realized_profit = float(ghost.get("realized_profit", self.realized_profit))
        self.total_trades = int(ghost.get("total_trades", self.total_trades))
        self.wins = int(ghost.get("wins", self.wins))
        self.positions = {
            sym: {
                "entry_price": float(pos.get("entry_price", 0.0)),
                "size": float(pos.get("size", 0.0)),
                "ts": float(pos.get("ts", time.time())),
                "route": list(pos.get("route", [])),
                "bus_index": int(pos.get("bus_index", 0)),
            }
            for sym, pos in ghost.get("positions", {}).items()
        }
        routes = ghost.get("routes")
        if isinstance(routes, dict):
            self.bus_routes = {sym: list(tokens) for sym, tokens in routes.items()}

    def _save_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        if not isinstance(state, dict):
            state = {}
        state["ghost_trading"] = {
            "stable_bank": self.stable_bank,
            "total_profit": self.total_profit,
            "realized_profit": self.realized_profit,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "positions": self.positions,
            "routes": self.bus_routes,
        }
        try:
            self.db.save_state(state)
        except Exception:
            pass
