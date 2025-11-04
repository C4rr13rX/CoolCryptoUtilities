from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from cache import CacheBalances, CacheTransfers
from db import TradingDatabase, get_db
from trading.data_stream import MarketDataStream
from trading.pipeline import TrainingPipeline
from trading.portfolio import PortfolioState
from trading.scheduler import BusScheduler, TradeDirective
from trading.metrics import FeedbackSeverity, MetricStage, MetricsCollector
from trading.swap_validator import SwapValidator

try:
    from router_wallet import UltraSwapBridge  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    UltraSwapBridge = None  # type: ignore


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
        self._model_input_order: Optional[List[str]] = None
        self._predict_fn: Optional[Callable[..., Any]] = None
        self._active_model_ref: Optional[tf.keras.Model] = None
        self._asset_vocab_limit: Optional[int] = None
        self.portfolio = PortfolioState(db=self.db)
        try:
            self.portfolio.refresh(force=True)
        except Exception as exc:
            print(f"[portfolio] initial refresh failed: {exc}")
        self._portfolio_next_refresh: float = time.time() + self.portfolio.refresh_interval
        self.scheduler = BusScheduler(db=self.db)
        self.metrics = MetricsCollector(self.db)
        self._ghost_trade_counter = 0
        self.swap_validator = SwapValidator(db=self.db)
        self._wallet_sync_lock = asyncio.Lock()
        self._bridge_init_attempted = False
        self._cache_balances = CacheBalances(db=self.db)
        self._cache_transfers = CacheTransfers(db=self.db)
        self._bridge = self._init_bridge()
        self._wallet_sync_last_reason: Optional[str] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.stream.register(self._handle_sample)
        await asyncio.gather(
            self.stream.start(),
            self._start_background_refinement(),
        )

    def _init_bridge(self) -> Optional["UltraSwapBridge"]:
        if getattr(self, "_bridge_init_attempted", False) and getattr(self, "_bridge", None) is not None:
            return self._bridge  # type: ignore[attr-defined]
        self._bridge_init_attempted = True
        if UltraSwapBridge is None:
            print("[trading-bot] UltraSwapBridge unavailable (web3 dependencies missing). Wallet sync disabled.")
            return None
        try:
            bridge = UltraSwapBridge()
            return bridge
        except Exception as exc:
            print(f"[trading-bot] unable to initialise UltraSwapBridge: {exc}")
            return None

    async def _run_wallet_sync(self, *, reason: str) -> None:
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return
        chains = list(self.portfolio.chains)
        start = time.time()
        errors: List[Tuple[str, str]] = []

        async with self._wallet_sync_lock:
            def _sync_work() -> List[Tuple[str, str]]:
                local_errors: List[Tuple[str, str]] = []
                try:
                    self._cache_transfers.rebuild_incremental(self._bridge, chains)
                except Exception as exc:
                    local_errors.append(("transfers", str(exc)))
                try:
                    self._cache_balances.rebuild_all(self._bridge, chains)
                except Exception as exc:
                    local_errors.append(("balances", str(exc)))
                return local_errors

            errors = await asyncio.to_thread(_sync_work)

        try:
            self.portfolio.refresh(force=True)
        except Exception as exc:
            errors.append(("portfolio", str(exc)))

        duration = time.time() - start
        metrics_payload = {
            "duration_sec": duration,
            "chain_count": len(chains),
            "errors": float(len(errors)),
        }
        self.metrics.record(
            MetricStage.LIVE_TRADING,
            metrics_payload,
            category="wallet_sync",
            meta={"reason": reason, "chains": chains, "errors": errors},
        )
        if errors:
            for domain, message in errors:
                severity = FeedbackSeverity.WARNING if domain != "balances" else FeedbackSeverity.CRITICAL
                self.metrics.feedback(
                    "wallet_sync",
                    severity=severity,
                    label=f"{domain}_failure",
                    details={"reason": reason, "message": message, "chains": chains},
                )
        self._wallet_sync_last_reason = reason

    def _ensure_model_bindings(self, model: tf.keras.Model) -> None:
        if self._active_model_ref is model and self._model_input_order is not None:
            return

        self._model_input_order = [tensor.name.split(":")[0] for tensor in model.inputs]
        input_signature: List[tf.TensorSpec] = []
        for tensor in model.inputs:
            keras_tensor = tensor[0] if isinstance(tensor, (list, tuple)) else tensor
            shape_tuple = tf.keras.backend.int_shape(keras_tensor)
            if shape_tuple is None:
                shape_tuple = (None,)
            shape_signature = tuple(None if idx == 0 else dim for idx, dim in enumerate(shape_tuple))
            input_signature.append(tf.TensorSpec(shape=shape_signature, dtype=keras_tensor.dtype))

        @tf.function(reduce_retracing=True, input_signature=input_signature)
        def _predict(*ordered):
            return model(list(ordered), training=False)

        self._predict_fn = _predict
        self._active_model_ref = model
        self._asset_vocab_limit = None

        for layer_name in ("asset_embedding", "asset_embedding_1"):
            try:
                layer = model.get_layer(layer_name)
                if hasattr(layer, "input_dim"):
                    self._asset_vocab_limit = int(layer.input_dim)
                    break
            except Exception:
                continue
        if self._asset_vocab_limit is None:
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Embedding) and hasattr(layer, "input_dim"):
                    self._asset_vocab_limit = int(layer.input_dim)
                    break

    def _invoke_model(self, inputs: Dict[str, np.ndarray]):
        if not self._model_input_order:
            raise RuntimeError("Model input order not initialised")

        ordered_inputs = [inputs[name] for name in self._model_input_order]
        if self._predict_fn is None or self._active_model_ref is None:
            preds = self.pipeline.ensure_active_model().predict(ordered_inputs, verbose=0)
        else:
            ordered_tensors = [tf.convert_to_tensor(arr) for arr in ordered_inputs]
            preds = self._predict_fn(*ordered_tensors)
            if isinstance(preds, (list, tuple)):
                preds = [p.numpy() for p in preds]
            elif isinstance(preds, dict):
                preds = {k: np.array(v) for k, v in preds.items()}
            else:
                preds = preds.numpy()
        if isinstance(preds, dict):
            preds = [preds[name] for name in [
                "exit_conf",
                "price_mu",
                "price_log_var",
                "price_dir",
                "net_margin",
                "net_pnl",
                "tech_recon",
                "price_gaussian",
            ] if name in preds]
        return preds

    def _summarise_predictions(self, preds) -> Dict[str, float]:
        summary = {
            "exit_conf": 0.5,
            "direction_prob": 0.5,
            "net_margin": 0.0,
            "delta": 0.0,
        }
        try:
            summary["exit_conf"] = float(preds[0][0][0])
        except Exception:
            pass
        try:
            summary["price_mu"] = float(preds[1][0][0])
            summary["delta"] = summary["price_mu"]
        except Exception:
            summary["price_mu"] = 0.0
        try:
            summary["price_log_var"] = float(preds[2][0][0])
        except Exception:
            summary["price_log_var"] = 0.0
        try:
            summary["direction_prob"] = float(preds[3][0][0])
        except Exception:
            pass
        try:
            summary["net_margin"] = float(preds[4][0][0])
        except Exception:
            pass
        try:
            summary["net_pnl"] = float(preds[5][0][0])
        except Exception:
            summary["net_pnl"] = summary.get("net_margin", 0.0)
        return summary

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
        now = float(sample.get("ts") or time.time())
        if now >= self._portfolio_next_refresh:
            try:
                self.portfolio.refresh()
                summary = self.portfolio.summary()
                print(
                    "[portfolio] wallet=%s stable≈%.2f native≈%.4f holdings=%d"
                    % (
                        summary.get("wallet"),
                        summary.get("stable_usd", 0.0),
                        summary.get("native_eth", 0.0),
                        int(summary.get("holdings", 0)),
                    )
                )
            except Exception as exc:
                print(f"[portfolio] refresh failed: {exc}")
            self._portfolio_next_refresh = now + self.portfolio.refresh_interval
        self._ensure_model_bindings(model)
        inputs = self._prepare_inputs(list(self._buffer))
        try:
            preds = self._invoke_model(inputs)
        except Exception as exc:
            print(f"[trading-bot] prediction failed: {exc}")
            return
        pred_summary = self._summarise_predictions(preds)
        await self._run_wallet_sync(reason="pre-schedule")
        directive = None
        try:
            directive = self.scheduler.evaluate(sample, pred_summary, self.portfolio)
        except Exception as exc:
            print(f"[bus-scheduler] evaluation failed: {exc}")
        decision = await self._interpret_predictions(preds, sample, directive, pred_summary)
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
        if self._asset_vocab_limit is not None and self._asset_vocab_limit > 0:
            max_index = self._asset_vocab_limit - 1
            asset = np.clip(asset, 0, max_index).astype(np.int32)

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

    async def _interpret_predictions(
        self,
        preds,
        sample: Dict[str, Any],
        directive: Optional[TradeDirective],
        pred_summary: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        summary = pred_summary or self._summarise_predictions(preds)
        exit_conf_val = float(summary.get("exit_conf", 0.5))
        direction_prob = float(summary.get("direction_prob", 0.5))
        delta = float(summary.get("delta", 0.0))
        margin = float(summary.get("net_margin", 0.0))
        pnl = float(summary.get("net_pnl", margin))
        sample_ts = float(sample.get("ts", time.time()))

        symbol = sample.get("symbol", "asset")
        price = float(sample.get("price", 0.0))
        volume = float(sample.get("volume", 0.0))
        route = self.bus_routes.get(symbol) or symbol.split("-")
        route = [t.upper() for t in route if t]
        if directive:
            route = [directive.base_token.upper(), directive.quote_token.upper()]
        if not route:
            route = [symbol.upper()]
        self.bus_routes[symbol] = route

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
            "route": route,
            "bus_plan": directive.to_dict() if directive else None,
        }

        trade_size = max(min(volume * self.max_trade_share, volume), 0.0)
        trade_size = min(trade_size, 100.0)
        if directive and directive.size > 0:
            trade_size = float(directive.size)
        if trade_size <= 0.0:
            if pos is not None:
                decision.update(
                    {
                        "unrealized": (price - pos["entry_price"]) * pos["size"],
                        "size": pos["size"],
                        "entry_price": pos["entry_price"],
                    }
                )
            return decision

        should_enter = False
        should_exit = False
        reason = ""

        if directive and directive.action == "enter":
            should_enter = True
            reason = directive.reason
        elif directive and directive.action == "exit":
            should_exit = True
            reason = directive.reason
        elif pos is None:
            if direction_prob > 0.58 and exit_conf_val >= 0.6 and margin > fees * 1.1:
                should_enter = True
                reason = "model-long"
        else:
            if direction_prob < 0.45 or exit_conf_val < 0.5:
                should_exit = True
                reason = "confidence_drop"
            elif margin <= fees:
                should_exit = True
                reason = "negative_margin"
            elif pnl < 0 and (time.time() - pos.get("ts", 0)) > 60 * 15:
                should_exit = True
                reason = "timed-exit"

        if should_enter:
            await self._run_wallet_sync(reason="pre-enter")
            allowed, guard_metrics, guard_reasons = self.swap_validator.validate(
                symbol=symbol,
                route=route,
                trade_size=trade_size,
                price=price,
                volume=volume,
                prediction=summary,
            )
            if not allowed:
                decision.update(
                    {
                        "action": "hold",
                        "status": "guard-blocked",
                        "reason": f"swap_guard:{'/'.join(guard_reasons) if guard_reasons else 'guard'}",
                        "swap_guard": guard_metrics,
                    }
                )
                return decision
            self._ghost_trade_counter += 1
            trade_id = f"{symbol}-{self._ghost_trade_counter}"
            self.positions[symbol] = {
                "entry_price": price,
                "size": trade_size,
                "ts": sample_ts,
                "entry_ts": sample_ts,
                "trade_id": trade_id,
                "route": route,
                "bus_index": 0,
                "target_price": directive.target_price if directive else None,
            }
            decision.update(
                {
                    "action": "enter",
                    "status": "ghost-entry",
                    "size": trade_size,
                    "entry_price": price,
                    "route": route,
                    "reason": reason or (directive.reason if directive else "model"),
                    "target_price": directive.target_price if directive else price * 1.05,
                    "horizon": directive.horizon if directive else None,
                    "trade_id": trade_id,
                    "entry_ts": sample_ts,
                }
            )
            entry_metrics = {
                "direction_prob": direction_prob,
                "exit_confidence": exit_conf_val,
                "expected_margin": margin,
                "trade_size": trade_size,
                "volume": volume,
                "route_length": len(route),
            }
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                entry_metrics,
                category="entry",
                meta={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "reason": decision["reason"],
                    "price": price,
                },
            )
            self.metrics.feedback(
                "ghost_trading",
                severity=FeedbackSeverity.INFO,
                label="entry",
                details={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "expected_margin": margin,
                    "direction_prob": direction_prob,
                    "route": route,
                },
            )
            print(
                "[ghost] enter %s size=%.4f price=%.4f dir=%.3f margin=%.6f (%s)"
                % (symbol, trade_size, price, direction_prob, margin, decision["reason"])
            )
            return decision

        if should_exit and pos is not None:
            await self._run_wallet_sync(reason="pre-exit")
            exit_size = min(pos["size"], trade_size)
            profit = (price - pos["entry_price"]) * exit_size
            self.total_trades += 1
            if profit > 0:
                self.wins += 1
            checkpoint = 0.0
            next_stop = route[1] if len(route) > 1 else None
            if profit > 0:
                checkpoint = profit * self.stable_checkpoint_ratio
                self.stable_bank += checkpoint
                profit -= checkpoint
            self.total_profit += profit
            self.realized_profit += profit
            trade_id = pos.get("trade_id") or f"{symbol}-{int(pos.get('ts', sample_ts))}"
            entry_ts = float(pos.get("entry_ts", pos.get("ts", sample_ts)))
            duration_sec = max(0.0, sample_ts - entry_ts)
            del self.positions[symbol]
            decision.update(
                {
                    "action": "exit",
                    "status": "ghost-exit",
                    "exit_reason": reason,
                    "size": exit_size,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "profit": profit,
                    "checkpoint": checkpoint,
                    "stable_token": stable_target,
                    "bank_balance": self.stable_bank,
                    "total_profit": self.total_profit,
                    "win_rate": (self.wins / self.total_trades) if self.total_trades else 0.0,
                    "next_bus": next_stop,
                    "horizon": directive.horizon if directive else None,
                    "trade_id": trade_id,
                    "entry_ts": entry_ts,
                    "exit_ts": sample_ts,
                    "duration_sec": duration_sec,
                }
            )
            exit_metrics = {
                "profit": profit,
                "checkpoint": checkpoint,
                "duration_sec": duration_sec,
                "bank_balance": self.stable_bank,
                "total_profit": self.total_profit,
                "win_rate": self.wins / max(1, self.total_trades),
                "exit_price": price,
                "entry_price": pos["entry_price"],
            }
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                exit_metrics,
                category="exit",
                meta={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "reason": reason,
                    "route": route,
                },
            )
            severity = FeedbackSeverity.INFO if profit > 0 else FeedbackSeverity.WARNING
            if profit <= 0 or reason in {"negative_margin", "confidence_drop", "timed-exit"}:
                severity = FeedbackSeverity.CRITICAL if profit < 0 else FeedbackSeverity.WARNING
            self.metrics.feedback(
                "ghost_trading",
                severity=severity,
                label=f"exit_{reason}",
                details={
                    "symbol": symbol,
                    "trade_id": trade_id,
                    "profit": profit,
                    "duration_sec": duration_sec,
                    "reason": reason,
                    "expected_margin": margin,
                    "direction_prob": direction_prob,
                },
            )
            print(
                "[ghost] exit %s size=%.4f price=%.4f profit=%.6f checkpoint=%.6f bank=%.6f reason=%s"
                % (symbol, exit_size, price, profit, self.stable_bank, reason or "exit")
            )
            return decision

        if pos is not None:
            decision.update(
                {
                    "unrealized": (price - pos["entry_price"]) * pos["size"],
                    "size": pos["size"],
                    "entry_price": pos["entry_price"],
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
