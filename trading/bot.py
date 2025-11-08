from __future__ import annotations

import asyncio
import math
import time
import os
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from cache import CacheBalances, CacheTransfers
from db import TradingDatabase, get_db
from trading.data_stream import MarketDataStream
from trading.pipeline import TrainingPipeline
from trading.portfolio import PortfolioState, NATIVE_SYMBOL
from trading.scheduler import BusScheduler, TradeDirective
from trading.equilibrium import EquilibriumTracker
from trading.metrics import FeedbackSeverity, MetricStage, MetricsCollector
from trading.swap_validator import SwapValidator
from trading.opportunity import OpportunityTracker
from trading.savings import StableSavingsPlanner, SavingsEvent
from trading.constants import (
    PRIMARY_CHAIN,
    PRIMARY_SYMBOL,
    MIN_CONFIDENCE,
    SMALL_PROFIT_FLOOR,
    MAX_QUOTE_SHARE,
    GAS_PROFIT_BUFFER,
    FALLBACK_NATIVE_PRICE,
)
from trading.brain import (
    NeuroGraph,
    MultiResolutionSwarm,
    PatternMemory,
    ScenarioReactor,
    VolatilityArbCell,
)
from trading.brain.event_engine import make_default_engine
from services.organism_state import build_snapshot

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
        self.stream = stream or MarketDataStream(symbol=PRIMARY_SYMBOL, chain=PRIMARY_CHAIN)
        self.window_size = window_size
        self.queue: List[Dict[str, Any]] = []
        self._bg_task: Optional[asyncio.Task] = None
        self._running = False
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.stable_bank: float = 0.0
        self.total_profit: float = 0.0
        self.realized_profit: float = 0.0
        self.total_trades: int = 0
        self.wins: int = 0
        self.max_trade_share: float = min(0.5, max(0.05, MAX_QUOTE_SHARE))
        self.stable_checkpoint_ratio: float = 0.15
        self.bus_routes: Dict[str, List[str]] = {}
        self.stable_tokens = {"USDC", "USDT", "DAI", "BUSD", "TUSD", "USDP", "USDD"}
        self.sim_quote_balances: Dict[Tuple[str, str], float] = {}
        self.sim_native_balances: Dict[str, float] = {}
        self._sim_initial_pool: float = 0.0
        self.ghost_session_id: int = 1
        self.active_exposure: Dict[str, float] = {}
        self.graph = NeuroGraph()
        self.swarm = MultiResolutionSwarm(
            [("fast", 20), ("medium", 60), ("slow", 180)]
        )
        self._brain_window = max(
            self.window_size,
            max((h for _, h in self.swarm.horizon_defs), default=self.window_size),
        )
        self._buffer = deque(maxlen=self._brain_window)
        self._price_history: Dict[str, deque] = {}
        self._sentiment_history: Dict[str, deque] = {}
        self._prev_prices: Dict[str, float] = {}
        self._graph_decay_counter: int = 0
        self.memory = PatternMemory(dim=6)
        self.scenario_reactor = ScenarioReactor()
        self.arb_cell = VolatilityArbCell()
        self.event_engine = make_default_engine(self._on_reflex_block)
        self._reflex_blocked_until: float = 0.0
        self._reflex_block_reason: Optional[str] = None
        self._volatility_avg: float = 0.0
        self._peak_equity: float = 0.0
        self._last_windows: Dict[str, Dict[str, np.ndarray]] = {}
        self._model_input_order: Optional[List[str]] = None
        self._predict_fn: Optional[Callable[..., Any]] = None
        self._active_model_ref: Optional[tf.keras.Model] = None
        self._asset_vocab_limit: Optional[int] = None
        self.portfolio = PortfolioState(db=self.db)
        self._snapshot_interval = max(1.0, float(os.getenv("ORGANISM_SNAPSHOT_INTERVAL", "5.0")))
        self._last_snapshot_ts: float = 0.0
        self._discovery_cache: Dict[str, Any] = {}
        self._discovery_cache_ts: float = 0.0
        try:
            self.portfolio.refresh(force=True)
        except Exception as exc:
            print(f"[portfolio] initial refresh failed: {exc}")
        self._portfolio_next_refresh: float = time.time() + self.portfolio.refresh_interval
        self.scheduler = BusScheduler(db=self.db)
        self.metrics = MetricsCollector(self.db)
        self._ghost_trade_counter = 0
        self.swap_validator = SwapValidator(db=self.db)
        self.opportunity_tracker = OpportunityTracker()
        savings_batch = float(os.getenv("SAVINGS_TRANSFER_MIN_USD", "50"))
        self.savings = StableSavingsPlanner(min_batch=savings_batch)
        self._wallet_sync_lock = asyncio.Lock()
        self._bridge_init_attempted = False
        self.equilibrium = EquilibriumTracker()
        self._nash_equilibrium_reached: bool = False
        self._cache_balances = CacheBalances(db=self.db)
        self._cache_transfers = CacheTransfers(db=self.db)
        self._bridge = self._init_bridge()
        self._wallet_sync_last_reason: Optional[str] = None
        self._latency_window: deque = deque(maxlen=500)
        self._pending_queue: deque = deque(maxlen=int(os.getenv("STREAM_QUEUE_MAX", "8")))
        self._processing_sample: bool = False
        self._last_sample_signature: Optional[Tuple[str, float]] = None
        self._last_gas_advisory_signature: Optional[str] = None
        self.primary_chain: str = PRIMARY_CHAIN
        self.primary_symbol: str = PRIMARY_SYMBOL
        self.gas_buffer_multiplier: float = max(1.0, float(os.getenv("GAS_BUFFER_MULTIPLIER", str(GAS_PROFIT_BUFFER))))
        self.gas_profit_guard: float = max(1.0, float(os.getenv("GAS_PROFIT_SAFETY", str(GAS_PROFIT_BUFFER))))
        self._latency_samples: int = 0
        self._enable_bg_refinement = os.getenv("ENABLE_BG_REFINEMENT", "0").lower() in {"1", "true", "yes", "on"}
        self.live_trading_enabled: bool = os.getenv("ENABLE_LIVE_TRADING", "0").lower() in {"1", "true", "yes", "on"}
        self.auto_promote_live: bool = os.getenv("AUTO_PROMOTE_LIVE", "0").lower() in {"1", "true", "yes", "on"}
        self.required_live_win_rate: float = float(os.getenv("LIVE_PROMOTION_WIN_RATE", "0.9"))
        self.required_live_trades: int = int(os.getenv("LIVE_PROMOTION_MIN_TRADES", "120"))
        self.required_live_profit: float = float(os.getenv("LIVE_PROMOTION_MIN_PROFIT", "50.0"))
        self.max_symbol_share: float = float(os.getenv("MAX_SYMBOL_SHARE", "0.25"))
        self.global_risk_budget: float = float(os.getenv("GLOBAL_RISK_BUDGET", "1.0"))
        self._load_state()
        if not self.sim_quote_balances:
            self._init_sim_balances()
        else:
            self._sim_initial_pool = sum(self.sim_quote_balances.values()) or self._sim_initial_pool
            if not self.sim_native_balances:
                self.sim_native_balances[self.primary_chain.lower()] = 0.5

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.stream.register(self._handle_sample)
        tasks = [self.stream.start()]
        if self._enable_bg_refinement:
            tasks.append(self._start_background_refinement())
        await asyncio.gather(*tasks)

    def _init_sim_balances(self) -> None:
        """Initialise simulated balances based on the real portfolio snapshot."""
        self.sim_quote_balances.clear()
        for (chain, symbol), holding in self.portfolio.holdings.items():
            if symbol.upper() in self.stable_tokens:
                self.sim_quote_balances[(chain.lower(), symbol.upper())] = holding.quantity
        # ensure primary quote exists even if wallet empty
        key = (self.primary_chain.lower(), "USDC")
        self.sim_quote_balances.setdefault(key, 0.0)
        self.sim_native_balances = {chain.lower(): max(balance, 0.1) for chain, balance in self.portfolio.native_balances.items()}
        self.sim_native_balances.setdefault(self.primary_chain.lower(), 0.5)
        self._sim_initial_pool = sum(self.sim_quote_balances.values())
        if self._sim_initial_pool > self._peak_equity:
            self._peak_equity = self._sim_initial_pool
        self.ghost_session_id = max(1, self.ghost_session_id)
        self.active_exposure.clear()

    def _token_key(self, chain: str, symbol: str) -> Tuple[str, str]:
        return (chain.lower(), symbol.upper())

    def _get_quote_balance(self, chain: str, symbol: str) -> float:
        key = self._token_key(chain, symbol)
        if not self.live_trading_enabled:
            return self.sim_quote_balances.get(key, 0.0)
        return self.portfolio.get_quantity(symbol, chain=chain)

    def _adjust_quote_balance(self, chain: str, symbol: str, delta: float) -> None:
        if self.live_trading_enabled:
            return
        key = self._token_key(chain, symbol)
        current = self.sim_quote_balances.get(key, 0.0)
        updated = max(0.0, current + delta)
        self.sim_quote_balances[key] = updated

    def _consume_sim_gas(self, chain: str, gas_native: float) -> None:
        if self.live_trading_enabled:
            return
        chain_l = chain.lower()
        current = self.sim_native_balances.get(chain_l, 0.5)
        current = max(0.0, current - gas_native)
        self.sim_native_balances[chain_l] = current

    def _check_sim_restart(self) -> None:
        if self.live_trading_enabled:
            return
        total = sum(self.sim_quote_balances.values())
        if self._sim_initial_pool <= 0:
            self._sim_initial_pool = total
            return
        threshold = float(os.getenv("SIM_BALANCE_RESET_RATIO", "0.1"))
        if total <= self._sim_initial_pool * threshold:
            print("[ghost] simulated bankroll depleted; resetting simulation state.")
            self._init_sim_balances()
            self.positions.clear()
            self.ghost_session_id += 1

    def _extract_sentiment_score(self, sample: Dict[str, Any]) -> float:
        raw_score = sample.get("sentiment_score")
        if raw_score is not None:
            try:
                return float(raw_score)
            except Exception:
                pass
        sentiment = sample.get("sentiment")
        if sentiment is None:
            raw = sample.get("raw")
            if isinstance(raw, dict):
                sentiment = raw.get("sentiment") or raw.get("label")
        if sentiment is None:
            return 0.0
        mapping = {
            "positive": 0.7,
            "bullish": 1.0,
            "accumulate": 0.5,
            "neutral": 0.0,
            "mixed": 0.1,
            "negative": -0.7,
            "bearish": -1.0,
            "sell": -0.6,
        }
        return float(mapping.get(str(sentiment).lower(), 0.0))

    def _on_reflex_block(self, context: Dict[str, float]) -> None:
        cooldown = float(context.get("cooldown", 60.0))
        reason = str(context.get("reflex_rule") or context.get("reason") or "reflex")
        self._reflex_blocked_until = time.time() + cooldown
        self._reflex_block_reason = reason
        details = {
            "reason": reason,
            "cooldown": cooldown,
            "drawdown": context.get("drawdown"),
            "volatility": context.get("volatility"),
        }
        if hasattr(self, "metrics"):
            try:
                self.metrics.feedback(
                    "reflex",
                    severity=FeedbackSeverity.CRITICAL,
                    label="block",
                    details=details,
                )
            except Exception:
                pass

    def _update_brain_state(
        self,
        sample: Dict[str, Any],
        history_window: List[Dict[str, Any]],
        pred_summary: Dict[str, float],
    ) -> Dict[str, Any]:
        symbol = str(sample.get("symbol") or self.primary_symbol)
        price = float(sample.get("price") or 0.0)
        volume = float(sample.get("volume") or 0.0)
        if volume <= 0.0 and history_window:
            try:
                volume = float(history_window[-1].get("volume", volume))
            except Exception:
                pass
        ts = float(sample.get("ts", time.time()))
        if price <= 0.0 and history_window:
            try:
                price = float(history_window[-1].get("price", price))
            except Exception:
                pass
        price_history = self._price_history.setdefault(symbol, deque(maxlen=self._brain_window))
        sentiment_history = self._sentiment_history.setdefault(symbol, deque(maxlen=self._brain_window))
        price_history.append(price)
        sentiment_history.append(self._extract_sentiment_score(sample))
        history_prices = np.asarray(list(price_history), dtype=np.float64)
        history_sentiment = np.asarray(list(sentiment_history), dtype=np.float64)
        price_windows: Dict[str, np.ndarray] = {}
        sentiment_windows: Dict[str, np.ndarray] = {}
        realized_returns: Dict[str, float] = {}
        for label, horizon in self.swarm.horizon_defs:
            if history_prices.size >= horizon:
                price_windows[label] = history_prices[-horizon:].copy()
                sentiment_windows[label] = history_sentiment[-horizon:].copy()
                base_index = history_prices.size - horizon - 1
                if base_index >= 0:
                    base_price = history_prices[base_index]
                else:
                    base_price = history_prices[0]
                if base_price != 0.0:
                    realized_returns[label] = (price - base_price) / max(abs(base_price), 1e-9)
        if realized_returns:
            try:
                self.swarm.learn(price_windows, sentiment_windows, realized_returns)
            except Exception:
                pass
        opportunity_signal = None
        try:
            opportunity_signal = self.opportunity_tracker.evaluate(symbol, history_prices)
        except Exception:
            opportunity_signal = None
        if opportunity_signal:
            try:
                self.metrics.feedback(
                    "opportunity",
                    severity=FeedbackSeverity.INFO if opportunity_signal.kind == "buy-low" else FeedbackSeverity.WARNING,
                    label=opportunity_signal.kind,
                    details=opportunity_signal.to_dict(),
                )
                self.scheduler.record_opportunity(opportunity_signal)
            except Exception:
                pass
        swarm_votes = []
        try:
            swarm_votes = self.swarm.vote(price_windows, sentiment_windows)
        except Exception:
            swarm_votes = []
        weight_sum = sum(v.confidence for v in swarm_votes) or 1.0
        swarm_bias = (
            sum(v.expected_return * v.confidence for v in swarm_votes) / weight_sum
            if swarm_votes
            else 0.0
        )
        volatility = float(sample.get("rolling_volatility") or 0.0)
        if volatility == 0.0 and history_prices.size > 3:
            volatility = float(np.std(np.diff(history_prices[-min(20, history_prices.size):])))
        alpha = 0.05
        self._volatility_avg = (1 - alpha) * self._volatility_avg + alpha * volatility
        self.graph.upsert_node(symbol, "asset", price, ts, volume=volume, volatility=volatility)
        prev_price = self._prev_prices.get(symbol)
        if prev_price is not None:
            rel_change = (price - prev_price) / max(abs(prev_price), 1e-9)
            self.graph.upsert_node(f"{symbol}:momentum", "volatility", rel_change, ts)
            self.graph.reinforce(symbol, f"{symbol}:momentum", ts, rel_change)
        self._prev_prices[symbol] = price
        self._graph_decay_counter += 1
        if self._graph_decay_counter >= 20:
            try:
                self.graph.decay_all()
            finally:
                self._graph_decay_counter = 0
        graph_conf = self.graph.confidence_adjustment(symbol)
        fingerprint = np.array(
            [
                price,
                volume,
                float(pred_summary.get("direction_prob", 0.5)),
                float(pred_summary.get("net_margin", 0.0)),
                float(pred_summary.get("delta", 0.0)),
                volatility,
            ],
            dtype=np.float32,
        )
        memory_bias = 0.0
        memory_meta: Dict[str, float] = {}
        try:
            match = self.memory.match(fingerprint)
        except Exception:
            match = None
        if match:
            memory_bias, memory_meta = match
        scenarios = self.scenario_reactor.analyse(
            float(pred_summary.get("net_margin", 0.0)),
            float(pred_summary.get("direction_prob", 0.5)),
            volatility,
        )
        scenario_spread = self.scenario_reactor.divergence(scenarios)
        scenario_defer = self.scenario_reactor.should_defer(scenarios)
        scenario_mod = 1.0
        if scenario_defer:
            scenario_mod = 0.0
        else:
            scenario_mod = max(0.4, 1.0 - min(0.4, scenario_spread * 10.0))
        arb_signal_payload: Optional[Dict[str, float]] = None
        symbol_upper = symbol.upper()
        if price > 0 and any(tok in symbol_upper for tok in {"ETH", "WETH"}) and any(
            stable in symbol_upper for stable in {"USDC", "USDT", "DAI"}
        ):
            try:
                arb_signal = self.arb_cell.observe(price, 1.0)
                arb_signal_payload = {
                    "action": arb_signal.action,
                    "spread": float(arb_signal.spread),
                    "implied_edge": float(arb_signal.implied_edge),
                    "confidence": float(arb_signal.confidence),
                }
            except Exception:
                arb_signal_payload = None
        self._last_windows[symbol] = {
            "prices": {label: window.copy() for label, window in price_windows.items()},
            "sentiment": {label: window.copy() for label, window in sentiment_windows.items()},
        }
        equity = self._current_equity()
        if equity > self._peak_equity:
            self._peak_equity = equity
        drawdown = 0.0
        if self._peak_equity > 0:
            drawdown = (equity - self._peak_equity) / max(self._peak_equity, 1e-9)
        context = {
            "drawdown": drawdown,
            "equity": equity,
            "volatility": volatility,
            "volatility_avg": self._volatility_avg,
            "pnl": self.total_profit,
            "cooldown": max(30.0, 60.0 * (1.0 + min(1.0, abs(drawdown)))),
        }
        if self._reflex_blocked_until and time.time() >= self._reflex_blocked_until:
            self._reflex_block_reason = None
            self._reflex_blocked_until = 0.0
        reflex_triggered: List[str] = []
        try:
            reflex_triggered = self.event_engine.process(context, ts)
        except Exception:
            reflex_triggered = []
        if reflex_triggered:
            for rule in reflex_triggered:
                try:
                    self.metrics.feedback(
                        "reflex",
                        severity=FeedbackSeverity.WARNING,
                        label=f"trigger_{rule}",
                        details={"drawdown": drawdown, "volatility": volatility},
                    )
                except Exception:
                    pass
        reflex_active = time.time() < self._reflex_blocked_until
        threshold_scale = 1.0
        if swarm_bias > 0:
            threshold_scale *= max(0.7, 1.0 - min(0.3, abs(swarm_bias) * 2.0))
        elif swarm_bias < 0:
            threshold_scale *= min(1.3, 1.0 + min(0.3, abs(swarm_bias) * 2.0))
        if memory_bias > 0:
            threshold_scale *= max(0.75, 1.0 - min(0.2, memory_bias / 5.0))
        elif memory_bias < 0:
            threshold_scale *= min(1.25, 1.0 + min(0.2, abs(memory_bias) / 5.0))
        brain_summary = {
            "graph_confidence": graph_conf,
            "swarm_bias": swarm_bias,
            "swarm_votes": [
                {"horizon": vote.horizon, "expected": vote.expected_return, "confidence": vote.confidence}
                for vote in swarm_votes
            ],
            "memory_bias": memory_bias,
            "memory_meta": memory_meta,
            "scenario_spread": scenario_spread,
            "scenario_defer": scenario_defer,
            "scenario_mod": scenario_mod,
            "scenarios": [
                {"label": s.label, "expected": s.expected_return, "confidence": s.confidence} for s in scenarios
            ],
            "arb_signal": arb_signal_payload,
            "volatility": volatility,
            "volatility_avg": self._volatility_avg,
            "reflex_triggered": reflex_triggered,
            "reflex_block_active": reflex_active,
            "reflex_block_until": self._reflex_blocked_until if reflex_active else None,
            "reflex_reason": self._reflex_block_reason,
            "opportunity": opportunity_signal.to_dict() if opportunity_signal else None,
            "threshold_scale": threshold_scale,
            "fingerprint": fingerprint.tolist(),
        }
        try:
            self.metrics.record(
                MetricStage.GHOST_TRADING,
                {
                    "graph_confidence": graph_conf,
                    "swarm_bias": swarm_bias,
                    "scenario_spread": scenario_spread,
                    "volatility": volatility,
                },
                category="brain_state",
                meta={"symbol": symbol, "reflex_block": reflex_active},
            )
        except Exception:
            pass
        return brain_summary

    def _total_stable(self, chain: str) -> float:
        chain_l = chain.lower()
        if self.live_trading_enabled:
            return self.portfolio.stable_liquidity(chain_l)
        total = sum(qty for (ch, _), qty in self.sim_quote_balances.items() if ch == chain_l)
        total += max(0.0, self.stable_bank)
        return total

    def _handle_savings_transfer(self, event: SavingsEvent) -> None:
        payload = event.to_dict()
        payload["checkpoint_ratio"] = float(self.stable_checkpoint_ratio)
        try:
            self.db.log_trade(
                wallet=event.mode,
                chain=self.primary_chain,
                symbol=event.token,
                action="savings_transfer",
                status="queued",
                details=payload,
            )
        except Exception:
            pass
        try:
            self.metrics.feedback(
                "savings",
                severity=FeedbackSeverity.INFO,
                label=event.reason,
                details=payload,
            )
        except Exception:
            pass
        try:
            self.metrics.record(
                MetricStage.SAVINGS,
                {"amount": float(event.amount), "equilibrium_score": event.equilibrium_score},
                category=event.mode,
                meta=payload,
            )
        except Exception:
            pass
        print(
            f"[savings] mode={event.mode} token={event.token} amount={event.amount:.4f} equilibrium={event.equilibrium_score:.3f}"
        )

    def _get_pair_adjustment(self, symbol: str) -> Dict[str, Any]:
        cache = self._pair_adjustments.get(symbol)
        now = time.time()
        if cache and (now - float(cache.get("_ts", 0.0))) < 30.0:
            return cache
        try:
            record = self.db.get_pair_adjustment(symbol) or {}
        except Exception:
            record = {}
        record["_ts"] = now
        self._pair_adjustments[symbol] = record
        return record

    def _tune_allocation(self, symbol: str, *, positive: bool, negative: bool) -> None:
        delta = 0.0
        if positive:
            delta += 0.02
        if negative:
            delta -= 0.05
        if abs(delta) < 1e-6:
            return
        try:
            self.db.adjust_pair_allocation(symbol, delta)
            self._pair_adjustments.pop(symbol, None)
        except Exception:
            pass

    def _current_equity(self) -> float:
        if self.live_trading_enabled:
            stable_liquidity = 0.0
            try:
                for holding in self.portfolio.holdings.values():
                    value = getattr(holding, "usd", None)
                    if value is None:
                        continue
                    stable_liquidity += float(value)
            except Exception:
                stable_liquidity = 0.0
            native_usd = 0.0
            try:
                for chain, balance in self.portfolio.native_balances.items():
                    price_row = self.db.fetch_price(chain, NATIVE_SYMBOL.get(chain, chain.upper()))
                    price = float(price_row.get("usd")) if price_row and price_row.get("usd") else FALLBACK_NATIVE_PRICE
                    native_usd += balance * price
            except Exception:
                native_usd = 0.0
            return max(0.0, stable_liquidity + native_usd + self.stable_bank + self.total_profit)
        return max(0.0, self._sim_initial_pool + self.stable_bank + self.total_profit)

    def current_equity(self) -> float:
        """Public helper exposed to observability layers."""
        return self._current_equity()

    def latency_stats(self) -> Dict[str, float]:
        if not self._latency_window:
            return {}
        try:
            window_arr = np.asarray(self._latency_window, dtype=np.float64)
        except Exception:
            window_arr = np.array(list(self._latency_window), dtype=np.float64)
        if window_arr.size == 0:
            return {}
        avg_ms = float(np.mean(window_arr) * 1000.0)
        p95_ms = float(np.percentile(window_arr, 95) * 1000.0)
        return {
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "count": int(window_arr.size),
        }

    def _compute_base_allocation(self, sample: Dict[str, Any]) -> Dict[str, float]:
        symbol = str(sample.get("symbol") or "")
        if not symbol:
            return {}
        chain = str(sample.get("chain", self.primary_chain)).lower() or self.primary_chain
        total_stable = self._total_stable(chain)
        if total_stable <= 0:
            return {}
        max_share = max(0.01, min(self.max_symbol_share, 1.0))
        allocation = total_stable * max_share
        current_exposure = self.active_exposure.get(symbol, 0.0)
        available = max(0.0, allocation - current_exposure)
        if available <= 0:
            return {}
        return {symbol: available}

    def _propagate_horizon_bias(self) -> None:
        if not hasattr(self.scheduler, "set_bucket_bias"):
            return
        bias = {}
        try:
            bias = self.pipeline.horizon_bias()
        except Exception:
            return
        if bias:
            self.scheduler.set_bucket_bias(bias)

    def _maybe_promote_to_live(self) -> None:
        if self.live_trading_enabled or not self.auto_promote_live:
            return
        if self.total_trades < max(1, self.required_live_trades):
            return
        win_rate = (self.wins / self.total_trades) if self.total_trades else 0.0
        if win_rate < self.required_live_win_rate:
            return
        if self.total_profit < self.required_live_profit:
            return
        self.live_trading_enabled = True
        self.metrics.feedback(
            "trading",
            severity=FeedbackSeverity.INFO,
            label="live_promotion",
            details={"win_rate": win_rate, "trades": self.total_trades, "profit": self.total_profit},
        )
        print("[trading-bot] conditions met; live trading enabled.")

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
        temp_scale = getattr(self.pipeline, "temperature_scale", 1.0)
        if temp_scale and temp_scale > 0:
            raw_prob = float(np.clip(summary["direction_prob"], 1e-6, 1.0 - 1e-6))
            logit = math.log(raw_prob / (1.0 - raw_prob))
            calibrated = 1.0 / (1.0 + math.exp(-logit / temp_scale))
            summary["direction_prob_raw"] = summary["direction_prob"]
            summary["direction_prob"] = float(np.clip(calibrated, 1e-6, 1.0 - 1e-6))
        return summary

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._bg_task:
            self._bg_task.cancel()
        await self.stream.stop()

    async def _handle_sample(self, sample: Dict[str, Any]) -> None:
        if self._processing_sample:
            if len(self._pending_queue) >= self._pending_queue.maxlen:
                dropped = self._pending_queue.popleft()
                self.metrics.feedback(
                    "stream",
                    severity=FeedbackSeverity.WARNING,
                    label="queue_drop",
                    details={"symbol": dropped.get("symbol"), "ts": dropped.get("ts")},
                )
            self._pending_queue.append(sample)
            return

        cycle_start = time.perf_counter()
        self._processing_sample = True
        try:
            self._buffer.append(sample)
            if len(self._buffer) < self.window_size:
                return

            sample_ts = float(sample.get("ts") or time.time())
            signature = (sample.get("symbol", ""), sample_ts)
            if signature == self._last_sample_signature:
                return
            self._last_sample_signature = signature

            model = self.pipeline.ensure_active_model()
            now = sample_ts
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
                    self._schedule_next_portfolio_refresh(now, success=True)
                except Exception as exc:
                    print(f"[portfolio] refresh failed: {exc}")
                    self._schedule_next_portfolio_refresh(now, success=False)
            self._ensure_model_bindings(model)
            history_snapshot = list(self._buffer)
            window_slice = history_snapshot[-self.window_size :]
            inputs = self._prepare_inputs(window_slice)
            try:
                preds = self._invoke_model(inputs)
            except Exception as exc:
                print(f"[trading-bot] prediction failed: {exc}")
                return
            pred_summary = self._summarise_predictions(preds)
            brain_summary = self._update_brain_state(sample, history_snapshot, pred_summary)
            await self._run_wallet_sync(reason="pre-schedule")
            directive = None
            try:
                allocation_map = self._compute_base_allocation(sample)
                self._propagate_horizon_bias()
                directive = self.scheduler.evaluate(
                    sample,
                    pred_summary,
                    self.portfolio,
                    base_allocation=allocation_map,
                    risk_budget=self.global_risk_budget,
                )
            except Exception as exc:
                print(f"[bus-scheduler] evaluation failed: {exc}")
            decision = await self._interpret_predictions(
                preds,
                sample,
                directive,
                pred_summary,
                brain_summary,
            )
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
            snapshot_latency = time.perf_counter() - cycle_start
            self._record_organism_snapshot(
                sample=sample,
                pred_summary=pred_summary,
                brain_summary=brain_summary,
                directive=directive,
                decision=decision,
                latency_s=snapshot_latency,
            )
        finally:
            latency = time.perf_counter() - cycle_start
            self._latency_window.append(latency)
            self.metrics.record(
                MetricStage.LIVE_TRADING,
                {"ttl_ms": latency * 1000.0},
                category="latency",
                meta={"window": len(self._buffer), "queue_depth": len(self._pending_queue)},
            )
            self._latency_samples += 1
            if self._latency_samples % 20 == 0 and self._latency_window:
                window_arr = np.asarray(self._latency_window, dtype=np.float64)
                ttl_avg_ms = float(np.mean(window_arr) * 1000.0)
                ttl_p95_ms = float(np.percentile(window_arr, 95) * 1000.0)
                self.metrics.record(
                    MetricStage.LIVE_TRADING,
                    {"ttl_ms_avg": ttl_avg_ms, "ttl_ms_p95": ttl_p95_ms},
                    category="latency_summary",
                    meta={"sample_count": len(window_arr)},
                )
                if ttl_p95_ms > 50.0:
                    self.metrics.feedback(
                        "latency",
                        severity=FeedbackSeverity.WARNING,
                        label="high_ttl_window",
                        details={"ttl_ms_p95": ttl_p95_ms, "ttl_ms_avg": ttl_avg_ms},
                    )
            if latency > 0.05:
                self.metrics.feedback(
                    "latency",
                    severity=FeedbackSeverity.WARNING,
                    label="high_ttl",
                    details={"ttl_ms": latency * 1000.0},
                )
            self._processing_sample = False
            if self._pending_queue:
                next_sample = self._pending_queue.popleft()
                asyncio.create_task(self._handle_sample(next_sample))

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
        brain_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = pred_summary or self._summarise_predictions(preds)
        brain = brain_summary or {}
        exit_conf_val = float(summary.get("exit_conf", 0.5))
        direction_prob = float(summary.get("direction_prob", 0.5))
        delta = float(summary.get("delta", 0.0))
        margin = float(summary.get("net_margin", 0.0))
        pnl = float(summary.get("net_pnl", margin))
        sample_ts = float(sample.get("ts", time.time()))
        enter_threshold = max(getattr(self.pipeline, "decision_threshold", 0.58), MIN_CONFIDENCE)
        exit_threshold = min(enter_threshold * 0.6, 0.5)
        max_hold_sec = float(os.getenv("MAX_HOLD_SECONDS", "3600"))
        graph_conf = float(brain.get("graph_confidence", 1.0) or 1.0)
        direction_prob = float(np.clip(direction_prob * graph_conf, 0.0, 1.0))
        swarm_bias = float(brain.get("swarm_bias", 0.0) or 0.0)
        margin += swarm_bias
        delta += swarm_bias
        memory_bias_val = brain.get("memory_bias")
        if memory_bias_val is not None:
            try:
                memory_bias = float(memory_bias_val)
                direction_prob = float(
                    np.clip(direction_prob + np.tanh(memory_bias) * 0.05, 0.0, 1.0)
                )
                margin += memory_bias * 0.02
            except Exception:
                pass
        opportunity_bias = brain.get("opportunity") or {}
        if opportunity_bias:
            try:
                opp_kind = str(opportunity_bias.get("kind") or "")
                opp_strength = abs(float(opportunity_bias.get("zscore") or 0.0))
                boost = min(0.15, opp_strength * 0.03)
                if opp_kind == "buy-low":
                    direction_prob = float(np.clip(direction_prob + boost, 0.0, 1.0))
                    margin += boost * 0.01
                elif opp_kind == "sell-high":
                    direction_prob = float(np.clip(direction_prob - boost, 0.0, 1.0))
                    margin -= boost * 0.01
            except Exception:
                pass
        threshold_scale = float(brain.get("threshold_scale", 1.0) or 1.0)
        enter_threshold = float(np.clip(enter_threshold * threshold_scale, 0.45, 0.9))
        exit_threshold = min(enter_threshold * 0.6, 0.5)
        scenario_defer = bool(brain.get("scenario_defer"))
        scenario_mod = float(brain.get("scenario_mod", 1.0) or 1.0)
        arb_signal = brain.get("arb_signal")
        if isinstance(arb_signal, dict):
            action = str(arb_signal.get("action") or "")
            confidence = float(arb_signal.get("confidence") or 0.0)
            sym_upper = str(sample.get("symbol", "")).upper()
            if "ETH" in sym_upper:
                if action == "buy_eth":
                    direction_prob = float(np.clip(direction_prob + confidence * 0.05, 0.0, 1.0))
                elif action == "sell_eth":
                    direction_prob = float(np.clip(direction_prob - confidence * 0.05, 0.0, 1.0))
                    margin -= confidence * 0.01

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

        base_token = route[0]
        quote_token = route[-1]
        chain_name = str(sample.get("chain", self.primary_chain)).lower() or self.primary_chain
        pos = self.positions.get(symbol)
        stable_target = next((tok for tok in route if tok.upper() in self.stable_tokens), "USDC")
        fees = 0.0015 + 0.005
        brain_payload = {}
        if brain:
            brain_payload = {
                key: value
                for key, value in brain.items()
                if key not in {"fingerprint"}
            }
            if brain.get("fingerprint") is not None:
                brain_payload["fingerprint"] = list(brain.get("fingerprint") or [])
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
            "session_id": self.ghost_session_id,
            "brain": brain_payload,
        }
        if brain.get("opportunity"):
            decision["opportunity"] = brain["opportunity"]
        if chain_name != self.primary_chain:
            required_float = float(os.getenv("CHAIN_EXPANSION_THRESHOLD", "2000"))
            if self.portfolio.stable_liquidity(self.primary_chain) < required_float:
                decision["status"] = "hold-nonprimary"
                decision["reason"] = "insufficient-float"
                return decision
        if time.time() < self._reflex_blocked_until:
            decision.update(
                {
                    "status": "reflex-blocked",
                    "reason": f"reflex:{self._reflex_block_reason or 'safety'}",
                    "blocked_until": self._reflex_blocked_until,
                }
            )
            return decision
        gas_required = self._estimate_gas_cost(chain_name, route)
        use_sim = not self.live_trading_enabled
        if use_sim:
            available_quote = self._get_quote_balance(chain_name, quote_token)
            available_base = float(pos.get("size", 0.0)) if pos else 0.0
            native_balance = max(
                self.sim_native_balances.get(chain_name, gas_required * self.gas_buffer_multiplier),
                gas_required,
            )
        else:
            available_quote = self.portfolio.get_quantity(quote_token, chain=chain_name)
            available_base = self.portfolio.get_quantity(base_token, chain=chain_name)
            native_balance = self.portfolio.get_native_balance(chain_name)

        trade_size = max(min(volume * self.max_trade_share, volume), 0.0)
        trade_size = min(trade_size, 100.0)
        if directive and directive.size > 0:
            trade_size = float(directive.size)
        if pos is None:
            trade_size *= max(0.0, scenario_mod)
        if scenario_defer and pos is None:
            decision.update(
                {
                    "status": "scenario-hold",
                    "reason": "scenario_spread",
                }
            )
            return decision
        if native_balance < gas_required:
            if self.live_trading_enabled:
                strategy = self._plan_gas_replenishment(
                    chain=chain_name,
                    route=route,
                    native_balance=native_balance,
                    gas_required=gas_required,
                    trade_size=trade_size,
                    price=price,
                    margin=margin,
                    pnl=pnl,
                    available_quote=available_quote,
                    symbol=symbol,
                )
                if strategy and strategy.get("profit_guard_passed") and strategy.get("stable_swap_plan"):
                    executed = self._rebalance_for_gas(chain_name, strategy)
                    if executed:
                        self.metrics.feedback(
                            "trading",
                            severity=FeedbackSeverity.INFO,
                            label="gas_rebalanced",
                            details={"chain": chain_name, "strategy": strategy, "mode": "auto"},
                        )
                        await self._run_wallet_sync(reason="post-gas-rebalance")
                        native_balance = self.portfolio.get_native_balance(chain_name)
                        available_quote = self.portfolio.get_quantity(quote_token, chain=chain_name)
                        if native_balance >= gas_required:
                            strategy = None
                if native_balance < gas_required:
                    strategy_severity = FeedbackSeverity.CRITICAL
                    details = {
                        "native_balance": native_balance,
                        "required": gas_required,
                        "chain": chain_name,
                    }
                    if strategy:
                        details["strategy"] = strategy
                        if strategy.get("profit_guard_passed") and strategy.get("stable_swap_plan"):
                            strategy_severity = FeedbackSeverity.WARNING
                        message = (
                            f"Native balance {native_balance:.6f} below required {gas_required:.6f} on {chain_name}."
                        )
                        self._record_advisory(
                            topic="gas_replenishment",
                            message=message,
                            severity=strategy_severity,
                            scope=f"{chain_name}:{symbol}",
                            recommendation=str(strategy.get("recommendation", "")),
                            meta=strategy,
                        )
                    self.metrics.feedback(
                        "trading",
                        severity=strategy_severity,
                        label="insufficient_gas",
                        details=details,
                    )
                    return decision
            else:
                # Simulated trading: replenish virtual gas buffer instead of touching live funds
                self.sim_native_balances[chain_name] = max(self.sim_native_balances.get(chain_name, 0.5), gas_required * self.gas_buffer_multiplier)
                native_balance = self.sim_native_balances[chain_name]
        if trade_size > 0.0 and price > 0.0 and available_quote > 0.0:
            max_affordable = max(0.0, available_quote / price)
            trade_size = min(trade_size, max_affordable * self.max_trade_share)
        adjustments = self._get_pair_adjustment(symbol)
        trade_size *= float(max(0.1, min(3.0, adjustments.get("size_multiplier", 1.0))))
        trade_size = max(0.0, trade_size)
        min_margin_required = max(fees * 1.5, SMALL_PROFIT_FLOOR)
        min_margin_required = max(0.0, min_margin_required + float(adjustments.get("margin_offset", 0.0)))
        expected_profit_units = margin * max(trade_size, 0.0) * max(price, 1e-9)
        if trade_size <= 0.0:
            if pos is not None:
                decision.update(
                    {
                        "unrealized": (price - pos["entry_price"]) * pos["size"],
                        "size": pos["size"],
                        "entry_price": pos["entry_price"],
                    }
                )
            else:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="insufficient_quote",
                    details={"quote_token": quote_token, "available": available_quote, "price": price},
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
            enter_threshold = max(0.5, min(0.99, enter_threshold + float(adjustments.get("enter_offset", 0.0))))
            if (
                direction_prob >= enter_threshold
                and exit_conf_val >= enter_threshold
                and margin >= min_margin_required
                and expected_profit_units >= SMALL_PROFIT_FLOOR
                and delta >= 0.0
            ):
                should_enter = True
                reason = "model-long"
        else:
            exit_threshold = max(0.05, min(enter_threshold * 0.95, exit_threshold + float(adjustments.get("exit_offset", 0.0))))
            if direction_prob < exit_threshold or exit_conf_val < 0.5:
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
                if guard_reasons and any("insufficient" in reason for reason in guard_reasons):
                    self._tune_allocation(symbol, positive=False, negative=True)
                return decision
            if price > 0.0 and available_quote > 0.0:
                trade_size = min(trade_size, max(0.0, available_quote / price))
            if trade_size <= 0.0:
                self._tune_allocation(symbol, positive=False, negative=True)
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
                "brain_snapshot": brain_payload,
                "expected_margin": margin,
                "entry_confidence": exit_conf_val,
                "direction_prob": direction_prob,
            }
            if brain.get("fingerprint"):
                try:
                    self.positions[symbol]["fingerprint"] = list(brain.get("fingerprint") or [])
                except Exception:
                    self.positions[symbol]["fingerprint"] = []
            decision.update(
                {
                    "action": "enter",
                    "status": f"{'live' if self.live_trading_enabled else 'ghost'}-entry",
                    "size": trade_size,
                    "entry_price": price,
                    "route": route,
                    "reason": reason or (directive.reason if directive else "model"),
                    "target_price": directive.target_price if directive else price * 1.05,
                    "horizon": directive.horizon if directive else None,
                    "trade_id": trade_id,
                    "entry_ts": sample_ts,
                    "wallet": "live" if self.live_trading_enabled else "ghost",
                    "session_id": self.ghost_session_id,
                }
            )
            if isinstance(decision.get("brain"), dict):
                decision["brain"]["entry_trade_id"] = trade_id
            exposure_delta = trade_size * price
            self.active_exposure[symbol] = self.active_exposure.get(symbol, 0.0) + exposure_delta
            self._tune_allocation(symbol, positive=True, negative=False)
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
            if not self.live_trading_enabled:
                quote_spent = trade_size * price
                self._adjust_quote_balance(chain_name, quote_token, -quote_spent)
                self._consume_sim_gas(chain_name, gas_required)
            return decision

        if should_exit and pos is not None:
            await self._run_wallet_sync(reason="pre-exit")
            held_size = float(pos["size"])
            exit_size = min(held_size, trade_size, available_base)
            if exit_size <= 0.0:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="insufficient_base",
                    details={
                        "base_token": base_token,
                        "available": available_base,
                        "held": held_size,
                    },
                )
                return decision
            profit = (price - pos["entry_price"]) * exit_size
            self.total_trades += 1
            if profit > 0:
                self.wins += 1
            checkpoint = 0.0
            next_stop = route[1] if len(route) > 1 else None
            notional = max(exit_size * pos["entry_price"], 1e-9)
            realized_margin = profit / notional if notional else 0.0
            predicted_margin = float(pos.get("expected_margin", margin))
            entry_confidence = float(pos.get("entry_confidence", exit_conf_val))
            self.equilibrium.observe(
                predicted_margin=predicted_margin,
                realized_margin=realized_margin,
                confidence=entry_confidence,
            )
            equilibrium_ready = self.equilibrium.is_equilibrium()
            self._nash_equilibrium_reached = equilibrium_ready
            savings_event = None
            equilibrium_score = self.equilibrium.score()
            if profit > 0 and equilibrium_ready:
                checkpoint = profit * self.stable_checkpoint_ratio
                self.stable_bank += checkpoint
                profit -= checkpoint
                savings_event = self.savings.record_allocation(
                    amount=checkpoint,
                    token=stable_target,
                    mode="live" if self.live_trading_enabled else "ghost",
                    equilibrium_score=self.equilibrium.score(),
                    trade_id=pos.get("trade_id") or f"{symbol}-{int(pos.get('ts', sample_ts))}",
                )
                decision.setdefault("savings", {})["checkpoint"] = savings_event.to_dict()
                transfers = self.savings.drain_ready_transfers()
                for transfer in transfers:
                    self._handle_savings_transfer(transfer)
            elif profit <= 0 and (sample_ts - pos.get("entry_ts", pos.get("ts", sample_ts))) < max_hold_sec:
                decision.update({"status": "hold-negative", "reason": reason or "hold"})
                return decision
            self.total_profit += profit
            self.realized_profit += profit
            trade_id = pos.get("trade_id") or f"{symbol}-{int(pos.get('ts', sample_ts))}"
            entry_ts = float(pos.get("entry_ts", pos.get("ts", sample_ts)))
            duration_sec = max(0.0, sample_ts - entry_ts)
            pos_fingerprint = pos.get("fingerprint")
            if pos_fingerprint is not None:
                try:
                    self.memory.add(
                        np.asarray(pos_fingerprint, dtype=np.float32),
                        profit,
                        duration=duration_sec,
                        size=exit_size,
                    )
                except Exception:
                    pass
            if profit != 0.0:
                try:
                    self.graph.upsert_node(
                        f"{symbol}:pnl",
                        "portfolio",
                        profit,
                        sample_ts,
                        duration=duration_sec,
                    )
                    strength = float(np.tanh(profit / max(abs(pos["entry_price"]), 1e-6)))
                    self.graph.reinforce(symbol, f"{symbol}:pnl", sample_ts, strength)
                except Exception:
                    pass
            del self.positions[symbol]
            decision.update(
                {
                    "action": "exit",
                    "status": f"{'live' if self.live_trading_enabled else 'ghost'}-exit",
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
                    "wallet": "live" if self.live_trading_enabled else "ghost",
                    "session_id": self.ghost_session_id,
                    "equilibrium_score": equilibrium_score,
                    "nash_equilibrium": equilibrium_ready,
                }
            )
            if isinstance(decision.get("brain"), dict):
                decision["brain"]["realized_profit"] = profit
            exposure_delta = exit_size * price
            remaining = max(0.0, self.active_exposure.get(symbol, 0.0) - exposure_delta)
            if remaining <= 1e-6:
                self.active_exposure.pop(symbol, None)
            else:
                self.active_exposure[symbol] = remaining
            exit_metrics = {
                "profit": profit,
                "checkpoint": checkpoint,
                "duration_sec": duration_sec,
                "bank_balance": self.stable_bank,
                "total_profit": self.total_profit,
                "win_rate": self.wins / max(1, self.total_trades),
                "exit_price": price,
                "entry_price": pos["entry_price"],
                "equilibrium_score": equilibrium_score,
                "nash_equilibrium": equilibrium_ready,
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
            if not self.live_trading_enabled:
                quote_gain = exit_size * price
                self._adjust_quote_balance(chain_name, quote_token, quote_gain)
                self._consume_sim_gas(chain_name, gas_required)
                self.sim_native_balances[chain_name] = max(self.sim_native_balances.get(chain_name, 0.5), 0.5)
                self._check_sim_restart()
            self._maybe_promote_to_live()
            if profit > 0 and decision.get("wallet", "ghost") == "live":
                strategy = self._plan_gas_replenishment(
                    chain=chain_name,
                    route=route,
                    native_balance=native_balance,
                    gas_required=gas_required,
                    trade_size=trade_size,
                    price=price,
                    margin=margin,
                    pnl=pnl,
                    available_quote=available_quote,
                    symbol=symbol,
                )
                if strategy:
                    executed = self._rebalance_for_gas(chain_name, strategy)
                    if executed:
                        self.metrics.feedback(
                            "trading",
                            severity=FeedbackSeverity.INFO,
                            label="gas_rebalanced",
                            details={"chain": chain_name, "strategy": strategy},
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
        if not self._enable_bg_refinement:
            return

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

    def _schedule_next_portfolio_refresh(self, now: float, *, success: bool) -> None:
        base = self.portfolio.refresh_interval
        if self._latency_window:
            avg_ttl = float(np.mean(self._latency_window))
        else:
            avg_ttl = 0.02
        adjustment = 1.0 + min(1.0, avg_ttl * 10.0)
        if not success:
            adjustment = 0.5
        interval = float(np.clip(base * adjustment, 60.0, 900.0))
        self._portfolio_next_refresh = now + interval
        self._maybe_expand_chains()

    def _maybe_expand_chains(self) -> None:
        """
        Start on the primary chain (Base) and progressively enable additional
        networks only when the stablecoin float justifies the added latency.
        """
        total_stable = self.portfolio.stable_liquidity(self.primary_chain)
        if total_stable < float(os.getenv("CHAIN_EXPANSION_THRESHOLD", "2000")):
            return
        desired_chains = {self.primary_chain}
        optional_chains = os.getenv("SECONDARY_CHAINS", "ethereum,arbitrum,optimism").split(",")
        for chain in optional_chains:
            chain_clean = chain.strip().lower()
            if not chain_clean:
                continue
            desired_chains.add(chain_clean)
            if len(desired_chains) >= 3:
                break
        current = set(self.portfolio.chains)
        if desired_chains != current:
            self.portfolio.chains = tuple(desired_chains)

    def _estimate_gas_cost(self, chain: str, route: List[str]) -> float:
        base_cost = float(os.getenv("ESTIMATED_GAS_NATIVE", "0.001"))
        hop_cost = float(os.getenv("ESTIMATED_GAS_HOP", "0.0002"))
        hops = max(0, len(route) - 1)
        return base_cost + hops * hop_cost

    def _estimate_native_price(self, chain: str, route: List[str], price: float, symbol: str) -> float:
        chain_l = chain.lower()
        native_symbol = NATIVE_SYMBOL.get(chain_l, chain.upper())
        price_candidate = float(price or 0.0)
        route_upper = [token.upper() for token in route]
        if (
            price_candidate > 0.0
            and native_symbol in route_upper
            and route_upper[-1] in self.stable_tokens
        ):
            return price_candidate
        try:
            row = self.db.fetch_price(chain, native_symbol)
            if row:
                candidate = row.get("usd")
                if candidate is not None:
                    price_candidate = float(candidate)
        except Exception:
            pass
        if price_candidate <= 0.0:
            env_key = f"FALLBACK_NATIVE_PRICE_{chain_l.upper()}"
            fallback_raw = os.getenv(env_key) or os.getenv("FALLBACK_NATIVE_PRICE")
            try:
                fallback_val = float(fallback_raw) if fallback_raw else FALLBACK_NATIVE_PRICE
            except Exception:
                fallback_val = FALLBACK_NATIVE_PRICE
            price_candidate = fallback_val
        if price_candidate <= 0.0:
            if native_symbol in {"ETH", "WETH"} or symbol.upper().endswith("WETH"):
                return 1800.0
            return 100.0
        return price_candidate

    def _plan_gas_replenishment(
        self,
        *,
        chain: str,
        route: List[str],
        native_balance: float,
        gas_required: float,
        trade_size: float,
        price: float,
        margin: float,
        pnl: float,
        available_quote: float,
        symbol: str,
    ) -> Optional[Dict[str, Any]]:
        deficit = max(0.0, gas_required - native_balance)
        if deficit <= 0.0:
            return None
        target_native = max(deficit * self.gas_buffer_multiplier, deficit)
        native_price = self._estimate_native_price(chain, route, price, symbol)
        expected_profit_unit = max(float(margin), float(pnl))
        expected_profit_usd = max(
            float(pnl),
            float(margin) * max(trade_size, 1.0),
        )
        estimated_gas_cost_usd = gas_required * native_price
        target_buffer_usd = target_native * native_price
        profit_guard_passed = expected_profit_usd > (estimated_gas_cost_usd * self.gas_profit_guard)

        chain_l = chain.lower()
        stable_holdings = [
            holding
            for (key_chain, sym), holding in self.portfolio.holdings.items()
            if key_chain == chain_l and sym.upper() in self.stable_tokens
        ]
        stable_holdings.sort(key=lambda entry: entry.usd, reverse=True)

        swap_plan: List[Dict[str, Any]] = []
        native_from_stables = 0.0
        remaining_native = target_native
        for holding in stable_holdings:
            if remaining_native <= 0.0:
                break
            max_native_from_holding = holding.usd / max(native_price, 1e-9)
            convert_native = min(remaining_native, max_native_from_holding)
            if convert_native <= 0.0:
                continue
            spend_stable = convert_native * native_price
            min_swap_usd = float(os.getenv("GAS_MIN_SWAP_USD", "1.0"))
            if spend_stable < min_swap_usd:
                spend_stable = min(min_swap_usd, holding.usd)
                convert_native = spend_stable / max(native_price, 1e-9)
            spend_stable = min(spend_stable, holding.usd, holding.quantity)
            convert_native = spend_stable / max(native_price, 1e-9)
            if spend_stable <= 0.0 or convert_native <= 0.0:
                continue
            swap_plan.append(
                {
                    "symbol": holding.symbol,
                    "token": holding.token,
                    "spend_stable": round(spend_stable, 6),
                    "obtain_native": round(convert_native, 8),
                    "available": holding.quantity,
                    "usd_value": holding.usd,
                }
            )
            native_from_stables += convert_native
            remaining_native -= convert_native

        bridge_candidates = [
            {"chain": other_chain, "native_balance": bal}
            for other_chain, bal in self.portfolio.native_balances.items()
            if other_chain != chain_l and bal > 0.0
        ]
        bridge_candidates.sort(key=lambda entry: entry["native_balance"], reverse=True)

        remaining_native_gap = max(0.0, remaining_native)
        signature = f"{chain_l}:{symbol}:{int(round(deficit * 1e6))}:{int(round(target_native * 1e6))}:{1 if profit_guard_passed else 0}"

        if not profit_guard_passed:
            recommendation = (
                "Predicted profit does not cover gas costs. Pause trade and replenish native balance before resuming."
            )
        elif swap_plan and remaining_native_gap <= 1e-6:
            primary_swap = swap_plan[0]
            recommendation = (
                f"Swap {primary_swap['spend_stable']:.2f} {primary_swap['symbol']} to native to restore gas buffer."
            )
        elif swap_plan:
            recommendation = (
                "Swap available stablecoins for native gas, then bridge or deposit remaining shortfall."
            )
        else:
            recommendation = (
                "Bridge or deposit native tokens to replenish gas; no stables available on the active chain."
            )

        plan = {
            "chain": chain_l,
            "symbol": symbol,
            "native_balance": native_balance,
            "gas_required": gas_required,
            "deficit_native": deficit,
            "target_native": target_native,
            "native_price_usd": native_price,
            "expected_profit_usd": expected_profit_usd,
            "estimated_gas_cost_usd": estimated_gas_cost_usd,
            "profit_guard_passed": profit_guard_passed,
            "stable_swap_plan": swap_plan,
            "bridge_candidates": bridge_candidates,
            "remaining_native_gap": remaining_native_gap,
            "available_quote": available_quote,
            "recommendation": recommendation,
            "signature": signature,
        }
        if swap_plan:
            plan["stable_coverage_native"] = native_from_stables
        return plan

    def _record_advisory(
        self,
        *,
        topic: str,
        message: str,
        severity: str,
        scope: str,
        recommendation: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = meta or {}
        signature = str(meta.get("signature") or "")
        if signature and signature == self._last_gas_advisory_signature:
            return
        advisory_id: Optional[int] = None
        try:
            advisory_id = self.db.record_advisory(
                topic=topic,
                message=message,
                severity=severity,
                scope=scope,
                recommendation=recommendation,
                meta=meta,
            )
        except Exception as exc:
            print(f"[advisory] failed to persist {topic}: {exc}")
        details = {
            "topic": topic,
            "scope": scope,
            "message": message,
            "recommendation": recommendation,
            "meta": meta,
        }
        if advisory_id is not None:
            details["advisory_id"] = advisory_id
        self.metrics.feedback("advisory", severity=severity, label=topic, details=details)
        if signature:
            self._last_gas_advisory_signature = signature

    def _rebalance_for_gas(self, chain: str, strategy: Dict[str, Any]) -> bool:
        if not strategy or not strategy.get("stable_swap_plan"):
            return False
        if self._bridge is None:
            self._bridge = self._init_bridge()
        if self._bridge is None:
            return False
        try:
            from services.swap_service import SwapService  # type: ignore
        except Exception as exc:
            self.metrics.feedback(
                "trading",
                severity=FeedbackSeverity.WARNING,
                label="gas_swap_unavailable",
                details={"reason": str(exc)},
            )
            return False

        swapper = SwapService(self._bridge)
        slippage = int(os.getenv("GAS_REFILL_SLIPPAGE_BPS", "75"))
        executed = False
        for plan in strategy.get("stable_swap_plan", []):
            spend = float(plan.get("spend_stable", 0.0))
            token = plan.get("token")
            if spend <= 0.0 or not token:
                continue
            try:
                swapper.swap(chain=chain, sell=token, buy="native", amount_human=f"{spend:.4f}", slippage_bps=slippage)
                executed = True
            except Exception as exc:
                self.metrics.feedback(
                    "trading",
                    severity=FeedbackSeverity.WARNING,
                    label="gas_swap_failed",
                    details={"token": token, "amount": spend, "reason": str(exc)},
                )
        return executed

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
        positions: Dict[str, Dict[str, Any]] = {}
        for sym, pos in ghost.get("positions", {}).items():
            if not isinstance(pos, dict):
                continue
            entry_ts_val = float(pos.get("entry_ts", pos.get("ts", time.time())))
            fingerprint_val = pos.get("fingerprint")
            if isinstance(fingerprint_val, np.ndarray):
                fingerprint_list = fingerprint_val.tolist()
            elif isinstance(fingerprint_val, list):
                fingerprint_list = fingerprint_val
            else:
                fingerprint_list = []
            positions[str(sym)] = {
                "entry_price": float(pos.get("entry_price", 0.0)),
                "size": float(pos.get("size", 0.0)),
                "ts": float(pos.get("ts", time.time())),
                "entry_ts": entry_ts_val,
                "route": list(pos.get("route", [])),
                "bus_index": int(pos.get("bus_index", 0)),
                "trade_id": pos.get("trade_id"),
                "target_price": pos.get("target_price"),
                "fingerprint": fingerprint_list,
                "brain_snapshot": pos.get("brain_snapshot"),
            }
        self.positions = positions
        routes = ghost.get("routes")
        if isinstance(routes, dict):
            self.bus_routes = {sym: list(tokens) for sym, tokens in routes.items()}
        sim_quotes = ghost.get("sim_quote_balances")
        if isinstance(sim_quotes, dict):
            self.sim_quote_balances = {
                (str(key[0]).lower(), str(key[1]).upper()) if isinstance(key, (list, tuple)) and len(key) == 2 else (self.primary_chain.lower(), str(key).upper()): float(value)
                for key, value in sim_quotes.items()
            }
        sim_native = ghost.get("sim_native_balances")
        if isinstance(sim_native, dict):
            self.sim_native_balances = {str(chain).lower(): float(value) for chain, value in sim_native.items()}
        self.ghost_session_id = int(ghost.get("session_id", self.ghost_session_id)) or 1
        exposure = ghost.get("active_exposure")
        if isinstance(exposure, dict):
            self.active_exposure = {str(sym): float(value) for sym, value in exposure.items()}

    def _record_organism_snapshot(
        self,
        *,
        sample: Optional[Dict[str, Any]],
        pred_summary: Optional[Dict[str, Any]],
        brain_summary: Optional[Dict[str, Any]],
        directive: Optional[TradeDirective],
        decision: Optional[Dict[str, Any]],
        latency_s: Optional[float],
    ) -> None:
        if not getattr(self, "db", None):
            return
        now = time.time()
        if (now - self._last_snapshot_ts) < self._snapshot_interval:
            return
        discovery_snapshot = self._get_discovery_snapshot(now)
        try:
            snapshot = build_snapshot(
                bot=self,
                sample=sample,
                pred_summary=pred_summary,
                brain_summary=brain_summary,
                directive=directive,
                decision=decision,
                latency_s=latency_s,
                latency_window=list(self._latency_window),
                pending_depth=len(self._pending_queue),
                discovery_snapshot=discovery_snapshot,
                last_windows=self._last_windows,
            )
        except Exception as exc:
            print(f"[organism] snapshot build failed: {exc}")
            return
        try:
            self.db.record_organism_snapshot(snapshot)
            self._last_snapshot_ts = now
        except Exception as exc:
            print(f"[organism] snapshot persist failed: {exc}")

    def _get_discovery_snapshot(self, now: float) -> Dict[str, Any]:
        if self._discovery_cache and (now - self._discovery_cache_ts) < 300:
            return self._discovery_cache
        try:
            snapshot = {
                "status_counts": self.db.discovery_status_counts(),
                "recent_events": self.db.discovery_recent_events(limit=12),
                "recent_honeypots": self.db.discovery_recent_honeypots(limit=6),
            }
        except Exception:
            snapshot = {}
        self._discovery_cache = snapshot
        self._discovery_cache_ts = now
        return snapshot

    def _save_state(self) -> None:
        try:
            state = self.db.load_state()
        except Exception:
            state = {}
        if not isinstance(state, dict):
            state = {}
        positions_payload: Dict[str, Dict[str, Any]] = {}
        for sym, pos in self.positions.items():
            pos_copy = dict(pos)
            fingerprint_val = pos_copy.get("fingerprint")
            if isinstance(fingerprint_val, np.ndarray):
                pos_copy["fingerprint"] = fingerprint_val.tolist()
            positions_payload[str(sym)] = pos_copy
        state["ghost_trading"] = {
            "stable_bank": self.stable_bank,
            "total_profit": self.total_profit,
            "realized_profit": self.realized_profit,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "positions": positions_payload,
            "routes": self.bus_routes,
            "sim_quote_balances": {f"{chain}:{symbol}": amount for (chain, symbol), amount in self.sim_quote_balances.items()},
            "sim_native_balances": self.sim_native_balances,
            "session_id": self.ghost_session_id,
            "active_exposure": self.active_exposure,
        }
        try:
            self.db.save_state(state)
        except Exception:
            pass
