from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from trading.metrics import MetricsCollector, MetricStage

from db import get_db, TradingDatabase
from trading.data_stream import _split_symbol
from trading.portfolio import PortfolioState
from trading.constants import PRIMARY_CHAIN
from trading.opportunity import OpportunitySignal

HORIZON_DEFAULTS: List[Tuple[str, int]] = [
    ("5m", 5 * 60),
    ("15m", 15 * 60),
    ("30m", 30 * 60),
    ("1h", 60 * 60),
    ("3h", 3 * 60 * 60),
    ("6h", 6 * 60 * 60),
    ("12h", 12 * 60 * 60),
    ("1d", 24 * 60 * 60),
    ("3d", 3 * 24 * 60 * 60),
    ("5d", 5 * 24 * 60 * 60),
    ("1w", 7 * 24 * 60 * 60),
    ("1m", 30 * 24 * 60 * 60),
    ("3m", 90 * 24 * 60 * 60),
    ("6m", 180 * 24 * 60 * 60),
]


class HorizonAccuracyTracker:
    def __init__(self, horizons: Sequence[Tuple[str, int]], window: int = 256) -> None:
        self.window = max(16, int(window))
        self._history: Dict[str, Deque[Dict[str, float]]] = {
            label: deque(maxlen=self.window) for label, _ in horizons
        }

    def record(self, label: str, predicted_return: float, realized_return: float) -> None:
        error = abs(predicted_return - realized_return)
        bucket = self._history.setdefault(label, deque(maxlen=self.window))
        bucket.append(
            {
                "error": error,
                "realized": realized_return,
            }
        )

    def mae(self, label: str) -> float:
        bucket = self._history.get(label)
        if not bucket:
            return 0.0
        return float(np.mean([entry["error"] for entry in bucket]))

    def quality(self, label: str) -> float:
        mae = self.mae(label)
        return float(1.0 / (1.0 + mae))

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            label: {
                "samples": float(len(records)),
                "mae": float(np.mean([r["error"] for r in records])) if records else 0.0,
                "avg_realized": float(np.mean([r["realized"] for r in records])) if records else 0.0,
            }
            for label, records in self._history.items()
        }

    def count(self, label: str) -> int:
        bucket = self._history.get(label)
        return len(bucket) if bucket else 0


@dataclass
class HorizonSignal:
    label: str
    seconds: int
    predicted_price: float
    expected_return: float
    zscore: float
    historical_mae: float = 0.0


@dataclass
class TradeDirective:
    action: str  # "enter" or "exit"
    symbol: str
    base_token: str
    quote_token: str
    size: float
    target_price: float
    horizon: str
    confidence: float
    expected_return: float
    reason: str

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["expected_return"] = float(self.expected_return)
        payload["confidence"] = float(self.confidence)
        payload["size"] = float(self.size)
        payload["target_price"] = float(self.target_price)
        return payload


@dataclass
class RouteState:
    symbol: str
    base_token: str
    quote_token: str
    samples: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=720))
    last_directive: Optional[TradeDirective] = None
    last_update: float = 0.0
    last_signature: Optional[Tuple[float, float]] = None
    cached_signals: Optional[List[HorizonSignal]] = None
    forecast_signature: Optional[Tuple[int, float, float]] = None
    pending_predictions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2048))


class TridentUSSATSolver:
    """
    Lightweight, portfolio-style search inspired by the TRIDENT-US SAT/UNSAT
    description. The solver treats candidate directives as assignments and
    runs a small family of scoring functions in a fair, weighted round-robin
    to pick the best verified directive.
    """

    def __init__(self, *, explore: float = 0.06, decay: float = 0.9) -> None:
        self.explore = max(0.01, float(explore))
        self.decay = max(0.5, min(0.99, float(decay)))
        self.weights: Dict[str, float] = {
            "expected_return": 0.34,
            "confidence_weighted": 0.33,
            "risk_buffered": 0.33,
        }
        self.last_trace: Dict[str, Any] = {}

    # Candidate shape: {"directive": TradeDirective, "score": float, "meta": {...}}
    def select(self, candidates: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> Optional[TradeDirective]:
        if not candidates:
            return None
        verified = [cand for cand in candidates if self._verify(cand, context)]
        if not verified:
            return None
        strategies = self._strategies()
        min_weight = self.explore / max(1, len(strategies))

        choice_scores: Dict[int, float] = {}
        trace: List[Dict[str, Any]] = []
        updated_weights: Dict[str, float] = {}

        for name, scorer in strategies.items():
            weight_prior = self.weights.get(name, 1.0 / max(1, len(strategies)))
            try:
                best = max(verified, key=lambda cand: scorer(cand, context))
            except Exception:
                continue
            raw_score = scorer(best, context)
            reward = self._normalise_reward(raw_score)
            blended_weight = self.decay * weight_prior + (1.0 - self.decay) * reward
            updated_weights[name] = blended_weight
            trace.append(
                {
                    "strategy": name,
                    "raw_score": float(raw_score),
                    "reward": reward,
                    "selected_action": getattr(best.get("directive"), "action", "unknown"),
                }
            )
            best_id = id(best)
            choice_scores[best_id] = choice_scores.get(best_id, 0.0) + blended_weight * raw_score

        if not updated_weights:
            return verified[0]["directive"]

        total_weight = sum(updated_weights.values()) or 1.0
        normalised_weights = {k: max(min_weight, v / total_weight) for k, v in updated_weights.items()}
        self.weights = normalised_weights

        # Apply normalised weights to scores for the final selection pass
        final_scores: Dict[int, float] = {}
        for name, scorer in strategies.items():
            weight = normalised_weights.get(name, min_weight)
            try:
                best = max(verified, key=lambda cand: scorer(cand, context))
            except Exception:
                continue
            best_id = id(best)
            final_scores[best_id] = final_scores.get(best_id, 0.0) + weight * scorer(best, context)

        if not final_scores:
            return verified[0]["directive"]

        selected_id = max(final_scores, key=final_scores.get)
        selected = next(cand for cand in verified if id(cand) == selected_id)
        self.last_trace = {"trace": trace, "weights": dict(self.weights)}
        return selected["directive"]

    def _normalise_reward(self, value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        clipped = max(-3.0, min(3.0, value))
        return 0.5 + 0.5 * math.tanh(clipped)

    def _verify(self, candidate: Dict[str, Any], context: Dict[str, Any]) -> bool:
        directive = candidate.get("directive")
        meta = candidate.get("meta", {}) or {}
        if not isinstance(directive, TradeDirective):
            return False
        if not math.isfinite(directive.size) or directive.size <= 0:
            return False
        if not math.isfinite(directive.target_price) or directive.target_price <= 0:
            return False
        native_balance = float(context.get("native_balance", 0.0))
        min_native = float(context.get("min_native", 0.01))
        if native_balance < min_native:
            return False
        risk_budget = float(context.get("risk_budget", 1.0))
        if risk_budget <= 0:
            return False
        confidence = float(meta.get("confidence", 0.0))
        direction_prob = float(meta.get("direction_prob", 0.5))
        if confidence <= 0.0 or direction_prob <= 0.0:
            return False
        return True

    def _strategies(self) -> Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], float]]:
        return {
            "expected_return": lambda cand, _: float(cand.get("score", 0.0)),
            "confidence_weighted": lambda cand, _: float(cand.get("score", 0.0))
            * float((cand.get("meta") or {}).get("confidence", 1.0)),
            "risk_buffered": lambda cand, ctx: float(cand.get("score", 0.0))
            - float(ctx.get("fee_rate", 0.0))
            - float((cand.get("meta") or {}).get("risk_penalty", 0.0)),
        }


class BusScheduler:
    """
    Maintains multi-horizon forecasts for each trading pair and produces
    actionable directives for the trading bot (ghost or live).
    """

    def __init__(
        self,
        *,
        horizons: Optional[Iterable[Tuple[str, int]]] = None,
        min_profit: float = 0.05,
        fee_buffer: float = 0.002,
        tax_buffer: float = 0.003,
        history_limit_sec: int = 3 * 24 * 60 * 60,
        db: Optional[TradingDatabase] = None,
    ) -> None:
        self.db = db or get_db()
        self.horizons = list(horizons) if horizons is not None else HORIZON_DEFAULTS
        self.min_profit = min_profit
        self.fee_buffer = fee_buffer
        self.tax_buffer = tax_buffer
        self.history_limit_sec = history_limit_sec
        self.routes: Dict[str, RouteState] = {}
        self.accuracy = HorizonAccuracyTracker(self.horizons)
        seconds = [sec for _, sec in self.horizons]
        self._horizon_min = min(seconds) if seconds else 60
        self._horizon_span = max(seconds) - self._horizon_min if seconds else 1
        self._opportunity_bias: Dict[str, OpportunitySignal] = {}
        self._bucket_bias: Dict[str, float] = {"short": 1.0, "mid": 1.0, "long": 1.0}
        self._gas_alert_cb: Optional[Callable[[str, float], None]] = None
        self._last_gas_alert_ts: float = 0.0
        self._gas_alert_interval: float = 180.0
        self._trident = TridentUSSATSolver()
        self.gas_roundtrip_fee = float(os.getenv("SCHEDULER_GAS_ROUNDTRIP_RATIO", os.getenv("GAS_ROUNDTRIP_FEE_RATIO", "0.0025")))
        self.slippage_bps = int(os.getenv("SCHEDULER_SLIPPAGE_BPS", "50"))
        self.spread_floor = float(os.getenv("SCHEDULER_SPREAD_FLOOR", "0.002"))
        self.depth_floor_usd = float(os.getenv("SCHEDULER_DEPTH_FLOOR_USD", "5000"))
        self._log_filters = os.getenv("SCHEDULER_LOG_FILTER", "0").lower() in {"1", "true", "yes", "on"}
        self._filter_metrics: List[Dict[str, Any]] = []
        self._metrics_collector = MetricsCollector(self.db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sample: Dict[str, float],
        pred_summary: Dict[str, float],
        portfolio: PortfolioState,
        *,
        base_allocation: Optional[Dict[str, float]] = None,
        risk_budget: float = 1.0,
    ) -> Optional[TradeDirective]:
        """Update internal state with the latest sample and return a directive."""
        state = self._update_state(sample)
        signals = self._forecast(state)
        if not signals:
            return None

        last_price = state.samples[-1][1]
        last_volume = state.samples[-1][2]
        direction_prob = float(pred_summary.get("direction_prob", 0.5))
        confidence = float(pred_summary.get("exit_conf", 0.5))
        net_margin = float(pred_summary.get("net_margin", 0.0))
        chain_name = str(sample.get("chain", PRIMARY_CHAIN)).lower()

        available_quote = portfolio.get_quantity(state.quote_token, chain=chain_name)
        available_base = portfolio.get_quantity(state.base_token, chain=chain_name)
        native_balance = portfolio.get_native_balance(chain_name)
        # crude gas safety
        if native_balance < 0.01:
            self._emit_gas_alert(chain_name, native_balance)
            return None

        best_long = max(signals, key=self._score_signal)
        best_short = min(signals, key=self._score_signal)

        fee_rate = self.fee_buffer + self.tax_buffer + self.gas_roundtrip_fee + (self.slippage_bps / 10000.0)
        candidates: List[Dict[str, Any]] = []
        context = {
            "native_balance": native_balance,
            "min_native": 0.01,
            "fee_rate": fee_rate,
            "risk_budget": risk_budget,
            "direction_prob": direction_prob,
        }
        # Pause if recent data is too sparse
        if len(state.samples) < 12:
            return None
        gap_cutoff = state.samples[-1][0] - state.samples[0][0]
        if gap_cutoff > self.history_limit_sec * 1.5 and len(state.samples) < state.samples.maxlen * 0.5:
            return None
        # crude spread/depth checks: skip if implied spread too high or volume too low
        implied_spread = abs(self._bias(state)) if hasattr(self, "_bias") else 0.0
        if implied_spread > max(self.spread_floor, 0.01):
            if self._log_filters:
                print(f"[bus-scheduler] skip {state.symbol}: spread {implied_spread:.4f} > floor {self.spread_floor:.4f}")
            self._filter_metrics.append(
                {
                    "symbol": state.symbol,
                    "reason": "spread",
                    "spread": implied_spread,
                    "spread_floor": self.spread_floor,
                    "ts": time.time(),
                }
            )
            return None
        if last_volume * last_price < self.depth_floor_usd:
            if self._log_filters:
                print(f"[bus-scheduler] skip {state.symbol}: depth {last_volume*last_price:.2f} < floor {self.depth_floor_usd:.2f}")
            self._filter_metrics.append(
                {
                    "symbol": state.symbol,
                    "reason": "depth",
                    "depth_usd": last_volume * last_price,
                    "depth_floor": self.depth_floor_usd,
                    "ts": time.time(),
                }
            )
            return None
        # Enter: use quote asset to buy base
        if available_quote > 0:
            long_quality = self.accuracy.quality(best_long.label)
            weight = self._horizon_weight(best_long.label, best_long.seconds)
            expected = best_long.expected_return * long_quality * weight
            expected -= fee_rate
            risk_factor = min(1.0, max(expected - self.min_profit, 0.0) * 5.0)
            allocation = base_allocation.get(state.symbol, 0.0) if base_allocation else 0.0
            max_allocation = max(allocation * risk_budget, 0.0)
            if (
                expected > self.min_profit
                and direction_prob >= 0.6
                and confidence >= 0.6
                and (net_margin >= self.min_profit)
            ):
                size_quote = available_quote * min(0.35, max(0.08, risk_factor * 0.25))
                if max_allocation > 0:
                    size_quote = min(size_quote, max_allocation)
                size_base = size_quote / max(last_price, 1e-9)
                if size_base > 0:
                    target_price = last_price * (1.0 + max(expected - fee_rate, self.min_profit))
                    directive = TradeDirective(
                        action="enter",
                        symbol=state.symbol,
                        base_token=state.base_token,
                        quote_token=state.quote_token,
                        size=float(size_base),
                        target_price=float(target_price),
                        horizon=best_long.label,
                        confidence=confidence,
                        expected_return=float(expected),
                        reason=f"forecast {best_long.label} {expected:.2%} w/ dir={direction_prob:.2f}",
                    )
                    candidates.append(
                        {
                            "directive": directive,
                            "score": float(expected),
                            "meta": {
                                "confidence": confidence,
                                "direction_prob": direction_prob,
                                "risk_factor": risk_factor,
                                "horizon_weight": weight,
                                "quality": long_quality,
                                "risk_penalty": fee_rate,
                            },
                        }
                    )

        # Exit: sell base into quote when projected drawdown
        if available_base > 0:
            short_quality = self.accuracy.quality(best_short.label)
            weight = self._horizon_weight(best_short.label, best_short.seconds)
            expected = best_short.expected_return * short_quality * weight - fee_rate
            if expected < -self.min_profit or direction_prob <= 0.4 or net_margin < 0:
                size_base = available_base * 0.5
                if size_base > 0:
                    target_price = last_price * (1.0 + expected + fee_rate)
                    directive = TradeDirective(
                        action="exit",
                        symbol=state.symbol,
                        base_token=state.base_token,
                        quote_token=state.quote_token,
                        size=float(size_base),
                        target_price=float(target_price),
                        horizon=best_short.label,
                        confidence=confidence,
                        expected_return=float(expected),
                        reason=f"drawdown {best_short.label} {expected:.2%}",
                    )
                    candidates.append(
                        {
                            "directive": directive,
                            "score": float(expected),
                            "meta": {
                                "confidence": confidence,
                                "direction_prob": direction_prob,
                                "risk_factor": 0.0,
                                "horizon_weight": weight,
                                "quality": short_quality,
                                "risk_penalty": fee_rate,
                            },
                        }
                    )

        if not candidates:
            return None
        chosen = self._trident.select(candidates, context)
        if chosen:
            state.last_directive = chosen
            return chosen
        fallback = max(candidates, key=lambda cand: cand.get("score", 0.0))
        state.last_directive = fallback["directive"]
        return fallback["directive"]

    def set_bucket_bias(self, bucket_bias: Dict[str, float]) -> None:
        if not bucket_bias:
            return
        for key, value in bucket_bias.items():
            if key not in self._bucket_bias:
                continue
            try:
                self._bucket_bias[key] = max(0.3, float(value))
            except (TypeError, ValueError):
                continue

    def set_gas_alert_callback(self, callback: Optional[Callable[[str, float], None]]) -> None:
        self._gas_alert_cb = callback

    def snapshot(self) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for state in self.routes.values():
            if not state.samples:
                continue
            price = state.samples[-1][1]
            summary = {
                "symbol": state.symbol,
                "price": price,
                "history_points": len(state.samples),
                "last_update": state.last_update,
            }
            if state.last_directive:
                summary["last_directive"] = state.last_directive.to_dict()
            out.append(summary)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_state(self, sample: Dict[str, float]) -> RouteState:
        symbol = sample.get("symbol") or ""
        state = self.routes.get(symbol)
        if state is None:
            base, quote = _split_symbol(symbol)
            state = RouteState(symbol=symbol, base_token=base, quote_token=quote)
            self._prefill_history(state)
            self.routes[symbol] = state

        ts = float(sample.get("ts") or time.time())
        price = float(sample.get("price") or 0.0)
        volume = float(sample.get("volume") or 0.0)
        signature = (ts, price)
        if state.last_signature == signature:
            return state
        state.samples.append((ts, price, volume))
        state.last_update = ts
        state.last_signature = signature
        state.cached_signals = None
        state.forecast_signature = None
        self._trim_history(state)
        self._resolve_predictions(state)
        return state

    def _prefill_history(self, state: RouteState) -> None:
        try:
            rows = self.db.fetch_market_samples_for(state.symbol, limit=720)
        except Exception:
            rows = []
        if not rows:
            return
        for row in reversed(rows):
            ts = float(row.get("ts") or time.time())
            price = float(row.get("price") or 0.0)
            volume = float(row.get("volume") or 0.0)
            if price <= 0:
                continue
            state.samples.append((ts, price, volume))
        if state.samples:
            last_ts, last_price, _ = state.samples[-1]
            state.last_signature = (last_ts, last_price)

    def _trim_history(self, state: RouteState) -> None:
        if not state.samples:
            return
        cutoff = state.samples[-1][0] - self.history_limit_sec
        while state.samples and state.samples[0][0] < cutoff:
            state.samples.popleft()

    def _forecast(self, state: RouteState) -> List[HorizonSignal]:
        if len(state.samples) < 12:
            return []
        signature_key = (len(state.samples), state.samples[-1][0], state.samples[-1][1])
        if state.cached_signals is not None and state.forecast_signature == signature_key:
            return state.cached_signals
        times = np.array([row[0] for row in state.samples], dtype=float)
        prices = np.array([row[1] for row in state.samples], dtype=float)
        prices = np.nan_to_num(prices, nan=0.0, neginf=0.0, posinf=0.0)
        if np.all(prices <= 0):
            return []
        # Use the last 120 points (approx 2 hours if sampled ~1 min)
        window = min(120, len(prices))
        times = times[-window:]
        prices = prices[-window:]
        rel_minutes = (times - times[-1]) / 60.0
        rel_minutes = np.nan_to_num(rel_minutes, nan=0.0)
        safe_prices = np.clip(prices, a_min=1e-9, a_max=1e9)
        log_prices = np.log(safe_prices)
        slope = 0.0
        intercept = log_prices[-1]
        try:
            if not np.allclose(rel_minutes, rel_minutes[0]):
                with np.errstate(all="ignore"):
                    slope, intercept = np.polyfit(rel_minutes, log_prices, 1)
        except Exception:
            slope = 0.0
            intercept = log_prices[-1]
        current_price = prices[-1]
        returns = np.diff(log_prices)
        returns = returns[np.isfinite(returns)]
        vol = float(np.std(returns)) if returns.size > 0 else 0.0
        if not math.isfinite(vol) or vol == 0.0:
            vol = 1e-6

        signals: List[HorizonSignal] = []
        for label, seconds in self.horizons:
            future_minutes = seconds / 60.0
            predicted_log = intercept + slope * future_minutes
            predicted_log = float(np.clip(predicted_log, -20.0, 20.0))
            predicted_price = float(math.exp(predicted_log))
            expected_return = (predicted_price - current_price) / max(current_price, 1e-9)
            expected_return = float(np.clip(expected_return, -5.0, 5.0))
            if not math.isfinite(expected_return):
                expected_return = 0.0
            # rough z-score relative to observed volatility scaled by horizon
            denom = max(abs(rel_minutes[0]), 1e-6)
            horizon_scale = max(future_minutes / denom, 1.0)
            try:
                horizon_vol = vol * math.sqrt(horizon_scale)
            except ValueError:
                horizon_vol = vol
            horizon_vol = max(horizon_vol, 1e-6)
            zscore = expected_return / horizon_vol
            signals.append(
                HorizonSignal(
                    label=label,
                    seconds=seconds,
                    predicted_price=predicted_price,
                    expected_return=expected_return,
                    zscore=zscore,
                    historical_mae=self.accuracy.mae(label),
                )
            )
        self._queue_predictions(state, signals)
        state.cached_signals = signals
        state.forecast_signature = signature_key
        return self._apply_opportunity_bias(state.symbol, signals)

    def _queue_predictions(self, state: RouteState, signals: List[HorizonSignal]) -> None:
        if not signals or not state.samples:
            return
        current_ts, current_price, _ = state.samples[-1]
        for signal in signals:
            state.pending_predictions.append(
                {
                    "label": signal.label,
                    "resolve_ts": current_ts + signal.seconds,
                    "predicted_return": signal.expected_return,
                    "start_price": current_price,
                }
            )

    def _resolve_predictions(self, state: RouteState) -> None:
        if not state.pending_predictions or not state.samples:
            return
        current_ts = state.samples[-1][0]
        while state.pending_predictions and state.pending_predictions[0]["resolve_ts"] <= current_ts:
            entry = state.pending_predictions.popleft()
            actual = self._realized_return(state, entry["resolve_ts"], entry["start_price"])
            if actual is None:
                continue
            self.accuracy.record(entry["label"], entry["predicted_return"], actual)

    def _realized_return(self, state: RouteState, target_ts: float, start_price: float) -> Optional[float]:
        if start_price <= 0:
            return None
        price = self._price_near_timestamp(state, target_ts)
        if price is None or price <= 0:
            return None
        return (price - start_price) / max(start_price, 1e-9)

    def _price_near_timestamp(self, state: RouteState, target_ts: float) -> Optional[float]:
        if not state.samples:
            return None
        best_price: Optional[float] = None
        best_delta: Optional[float] = None
        for ts, price, _ in reversed(state.samples):
            delta = abs(ts - target_ts)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_price = price
            if ts < target_ts and best_delta is not None and ts < target_ts:
                break
        return best_price

    def record_opportunity(self, signal: OpportunitySignal) -> None:
        self._opportunity_bias[signal.symbol] = signal

    def _score_signal(self, signal: HorizonSignal) -> float:
        return signal.expected_return * self.accuracy.quality(signal.label)

    def _bucket_for_seconds(self, seconds: int) -> str:
        if seconds <= 30 * 60:
            return "short"
        if seconds <= 24 * 3600:
            return "mid"
        return "long"

    def _horizon_weight(self, label: str, seconds: int) -> float:
        span = max(self._horizon_span, 1)
        position = (seconds - self._horizon_min) / span
        position = float(np.clip(position, 0.0, 1.0))
        bell = 0.75 + 0.25 * (1.0 - abs(0.5 - position) * 2.0)
        samples = self.accuracy.count(label)
        scarcity = 1.0 - min(0.6, samples / 256.0)
        bucket = self._bucket_for_seconds(seconds)
        bias = self._bucket_bias.get(bucket, 1.0)
        return float(max(0.4, (bell + 0.2 * scarcity) * bias))

    def _emit_gas_alert(self, chain: str, native_balance: float) -> None:
        if not self._gas_alert_cb:
            return
        now = time.time()
        if now - self._last_gas_alert_ts < self._gas_alert_interval:
            return
        self._last_gas_alert_ts = now
        try:
            self._gas_alert_cb(chain, native_balance)
        except Exception:
            pass

    def _apply_opportunity_bias(self, symbol: str, signals: List[HorizonSignal]) -> List[HorizonSignal]:
        if not signals:
            return signals
        opportunity = self._opportunity_bias.get(symbol)
        if not opportunity:
            return signals
        if time.time() - opportunity.timestamp > 1800:
            self._opportunity_bias.pop(symbol, None)
            return signals
        direction = 1.0 if opportunity.kind == "buy-low" else -1.0
        bias_strength = min(0.03, abs(opportunity.zscore) * 0.005)
        for signal in signals:
            horizon_scale = min(1.0, signal.seconds / (12 * 3600))  # emphasize <= 12h windows
            signal.expected_return += direction * bias_strength * (0.5 + 0.5 * horizon_scale)
        return signals

    def filter_metrics(self) -> List[Dict[str, Any]]:
        """
        Expose recent filter decisions (spread/depth) for telemetry.
        """
        metrics = list(self._filter_metrics)
        self._filter_metrics.clear()
        if metrics:
            try:
                self._metrics_collector.record(
                    MetricStage.LIVE_TRADING,
                    {"count": float(len(metrics))},
                    category="scheduler_filters",
                    meta={"items": metrics[:32]},
                )
            except Exception:
                pass
        return metrics
