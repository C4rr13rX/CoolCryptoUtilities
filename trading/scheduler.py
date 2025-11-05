from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from db import get_db, TradingDatabase
from trading.data_stream import _split_symbol
from trading.portfolio import PortfolioState

HORIZON_DEFAULTS: List[Tuple[str, int]] = [
    ("5m", 5 * 60),
    ("20m", 20 * 60),
    ("30m", 30 * 60),
    ("1h", 60 * 60),
    ("5h", 5 * 60 * 60),
    ("10h", 10 * 60 * 60),
    ("1d", 24 * 60 * 60),
    ("3d", 3 * 24 * 60 * 60),
    ("5d", 5 * 24 * 60 * 60),
    ("1w", 7 * 24 * 60 * 60),
]


@dataclass
class HorizonSignal:
    label: str
    seconds: int
    predicted_price: float
    expected_return: float
    zscore: float


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        sample: Dict[str, float],
        pred_summary: Dict[str, float],
        portfolio: PortfolioState,
    ) -> Optional[TradeDirective]:
        """Update internal state with the latest sample and return a directive."""
        state = self._update_state(sample)
        signals = self._forecast(state)
        if not signals:
            return None

        last_price = state.samples[-1][1]
        direction_prob = float(pred_summary.get("direction_prob", 0.5))
        confidence = float(pred_summary.get("exit_conf", 0.5))
        net_margin = float(pred_summary.get("net_margin", 0.0))

        available_quote = portfolio.get_quantity(state.quote_token)
        available_base = portfolio.get_quantity(state.base_token)
        native_balance = portfolio.get_native_balance("ethereum")
        # crude gas safety
        if native_balance < 0.01:
            return None

        best_long = max(signals, key=lambda s: s.expected_return)
        best_short = min(signals, key=lambda s: s.expected_return)

        fee_rate = self.fee_buffer + self.tax_buffer
        # Enter: use quote asset to buy base
        if available_quote > 0:
            expected = best_long.expected_return
            if (
                expected > (self.min_profit + fee_rate)
                and direction_prob >= 0.52
                and (net_margin >= 0 or confidence >= 0.55)
            ):
                size_quote = min(available_quote * 0.25, max(available_quote * 0.1, 0.0))
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
                    state.last_directive = directive
                    return directive

        # Exit: sell base into quote when projected drawdown
        if available_base > 0:
            expected = best_short.expected_return
            if expected < -(self.min_profit / 2.0) or direction_prob <= 0.45 or net_margin < 0:
                size_base = available_base * 0.4
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
                    state.last_directive = directive
                    return directive

        return None

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
        self._trim_history(state)
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
                )
            )
        return signals
