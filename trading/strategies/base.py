"""Strategy plugin architecture.

Each strategy is an independent, CPU-only signal generator that inspects a
pair's rolling sample window (``RouteState.samples``) and emits candidate
trade directives in the exact ``{"directive", "score", "meta"}`` shape the
CDCL solver already arbitrates. Strategies never import TensorFlow and never
block: everything is arithmetic over the in-memory deque.

Contract with the CDCL solver (trading/cdcl_solver.py clause DB):
  - ``directive.expected_return`` must be the NET-of-nothing raw edge and must
    exceed ``context["fee_rate"]`` to pass ``return_above_fees`` — for exit
    candidates this is the positive expected benefit of exiting now.
  - ``meta["confidence"]`` and ``meta["direction_prob"]`` must be > 0.
  - ``meta["strategy"]`` carries the strategy_id for the per-strategy ledger.
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def env_float(name: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    try:
        val = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        val = default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return val


def env_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class StrategyContext:
    """Everything a strategy may consult beyond the RouteState window."""

    chain: str
    last_price: float
    last_volume: float
    fee_rate: float
    available_quote: float
    available_base: float
    risk_budget: float = 1.0
    live_trading: bool = False
    # TF model summary — neutral (0.5/0.5/0.0) whenever TF is unavailable.
    direction_prob: float = 0.5
    confidence: float = 0.5
    net_margin: float = 0.0
    opportunity: Optional[Any] = None  # OpportunitySignal or None
    extras: Dict[str, Any] = field(default_factory=dict)


def sample_arrays(
    state: Any,
    lookback_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(timestamps, prices, volumes) from RouteState.samples, oldest first."""
    samples = list(getattr(state, "samples", []) or [])
    if lookback_sec and samples:
        cutoff = samples[-1][0] - float(lookback_sec)
        samples = [s for s in samples if s[0] >= cutoff]
    if not samples:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty
    arr = np.asarray(samples, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def ema(values: np.ndarray, span: int) -> np.ndarray:
    if values.size == 0:
        return values
    alpha = 2.0 / (max(int(span), 1) + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, period: int = 14) -> float:
    """Wilder RSI of the last `period` moves; 50.0 when undefined."""
    if prices.size < period + 1:
        return 50.0
    deltas = np.diff(prices[-(period + 1):])
    gains = np.clip(deltas, 0.0, None)
    losses = np.clip(-deltas, 0.0, None)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss <= 1e-12:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def log_slope_per_min(ts: np.ndarray, prices: np.ndarray) -> float:
    """OLS slope of log-price per minute; 0.0 when degenerate."""
    if prices.size < 4:
        return 0.0
    rel_min = (ts - ts[-1]) / 60.0
    if np.allclose(rel_min, rel_min[0]):
        return 0.0
    safe = np.clip(prices, 1e-12, None)
    try:
        with np.errstate(all="ignore"):
            slope, _ = np.polyfit(rel_min, np.log(safe), 1)
        return float(slope) if math.isfinite(slope) else 0.0
    except Exception:
        return 0.0


class Strategy(ABC):
    """Independent buy-low/sell-high signal generator."""

    strategy_id: str = "base"
    default_horizon: str = "30m"
    #: minimum samples in the window before this strategy will evaluate
    min_samples: int = 16

    def enabled(self) -> bool:
        return env_flag(f"STRATEGY_{self.strategy_id.upper()}_ENABLED", "1")

    @abstractmethod
    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        """Return a candidate dict or None. Must not raise for bad data."""

    # ------------------------------------------------------------------
    # Candidate builders
    # ------------------------------------------------------------------

    def _size_enter(self, ctx: StrategyContext, edge: float) -> float:
        """Quote-denominated size: modest, edge-scaled fraction of quote."""
        min_frac = env_float("STRATEGY_MIN_QUOTE_FRAC", 0.03, lo=0.0, hi=0.5)
        max_frac = env_float("STRATEGY_MAX_QUOTE_FRAC", 0.12, lo=0.01, hi=0.5)
        frac = min(max_frac, max(min_frac, 0.04 + min(0.08, max(edge, 0.0) * 2.0)))
        return ctx.available_quote * frac * max(0.0, min(1.0, ctx.risk_budget))

    def _size_exit(self, ctx: StrategyContext, confidence: float) -> float:
        base_frac = env_float("STRATEGY_EXIT_BASE_FRAC", 0.5, lo=0.1, hi=1.0)
        frac = min(1.0, base_frac + 0.5 * max(0.0, confidence - 0.5))
        return ctx.available_base * frac

    def make_candidate(
        self,
        state: Any,
        ctx: StrategyContext,
        *,
        action: str,
        expected_return: float,
        target_price: float,
        confidence: float,
        reason: str,
        direction_prob: Optional[float] = None,
        horizon: Optional[str] = None,
        quote_size: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build a sized `{"directive", "score", "meta"}` candidate.

        ``expected_return`` is the net edge the strategy expects to capture by
        taking `action` NOW — always positive when the trade is worth taking
        (the CDCL ``return_above_fees`` clause rejects anything <= fee_rate).
        """
        if not (math.isfinite(expected_return) and math.isfinite(target_price)):
            return None
        if target_price <= 0 or expected_return <= 0:
            return None
        confidence = max(0.01, min(1.0, float(confidence)))
        if action == "enter":
            requested_quote = ctx.available_quote if quote_size is None else float(quote_size)
            if requested_quote <= 0 or ctx.last_price <= 0:
                return None
            # Only enter USD-stable-quoted pairs. Base/base pairs (JITOSOL-CBBTC,
            # WBTC-WETH, AERO-WETH ...) carry their 'price' as a token ratio, not
            # USD, so every strategy's PnL/RSI/VWAP math is poisoned — producing
            # dust enters that loop, fantasy profits, and RSI-0 garbage. Exits are
            # still allowed so any existing non-USD position can be closed.
            if env_flag("STRATEGY_STABLE_QUOTE_ONLY", "1"):
                _stable = {"USDC", "USDT", "DAI", "USDBC", "USDC.E", "BUSD", "EURC", "USD", "TUSD", "FDUSD"}
                if str(getattr(state, "quote_token", "")).upper() not in _stable:
                    return None
                # Stable-stable pairs (USDT-USDC, DAI-USDC ...) never move more
                # than fees; entering them just churns near-zero "drawdown"
                # losses that dilute win rates and block graduation.
                if str(getattr(state, "base_token", "")).upper() in _stable:
                    return None
            size_quote = (
                self._size_enter(ctx, expected_return - ctx.fee_rate)
                if quote_size is None else requested_quote
            )
            size = size_quote / max(ctx.last_price, 1e-12)
        elif action == "exit":
            if ctx.available_base <= 0:
                return None
            size = self._size_exit(ctx, confidence)
        else:
            return None
        if size <= 0 or not math.isfinite(size):
            return None

        from trading.scheduler import TradeDirective  # local: avoids import cycle

        directive = TradeDirective(
            action=action,
            symbol=state.symbol,
            base_token=state.base_token,
            quote_token=state.quote_token,
            size=float(size),
            target_price=float(target_price),
            horizon=horizon or self.default_horizon,
            confidence=confidence,
            expected_return=float(expected_return),
            reason=f"{self.strategy_id}: {reason}",
            strategy_id=self.strategy_id,
        )
        meta: Dict[str, Any] = {
            "strategy": self.strategy_id,
            "confidence": confidence,
            "direction_prob": float(direction_prob if direction_prob is not None else max(0.5, confidence)),
            "risk_penalty": float(ctx.fee_rate),
            "horizon_weight": 1.0,
            "quality": 1.0,
        }
        if extra_meta:
            meta.update(extra_meta)
        return {
            "directive": directive,
            "score": float(expected_return - ctx.fee_rate),
            "meta": meta,
        }


class StrategyRegistry:
    """Holds strategy instances and fans evaluation out across them.

    One registry per BusScheduler; strategies that keep per-symbol state must
    key it by ``state.symbol`` because a scheduler tracks multiple routes.
    """

    def __init__(self, strategies: Optional[Sequence[Strategy]] = None) -> None:
        self._strategies: Dict[str, Strategy] = {}
        for strat in strategies or []:
            self.register(strat)

    def register(self, strategy: Strategy) -> None:
        self._strategies[strategy.strategy_id] = strategy

    def get(self, strategy_id: str) -> Optional[Strategy]:
        return self._strategies.get(strategy_id)

    def all(self) -> List[Strategy]:
        return list(self._strategies.values())

    def ids(self) -> List[str]:
        return list(self._strategies.keys())

    def evaluate_all(self, state: Any, ctx: StrategyContext) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        n_samples = len(getattr(state, "samples", []) or [])
        for strat in self._strategies.values():
            if n_samples < strat.min_samples:
                continue
            try:
                if not strat.enabled():
                    continue
                cand = strat.evaluate(state, ctx)
            except Exception:
                continue
            if cand:
                candidates.append(cand)
        return candidates
