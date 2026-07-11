"""SuperTrend-style ATR band trend-follow strategy.

Volatility-banded trend detection: price crossing above the upper ATR band
around a rolling median flips the regime bullish (enter); crossing below the
lower band flips it bearish (exit a held position). Samples are ticks (no
OHLC), so ATR is proxied by the rolling mean absolute price change — same
robustness, tick-friendly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class SupertrendFollowStrategy(Strategy):
    strategy_id = "supertrend_follow"
    default_horizon = "45m"
    min_samples = 36

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        period = int(env_float("SUPERTREND_PERIOD", 14, lo=6, hi=60))
        mult = env_float("SUPERTREND_MULT", 2.0, lo=1.0, hi=5.0)
        min_net = env_float("SUPERTREND_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, _ = sample_arrays(state)
        if prices.size < max(self.min_samples, period * 2 + 2) or ctx.last_price <= 0:
            return None
        deltas = np.abs(np.diff(prices[-(period + 1):]))
        atr = float(np.mean(deltas))
        mid = float(np.median(prices[-period:]))
        if atr <= 0 or mid <= 0:
            return None
        upper = mid + mult * atr
        lower = mid - mult * atr
        prev = float(prices[-2])

        # Bullish flip: price crosses above the upper band (fresh, not chased).
        if prev <= upper < ctx.last_price and ctx.available_quote > 0:
            target = ctx.last_price + mult * atr  # ride one more band-width
            expected = (target - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            strength = (ctx.last_price - upper) / max(atr, 1e-12)
            confidence = min(0.85, 0.55 + min(0.25, strength * 0.1) + min(0.1, atr / mid * 10.0))
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=target,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"SuperTrend bullish flip above band ({atr / mid:.3%} ATR)",
                extra_meta={"atr": atr, "upper_band": upper, "lower_band": lower},
            )

        # Bearish flip with a position: exit the trend.
        if prev >= lower > ctx.last_price and ctx.available_base > 0:
            expected = mult * atr / max(ctx.last_price, 1e-12) * 0.5
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + min(0.25, (lower - ctx.last_price) / max(atr, 1e-12) * 0.1))
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"SuperTrend bearish flip below band, exiting",
                extra_meta={"atr": atr, "lower_band": lower},
            )
        return None
