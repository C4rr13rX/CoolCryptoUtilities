"""Stochastic oscillator reversal strategy.

Lane's stochastic: %K measures where price sits in the recent range, %D is
its smoothing. Enter on a %K/%D bullish cross inside the oversold zone; exit
a held position on a bearish cross in the overbought zone. Fires frequently
and resolves quickly — designed to accumulate graduation trades fast.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


def _stochastic(prices: np.ndarray, period: int, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    """(%K, %D) series over the trailing window; empty when degenerate."""
    if prices.size < period + smooth + 1:
        empty = np.empty(0)
        return empty, empty
    n = prices.size - period + 1
    k = np.empty(n)
    for i in range(n):
        win = prices[i:i + period]
        lo, hi = float(np.min(win)), float(np.max(win))
        k[i] = 50.0 if hi - lo <= 1e-15 else (float(win[-1]) - lo) / (hi - lo) * 100.0
    d = np.convolve(k, np.ones(smooth) / smooth, mode="valid")
    return k[smooth - 1:], d


class StochasticReversalStrategy(Strategy):
    strategy_id = "stochastic_reversal"
    default_horizon = "20m"
    min_samples = 30

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        period = int(env_float("STOCH_PERIOD", 14, lo=5, hi=50))
        smooth = int(env_float("STOCH_SMOOTH", 3, lo=2, hi=10))
        oversold = env_float("STOCH_OVERSOLD", 20.0, lo=5.0, hi=40.0)
        overbought = env_float("STOCH_OVERBOUGHT", 80.0, lo=60.0, hi=95.0)
        min_net = env_float("STOCH_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, _ = sample_arrays(state)
        if prices.size < max(self.min_samples, period + smooth + 2) or ctx.last_price <= 0:
            return None
        k, d = _stochastic(prices, period, smooth)
        if k.size < 2 or d.size < 2:
            return None
        k_now, k_prev = float(k[-1]), float(k[-2])
        d_now, d_prev = float(d[-1]), float(d[-2])

        # Bullish cross in the oversold zone.
        if k_prev <= d_prev and k_now > d_now and k_now < oversold + 10.0 and ctx.available_quote > 0:
            mid = float(np.median(prices[-period * 2:]))
            expected = (mid - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            depth = max(0.0, oversold - min(k_now, k_prev))
            confidence = min(0.85, 0.55 + depth / 100.0 + 0.05)
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=mid,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"stochastic bullish cross %K {k_now:.0f} oversold",
                extra_meta={"k": k_now, "d": d_now},
            )

        # Bearish cross in the overbought zone with a position: harvest.
        if k_prev >= d_prev and k_now < d_now and k_now > overbought - 10.0 and ctx.available_base > 0:
            mid = float(np.median(prices[-period * 2:]))
            expected = (ctx.last_price - mid) / max(ctx.last_price, 1e-12)
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + max(0.0, k_now - overbought) / 100.0 + 0.05)
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"stochastic bearish cross %K {k_now:.0f} overbought, harvesting {expected:.2%}",
                extra_meta={"k": k_now, "d": d_now},
            )
        return None
