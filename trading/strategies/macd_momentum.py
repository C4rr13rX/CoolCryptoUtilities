"""MACD histogram momentum strategy.

The classic Appel MACD: enter when the histogram (MACD - signal) flips
positive while still below the zero line — momentum inflecting up before the
crowd sees the crossover — and exit a held position on the bearish flip.
Proven, fast-firing, and complements ema_cross (which needs the full cross).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, ema, env_float, sample_arrays


class MacdMomentumStrategy(Strategy):
    strategy_id = "macd_momentum"
    default_horizon = "30m"
    min_samples = 40

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        fast = int(env_float("MACD_FAST", 12, lo=4, hi=48))
        slow = int(env_float("MACD_SLOW", 26, lo=8, hi=96))
        sig_span = int(env_float("MACD_SIGNAL", 9, lo=3, hi=32))
        min_net = env_float("MACD_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, _ = sample_arrays(state)
        if prices.size < max(self.min_samples, slow + sig_span) or ctx.last_price <= 0:
            return None
        macd = ema(prices, fast) - ema(prices, slow)
        signal = ema(macd, sig_span)
        hist = macd - signal
        if hist.size < 3:
            return None
        h_now, h_prev = float(hist[-1]), float(hist[-2])
        # Normalize by price so thresholds are scale-free.
        rel = h_now / max(ctx.last_price, 1e-12)

        # Bullish inflection: histogram flips positive while MACD still below zero.
        if h_prev <= 0.0 < h_now and float(macd[-1]) < 0.0 and ctx.available_quote > 0:
            # Expected reversion toward the slow EMA (trend equilibrium).
            slow_ema = float(ema(prices, slow)[-1])
            expected = (slow_ema - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + min(0.25, abs(rel) * 400.0))
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=slow_ema,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"MACD histogram bullish flip below zero ({rel:+.4%})",
                extra_meta={"macd": float(macd[-1]), "histogram": h_now},
            )

        # Bearish inflection with position: harvest.
        if h_prev >= 0.0 > h_now and ctx.available_base > 0:
            recent_low = float(np.min(prices[-slow:]))
            expected = (ctx.last_price - recent_low) / max(ctx.last_price, 1e-12) * 0.5
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + min(0.25, abs(rel) * 400.0))
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"MACD histogram bearish flip, harvesting ({rel:+.4%})",
                extra_meta={"macd": float(macd[-1]), "histogram": h_now},
            )
        return None
