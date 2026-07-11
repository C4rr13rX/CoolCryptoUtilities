"""Momentum breakout strategy.

Enter when price clears the recent high with volume confirmation; exit when
price loses the recent low while holding.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class MomentumBreakoutStrategy(Strategy):
    strategy_id = "momentum_breakout"
    default_horizon = "30m"
    min_samples = 30

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        lookback = env_float("MOMENTUM_BREAKOUT_LOOKBACK_SEC", 3600.0, lo=900.0, hi=8 * 3600.0)
        buffer_pct = env_float("MOMENTUM_BREAKOUT_BUFFER", 0.002, lo=0.0, hi=0.05)
        vol_mult = env_float("MOMENTUM_BREAKOUT_VOL_MULT", 1.5, lo=1.0, hi=10.0)
        min_net = env_float("MOMENTUM_BREAKOUT_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, volumes = sample_arrays(state, lookback)
        if prices.size < self.min_samples or ctx.last_price <= 0:
            return None
        # Reference window excludes the most recent bars that form the breakout.
        head = prices[:-3]
        if head.size < 10:
            return None
        prev_high = float(np.max(head))
        prev_low = float(np.min(head))
        avg_vol = float(np.mean(volumes[:-3])) if volumes.size > 3 else 0.0
        rng = (prev_high - prev_low) / max(prev_high, 1e-12)

        if (
            ctx.available_quote > 0
            and ctx.last_price > prev_high * (1.0 + buffer_pct)
            and avg_vol > 0
            and ctx.last_volume >= avg_vol * vol_mult
        ):
            # Expect continuation of roughly half the consolidation range.
            expected = max(rng * 0.5, 0.0)
            if expected - ctx.fee_rate < min_net:
                return None
            vol_edge = min(1.0, ctx.last_volume / (avg_vol * vol_mult))
            confidence = min(0.85, 0.5 + 0.2 * vol_edge + 0.1 * min(rng * 10.0, 1.0))
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=ctx.last_price * (1.0 + expected),
                confidence=confidence,
                direction_prob=confidence,
                reason=f"breakout above {prev_high:.6g} on {ctx.last_volume/max(avg_vol,1e-12):.1f}x volume",
                extra_meta={"prev_high": prev_high, "range": rng},
            )

        if ctx.available_base > 0 and ctx.last_price < prev_low * (1.0 - buffer_pct):
            # Breakdown: expected benefit of exiting = avoiding the same continuation down.
            expected = max(rng * 0.5, 0.0)
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.5 + 0.15 * min(rng * 10.0, 1.0))
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"breakdown below {prev_low:.6g}, avoiding ~{expected:.2%} continuation",
                extra_meta={"prev_low": prev_low, "range": rng},
            )
        return None
