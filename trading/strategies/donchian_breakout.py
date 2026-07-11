"""Donchian channel breakout strategy (turtle trading).

Dennis/Eckhardt's turtle rule: enter when price breaks the N-bar high
(strength begets strength), exit a held position when price falls back
through the channel midline. One of the most durably profitable systematic
rules ever published, and cheap to compute.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class DonchianBreakoutStrategy(Strategy):
    strategy_id = "donchian_breakout"
    default_horizon = "1h"
    min_samples = 40

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        channel = int(env_float("DONCHIAN_CHANNEL", 20, lo=10, hi=96))
        min_net = env_float("DONCHIAN_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, volumes = sample_arrays(state)
        if prices.size < max(self.min_samples, channel + 4) or ctx.last_price <= 0:
            return None
        # Channel excludes the newest bar so the breakout compares against
        # established structure, not itself.
        window = prices[-(channel + 1):-1]
        hi = float(np.max(window))
        lo = float(np.min(window))
        mid = (hi + lo) / 2.0
        if hi <= 0 or hi - lo <= 1e-15:
            return None
        width = (hi - lo) / hi

        # Breakout above the N-bar high, with modest volume confirmation.
        if ctx.last_price > hi and ctx.available_quote > 0:
            vol_ok = True
            if volumes.size >= channel + 1:
                recent_vol = float(np.mean(volumes[-3:]))
                base_vol = float(np.mean(volumes[-(channel + 1):-1]))
                vol_ok = base_vol <= 0 or recent_vol >= base_vol * 0.8
            if not vol_ok:
                return None
            # Measured-move target: channel width projected from the breakout.
            target = ctx.last_price * (1.0 + width * 0.5)
            expected = (target - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + min(0.2, width * 2.0))
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=target,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"Donchian {channel}-bar breakout, channel {width:.2%}",
                extra_meta={"channel_high": hi, "channel_low": lo, "width": width},
            )

        # Turtle exit: price back through the channel midline with a position.
        if ctx.last_price < mid and ctx.available_base > 0:
            expected = (ctx.last_price - lo) / max(ctx.last_price, 1e-12) * 0.5
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.8, 0.55 + (mid - ctx.last_price) / max(mid, 1e-12))
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"price below Donchian midline, exiting trend",
                extra_meta={"channel_mid": mid},
            )
        return None
