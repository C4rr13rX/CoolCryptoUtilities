"""On-balance volume accumulation strategy.

Granville's OBV: cumulative signed volume leads price. Enter when OBV makes a
sustained higher trend while price is still flat/lower (institutional
accumulation); exit a held position when OBV rolls over while price is still
elevated (distribution). Orthogonal to every price-only strategy — it reads
volume flow, so its ledger wins are independent evidence.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class ObvAccumulationStrategy(Strategy):
    strategy_id = "obv_accumulation"
    default_horizon = "45m"
    min_samples = 36

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        window = int(env_float("OBV_WINDOW", 24, lo=12, hi=96))
        div_floor = env_float("OBV_DIVERGENCE_FLOOR", 0.15, lo=0.02, hi=1.0)
        min_net = env_float("OBV_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, volumes = sample_arrays(state)
        if prices.size < max(self.min_samples, window + 2) or ctx.last_price <= 0:
            return None
        if volumes.size != prices.size or float(np.sum(volumes[-window:])) <= 0:
            return None
        deltas = np.diff(prices)
        obv = np.cumsum(np.sign(deltas) * volumes[1:])
        if obv.size < window:
            return None
        seg = obv[-window:]
        seg_range = float(np.max(seg) - np.min(seg))
        if seg_range <= 1e-15:
            return None
        # Normalized slopes over the window: OBV vs price, both in [-1, 1]-ish.
        obv_slope = float(seg[-1] - seg[0]) / seg_range
        p_seg = prices[-window:]
        p_range = float(np.max(p_seg) - np.min(p_seg))
        price_slope = 0.0 if p_range <= 1e-15 else float(p_seg[-1] - p_seg[0]) / p_range

        divergence = obv_slope - price_slope

        # Accumulation: volume flowing in while price hasn't moved yet.
        if divergence >= div_floor and obv_slope > 0 and ctx.available_quote > 0:
            recent_high = float(np.max(p_seg))
            expected = (recent_high - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + min(0.25, divergence * 0.4))
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=recent_high,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"OBV accumulation divergence {divergence:+.2f}",
                extra_meta={"obv_slope": obv_slope, "price_slope": price_slope},
            )

        # Distribution: volume flowing out while price is still elevated.
        if divergence <= -div_floor and obv_slope < 0 and ctx.available_base > 0:
            recent_low = float(np.min(p_seg))
            expected = (ctx.last_price - recent_low) / max(ctx.last_price, 1e-12) * 0.5
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.55 + min(0.25, abs(divergence) * 0.4))
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"OBV distribution divergence {divergence:+.2f}, harvesting",
                extra_meta={"obv_slope": obv_slope, "price_slope": price_slope},
            )
        return None
