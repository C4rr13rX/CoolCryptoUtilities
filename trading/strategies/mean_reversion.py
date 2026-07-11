"""Mean-reversion z-score strategy.

Buy when price sits well below its rolling mean (controlled dip), targeting
reversion to the mean; flag an exit when price stretches equally far above.
Same statistical core as OpportunityTracker but self-contained and sized.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import (
    Strategy,
    StrategyContext,
    env_float,
    log_slope_per_min,
    sample_arrays,
)


class MeanReversionStrategy(Strategy):
    strategy_id = "mean_reversion"
    default_horizon = "45m"
    min_samples = 24

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        lookback = env_float("MEAN_REVERSION_LOOKBACK_SEC", 3600.0, lo=600.0, hi=6 * 3600.0)
        z_entry = env_float("MEAN_REVERSION_Z_ENTRY", 1.8, lo=0.5, hi=4.0)
        min_net = env_float("MEAN_REVERSION_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        ts, prices, _ = sample_arrays(state, lookback)
        if prices.size < self.min_samples or ctx.last_price <= 0:
            return None
        mean = float(np.mean(prices))
        std = float(np.std(prices))
        if std <= 1e-12 or mean <= 0:
            return None
        z = (ctx.last_price - mean) / std

        if z <= -z_entry and ctx.available_quote > 0:
            # Knife guard: recent trend must not be in free fall.
            slope = log_slope_per_min(ts[-max(prices.size // 4, 6):], prices[-max(prices.size // 4, 6):])
            if slope < env_float("MEAN_REVERSION_SLOPE_FLOOR", -0.0004, lo=-0.01, hi=0.0):
                return None
            expected = (mean - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.9, 0.5 + abs(z) * 0.08)
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=mean,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"z={z:.2f} below mean, target reversion {expected:.2%}",
                extra_meta={"zscore": float(z)},
            )

        if z >= z_entry and ctx.available_base > 0:
            # Price stretched above the mean: lock the extension in now.
            expected = (ctx.last_price - mean) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.9, 0.5 + abs(z) * 0.08)
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"z={z:.2f} above mean, harvesting extension {expected:.2%}",
                extra_meta={"zscore": float(z)},
            )
        return None
