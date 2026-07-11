"""Volume spike (momentum ignition) strategy.

An abnormal volume burst with a directional price move tends to continue
short-term: burst up enters, burst down exits a held position.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class VolumeSpikeStrategy(Strategy):
    strategy_id = "volume_spike"
    default_horizon = "20m"
    min_samples = 24

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        spike_mult = env_float("VOLUME_SPIKE_MULT", 3.0, lo=1.5, hi=20.0)
        move_window = int(env_float("VOLUME_SPIKE_MOVE_WINDOW", 5, lo=2, hi=30))
        min_net = env_float("VOLUME_SPIKE_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, volumes = sample_arrays(state)
        if prices.size < self.min_samples or ctx.last_price <= 0 or volumes.size < self.min_samples:
            return None
        baseline = float(np.median(volumes[:-1]))
        if baseline <= 0 or ctx.last_volume < baseline * spike_mult:
            return None
        ref = float(prices[-min(move_window + 1, prices.size)])
        if ref <= 0:
            return None
        move = (ctx.last_price - ref) / ref
        # Continuation expectation: half the ignition move.
        expected = abs(move) * 0.5
        if expected - ctx.fee_rate < min_net:
            return None
        spike_ratio = ctx.last_volume / (baseline * spike_mult)
        confidence = min(0.8, 0.5 + 0.15 * min(spike_ratio, 2.0) + min(abs(move) * 5.0, 0.15))

        if move > 0 and ctx.available_quote > 0:
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=ctx.last_price * (1.0 + expected),
                confidence=confidence,
                direction_prob=confidence,
                reason=f"volume spike {ctx.last_volume/baseline:.1f}x median with +{move:.2%} move",
                extra_meta={"spike_ratio": spike_ratio, "move": move},
            )
        if move < 0 and ctx.available_base > 0:
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"volume spike {ctx.last_volume/baseline:.1f}x median with {move:.2%} move",
                extra_meta={"spike_ratio": spike_ratio, "move": move},
            )
        return None
