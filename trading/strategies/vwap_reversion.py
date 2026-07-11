"""VWAP reversion strategy.

Price trading materially below rolling VWAP enters targeting VWAP; price
stretched above VWAP exits a held position.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class VwapReversionStrategy(Strategy):
    strategy_id = "vwap_reversion"
    default_horizon = "30m"
    min_samples = 20

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        lookback = env_float("VWAP_LOOKBACK_SEC", 2 * 3600.0, lo=900.0, hi=12 * 3600.0)
        dev_entry = env_float("VWAP_DEV_ENTRY", 0.01, lo=0.002, hi=0.10)
        min_net = env_float("VWAP_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)
        # A real reversion is a few %; anything past this is a corrupt window
        # (e.g. mixed price denominations across exchanges) — never a signal.
        max_dev = env_float("VWAP_MAX_DEV", 0.25, lo=0.05, hi=1.0)

        _, prices, volumes = sample_arrays(state, lookback)
        if prices.size < self.min_samples or ctx.last_price <= 0:
            return None
        total_vol = float(np.sum(volumes))
        if total_vol <= 0:
            return None
        vwap = float(np.dot(prices, volumes) / total_vol)
        if vwap <= 0:
            return None
        dev = (ctx.last_price - vwap) / vwap
        # Reject implausible deviations: the price series is contaminated
        # (unit mismatch / bad tick), not a tradeable dislocation.
        if abs(dev) > max_dev:
            return None

        if dev <= -dev_entry and ctx.available_quote > 0:
            expected = (vwap - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.5 + abs(dev) * 10.0)
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=vwap,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"{dev:.2%} below VWAP, reversion target {expected:.2%}",
                extra_meta={"vwap": vwap, "deviation": dev},
            )

        if dev >= dev_entry and ctx.available_base > 0:
            expected = (ctx.last_price - vwap) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.5 + abs(dev) * 10.0)
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"{dev:.2%} above VWAP, harvesting {expected:.2%}",
                extra_meta={"vwap": vwap, "deviation": dev},
            )
        return None
