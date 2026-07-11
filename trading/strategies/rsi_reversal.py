"""RSI reversal strategy.

Oversold RSI enters (with a bullish-divergence confidence boost); overbought
RSI exits a held position.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, rsi, sample_arrays


class RsiReversalStrategy(Strategy):
    strategy_id = "rsi_reversal"
    default_horizon = "45m"
    min_samples = 32

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        period = int(env_float("RSI_PERIOD", 14, lo=5, hi=50))
        oversold = env_float("RSI_OVERSOLD", 30.0, lo=5.0, hi=45.0)
        overbought = env_float("RSI_OVERBOUGHT", 70.0, lo=55.0, hi=95.0)
        min_net = env_float("RSI_MIN_NET_RETURN", 0.004, lo=0.0, hi=0.1)

        _, prices, _ = sample_arrays(state)
        if prices.size < max(self.min_samples, 2 * (period + 1)) or ctx.last_price <= 0:
            return None
        value = rsi(prices, period)
        mean = float(np.mean(prices[-4 * period:]))
        if mean <= 0:
            return None

        if value <= oversold and ctx.available_quote > 0:
            expected = (mean - ctx.last_price) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            # Bullish divergence: price lower low vs prior window while RSI improved.
            prior = prices[:-(period + 1)]
            divergence = False
            if prior.size >= period + 1:
                prior_rsi = rsi(prior, period)
                if ctx.last_price < float(np.min(prior[-period:])) and value > prior_rsi:
                    divergence = True
            confidence = min(0.85, 0.5 + (oversold - value) / 100.0 + (0.15 if divergence else 0.0))
            return self.make_candidate(
                state, ctx,
                action="enter",
                expected_return=expected,
                target_price=mean,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"RSI {value:.0f} oversold{' + divergence' if divergence else ''}",
                extra_meta={"rsi": value, "divergence": divergence},
            )

        if value >= overbought and ctx.available_base > 0:
            expected = (ctx.last_price - mean) / ctx.last_price
            if expected - ctx.fee_rate < min_net:
                return None
            confidence = min(0.85, 0.5 + (value - overbought) / 100.0)
            return self.make_candidate(
                state, ctx,
                action="exit",
                expected_return=expected,
                target_price=ctx.last_price,
                confidence=confidence,
                direction_prob=confidence,
                reason=f"RSI {value:.0f} overbought, harvesting {expected:.2%}",
                extra_meta={"rsi": value},
            )
        return None
