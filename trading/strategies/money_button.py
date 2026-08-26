"""The Money Button — a 5-10 minute lane that only fires when the move clears costs.

Every other strategy in this registry resolves over hours to a week. That is
why the ghost ledger fills slowly: evidence accrues at the speed of the slowest
horizon, and a strategy needs 20 closed trades before it can graduate. This
lane exists to shorten that loop by trading the shortest horizon the feed can
actually support.

The danger of a fast lane is that it is trivially easy to build one that trades
constantly and proves nothing. A round trip costs ``2 * fee_rate`` plus
slippage, so a 5-minute strategy that ignores costs will post a long string of
small, confident, losing trades and pollute the very ledger it was meant to
fill. Measured on this feed, only ~2% of observed 5-minute moves clear a 0.6%
round trip.

So the gate here is deliberately strict and cost-first:

  * the projected move must clear the full round-trip cost with margin, using
    a *conservative* estimate of the move rather than the raw one;
  * momentum must be confirmed by an independent, longer window, so a single
    noisy print cannot trigger an entry;
  * the price must be moving on real volume;
  * and the pair must be genuinely liquid enough for the sample window to be
    trustworthy.

A strategy that declines to trade is doing its job. Firing rarely and being
right is what earns graduation; firing often is what destroyed the previous
ledger.
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


class MoneyButtonStrategy(Strategy):
    """Short-horizon momentum continuation, gated hard on round-trip cost."""

    strategy_id = "money_button"
    default_horizon = "10m"
    #: 5-minute decisions off fewer than ~20 prints are noise, not signal.
    min_samples = 20

    #: Window actually inspected. Wider than the horizon on purpose: the
    #: entry is short, but the evidence for it should not be.
    LOOKBACK_SEC = 45.0 * 60.0

    @staticmethod
    def _return_over(ts: np.ndarray, prices: np.ndarray, seconds: float) -> float:
        """Fractional price change over the trailing `seconds`."""
        if prices.size < 2:
            return 0.0
        cutoff = float(ts[-1]) - seconds
        idx = int(np.searchsorted(ts, cutoff, side="left"))
        idx = min(max(idx, 0), prices.size - 1)
        anchor = float(prices[idx])
        if anchor <= 0:
            return 0.0
        return float(prices[-1]) / anchor - 1.0

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        if ctx.last_price <= 0:
            return None

        ts, prices, volumes = sample_arrays(state, self.LOOKBACK_SEC)
        if prices.size < self.min_samples:
            return None

        # Require real elapsed time. A burst of prints inside one minute can
        # satisfy min_samples while telling us nothing about a 10m horizon.
        span_sec = float(ts[-1] - ts[0])
        if span_sec < 12.0 * 60.0:
            return None

        # ------------------------------------------------------------------
        # Reject frozen and near-frozen feeds outright.
        #
        # Most symbols here are Base-chain DEX tokens; when a feed cannot
        # price one it repeats a seed value forever. A strategy reading that
        # sees a perfectly stable price and infers a risk-free entry. Every
        # such trade exits at exactly its entry and drags the win rate toward
        # zero, which is precisely how 89 of 205 ghost exits landed flat.
        # ------------------------------------------------------------------
        distinct = int(np.unique(prices).size)
        if distinct < max(6, prices.size // 4):
            return None

        # ------------------------------------------------------------------
        # Cost floor. This is the whole point of the strategy.
        # ------------------------------------------------------------------
        slippage = env_float("MONEY_BUTTON_SLIPPAGE", 0.001, lo=0.0, hi=0.05)
        # Both legs pay the fee; a "profitable" one-way move that cannot pay
        # for its own exit is a loss that has not been realised yet.
        round_trip_cost = 2.0 * float(ctx.fee_rate) + slippage
        # Margin is small on purpose. The projected move is already discounted
        # twice below (momentum decay, then a volatility penalty), so stacking
        # a large margin on top means requiring roughly 3x the round trip in
        # raw movement -- which rejects a clean 6%-in-40min trend and leaves
        # the lane unable to produce the evidence it exists to produce.
        margin = env_float("MONEY_BUTTON_MARGIN", 1.1, lo=1.0, hi=5.0)
        required_edge = round_trip_cost * margin

        # ------------------------------------------------------------------
        # Momentum, confirmed across independent windows.
        # ------------------------------------------------------------------
        r5 = self._return_over(ts, prices, 5.0 * 60.0)
        r10 = self._return_over(ts, prices, 10.0 * 60.0)
        r30 = self._return_over(ts, prices, 30.0 * 60.0)

        # Direction must agree across timescales. Requiring the 30m to also be
        # up is what separates a trend from a dead-cat bounce inside a slide.
        if r5 <= 0.0 or r10 <= 0.0 or r30 <= 0.0:
            return None
        # The recent leg must lead, otherwise the move is already exhausted
        # and we would be buying the top of it.
        if r5 < r10 * 0.35:
            return None

        slope_per_min = log_slope_per_min(ts, prices)
        if slope_per_min <= 0.0:
            return None

        # ------------------------------------------------------------------
        # Project the move over the holding period, then discount it.
        # ------------------------------------------------------------------
        hold_minutes = env_float("MONEY_BUTTON_HOLD_MIN", 8.0, lo=5.0, hi=10.0)
        projected = float(np.expm1(slope_per_min * hold_minutes))
        if projected <= 0.0:
            return None

        # Momentum decays; assuming the trailing slope simply continues is the
        # standard way a fast strategy fools itself. Halve it by default.
        decay = env_float("MONEY_BUTTON_DECAY", 0.5, lo=0.1, hi=1.0)
        step_returns = np.diff(np.log(np.clip(prices, 1e-12, None)))
        volatility = float(np.std(step_returns)) if step_returns.size else 0.0
        # Noisy series get discounted further: dispersion here is as likely to
        # be against us as with us.
        conservative_edge = max(0.0, projected * decay - 0.5 * volatility)

        if conservative_edge < required_edge:
            return None

        # ------------------------------------------------------------------
        # Liquidity: the move must be carried by volume, not by one print.
        # ------------------------------------------------------------------
        if volumes.size >= 6:
            recent_vol = float(np.mean(volumes[-3:]))
            baseline_vol = float(np.median(volumes[:-3]))
            if baseline_vol > 0 and recent_vol < baseline_vol * env_float(
                "MONEY_BUTTON_MIN_VOL_RATIO", 0.6, lo=0.0, hi=5.0
            ):
                return None

        # ------------------------------------------------------------------
        # Confidence, and a floor under it.
        # ------------------------------------------------------------------
        headroom = conservative_edge / max(required_edge, 1e-12)
        confidence = 0.55
        confidence += min(0.20, (headroom - 1.0) * 0.20)      # edge over the bar
        confidence += 0.06 if r30 > 0 and r10 > 0 else 0.0    # aligned windows
        confidence += min(0.08, max(0.0, slope_per_min) * 40.0)
        confidence -= min(0.15, volatility * 4.0)             # penalise noise
        confidence = max(0.01, min(0.95, confidence))

        min_confidence = env_float("MONEY_BUTTON_MIN_CONFIDENCE", 0.62, lo=0.5, hi=0.99)
        if confidence < min_confidence:
            return None

        # Snap to the lane this strategy is defined by: 5-10 minutes.
        horizon_minutes = int(min(10, max(5, round(hold_minutes))))

        target_price = float(ctx.last_price) * (1.0 + conservative_edge)

        return self.make_candidate(
            state,
            ctx,
            action="enter",
            # The CDCL solver compares expected_return against fee_rate; pass
            # the conservative figure so a candidate can never look better to
            # the solver than it did to the gate above.
            expected_return=conservative_edge,
            target_price=target_price,
            confidence=confidence,
            direction_prob=confidence,
            reason=(
                f"{horizon_minutes}m momentum {slope_per_min * 100:.3f}%/min; "
                f"net edge {conservative_edge * 100:.2f}% vs round-trip "
                f"{round_trip_cost * 100:.2f}%"
            ),
            horizon=f"{horizon_minutes}m",
            extra_meta={
                "lane": "money_button",
                "returns": {"5m": r5, "10m": r10, "30m": r30},
                "slope_per_min": slope_per_min,
                "projected_edge": projected,
                "conservative_edge": conservative_edge,
                "round_trip_cost": round_trip_cost,
                "required_edge": required_edge,
                "edge_headroom": headroom,
                "volatility": volatility,
                "distinct_prices": distinct,
                "window_sec": span_sec,
            },
        )
