"""Ghost-only conversion of wallet dust into a cost-covered micro target.

The strategy treats the USD value of real sub-threshold holdings as a USDC
paper notional.  It never swaps those holdings itself and never emits in live
mode.  Existing scheduler/CDCL/ledger machinery therefore measures whether a
$0.50-$1.00 consolidation-and-redeployment idea survives fees before any live
promotion can even be considered.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from trading.strategies.base import Strategy, StrategyContext, env_float, sample_arrays


class DustMicroSwingStrategy(Strategy):
    strategy_id = "dust_micro_swing"
    default_horizon = "15m"
    min_samples = 24

    @staticmethod
    def _return_at(ts: np.ndarray, prices: np.ndarray, seconds: float) -> float:
        cutoff = float(ts[-1]) - seconds
        index = int(np.searchsorted(ts, cutoff, side="left"))
        index = min(max(index, 0), prices.size - 1)
        anchor = float(prices[index])
        return (float(prices[-1]) / anchor - 1.0) if anchor > 0 else 0.0

    def evaluate(self, state: Any, ctx: StrategyContext) -> Optional[Dict[str, Any]]:
        dust = ctx.extras.get("dust_micro") if isinstance(ctx.extras, dict) else None
        if ctx.live_trading or not isinstance(dust, dict) or not dust.get("enabled"):
            return None
        if str(getattr(state, "quote_token", "")).upper() != "USDC":
            return None

        budget = min(
            env_float("DUST_MICRO_MAX_USDC", 1.0, lo=0.25, hi=5.0),
            max(0.0, float(dust.get("budget_usdc") or 0.0)),
        )
        min_budget = env_float("DUST_MICRO_MIN_USDC", 0.50, lo=0.01, hi=5.0)
        if budget < min_budget:
            return None

        ts, prices, _ = sample_arrays(state, 30.0 * 60.0)
        if prices.size < self.min_samples or ctx.last_price <= 0:
            return None
        if float(ts[-1] - ts[0]) < 25.0 * 60.0:
            return None

        # A micro redeployment must be a visible dip that has started to
        # reverse, not a falling knife.  The 5m bounce plus 15m/30m drawdown
        # is the short-lived fluctuating pattern this lane is meant to learn.
        r5 = self._return_at(ts, prices, 5.0 * 60.0)
        r15 = self._return_at(ts, prices, 15.0 * 60.0)
        r30 = self._return_at(ts, prices, 30.0 * 60.0)
        if r5 <= 0.0 or min(r15, r30) >= 0.0:
            return None
        if prices.size < 4 or float(prices[-1]) <= float(prices[-3]):
            return None

        reference = float(np.median(prices[:-3]))
        raw_edge = reference / float(ctx.last_price) - 1.0
        if raw_edge <= 0.0:
            return None

        step_returns = np.diff(np.log(np.clip(prices, 1e-12, None)))
        volatility = float(np.std(step_returns)) if step_returns.size else 0.0
        mad = float(np.median(np.abs(prices - np.median(prices))))
        robust_sigma = 1.4826 * mad / max(reference, 1e-12)
        # Discount the apparent reversion edge for noisy bars.  Capping the
        # discount at 35% of the edge avoids treating volatility as certainty.
        uncertainty = min(raw_edge * 0.35, 0.5 * volatility + 0.10 * robust_sigma)
        conservative_edge = max(0.0, raw_edge - uncertainty)

        conversion_cost = env_float("DUST_CONVERSION_COST_RATIO", 0.005, lo=0.0, hi=0.25)
        min_profit_usdc = env_float("DUST_MICRO_MIN_PROFIT_USDC", 0.01, lo=0.001, hi=1.0)
        all_in_cost = float(ctx.fee_rate) + conversion_cost
        expected_net_usdc = budget * (conservative_edge - all_in_cost)
        if expected_net_usdc < min_profit_usdc:
            return None

        std = float(np.std(prices))
        zscore = (float(ctx.last_price) - float(np.mean(prices))) / max(std, 1e-12)
        confidence = 0.72
        confidence += min(0.12, max(0.0, -zscore) * 0.03)
        confidence += 0.06  # confirmed 5m bounce
        confidence += 0.04 if r15 < 0.0 and r30 < 0.0 else 0.0
        confidence += min(0.04, max(0.0, raw_edge - all_in_cost))
        confidence = min(0.96, confidence)
        min_confidence = env_float("DUST_MICRO_MIN_CONFIDENCE", 0.82, lo=0.5, hi=0.99)
        if confidence < min_confidence:
            return None

        # Estimate recovery time from the confirmed 5m rebound and snap it to
        # the scheduler's supported short horizons.
        rebound_per_min = max(r5 / 5.0, 1e-6)
        recovery_minutes = raw_edge / rebound_per_min
        horizon_minutes = min((5, 15, 30), key=lambda value: abs(value - recovery_minutes))
        horizon = f"{horizon_minutes}m"
        sources = [str(value) for value in (dust.get("source_tokens") or [])]
        return self.make_candidate(
            state,
            ctx,
            action="enter",
            expected_return=raw_edge,
            target_price=reference,
            confidence=confidence,
            direction_prob=confidence,
            reason=(
                f"ghost dust->USDC->{state.base_token} ${budget:.2f}; "
                f"conservative net ${expected_net_usdc:.3f}"
            ),
            horizon=horizon,
            quote_size=budget,
            extra_meta={
                "ghost_only": True,
                "dust_source_tokens": sources,
                "dust_budget_usdc": budget,
                "conversion_cost_ratio": conversion_cost,
                "all_in_cost_ratio": all_in_cost,
                "raw_edge": raw_edge,
                "conservative_edge": conservative_edge,
                "uncertainty_ratio": uncertainty,
                "expected_net_usdc": expected_net_usdc,
                "returns": {"5m": r5, "15m": r15, "30m": r30},
                "target_token": str(getattr(state, "base_token", "")),
            },
        )
