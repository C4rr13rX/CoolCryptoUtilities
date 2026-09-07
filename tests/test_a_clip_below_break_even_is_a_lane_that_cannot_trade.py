"""A published live clip must be able to pay for its own round trip.

Measured 2026-09-07 over every live round trip this account has ever settled
(``trade_outcomes``, 18 rows, entry/exit prices and quantity as recorded),
replayed against ``services/roundtrip_cost`` with nothing changed but the size:

    clip      gross$      cost$        net$
    $ 0.75    +0.0770     0.1159      -0.0389     <- the clip actually taken
    $ 1.00    +0.1026     0.1302      -0.0276     <- LIVE_MICRO_MIN_CLIP_USD
    $ 2.00    +0.2053     0.1876      +0.0177
    $ 6.00    +0.6158     0.4170      +0.1987     <- LIVE_MIN_CLIP_USD
    $19.66    +2.0177     1.2007      +0.8170     <- deployable_stable_usd

Same fills, same direction calls, same hold times. The book changes SIGN
between $1 and $2, because the round trip costs ``0.004047 + 0.003187 * n``
and the fixed leg does not shrink with the trade: 0.858%/round trip at $0.75
against 0.386% at $6.00. Those 18 trades ran at a median notional of $0.750
and a maximum of $3.00 against a plan publishing ``min_clip_usd`` $6.00, so
the sign of the live book was decided by the clip and not by the market. Their
gross return averaged +0.5702%/trade, putting empirical break-even at
``0.004047 / (0.005702 - 0.003187)`` = $1.61.

The hole this pins: the transition plan's micro branch publishes
``LIVE_MICRO_MIN_CLIP_USD`` -- .env sets it to 1.00, the code default is 0.05 --
and neither can pay for itself. Downstream the entry gate is already
cost-proportional (``_entry_profit_floor_ratio`` demands gross >= 1.25x cost,
which at the measured 0.576% edge implies a $2.85 notional), so a sub-viable
clip does not produce small trades. It produces a live lane that refuses 100%
of entries while every gate in front of it reads open -- the same as being
switched off, and switched off precisely when the wallet is small enough for
micro mode to engage.

So the plan may not publish a clip below the notional at which a round trip
can pay for itself. It raises it instead, and records what it overrode. If the
wallet cannot fund the raised clip, the EXISTING ``min_clip_block`` path fires
with a reason -- an explicit block, which is the outcome that can be acted on.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from services.roundtrip_cost import (
    DEFAULT_FIXED_USD,
    DEFAULT_RATE,
    min_viable_notional_usd,
    roundtrip_cost_rate,
    roundtrip_cost_usd,
)


#: The 18 settled live round trips, as (symbol, gross_return). Returns are
#: ``exit_price / entry_price - 1`` off the recorded fills, so they are
#: independent of the notional each was actually taken at -- which is the whole
#: point: only the size is varied below.
LIVE_ROUND_TRIP_RETURNS = [
    0.001653658,   # AERO-USDC
    0.033760650,   # CBETH-USDC
    -0.007590863,  # CBETH-USDC
    -0.001393333,  # AERO-USDC
    -0.001367333,  # CBBTC-USDC
    0.341022667,   # BSTONK-USDC
    -0.038814667,  # BASECAT-USDC
    -0.014211362,  # CBBTC-USDC
    -0.023280000,  # AERO-USDC
    -0.003536409,  # CBBTC-USDC
    -0.016789333,  # CBXRP-USDC
    0.002296044,   # AERO-USDC
    0.007997676,   # AERO-USDC
    0.013573926,   # AERO-USDC
    -0.166595333,  # BPAD-USDC
    0.007361635,   # AERO-USDC
    -0.013516844,  # AERO-USDC
    -0.017942647,  # AERO-USDC
]


def _replay(clip_usd: float) -> float:
    """Net USD of the whole live book, had every trade been taken at one clip."""
    gross = sum(ret * clip_usd for ret in LIVE_ROUND_TRIP_RETURNS)
    cost = roundtrip_cost_usd(clip_usd) * len(LIVE_ROUND_TRIP_RETURNS)
    return gross - cost


class TestTheCostModelIsSizeDependent:
    def test_a_small_clip_pays_a_larger_rate(self):
        """The fixed leg is what makes a $0.75 clip a different game."""
        assert roundtrip_cost_rate(0.75) > roundtrip_cost_rate(6.00)
        # and not marginally: it is more than double.
        assert roundtrip_cost_rate(0.75) > 2.0 * roundtrip_cost_rate(6.00)

    def test_the_live_book_changes_sign_with_the_clip_alone(self):
        """Same trades, same fills. Only the size differs."""
        assert _replay(0.75) < 0.0, "the clip actually taken loses"
        assert _replay(1.00) < 0.0, "the micro clip loses"
        assert _replay(2.00) > 0.0
        assert _replay(6.00) > 0.0, "the clip the plan authorises wins"
        # Monotone in the clip, which is why the floor is the lever.
        assert _replay(0.75) < _replay(1.00) < _replay(2.00) < _replay(6.00)


class TestTheViableFloor:
    def test_the_floor_sits_above_break_even_and_below_the_authorised_clip(self):
        floor = min_viable_notional_usd()
        # Empirical break-even off the measured +0.5702%/trade gross return.
        empirical = DEFAULT_FIXED_USD / (0.005702 - DEFAULT_RATE)
        assert empirical == pytest.approx(1.609, abs=0.01)
        assert floor > empirical, "a floor at or below break-even refuses nothing"
        assert floor < 6.00, "a floor above the authorised clip blocks everything"
        assert _replay(floor) > 0.0, "the floor must be in the profitable region"

    def test_the_floor_is_where_the_cost_rate_meets_its_ceiling(self):
        """Not a hand-picked constant: it inverts the cost model."""
        floor = min_viable_notional_usd()
        assert roundtrip_cost_rate(floor) == pytest.approx(0.005, rel=1e-9)

    def test_an_unsatisfiable_ceiling_disables_the_floor_rather_than_everything(self):
        """A ceiling below the proportional rate cannot be met at any size.

        A guard that refuses every notional is the failure this whole test
        exists to prevent, so it must not be reachable by misconfiguring the
        guard itself.
        """
        with mock.patch.dict(
            os.environ, {"LIVE_MAX_ROUNDTRIP_COST_RATE": "0.001"}, clear=False
        ):
            assert min_viable_notional_usd() == 0.0


class TestThePlanNeverPublishesASubViableClip:
    """The regression: LIVE_MICRO_MIN_CLIP_USD reaching `min_clip_usd` intact.

    Driven through the real source of ``_build_transition_plan`` rather than a
    rebuilt copy of it -- a test that re-implements the branch it is guarding
    passes whatever the branch does.
    """

    @staticmethod
    def _plan_source() -> str:
        import inspect

        from trading.pipeline import TrainingPipeline

        return inspect.getsource(TrainingPipeline._build_transition_plan)

    def test_the_micro_clip_is_raised_to_the_viable_floor(self):
        """.env's LIVE_MICRO_MIN_CLIP_USD=1.00 is below break-even."""
        with mock.patch.dict(
            os.environ, {"LIVE_MICRO_MIN_CLIP_USD": "1.00"}, clear=False
        ):
            configured = float(os.environ["LIVE_MICRO_MIN_CLIP_USD"])
            assert _replay(configured) < 0.0, "premise: the configured clip loses"
            assert min_viable_notional_usd() > configured, (
                "the floor must actually raise it"
            )

    def test_the_raise_happens_after_the_micro_branch_not_before(self):
        """Ordering is the bug. A floor applied first is overwritten by micro.

        ``min_clip_usd`` is set from ``LIVE_MIN_CLIP_USD``, then REPLACED
        wholesale inside ``if wallet_state.get("micro_mode")``. A viable-floor
        clamp placed above that assignment reads as correct and does nothing.
        """
        source = self._plan_source()
        micro_at = source.find("LIVE_MICRO_MIN_CLIP_USD")
        floor_at = source.find("min_viable_notional_usd")
        assert micro_at != -1, "micro branch not found; this test needs rewriting"
        assert floor_at != -1, "the viable-clip floor is not applied in the plan"
        assert floor_at > micro_at, (
            "the viable-clip floor runs BEFORE the micro branch overwrites "
            "min_clip_usd, so micro mode still publishes a sub-viable clip"
        )

    def test_the_override_is_recorded_rather_than_silent(self):
        source = self._plan_source()
        # The NAME appearing is not the property: a zeroed placeholder mentions
        # it too. What matters is that the overridden value is captured before
        # it is lost, and that the plan carries it out.
        assert "min_clip_raised_from_usd = min_clip_usd" in source, (
            "a clip raised with no trace reads to the next reader as the clip "
            "that was always configured"
        )
        assert '"min_clip_raised_from_usd": min_clip_raised_from_usd' in source, (
            "the override is computed but never published, so nothing "
            "downstream -- the gate map, a snapshot, this test's next reader -- "
            "can see that a clip was overridden"
        )

    def test_the_floor_never_lowers_an_authorised_clip(self):
        """It may only raise. $6.00 is already viable and must survive intact."""
        source = self._plan_source()
        assert "min_clip_usd < viable_clip_usd" in source, (
            "the floor must be applied as a one-sided raise, never as an "
            "assignment that could shrink a larger authorised clip"
        )
