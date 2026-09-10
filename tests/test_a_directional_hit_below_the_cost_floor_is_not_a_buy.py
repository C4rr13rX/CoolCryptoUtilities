"""A correct direction call on a move smaller than the round trip is a LOSS.

Operator direction, 2026-09-10 15:07: stop leading with accuracy. The goal is
being accurate specifically about WHEN WE CAN BUY LOW AND SELL HIGH, so the
head has to be scored as a money rule. ``buy_rule_profile`` in
``scripts/head_vs_realised_census.py`` is that scoreboard, and this file pins
the three properties that make it a money number rather than an accuracy
number dressed up as one.

WHY THIS TEST EXISTS. The obvious implementation of "buy precision" is

    paid = sum(1 for r in group if r["realised"] > 0)

-- the share of up-calls the tape agreed with. That number is the one this
repo has already been fooled by: measured in the same census, only 30.5% of
15-minute ticks move further than the 0.3187% proportional cost at all, so a
sign-based precision counts roughly two thirds of its "correct" calls on moves
that could not have paid for themselves. Every assertion below is written so
that a sign-based implementation FAILS it.

THE THREE PROPERTIES:

  1. A hit must CLEAR THE FLOOR. floor = pct_cost/100 + fixed_cost/clip, and a
     realised move strictly between 0 and the floor scores as a miss.
  2. AN EXIT PAYS ONE LEG, NOT A ROUND TRIP. Leaving a position we already
     hold is a single swap, so the sell side is scored against floor/2. Using
     the round trip there would understate the exit rule by half its cost and
     is the "round trip billed twice to one leg" shape that
     ``services/profit_logic_audit`` flags.
  3. THE BASELINE IS THE EVERY-BAR MONEY RULE, over the SAME rows. Not 0.5 and
     not the majority class -- both flatter a head that has stopped calling
     up. Measured post-collapse: the head says UP on 65 of 2734 rows, so a
     baseline drawn from anywhere but these rows is not a comparison.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from head_vs_realised_census import buy_rule_profile  # noqa: E402

# clip 1.0 makes the fixed leg its own full value, so the floor is exactly
# 0.3187% + 0.4047% = 0.7234%. Chosen so the numbers below are readable.
PCT_COST = 0.3187
FIXED_COST = 0.004047
CLIP = 1.0
FLOOR = PCT_COST / 100.0 + FIXED_COST / CLIP  # 0.0072337
ONE_LEG = FLOOR / 2.0


def _row(dp: float, realised: float) -> dict:
    return {"ts": 0.0, "symbol": "AERO-USDC", "direction_prob": dp, "realised": realised}


def test_the_floor_is_the_round_trip_and_the_exit_pays_one_leg() -> None:
    """The arithmetic the whole scoreboard rests on, spelled out."""
    profile = buy_rule_profile([_row(0.9, 0.01)], PCT_COST, CLIP, FIXED_COST)
    assert profile["floor"] == pytest.approx(FLOOR)
    assert profile["one_leg"] == pytest.approx(FLOOR / 2.0)
    # And the fixed leg amortises over the clip -- the only thing clip moves.
    fatter = buy_rule_profile([_row(0.9, 0.01)], PCT_COST, 10.0, FIXED_COST)
    assert fatter["floor"] == pytest.approx(PCT_COST / 100.0 + FIXED_COST / 10.0)
    assert fatter["floor"] < profile["floor"]


def test_an_up_call_on_a_sub_floor_up_move_scores_zero_precision() -> None:
    """THE BUG THIS FILE IS NAMED FOR. Sign agreement is not a paid trade.

    All four rows are up-calls and all four moved UP, so a sign-based
    precision reads 1.0. All four moved LESS than the round trip, so not one
    of them could have paid for itself and the honest precision is 0.0.
    """
    rows = [_row(0.9, FLOOR * f) for f in (0.05, 0.30, 0.75, 0.99)]
    profile = buy_rule_profile(rows, PCT_COST, CLIP, FIXED_COST)

    assert profile["buy"]["n"] == 4
    assert profile["buy"]["precision"] == 0.0, (
        "a sign-based precision reads 1.0 here; every move is below the floor"
    )
    # Gross is positive and net is still negative: that IS the cost floor.
    assert profile["buy"]["gross_per_trade"] > 0.0
    assert profile["buy"]["net_per_trade"] < 0.0


def test_precision_counts_only_the_moves_that_cleared_the_floor() -> None:
    """Two of five up-calls clear it; precision is 0.4, not the 0.8 sign gives."""
    rows = [
        _row(0.9, FLOOR * 2.0),    # paid
        _row(0.8, FLOOR * 1.5),    # paid
        _row(0.7, FLOOR * 0.5),    # up, but did not cover the round trip
        _row(0.6, FLOOR * 0.1),    # up, but did not cover the round trip
        _row(0.9, -FLOOR * 2.0),   # wrong outright
    ]
    profile = buy_rule_profile(rows, PCT_COST, CLIP, FIXED_COST)
    assert profile["buy"]["precision"] == pytest.approx(0.4)
    # Sign agreement would have been 4/5.
    assert profile["buy"]["precision"] != pytest.approx(0.8)


def test_the_sell_side_is_scored_against_one_leg_not_the_round_trip() -> None:
    """A fall worth exiting is bigger than ONE swap, not two.

    The row falls by 0.6 of the round trip -- more than one leg, less than
    both. Scoring the exit against the round trip would call it a miss and
    halve the measured value of every exit rule.
    """
    rows = [_row(0.1, -FLOOR * 0.6)]
    profile = buy_rule_profile(rows, PCT_COST, CLIP, FIXED_COST)
    assert profile["sell"]["n"] == 1
    assert profile["sell"]["precision"] == 1.0, (
        "scoring the exit against the round trip instead of one leg loses this row"
    )
    # An exit's P/L is the loss AVOIDED, so a fall is a positive net.
    assert profile["sell"]["net_per_trade"] > 0.0


def test_the_baseline_is_every_bar_in_the_same_rows() -> None:
    """The head firing on 2 of 10 rows is compared against all 10, not 0.5."""
    rows = [_row(0.9, FLOOR * 2.0), _row(0.9, FLOOR * 2.0)]
    rows += [_row(0.3, -FLOOR * 2.0) for _ in range(8)]
    profile = buy_rule_profile(rows, PCT_COST, CLIP, FIXED_COST)

    assert profile["buy"]["n"] == 2
    assert profile["buy"]["precision"] == 1.0
    # Buy-every-bar clears the floor on 2 of 10, and THAT is what 1.0 beats.
    assert profile["buy_baseline"]["n"] == 10
    assert profile["buy_baseline"]["precision"] == pytest.approx(0.2)


def test_a_head_that_calls_nothing_up_gets_no_free_baseline() -> None:
    """Post-collapse the head's whole dp distribution sits below 0.5.

    Measured 2026-09-10 over 26h at a 15-minute horizon: post-collapse the top
    DECILE of direction_prob starts at 0.1106, so 'head says UP' fires 65
    times in 2734 rows. The top-decile rule must still be scored, because a
    rank threshold reads the ORDERING and is indifferent to the level -- that
    is the whole point of separating the two properties.
    """
    rows = [_row(0.02 + i * 0.001, FLOOR * (2.0 if i >= 90 else -1.0)) for i in range(100)]
    profile = buy_rule_profile(rows, PCT_COST, CLIP, FIXED_COST)

    assert profile["buy"]["n"] == 0, "no row is above 0.5, so the LEVEL rule never fires"
    assert profile["buy"]["precision"] != profile["buy"]["precision"]  # NaN
    # The ORDERING rule still fires, and here it is the informative one.
    assert profile["buy_top_decile"]["n"] == 10
    assert profile["buy_top_decile"]["precision"] == 1.0
    assert profile["buy_top_decile"]["precision"] > profile["buy_baseline"]["precision"]


def test_higher_precision_can_come_with_a_worse_net_per_trade() -> None:
    """Volatility selection, which is what the ordering actually buys.

    MEASURED, pre-collapse, 26h, 15-minute horizon, clip $5: the top decile of
    direction_prob clears the floor UPWARD on 17.60% of bars against 9.48% for
    buying every bar -- +8.1 points, the largest effect in the table -- while
    its mean return is -0.0358% against the baseline's +0.0191%. It selects
    bigger moves in BOTH directions, so it raises the share that pays AND the
    tail that does not. A scoreboard reporting precision without net/trade
    would have published that as an edge.
    """
    # Two rows that pay big, two that lose bigger: precision beats a flat
    # baseline while the mean is worse.
    volatile = [_row(0.99, FLOOR * 3.0), _row(0.98, FLOOR * 3.0),
                _row(0.97, -FLOOR * 5.0), _row(0.96, -FLOOR * 5.0)]
    flat = [_row(0.10, FLOOR * 0.2) for _ in range(16)]
    profile = buy_rule_profile(volatile + flat, PCT_COST, CLIP, FIXED_COST, top_frac=0.20)

    top = profile["buy_top_decile"]
    base = profile["buy_baseline"]
    assert top["n"] == 4
    assert top["precision"] > base["precision"], "the ordering does select payers"
    assert top["net_per_trade"] < base["net_per_trade"], (
        "and it still loses more per trade -- precision alone is not an edge"
    )
