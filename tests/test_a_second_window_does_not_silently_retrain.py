"""The two-window protocol's load-bearing property, and the leak beside it.

The standing rule for the brain is that held-out edge means an UP window AND
a DOWN window, measured BACK-TO-BACK ON ONE FABRIC. Node run-to-run variance
is large enough that cross-session numbers are noise -- the same fabric and
samples gave 89.2%% and 93.6%% thirty-four minutes apart -- so "one fabric" is
not a nicety, it is the only thing that makes the two numbers comparable.

Before ``plan_windows`` existed, ``omen_experiment`` derived the training
window FROM the test window::

    test_stop  = len(bars) - horizon - 1     # always the end of the corpus
    test_start = test_stop - test
    train_stop = test_start - horizon        # <-- moves when the test moves

That has two consequences, and this file pins both:

  1. The held-out window could not be moved at all. Every run scored the
     last ``--test`` bars, so a second window was unreachable and the
     UP-and-DOWN requirement could not be satisfied even in principle.
  2. Moving it would have silently retrained. Two windows scored against
     two different training sets are two experiments, not two measurements
     of one fabric, and the difference between them measures the training
     data rather than the topology under test.

The second is the dangerous one, because it produces a number that looks
like the number you wanted.
"""
from __future__ import annotations

import pytest

from scripts.omen_experiment import (
    WindowError, plan_windows, window_regime,
)
from trading.omen_brain import LOOKBACK_BARS


TRAIN = 500
TEST = 100
HORIZON = 12
TOTAL = LOOKBACK_BARS + TRAIN + HORIZON + TEST + HORIZON + 2000


def test_pinned_train_end_does_not_move_when_the_test_window_moves():
    """THE property. Same --train-end, two test windows, one training set.

    Against the old derivation this fails: train_stop tracked test_start, so
    the two windows would have been scored on fabrics taught different bars.
    """
    pinned = LOOKBACK_BARS + TRAIN

    early = plan_windows(TOTAL, TRAIN, TEST, HORIZON,
                         train_end=pinned, test_end=pinned + HORIZON + TEST)
    late = plan_windows(TOTAL, TRAIN, TEST, HORIZON,
                        train_end=pinned, test_end=TOTAL - HORIZON - 1)

    assert early["test_stop"] != late["test_stop"], (
        "the two windows must actually be different windows")
    assert (early["train_start"], early["train_stop"]) == \
           (late["train_start"], late["train_stop"]), (
        "train window moved with the test window: the second measurement is "
        "against a DIFFERENT fabric and the comparison is meaningless")


def test_unpinned_train_end_still_tracks_the_test_window():
    """The default is unchanged -- this is a new capability, not a new default."""
    a = plan_windows(TOTAL, TRAIN, TEST, HORIZON, test_end=TOTAL - HORIZON - 1)
    b = plan_windows(TOTAL, TRAIN, TEST, HORIZON,
                     test_end=TOTAL - HORIZON - 1 - 400)
    assert a["train_stop"] != b["train_stop"]
    # and the default test window is still the end of the corpus
    assert a["test_stop"] == TOTAL - HORIZON - 1


def test_the_purge_gap_survives_every_split():
    """No training sample's future may reach into the held-out window."""
    for test_end in (TOTAL - HORIZON - 1, TOTAL - 900, LOOKBACK_BARS + TRAIN + HORIZON + TEST):
        plan = plan_windows(TOTAL, TRAIN, TEST, HORIZON, test_end=test_end)
        assert plan["train_stop"] + HORIZON <= plan["test_start"], (
            f"train future overlaps held-out bars at test_end={test_end}")
        assert plan["train_start"] >= LOOKBACK_BARS


def test_a_train_end_that_overruns_the_test_window_is_refused():
    """A pin that leaks the answer must be an error, not a quiet overlap."""
    test_end = LOOKBACK_BARS + TRAIN + HORIZON + TEST
    test_start = test_end - TEST
    with pytest.raises(WindowError, match="overlap|overrun"):
        plan_windows(TOTAL, TRAIN, TEST, HORIZON,
                     train_end=test_start - HORIZON + 1, test_end=test_end)


def test_a_window_too_early_for_its_lookback_is_refused():
    with pytest.raises(WindowError, match="lookback"):
        plan_windows(TOTAL, TRAIN, TEST, HORIZON, test_end=LOOKBACK_BARS + 1)


def _bars(closes):
    return [{"timestamp": 1_700_000_000 + i * 3600, "close": c}
            for i, c in enumerate(closes)]


def test_window_regime_names_up_and_down_from_the_bars():
    rising = _bars([100.0 * (1.01 ** i) for i in range(200)])
    falling = _bars([100.0 * (0.99 ** i) for i in range(200)])

    up = window_regime(rising, 0, 200, HORIZON)
    down = window_regime(falling, 0, 200, HORIZON)

    assert up["regime"] == "UP" and up["up_rate"] > 0.9
    assert down["regime"] == "DOWN" and down["up_rate"] < 0.1
    assert up["mean_forward"] > 0 > down["mean_forward"]


def test_a_coin_flip_window_is_FLAT_not_a_direction():
    """50.4%% must not be reported as "the UP window".

    A window inside the noise band is not a direction, and calling one UP is
    how a long-only rule gets credited with an edge it does not have.
    """
    # Odd horizon, so the alternation is a genuine coin flip rather than a
    # sequence of exactly-zero forwards (that case is the next test).
    alternating = _bars([100.0 + (1.0 if i % 2 else -1.0) for i in range(200)])
    info = window_regime(alternating, 0, 200, horizon=11)
    assert info["regime"] == "FLAT", (
        f"up_rate {info['up_rate']:.3f} was called {info['regime']}")


def test_a_frozen_feed_is_FLAT_not_a_DOWN_window():
    """A window that never moved must not be scored as a down market.

    Every forward is exactly 0.0, which is not > 0. Counting those as "not
    up" reads up_rate 0.0 and names a dead feed the DOWN window -- and this
    repo has shipped frozen feeds, once with 82 of 94 symbols holding a seed
    price. A stall must be visible as a stall.
    """
    frozen = _bars([100.0] * 200)
    info = window_regime(frozen, 0, 200, HORIZON)
    assert info["zero_share"] == pytest.approx(1.0)
    assert info["regime"] == "FLAT", (
        f"a frozen window was called {info['regime']}")


def test_a_mostly_frozen_window_does_not_get_a_direction_from_its_few_live_bars():
    """80%% stalled with the live 20%% all rising is still not an UP window."""
    closes = [100.0] * 160 + [100.0 * (1.02 ** i) for i in range(1, 41)]
    info = window_regime(_bars(closes), 0, 200, HORIZON)
    assert info["zero_share"] > 0.5
    assert info["regime"] == "FLAT"
