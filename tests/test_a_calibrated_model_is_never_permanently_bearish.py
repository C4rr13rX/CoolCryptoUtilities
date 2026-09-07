"""A Platt calibration must not move the point every threshold calls neutral.

MEASURED 2026-09-07, production, 3007 paired evaluations over 72h taken from
``organism_snapshots``:

    the model's own output (direction_prob_raw)   median 0.5985   max 0.9498
    what the decision path actually saw           median 0.1414   max 0.7183
    >= 0.58 (bot.py enter_threshold)              16 / 3007  =  0.53%
    >= 0.60 (SCHEDULER_MIN_DIRECTION_PROB)        15 / 3007  =  0.50%
    read as bearish (< 0.5)                     2910 / 3007  = 96.8%

The model was bullish just over half the time and the decision path read it as
bearish 96.8% of the time. Fitting the transform on the 1324 evaluations where
``graph_confidence`` was exactly 1.0 (so nothing else touched the number) gives

    logit_out = 0.9806 * logit_in - 2.0784      median |residual| = 0.048

which is the active model's Platt calibration. Its no-information point -- where
it sends a model that said 0.5 -- is sigmoid(-2.0784) = 0.111, not 0.5. On that
scale, clearing the 0.58 entry gate needs a raw model output of 0.906.

Every threshold that reads ``direction_prob`` was chosen against a 0.5-neutral
scale, and services/env_loader.py says so out loud: it pins
MONEY_BUTTON_MIN_DIR_PROB to "0.50" with the comment "so the neutral case
PASSES". So the calibration silently re-scaled the quantity under all of them,
and 379 of 532 scheduler route evaluations in 6h terminated at
``no_candidates (thresholds not met)``.

Two separate places did it. Both are covered here:

  * ``_summarise_predictions`` -- the calibration offset, which carries the
    BASE RATE, was left in a number that is compared against a 0.5 neutral.
  * ``_interpret_predictions`` -- ``direction_prob * graph_conf``, a
    probability multiplied by a confidence, which shrinks toward "certainly
    down" instead of toward "no opinion".
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.bot import TradingBot, damp_direction_prob  # noqa: E402


#: The active model's fitted calibration, measured as described above.
MEASURED_SCALE = 0.9806
MEASURED_OFFSET = -2.0784

#: bot.py's ``enter_threshold`` default and SCHEDULER_MIN_DIRECTION_PROB.
ENTER_THRESHOLD = 0.58
SCHEDULER_DIR_FLOOR = 0.60


def _summarise(direction_prob, *, cal_scale=None, cal_offset=None, temperature=1.0):
    """Run the real ``_summarise_predictions`` against a stub pipeline.

    Unbound on purpose: the method reads nothing off ``self`` except
    ``self.pipeline``, and constructing a TradingBot would drag in TF, the
    database and the network.
    """
    pipeline = types.SimpleNamespace(
        calibration_scale=cal_scale,
        calibration_offset=cal_offset,
        temperature_scale=temperature,
    )
    # horizon_forecast is consulted after the calibration block; make it fail
    # the way a missing model does so the method takes its own except branch.
    def _boom(*_args, **_kwargs):
        raise RuntimeError("no model in this test")

    pipeline.horizon_forecast = _boom
    stub = types.SimpleNamespace(pipeline=pipeline)
    preds = [
        [[0.5]],            # exit_conf
        [[0.0]],            # price_mu
        [[0.0]],            # price_log_var
        [[direction_prob]],  # price_dir  <- the one under test
        [[0.0]],            # net_margin
        [[0.0]],            # net_pnl
    ]
    return TradingBot._summarise_predictions(stub, preds, current_price=1.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# The calibration offset
# ---------------------------------------------------------------------------


def test_a_bullish_model_survives_the_measured_calibration():
    """0.60 from the model must not reach the gates as 0.157.

    This is the regression. Against the old code the value published here was
    ``sigmoid(0.9806 * logit(0.60) - 2.0784)`` = 0.157, so a bullish call
    arrived at every gate as a strongly bearish one.
    """
    summary = _summarise(0.60, cal_scale=MEASURED_SCALE, cal_offset=MEASURED_OFFSET)

    old_behaviour = _sigmoid(MEASURED_SCALE * _logit(0.60) + MEASURED_OFFSET)
    assert old_behaviour < 0.20, "premise check: the old transform really did this"

    assert summary["direction_prob"] > 0.5, (
        "a model that says 'up' must reach the gates as 'up'; got "
        f"{summary['direction_prob']:.4f}"
    )
    assert summary["direction_prob"] >= ENTER_THRESHOLD


def test_the_neutral_model_lands_exactly_on_neutral():
    """A model with no opinion must arrive as no opinion, not as 0.111.

    sigmoid(cal_offset) = 0.111 is where the old code sent 0.5, which is below
    every bearish floor in the system -- so an abstaining model voted SELL.
    """
    summary = _summarise(0.5, cal_scale=MEASURED_SCALE, cal_offset=MEASURED_OFFSET)
    assert summary["direction_prob"] == pytest.approx(0.5, abs=1e-9)
    assert summary["direction_prob_neutral"] == pytest.approx(0.111, abs=0.002)


def test_a_bearish_model_is_still_bearish():
    """The guard is not removed. Down must still read down."""
    summary = _summarise(0.30, cal_scale=MEASURED_SCALE, cal_offset=MEASURED_OFFSET)
    assert summary["direction_prob"] < 0.5
    assert summary["direction_prob"] < ENTER_THRESHOLD
    assert summary["direction_prob"] < SCHEDULER_DIR_FLOOR


def test_the_sharpening_half_of_the_calibration_is_kept():
    """Only the base rate is divided out; the scale still does its work.

    A scale of 2.0 with no offset must still sharpen, and must be untouched by
    the re-centring -- otherwise this fix would be quietly deleting the
    calibration rather than putting it on the right scale.
    """
    summary = _summarise(0.60, cal_scale=2.0, cal_offset=0.0)
    expected = _sigmoid(2.0 * _logit(0.60))
    assert summary["direction_prob"] == pytest.approx(expected, abs=1e-9)
    assert summary["direction_prob"] > 0.60, "scale > 1 must sharpen, not flatten"


def test_the_true_calibrated_probability_is_still_published():
    """Anything that needs a genuine P(up) can still have one."""
    summary = _summarise(0.60, cal_scale=MEASURED_SCALE, cal_offset=MEASURED_OFFSET)
    expected = _sigmoid(MEASURED_SCALE * _logit(0.60) + MEASURED_OFFSET)
    assert summary["direction_prob_calibrated"] == pytest.approx(expected, abs=1e-9)
    assert summary["direction_prob_raw"] == pytest.approx(0.60, abs=1e-9)


def test_an_uncalibrated_model_is_judged_exactly_as_before():
    """No calibration configured means the temperature branch, untouched."""
    summary = _summarise(0.60, cal_scale=None, cal_offset=None, temperature=1.0)
    assert summary["direction_prob"] == pytest.approx(0.60, abs=1e-9)
    assert "direction_prob_calibrated" not in summary


# ---------------------------------------------------------------------------
# The graph-confidence discount
# ---------------------------------------------------------------------------


def test_a_confidence_discount_does_not_flip_a_bullish_call():
    """0.62 at graph_conf 0.80 came out as 0.496 -- bearish, on a discount."""
    assert 0.62 * 0.80 < 0.5, "premise check: the old multiply really flipped it"
    assert damp_direction_prob(0.62, 0.80) > 0.5


def test_no_confidence_means_no_opinion_not_certainly_down():
    """graph_conf 0 must give 0.5. The old code gave 0.0."""
    assert damp_direction_prob(0.90, 0.0) == pytest.approx(0.5, abs=1e-9)
    assert damp_direction_prob(0.10, 0.0) == pytest.approx(0.5, abs=1e-9)


def test_full_confidence_changes_nothing():
    for prob in (0.05, 0.5, 0.62, 0.95):
        assert damp_direction_prob(prob, 1.0) == pytest.approx(prob, abs=1e-9)


def test_the_discount_no_longer_caps_the_entry_gate():
    """A 0.80-confidence tick could never clear 0.58 under the old multiply.

    ``0.58 / 0.80 = 0.725`` was the old cap's demand on the model, and the cap
    itself was ``direction_prob <= graph_conf``: at graph_conf 0.57 the gate
    was unreachable at ANY model output.
    """
    assert 0.95 * 0.57 < ENTER_THRESHOLD, "premise check: the old cap really bound"
    assert damp_direction_prob(0.95, 0.57) >= ENTER_THRESHOLD


def test_a_bearish_call_stays_bearish_under_a_discount():
    """The discount must shrink conviction, never invent the opposite one."""
    damped = damp_direction_prob(0.20, 0.50)
    assert damped < 0.5
    assert damped > 0.20, "a discount must move toward neutral, not away from it"


def test_an_over_unity_confidence_cannot_leave_the_unit_interval():
    """graph_confidence is observed up to 1.1046 in production."""
    assert 0.0 <= damp_direction_prob(0.99, 1.1046) <= 1.0
    assert 0.0 <= damp_direction_prob(0.01, 1.1046) <= 1.0


def test_a_broken_confidence_is_an_abstention():
    assert damp_direction_prob(0.9, float("nan")) == pytest.approx(0.5)
    assert damp_direction_prob(None, 1.0) == pytest.approx(0.5)


def test_the_decision_path_actually_calls_the_damping_helper():
    """A helper nothing calls fixes nothing.

    This is not paranoia: while this pass was being written, a concurrent
    write to trading/bot.py from a stale buffer reverted the call site and
    left the helper in place, and every test above still passed. Reading the
    compiled code object is the cheapest way to assert the wiring rather than
    the arithmetic -- ``co_names`` carries the global a function actually
    looks up when it runs.
    """
    names = TradingBot._interpret_predictions.__code__.co_names
    assert "damp_direction_prob" in names, (
        "_interpret_predictions no longer calls damp_direction_prob; the "
        "probability-times-confidence bug is back"
    )
