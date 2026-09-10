"""A hit rate above baseline but inside sampling noise is NOT an edge.

This is the bug the census caught in its own first draft. ``score_era`` measured
the head at 0.5421 against a 0.5308 majority baseline and the verdict function
read ``if edge > 0.0: return "INFORMATIVE"`` -- so a +0.0113 beat on 2721
samples was about to be published as "the head still carries signal, the
collapse is only calibration". The standard error at that sample size is
0.0096, so the beat was 1.18 SE: a coin flip clears it roughly one run in eight.

Publishing that would have sent the next pass to fix a calibrator on the
strength of noise, while the real finding -- the tape MOVED and the head had no
measurable direction signal at any horizon -- went unreported. This repo's
history is a list of fake edges found later at greater cost, so the guard is
that INFORMATIVE requires the 95% LOWER BOUND to clear the baseline, not the
point estimate.

Run: python -X utf8 -m pytest tests/test_a_hit_rate_inside_noise_is_not_an_edge.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "head_vs_realised_census.py",
)


def _load():
    spec = importlib.util.spec_from_file_location("head_vs_realised_census", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


census = _load()


def _rows(n_up, n_down, dp_up, dp_down, move=0.005):
    """Rows whose realised direction and head call are set explicitly."""
    out = []
    for _ in range(n_up):
        out.append({"ts": 0.0, "symbol": "X", "direction_prob": dp_up, "realised": move})
    for _ in range(n_down):
        out.append({"ts": 0.0, "symbol": "X", "direction_prob": dp_down, "realised": -move})
    return out


def test_a_beat_of_one_standard_error_is_not_called_informative():
    """The exact shape measured on 2026-09-10: +0.011 over baseline at n~2700."""
    # 1450 down / 1300 up: baseline (majority) = 1450/2750 = 0.5273.
    # The head calls DOWN on everything, so it hits every down row and misses
    # every up row -- hit rate 0.5273, identical to baseline by construction.
    # Then flip 31 up-rows to an UP call so the head edges ahead by ~0.011.
    rows = _rows(1300, 1450, dp_up=0.2, dp_down=0.2)
    for row in rows[:31]:
        row["direction_prob"] = 0.8

    era = census.score_era(rows, flat_bp=10.0)
    assert era["edge"] > 0.0, "precondition: the head is ahead on the point estimate"
    assert era["z_vs_majority"] < 1.96, "precondition: and that lead is inside noise"
    assert not era["significant"]

    kind, reason = census.verdict(era, min_scored=30)
    assert kind != "INFORMATIVE", (
        f"a {era['edge']:+.4f} lead at z={era['z_vs_majority']:+.2f} was published as "
        f"an edge: {reason}"
    )
    assert "inside sampling noise" in reason


def test_a_genuine_edge_is_still_called_informative():
    """The guard must not be so strict that a real signal is thrown away."""
    # Head calls the direction correctly on 80% of a balanced tape.
    rows = _rows(1000, 1000, dp_up=0.8, dp_down=0.2)
    for row in rows[:200]:  # 200 of the 1000 up rows get a wrong DOWN call
        row["direction_prob"] = 0.2
    for row in rows[1000:1200]:  # and 200 down rows get a wrong UP call
        row["direction_prob"] = 0.8

    era = census.score_era(rows, flat_bp=10.0)
    assert era["hit_rate"] == pytest.approx(0.8, abs=1e-9)
    assert era["significant"], "an 80% hit rate on n=2000 must clear the baseline"
    kind, _ = census.verdict(era, min_scored=30)
    assert kind == "INFORMATIVE"


def test_a_down_calling_head_scores_baseline_not_skill_in_a_down_tape():
    """Calling DOWN on everything is the majority baseline, never an edge.

    This is why the baseline is majority-class and not 0.5: the collapsed head
    reads direction_prob p50 0.041 -- DOWN on essentially every symbol -- over a
    tape that was 60.6% down at 60 minutes. Scored against a coin it looks like
    a 0.61 hit rate and a discovery; scored against the tape's own majority it
    is exactly zero information.
    """
    rows = _rows(394, 606, dp_up=0.04, dp_down=0.04)  # head says DOWN on all
    era = census.score_era(rows, flat_bp=10.0)

    assert era["hit_rate"] == pytest.approx(0.606, abs=1e-9)
    assert era["majority"] == pytest.approx(0.606, abs=1e-9)
    assert era["edge"] == pytest.approx(0.0, abs=1e-9)
    assert not era["significant"]
    assert census.verdict(era, min_scored=30)[0] != "INFORMATIVE"


def test_a_moving_tape_with_a_flat_head_is_the_model_not_the_market():
    """MODEL and MARKET must not be interchangeable -- they are opposite actions."""
    moving = _rows(500, 500, dp_up=0.04, dp_down=0.04, move=0.005)  # 0.5% moves
    assert census.verdict(census.score_era(moving, flat_bp=10.0), 30)[0] == "MODEL"

    flat = _rows(500, 500, dp_up=0.04, dp_down=0.04, move=0.00001)  # 0.001% moves
    assert census.verdict(census.score_era(flat, flat_bp=10.0), 30)[0] == "MARKET"


def test_the_no_model_sentinel_is_not_counted_as_a_prediction():
    """(dp==0.5, net_margin==0.0) is bot.py's no-model row, not a head reading."""
    assert census.SENTINEL_DIRECTION_PROB == 0.5
    assert census.SENTINEL_NET_MARGIN == 0.0
    # A tie at exactly 0.5 is an abstention and must not be scored either way.
    rows = _rows(500, 500, dp_up=0.5, dp_down=0.5)
    era = census.score_era(rows, flat_bp=10.0)
    assert era["scored"] == 0, "abstentions were scored as directional calls"


def test_a_non_monotonic_ranking_is_reported_as_non_monotonic():
    """A calibrator preserves ORDER, so a broken order is not a calibration bug.

    The measured pre-collapse head ran quintile up-rates 50.3 / 52.7 / 31.9 /
    37.9 / 43.0 -- its most confident UP calls were LESS likely to go up than
    its least confident ones. If this flag ever reads YES on that shape, the
    next pass gets sent to fix a calibrator that cannot help.
    """
    rows = []
    # Confidence rises with dp, but the realised up-rate falls: an INVERTED head.
    for i, up_rate in enumerate((0.9, 0.7, 0.5, 0.3, 0.1)):
        for j in range(100):
            rows.append(
                {
                    "ts": 0.0,
                    "symbol": "X",
                    "direction_prob": 0.1 + i * 0.2,
                    "realised": 0.005 if j < up_rate * 100 else -0.005,
                }
            )
    profile = census.rank_profile(rows)
    assert not profile["monotonic"]
    assert profile["up_spread"] < 0, "an inverted head must show a negative spread"

    # And the honest positive control: a head whose order genuinely works.
    good = []
    for i, up_rate in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        for j in range(100):
            good.append(
                {
                    "ts": 0.0,
                    "symbol": "X",
                    "direction_prob": 0.1 + i * 0.2,
                    "realised": 0.005 if j < up_rate * 100 else -0.005,
                }
            )
    good_profile = census.rank_profile(good)
    assert good_profile["monotonic"]
    assert good_profile["up_spread"] > 0


def test_a_non_monotonic_ranking_is_not_assumed_to_be_uninformative():
    """The correction: non-monotonic across five buckets is not "no signal".

    An earlier version of the census reported only the quintile profile and
    concluded "the order carries nothing, so a calibrator cannot rescue it"
    from its non-monotonicity. A noisy but genuinely informative ranking is
    routinely non-monotonic across five buckets, so AUC has to be the arbiter.
    This pins that a head can be non-monotonic AND above chance at once.
    """
    rows = []
    # Overall up-rate rises with dp, but bucket 3 dips below bucket 2.
    for i, up_rate in enumerate((0.30, 0.55, 0.40, 0.60, 0.75)):
        for j in range(400):
            rows.append(
                {
                    "ts": 0.0,
                    "symbol": "X",
                    "direction_prob": 0.1 + i * 0.2,
                    "realised": 0.005 if j < up_rate * 400 else -0.005,
                }
            )
    profile = census.rank_profile(rows)
    assert not profile["monotonic"], "precondition: the quintile profile dips"
    assert profile["auc"] > 0.5
    assert profile["auc_beats_chance"], (
        "a non-monotonic but informative ranking was written off as chance"
    )
    assert not profile["auc_inverted"]


def test_an_inverted_head_is_named_inverted_not_merely_unskilled():
    """Inverted and unskilled are different diagnoses with different actions.

    The pre-collapse head reads AUC 0.4076 +/- 0.0130 -- significantly BELOW
    chance. Restoring that state, which is what a 'direction_prob p50 rises
    back above 0.5' criterion rewards, restores a head that is more wrong the
    more confident it is.
    """
    rows = []
    for i, up_rate in enumerate((0.75, 0.60, 0.55, 0.40, 0.25)):
        for j in range(400):
            rows.append(
                {
                    "ts": 0.0,
                    "symbol": "X",
                    "direction_prob": 0.1 + i * 0.2,
                    "realised": 0.005 if j < up_rate * 400 else -0.005,
                }
            )
    profile = census.rank_profile(rows)
    assert profile["auc"] < 0.5
    assert profile["auc_inverted"]
    assert not profile["auc_beats_chance"]


def test_a_coin_flip_ranking_clears_neither_bound():
    """The null case must be called chance, not skill and not inversion."""
    rows = []
    for i in range(5):
        for j in range(400):
            rows.append(
                {
                    "ts": 0.0,
                    "symbol": "X",
                    "direction_prob": 0.1 + i * 0.2,
                    "realised": 0.005 if j % 2 == 0 else -0.005,
                }
            )
    profile = census.rank_profile(rows)
    assert profile["auc"] == pytest.approx(0.5, abs=0.02)
    assert not profile["auc_beats_chance"]
    assert not profile["auc_inverted"]


def test_a_nearest_tick_outside_tolerance_is_refused():
    """A stale price silently reached for is a fabricated forward return."""
    series = ([100.0, 200.0], [1.0, 2.0])
    assert census.price_at(series, 205.0, tolerance_sec=10.0) == 2.0
    assert census.price_at(series, 260.0, tolerance_sec=10.0) is None
    assert census.price_at(series, 150.0, tolerance_sec=10.0) is None
