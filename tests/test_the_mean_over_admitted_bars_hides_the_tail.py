"""A rule that CHOOSES when to fire is not described by its mean.

Every omen generalisation number so far was a mean over the bars a threshold
admitted, and the threshold search itself could not look past the 95th
percentile (`choose_threshold` walks `range(5, 96, 5)`) and refused any cut
holding fewer than `--min-trades` bars. So the question "does the score know
when the move is BIG" had no way of being answered: the top 1% never appeared
in a reported number.

That distinction decides the omen. Peak edge measured 2026-09-07 was +0.0916%
against a 0.6500% round trip -- a 7x shortfall no accuracy improvement closes.
The only shape that closes it is a score that ranks MAGNITUDE, so that a rare,
tight cut holds moves several times the mean. `tail_profile` reports exactly
that, out of sample, and reports it whether or not a threshold cleared the
trade floor.

Against the old code these tests fail at import: there was no `tail_profile`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_generalisation import (  # noqa: E402
    TAIL_QUANTILES, tail_profile,
)


def _rows(forwards):
    return [{"index": i, "forward": f} for i, f in enumerate(forwards)]


def test_a_perfect_ranker_shows_its_edge_only_in_the_tail():
    """100 bars, 1 of them +10%, the rest flat. The mean says nothing."""
    forwards = [0.0] * 99 + [0.10]
    scores = [0.0] * 99 + [1.0]  # the score fingers the one that moved
    profile = tail_profile(_rows(forwards), scores, cost=0.0065)

    assert profile[0.50][0] == 50  # bars kept
    # top 1% is the single +10% bar; top 50% dilutes it 50-fold.
    assert profile[0.01][1] / profile[0.01][0] == 0.10
    assert abs(profile[0.50][1] / profile[0.50][0] - 0.10 / 50) < 1e-12
    # ...and only the tail clears a 0.65% round trip.
    assert profile[0.01][2] == 1
    assert profile[0.50][2] == 1


def test_a_score_blind_to_magnitude_is_flat_across_every_quantile():
    """The null the omen has to beat: ranking uncorrelated with the move."""
    forwards = [0.01 if i % 2 else -0.01 for i in range(400)]
    scores = [float(i) for i in range(400)]  # ranks by index, not by outcome
    profile = tail_profile(_rows(forwards), scores, cost=0.0065)
    means = [profile[q][1] / profile[q][0] for q in profile]
    assert max(means) - min(means) < 0.011, (
        "an index-ordered score must not separate alternating outcomes")


def test_the_tail_ranks_within_the_symbol_not_across_the_pool():
    """Scores are deviations from a per-symbol base rate and are not pooled.

    Feeding one symbol's rows in must never depend on another's scale: doubling
    every score leaves the ranking, and therefore every reported bucket,
    identical.
    """
    forwards = [0.03, -0.01, 0.005, 0.02, -0.04, 0.001, 0.011, -0.002] * 25
    scores = [i * 0.000001 for i in range(len(forwards))]
    base = tail_profile(_rows(forwards), scores, cost=0.0065)
    scaled = tail_profile(_rows(forwards), [s * 1000.0 for s in scores],
                          cost=0.0065)
    assert base == scaled


def test_a_quantile_holding_less_than_one_bar_is_omitted_not_reported_as_zero():
    """20 rows cannot report a top-1%: 0.2 bars is not 0.0% forward return."""
    profile = tail_profile(_rows([0.01] * 20), list(range(20)), cost=0.0065)
    assert 0.01 not in profile and 0.02 not in profile
    assert profile[0.05][0] == 1
    assert profile[0.50][0] == 10


def test_an_empty_window_profiles_to_nothing():
    assert tail_profile([], [], cost=0.0065) == {}


def test_the_quantiles_are_ordered_tightest_last_and_include_the_top_percent():
    assert TAIL_QUANTILES == tuple(sorted(TAIL_QUANTILES, reverse=True))
    assert 0.01 in TAIL_QUANTILES, (
        "the 95th percentile was the old ceiling; the whole point is to look "
        "past it")
    assert max(TAIL_QUANTILES) <= 0.50
