"""A take equal to its stop has zero pre-cost expectancy, so it pays the fee.

The bug this prevents: the omen brain's target was labelled with a single
threshold -- ``omen_threshold()`` -- used as BOTH the move that counts as a
win and, implicitly, the move that counts as a loss. Three passes then spent
their effort raising the brain's recall on that target, from 89.2% to 98.7%,
and none of them produced held-out edge, because the target could not have
one: a barrier race on a driftless price is a martingale, its expectancy
before costs is exactly zero for EVERY (take, stop) pair, and a symmetric
pair therefore needs a 75-83% win rate purely to hand the round trip back.

Measured 2026-09-07 over 94,714 walks on ten five-minute corpora: 0 of 120
(take, stop, horizon) combinations had positive unconditional net, and a
symmetric take == stop == 1.30% race came out at p = 49.80% against the
50.00% the martingale identity predicts.

These tests fail against the pre-fix world, where nothing anywhere computed
the win rate a barrier pair requires.
"""
from __future__ import annotations

import math

import pytest

from trading.omen_path import (
    LONG, PATH_FLAT, PATH_STOP, PATH_WIN, barriers_are_payable,
    break_even_win_rate, indiscriminate_win_rate, max_bearable_cost,
    skill_required, walk_path,
)


def _bars(closes, highs=None, lows=None):
    """Minimal OHLC bars. Highs/lows default to the close (no intrabar range)."""
    highs = closes if highs is None else highs
    lows = closes if lows is None else lows
    return [{"timestamp": 1_700_000_000 + 300 * i, "open": c,
             "high": h, "low": l, "close": c}
            for i, (c, h, l) in enumerate(zip(closes, highs, lows))]


class TestSymmetricBarriersArePreCostZero:
    def test_a_symmetric_pair_needs_far_more_than_a_coin_flip(self):
        # take == stop == 0.975% -- exactly what omen_threshold() produces --
        # against the measured 0.65% round trip.
        needed = break_even_win_rate(0.00975, 0.00975, 0.0065)
        assert needed == pytest.approx(0.8333, abs=1e-3), (
            "a symmetric omen barrier at the live round trip needs an 83.3% "
            f"win rate, got {needed:.4%}")

    def test_indiscriminate_entry_on_a_symmetric_pair_is_a_coin_flip(self):
        assert indiscriminate_win_rate(0.00975, 0.00975) == pytest.approx(0.5)

    def test_a_symmetric_pair_has_exactly_zero_expectancy_before_cost(self):
        # p0 * take - (1 - p0) * stop must be 0 for every symmetric pair, or
        # the barriers themselves would be an edge and nobody would need a
        # brain at all.
        for width in (0.001, 0.0065, 0.00975, 0.02, 0.05):
            p0 = indiscriminate_win_rate(width, width)
            expectancy = p0 * width - (1.0 - p0) * width
            assert expectancy == pytest.approx(0.0, abs=1e-12), (
                f"take == stop == {width} should be a fair game, got "
                f"{expectancy}")

    def test_every_barrier_pair_is_pre_cost_zero_not_only_symmetric_ones(self):
        # The martingale identity is why widening the take does not create
        # an edge on its own -- it only lowers the SKILL required.
        for take, stop in ((0.039, 0.0033), (0.0065, 0.013), (0.02, 0.005)):
            p0 = indiscriminate_win_rate(take, stop)
            expectancy = p0 * take - (1.0 - p0) * stop
            assert expectancy == pytest.approx(0.0, abs=1e-12), (
                f"take {take} / stop {stop} should be pre-cost fair, got "
                f"{expectancy}")

    def test_the_fee_is_charged_on_the_loss_as_well_as_the_win(self):
        # The version of this arithmetic that drops the second cost term
        # gives (stop)/(take+stop) = 50% here and calls a symmetric pair a
        # coin flip. It is 83.3%.
        naive = 0.00975 / (0.00975 + 0.00975)
        real = break_even_win_rate(0.00975, 0.00975, 0.0065)
        assert real > naive + 0.30, (
            "forgetting that cost is paid on the losing leg understates the "
            f"bar by {real - naive:.2%}")


class TestSkillRequiredIsCostOverWidth:
    def test_skill_required_is_the_cost_divided_by_the_barrier_width(self):
        assert skill_required(0.00975, 0.00975, 0.0065) == pytest.approx(
            0.0065 / 0.0195)

    def test_it_equals_the_gap_between_break_even_and_random_entry(self):
        for take, stop, cost in ((0.00975, 0.00975, 0.0065),
                                 (0.039, 0.0033, 0.0065),
                                 (0.013, 0.0065, 0.001)):
            gap = (break_even_win_rate(take, stop, cost)
                   - indiscriminate_win_rate(take, stop))
            assert skill_required(take, stop, cost) == pytest.approx(gap)

    def test_widening_the_barriers_is_the_only_lever_that_is_not_the_cost(self):
        tight = skill_required(0.00975, 0.00975, 0.0065)
        wide = skill_required(0.065, 0.065, 0.0065)
        assert tight > 0.24, f"omen width should demand >24 points, got {tight:.2%}"
        assert wide < 0.06, f"a 13% width should demand <6 points, got {wide:.2%}"

    def test_a_cheaper_round_trip_lowers_the_skill_in_proportion(self):
        assert skill_required(0.02, 0.01, 0.0065) == pytest.approx(
            2.0 * skill_required(0.02, 0.01, 0.00325))


class TestMaxBearableCost:
    def test_it_inverts_the_break_even_rate(self):
        take, stop, cost = 0.02, 0.008, 0.0065
        p = break_even_win_rate(take, stop, cost)
        assert max_bearable_cost(p, take, stop) == pytest.approx(cost)

    def test_a_coin_flip_on_symmetric_barriers_bears_no_cost_at_all(self):
        # The exact statement of "geared to lose": at 50% on take == stop,
        # the largest fee the scheme can survive is zero.
        assert max_bearable_cost(0.5, 0.00975, 0.00975) == pytest.approx(0.0)

    def test_a_scheme_below_random_cannot_be_rescued_by_free_trading(self):
        assert max_bearable_cost(0.30, 0.00975, 0.00975) < 0.0


class TestTheGuardRefusesAnUnpayableScheme:
    def test_the_omens_own_barrier_at_its_demonstrated_skill_is_refused(self):
        # Held-out, measured 2026-09-07: 31.2% exact against a 31.2%
        # majority class -- zero demonstrated skill. Arming on that is the
        # behaviour this guard exists to stop.
        payable, reason = barriers_are_payable(0.00975, 0.00975, 0.312, 0.0065)
        assert payable is False
        assert "83.33%" in reason or "83.3" in reason, reason

    def test_a_take_inside_the_round_trip_is_refused_even_at_perfect_skill(self):
        payable, reason = barriers_are_payable(0.005, 0.005, 1.0, 0.0065)
        assert payable is False
        assert "even a win loses money" in reason

    def test_a_scheme_with_enough_demonstrated_skill_is_allowed(self):
        payable, reason = barriers_are_payable(0.00975, 0.00975, 0.90, 0.0065)
        assert payable is True
        assert "90.00%" in reason

    def test_the_reason_is_populated_on_success_too(self):
        _, reason = barriers_are_payable(0.05, 0.01, 0.95, 0.0065)
        assert reason.strip(), "a guard that is silent when it passes cannot be audited"

    def test_a_win_rate_that_is_not_a_fraction_is_refused_not_coerced(self):
        for bad in (1.5, -0.1, 83.3):
            payable, reason = barriers_are_payable(0.02, 0.01, bad, 0.0065)
            assert payable is False, f"{bad} should be refused"
            assert "not a fraction" in reason
        # 83.3 is the shape of the bug: a percentage passed where a fraction
        # belongs would otherwise arm every scheme unconditionally.

    def test_a_margin_raises_the_bar_rather_than_lowering_it(self):
        ok_without, _ = barriers_are_payable(0.00975, 0.00975, 0.84, 0.0065)
        ok_with, _ = barriers_are_payable(0.00975, 0.00975, 0.84, 0.0065,
                                          margin=0.05)
        assert ok_without is True and ok_with is False


class TestWalkPathMatchesTheMartingale:
    def test_a_win_fills_at_the_take_and_reports_when_it_happened(self):
        bars = _bars([100.0, 100.2, 101.5, 100.0, 100.0],
                     highs=[100.0, 100.3, 101.6, 100.1, 100.1],
                     lows=[100.0, 100.1, 100.4, 99.9, 99.9])
        outcome = walk_path(bars, 0, horizon_bars=4, take=0.01, stop=0.01)
        assert outcome is not None
        assert outcome.outcome == PATH_WIN
        assert outcome.bars_held == 2
        assert outcome.ret == pytest.approx(0.01)
        assert outcome.net(0.0065) == pytest.approx(0.0035)

    def test_a_stop_hit_before_the_target_is_a_loss_however_it_ends(self):
        # The endpoint label calls this a WIN: the close 4 bars on is +2%.
        # A real position with a stop was closed at bar 1 for -1%.
        bars = _bars([100.0, 99.0, 100.0, 101.0, 102.0],
                     highs=[100.0, 100.0, 100.5, 101.5, 102.5],
                     lows=[100.0, 98.5, 99.5, 100.5, 101.5])
        outcome = walk_path(bars, 0, horizon_bars=4, take=0.01, stop=0.01)
        assert outcome is not None
        assert outcome.outcome == PATH_STOP
        assert outcome.bars_held == 1
        assert outcome.net(0.0065) == pytest.approx(-0.0165)

    def test_both_barriers_in_one_bar_is_taken_as_the_loss_and_flagged(self):
        bars = _bars([100.0, 100.0, 100.0],
                     highs=[100.0, 102.0, 100.0],
                     lows=[100.0, 98.0, 100.0])
        outcome = walk_path(bars, 0, horizon_bars=2, take=0.01, stop=0.01)
        assert outcome is not None
        assert outcome.outcome == PATH_STOP
        assert outcome.ambiguous is True, (
            "an unknowable intrabar order must be marked, not silently "
            "resolved in our favour")

    def test_neither_barrier_closes_at_the_horizon_rather_than_never(self):
        bars = _bars([100.0] * 6)
        outcome = walk_path(bars, 0, horizon_bars=5, take=0.05, stop=0.05)
        assert outcome is not None
        assert outcome.outcome == PATH_FLAT
        assert outcome.bars_held == 5
        assert outcome.ret == pytest.approx(0.0)

    def test_a_missing_future_is_none_not_a_flat_outcome(self):
        bars = _bars([100.0, 100.0, 100.0])
        assert walk_path(bars, 1, horizon_bars=5, take=0.01, stop=0.01) is None, (
            "running off the end of the corpus must drop the sample, not "
            "teach the brain that the end of the file is a reason to hold")

    def test_a_corpus_without_high_and_low_still_walks_on_closes(self):
        bars = [{"timestamp": 1_700_000_000 + 300 * i, "close": c}
                for i, c in enumerate([100.0, 100.0, 102.0, 100.0])]
        outcome = walk_path(bars, 0, horizon_bars=3, take=0.01, stop=0.01)
        assert outcome is not None and outcome.outcome == PATH_WIN

    def test_the_direction_is_long_unless_asked_and_a_bad_one_raises(self):
        bars = _bars([100.0] * 4)
        assert walk_path(bars, 0, horizon_bars=3, take=0.01,
                         stop=0.01).direction == LONG
        with pytest.raises(ValueError):
            walk_path(bars, 0, horizon_bars=3, take=0.01, stop=0.01,
                      direction="sideways")

    def test_a_short_reads_the_barriers_the_other_way_round(self):
        bars = _bars([100.0, 98.0, 98.0],
                     highs=[100.0, 100.0, 98.5],
                     lows=[100.0, 97.5, 97.5])
        outcome = walk_path(bars, 0, horizon_bars=2, take=0.01, stop=0.01,
                            direction="short")
        assert outcome is not None
        assert outcome.outcome == PATH_WIN
        assert outcome.ret == pytest.approx(0.01)

    def test_returns_are_fractions_not_percentages_or_basis_points(self):
        bars = _bars([100.0, 100.0, 102.0],
                     highs=[100.0, 100.0, 102.0], lows=[100.0, 100.0, 100.0])
        outcome = walk_path(bars, 0, horizon_bars=2, take=0.01, stop=0.01)
        assert outcome is not None
        assert 0.0 < abs(outcome.ret) < 1.0, (
            f"ret {outcome.ret} is not a fraction of entry")
        assert math.isfinite(outcome.net(0.0065))
