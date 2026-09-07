"""A forecast horizon is wall clock, and expressing it in bars hid a 6x spread.

`scripts/omen_generalisation.py` took `--horizons` in BARS and selected its
corpus by FILE SIZE. The corpus mixes cadences -- 300s, 598s, 600s, 612s, 618s,
620s, 654s, 732s, 3600s -- and the largest files are the hourly ones, so
`--symbols 10` picked a set spanning 598s to 3600s. Every omen number this
repo produced was therefore a trade-weighted average of a 119.6-minute forecast
on cbBTC and a 720.0-minute forecast on SHIB, reported as one row labelled
"h=12".

That is the same class of defect as the 20-step window fed to a 60-step model:
a quantity crossed a boundary without its unit. These tests pin the unit at the
boundary. Against the old code -- which passed `--horizons` straight through as
a bar count -- `test_the_same_horizon_is_the_same_wall_clock_on_every_cadence`
and `test_the_hourly_corpus_cannot_answer_a_mandate_horizon` both fail, because
there was no conversion to fail on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_generalisation import bar_seconds, horizon_bars  # noqa: E402


#: The exact cadences the corpus selector actually returned for `--symbols 10`,
#: measured 2026-09-07 over data/historical_ohlcv/base.
CORPUS_CADENCES = (598, 618, 600, 598, 620, 732, 654, 612, 3600, 3600)


def test_the_same_horizon_is_the_same_wall_clock_on_every_cadence():
    """20 minutes is 20 minutes whether the bars are 5 minutes or an hour."""
    for cadence in (60, 300, 598, 600, 900, 3600):
        bars = horizon_bars(20, cadence)
        realised_minutes = bars * cadence / 60.0
        # Within one bar of the request -- the rounding error, and nothing more.
        assert abs(realised_minutes - 20) <= cadence / 60.0, (
            f"{cadence}s cadence realised {realised_minutes:.1f} min for a "
            f"20-minute horizon")


def test_a_bar_count_is_six_different_forecasts_across_this_corpus():
    """The bug itself: one bar count, six wall clocks, averaged into one row."""
    realised = {12 * cadence / 60.0 for cadence in CORPUS_CADENCES}
    assert max(realised) / min(realised) > 5.9, (
        "the corpus no longer spans the cadences that made this a bug; if that "
        "is real, re-measure rather than deleting the test")
    # And the conversion collapses that spread to nothing.
    converted = {horizon_bars(120, c) * c / 60.0 for c in CORPUS_CADENCES}
    assert max(converted) / min(converted) < 1.10


def test_the_hourly_corpus_cannot_answer_a_mandate_horizon():
    """R3V3N!R trades 5-30 minute round trips; hourly bars cannot see them.

    On 3600s bars every horizon under 90 minutes collapses to one bar, so an
    hourly symbol answers "10 minutes" and "60 minutes" with the identical
    forecast. That is why the cadence filter exists and why a minute-scale run
    must exclude hourly data rather than silently round it up.
    """
    assert horizon_bars(10, 3600) == horizon_bars(60, 3600) == 1
    # A 5-minute corpus separates them.
    assert horizon_bars(10, 300) == 2
    assert horizon_bars(60, 300) == 12
    assert len({horizon_bars(m, 300) for m in (10, 15, 20, 30, 60)}) == 5


def test_a_horizon_never_rounds_to_zero_bars():
    """A zero-bar horizon compares a close against itself: a free, fake edge."""
    for minutes in (1, 2, 5, 10):
        assert horizon_bars(minutes, 3600) >= 1
    assert horizon_bars(0.0001, 86400) == 1


def test_a_non_positive_cadence_is_refused_not_divided_by():
    for bad in (0, -300):
        with pytest.raises(ValueError):
            horizon_bars(20, bad)


def test_bar_seconds_reads_the_median_gap_not_the_first():
    """One missing candle must not redefine the cadence of a whole symbol."""
    bars = [{"timestamp": 1000 + i * 300, "close": 1.0} for i in range(60)]
    del bars[10]  # a gap of 600s in the middle of a 300s series
    assert bar_seconds(bars) == 300
