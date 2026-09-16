"""A dip that lasts longer than the consumer's memory must keep re-arming.

TWO HALVES THAT DISAGREED ABOUT LIFETIME.

  producer  trading/opportunity.py   -- emits a buy-low signal, then
            suppresses same-kind successors whose z-score has not moved by
            0.2. The reference point is written only on emit, so it never
            decayed.

  consumer  trading/scheduler.py:1505-1506 -- DELETES its copy of that
            signal once it is 1800s old, and trading/scheduler.py:1665
            refuses the money_button dip candidate outright when the copy
            is missing.

A mean-reverting 2-5% drawdown holds its z-score roughly still for as long
as it lasts. Under the unbounded debounce that is the worst case, not the
best: the tracker emits once, the scheduler forgets 30 minutes later, and
the producer declines to say the thing that is still true. The rule is then
refused for a reason with nothing to do with the price.

This test pins the repaired contract: while the condition holds, the
producer re-emits at least as often as the consumer forgets.
"""
from __future__ import annotations

import numpy as np

from trading.opportunity import OpportunityTracker


def _dip_series(n: int = 60) -> np.ndarray:
    """Flat history with the last print pushed well below the mean."""
    series = np.full(n, 100.0, dtype=np.float64)
    series[-8:] = [100.0, 99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 97.0]
    return series


def test_a_sustained_dip_re_arms_after_the_consumer_forgets(monkeypatch):
    tracker = OpportunityTracker()
    series = _dip_series()

    clock = {"now": 1_000_000.0}
    monkeypatch.setattr("trading.opportunity.time.time", lambda: clock["now"])

    first = tracker.evaluate("VVV-USDC", series)
    assert first is not None, "the dip should produce a signal at all"
    assert first.kind == "buy-low"

    # Same condition, one tick later: suppressing this IS the point of the
    # debounce, and it must still happen.
    clock["now"] += 30.0
    assert tracker.evaluate("VVV-USDC", series) is None

    # Still the same condition, but now past the window in which the
    # scheduler keeps the bias (scheduler.py:1505). The consumer has
    # nothing; the producer must supply it again.
    clock["now"] += tracker.refresh_sec + 1.0
    again = tracker.evaluate("VVV-USDC", series)
    assert again is not None, (
        "the dip is still true and the consumer has forgotten it -- "
        "the tracker must re-emit rather than stay muted"
    )
    assert again.kind == "buy-low"
    assert again.timestamp == clock["now"]


def test_the_debounce_still_suppresses_inside_the_window(monkeypatch):
    """The repair must not turn the tracker into a per-tick firehose."""
    tracker = OpportunityTracker()
    series = _dip_series()

    clock = {"now": 2_000_000.0}
    monkeypatch.setattr("trading.opportunity.time.time", lambda: clock["now"])

    assert tracker.evaluate("DRB-USDC", series) is not None
    emitted = 0
    for _ in range(50):
        clock["now"] += 20.0          # 1000s total, inside the 1800s window
        if tracker.evaluate("DRB-USDC", series) is not None:
            emitted += 1
    assert emitted == 0, f"debounce leaked {emitted} duplicate signals inside its own window"


def test_a_changed_zscore_still_re_emits_immediately(monkeypatch):
    """Movement, not age, is still the primary reason to re-emit."""
    tracker = OpportunityTracker()

    clock = {"now": 3_000_000.0}
    monkeypatch.setattr("trading.opportunity.time.time", lambda: clock["now"])

    shallow = _dip_series()
    assert tracker.evaluate("ZORA-USDC", shallow) is not None

    deeper = _dip_series()
    deeper[-1] = 90.0                 # the dip deepens sharply
    clock["now"] += 5.0
    moved = tracker.evaluate("ZORA-USDC", deeper)
    assert moved is not None, "a materially different z-score must not be debounced"
    assert abs(moved.zscore) > 1.5
