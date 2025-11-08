from __future__ import annotations

import numpy as np

from trading.opportunity import OpportunityTracker


def test_opportunity_tracker_flags_extremes() -> None:
    tracker = OpportunityTracker(min_points=20, zscore_threshold=1.0)
    prices = np.linspace(100, 110, 40)
    signal = tracker.evaluate("ETH-USDC", prices)
    assert signal is not None
    assert signal.kind == "sell-high"
    assert signal.zscore > 0


def test_opportunity_tracker_requires_enough_points() -> None:
    tracker = OpportunityTracker(min_points=50)
    prices = np.linspace(10, 11, 10)
    signal = tracker.evaluate("ETH-USDC", prices)
    assert signal is None
