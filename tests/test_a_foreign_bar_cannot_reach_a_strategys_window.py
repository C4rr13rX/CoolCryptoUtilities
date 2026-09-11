"""38.4% of 60-bar windows on the contaminated symbols held a foreign-regime bar.

`sample_arrays` is the one place every strategy in this package obtains its
window, and seventeen call sites read its extrema. Measured over the 7 days to
2026-09-11 (data/feed_regime_contamination_census.md): 22 of 198 streamed
symbols carry a second price regime, 323 of 43,920 ticks (0.74%) sit in it, and
because a 60-bar window is 60 chances to include one, 1,497 of 3,894 windows
(38.4%) hold at least one. `sanitize_model_price_window` fully repairs 1,410 of
those 1,497 (94.2%).

The loud failure -- CLANKER-USDC's `expected_return = 12551318.65` -- is bounded
in `make_candidate`. The quiet ones are what this file is for: a foreign HIGH
makes donchian_breakout's band unreachable so it silently stops entering, and a
seven-decade range drives stochastic_reversal's %K to ~0 so the symbol reads
permanently oversold. Neither trips any bound anyone would write.
"""

from __future__ import annotations

import numpy as np

from trading.strategies.base import sample_arrays


class _State:
    """RouteState-shaped: samples are (ts, price, volume), oldest first."""

    def __init__(self, prices):
        self.symbol = "CLANKER-USDC"
        self.samples = [
            (1_757_000_000.0 + i * 300.0, float(p), 1000.0)
            for i, p in enumerate(prices)
        ]


def _clanker(foreign_at=()):
    """CLANKER's real shape: ~1e-06, with 13.01897021 published at some index."""
    prices = [1.0e-06 + (i % 7) * 1.0e-09 for i in range(60)]
    for i in foreign_at:
        prices[i] = 13.01897021
    return prices


class TestTheForeignBarNeverReachesAConsumer:
    def test_the_three_clanker_ticks_are_repaired(self):
        ts, prices, vols = sample_arrays(_State(_clanker(foreign_at=(20, 21, 30))))
        assert prices.size == 60, "the window must keep its length"
        assert float(np.max(prices)) < 1.0e-05, (
            f"a foreign bar survived: max {float(np.max(prices))}"
        )

    def test_the_window_range_stops_being_seven_decades(self):
        """stochastic_reversal's %K denominator, and donchian's band."""
        _, dirty, _ = sample_arrays(_State(_clanker()))
        _, repaired, _ = sample_arrays(_State(_clanker(foreign_at=(20,))))
        clean_range = float(np.max(dirty) - np.min(dirty))
        assert float(np.max(repaired) - np.min(repaired)) <= clean_range * 1.5

    def test_the_timestamps_and_volumes_are_untouched(self):
        ts, _, vols = sample_arrays(_State(_clanker(foreign_at=(20,))))
        assert ts.size == 60 and vols.size == 60
        assert float(ts[1] - ts[0]) == 300.0
        assert float(np.min(vols)) == 1000.0


class TestACleanWindowIsUnchanged:
    """The repair must never be the reason a real move is not seen."""

    def test_a_clean_window_is_the_same_numbers(self):
        prices = _clanker()
        _, out, _ = sample_arrays(_State(prices))
        assert np.allclose(out, np.asarray(prices, dtype=np.float64))

    def test_a_real_large_move_survives(self):
        """A 40% move is a move, not contamination, and must not be flattened."""
        prices = [1.0e-06] * 40 + [1.4e-06] * 20
        _, out, _ = sample_arrays(_State(prices))
        assert float(np.max(out)) == 1.4e-06

    def test_an_empty_window_is_still_empty(self):
        ts, prices, vols = sample_arrays(_State([]))
        assert ts.size == 0 and prices.size == 0 and vols.size == 0


class TestTheRepairIsVisible:
    def test_the_count_is_recorded_on_the_state(self):
        """A serving path repairing every tick is an upstream feed bug."""
        state = _State(_clanker(foreign_at=(20, 21, 30)))
        sample_arrays(state)
        assert getattr(state, "window_rows_repaired", 0) == 3

    def test_a_clean_window_records_nothing(self):
        state = _State(_clanker())
        sample_arrays(state)
        assert getattr(state, "window_rows_repaired", 0) == 0
