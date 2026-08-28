"""A stop-loss is only as good as the feed that triggers it.

The stop is evaluated when a tick ARRIVES. On a sparse feed the price gaps
straight past it, so the realised loss is unbounded no matter what the stop is
set to.

Measured 2026-08-27 on real ghost trades: 4 of 6 stop_loss exits breached an 8%
stop, losing 11.2%, 11.3% and 22.2%. SOL-USDC -- the worst, at -22.19% -- had
14 ticks total and a 663-minute hole between them. That single exit pushed tail
risk to 0.095 against a 0.08 guardrail, which was the only thing blocking live
trading (ghost_validation_block, reason=tail_risk).

Entering a position whose downside cannot be bounded is not a risk being
chosen, it is one that cannot be seen. Refuse it.
"""

from __future__ import annotations

import os
import time
import unittest
from unittest import mock

from services.atf_static_strategy import _feed_is_dense_enough


class _DB:
    """Returns (price, ts) rows the way db.recent_market_prices does."""

    def __init__(self, gaps_sec=None, count=None, raise_on_call=False):
        self._gaps = gaps_sec
        self._count = count
        self._raise = raise_on_call

    def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=200):
        if self._raise:
            raise RuntimeError("db down")
        now = time.time()
        if self._gaps is not None:
            stamps, t = [], now
            for gap in reversed(self._gaps):
                stamps.append(t)
                t -= gap
            return [(1.0, s) for s in sorted(stamps)]
        return [(1.0, now - i * 60.0) for i in range(self._count or 0)]


class FeedDensityGateTest(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {
            "ATF_STATIC_REQUIRE_DENSE_FEED": "1",
            "ATF_STATIC_MAX_MEDIAN_TICK_GAP_SEC": "300",
            "ATF_STATIC_MIN_TICKS_FOR_ENTRY": "6",
        })
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_dense_feed_is_accepted(self):
        """One tick a minute can enforce a stop."""
        self.assertTrue(_feed_is_dense_enough(_DB(count=30), "AERO-USDC", "base"))

    def test_sparse_feed_is_refused(self):
        """The SOL case: ~10 minutes between ticks."""
        self.assertFalse(
            _feed_is_dense_enough(_DB(gaps_sec=[600] * 10), "SOL-USDC", "base")
        )

    def test_too_few_ticks_is_refused(self):
        """A handful of ticks is not a feed, whatever the spacing."""
        self.assertFalse(_feed_is_dense_enough(_DB(count=3), "THIN-USDC", "base"))

    def test_one_long_hole_disqualifies_an_otherwise_good_feed(self):
        """A single hole is exactly what breaches the stop.

        This assertion was originally the opposite -- a lone outage was
        allowed on the reasoning that it should not veto solid coverage.
        Measured 2026-08-28, that reasoning was wrong: every stop_loss breach
        in the ledger had healthy entry-time density and one long hole during
        the hold (BSTONK 10.9min, BASEJUICE 29.8min, BASECAT 9.6min). The
        median test passed all three. Those three trades were the entire tail
        that blocked live trading.
        """
        gaps = [30] * 20 + [40000] + [30] * 20
        self.assertFalse(_feed_is_dense_enough(_DB(gaps_sec=gaps), "OK-USDC", "base"))

    def test_short_hole_within_budget_is_tolerated(self):
        """The bound is the stop's survivable hole, not zero variance."""
        gaps = [30] * 20 + [420] + [30] * 20
        self.assertTrue(_feed_is_dense_enough(_DB(gaps_sec=gaps), "OK-USDC", "base"))

    def test_bstonk_hold_gap_is_refused(self):
        """The exact shape that booked -8.39% and blocked live trading."""
        gaps = [42] * 19 + [654]
        self.assertFalse(
            _feed_is_dense_enough(_DB(gaps_sec=gaps), "BSTONK-USDC", "base")
        )

    def test_stale_feed_is_refused_even_when_history_is_dense(self):
        """Dense an hour ago is not dense now."""

        class _Stale:
            def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=200):
                now = time.time()
                # 30 tight ticks that all stopped 20 minutes ago.
                return [(1.0, now - 1200.0 - i * 30.0) for i in range(30)]

        self.assertFalse(_feed_is_dense_enough(_Stale(), "STALE-USDC", "base"))

    def test_hole_budget_is_configurable(self):
        gaps = [30] * 20 + [40000] + [30] * 20
        with mock.patch.dict(os.environ, {"ATF_STATIC_MAX_TICK_HOLE_SEC": "0"}):
            self.assertTrue(
                _feed_is_dense_enough(_DB(gaps_sec=gaps), "OK-USDC", "base")
            )

    def test_consistently_thin_feed_is_refused(self):
        gaps = [900] * 12
        self.assertFalse(_feed_is_dense_enough(_DB(gaps_sec=gaps), "THIN-USDC", "base"))

    def test_boundary_gap_is_accepted(self):
        gaps = [300] * 12
        self.assertTrue(_feed_is_dense_enough(_DB(gaps_sec=gaps), "EDGE-USDC", "base"))

    def test_db_failure_refuses_rather_than_assumes(self):
        """Unknown coverage must not be treated as good coverage."""
        self.assertFalse(
            _feed_is_dense_enough(_DB(raise_on_call=True), "X-USDC", "base")
        )

    def test_empty_feed_is_refused(self):
        self.assertFalse(_feed_is_dense_enough(_DB(count=0), "X-USDC", "base"))

    def test_gate_can_be_disabled_explicitly(self):
        """An operator opt-out exists, but it must be deliberate."""
        with mock.patch.dict(os.environ, {"ATF_STATIC_REQUIRE_DENSE_FEED": "0"}):
            self.assertTrue(_feed_is_dense_enough(_DB(count=0), "X-USDC", "base"))


if __name__ == "__main__":
    unittest.main()
