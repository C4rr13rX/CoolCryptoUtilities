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

    def test_bstonk_entry_window_was_genuinely_healthy(self):
        """Honest about what this gate can and cannot catch.

        BSTONK-USDC booked the -8.39% that blocks live trading, but its ENTRY
        window was fine: 20 ticks at a 42s median, max gap 654s -- inside the
        calibrated budget. The hole that breached the stop (10.9min) opened
        AFTER entry, during the hold, where an entry-time gate cannot see it.

        So this asserts the entry was correctly allowed. The residual risk is
        real and lives in the exit path, not here; see
        test_pathological_hole_is_refused for what this gate does catch.
        """
        gaps = [42] * 19 + [654]
        self.assertTrue(
            _feed_is_dense_enough(_DB(gaps_sec=gaps), "BSTONK-USDC", "base")
        )

    def test_pathological_hole_is_refused(self):
        """BASEJUICE's 29.8min hole is beyond any stop's reach."""
        gaps = [120] * 19 + [1788]
        self.assertFalse(
            _feed_is_dense_enough(_DB(gaps_sec=gaps), "BASEJUICE-USDC", "base")
        )

    def test_healthy_jitter_is_not_refused(self):
        """Measured 2026-08-27: healthy symbols reach ~1000s max gap.

        A budget below this refuses every symbol permanently, which is a worse
        failure than the one being fixed.
        """
        for sym, mx in (("CBBTC", 624), ("AERO", 1052), ("MAMO", 770),
                        ("BASECAT", 567), ("BSTONK", 983)):
            gaps = [180] * 19 + [mx]
            self.assertTrue(
                _feed_is_dense_enough(_DB(gaps_sec=gaps), sym + "-USDC", "base"),
                "%s (max gap %ds) is a healthy feed and must not be refused"
                % (sym, mx),
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


class _FeedAwareDB(_DB):
    """``_DB`` that can also answer "was the rest of the feed ticking?".

    ``_DB`` deliberately does not implement ``feed_tick_times``. Every test
    above therefore exercises the fail-closed path, where a database that
    cannot answer leaves the guard exactly as strict as it was before.
    """

    def __init__(self, gaps_sec=None, count=None, feed_cadence=None,
                 feed_covers=1.0):
        super().__init__(gaps_sec=gaps_sec, count=count)
        self._cadence = feed_cadence
        self._covers = feed_covers

    def feed_tick_times(self, chain, *, since_ts, until_ts,
                        exclude_symbol=None, limit=5000):
        if not self._cadence:
            return []               # the whole pipeline was dark
        end = since_ts + (until_ts - since_ts) * self._covers
        ticks, t = [], since_ts + self._cadence
        while t < end:
            ticks.append(t)
            t += self._cadence
        return ticks


class OutageIsNotACoverageFailureTest(unittest.TestCase):
    """One dark pipeline is one event, not one failure per symbol.

    Measured 2026-09-05 09:30: the largest 1h gap for nine unrelated symbols
    all ENDED within a second of 09:11:42, spanning the production restart at
    09:00:03 -- AERO 2347s, CBETH 2421s, CBBTC 1387s, CBDOGE 1384s, BSTONK
    1489s, CBXRP 1488s, BASECAT 1389s, COMP 1410s, DAI 2407s. Charged per
    symbol, that refused 32 of 36 symbols, and because the window is an hour
    long it blocked every entry for the hour after each restart. The four that
    passed were simply too new to contain the outage.
    """

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {
            "ATF_STATIC_REQUIRE_DENSE_FEED": "1",
            "ATF_STATIC_MAX_MEDIAN_TICK_GAP_SEC": "300",
            "ATF_STATIC_MIN_TICKS_FOR_ENTRY": "6",
        })
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_a_pipeline_wide_outage_does_not_condemn_the_symbol(self):
        """Nothing ticked anywhere, so the gap says nothing about this symbol."""
        gaps = [30] * 20 + [1400] + [30] * 20
        self.assertTrue(
            _feed_is_dense_enough(
                _FeedAwareDB(gaps_sec=gaps, feed_cadence=None),
                "AERO-USDC", "base",
            )
        )

    def test_a_hole_while_the_feed_was_healthy_is_still_refused(self):
        """The guard's real target: this symbol lost coverage, nothing else did."""
        gaps = [30] * 20 + [1400] + [30] * 20
        self.assertFalse(
            _feed_is_dense_enough(
                _FeedAwareDB(gaps_sec=gaps, feed_cadence=60.0),
                "LONELY-USDC", "base",
            )
        )

    def test_only_the_feed_up_portion_of_a_hole_is_charged(self):
        """A hole that is mostly outage is scored on the part that was not."""
        gaps = [30] * 20 + [1400] + [30] * 20
        self.assertTrue(
            _feed_is_dense_enough(
                _FeedAwareDB(gaps_sec=gaps, feed_cadence=60.0, feed_covers=0.25),
                "AERO-USDC", "base",
            )
        )

    def test_basejuice_is_still_refused_when_the_feed_was_up(self):
        """The calibration case must survive the outage carve-out."""
        gaps = [120] * 19 + [1788]
        self.assertFalse(
            _feed_is_dense_enough(
                _FeedAwareDB(gaps_sec=gaps, feed_cadence=60.0),
                "BASEJUICE-USDC", "base",
            )
        )

    def test_a_db_that_cannot_answer_stays_as_strict_as_before(self):
        """Fail closed: unknown feed liveness must not soften the guard."""
        gaps = [30] * 20 + [40000] + [30] * 20
        self.assertFalse(
            _feed_is_dense_enough(_DB(gaps_sec=gaps), "OK-USDC", "base")
        )


class _CoverageDB(_DB):
    """A symbol that ticks only in the last ``covers_sec`` of the window.

    The rest of the feed ticks steadily across the whole window, so this
    isolates coverage from every other property the gate measures: the
    symbol's own median gap, worst hole and staleness are all healthy by
    construction, and only the share of live minutes it appears in varies.
    """

    def __init__(self, covers_sec, cadence=60.0, window=3600.0):
        super().__init__()
        self._covers_sec = covers_sec
        self._cadence = cadence
        self._window = window

    def recent_market_prices(self, symbol, chain, *, since_ts=None, limit=200):
        now = time.time()
        n = int(self._covers_sec // self._cadence)
        return [(1.0, now - i * self._cadence) for i in range(n)]

    def feed_tick_times(self, chain, *, since_ts, until_ts,
                        exclude_symbol=None, limit=5000):
        ticks, t = [], since_ts
        while t < until_ts:
            ticks.append(t)
            t += self._cadence
        return ticks


class BurstyCoverageIsRefusedTest(unittest.TestCase):
    """The BPAD-USDC hole, and the three tests that all missed it.

    Live entry 2026-09-05 17:48:02. In the trailing hour BPAD had 22 ticks at
    a 1.3s median gap -- but 20 of them were a single 90-second burst, and the
    hour held a 2864s hole. The median described the burst. The hole escalated
    to ``_symbol_specific_hole``, which scores the longest run of CONSECUTIVE
    live buckets; the feed delivered 426 ticks inside that hole but they fell
    in only 12 of its 47 minutes, so the longest unbroken run was 300s against
    a 1200s budget and the entry was allowed.

    The position then went 1678s unpriced and the first tick after the hole
    fired the stop at -19.21% against a 1.5% stop: -0.2549 on a $1.50 clip,
    larger than the whole live book (18 round trips, -0.1864, or +0.0686
    without it).

    Coverage measured at the entry instant of all 18 live round trips put BPAD
    at 27.3%, in a cluster where four trades of four lost money and which is
    separated from the rest by the distribution's widest gap (27.8% -> 34.1%).
    """

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {
            "ATF_STATIC_REQUIRE_DENSE_FEED": "1",
            "ATF_STATIC_MAX_MEDIAN_TICK_GAP_SEC": "300",
            "ATF_STATIC_MIN_TICKS_FOR_ENTRY": "6",
            "ATF_STATIC_MIN_FEED_COVERAGE": "0.30",
        })
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_a_symbol_seen_in_a_quarter_of_live_minutes_is_refused(self):
        """BPAD's shape: healthy median, healthy hole, no coverage."""
        db = _CoverageDB(covers_sec=900.0)      # 15 of 60 live minutes = 25%
        self.assertFalse(_feed_is_dense_enough(db, "BPAD-USDC", "base"))

    def test_a_fully_covered_symbol_is_accepted(self):
        db = _CoverageDB(covers_sec=3600.0)     # 60 of 60 = 100%
        self.assertTrue(_feed_is_dense_enough(db, "AERO-USDC", "base"))

    def test_the_refusal_is_coverage_and_not_some_other_test(self):
        """Everything else about the sparse case is healthy."""
        db = _CoverageDB(covers_sec=900.0)
        with mock.patch.dict(os.environ, {"ATF_STATIC_MIN_FEED_COVERAGE": "0"}):
            self.assertTrue(
                _feed_is_dense_enough(db, "BPAD-USDC", "base"),
                "median gap, worst hole and staleness must all pass here, so a "
                "refusal with the coverage test off would mean this fixture is "
                "proving something other than what it claims",
            )

    def test_coverage_cannot_cause_a_blackout_when_it_cannot_be_measured(self):
        """Fails OPEN, unlike the tests above it.

        The three earlier checks have already passed by the time coverage is
        read. A database that cannot report feed liveness must leave the gate
        exactly as strict as it was, never stricter -- refusing every symbol
        because a measurement failed is the blackout this repo has already
        shipped once, when one outage was charged to 32 of 36 symbols.
        """
        db = _CoverageDB(covers_sec=900.0)
        db.feed_tick_times = mock.Mock(side_effect=RuntimeError("db down"))
        self.assertTrue(_feed_is_dense_enough(db, "BPAD-USDC", "base"))


if __name__ == "__main__":
    unittest.main()
