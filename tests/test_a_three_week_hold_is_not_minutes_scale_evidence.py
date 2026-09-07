"""A round trip held for weeks must not count as evidence for a minutes-scale bot.

THE BUG, measured 2026-09-07 by replaying the 30-day ghost book at the $6 live
clip through ``services.roundtrip_cost`` and splitting on
``trading.strategies.ledger._live_tradeable``:

    live-tradeable round trips           312   net  +7.157
      of which held longer than 4h        20   net  +7.938
      held inside 4h                     292   net  -0.779

Six percent of the rows carried the entire positive case for putting real money
behind a strategy, and they were the rows whose holding period had nothing to do
with the horizon the strategy is graded on. The single worst was CBBTC-USDC at
+22.20% held 30,617 minutes -- 21.3 days, against a ``MAX_HOLD_SECONDS`` of
3600. That one row IS ``atf_static``/CBBTC's whole +0.6705, and it made that the
top-ranked live-tradeable pair in the system; without it the pair is -0.6384
over 40 trades. On tradeable symbols ``atf_static`` reads +1.0596 over 181
trades with all holds, and -0.9080 over 175 once the six long ones are removed.

``_is_implausible`` could not see any of it, because it bounds an outcome in
DOLLARS and this artifact is in TIME: a 21-day drift on cbBTC nets $1.31 at the
$6 clip, under the $2.00 absolute cap and nowhere near 25x the strategy's own
scale.

These tests fail against the old behaviour: before ``held_sec`` existed,
``record()`` accepted every one of these outcomes and the ledger counted them.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from trading.strategies import ledger as ledger_mod
from trading.strategies.ledger import StrategyLedger


class ThreeWeekHoldIsNotEvidence(unittest.TestCase):
    def setUp(self):
        # Isolate the ledger path. `record()` mirrors into the lifetime
        # registry only when the path is the DEFAULT one, so a temp path also
        # keeps these fixtures out of data/strategy_registry.json -- the exact
        # leak tests/test_ledger_rejects_artifacts.py documents.
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ledger.json"
        self.addCleanup(self._tmp.cleanup)
        self._env = mock.patch.dict(os.environ, {"MAX_HOLD_SECONDS": "3600"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _ledger(self) -> StrategyLedger:
        return StrategyLedger(path=str(self.path))

    def _trades(self, led: StrategyLedger, sid: str) -> int:
        return int(((led._entry(sid).get("ghost") or {}).get("trades") or 0))

    # -- the horizon itself ------------------------------------------------

    def test_the_horizon_is_derived_from_the_hold_timer_not_a_fresh_number(self):
        """The bar must move with MAX_HOLD_SECONDS, or the two drift apart."""
        with mock.patch.dict(os.environ, {"MAX_HOLD_SECONDS": "3600"}):
            self.assertAlmostEqual(ledger_mod._max_evidence_hold_sec(),
                                   3600.0 * ledger_mod._MAX_HOLD_MULTIPLE)
        with mock.patch.dict(os.environ, {"MAX_HOLD_SECONDS": "900"}):
            self.assertAlmostEqual(ledger_mod._max_evidence_hold_sec(),
                                   900.0 * ledger_mod._MAX_HOLD_MULTIPLE)

    def test_a_disabled_multiple_switches_the_check_off_rather_than_banning_all(self):
        with mock.patch.object(ledger_mod, "_MAX_HOLD_MULTIPLE", 0.0):
            self.assertFalse(ledger_mod._exceeds_evidence_horizon(21 * 86400.0))

    # -- the measured row --------------------------------------------------

    def test_the_cbbtc_row_held_21_days_is_refused(self):
        """The exact trade that carried the whole tradeable book.

        +22.20% on cbBTC held 30,617 minutes. Its USD outcome at the $6 clip is
        +1.31, which passes `_is_implausible` -- so if the horizon check is
        absent, this is recorded.
        """
        led = self._ledger()
        led.record("atf_static", profit=1.3089, mode="ghost",
                   symbol="CBBTC-USDC", held_sec=30617.5 * 60.0,
                   mirror_registry=False)
        self.assertEqual(self._trades(led, "atf_static"), 0,
                         "a 21.3-day round trip was counted as evidence for a "
                         "strategy graded on a 3600s horizon")

    def test_that_same_row_passes_the_dollar_guard_so_only_time_can_catch_it(self):
        """Proves the two guards are independent, not one wearing two names.

        If `_is_implausible` already rejected +1.3089 the new check would be
        redundant and this whole finding would be a duplicate. It does not.
        """
        self.assertFalse(
            ledger_mod._is_implausible(1.3089, relative_to=None),
            "the size guard would have caught it; the time guard is redundant",
        )
        self.assertTrue(ledger_mod._exceeds_evidence_horizon(30617.5 * 60.0))

    # -- what must still get through ---------------------------------------

    def test_a_normal_round_trip_inside_the_horizon_is_still_recorded(self):
        """A gate that refuses everything is switched off, not safe.

        The median hold in the same 30-day book is 1.01h -- comfortably inside
        the 4h bar -- so the ordinary case must be untouched.
        """
        led = self._ledger()
        led.record("atf_static", profit=0.031, mode="ghost",
                   symbol="AERO-USDC", held_sec=3630.0, mirror_registry=False)
        self.assertEqual(self._trades(led, "atf_static"), 1)

    def test_an_outcome_with_no_known_hold_is_recorded_exactly_as_before(self):
        """`held_sec` is new; most historical callers cannot supply it.

        Rejecting on its absence would discard the whole book to catch 6% of
        it. Unknown must fail OPEN here -- same shape as `pl_ref`/`dd_ref`.
        """
        led = self._ledger()
        led.record("atf_static", profit=0.031, mode="ghost",
                   symbol="AERO-USDC", mirror_registry=False)
        led.record("atf_static", profit=0.021, mode="ghost",
                   symbol="AERO-USDC", held_sec=None, mirror_registry=False)
        self.assertEqual(self._trades(led, "atf_static"), 2)

    def test_an_unparseable_hold_does_not_silently_delete_the_trade(self):
        led = self._ledger()
        for bad in ("", "not-a-number", float("nan")):
            led.record("s", profit=0.01, mode="ghost", symbol="AERO-USDC",
                       held_sec=bad, mirror_registry=False)
        self.assertEqual(self._trades(led, "s"), 3)

    # -- direction: this must not launder a losing record ------------------

    def test_a_long_hold_LOSS_is_refused_too_so_the_filter_cannot_launder(self):
        """The failure mode `_is_implausible` documents, checked from the front.

        A filter that only removed long-held LOSSES would improve every record
        it judges -- which is how a guard against fiction becomes a way to
        launder a losing book. The bar is the horizon, not the sign.
        """
        led = self._ledger()
        led.record("s", profit=-0.9, mode="ghost", symbol="AERO-USDC",
                   held_sec=20 * 86400.0, mirror_registry=False)
        led.record("s", profit=+0.9, mode="ghost", symbol="AERO-USDC",
                   held_sec=20 * 86400.0, mirror_registry=False)
        self.assertEqual(self._trades(led, "s"), 0)

    def test_the_refused_outcome_never_reaches_the_lifetime_registry(self):
        """An outcome that was never evidence must not enter the append-only book.

        The registry is the ledger's only independent check, so a row refused
        here has to be refused there in the same call -- which means the check
        must sit AHEAD of the mirror, beside the size guard.
        """
        calls = []
        with mock.patch("services.strategy_registry.record_outcome",
                        side_effect=lambda *a, **k: calls.append((a, k))):
            led = self._ledger()
            led.record("atf_static", profit=1.3089, mode="ghost",
                       symbol="CBBTC-USDC", held_sec=30617.5 * 60.0)
        self.assertEqual(calls, [],
                         "a refused outcome was still written to the lifetime "
                         "registry, where nothing can take it back")

    # -- the re-arm path is what this actually protects ---------------------

    def test_a_long_hold_cannot_supply_the_fresh_evidence_a_re_arm_needs(self):
        """The reason this matters for money.

        `_maybe_rearm_locked` wants 20 fresh live-tradeable ghost round trips
        before a demoted strategy may spend real money again. Nineteen honest
        ones plus a 21-day mark-out is not twenty.
        """
        led = self._ledger()
        for _ in range(19):
            led.record("atf_static", profit=0.02, mode="ghost",
                       symbol="AERO-USDC", held_sec=1800.0,
                       mirror_registry=False)
        led.record("atf_static", profit=0.02, mode="ghost",
                   symbol="AERO-USDC", held_sec=21 * 86400.0,
                   mirror_registry=False)
        sub = ((led._entry("atf_static").get("ghost") or {}).get("tradeable") or {})
        self.assertEqual(int(sub.get("trades") or 0), 19,
                         "the 20th trade toward a live licence was a 21-day "
                         "drift, not a decision this strategy made")


class ProductionCallSitesPassTheHoldingPeriod(unittest.TestCase):
    """A guard nothing calls is not a guard.

    Both production writers must supply `held_sec`, or every outcome arrives
    unknown and the check above fails open on 100% of the book. Asserts on the
    CODE, because this is exactly the kind of thing a passing unit test on the
    ledger alone would miss.
    """

    def test_the_bot_exit_path_passes_a_holding_period(self):
        src = Path("trading/bot.py").read_text(encoding="utf-8")
        head = src.index("self.strategy_ledger.record(")
        call = src[head:head + 2000]
        self.assertIn("held_sec=", call,
                      "trading/bot.py exits without telling the ledger how "
                      "long the position was held")

    def test_the_scout_exit_path_passes_a_holding_period(self):
        src = Path("services/atf_static_strategy.py").read_text(encoding="utf-8")
        self.assertIn("held_sec=age", src,
                      "services/atf_static_strategy.py publishes age_sec into "
                      "trading_ops but not into the ledger that gates money")

    def test_the_scout_hands_over_the_same_number_it_publishes(self):
        """`age_sec` in trading_ops and `held_sec` in the ledger must be one value.

        Two clocks for one quantity is how this repo shipped a fee in the wrong
        currency and gas priced in the traded pair.
        """
        src = Path("services/atf_static_strategy.py").read_text(encoding="utf-8")
        self.assertIn('"age_sec": age,', src)
        self.assertIn("held_sec=age", src)


if __name__ == "__main__":
    unittest.main()
