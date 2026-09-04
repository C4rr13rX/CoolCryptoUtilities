"""A losing streak is not a verdict; the account balance is.

Measured 2026-09-03: ``atf_static`` -- the only strategy that had ever traded
live -- was demoted for "2 consecutive live losses", with
``STRATEGY_DEMOTE_MAX_LIVE_LOSSES=2`` in .env against a code default of 4. Both
losses were the CBETH exits that sold only 0.000162 and 0.000111 CBETH against
roughly 0.00026 held: our own exit-sizing bug booking losses the market never
produced.

The consequence was total. That demotion left EVERY strategy in
``data/strategy_ledger.json`` at ``live_approved=False``, so nothing could
trade at all, and the burst of rapid profitable swapping never came back on
its own -- the path check read "no strategy approved for live" for three
straight passes.

A strategy can lose four small trades, win one larger one, and still be ahead.
Demoting it there discards a winner for the shape of its variance rather than
its result. So the streak still guards against a run of real losses, but it
cannot fire while the strategy is net positive on live money.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.strategies.ledger import StrategyLedger  # noqa: E402


class DemotionNeedsANetLossTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "ledger.json")

    def _approved(self, sid="atf_static"):
        led = StrategyLedger(self.path)
        ent = led._entry(sid)
        ent["live_approved"] = True
        return led, sid, ent

    def test_a_streak_while_net_positive_does_not_demote(self):
        """The exact case that locked the account out of trading."""
        led, sid, ent = self._approved()
        ent["live"].update({"trades": 5, "consecutive_losses": 2,
                            "total_profit": 0.0051})
        with mock.patch.dict(os.environ, {"STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "2"}):
            led._evaluate_demotion_locked(sid)
        self.assertTrue(
            led._entry(sid)["live_approved"],
            "a strategy that is UP on real money must keep trading",
        )

    def test_a_streak_while_net_negative_does_demote(self):
        """The guard still works when the account is actually shrinking."""
        led, sid, ent = self._approved()
        ent["live"].update({"trades": 5, "consecutive_losses": 2,
                            "total_profit": -0.25})
        with mock.patch.dict(os.environ, {"STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "2"}):
            led._evaluate_demotion_locked(sid)
        self.assertFalse(led._entry(sid)["live_approved"])
        self.assertIn("net", str(led._entry(sid).get("demote_reason") or ""))

    def test_breaking_even_on_a_streak_still_demotes(self):
        """Zero is not growth, and the round trip costs fees to reach it."""
        led, sid, ent = self._approved()
        ent["live"].update({"trades": 5, "consecutive_losses": 3,
                            "total_profit": 0.0})
        with mock.patch.dict(os.environ, {"STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "2"}):
            led._evaluate_demotion_locked(sid)
        self.assertFalse(led._entry(sid)["live_approved"])

    def test_a_short_streak_alone_does_not_demote(self):
        """One loss is not a streak, and the sample is too small to judge.

        The min-sample guard is held above the trade count on purpose here:
        .env sets STRATEGY_DEMOTE_MIN_LIVE_TRADES=3 against a code default of
        8, so on cent-sized trades a strategy gets judged on net P/L after
        three results, which is noise rather than evidence. That is a second,
        independent route to the same total lockout.
        """
        led, sid, ent = self._approved()
        ent["live"].update({"trades": 3, "consecutive_losses": 1,
                            "total_profit": -0.01})
        with mock.patch.dict(os.environ, {
                "STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "4",
                "STRATEGY_DEMOTE_MIN_LIVE_TRADES": "8"}):
            led._evaluate_demotion_locked(sid)
        self.assertTrue(led._entry(sid)["live_approved"])

    def test_net_loss_over_a_full_sample_still_demotes_without_a_streak(self):
        """The profitability rule is untouched: alternating losers still go."""
        led, sid, ent = self._approved()
        ent["live"].update({"trades": 10, "consecutive_losses": 0,
                            "total_profit": -0.30})
        with mock.patch.dict(os.environ, {
                "STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "4",
                "STRATEGY_DEMOTE_MIN_LIVE_TRADES": "8"}):
            led._evaluate_demotion_locked(sid)
        self.assertFalse(
            led._entry(sid)["live_approved"],
            "a strategy that is down over a fair sample must still be demoted",
        )


if __name__ == "__main__":
    unittest.main()
