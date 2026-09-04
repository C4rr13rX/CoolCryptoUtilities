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


class DemotionIsAPauseNotADeathSentenceTest(unittest.TestCase):
    """A demoted strategy must be able to come back when the money recovers.

    Measured 2026-09-04: atf_static was demoted at net -0.4135, its live P/L
    then RECOVERED to +0.2221, and it stayed locked out for three hours --
    while being the only strategy able to trade at all. Overnight production
    settled six swaps against twenty-plus from the same machinery the previous
    afternoon. Demotion also wiped the ghost record, so it faced a 20-trade
    re-graduation bar from zero and could never have returned within a session.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "ledger.json")

    def _demoted(self, sid="atf_static"):
        led = StrategyLedger(self.path)
        ent = led._entry(sid)
        ent["live_approved"] = True
        ent["ghost"].update({"trades": 30, "wins": 20, "total_profit": 0.9})
        led._demote_locked(sid, "3 consecutive live losses with net -0.4135")
        return led, sid, led._entry(sid)

    def test_demotion_keeps_the_ghost_record(self):
        """Wiping it made demotion permanent within a session."""
        led, sid, ent = self._demoted()
        self.assertEqual(int(ent["ghost"].get("trades") or 0), 30,
                         "the proving-ground record must survive a demotion")

    def test_a_recovered_strategy_is_re_armed(self):
        led, sid, ent = self._demoted()
        ent["live"].update({"trades": 6, "consecutive_losses": 0,
                            "total_profit": 0.2221})
        led._evaluate_demotion_locked(sid)
        self.assertTrue(led._entry(sid)["live_approved"],
                        "net positive with no active streak must trade again")
        self.assertIsNone(led._entry(sid).get("demote_reason"))

    def test_a_still_losing_strategy_stays_demoted(self):
        led, sid, ent = self._demoted()
        ent["live"].update({"trades": 6, "consecutive_losses": 0,
                            "total_profit": -0.30})
        led._evaluate_demotion_locked(sid)
        self.assertFalse(led._entry(sid)["live_approved"])

    def test_an_active_losing_streak_blocks_re_arming(self):
        """Recovering P/L while still losing in a row is not recovered."""
        led, sid, ent = self._demoted()
        ent["live"].update({"trades": 6, "consecutive_losses": 2,
                            "total_profit": 0.05})
        led._evaluate_demotion_locked(sid)
        self.assertFalse(led._entry(sid)["live_approved"])

    def test_a_permanent_block_is_never_re_armed(self):
        led = StrategyLedger(self.path)
        sid = "bad"
        ent = led._entry(sid)
        ent["live_approved"] = True
        led._demote_locked(sid, "fabricated record", permanent=True)
        ent["live"].update({"trades": 9, "consecutive_losses": 0,
                            "total_profit": 5.0})
        led._evaluate_demotion_locked(sid)
        self.assertFalse(led._entry(sid)["live_approved"],
                         "a permanent block is a decision, not a bad stretch")
