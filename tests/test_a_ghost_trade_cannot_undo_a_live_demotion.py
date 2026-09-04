"""A ghost outcome must not hand a demoted strategy back its chequebook.

Measured 2026-09-04 by replaying the shipped ledger code against a copy of the
real data/strategy_ledger.json. atf_static, the only strategy on this account
that has ever spent real money, sat demoted with

    demote_reason = "live drawdown: +0.1423 from peak +0.2221"

and one simulated trade put it straight back on real money:

    start    approved=False demotions=5  reason="live drawdown: +0.1423 ..."
    +ghost   approved=True  demotions=5  reason="live drawdown: +0.1423 ..."
    +live-L  approved=False demotions=6  reason="live drawdown: +0.1223 ..."
    +ghost   approved=True  demotions=6  ...

Two independent holes made that possible, and both are pinned here.

1. ``_evaluate_graduation_locked`` consulted ``graduation_blocked`` but never
   ``demote_reason``, and ``record()`` runs the demotion check only for live
   outcomes. So a ghost trade re-approved the strategy on the very ghost book
   it already held when it was demoted, and the entry then said both "pulled
   off real money for live drawdown" and "cleared to spend real money".

2. ``_maybe_rearm_locked`` -- written to break the demotion lockout -- was
   unreachable. It is called only when the strategy is NOT approved, and
   graduation had already approved it. Its conditions were vacuous anyway:
   ``net > 0`` is true by construction for the drawdown brake, which only fires
   on strategies that are still up, and ``consecutive_losses == 0`` is set by
   ``_demote_locked`` itself on its last line.

The money this cost: each cycle re-funded a live book of 9 round trips, 2W/7L,
net +0.1423 -- of which a single BSTONK exit is +0.2420. The other eight sum to
-0.0997 and lose -0.0503 GROSS, before any fee.
"""

import os
import tempfile
import unittest
from unittest import mock

from trading.strategies.ledger import StrategyLedger


class GhostCannotUndoALiveDemotionTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "ledger.json")
        # Pin the bars this test reasons about; .env must not steer it.
        self._env = mock.patch.dict(os.environ, {
            "STRATEGY_GRADUATION_MIN_TRADES": "20",
            "STRATEGY_GRADUATION_MIN_WINRATE": "0.55",
            "STRATEGY_GRADUATION_MIN_PROFIT": "0.0",
            "STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES": "8",
            "STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN": "0.25",
            "STRATEGY_DEMOTE_MIN_LIVE_TRADES": "12",
            "STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "2",
        })
        self._env.start()
        self.addCleanup(self._env.stop)

    def _atf_static_as_measured(self):
        """The real entry, as data/strategy_ledger.json held it on 2026-09-04."""
        led = StrategyLedger(self.path)
        ent = led._entry("atf_static")
        ent["ghost"].update({"trades": 35, "wins": 22, "losses": 2,
                             "total_profit": 0.9191999015361599,
                             "peak_profit": 0.9924689923694343})
        ent["live"].update({"trades": 9, "wins": 2, "losses": 7,
                            "total_profit": 0.14230439571137737,
                            "peak_profit": 0.2221068671625748,
                            "consecutive_losses": 0})
        ent["live_approved"] = False
        ent["demote_reason"] = "live drawdown: +0.1423 from peak +0.2221"
        ent["demotions"] = 5
        ent["ghost_at_demotion"] = dict(ent["ghost"])
        led._save()
        return led, "atf_static"

    def test_one_ghost_trade_does_not_re_approve_a_demoted_strategy(self):
        led, sid = self._atf_static_as_measured()
        led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                   mirror_registry=False)
        ent = led._entry(sid)
        self.assertFalse(ent["live_approved"],
                         "a simulated trade must not undo a live demotion")
        self.assertEqual(ent["demote_reason"],
                         "live drawdown: +0.1423 from peak +0.2221")

    def test_an_entry_is_never_both_demoted_and_approved(self):
        """The two fields have to agree; the thrash left them contradicting."""
        led, sid = self._atf_static_as_measured()
        for _ in range(10):
            led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                       mirror_registry=False)
            ent = led._entry(sid)
            if ent["live_approved"]:
                self.assertIsNone(ent.get("demote_reason"))
            else:
                self.assertTrue(ent.get("demote_reason"))

    def test_a_re_earned_ghost_book_does_re_approve_it(self):
        """Demotion is a pause. Twenty fresh winning ghost trades end it."""
        led, sid = self._atf_static_as_measured()
        for _ in range(19):
            led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                       mirror_registry=False)
        self.assertFalse(led._entry(sid)["live_approved"],
                         "19 of a 20-trade bar is not a re-earned book")
        led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                   mirror_registry=False)
        ent = led._entry(sid)
        self.assertTrue(ent["live_approved"])
        self.assertIsNone(ent.get("demote_reason"))
        self.assertEqual(int(ent.get("rearms") or 0), 1)

    def test_re_arming_rebases_the_brake_but_not_the_reported_peak(self):
        led, sid = self._atf_static_as_measured()
        for _ in range(20):
            led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                       mirror_registry=False)
        live = led._entry(sid)["live"]
        self.assertTrue(led._entry(sid)["live_approved"])
        # The brake now measures from the new licence...
        self.assertAlmostEqual(led._dd_ref(live), 0.14230439571137737, places=9)
        # ...while peak_profit still means "the most it has ever been up".
        self.assertAlmostEqual(float(live["peak_profit"]), 0.2221068671625748,
                               places=9)
        # And the first live outcome after re-arming does not instantly
        # re-demote it against a peak from the licence it lost. Under the old
        # code this exact call took it from approved to demoted.
        led.record(sid, profit=-0.02, mode="live", symbol="AERO-USDC",
                   mirror_registry=False)
        self.assertTrue(led._entry(sid)["live_approved"],
                        "one normal losing trade is not a give-back of the peak")

    def test_the_brake_still_fires_on_a_real_give_back(self):
        """Re-basing must not disarm the brake, only re-scope it."""
        led, sid = self._atf_static_as_measured()
        for _ in range(20):
            led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                       mirror_registry=False)
        self.assertTrue(led._entry(sid)["live_approved"])
        # dd_ref is +0.14230. The 25% bar is +0.10673. Two losses of 0.02 leave
        # +0.10230, under it.
        led.record(sid, profit=-0.02, mode="live", symbol="AERO-USDC",
                   mirror_registry=False)
        led.record(sid, profit=-0.02, mode="live", symbol="AERO-USDC",
                   mirror_registry=False)
        ent = led._entry(sid)
        self.assertFalse(ent["live_approved"])
        self.assertIn("drawdown", str(ent.get("demote_reason") or ""))

    def test_a_strategy_that_lost_real_money_is_not_excused_by_ghost(self):
        led = StrategyLedger(self.path)
        sid = "loser"
        ent = led._entry(sid)
        ent["ghost"].update({"trades": 30, "wins": 20, "total_profit": 0.9})
        ent["live"].update({"trades": 12, "wins": 2, "losses": 10,
                            "total_profit": -0.55})
        ent["live_approved"] = False
        ent["demote_reason"] = "live P/L -0.5500 over 12 trades is not profitable"
        ent["ghost_at_demotion"] = dict(ent["ghost"])
        for _ in range(40):
            led.record(sid, profit=0.01, mode="ghost", symbol="AERO-USDC",
                       mirror_registry=False)
        self.assertFalse(led._entry(sid)["live_approved"],
                         "a negative live record is not cured by simulation")

    def test_a_demotion_without_a_snapshot_baselines_from_now(self):
        """Fail closed: a hand-edited demotion must not read as recovered."""
        led = StrategyLedger(self.path)
        sid = "handedited"
        ent = led._entry(sid)
        ent["ghost"].update({"trades": 90, "wins": 70, "total_profit": 3.0})
        ent["live"].update({"trades": 9, "total_profit": 0.10})
        ent["live_approved"] = False
        ent["demote_reason"] = "demoted by hand"
        ent.pop("ghost_at_demotion", None)
        led.record(sid, profit=0.001, mode="ghost", symbol="AERO-USDC",
                   mirror_registry=False)
        self.assertFalse(led._entry(sid)["live_approved"],
                         "a 90-trade book from before the demotion is not fresh")
        self.assertIsInstance(led._entry(sid).get("ghost_at_demotion"), dict)


if __name__ == "__main__":
    unittest.main()
