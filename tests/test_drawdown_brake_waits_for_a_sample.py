"""
The give-back brake must not fire before there is a sample to measure.

On 2026-09-03 the only strategy that has ever spent real money on this account
was demoted while it was winning. atf_static had three live trades, all three
settled on-chain from wallet 0x291c854811e92906a658Fb94Aa511bF919f968ad:

    0x77f0075e...  AERO-USDC    +0.0012394
    0x927834717...  CBETH-USDC   +0.0097748
    0x9ffdd1cf...   CBETH-USDC   -0.0058591
                                 ----------
                                 +0.0051551   2 wins, 1 loss, net positive

The peak was just the running total after trade two (+0.0110142). Under the
configured 25% give-back that tolerated a loss of 0.00275 at trade three, while
a typical trade on this feed is 0.00562 -- so the ordinary closing loss demoted
it. It was fired for being 2W/1L and profitable.

The defect was a missing sample gate, not a wrong threshold. `.env` sets
STRATEGY_DEMOTE_MIN_LIVE_TRADES=3 because "did the account grow?" is a sign test
on a sum and three trades is a fair sample of it. Give-back is a RATIO against a
running maximum, and a running maximum over three points is not a peak -- it is
whichever trade landed last. Reusing the sign test's sample for the ratio made
the brake fire on any loss worth half a normal trade, which no strategy that
ever loses can clear.

The blast radius was larger than the flag: `_demote_locked` blanks the ghost
book, so the erroneous demotion also destroyed 22 ghost trades of evidence and
put re-graduation 20 fresh ghost trades away. That is why the graduation link
reported "no strategy approved for live" while holding a profitable strategy.

These tests set the thresholds explicitly rather than inheriting .env, so they
pin the RULE and cannot be silently changed by a config edit. The production
values are pinned separately, in `TheConfiguredProductionPolicy`.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from trading.strategies.ledger import StrategyLedger


#: The three real atf_static live outcomes, in the order they settled.
ATF_STATIC_LIVE = (0.0012393999763547683, 0.009774763168259131, -0.005859083287470101)

#: What .env actually runs in production.
PRODUCTION_POLICY = {
    "STRATEGY_DEMOTE_MIN_LIVE_TRADES": "3",
    "STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "2",
    "STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN": "0.25",
    "STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES": "8",
}


class _LedgerCase(unittest.TestCase):
    POLICY: dict = {}

    def setUp(self):
        patcher = mock.patch.dict(os.environ, self.POLICY)
        patcher.start()
        self.addCleanup(patcher.stop)

    def graduated(self) -> StrategyLedger:
        """A ledger holding one strategy that has earned its way to live."""
        ledger = StrategyLedger(os.path.join(tempfile.mkdtemp(), "ledger.json"))
        for _ in range(20):
            ledger.record("s", profit=0.01, mode="ghost")
        assert ledger.is_live_approved("s")
        return ledger


class TheConfiguredProductionPolicy(_LedgerCase):
    """The incident must be impossible under the values production actually runs."""

    POLICY = PRODUCTION_POLICY

    def test_the_real_atf_static_record_keeps_its_licence(self):
        ledger = self.graduated()
        for profit in ATF_STATIC_LIVE:
            ledger.record("s", profit=profit, mode="live")

        live = ledger._data["s"]["live"]
        self.assertEqual(live["trades"], 3)
        self.assertEqual(live["wins"], 2)
        self.assertEqual(live["losses"], 1)
        self.assertAlmostEqual(live["total_profit"], 0.0051550798571437986)
        self.assertTrue(
            ledger.is_live_approved("s"),
            "a profitable 2W/1L live record must not be demoted on give-back",
        )

    def test_the_ghost_evidence_survives(self):
        """A wrong demotion also costs the ghost book that earned graduation."""
        ledger = self.graduated()
        for profit in ATF_STATIC_LIVE:
            ledger.record("s", profit=profit, mode="live")
        self.assertEqual(ledger._data["s"]["ghost"]["trades"], 20)

    def test_a_losing_record_is_still_demoted_at_the_third_trade(self):
        """
        Adam's standard -- live P/L must never be negative -- is untouched.

        This is the rule that protects the account early, and it must still
        fire at trade three now that the give-back brake waits for eight.

        The losses are kept non-consecutive so this exercises the net-P/L rule
        rather than the 2-consecutive-loss breaker.
        """
        ledger = self.graduated()
        ledger.record("s", profit=-0.02, mode="live")
        ledger.record("s", profit=0.005, mode="live")
        self.assertTrue(ledger.is_live_approved("s"), "no verdict before a sample")
        ledger.record("s", profit=-0.004, mode="live")
        entry = ledger._data["s"]
        self.assertLess(entry["live"]["total_profit"], 0.0)
        self.assertFalse(ledger.is_live_approved("s"))
        self.assertIn("not profitable", str(entry.get("demote_reason", "")))

    def test_two_consecutive_losses_still_demote_immediately(self):
        """The fast breaker is deliberately not sample-gated, and still isn't."""
        ledger = self.graduated()
        for _ in range(2):
            ledger.record("s", profit=-0.01, mode="live")
        self.assertFalse(ledger.is_live_approved("s"))
        self.assertIn("consecutive", str(ledger._data["s"].get("demote_reason", "")))


class TheGiveBackRule(_LedgerCase):
    """The rule itself, pinned independently of any config values."""

    POLICY = {
        "STRATEGY_DEMOTE_MIN_LIVE_TRADES": "3",
        "STRATEGY_DEMOTE_MAX_LIVE_LOSSES": "4",
        "STRATEGY_DEMOTE_MAX_LIVE_DRAWDOWN": "0.25",
        "STRATEGY_DEMOTE_MIN_DRAWDOWN_TRADES": "8",
    }

    def test_one_win_then_one_loss_does_not_demote(self):
        """
        The minimal shape of the bug.

        On a two-trade record any loss is a large fraction of the running
        maximum, so an ungated brake fires on the second live trade a strategy
        ever makes.
        """
        ledger = self.graduated()
        ledger.record("s", profit=0.01, mode="live")
        ledger.record("s", profit=-0.006, mode="live")
        self.assertTrue(ledger.is_live_approved("s"))

    def test_it_still_fires_once_there_is_a_sample(self):
        """Not a licence to bleed: past the sample the brake works as before."""
        ledger = self.graduated()
        for _ in range(8):
            ledger.record("s", profit=0.05, mode="live")       # peak +0.40
        self.assertTrue(ledger.is_live_approved("s"))
        ledger.record("s", profit=-0.15, mode="live")          # hand back 37%
        entry = ledger._data["s"]
        self.assertGreater(entry["live"]["total_profit"], 0.0, "still net positive")
        self.assertFalse(ledger.is_live_approved("s"), "but demoted on give-back")
        self.assertIn("drawdown", str(entry.get("demote_reason", "")))

    def test_it_does_not_fire_one_trade_early(self):
        """The gate is a real boundary, not an off-by-one."""
        ledger = self.graduated()
        for _ in range(7):
            ledger.record("s", profit=0.05, mode="live")       # peak +0.35
        ledger.record("s", profit=-0.15, mode="live")          # 8th: gate open
        self.assertFalse(ledger.is_live_approved("s"))

        ledger = self.graduated()
        for _ in range(6):
            ledger.record("s", profit=0.05, mode="live")       # peak +0.30
        ledger.record("s", profit=-0.15, mode="live")          # 7th: gate shut
        self.assertTrue(
            ledger.is_live_approved("s"),
            "the brake must not judge a give-back before the sample is reached",
        )

    def test_an_alternating_drain_is_still_caught(self):
        """
        Gating the brake must not let a coin-flip drain trade forever.

        It never reaches four consecutive losses here, so net P/L is what has to
        catch it -- on evidence rather than on noise.
        """
        ledger = self.graduated()
        for i in range(10):
            ledger.record("s", profit=(0.008 if i % 2 == 0 else -0.012), mode="live")
        entry = ledger._data["s"]
        self.assertLess(entry["live"]["total_profit"], 0.0)
        self.assertLess(entry["live"]["consecutive_losses"], 4)
        self.assertFalse(ledger.is_live_approved("s"))


if __name__ == "__main__":
    unittest.main()
