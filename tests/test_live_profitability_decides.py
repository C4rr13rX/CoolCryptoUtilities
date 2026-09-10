"""
Live profitability is the metric that decides, above all others.

Demotion previously fired only on four CONSECUTIVE live losses. A strategy that
alternates win/loss/win/loss while paying a ~1.0% round trip each time never
reaches four in a row, so it could drain real funds indefinitely and never be
demoted.

That alternating pattern is not hypothetical here: measured on this feed, a
rising price continues rising only 44-50% of the time. A strategy that
graduates on a lucky ghost sample and then reverts to coin-flip behaviour is
the expected case, not the unlucky one.

So once a strategy has a fair sample of LIVE trades it must be net positive on
real money. Ghost record, win rate and consecutive-loss counts are all
secondary to whether the account actually grew.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from trading.strategies.ledger import StrategyLedger


#: The symbol every fixture books its evidence in.
#:
#: It must be one `ledger._live_tradeable` accepts, and since 2b58f50 that
#: predicate also consults `services.symbol_edge_gate` -- which reads the LIVE
#: trade_outcomes book. A fixture naming AERO-USDC or BASECAT-USDC would
#: therefore pass or fail with the market rather than with the code, so the
#: gate is stubbed in setUp and this name is arbitrary.
EVIDENCE_SYMBOL = "FIXTURE-USDC"


class LiveProfitabilityDecides(unittest.TestCase):
    def setUp(self):
        # Isolate the DEMOTION rule from the two gates that decide whether an
        # outcome counts as tradeable evidence at all. Without this the six
        # tests below fail at their FIRST assertion -- `is_live_approved` is
        # False because nothing ever reached the tradeable sub-book -- and the
        # rule they exist to test is never exercised. That is exactly how the
        # only test of the demotion rule sat red while pass_gate reported OK.
        self._gate = mock.patch(
            "services.symbol_edge_gate.refusal_reason", return_value=None
        )
        self._gate.start()
        self.addCleanup(self._gate.stop)
        self._stop = mock.patch(
            "trading.pipeline.stop_is_unenforceable", return_value=False
        )
        self._stop.start()
        self.addCleanup(self._stop.stop)
    def _graduated(self) -> StrategyLedger:
        """A ledger holding one strategy that has earned its way to live."""
        ledger = StrategyLedger(os.path.join(tempfile.mkdtemp(), "ledger.json"))
        for _ in range(20):
            ledger.record("s", profit=0.01, mode="ghost", symbol=EVIDENCE_SYMBOL, held_sec=900.0)
        assert ledger.is_live_approved("s")
        return ledger

    def test_an_alternating_drain_is_demoted(self):
        """
        The gap this closes.

        Win/loss/win/loss never reaches four consecutive losses, so the old
        rule would have funded this pattern forever while it bled fees.
        """
        ledger = self._graduated()
        for i in range(10):
            ledger.record("s", profit=(0.008 if i % 2 == 0 else -0.012), mode="live")
        entry = ledger._data["s"]
        self.assertLess(entry["live"]["total_profit"], 0.0)
        self.assertLess(entry["live"]["consecutive_losses"], 4)
        self.assertFalse(entry.get("live_approved"), "a losing strategy must be demoted")

    def test_a_profitable_strategy_is_kept(self):
        """The check must not punish the case it exists to protect."""
        ledger = self._graduated()
        for _ in range(10):
            ledger.record("s", profit=0.02, mode="live", symbol=EVIDENCE_SYMBOL, held_sec=900.0)
        self.assertTrue(ledger._data["s"].get("live_approved"))

    def test_judgement_waits_for_a_fair_sample(self):
        """
        One early loss is noise, not evidence.

        Demoting on the first red trade would make live trading impossible.
        """
        ledger = self._graduated()
        ledger.record("s", profit=-0.05, mode="live", symbol=EVIDENCE_SYMBOL, held_sec=900.0)
        self.assertTrue(
            ledger._data["s"].get("live_approved"),
            "a single loss must not demote before there is a sample",
        )

    def test_consecutive_losses_still_demote_immediately(self):
        """The fast circuit breaker is unchanged and still fires first."""
        ledger = self._graduated()
        for _ in range(4):
            ledger.record("s", profit=-0.01, mode="live", symbol=EVIDENCE_SYMBOL, held_sec=900.0)
        entry = ledger._data["s"]
        self.assertFalse(entry.get("live_approved"))
        self.assertIn("consecutive", str(entry.get("demote_reason", "")))

    def test_giving_back_the_gains_demotes_even_while_net_positive(self):
        """
        A strategy handing back its peak is not one to keep funding.

        Waiting for it to cross zero means giving back everything first.
        """
        ledger = self._graduated()
        for _ in range(8):
            ledger.record("s", profit=0.05, mode="live", symbol=EVIDENCE_SYMBOL, held_sec=900.0)      # peak +0.40
        self.assertTrue(ledger._data["s"].get("live_approved"))
        for _ in range(3):
            ledger.record("s", profit=-0.09, mode="live", symbol=EVIDENCE_SYMBOL, held_sec=900.0)     # give most back
        entry = ledger._data["s"]
        self.assertGreater(entry["live"]["total_profit"], 0.0, "still net positive")
        self.assertFalse(entry.get("live_approved"), "but demoted on drawdown")

    def test_ghost_trading_is_unaffected(self):
        """
        Ghost is where a strategy learns; only live money is judged this way.

        Applying the live rule to ghost would stop strategies ever graduating.
        """
        ledger = StrategyLedger(os.path.join(tempfile.mkdtemp(), "ledger.json"))
        for i in range(20):
            ledger.record("s", profit=(0.02 if i % 3 else -0.01), mode="ghost",
                          symbol=EVIDENCE_SYMBOL, held_sec=900.0)
        self.assertTrue(ledger.is_live_approved("s"))


if __name__ == "__main__":
    unittest.main()
