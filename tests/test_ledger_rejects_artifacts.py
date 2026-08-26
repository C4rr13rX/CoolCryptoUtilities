"""
The graduation gate must not accept profits that never happened.

A ghost position entered while its feed was frozen and closed after the feed
was repaired books the entire repricing as profit. Observed live on
2026-08-26: AERO-USDC was entered at a stale 0.436805 on 2026-08-19 and exited
at the corrected 1.14, recording **+3.2066 on a 4.58-unit position** -- a
+161% "gain" that never occurred in the market. The same record carried
`net_pnl: 0.0`, so by its own accounting the trade both did and did not make
money.

Why this matters more than one bad row: `StrategyLedger.record()` is the only
input to the promotion decision. A strategy needs 20 ghost trades at a 55% win
rate to go live, so a handful of fantasy wins can carry a losing strategy
across that line and start it trading real money. This is the same class of
failure that put 969 trades at a 0% win rate into the ledger and forced a full
reset.

Discarding a real outcome costs one data point. Accepting a fabricated one
costs the integrity of the gate, so the check is deliberately asymmetric --
but also deliberately conservative: an outcome must clear both an absolute
bound and the strategy's own recent scale before it is refused.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from trading.strategies.ledger import StrategyLedger, _is_implausible


class ImplausibilityCheck(unittest.TestCase):
    def test_ordinary_outcomes_are_accepted(self):
        for profit in (0.0036, -0.0036, 0.12, -0.4, 0.0):
            self.assertFalse(
                _is_implausible(profit, relative_to=0.004),
                f"{profit} is a normal outcome and must be recorded",
            )

    def test_the_aero_repricing_artifact_is_rejected(self):
        """The exact number this guard was written for."""
        self.assertTrue(_is_implausible(3.2066242686186053, relative_to=0.004))

    def test_a_large_outcome_is_rejected_when_there_is_no_history(self):
        """With nothing to compare against, the absolute bound governs."""
        self.assertTrue(_is_implausible(3.2066, relative_to=None))

    def test_a_strategy_that_genuinely_trades_big_is_not_penalised(self):
        """
        Scale is per strategy on purpose.

        A strategy whose outcomes average ~1.0 should not have a 3.5 result
        thrown away just because a different, smaller strategy would never
        produce one.
        """
        self.assertFalse(_is_implausible(3.5, relative_to=1.0))

    def test_large_losses_are_always_believed(self):
        """
        The asymmetry is the whole design, and it is load-bearing.

        Large losses are ordinary -- a stop-loss, a rug, a crash. Filtering
        them removes the evidence that a strategy is losing money. An earlier
        version of this guard rejected a -10.0 loss, which left only wins in
        the record and GRADUATED a strategy that should have been blocked
        (caught by tests/test_strategy_ledger.py::test_unprofitable_never_
        graduates). A filter against fiction must never become a way to
        launder a losing record.
        """
        for loss in (-10.0, -50.0, -3.2066, -1000.0):
            self.assertFalse(
                _is_implausible(loss, relative_to=0.004),
                f"loss {loss} must be recorded, not filtered away",
            )

    def test_non_finite_values_are_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertTrue(_is_implausible(bad, relative_to=0.004))

    def test_unparseable_values_are_rejected(self):
        self.assertTrue(_is_implausible("not a number", relative_to=0.004))  # type: ignore[arg-type]


class LedgerRejectsArtifacts(unittest.TestCase):
    def setUp(self):
        # StrategyLedger takes its path as a constructor argument, NOT an env
        # var. Setting an env var here silently wrote test data into the real
        # production ledger at data/strategy_ledger.json -- 25 fabricated
        # money_button trades, which is precisely the fictional evidence this
        # module exists to keep out of the promotion gate.
        self.tmp = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmp, "ledger.json")

    def _ledger(self) -> StrategyLedger:
        ledger = StrategyLedger(self.ledger_path)
        assert str(ledger.path) == self.ledger_path, "test would touch the real ledger"
        return ledger

    def test_an_artifact_does_not_reach_the_stats(self):
        """
        A rejected outcome must leave no trace at all.

        Counting it as a trade-but-not-a-win would be its own distortion: it
        would drag the win rate down on a trade that never happened.
        """
        ledger = self._ledger()
        for _ in range(3):
            ledger.record("money_button", profit=0.01, mode="ghost")
        before = dict((ledger._data.get("money_button") or {}).get("ghost") or {})

        ledger.record("money_button", profit=3.2066, mode="ghost")
        after = dict((ledger._data.get("money_button") or {}).get("ghost") or {})

        self.assertEqual(
            before.get("trades"), after.get("trades"),
            "an implausible outcome must not increment the trade count",
        )
        self.assertAlmostEqual(
            float(before.get("total_profit", 0.0)),
            float(after.get("total_profit", 0.0)),
            places=9,
            msg="an implausible outcome must not move total profit",
        )

    def test_real_outcomes_still_record_normally(self):
        """The guard must not break ordinary bookkeeping."""
        ledger = self._ledger()
        for profit in (0.01, -0.005, 0.02):
            ledger.record("money_button", profit=profit, mode="ghost")
        stats = (ledger._data.get("money_button") or {}).get("ghost") or {}
        self.assertEqual(int(stats.get("trades", 0)), 3)
        self.assertEqual(int(stats.get("wins", 0)), 2)
        self.assertAlmostEqual(float(stats.get("total_profit", 0.0)), 0.025, places=9)

    def test_an_artifact_cannot_push_a_strategy_to_graduation(self):
        """
        The property that actually protects real money.

        Nineteen genuine losing trades plus one fabricated windfall must not
        add up to a live-approved strategy.
        """
        ledger = self._ledger()
        for _ in range(19):
            ledger.record("money_button", profit=-0.01, mode="ghost")
        ledger.record("money_button", profit=50.0, mode="ghost")
        entry = ledger._data.get("money_button") or {}
        self.assertFalse(
            entry.get("live_approved"),
            "a fabricated windfall must not graduate a losing strategy",
        )


if __name__ == "__main__":
    unittest.main()
