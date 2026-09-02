"""Repairing the ledger must not add trades to the lifetime registry.

The ledger and the registry are each other's only independent check, and they
come apart in exactly one direction: the concurrency bug fixed on 2026-09-02
let a registry write land while the ledger write was lost. Measured after that
fix, six ghost exits sat in the registry with no ledger row -- including
money_button's ONLY real round trip, which is why the strategy the loop cares
about most could never graduate no matter how it traded.

Replaying those through the normal path repairs the ledger and inflates the
registry by the same six. That is worse than the gap it closes: the registry is
append-only, it is what "how has this strategy ever actually done" is answered
from, and a lifetime record that gains trades which never happened is the same
fiction the purge scripts exist to remove.

Also pinned here: which ledger id a replayed ATF exit lands under. Rows written
before the scout was split out carry ``strategy_id: "atf_static"`` for BOTH
executors, and replaying them on the stored id would re-credit the ghost-only
scout's 368 trades to the id the live gate reads -- silently re-graduating a
strategy on trades it did not take.
"""

from __future__ import annotations

import unittest
from unittest import mock

from trading.strategies.ledger import StrategyLedger


class MirrorRegistryTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "ledger.json"
        # record() only mirrors from the PRODUCTION path, so the test ledger
        # has to claim that identity for the mirroring branch to be reachable
        # at all -- otherwise this test would pass without testing anything.
        patch = mock.patch.object(StrategyLedger, "DEFAULT_PATH", self.path)
        patch.start()
        self.addCleanup(patch.stop)

    def test_replay_can_skip_the_registry(self):
        ledger = StrategyLedger(path=self.path)
        with mock.patch("services.strategy_registry.record_outcome") as rec:
            ledger.record("money_button", profit=0.0053, mode="ghost",
                          symbol="TOAD-USDC", mirror_registry=False)
        rec.assert_not_called()
        self.assertEqual(ledger.stats("money_button")["ghost"]["trades"], 1)

    def test_normal_recording_still_mirrors(self):
        """The skip must be opt-in: a live outcome still reaches the registry."""
        ledger = StrategyLedger(path=self.path)
        with mock.patch("services.strategy_registry.record_outcome") as rec:
            ledger.record("money_button", profit=0.0053, mode="ghost",
                          symbol="TOAD-USDC")
        rec.assert_called_once()
        self.assertEqual(rec.call_args.kwargs["symbol"], "TOAD-USDC")


class BackfillAttributionTest(unittest.TestCase):
    """A legacy ATF row is attributed by its executor, not by its stored id."""

    def _attribute(self, detail):
        from scripts.backfill_ghost_ledger import _attributed_id

        return _attributed_id(detail)

    def test_scout_row_goes_to_the_scout_id(self):
        self.assertEqual(
            self._attribute({"strategy_id": "atf_static",
                             "source": "c0d3rv2_atf_static"}),
            "atf_static_scout",
        )

    def test_bot_row_goes_to_the_bot_id(self):
        self.assertEqual(
            self._attribute({"strategy_id": "atf_static"}), "atf_static"
        )

    def test_other_strategies_are_untouched(self):
        self.assertEqual(
            self._attribute({"strategy_id": "money_button"}), "money_button"
        )
        self.assertEqual(self._attribute({}), "unclassified")


if __name__ == "__main__":
    unittest.main()
