"""A strategy record must be answerable to the trade log.

scripts/purge_test_artifacts.py catches fabrications with a perfect win rate.
money_button's did not have one -- 16W/60L at -0.3997 -- so it survived, and
it is the most expensive record in the file to get wrong: it is the lane the
system most wants short-horizon evidence for, and a fabricated LOSING record
reads as "already tried, does not work".

Measured 2026-09-02 the database held exactly ONE money_button round trip
(TOAD-USDC, +0.005278 net over 304 seconds) against 77 claimed. The other 76
are test fixtures that reached the production registry.

These tests pin the detector's two halves: it must catch a losing fabrication,
and it must never touch a record the log corroborates.
"""

from __future__ import annotations

import unittest

from scripts.purge_unwitnessed_records import judge, rebuild_ghost


def _ghost(trades, *, symbols=None, profit=0.0):
    return {"trades": trades, "total_profit": profit, "symbols": symbols or {}}


class DetectorScope(unittest.TestCase):
    def test_catches_a_losing_fabrication(self):
        """The money_button case: many claimed, one witnessed, no symbols."""
        unwitnessed, reason = judge("money_button", _ghost(77, profit=-0.3897), witnessed=1)
        self.assertTrue(unwitnessed, reason)

    def test_catches_a_record_with_no_trades_in_the_log_at_all(self):
        """volume_spike: 21 claimed, 0 witnessed, -29.82 on a stack whose
        largest real trade is +0.41."""
        unwitnessed, _ = judge("volume_spike", _ghost(21, profit=-29.82), witnessed=0)
        self.assertTrue(unwitnessed)

    def test_keeps_a_record_the_log_corroborates(self):
        """atf_static: 245 of 250 witnessed. Must never be purged."""
        unwitnessed, reason = judge(
            "atf_static",
            _ghost(250, symbols={"BASECAT-USDC": 63}, profit=2.6684),
            witnessed=245,
        )
        self.assertFalse(unwitnessed, reason)

    def test_named_symbols_alone_protect_a_record(self):
        """Belt and braces: op rows can be pruned, symbol counts cannot be
        invented by the fixtures this guards against."""
        unwitnessed, _ = judge("atf_static", _ghost(250, symbols={"X-USDC": 1}), witnessed=0)
        self.assertFalse(unwitnessed)

    def test_a_short_history_is_not_a_fabrication(self):
        """A strategy with 2 trades is new, not fictional."""
        unwitnessed, _ = judge("obv_accumulation@1w", _ghost(2), witnessed=0)
        self.assertFalse(unwitnessed)

    def test_a_partly_witnessed_record_is_left_alone(self):
        """Above the floor the record is treated as real but incomplete --
        ledger resets and pruning are ordinary, fabrication is not."""
        unwitnessed, _ = judge("s", _ghost(10), witnessed=3)
        self.assertFalse(unwitnessed)


class RebuildIsAMeasurement(unittest.TestCase):
    def test_rebuild_reproduces_the_surviving_trade(self):
        rebuilt = rebuild_ghost([(1788207983.0, 0.005277519134108606, "TOAD-USDC")])
        self.assertEqual(rebuilt["trades"], 1)
        self.assertEqual(rebuilt["wins"], 1)
        self.assertEqual(rebuilt["losses"], 0)
        self.assertAlmostEqual(rebuilt["total_profit"], 0.005277519134108606, places=12)
        self.assertEqual(rebuilt["symbols"], {"TOAD-USDC": 1})

    def test_rebuild_names_symbols_so_the_record_can_be_analysed(self):
        """The empty symbols map is what made per-symbol analysis impossible;
        a rebuilt record must not reintroduce it."""
        rebuilt = rebuild_ghost([
            (1.0, 0.01, "A-USDC"),
            (2.0, -0.02, "B-USDC"),
            (3.0, 0.03, "A-USDC"),
        ])
        self.assertEqual(rebuilt["symbols"], {"A-USDC": 2, "B-USDC": 1})
        self.assertEqual(rebuilt["wins"], 2)
        self.assertEqual(rebuilt["losses"], 1)
        self.assertAlmostEqual(rebuilt["total_profit"], 0.02, places=12)
        self.assertAlmostEqual(rebuilt["max_drawdown"], 0.02, places=12)
        self.assertEqual(rebuilt["max_consecutive_losses"], 1)

    def test_rebuild_of_nothing_is_empty_not_zeroed(self):
        self.assertEqual(rebuild_ghost([])["trades"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
