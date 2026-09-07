"""A licence to spend real money must stand on evidence that could be spent.

Measured 2026-09-07 on atf_static, the only live-capable strategy, over its 9
fresh ghost closes since ``demoted_ts``:

    ALL fresh        9 trades  4 wins  0.4444  net +0.584094
    TRADEABLE fresh  7 trades  2 wins  0.2857  net -0.271454
    UNTRADEABLE      2 trades  2 wins  1.0000  net +0.855548

Both untradeable rows are BSTONK-USDC, for which
``trading.pipeline.stop_is_unenforceable`` is True -- no stop binds on it, so
the live lane will not place it. ``_maybe_rearm_locked`` and
``_evaluate_graduation_locked`` both read the POOLED number, so the entire
profit case for putting real money back behind atf_static rested on two trades
it could never have placed.

The 0.55 win-rate bar was refusing it, but for the wrong reason and only by
luck: a run of fee-scraping micro-wins on an untradeable symbol clears both the
count and the hit rate. These tests pin the population the bar reads, not the
bar itself -- no threshold is changed anywhere in this fix.

Against the old code the first two tests FAIL: the strategy graduates and
re-arms on untradeable evidence.
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

# Real symbols, with the tradeability the live lane actually assigns them --
# asserted in test_the_fixture_symbols_have_the_tradeability_this_test_assumes
# so this file cannot quietly rot into testing nothing.
UNTRADEABLE = "BSTONK-USDC"
TRADEABLE = "AERO-USDC"


class LicenceNeedsSpendableEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ledger.json"
        self.addCleanup(self._tmp.cleanup)
        # mirror_registry is gated on `self.path == self.DEFAULT_PATH`, so an
        # isolated path cannot reach data/strategy_registry.json. Belt and
        # braces: this module has already written fabricated trades into the
        # production registry once (see StrategyLedger.record's docstring).
        self._env = mock.patch.dict(
            os.environ,
            {
                "STRATEGY_GRADUATION_MIN_TRADES": "4",
                "STRATEGY_GRADUATION_MIN_WINRATE": "0.55",
                "STRATEGY_GRADUATION_MIN_PROFIT": "0.0",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _ledger(self) -> StrategyLedger:
        return StrategyLedger(path=str(self.path))

    def _feed(self, led: StrategyLedger, sid: str, rows) -> None:
        for symbol, profit in rows:
            led.record(sid, profit=profit, mode="ghost", symbol=symbol)

    def _demote(self, led: StrategyLedger, sid: str) -> None:
        """Demote and persist.

        In production `_demote_locked` runs inside `record()`, which saves
        before returning. Called bare it mutates only this instance's cached
        dict, and the very next `record()` begins with `_load()` -- which
        re-reads the file and discards it.
        """
        led._demote_locked(sid, "live P/L is not profitable")
        led._save()

    # -- first licence ------------------------------------------------------
    def test_a_strategy_cannot_graduate_on_a_symbol_the_live_lane_refuses(self):
        """Four profitable BSTONK round trips are not a licence to spend."""
        led = self._ledger()
        self._feed(led, "s", [(UNTRADEABLE, 0.5)] * 6)
        ent = led._entry("s")
        self.assertEqual(ent["ghost"]["trades"], 6, "pooled book still records them")
        self.assertEqual(
            ent["ghost"]["tradeable"]["trades"],
            0,
            "none of them were placeable live",
        )
        self.assertFalse(
            ent["live_approved"],
            "graduated on 6 wins the live lane could never have placed",
        )

    def test_a_strategy_still_graduates_on_evidence_it_could_have_spent(self):
        """The gate is not switched off -- tradeable wins still graduate it."""
        led = self._ledger()
        self._feed(led, "s", [(TRADEABLE, 0.5)] * 6)
        ent = led._entry("s")
        self.assertEqual(ent["ghost"]["tradeable"]["trades"], 6)
        self.assertTrue(
            ent["live_approved"],
            "a real tradeable book must still earn a licence",
        )

    def test_untradeable_wins_cannot_carry_a_losing_tradeable_book(self):
        """The dangerous case: the POOLED book clears every bar and is a lie.

        atf_static's real fresh window is the seven TRADEABLE rows below --
        -0.271454 at 2/7 -- plus two BSTONK winners worth +0.855548. Pooled
        that is 9 trades, 4 wins, +0.584094: profitable, and it only missed
        the 0.55 hit rate by luck of the count.

        So this extends the measured shape by the one thing that was about to
        happen anyway. The ghost entry rate is rising, BSTONK is one of the two
        symbols this book fires on most, and its wins are fee-scraping micro
        wins (+0.014830 is a real row). A handful more of them and the pooled
        book reads 11 trades / 8 wins / 0.727 / +0.90 -- clearing the count,
        the hit rate and the profit floor at once -- while the money it could
        actually spend is still -0.271454 at a 29% hit rate.

        That is the trade this test exists to refuse. The old rule grants the
        licence here.
        """
        led = self._ledger()
        self._feed(
            led,
            "s",
            [
                # The real fresh tradeable window, verbatim.
                (TRADEABLE, -0.008920),
                (TRADEABLE, -0.042150),
                (TRADEABLE, -0.048250),
                (TRADEABLE, -0.070430),
                (TRADEABLE, 0.001400),
                (TRADEABLE, -0.106100),
                (TRADEABLE, 0.002980),
                # The two real BSTONK rows, plus four more of the small kind.
                (UNTRADEABLE, 0.014830),
                (UNTRADEABLE, 0.840718),
                (UNTRADEABLE, 0.014830),
                (UNTRADEABLE, 0.014830),
                (UNTRADEABLE, 0.014830),
                (UNTRADEABLE, 0.014830),
            ],
        )
        ent = led._entry("s")
        pooled, sub = ent["ghost"], ent["ghost"]["tradeable"]

        # The pooled book clears all three bars set in setUp.
        self.assertEqual(pooled["trades"], 13)
        self.assertEqual(pooled["wins"], 8)
        self.assertGreaterEqual(pooled["wins"] / pooled["trades"], 0.55)
        self.assertAlmostEqual(pooled["total_profit"], 0.643398, places=5)

        # The money it could actually have placed does not.
        self.assertEqual(sub["trades"], 7)
        self.assertEqual(sub["wins"], 2)
        self.assertLess(sub["wins"] / sub["trades"], 0.55)
        self.assertAlmostEqual(sub["total_profit"], -0.271470, places=5)

        self.assertFalse(
            ent["live_approved"],
            "granted a licence on a pooled +0.643 that is -0.271 where it "
            "can spend, at a 29% hit rate",
        )

    # -- re-arm after a demotion -------------------------------------------
    def test_a_demoted_strategy_cannot_rearm_on_untradeable_evidence(self):
        led = self._ledger()
        self._feed(led, "s", [(TRADEABLE, 0.5)] * 6)
        self.assertTrue(led._entry("s")["live_approved"])

        self._demote(led, "s")
        ent = led._entry("s")
        self.assertFalse(ent["live_approved"])
        # The demotion snapshot must be an independent copy, or the fresh
        # delta below is structurally pinned at zero.
        self.assertIsNot(
            ent["ghost_at_demotion"]["tradeable"],
            ent["ghost"]["tradeable"],
            "ghost_at_demotion aliases the live counter; every fresh delta "
            "would be zero forever",
        )

        self._feed(led, "s", [(UNTRADEABLE, 0.5)] * 8)
        self.assertFalse(
            led._entry("s")["live_approved"],
            "re-armed on 8 wins in a symbol the live lane refuses",
        )

    def test_a_demoted_strategy_rearms_on_fresh_tradeable_evidence(self):
        """The re-arm path is a pause, not a lockout -- prove it still opens."""
        led = self._ledger()
        self._feed(led, "s", [(TRADEABLE, 0.5)] * 6)
        self._demote(led, "s")
        self.assertFalse(led._entry("s")["live_approved"])

        self._feed(led, "s", [(TRADEABLE, 0.5)] * 6)
        ent = led._entry("s")
        self.assertTrue(
            ent["live_approved"],
            "fresh tradeable evidence must re-arm the strategy",
        )
        self.assertGreater(float(ent.get("rearmed_ts") or 0.0), 0.0)

    # -- the aliasing trap, directly ---------------------------------------
    def test_the_demotion_snapshot_does_not_alias_the_live_book(self):
        """The snapshot must be independent IN MEMORY, on its own terms.

        The subset is nested, so a shallow ``dict()`` leaves the baseline and
        the live book sharing one inner object -- and a baseline that advances
        with the book holds the fresh delta at zero.

        This is hardening, not a shipped bug, and the distinction is worth
        stating: `_save()` serialises the two and the next `_load()` reads
        them back as separate dicts, so the file round trip hides the aliasing
        in production. That is luck from the persistence layer. Asserted here
        BEFORE any save, which is where the guarantee has to hold.
        """
        led = self._ledger()
        self._feed(led, "s", [(TRADEABLE, 0.5)] * 6)
        led._demote_locked("s", "live P/L is not profitable")  # deliberately unsaved
        ent = led._entry("s")
        self.assertIsNot(ent["ghost_at_demotion"], ent["ghost"])
        self.assertIsNot(
            ent["ghost_at_demotion"]["tradeable"],
            ent["ghost"]["tradeable"],
            "shallow copy: the baseline shares the live book's inner dict",
        )
        ent["ghost"]["tradeable"]["trades"] += 1
        self.assertEqual(
            ent["ghost_at_demotion"]["tradeable"]["trades"],
            6,
            "the baseline moved with the book -- the snapshot is aliased",
        )

    # -- migration ----------------------------------------------------------
    def test_a_ledger_written_before_this_fix_baselines_at_zero(self):
        """Old entries must not have their pooled totals laundered in.

        Back-filling `tradeable` from `trades`/`wins`/`total_profit` would
        import exactly the untradeable evidence this counter exists to
        exclude, and would re-arm atf_static on the BSTONK rows immediately.
        """
        legacy = {
            "atf_static": {
                "ghost": {
                    "trades": 48,
                    "wins": 27,
                    "losses": 21,
                    "total_profit": 1.5593800779725517,
                },
                "live": {"trades": 18, "wins": 5, "total_profit": -0.18637145913731235},
                "live_approved": False,
                "demote_reason": "live P/L is not profitable",
            }
        }
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        led = self._ledger()
        ent = led._entry("atf_static")
        self.assertEqual(ent["ghost"]["trades"], 48, "pooled history is preserved")
        self.assertEqual(
            ent["ghost"]["tradeable"],
            {"trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0},
            "the tradeable window must start now, not inherit the pooled book",
        )
        self.assertFalse(ent["live_approved"])

    # -- the fixture's own premise -----------------------------------------
    def test_the_fixture_symbols_have_the_tradeability_this_test_assumes(self):
        """Pin the premise. If BSTONK becomes tradeable this file is a no-op.

        A test whose fixture silently stops exercising the branch it names is
        the failure mode this repo has already shipped -- a gating test that
        passed while its candidate was rejected upstream.
        """
        self.assertFalse(
            ledger_mod._live_tradeable(UNTRADEABLE),
            f"{UNTRADEABLE} is expected to be un-placeable by the live lane",
        )
        self.assertTrue(
            ledger_mod._live_tradeable(TRADEABLE),
            f"{TRADEABLE} is expected to be placeable by the live lane",
        )

    def test_an_unknown_symbol_is_not_counted_as_tradeable_evidence(self):
        """"Could not establish the symbol" is not proof the lane could place it."""
        led = self._ledger()
        self._feed(led, "s", [("", 0.5)] * 6)
        ent = led._entry("s")
        self.assertEqual(ent["ghost"]["trades"], 6)
        self.assertEqual(ent["ghost"]["tradeable"]["trades"], 0)
        self.assertFalse(ent["live_approved"])


if __name__ == "__main__":
    unittest.main()
