"""A correction applied to one book of record must reach the other.

There are two books. ``data/strategy_ledger.json`` is the rolling promotion
window that gates graduation; ``data/strategy_registry.json`` is the
append-only lifetime record that ``scripts/live_path_check.py`` link 10 reads
to decide whether live trading has made money. ``StrategyLedger.record``
mirrors every new outcome into the registry, so in normal operation they agree.

They diverged on 2026-09-03. Four live exits with no settling ERC-20 Transfer
on base were annulled -- struck from the ledger and marked
``trade_outcomes.status='annulled'`` -- but the registry was never touched. The
gate went on reporting

    [FAIL ] 10 PROFIT   live P/L -0.1405 over 7 trades

for 7 hours while the ledger and the chain both said +0.0052 over 3. Nothing
was wrong with the chain, the executor or the strategy: one of the two books
had simply not been told.

These tests pin both halves of the repair:
  * the correction primitive (``rebuild_lifetime``) computes a lifetime block
    identically to the record primitive, so a replay can never drift from a
    record; and
  * the two shipped books agree about live P/L, which is the invariant whose
    violation caused the failure.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from services import strategy_registry as reg
from trading.strategies.ledger import StrategyLedger


class RebuildMatchesRecordTest(unittest.TestCase):
    """``rebuild_lifetime`` and ``record_outcome`` must fold identically.

    They share ``_fold`` precisely so that a correction cannot quietly use
    different arithmetic from the path that wrote the numbers being corrected.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(reg, "REGISTRY_PATH", Path(self.tmp.name) / "reg.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    HISTORY = [
        {"profit": 0.0012394, "symbol": "AERO-USDC", "ts": 1000.0},
        {"profit": 0.0097748, "symbol": "CBETH-USDC", "ts": 2000.0},
        {"profit": -0.0058591, "symbol": "CBETH-USDC", "ts": 3000.0},
    ]

    def _live(self, sid="s1"):
        return ((reg.get_strategy(sid) or {}).get("lifetime") or {}).get("live") or {}

    def test_replaying_a_history_equals_recording_it(self):
        for o in self.HISTORY:
            reg.record_outcome("s1", profit=o["profit"], mode="live",
                               symbol=o["symbol"], ts=o["ts"])
        recorded = dict(self._live())

        # s2 exists but has never traded live; record_outcome auto-registers it
        # exactly as it auto-registered atf_static.
        reg.record_outcome("s2", profit=0.5, mode="ghost", symbol="Z-USDC", ts=1.0)
        reg.rebuild_lifetime("s2", mode="live", outcomes=self.HISTORY, reason="replay")
        rebuilt = dict(self._live("s2"))

        self.assertEqual(recorded, rebuilt)

    def test_striking_the_worst_trade_recomputes_the_path_dependent_fields(self):
        """max_drawdown cannot be repaired by subtraction, only by replay.

        This is the shape of the real defect: the struck trade was the one that
        SET peak-to-trough. Reversing it arithmetically would leave a drawdown
        no surviving trade ever produced.
        """
        history = self.HISTORY + [
            {"profit": -0.1428655, "symbol": "BSTONK-USDC", "ts": 4000.0},
        ]
        for o in history:
            reg.record_outcome("s1", profit=o["profit"], mode="live",
                               symbol=o["symbol"], ts=o["ts"])
        before = dict(self._live())
        self.assertEqual(before["trades"], 4)
        # peak +0.0110142 (after trade 2), trough -0.1377104 (after trade 4).
        self.assertAlmostEqual(before["max_drawdown"], 0.1487246, places=7)
        self.assertIn("BSTONK-USDC", before["symbols"])

        reg.rebuild_lifetime("s1", mode="live", outcomes=self.HISTORY,
                             reason="no settling transfer on chain")
        after = self._live()

        self.assertEqual(after["trades"], 3)
        self.assertEqual((after["wins"], after["losses"]), (2, 1))
        self.assertAlmostEqual(after["total_profit"], 0.0051551, places=7)
        # The drawdown is now the one the surviving trades actually made,
        # not the old figure minus the struck trade.
        self.assertAlmostEqual(after["max_drawdown"], 0.0058591, places=7)
        self.assertAlmostEqual(after["worst"], -0.0058591, places=7)
        # The struck trade's symbol leaves the record entirely.
        self.assertNotIn("BSTONK-USDC", after["symbols"])
        # ...and the ghost book is untouched by a live correction.
        self.assertEqual(((reg.get_strategy("s1") or {}).get("lifetime") or {}).get("ghost"), None)

    def test_a_correction_is_audited(self):
        reg.record_outcome("s1", profit=-1.0, mode="live", symbol="X-USDC", ts=1.0)
        reg.rebuild_lifetime("s1", mode="live", outcomes=[], reason="never happened",
                             struck=[{"symbol": "X-USDC", "net_profit": -1.0}])
        corr = (reg.get_strategy("s1") or {}).get("corrections") or []
        self.assertEqual(len(corr), 1)
        self.assertEqual(corr[0]["reason"], "never happened")
        self.assertEqual(corr[0]["was"]["trades"], 1)
        self.assertEqual(corr[0]["now"]["trades"], 0)
        self.assertEqual(corr[0]["struck"][0]["symbol"], "X-USDC")

    def test_a_correction_never_invents_the_strategy_it_corrects(self):
        self.assertIsNone(
            reg.rebuild_lifetime("nobody", mode="live", outcomes=self.HISTORY, reason="x")
        )
        self.assertIsNone(reg.get_strategy("nobody"))


class ShippedBooksAgreeTest(unittest.TestCase):
    """The two files in data/ must tell the same story about live money.

    Read-only: this asserts against the real books rather than fixtures,
    because the divergence that caused the failure existed only in the real
    files. A fixture cannot catch a correction someone forgot to mirror.
    """

    def setUp(self):
        self.registry = json.loads(reg._DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
        self.ledger = json.loads(StrategyLedger.DEFAULT_PATH.read_text(encoding="utf-8"))

    def test_live_pl_matches_between_registry_and_ledger(self):
        for sid, ent in (self.registry.get("strategies") or {}).items():
            live = ((ent.get("lifetime") or {}).get("live")) or {}
            if not int(live.get("trades") or 0):
                continue
            book = (self.ledger.get(sid) or {}).get("live") or {}
            self.assertEqual(
                int(live["trades"]), int(book.get("trades") or 0),
                f"{sid}: registry says {live['trades']} live trades, ledger says "
                f"{book.get('trades')}. One book has been corrected and the other "
                f"has not -- see scripts/annul_registry_unsettled_live.py",
            )
            self.assertAlmostEqual(
                float(live["total_profit"]), float(book.get("total_profit") or 0.0),
                places=9,
                msg=f"{sid}: registry live P/L {live['total_profit']!r} != ledger "
                    f"{book.get('total_profit')!r}",
            )

    def test_registry_live_trades_match_the_settled_outcome_rows(self):
        """The chain is the third book, and it outranks both.

        ``trade_outcomes`` rows carry ``status='annulled'`` for exits with no
        settling ERC-20 Transfer. A live lifetime record must count only the
        rows that settled.
        """
        import sqlite3

        db = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"
        if not db.exists():  # pragma: no cover - fresh checkout
            self.skipTest("no trading_cache.db")
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        settled = 0
        total = 0.0
        for row in conn.execute("SELECT status, net_profit, details FROM trade_outcomes"):
            det = json.loads(row["details"] or "{}")
            if str(det.get("mode")) == "live" and str(row["status"]) == "closed":
                settled += 1
                total += float(row["net_profit"] or 0.0)
        conn.close()

        booked = sum(
            int((((e.get("lifetime") or {}).get("live")) or {}).get("trades") or 0)
            for e in (self.registry.get("strategies") or {}).values()
        )
        self.assertEqual(
            booked, settled,
            f"the registry books {booked} live trades but only {settled} live "
            f"outcome rows settled on chain",
        )
        booked_pl = sum(
            float((((e.get("lifetime") or {}).get("live")) or {}).get("total_profit") or 0.0)
            for e in (self.registry.get("strategies") or {}).values()
        )
        self.assertAlmostEqual(booked_pl, total, places=9)


if __name__ == "__main__":
    unittest.main()
