"""Outcomes recorded concurrently must all survive.

The ledger gates graduation and the registry holds the lifetime record, and
both were written by several processes at once through a `threading.Lock` --
which orders writers inside ONE interpreter and nothing between processes --
onto a temp file with the same name for every writer, with the failing rename
swallowed.

Measured 2026-09-02 against ghost exits in the database:

    rsi_reversal@5h     13 exits   ledger holds 1
    money_button         1 exit    ledger holds none -- ABSENT
    obv_accumulation@5d  1 exit    ledger holds none -- ABSENT

money_button could therefore never graduate no matter how well it traded: it
was not in the file that decides. Reproduced directly -- forty outcomes through
separate ledger instances left TWO on disk.

A separate StrategyLedger instance is a faithful stand-in for a separate
process here: each has its own threading.Lock, so nothing but the cross-process
file lock orders them.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import services.strategy_registry as registry
from trading.strategies.ledger import StrategyLedger


def _run_concurrently(fn, n):
    threads = [threading.Thread(target=fn, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class LedgerDurabilityTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "strategy_ledger.json"

    def test_every_concurrent_outcome_is_kept(self):
        n = 60
        _run_concurrently(
            lambda i: StrategyLedger(self.path).record(
                "s%d" % (i % 4), profit=0.01, mode="ghost", symbol="A-USDC"
            ),
            n,
        )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        kept = sum(v["ghost"]["trades"] for v in data.values())
        self.assertEqual(
            kept, n,
            "%d of %d outcomes were lost; this is the defect that kept "
            "money_button out of the ledger entirely" % (n - kept, n),
        )
        booked = sum(v["ghost"]["total_profit"] for v in data.values())
        self.assertAlmostEqual(booked, n * 0.01, places=6)

    def test_a_single_strategy_keeps_every_one_of_its_trades(self):
        """rsi_reversal@5h wrote 13 and the ledger kept 1."""
        n = 13
        _run_concurrently(
            lambda i: StrategyLedger(self.path).record(
                "rsi_reversal@5h", profit=-0.147, mode="ghost", symbol="BASECAT-USDC"
            ),
            n,
        )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["rsi_reversal@5h"]["ghost"]["trades"], n)

    def test_writers_do_not_share_a_temp_filename(self):
        """A shared temp name is what made the loss routine rather than rare."""
        seen = []
        real_write_text = Path.write_text

        def spy(self_path, *a, **k):
            if self_path.name.endswith(".tmp"):
                seen.append(self_path.name)
            return real_write_text(self_path, *a, **k)

        with mock.patch.object(Path, "write_text", spy):
            _run_concurrently(
                lambda i: StrategyLedger(self.path).record(
                    "s", profit=0.01, mode="ghost", symbol="A-USDC"
                ),
                8,
            )
        self.assertGreater(len(seen), 0, "no temp file was written at all")
        self.assertEqual(
            len(seen), len(set(seen)),
            "concurrent writers reused a temp filename: %r" % (seen,),
        )

    def test_no_temp_or_lock_files_are_left_behind(self):
        _run_concurrently(
            lambda i: StrategyLedger(self.path).record(
                "s", profit=0.01, mode="ghost", symbol="A-USDC"
            ),
            12,
        )
        leftovers = sorted(p.name for p in self.dir.iterdir() if p.name != self.path.name)
        self.assertEqual(leftovers, [], "stray files left: %r" % (leftovers,))

    def test_an_unreadable_ledger_is_never_overwritten_with_a_blank_one(self):
        """A failed read used to blank _data, which then got saved over everything.

        Losing one outcome is a data point. Losing the file is every strategy's
        promotion history, so the write is refused instead.
        """
        ledger = StrategyLedger(self.path)
        ledger.record("keeper", profit=0.05, mode="ghost", symbol="A-USDC")
        before = self.path.read_text(encoding="utf-8")

        with mock.patch(
            "trading.strategies.ledger.read_json", return_value=(None, False)
        ):
            StrategyLedger(self.path).record(
                "newcomer", profit=0.01, mode="ghost", symbol="B-USDC"
            )

        after = self.path.read_text(encoding="utf-8")
        self.assertEqual(after, before, "an unreadable read blanked the ledger")
        self.assertIn("keeper", json.loads(after))

    def test_graduation_still_requires_the_full_sample(self):
        """Durability must not become a way to graduate on fewer trades."""
        with mock.patch.dict(
            "os.environ",
            {"STRATEGY_GRADUATION_MIN_TRADES": "20", "STRATEGY_GRADUATION_MIN_WINRATE": "0.55"},
            clear=False,
        ):
            for _ in range(19):
                StrategyLedger(self.path).record(
                    "almost", profit=0.02, mode="ghost", symbol="A-USDC"
                )
            self.assertFalse(StrategyLedger(self.path).is_live_approved("almost"))
            StrategyLedger(self.path).record(
                "almost", profit=0.02, mode="ghost", symbol="A-USDC"
            )
            self.assertTrue(StrategyLedger(self.path).is_live_approved("almost"))


class RegistryDurabilityTest(unittest.TestCase):
    """The lifetime record had the identical three defects."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "strategy_registry.json"
        patcher = mock.patch.object(registry, "REGISTRY_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_concurrent_outcome_is_kept(self):
        n = 60
        _run_concurrently(
            lambda i: registry.record_outcome(
                "s%d" % (i % 4), profit=0.01, mode="ghost", symbol="A-USDC"
            ),
            n,
        )
        state = json.loads(self.path.read_text(encoding="utf-8"))["strategies"]
        kept = sum(v["lifetime"]["ghost"]["trades"] for v in state.values())
        self.assertEqual(kept, n, "%d lifetime outcomes lost" % (n - kept))

    def test_symbols_are_recorded_for_every_outcome(self):
        """A record with no symbols cannot be checked for concentration.

        This is how a strategy takes 13 correlated positions in ONE token and
        it looks like 13 independent trades.
        """
        _run_concurrently(
            lambda i: registry.record_outcome(
                "mb", profit=0.005, mode="ghost", symbol="TOAD-USDC"
            ),
            20,
        )
        entry = json.loads(self.path.read_text(encoding="utf-8"))["strategies"]["mb"]
        self.assertEqual(entry["lifetime"]["ghost"]["symbols"], {"TOAD-USDC": 20})

    def test_an_unreadable_registry_is_never_blanked(self):
        registry.record_outcome("keeper", profit=0.05, mode="ghost", symbol="A-USDC")
        before = self.path.read_text(encoding="utf-8")
        with mock.patch(
            "services.strategy_registry.read_json", return_value=(None, False)
        ):
            with self.assertRaises(OSError):
                registry.record_outcome("x", profit=0.01, mode="ghost", symbol="B-USDC")
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
