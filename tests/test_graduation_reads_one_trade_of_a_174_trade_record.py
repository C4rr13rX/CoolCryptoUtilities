"""The promotion gate judged a 174-trade record on a sample of one.

Graduation reads ``_tradeable_of(ghost)`` -- the live-tradeable subset of a
strategy's ghost book -- and that sub-counter is only maintained forward, by
``record()``. It landed in commit ``dcb7517`` and was never backfilled, so
measured 4.7 hours later on the real ledger:

    ledger ghost trades, all strategies             394
    ledger `tradeable` counters, all strategies       7
    entries with no `tradeable` key at all       33 of 37

``atf_static`` -- the only executor with a live branch -- read
``tradeable.trades == 1`` while its recorded history held 174 live-tradeable,
in-horizon round trips at a 44.8% win rate. Those are opposite findings: 1 of 20
says "collect more evidence", 78/174 against a 55% bar says "this strategy does
not have an edge and more evidence will not create one".

These tests pin the reconstruction that closes that gap. They fail against the
behaviour that shipped the bug -- reading the ledger's own counter -- because
that counter reports zero for a strategy whose entire recorded history is
live-tradeable.
"""

from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from services import tradeable_evidence


def _make_db(path: Path, rows) -> None:
    """A trading_ops table holding exactly the given ghost exits."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trading_ops (id INTEGER PRIMARY KEY, ts REAL, wallet TEXT, "
        "chain TEXT, symbol TEXT, action TEXT, status TEXT, details TEXT)"
    )
    for i, row in enumerate(rows):
        conn.execute(
            "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details) "
            "VALUES (?,?,?,?,?,?,?)",
            (
                float(i),
                "ghost",
                "base",
                row.get("symbol", ""),
                row.get("action", "exit"),
                row.get("status", "ghost-exit"),
                json.dumps(row),
            ),
        )
    conn.commit()
    conn.close()


def _exit(strategy_id, symbol, profit, *, held_sec=60.0, reason="target_hit"):
    return {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "profit": profit,
        "age_sec": held_sec,
        "exit_reason": reason,
    }


class TradeableEvidenceIsReconstructedFromHistory(unittest.TestCase):
    """The recorded history, not the four-hour-old counter, is the population."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.db = self.tmp / "ops.db"

    def test_every_tradeable_round_trip_in_history_is_counted(self):
        """The gap the bug left: a full history against a counter that says zero.

        Twenty-five recorded round trips, all on a tradeable symbol, all inside
        the horizon. The ledger entry carries NO ``tradeable`` key -- exactly the
        state 33 of 37 real entries were in -- so the shipped gate saw 0 while
        the strategy had 25.
        """
        rows = [
            _exit("s1", "AERO-USDC", 0.01 if i % 4 else -0.02) for i in range(25)
        ]
        _make_db(self.db, rows)

        with mock.patch.object(
            tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)
        ), mock.patch(
            "trading.strategies.ledger._live_tradeable", return_value=True
        ), mock.patch(
            "trading.strategies.ledger._exceeds_evidence_horizon", return_value=False
        ):
            found = tradeable_evidence.reconstruct(self.db)

        self.assertEqual(found["s1"].trades, 25)
        self.assertEqual(found["s1"].wins, 18)      # i % 4 != 0 -> 18 of 25
        self.assertEqual(found["s1"].losses, 7)

        # This is the assertion that fails against the shipped behaviour. The
        # ledger's counter -- what graduation actually read -- is absent, i.e.
        # zero, for the same 25 round trips.
        ledger_counter = ({}).get("tradeable", {}).get("trades", 0)
        self.assertEqual(ledger_counter, 0)
        self.assertGreater(found["s1"].trades, ledger_counter)

    def test_an_untradeable_symbol_is_not_evidence_that_money_could_be_spent(self):
        """A record earned where the live lane refuses is excluded, and counted."""
        rows = [_exit("s1", "BSTONK-USDC", 0.05) for _ in range(10)]
        rows += [_exit("s1", "AERO-USDC", 0.01) for _ in range(3)]
        _make_db(self.db, rows)

        def tradeable(symbol):
            return symbol != "BSTONK-USDC"

        with mock.patch.object(
            tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)
        ), mock.patch(
            "trading.strategies.ledger._live_tradeable", side_effect=tradeable
        ), mock.patch(
            "trading.strategies.ledger._exceeds_evidence_horizon", return_value=False
        ):
            found = tradeable_evidence.reconstruct(self.db)

        self.assertEqual(found["s1"].trades, 3)
        self.assertEqual(found["s1"].dropped_untradeable, 10)
        # The drop is reported rather than silent -- a funnel that hides its own
        # filter is how the untradeable book carried the profit case twice.
        self.assertEqual(found["s1"].exits, 13)

    def test_an_out_of_horizon_round_trip_is_market_drift_not_a_decision(self):
        """The horizon filter is the ledger's, applied in the ledger's order."""
        rows = [_exit("s1", "AERO-USDC", 0.9, held_sec=90000.0)]
        rows += [_exit("s1", "AERO-USDC", 0.01, held_sec=60.0)]
        _make_db(self.db, rows)

        with mock.patch.object(
            tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)
        ), mock.patch(
            "trading.strategies.ledger._live_tradeable", return_value=True
        ):
            found = tradeable_evidence.reconstruct(self.db)

        self.assertEqual(found["s1"].trades, 1)
        self.assertEqual(found["s1"].dropped_out_of_horizon, 1)
        # The +0.9 must NOT reach the net; that row is why the tradeable book
        # read +7.395 or -0.748 depending on 19 long holds.
        self.assertAlmostEqual(found["s1"].net, 0.01, places=6)

    def test_an_implausible_outcome_is_an_artifact_not_a_fill(self):
        rows = [_exit("s1", "AERO-USDC", 3.5)]
        rows += [_exit("s1", "AERO-USDC", 0.02)]
        _make_db(self.db, rows)

        with mock.patch.object(
            tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)
        ), mock.patch(
            "trading.strategies.ledger._live_tradeable", return_value=True
        ), mock.patch(
            "trading.strategies.ledger._exceeds_evidence_horizon", return_value=False
        ):
            found = tradeable_evidence.reconstruct(self.db)

        self.assertEqual(found["s1"].trades, 1)
        self.assertEqual(found["s1"].dropped_implausible, 1)

    def test_a_flat_exit_books_as_a_loss_exactly_as_record_books_it(self):
        """Zero profit still paid the round trip; the two books must reconcile."""
        _make_db(self.db, [_exit("s1", "AERO-USDC", 0.0)])

        with mock.patch.object(
            tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)
        ), mock.patch(
            "trading.strategies.ledger._live_tradeable", return_value=True
        ), mock.patch(
            "trading.strategies.ledger._exceeds_evidence_horizon", return_value=False
        ):
            found = tradeable_evidence.reconstruct(self.db)

        self.assertEqual(found["s1"].wins, 0)
        self.assertEqual(found["s1"].losses, 1)


class ExitRoutingCanManufactureAWinRate(unittest.TestCase):
    """``atf_static_scout`` reads 92.5% because its losers leave by another door.

    78 of its 107 recorded exits are ``max_hold`` and ALL 78 are wins, because an
    underwater position is routed into ``stale_underwater`` before the max-hold
    timer can claim it. The headline is therefore a property of exit ROUTING and
    is not comparable to a strategy whose losers exit through ``max_hold``. The
    per-reason split has to survive into the report or that caveat is invisible.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.db = self.tmp / "ops.db"

    def test_the_exit_reason_split_is_reported_so_the_artifact_is_visible(self):
        rows = [_exit("scout", "AERO-USDC", 0.01, reason="max_hold") for _ in range(8)]
        rows += [
            _exit("scout", "AERO-USDC", -0.01, reason="stale_underwater")
            for _ in range(2)
        ]
        _make_db(self.db, rows)

        with mock.patch.object(
            tradeable_evidence, "_bounds", return_value=(2.0, 14400.0)
        ), mock.patch(
            "trading.strategies.ledger._live_tradeable", return_value=True
        ), mock.patch(
            "trading.strategies.ledger._exceeds_evidence_horizon", return_value=False
        ):
            found = tradeable_evidence.reconstruct(self.db)

        reasons = found["scout"].exit_reasons
        self.assertEqual(reasons["max_hold"], {"wins": 8, "losses": 0})
        self.assertEqual(reasons["stale_underwater"], {"wins": 0, "losses": 2})
        # 80% headline, and every loss is in a different bucket from every win.
        self.assertAlmostEqual(found["scout"].win_rate, 0.8, places=6)


class ReconciliationFailsClosedWhereABackfillWouldGuess(unittest.TestCase):
    """A window that cannot be located must not be written into the gate's state.

    The ledger is a rolling window that gets reset, so its ``ghost.trades`` is a
    SUFFIX of the recorded history. Writing a ``tradeable`` sub-book into an
    entry requires locating that suffix, and the only check available is the
    profit sum. It fails for both live-relevant strategies on the real data --
    ``atf_static`` (ledger 49 / +1.5777 vs a last-49 suffix of +1.1342) and
    ``atf_static_scout`` (ledger 235 against 107 recorded exits) -- so the
    reconstruction is reported as an independent measurement and never written.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.db = self.tmp / "ops.db"
        self.ledger = self.tmp / "ledger.json"

    def test_a_matching_suffix_reconciles(self):
        _make_db(
            self.db,
            [_exit("s1", "AERO-USDC", 0.5)] + [_exit("s1", "AERO-USDC", 0.25)] * 2,
        )
        self.ledger.write_text(
            json.dumps({"s1": {"ghost": {"trades": 2, "total_profit": 0.5}}}), "utf-8"
        )
        out = tradeable_evidence.reconcile_window(self.ledger, self.db)
        self.assertTrue(out["s1"]["reconciles"])

    def test_a_ledger_holding_more_trades_than_history_cannot_be_located(self):
        """atf_static_scout's shape: 235 booked against 107 recorded exits."""
        _make_db(self.db, [_exit("scout", "AERO-USDC", 0.1)] * 3)
        self.ledger.write_text(
            json.dumps({"scout": {"ghost": {"trades": 235, "total_profit": 6.5}}}),
            "utf-8",
        )
        out = tradeable_evidence.reconcile_window(self.ledger, self.db)
        self.assertFalse(out["scout"]["reconciles"])
        self.assertIn("cannot be located", out["scout"]["reason"])

    def test_a_suffix_whose_profit_disagrees_does_not_reconcile(self):
        """atf_static's shape: the right count, the wrong money."""
        _make_db(self.db, [_exit("s1", "AERO-USDC", 0.1)] * 5)
        self.ledger.write_text(
            json.dumps({"s1": {"ghost": {"trades": 5, "total_profit": 1.5777}}}),
            "utf-8",
        )
        out = tradeable_evidence.reconcile_window(self.ledger, self.db)
        self.assertFalse(out["s1"]["reconciles"])
        self.assertIn("suffix sums", out["s1"]["reason"])


class TheVerdictUsesTheGatesOwnThresholds(unittest.TestCase):
    """A blocker line quoting a different bar than the gate uses is worse than none."""

    def test_a_full_sample_below_the_win_bar_reads_as_no_edge_not_as_thin_data(self):
        ev = tradeable_evidence.Evidence(
            strategy_id="atf_static", exits=298, trades=174, wins=78, losses=96,
            net=0.0674,
        )
        verdict = tradeable_evidence.graduation_verdict(ev)
        self.assertFalse(verdict["clears_bar"])
        joined = "; ".join(verdict["blockers"])
        # The sample is ABOVE the 20-trade bar, so "not enough trades" must not
        # appear -- that misreading is the whole reason this module exists.
        self.assertNotIn("against a 20 bar", joined)
        self.assertIn("win rate 44.8%", joined)

    def test_thresholds_come_from_the_same_env_vars_the_ledger_reads(self):
        ev = tradeable_evidence.Evidence(strategy_id="s", trades=0)
        thresholds = tradeable_evidence.graduation_verdict(ev)["thresholds"]
        from trading.strategies import ledger as ledger_mod

        self.assertEqual(
            thresholds["min_trades"],
            ledger_mod._env_int("STRATEGY_GRADUATION_MIN_TRADES", 20),
        )
        self.assertEqual(
            thresholds["min_winrate"],
            ledger_mod._env_float("STRATEGY_GRADUATION_MIN_WINRATE", 0.55),
        )


if __name__ == "__main__":
    unittest.main()
