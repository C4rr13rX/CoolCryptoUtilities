"""Ghost exits must pair with their OWN entries.

db.fetch_trades returns ORDER BY ts DESC, but pairing is causal: an exit can
only match an entry already seen. Iterating newest-first meant every exit
arrived before its own entry, the keyed lookup always missed, and the
best-effort symbol fallback matched the exit to whatever entry sat nearest the
top of the list -- reusing one entry for several exits and dropping the rest.

Measured 2026-08-28 on the live ghost book: all 56 paired trades came back with
a NEGATIVE hold time (down to -105743s, an exit 29 hours before its entry),
realized_delta values belonging to other positions (-59.25 recorded against a
-0.00075 trade), and 4 of 60 real exits silently discarded.

This matters beyond cosmetics: the risk layer computes loss streaks and max
drawdown by walking this list in order, and both are order-dependent. They were
being measured on a time-REVERSED series.
"""

from __future__ import annotations

import unittest

from trading.metrics import MetricsCollector


class _DB:
    """Serves rows newest-first, the way db.fetch_trades does."""

    def __init__(self, rows):
        self._rows = sorted(rows, key=lambda r: float(r["ts"]), reverse=True)

    def fetch_trades(self, *, limit=200, statuses=None, wallets=None,
                     symbol=None, since_ts=None):
        return list(self._rows[:limit])


def _entry(ts, symbol, price, **extra):
    d = {"symbol": symbol, "entry_ts": ts, "entry_price": price}
    d.update(extra)
    return {"ts": ts, "status": "ghost-entry", "symbol": symbol, "details": d}


def _exit(ts, symbol, profit, **extra):
    d = {"symbol": symbol, "profit": profit}
    d.update(extra)
    return {"ts": ts, "status": "ghost-exit", "symbol": symbol, "details": d}


class GhostTradePairingTest(unittest.TestCase):
    def _snap(self, rows):
        return MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)

    def test_hold_time_is_never_negative(self):
        rows = [
            _entry(1000.0, "BASECAT-USDC", 0.02489),
            _exit(3433.0, "BASECAT-USDC", -0.08517,
                  entry_ts=1000.0, exit_ts=3433.0,
                  entry_price=0.02489, exit_price=0.02277, reason="stop_loss"),
        ]
        trades = self._snap(rows)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_ts - trades[0].entry_ts, 2433.0)

    def test_one_entry_is_not_reused_by_several_exits(self):
        """Three BASECAT exits must consume three distinct entries."""
        rows = [
            _entry(100.0, "BASECAT-USDC", 1.0),
            _entry(200.0, "BASECAT-USDC", 2.0),
            _entry(300.0, "BASECAT-USDC", 3.0),
            _exit(150.0, "BASECAT-USDC", -0.01, exit_price=0.99),
            _exit(250.0, "BASECAT-USDC", -0.02, exit_price=1.96),
            _exit(350.0, "BASECAT-USDC", -0.03, exit_price=2.91),
        ]
        trades = self._snap(rows)
        self.assertEqual(len(trades), 3)
        self.assertEqual(
            sorted(t.entry_ts for t in trades), [100.0, 200.0, 300.0],
            "each exit must claim its own entry, FIFO",
        )
        for t in trades:
            self.assertGreaterEqual(t.exit_ts, t.entry_ts)

    def test_exit_carrying_its_own_entry_data_is_not_dropped(self):
        """atf_static writes entry_ts on the exit; no entry row is required."""
        rows = [
            _exit(500.0, "BASENOUN-USDC", 0.13482,
                  entry_ts=384.0, exit_ts=500.0,
                  entry_price=1.0, exit_price=1.13482, reason="target_hit"),
        ]
        trades = self._snap(rows)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].entry_ts, 384.0)
        self.assertEqual(trades[0].reason, "target_hit")

    def test_legacy_rows_recover_entry_ts_from_position_or_age(self):
        """Rows written before atf_static published entry_ts still pair."""
        rows = [
            _exit(900.0, "BSTONK-USDC", -0.08388,
                  age_sec=1736.0, reason="stop_loss",
                  entry_price=0.002134, exit_price=0.001955),
            _exit(950.0, "MAMO-USDC", 0.01,
                  position={"entry_ts": 700.0}, reason="max_hold",
                  entry_price=1.0, exit_price=1.01),
        ]
        trades = self._snap(rows)
        self.assertEqual(len(trades), 2)
        by_symbol = {t.symbol: t for t in trades}
        self.assertEqual(by_symbol["BSTONK-USDC"].entry_ts, 900.0 - 1736.0)
        self.assertEqual(by_symbol["MAMO-USDC"].entry_ts, 700.0)
        for t in trades:
            self.assertGreater(t.exit_ts, t.entry_ts)

    def test_reason_falls_back_from_exit_reason_to_reason(self):
        """atf_static writes "reason"; trading/bot.py writes "exit_reason".

        Reading only exit_reason reported 58 of 60 real exits as "unspecified",
        blinding the risk layer to whether a loss was a stop-loss or a timer.
        """
        rows = [
            _exit(10.0, "A-USDC", -0.08, entry_ts=1.0, reason="stop_loss"),
            _exit(20.0, "B-USDC", 0.05, entry_ts=2.0, exit_reason="take_profit_limit"),
            _exit(30.0, "C-USDC", 0.01, entry_ts=3.0),
        ]
        reasons = {t.symbol: t.reason for t in self._snap(rows)}
        self.assertEqual(reasons["A-USDC"], "stop_loss")
        self.assertEqual(reasons["B-USDC"], "take_profit_limit")
        self.assertEqual(reasons["C-USDC"], "unspecified")

    def test_snapshot_is_chronological_so_streaks_are_measured_forward(self):
        rows = []
        for i, profit in enumerate([-0.01, -0.02, 0.05, -0.03]):
            t = 100.0 * (i + 1)
            rows.append(_entry(t, "X-USDC", 1.0))
            rows.append(_exit(t + 50.0, "X-USDC", profit, entry_ts=t, exit_ts=t + 50.0))
        trades = self._snap(rows)
        self.assertEqual([t.profit for t in trades], [-0.01, -0.02, 0.05, -0.03])

    def test_trade_id_pairs_exactly_even_when_interleaved(self):
        rows = [
            _entry(100.0, "X-USDC", 1.0, trade_id="t1"),
            _entry(110.0, "X-USDC", 2.0, trade_id="t2"),
            _exit(200.0, "X-USDC", 0.10, trade_id="t2", exit_price=2.2),
            _exit(210.0, "X-USDC", 0.20, trade_id="t1", exit_price=1.2),
        ]
        trades = self._snap(rows)
        self.assertEqual(len(trades), 2)
        by_profit = {round(t.profit, 2): t for t in trades}
        self.assertEqual(by_profit[0.10].entry_ts, 110.0)
        self.assertEqual(by_profit[0.20].entry_ts, 100.0)


if __name__ == "__main__":
    unittest.main()
