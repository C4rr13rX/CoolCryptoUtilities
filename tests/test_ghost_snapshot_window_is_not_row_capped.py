"""A time-bounded ghost book must not be silently truncated by a row cap.

``ghost_trade_snapshot`` takes BOTH a ``lookback_sec`` window and a ``limit``
counting trading_ops ROWS. Every production caller passes a window plus 500:

    production.py:733               _task_ghost_metrics
    trading/pipeline.py:3995        _ghost_validation      <- the live gate
    trading/pipeline.py:5554        _ghost_focus_assets
    trading/bot.py:2437             _refresh_live_transition
    revenir_service/task_runner.py:330

``fetch_trades`` is ORDER BY ts DESC, so the cap keeps the NEWEST rows. That
does not shrink the book evenly. Ghost entries outnumber ghost exits roughly
8:1 (measured 2026-09-03: 1062 entries against 134 exits in 48h), so the newest
rows are overwhelmingly entries whose exits have not happened yet, while the
completed round trips inside the window lose the older entry rows they need to
pair against -- and pairing is causal, so each of those exits is dropped as an
orphan.

Measured on the live 48h book at the time this was written:

    limit=500      atf_static  7 paired, net -0.16220 | pooled  28, -0.01032
    limit>=1500    atf_static 36 paired, net +0.53014 | pooled 134, +0.68025

The gate read the first row and reported ``insufficient_samples`` for
atf_static and ``negative_margin`` for the pooled book -- ``ghost_validation_block``,
the reason live trading was shut -- off a sample that hid 80% of the completed
trades and inverted the sign of the P&L on both.
"""

from __future__ import annotations

import os
import unittest

from trading.metrics import MetricsCollector


class _DB:
    """Stands in for TradingDatabase.fetch_trades: newest-first, then LIMIT."""

    def __init__(self, rows):
        self._rows = list(rows)
        self.last_limit = None

    def fetch_trades(self, *, limit=200, statuses=None, wallets=None,
                     symbol=None, since_ts=None):
        self.last_limit = int(limit)
        rows = [r for r in self._rows
                if since_ts is None or float(r["ts"]) >= float(since_ts)]
        rows.sort(key=lambda r: float(r["ts"]), reverse=True)
        return rows[: int(limit)]


def _entry(ts, symbol, price, strategy):
    return {
        "ts": ts,
        "status": "ghost-entry",
        "symbol": symbol,
        "wallet": "ghost",
        "details": {
            "symbol": symbol,
            "entry_ts": ts,
            "entry_price": price,
            "strategy_id": strategy,
            "trade_id": f"{symbol}-{ts}",
        },
    }


def _exit(ts, symbol, profit, entry_ts, strategy):
    return {
        "ts": ts,
        "status": "ghost-exit",
        "symbol": symbol,
        "wallet": "ghost",
        "details": {
            "symbol": symbol,
            "profit": profit,
            "entry_ts": entry_ts,
            "exit_ts": ts,
            "strategy_id": strategy,
            "trade_id": f"{symbol}-{entry_ts}",
        },
    }


class GhostSnapshotWindowTest(unittest.TestCase):
    def setUp(self):
        # now is far in the future of these synthetic stamps, so build the book
        # relative to real time and use a window that covers all of it.
        import time

        self.now = time.time()
        rows = []
        # 20 completed round trips, oldest. Each is a WIN, so if they are
        # dropped the book's sign flips -- the exact failure being pinned.
        for i in range(20):
            ts = self.now - 40000 + i * 10
            rows.append(_entry(ts, "AERO-USDC", 1.0, "atf_static"))
            rows.append(_exit(ts + 5, "AERO-USDC", +0.10, ts, "atf_static"))
        # 600 newer, still-open entries: the 8:1 flood that pushes the cap.
        for i in range(600):
            rows.append(_entry(self.now - 3600 + i, "BASENOUN-USDC", 1.0, "atf_static"))
        # 3 newer losing round trips.
        for i in range(3):
            ts = self.now - 600 + i * 10
            rows.append(_entry(ts, "CBETH-USDC", 1.0, "atf_static"))
            rows.append(_exit(ts + 5, "CBETH-USDC", -0.05, ts, "atf_static"))
        self.rows = rows
        self.db = _DB(rows)
        self.collector = MetricsCollector(self.db)

    def test_window_book_is_complete_despite_a_small_row_limit(self):
        trades = self.collector.ghost_trade_snapshot(limit=500, lookback_sec=86400)
        self.assertEqual(
            len(trades), 23,
            "all 23 completed round trips in the window must survive a 500-row cap",
        )
        net = sum(t.profit for t in trades)
        self.assertAlmostEqual(net, 20 * 0.10 - 3 * 0.05, places=6)
        self.assertGreater(net, 0.0, "the truncated book reported a LOSS on a winning record")

    def test_the_row_cap_was_actually_raised_for_a_windowed_call(self):
        self.collector.ghost_trade_snapshot(limit=500, lookback_sec=86400)
        self.assertGreater(
            self.db.last_limit, 500,
            "a windowed call must ask for more rows than the caller's trade-shaped cap",
        )

    def test_a_capped_book_would_have_lost_the_winners(self):
        """Pins WHY: with the old behaviour the same book reads as a loss."""
        capped = _DB(self.rows)
        trades = MetricsCollector(capped).ghost_trade_snapshot(
            limit=500, lookback_sec=None,
        )
        net = sum(t.profit for t in trades)
        self.assertEqual(capped.last_limit, 500,
                         "without a window the caller's cap is still the only bound")
        self.assertLess(len(trades), 23)
        self.assertLess(net, 0.0,
                        "this is the inverted record the live gate was refusing on")

    def test_no_window_still_honours_the_callers_limit(self):
        db = _DB(self.rows)
        MetricsCollector(db).ghost_trade_snapshot(limit=17, lookback_sec=None)
        self.assertEqual(db.last_limit, 17)

    def test_cap_is_configurable(self):
        db = _DB(self.rows)
        prior = os.environ.get("GHOST_SNAPSHOT_MAX_ROWS")
        os.environ["GHOST_SNAPSHOT_MAX_ROWS"] = "1234"
        try:
            MetricsCollector(db).ghost_trade_snapshot(limit=500, lookback_sec=86400)
        finally:
            if prior is None:
                os.environ.pop("GHOST_SNAPSHOT_MAX_ROWS", None)
            else:
                os.environ["GHOST_SNAPSHOT_MAX_ROWS"] = prior
        self.assertEqual(db.last_limit, 1234)

    def test_a_caller_asking_for_more_than_the_cap_keeps_its_own_limit(self):
        db = _DB(self.rows)
        MetricsCollector(db).ghost_trade_snapshot(limit=99999, lookback_sec=86400)
        self.assertEqual(db.last_limit, 99999)

    def test_strategy_filter_still_narrows_the_completed_book(self):
        trades = self.collector.ghost_trade_snapshot(
            limit=500, lookback_sec=86400, strategy_id="atf_static",
        )
        self.assertEqual(len(trades), 23)
        trades = self.collector.ghost_trade_snapshot(
            limit=500, lookback_sec=86400, strategy_id="money_button",
        )
        self.assertEqual(trades, [], "a strategy with no trades must read empty, not borrowed")


if __name__ == "__main__":
    unittest.main()
