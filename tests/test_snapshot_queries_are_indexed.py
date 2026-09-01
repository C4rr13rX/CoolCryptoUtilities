"""The per-tick snapshot must not full-scan the database to find 24 rows.

`TradingBot._handle_sample` -- the callback every market stream feeds -- calls
`_record_organism_snapshot` -> `build_snapshot` -> four "newest N rows"
queries, synchronously, on the event loop all ~20 streams share.

`feedback_events` and `metrics` each carried a composite index whose leading
column is `source`/`stage`, which cannot serve an unfiltered `ORDER BY ts
DESC`; `trading_ops` and `market_stream` had no index at all. So each of those
queries scanned the whole table and sorted it in a temp b-tree. Measured on
the live database 2026-09-01:

    feedback_events   162,731 rows   129 ms   SCAN + TEMP B-TREE
    metrics           167,848 rows   229 ms   SCAN + TEMP B-TREE
    trading_ops        61,488 rows   161 ms   SCAN (no index existed)

That is ~0.68s of blocking scans per snapshot, at ORGANISM_SNAPSHOT_INTERVAL=5s
per bot. With ~18 bots that is a snapshot every 0.28s costing 0.68s -- the loop
is oversubscribed roughly 2.4x and can never catch up. py-spy caught it 6 dumps
out of 6 parked in `fetch_feedback_events`, and the ticks showed the
matching signature: six symbols writing within 0.0s of each other, then 20-45s
of nothing.

These tests assert the planner uses an index instead of scanning, which is the
property that keeps the callback off the loop's critical path. They check the
plan rather than a stopwatch so they mean the same thing on a fast machine.
"""

from __future__ import annotations

import sqlite3
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from db import TradingDatabase

#: The four "newest N rows" reads build_snapshot issues per snapshot.
SNAPSHOT_QUERIES = {
    "feedback_events": (
        "SELECT ts, source, severity, label, details FROM feedback_events "
        "ORDER BY ts DESC LIMIT 24"
    ),
    "metrics": (
        "SELECT ts, stage, category, name, value, meta FROM metrics "
        "ORDER BY ts DESC LIMIT 24"
    ),
    "trading_ops": (
        "SELECT ts, wallet, chain, symbol, action, status, details FROM trading_ops "
        "WHERE wallet IN ('ghost') ORDER BY ts DESC LIMIT 24"
    ),
    #: not build_snapshot, but the same shape and it runs per tick during
    #: quote corroboration (db.recent_prices / get_market_price).
    "market_stream": (
        "SELECT price, ts FROM market_stream "
        "WHERE symbol='AERO-USDC' AND chain='base' AND price > 0 "
        "ORDER BY ts DESC LIMIT 25"
    ),
}


class SnapshotQueriesAreIndexed(unittest.TestCase):
    ROWS = 5000

    def setUp(self) -> None:
        # The sqlite connection TradingDatabase holds keeps the file open, and
        # Windows refuses to unlink an open file.
        self._tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        path = Path(self._tmp.name) / "indexed.db"
        # Building through TradingDatabase is the point: the schema and its
        # indexes must arrive together, so an existing deployment picks the
        # indexes up on its next start.
        self.db = TradingDatabase(path=str(path))
        self.conn = sqlite3.connect(str(path))
        self.addCleanup(self.conn.close)
        self._populate()

    def _populate(self) -> None:
        """Give the planner a table worth planning for.

        On an empty table sqlite reasonably picks a scan whatever indexes
        exist, so an unpopulated fixture would pass with no index at all.
        """
        now = time.time()
        rows = range(self.ROWS)
        with self.conn:
            self.conn.executemany(
                "INSERT INTO feedback_events(ts, source, severity, label, details) "
                "VALUES (?, ?, ?, ?, ?)",
                [(now - i, "market_stream", "warning", "endpoint_failure", "{}") for i in rows],
            )
            self.conn.executemany(
                "INSERT INTO metrics(ts, stage, category, name, value, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(now - i, "data_stream", "dedupe", "price", 1.0, "{}") for i in rows],
            )
            self.conn.executemany(
                "INSERT INTO trading_ops(ts, wallet, chain, symbol, action, status, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(now - i, "ghost", "base", "AERO-USDC", "buy", "ghost-entry", "{}") for i in rows],
            )
            self.conn.executemany(
                "INSERT INTO market_stream(ts, chain, symbol, price, volume, raw) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(now - i, "base", "AERO-USDC", 1.14, 0.0, "{}") for i in rows],
            )

    def _plan(self, query: str) -> str:
        rows = list(self.conn.execute("EXPLAIN QUERY PLAN " + query))
        return " | ".join(str(row[3]) for row in rows)

    def test_no_snapshot_query_scans_its_table(self):
        for table, query in SNAPSHOT_QUERIES.items():
            with self.subTest(table=table):
                plan = self._plan(query)
                # An ORDER BY ts walk shows up as "SCAN t USING INDEX ..." --
                # that reads the index in order and stops at LIMIT, so it is
                # the good case. What must not appear is a sort of the whole
                # table, or a scan with no index behind it.
                self.assertNotIn(
                    "TEMP B-TREE", plan,
                    f"{table}: sorting the whole table to answer LIMIT 24 -- {plan}",
                )
                self.assertIn(
                    "USING", plan,
                    f"{table}: no index used, so the cost grows with the table -- {plan}",
                )

    def test_recent_rows_stay_cheap_as_the_table_grows(self):
        """A behavioural backstop: volume must not change the cost much."""
        now = time.time()
        with self.conn:
            self.conn.executemany(
                "INSERT INTO feedback_events(ts, source, severity, label, details) "
                "VALUES (?, ?, ?, ?, ?)",
                [(now - i, "market_stream", "warning", "endpoint_failure", "{}")
                 for i in range(40000)],
            )
        query = SNAPSHOT_QUERIES["feedback_events"]
        list(self.conn.execute(query))  # warm the page cache; measure the plan
        started = time.perf_counter()
        rows = list(self.conn.execute(query))
        elapsed = time.perf_counter() - started
        self.assertEqual(len(rows), 24)
        self.assertLess(
            elapsed, 0.05,
            f"newest 24 of 40k rows took {elapsed * 1000:.0f}ms -- that cost is "
            f"paid on the event loop every stream shares",
        )


if __name__ == "__main__":
    unittest.main()
