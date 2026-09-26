"""The metrics write path must not commit on the caller's thread.

[20e2f185]. Production calls TradingDatabase.record_metrics from the asyncio
loop thread, and py-spy caught that thread parked inside the executemany twice.
Measured against the live store on 2026-09-26: 28,029 metric rows over 5,618
distinct timestamps in one hour, i.e. **93.6 sqlite commits per minute on the
event loop** at 5 rows each. These tests pin the two properties that fix rests
on -- the caller does no sqlite work, and N calls become ONE transaction.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from db import TradingDatabase


@pytest.fixture()
def dbh(tmp_path):
    handle = TradingDatabase(str(tmp_path / "metrics.db"))
    yield handle
    handle._metrics_stop.set()


def _raw_count(dbh) -> int:
    conn = sqlite3.connect(dbh.path)
    try:
        return conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    finally:
        conn.close()


def test_record_metrics_does_not_touch_sqlite_on_the_caller(dbh, monkeypatch):
    """The whole point: the calling thread must not reach the connection."""
    dbh._metrics_stop.set()  # keep the flusher from racing this assertion

    def _boom(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("record_metrics executed sqlite on the caller thread")

    monkeypatch.setattr(dbh, "_flush_metric_rows", _boom)
    dbh.record_metrics(stage="data_stream", metrics={"price": 1.0, "volume": 2.0})

    assert len(dbh._metrics_buffer) == 2


def test_many_calls_coalesce_into_one_transaction(dbh):
    """40 calls must not be 40 commits -- that is the starvation mechanism."""
    dbh._metrics_stop.set()
    commits: list[int] = []
    original = dbh._flush_metric_rows

    def _counting(rows):
        commits.append(len(rows))
        original(rows)

    dbh._flush_metric_rows = _counting  # type: ignore[method-assign]
    for i in range(40):
        dbh.record_metrics(stage="live_trading", metrics={"errors": float(i)})

    assert commits == []  # nothing written yet
    written = dbh.flush_metrics()

    assert written == 40
    assert commits == [40], "40 record_metrics calls must become ONE executemany"
    assert _raw_count(dbh) == 40


def test_fetch_metrics_sees_unflushed_rows(dbh):
    """Read-your-writes survives the buffering, or every dashboard lies."""
    dbh._metrics_stop.set()
    dbh.record_metrics(stage="pipeline", metrics={"ghost_multiplier": 0.5})

    rows = dbh.fetch_metrics(stage="pipeline", limit=10)

    assert [(r["name"], r["value"]) for r in rows] == [("ghost_multiplier", 0.5)]


def test_buffer_is_bounded_and_drops_oldest(dbh, monkeypatch):
    """A process that stops draining must not grow without bound."""
    dbh._metrics_stop.set()
    monkeypatch.setattr(TradingDatabase, "_METRICS_BUFFER_MAX", 1000)
    for i in range(300):
        dbh.record_metrics(
            stage="data_stream",
            metrics={f"m{j}": float(i) for j in range(5)},
        )

    assert len(dbh._metrics_buffer) == 1000
    assert dbh._metrics_dropped == 500
    # oldest dropped, newest kept: the last row buffered is from the last call
    assert dbh._metrics_buffer[-1][4] == 299.0


def test_background_flusher_drains_without_an_explicit_flush(dbh, monkeypatch):
    """The loop thread never calls flush_metrics -- the thread has to."""
    monkeypatch.setattr(TradingDatabase, "_METRICS_FLUSH_INTERVAL_SEC", 0.1)
    dbh.record_metrics(stage="ghost_trading", metrics={"held_sec": 73012.3})

    deadline = time.time() + 10.0
    while time.time() < deadline and _raw_count(dbh) == 0:
        time.sleep(0.1)

    assert _raw_count(dbh) == 1
    assert dbh._metrics_flusher is not None and dbh._metrics_flusher.is_alive()
