"""The news crawl must not hold the database lock once per day-per-symbol.

Observed 2026-09-02: `collect_news_for_terms` walked a 30-day window for every
symbol key, doing a separate `get_json`/`set_json` -- and therefore a separate
acquisition of the single global database lock -- for each cell. A 30-day, 5-key
window was ~600 locked round trips. Every other thread queued behind them, the
90s `data_ingest` timeout fired, and the abandoned worker kept holding the lock
while the next cycle started another one.

These tests pin two things: the batch KV helpers behave exactly like the
per-key ones they replace, and the news bookkeeping does the whole window in a
constant number of database round trips rather than one per cell.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from db import TradingDatabase
from services.news_lab import (
    NEWS_LOG_PREFIX,
    _load_seen_entries,
    _record_news_attempt,
)


@pytest.fixture()
def db(tmp_path):
    return TradingDatabase(path=str(tmp_path / "news_kv.db"))


class TestBatchKVMatchesPerKey:
    def test_get_json_many_reads_what_set_json_wrote(self, db):
        db.set_json("a", {"n": 1})
        db.set_json("b", {"n": 2})
        assert db.get_json_many(["a", "b"]) == {"a": {"n": 1}, "b": {"n": 2}}

    def test_set_json_many_is_readable_one_key_at_a_time(self, db):
        db.set_json_many({"a": {"n": 1}, "b": {"n": 2}})
        assert db.get_json("a") == {"n": 1}
        assert db.get_json("b") == {"n": 2}

    def test_missing_keys_are_absent_not_none(self, db):
        db.set_json("present", {"n": 1})
        out = db.get_json_many(["present", "missing"])
        assert out == {"present": {"n": 1}}
        assert "missing" not in out

    def test_set_json_many_overwrites_like_set_json(self, db):
        db.set_json("a", {"n": 1})
        db.set_json_many({"a": {"n": 99}})
        assert db.get_json("a") == {"n": 99}

    def test_empty_inputs_are_no_ops(self, db):
        assert db.get_json_many([]) == {}
        db.set_json_many({})  # must not raise


class _CountingDB:
    """Wraps a real database and counts lock-taking round trips."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = {"get_json": 0, "set_json": 0, "get_json_many": 0, "set_json_many": 0}

    def get_json(self, key):
        self.calls["get_json"] += 1
        return self._inner.get_json(key)

    def set_json(self, key, payload):
        self.calls["set_json"] += 1
        return self._inner.set_json(key, payload)

    def get_json_many(self, keys):
        self.calls["get_json_many"] += 1
        return self._inner.get_json_many(keys)

    def set_json_many(self, payloads):
        self.calls["set_json_many"] += 1
        return self._inner.set_json_many(payloads)

    @property
    def round_trips(self):
        return sum(self.calls.values())


def _window(days):
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    end = datetime(2026, 8, days, tzinfo=timezone.utc)
    return start, end


class TestNewsBookkeepingIsBatched:
    def test_recording_a_30_day_window_is_a_constant_number_of_round_trips(self, db):
        counting = _CountingDB(db)
        start, end = _window(30)
        keys = ["BTC", "ETH", "USDC", "ARB", "SOL"]
        items = [
            {"url": "u1", "title": "t1", "source": "x", "timestamp": int(start.timestamp())},
        ]

        _record_news_attempt(counting, keys, start, end, items)

        # 30 days x 5 keys = 150 cells. The old code took 300 locks for this.
        assert counting.calls["get_json"] == 0
        assert counting.calls["set_json"] == 0
        assert counting.round_trips == 2, counting.calls

    def test_loading_seen_entries_is_one_round_trip(self, db):
        counting = _CountingDB(db)
        start, end = _window(30)
        _load_seen_entries(counting, ["BTC", "ETH"], start, end)
        assert counting.calls["get_json"] == 0
        assert counting.round_trips == 1, counting.calls

    def test_round_trips_do_not_grow_with_the_window(self, db):
        start, end = _window(30)
        wide = _CountingDB(db)
        _record_news_attempt(wide, ["BTC", "ETH", "USDC"], start, end, [])

        narrow = _CountingDB(db)
        s2, e2 = _window(2)
        _record_news_attempt(narrow, ["BTC"], s2, e2, [])

        assert wide.round_trips == narrow.round_trips


class TestNewsBookkeepingStillRecords:
    """Batching must not change what gets stored."""

    def test_urls_and_titles_round_trip_through_the_batched_write(self, db):
        start, end = _window(1)
        keys = ["BTC"]
        items = [
            {"url": "http://a", "title": "Alpha", "source": "Feed1",
             "timestamp": int(start.timestamp())},
            {"url": "http://b", "title": "Beta", "source": "feed2",
             "timestamp": int(start.timestamp())},
        ]
        _record_news_attempt(db, keys, start, end, items)

        entry = db.get_json(f"{NEWS_LOG_PREFIX}:BTC:2026-08-01")
        assert entry["urls"] == ["http://a", "http://b"]
        assert entry["titles"] == ["Alpha", "Beta"]
        assert entry["sources"]["feed1"] == ["http://a"]
        assert entry["attempts"] == 1

        seen_urls, seen_titles = _load_seen_entries(db, keys, start, end)
        assert seen_urls == {"http://a", "http://b"}
        assert seen_titles == {"Alpha", "Beta"}

    def test_attempts_accumulate_across_calls(self, db):
        start, end = _window(1)
        for _ in range(3):
            _record_news_attempt(db, ["ETH"], start, end, [])
        entry = db.get_json(f"{NEWS_LOG_PREFIX}:ETH:2026-08-01")
        assert entry["attempts"] == 3

    def test_a_second_call_does_not_drop_earlier_urls(self, db):
        start, end = _window(1)
        ts = int(start.timestamp())
        _record_news_attempt(db, ["ARB"], start, end,
                             [{"url": "http://one", "title": "One", "source": "s", "timestamp": ts}])
        _record_news_attempt(db, ["ARB"], start, end,
                             [{"url": "http://two", "title": "Two", "source": "s", "timestamp": ts}])
        entry = db.get_json(f"{NEWS_LOG_PREFIX}:ARB:2026-08-01")
        assert entry["urls"] == ["http://one", "http://two"]
