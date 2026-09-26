"""A print that is not a price for this symbol must not manufacture a ban.

Dune, pass 715, item [a46e065b].

MEASURED over 30 days of ``market_stream``: 1553 of 199220 rows (0.78%) across
49 of 496 symbols sit more than 50x from their own symbol's median, and the
newest was 0.58 days old, so the writer -- ``db.py:1588 insert_market_sample``,
which validates nothing -- still produces them.

``stop_survivability_gate`` bans a symbol when the p99 adjacent-tick jump
exceeds ``STOP_PCT * MAX_JUMP_RATIO`` (4.00% by default). Its ``MAX_TICK_GAP_SEC``
cap bounds how far apart two prints may be in TIME but not how far apart they
may be in MAGNITUDE, so one fabricated print inside the gap was scored at full
size -- twice, once in and once out. Against the shipped module this banned
PEPE-USDC off nine rows, TRUMP-USDC off fifteen and DOT-USDC off thirty-nine.

The two tests below are the pair that matters. The first is the fix; the
second is the thing the fix must not break, because a filter that exonerates
every symbol is worth nothing.
"""

import importlib
import os
import sqlite3
import time

import pytest


def _stream_db(tmp_path, rows):
    # A unique file per call: a test that builds the stream twice must not
    # reopen the first one and try to CREATE TABLE into it again.
    _stream_db.counter = getattr(_stream_db, "counter", 0) + 1
    path = tmp_path / ("stream-%d.db" % _stream_db.counter)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE market_stream (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts REAL, chain TEXT, symbol TEXT, price REAL, volume REAL, raw TEXT)"
    )
    conn.executemany(
        "INSERT INTO market_stream (ts, chain, symbol, price) VALUES (?, 'base', ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(path)


def _gate_over(monkeypatch, tmp_path, rows):
    """Load the gate against a throwaway stream containing exactly ``rows``."""
    monkeypatch.setenv("STOP_SURVIVE_CACHE_SEC", "0")
    gate = importlib.reload(
        importlib.import_module("services.stop_survivability_gate")
    )
    # DB_PATH is a module constant built from __file__, not an env var, so it
    # has to be replaced after the reload or the gate reads production.
    gate.DB_PATH = _stream_db(tmp_path, rows)
    gate._cache_built_at = 0.0
    return gate


def _calm_series(symbol, now, count=100, step=30.0, base=1.0):
    """A symbol a 2% stop binds on comfortably: alternating +/-0.2% ticks.

    THIN ON PURPOSE, and this is the part that took a failing test to learn.
    At 400 ticks a single fabricated print contributes two bad jumps out of
    401, and ``_percentile(jumps, 0.99)`` indexes the 397th of 401 -- it
    steps straight over them and returns 0.2% whether the filter runs or not.
    A 400-tick fixture therefore passes for the WRONG REASON and cannot
    distinguish the fix from its absence.

    Production's bans are on thin feeds for exactly this reason: PEPE-USDC
    had 121 jumps, TRUMP-USDC 51, DOT-USDC 85, and at that width two bad
    jumps ARE the top 1%. ``MIN_BREACHES`` (2) is what lets a feed below
    ``MIN_TICKS`` (200) carry a ban at all. 100 ticks reproduces the shape
    the gate actually bans on.
    """
    out = []
    for index in range(count):
        price = base * (1.002 if index % 2 else 1.0)
        out.append((now - (count - index) * step, symbol, price))
    return out


def test_a_fabricated_print_cannot_ban_a_symbol(monkeypatch, tmp_path):
    now = time.time()
    rows = _calm_series("FAKE-USDC", now)

    gate = _gate_over(monkeypatch, tmp_path, rows)
    assert gate.refusal_reason("FAKE-USDC") is None, (
        "the clean series must not be banned, or this test proves nothing"
    )

    # One print 4175x the median, the XRP-USDC shape measured in production
    # (1.4216789307296556 -> 5937.095685263188), landing 30s from its
    # neighbours so MAX_TICK_GAP_SEC cannot drop the pair.
    contaminated = list(rows)
    contaminated.insert(50, (now - 50 * 30.0 + 15.0, "FAKE-USDC", 4175.0))

    gate = _gate_over(monkeypatch, tmp_path, contaminated)
    assert gate.refusal_reason("FAKE-USDC") is None, (
        "one fabricated print banned a symbol whose real feed never moves more "
        "than 0.2% between ticks: " + str(gate.refusal_reason("FAKE-USDC"))
    )


def test_a_genuinely_unstoppable_symbol_is_still_banned(monkeypatch, tmp_path):
    """The filter must not exonerate a symbol that really is too jumpy.

    Every tick moves 30%, which is 7.5x the 4.00% ceiling, and no print is
    anywhere near 50x the median -- so the scale filter has nothing to remove
    and the ban must survive it. This is the XRP-USDC/SPIKE-USDC case: in
    production those two stayed banned at 5.755% and 2845.02% after their bad
    prints were dropped.
    """
    now = time.time()
    rows = []
    for index in range(400):
        rows.append((now - (400 - index) * 30.0, "JUMPY-USDC", 1.3 if index % 2 else 1.0))

    gate = _gate_over(monkeypatch, tmp_path, rows)
    reason = gate.refusal_reason("JUMPY-USDC")
    assert reason is not None, (
        "a symbol moving 30% between adjacent ticks must still be refused; "
        "a filter that exonerates everything is not a filter"
    )
    assert "stop cannot bind" in reason


def test_the_scale_limit_is_configurable_and_disablable(monkeypatch, tmp_path):
    """0 restores the old behaviour, so the fix can be turned off in one env var."""
    now = time.time()
    contaminated = _calm_series("FAKE-USDC", now)
    contaminated.insert(50, (now - 50 * 30.0 + 15.0, "FAKE-USDC", 4175.0))

    monkeypatch.setenv("STOP_SURVIVE_MAX_PRICE_SCALE", "0")
    gate = _gate_over(monkeypatch, tmp_path, contaminated)
    assert gate.PRICE_SCALE_LIMIT == 0.0
    assert gate.refusal_reason("FAKE-USDC") is not None, (
        "with the filter disabled the fabricated print must ban again, "
        "otherwise the filter is not what changed the verdict"
    )
