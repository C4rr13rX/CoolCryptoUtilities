"""A replayed stop must fill where the price IS, and must not invent a path.

Two failures this pins, both of which would have made scripts/stop_width_replay
report that a tighter stop rescues the live-tradeable ghost book when it does
not:

  1. FILLING AT THE LEVEL. A stop is a market order. If the replay booked the
     stop level instead of the tick that breached it, every replayed stop would
     lose exactly its width and never a cent more -- which is precisely the
     overshoot the real book takes (services/stop_survivability_gate.py: a 2%
     stop realised -12.41%). That flatters every tight width and would have
     inverted the verdict.

  2. TREATING AN UNPLACEABLE TRIP AS AN UNSTOPPABLE ONE. ``locate_path``
     returns None when no tick matches the recorded entry price and [] when the
     entry was the last tick before the exit. Collapsing the two would let a
     trip whose path could not be found count as a trip no stop could have
     touched, silently building the book out of the paths that happened to
     match.

Also pinned: the first breach wins (no re-entry is credited), and a path that
never breaches keeps the outcome the trip actually had.
"""

from __future__ import annotations

import sqlite3

import pytest

from scripts.stop_width_replay import locate_path, replay_row


def _path(*prices, start=1000.0, step=60.0):
    return [{"ts": start + i * step, "price": float(p)}
            for i, p in enumerate(prices)]


def _row(entry=100.0, qty=2.0, booked=-0.5):
    return {"entry_price": entry, "quantity": qty, "cgross": booked,
            "symbol": "T-USDC", "notional": entry * qty}


def test_a_stop_fills_at_the_breaching_tick_not_at_its_own_level():
    # Entry 100, a 2% stop, and the price gaps straight to 90 -- a 10% move in
    # one tick. The fill is 90, so the loss is -10.00 on 2 units, NOT the
    # -4.00 a fill at the 98 level would have booked.
    res = replay_row(_row(), _path(99.5, 90.0), width=0.02)
    assert res["stopped"] is True
    assert res["gross"] == pytest.approx((90.0 - 100.0) * 2.0)
    # The level-filling bug would produce this instead:
    assert res["gross"] != pytest.approx((98.0 - 100.0) * 2.0)


def test_the_first_breach_wins_because_no_re_entry_is_credited():
    # 97 breaches a 2% stop; 95 later is worse. The replay must take the FIRST
    # one -- crediting the later, or the best, would be modelling a re-entry
    # the replay explicitly does not simulate.
    res = replay_row(_row(), _path(99.0, 97.0, 101.0, 95.0), width=0.02)
    assert res["gross"] == pytest.approx((97.0 - 100.0) * 2.0)


def test_a_path_that_never_breaches_keeps_the_outcome_the_trip_actually_had():
    res = replay_row(_row(booked=+0.75), _path(99.5, 100.5, 101.0), width=0.02)
    assert res["stopped"] is False
    assert res["gross"] == pytest.approx(+0.75)


def test_an_empty_path_is_placed_but_unstoppable_not_a_free_win():
    # The entry was the last tick before the exit: there is no tick to stop on,
    # so the booked outcome stands. It must not be scored as a stop.
    res = replay_row(_row(booked=-0.30), [], width=0.0025)
    assert res["stopped"] is False
    assert res["gross"] == pytest.approx(-0.30)


def _db_with(rows):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE market_stream (id INTEGER PRIMARY KEY, ts REAL, "
                "chain TEXT, symbol TEXT, price REAL, volume REAL, raw TEXT)")
    con.executemany(
        "INSERT INTO market_stream (ts, chain, symbol, price, volume, raw) "
        "VALUES (?,?,?,?,?,?)",
        [(t, "base", s, p, 0.0, "{}") for t, s, p in rows])
    return con


def test_an_entry_price_no_tick_ever_published_is_unplaceable_not_empty():
    con = _db_with([(100.0, "T-USDC", 5.0), (200.0, "T-USDC", 6.0)])
    # 5.5 was never streamed, so the trip cannot be placed on the path.
    assert locate_path(con, {"symbol": "T-USDC", "ts": 300.0,
                             "entry_price": 5.5}) is None
    # 5.0 was, and the path is everything after it.
    path = locate_path(con, {"symbol": "T-USDC", "ts": 300.0,
                             "entry_price": 5.0})
    assert path is not None
    assert [p["price"] for p in path] == [6.0]


def test_the_entry_is_the_latest_matching_tick_not_the_earliest():
    # The same price printed twice. Placing the entry at the FIRST occurrence
    # would hand the trip a path it never lived through -- here, an extra dip
    # to 4.0 that happened before it ever opened.
    con = _db_with([(100.0, "T-USDC", 5.0), (150.0, "T-USDC", 4.0),
                    (200.0, "T-USDC", 5.0), (250.0, "T-USDC", 5.1)])
    path = locate_path(con, {"symbol": "T-USDC", "ts": 300.0,
                             "entry_price": 5.0})
    assert [p["price"] for p in path] == [5.1]


def test_an_entry_older_than_the_hold_ceiling_is_not_matched():
    from scripts.stop_width_replay import MAX_HOLD_SECS

    con = _db_with([(100.0, "T-USDC", 5.0),
                    (100.0 + MAX_HOLD_SECS + 500.0, "T-USDC", 7.0)])
    exit_ts = 100.0 + MAX_HOLD_SECS + 600.0
    assert locate_path(con, {"symbol": "T-USDC", "ts": exit_ts,
                             "entry_price": 5.0}) is None
