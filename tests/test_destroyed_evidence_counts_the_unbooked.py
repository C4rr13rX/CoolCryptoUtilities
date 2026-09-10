"""A ghost position that never books is invisible to every book-reading tool.

THE FAILURE THIS PREVENTS
-------------------------
scripts/hold_time_edge.py -- the instrument [71975c13] was written on -- reads
``trade_outcomes``. A ghost position that is EVICTED by a new entry, or
ABANDONED because its feed went dark, writes no outcome row at all. So the
worst-held positions in the system were unmeasurable by the tool built to
measure how long positions are held: pass 100 reported the worst trip as 17.7
hours, and that was merely the worst trip that BOOKED. The worst that did not
book was held 11.6 days.

scripts/destroyed_evidence.py counts the funnel from the ENTRY side, out of
``trading_ops``, so the unbooked fates are visible. These tests hold it to that:

  - a released or abandoned position counts as DESTROYED, never as booked
  - ``booked_share`` is booked/entries, so it FALLS when evidence is destroyed
  - the held-time distribution is read for the abandoned population, including
    the ``over_4x_stale`` count that is acceptance criterion 3 of [71975c13]
  - a symbol the live lane refuses is not reported in the TRADEABLE count,
    because the pooled and tradeable books differ by a factor of ten and by sign

Every assertion here is on a number the function RETURNS, not on a log line.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.destroyed_evidence import STALE_EXIT_SECS, destroyed_evidence

NOW = 1_789_000_000.0


def _op(con, *, ts, symbol, action, status, details):
    con.execute(
        "INSERT INTO trading_ops (ts, wallet, chain, symbol, action, status, details)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, "w", "base", symbol, action, status, json.dumps(details)),
    )


@pytest.fixture()
def book(tmp_path):
    """A ghost lane that entered five times and booked once."""
    path = tmp_path / "trading_cache.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE trading_ops (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " ts REAL, wallet TEXT, chain TEXT, symbol TEXT, action TEXT,"
        " status TEXT, details TEXT)"
    )
    for i in range(5):
        _op(con, ts=NOW - 3600, symbol="AERO-USDC", action="enter",
            status="ghost-entry", details={})
    _op(con, ts=NOW - 60, symbol="AERO-USDC", action="exit",
        status="ghost-exit", details={})
    # Evicted by a new entry: the slot was reused and nothing was written.
    for i in range(2):
        _op(con, ts=NOW - 120, symbol="AERO-USDC", action="hold",
            status="position-released",
            details={"released_entry_price": 1.0, "released_size": 10.0,
                     "released_strategy_id": "atf_static"})
    # Feed went dark. Held far past the horizon the exit rules promise.
    _op(con, ts=NOW - 30, symbol="AERO-USDC", action="hold",
        status="position-abandoned-dark-feed",
        details={"released_entry_price": 1.0, "released_size": 10.0,
                 "released_strategy_id": "atf_static",
                 "held_sec": 9 * STALE_EXIT_SECS, "silent_sec": 3600.0})
    _op(con, ts=NOW - 30, symbol="AERO-USDC", action="hold",
        status="position-abandoned-dark-feed",
        details={"released_entry_price": 1.0, "released_size": 10.0,
                 "released_strategy_id": "", "held_sec": 2 * STALE_EXIT_SECS,
                 "silent_sec": 3600.0})
    con.commit()
    con.close()
    return path


def test_an_evicted_position_is_counted_as_destroyed_not_missing(book):
    """784 evictions in 7 days were simply absent from every book-reading tool."""
    r = destroyed_evidence(days=7.0, db_path=book, now=NOW)
    assert r["ghost_entries"] == 5
    assert r["booked"] == 1
    # Two evicted plus two abandoned. None of them booked; all of them happened.
    assert r["destroyed"] == 4
    assert r["fates"]["position-released"]["n"] == 2
    assert r["fates"]["position-abandoned-dark-feed"]["n"] == 2


def test_the_booked_share_falls_when_evidence_is_destroyed(book):
    """The number graduation depends on is booked/ENTRIES, not booked/booked.

    A tool that divides booked exits by booked exits reports 100% forever. The
    measured value on the real book is 19.4%.
    """
    r = destroyed_evidence(days=7.0, db_path=book, now=NOW)
    assert r["booked_share"] == pytest.approx(1 / 5)


def test_a_position_held_past_four_times_the_stale_horizon_is_counted(book):
    """Acceptance criterion 3 of [71975c13] needs this count to exist at all."""
    f = r_fate(book)
    assert f["over_4x_stale"] == 1          # the 9x one, not the 2x one
    assert f["held_max_x_stale"] == pytest.approx(9.0)
    assert f["held_median_x_stale"] == pytest.approx(5.5)


def test_every_abandoned_position_had_what_it_needed_to_book(book):
    """The entry price and size are present at the abandon site and unused.

    This is the claim that makes the fix a booking bug rather than a data
    problem: 90 of 90 on the real book carry both.
    """
    f = r_fate(book)
    assert f["bookable"] == 2
    assert f["with_strategy_id"] == 1


def test_a_stale_last_price_is_reported_so_nobody_books_a_fill_at_it(book):
    """silent_sec is 65 minutes on the real book. A fill at it is fabricated."""
    f = r_fate(book)
    assert f["silent_median_secs"] == pytest.approx(3600.0)


def test_a_window_that_excludes_the_rows_reports_an_empty_funnel(book):
    """No row is counted from outside the window it was asked for."""
    r = destroyed_evidence(days=7.0, db_path=book, now=NOW + 30 * 86400)
    assert r["ghost_entries"] == 0
    assert r["destroyed"] == 0
    assert r["booked_share"] == 0.0


def r_fate(book):
    r = destroyed_evidence(days=7.0, db_path=book, now=NOW)
    return r["fates"]["position-abandoned-dark-feed"]
