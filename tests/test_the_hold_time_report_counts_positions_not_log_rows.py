"""The destroyed-evidence count must be POSITIONS, not trading_ops rows.

``_abandon_dark_feed_positions`` walks ``self.positions``, which is the MERGED
book of every bot in the pool, and it logs one row per drop. Two things make
that a many-to-one map onto actual positions:

  * every bot in the pool sweeps the same merged book, so N bots log the same
    drop within the same sweep minute;
  * a position re-added from the persisted book is dropped again on a later
    sweep -- CRV-USDC produced 17 rows over 8.2 hours for ONE trade_id.

Measured on the live cache 2026-09-10: 90 rows for 20 positions. A report that
counted rows would have claimed 4.5x the evidence loss that actually happened,
and this item is an argument ABOUT that number -- inflating it would have
argued for weakening a guard that is behaving correctly.

The report is also the only place the abandoned population is visible at all:
every other number in hold_time_edge is drawn from ``trade_outcomes``, which
these positions never reach.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from scripts.hold_time_edge import STALE_EXIT_MINS, abandoned_positions


def _db(tmp_path, rows):
    path = tmp_path / "cache.db"
    con = sqlite3.connect(str(path))
    con.execute(
        "create table trading_ops (id integer primary key, ts real, wallet text,"
        " chain text, symbol text, action text, status text, details text)")
    for ts, detail in rows:
        con.execute(
            "insert into trading_ops (ts, status, symbol, details)"
            " values (?, 'position-abandoned-dark-feed', ?, ?)",
            (ts, detail.get("symbol", "?"), json.dumps(detail)))
    con.commit()
    con.close()
    return path


def _drop(symbol, tid, held_sec, mode="ghost", strategy="rsi_reversal"):
    return {
        "symbol": symbol,
        "released_mode": mode,
        "released_strategy_id": strategy,
        "released_trade_id": tid,
        "held_sec": held_sec,
        "silent_sec": 3600.0,
    }


def test_one_position_dropped_many_times_counts_once(tmp_path):
    """The CRV shape: 3 rows, 1 trade_id, 1 destroyed position."""
    tid = "2:CRV-USDC:78a5458c"
    path = _db(tmp_path, [
        (1000.0, _drop("CRV-USDC", tid, 3600.0)),
        (2000.0, _drop("CRV-USDC", tid, 4600.0)),
        (3000.0, _drop("CRV-USDC", tid, 5600.0)),
    ])

    rep = abandoned_positions(path, 0.0, now=4000.0)

    assert rep["rows"] == 3, "the log rows are still reported"
    assert rep["positions"] == 1, (
        "three drops of ONE trade_id are one destroyed position, not three -- "
        "counting rows overstated the live number by 4.5x")


def test_the_last_drop_sets_the_hold_time(tmp_path):
    """A position dropped repeatedly was held up to its FINAL drop."""
    tid = "2:CRV-USDC:78a5458c"
    path = _db(tmp_path, [
        (1000.0, _drop("CRV-USDC", tid, 3600.0)),
        (3000.0, _drop("CRV-USDC", tid, 5600.0)),
    ])

    rep = abandoned_positions(path, 0.0, now=4000.0)

    assert rep["max_held_mins"] == 5600.0 / 60.0, (
        "the earlier row is the same position earlier in its life, not a "
        "shorter one")


def test_distinct_positions_are_counted_separately(tmp_path):
    path = _db(tmp_path, [
        (1000.0, _drop("CRV-USDC", "2:CRV-USDC:aaa", 3600.0)),
        (1001.0, _drop("CRV-USDC", "2:CRV-USDC:aaa", 3600.0)),
        (2000.0, _drop("LINK-USDC", "2:LINK-USDC:bbb", 7200.0,
                       strategy="obv_accumulation@1d")),
    ])

    rep = abandoned_positions(path, 0.0, now=3000.0)

    assert rep["positions"] == 2
    assert rep["by_strategy"] == {"rsi_reversal": 1, "obv_accumulation@1d": 1}


def test_a_live_position_is_not_counted_as_abandoned_evidence(tmp_path):
    """The sweep never abandons a LIVE position; do not credit it with one.

    A live position is the only record of tokens the wallet actually holds, so
    ``_abandon_dark_feed_positions`` reports it and KEEPS it. Counting such a
    row as destroyed ghost evidence would double-count a position that is
    still open.
    """
    path = _db(tmp_path, [
        (1000.0, _drop("AERO-USDC", "2:AERO-USDC:live", 3600.0, mode="live")),
        (2000.0, _drop("CRV-USDC", "2:CRV-USDC:aaa", 3600.0)),
    ])

    rep = abandoned_positions(path, 0.0, now=3000.0)

    assert rep["positions"] == 1
    assert "2:AERO-USDC:live" not in json.dumps(rep)


def test_positions_held_past_four_times_the_promise_are_counted(tmp_path):
    """Criterion 3 of [71975c13] is measured on THIS population.

    stale_exit_secs promises a close at 15 minutes. The dark-feed sweep does
    not fire until 60. Every position it drops is therefore past 4x the
    promise by construction, and the report has to say so rather than leaving
    them out of the hold-time table entirely.
    """
    path = _db(tmp_path, [
        (1000.0, _drop("CRV-USDC", "2:CRV-USDC:aaa",
                       (4.0 * STALE_EXIT_MINS + 1.0) * 60.0)),
        (2000.0, _drop("LINK-USDC", "2:LINK-USDC:bbb",
                       (4.0 * STALE_EXIT_MINS - 1.0) * 60.0)),
    ])

    rep = abandoned_positions(path, 0.0, now=3000.0)

    assert rep["positions"] == 2
    assert rep["over_4x_stale"] == 1


def test_a_window_excludes_older_drops(tmp_path):
    path = _db(tmp_path, [
        (1000.0, _drop("CRV-USDC", "2:CRV-USDC:old", 3600.0)),
        (5000.0, _drop("LINK-USDC", "2:LINK-USDC:new", 3600.0)),
    ])

    rep = abandoned_positions(path, 4000.0, now=6000.0)

    assert rep["positions"] == 1
    assert rep["rows"] == 1


def test_a_missing_database_reports_nothing_rather_than_raising(tmp_path):
    """The report must not die on a box where the cache has not been created."""
    rep = abandoned_positions(tmp_path / "absent.db", 0.0, now=1.0)

    assert rep["positions"] == 0
    assert rep["rows"] == 0
