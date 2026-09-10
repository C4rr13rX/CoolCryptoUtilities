"""Counting ``trading_ops`` rows as decision cycles overstates the waste 4x.

[ed0d721e] was filed as "75% of the ghost lane's decision budget re-decides
one banned symbol", measured by counting ``entry-predropped-edge-ban`` rows.
Measured 2026-09-10 those 96 rows in one hour were 25 ticks: four
``evaluate()`` calls land on a single tick and each logs its own drop. The
symbol took 13.1% of decision cycles, against an 11.5% share of ticks.

The second half of the same mistake is the head series: ``bot.py`` seeds its
prediction summary with ``direction_prob=0.5`` and ``net_margin=0.0`` when the
model produced no reading, so counting those as readings lifts a collapsed
head's MAXIMUM back to its ceiling and hides exactly the condition that closes
the entry conjunct.

Both are asserted here so the next census cannot re-derive the wrong number.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from scripts.hold_attribution_census import ban_row_multiplicity, census


def _snapshot(ts: float, symbol: str, action: str, direction_prob: float,
              net_margin: float):
    return (
        ts,
        json.dumps(
            {
                "sample": {"symbol": symbol},
                "decision": {
                    "symbol": symbol,
                    "action": action,
                    "direction_prob": direction_prob,
                    "net_margin": net_margin,
                },
            }
        ),
    )


def test_four_ban_rows_on_one_tick_are_one_tick_not_four() -> None:
    """The row count is 4x the tick count and the census must say so."""
    now = time.time()
    conn = sqlite3.connect(":memory:")
    conn.execute("create table trading_ops (ts real, symbol text, status text)")
    # Two ticks, 60s apart, each logging four separate drops at one instant --
    # the exact shape measured in trading_ops.
    for tick in (now - 600.0, now - 540.0):
        for offset in (0.0, 0.01, 0.02, 0.03):
            conn.execute(
                "insert into trading_ops values (?,?,?)",
                (tick + offset, "AERO-USDC", "entry-predropped-edge-ban"),
            )
    conn.commit()

    report = ban_row_multiplicity(conn, now=now, hours=1.0)

    assert report["rows"] == 8, "eight rows were written"
    # The pre-fix reading treated all eight as eight wasted decision cycles.
    assert report["clusters"] == 2, "but they land on exactly two ticks"
    assert report["rows_per_cluster"] == pytest.approx(4.0)


def test_a_no_prediction_default_does_not_lift_a_collapsed_head_maximum() -> None:
    """(0.5, 0.0) is the absence of a reading, not a reading at the ceiling."""
    now = time.time()
    rows = [
        _snapshot(now - 300.0, "AERO-USDC", "hold", 0.13, -1.20),
        _snapshot(now - 240.0, "AERO-USDC", "hold", 0.11, -0.95),
        # The sentinel bot.py writes when the model produced nothing.
        _snapshot(now - 180.0, "AERO-USDC", "hold", 0.5, 0.0),
    ]

    report = census(rows, now=now, bucket_hours=2.0)

    assert report["cycles"] == 3
    assert report["default_pairs"] == 1
    series = report["series"]
    assert len(series) == 1
    assert series[0]["n"] == 2, "the sentinel is excluded from the head series"
    # Counting the sentinel would report 0.500 / +0.000 -- a head that reaches
    # its ceiling and a conjunct that is satisfiable. Neither is true here.
    assert series[0]["direction_prob_max"] == pytest.approx(0.13)
    assert series[0]["net_margin_max"] == pytest.approx(-0.95)
    assert report["buckets_with_negative_net_margin_max"] == 1


def test_a_hold_is_counted_even_though_it_writes_no_trading_ops_row() -> None:
    """`bot.py` logs only when action != 'hold', so holds exist only here."""
    now = time.time()
    rows = [
        _snapshot(now - 100.0, "AERO-USDC", "hold", 0.10, -1.10),
        _snapshot(now - 90.0, "ZORA-USDC", "hold", 0.12, -1.40),
        _snapshot(now - 80.0, "VVV-USDC", "enter", 0.80, 0.40),
    ]

    report = census(rows, now=now, bucket_hours=2.0)

    assert report["cycles"] == 3
    assert report["holds"] == 2
    assert report["non_hold"] == 1
    assert report["hold_share"] == pytest.approx(2.0 / 3.0)
    assert report["by_symbol"]["AERO-USDC"] == {"hold": 1}
    # A positive maximum anywhere in the bucket means the conjunct WAS
    # satisfiable, however bad the median.
    assert report["buckets_with_negative_net_margin_max"] == 0


def test_an_unreadable_snapshot_is_not_counted_as_a_hold() -> None:
    """An unknown cycle must not inflate the very number this reports."""
    now = time.time()
    rows = [
        (now - 100.0, "{not json"),
        (now - 90.0, json.dumps({"sample": {"symbol": "AERO-USDC"}})),  # no decision
        _snapshot(now - 80.0, "AERO-USDC", "hold", 0.10, -1.10),
    ]

    report = census(rows, now=now, bucket_hours=2.0)

    assert report["cycles"] == 1
    assert report["holds"] == 1
