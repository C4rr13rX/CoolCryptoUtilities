"""A strategy that publishes but holds no symbol slot gets ZERO cycles.

THE FAILURE THIS PREVENTS. The 2026-09-10 brief reported "decision cycles,
last 6h: atf_static 179, everything else 7 between them" and concluded the
population was starved because one strategy ate the budget. That count came
from ``trading_ops``, which is an append-only op LOG: a strategy that
publishes bus actions writes rows there while receiving no decision cycles at
all, and a refused cycle writes no row. Measured against the cycle table, the
strategy with 179 log rows had ZERO cycles.

The converse does NOT hold, and asserting it here was retracted: a strategy
absent from the log was very likely still ASKED and simply returned no signal
(65 of 72 do, every tick). Absence proves silence, not starvation.

``scripts/decision_budget_census.py`` exists so that number is not re-derived
from the wrong table. These tests pin the two properties that make it right:
a cycle is attributed to whoever holds THAT SYMBOL'S slot, and a busy
publisher with no slot is credited with nothing.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.decision_budget_census import (
    NO_DIRECTIVE,
    breadth_trend,
    candidate_channel,
    census,
    slot_owners,
)


def _snapshot(symbol, action, owners, *, bus_publisher=None):
    """One organism snapshot: a decision on ``symbol`` plus the slot table.

    ``bus_publisher`` names a strategy that published work on this tick
    WITHOUT holding a slot -- the shape that produced the bad number.
    """
    scheduler = [
        {
            "symbol": name,
            "last_directive": ({"strategy_id": owner} if owner else None),
        }
        for name, owner in owners.items()
    ]
    payload = {
        "scheduler": scheduler,
        "decision": {"symbol": symbol, "action": action},
    }
    if bus_publisher:
        payload["activity"] = {
            "ghost_trades": [
                {
                    "action": "strategy_publish",
                    "details": {"bus_actions": [{"strategy_id": bus_publisher}]},
                }
            ]
        }
    return (0.0, json.dumps(payload))


def test_a_publisher_with_no_slot_is_credited_with_zero_cycles():
    """The exact shape that produced "atf_static 179 cycles"."""
    owners = {"AAA-USDC": "ema_cross", "BBB-USDC": "rsi_reversal@5h"}
    rows = [
        _snapshot("AAA-USDC", "hold", owners, bus_publisher="atf_static")
        for _ in range(50)
    ]
    report = census(rows, hours=1.0)

    credited = {row["strategy"]: row["cycles"] for row in report["per_strategy"]}
    # The publisher appears on every single snapshot and must still score nil:
    # publishing is not a decision cycle.
    assert "atf_static" not in credited, credited
    assert credited == {"ema_cross": 50}, credited


def test_a_cycle_is_attributed_to_the_owner_of_its_own_symbol():
    """Not to the busiest strategy, and not to the first slot in the list."""
    owners = {"AAA-USDC": "ema_cross", "BBB-USDC": "rsi_reversal@5h"}
    rows = [_snapshot("AAA-USDC", "hold", owners) for _ in range(9)]
    rows += [_snapshot("BBB-USDC", "hold", owners)]
    report = census(rows, hours=1.0)

    credited = {row["strategy"]: row["cycles"] for row in report["per_strategy"]}
    assert credited == {"ema_cross": 9, "rsi_reversal@5h": 1}, credited


def test_an_unclaimed_symbol_is_its_own_bucket_not_a_strategys():
    """Cycles on a symbol nobody has claimed are budget, but nobody's credit."""
    rows = [_snapshot("AAA-USDC", "hold", {"AAA-USDC": None}) for _ in range(4)]
    report = census(rows, hours=1.0)

    credited = {row["strategy"]: row["cycles"] for row in report["per_strategy"]}
    assert credited == {NO_DIRECTIVE: 4}, credited
    assert report["strategies_holding_a_slot"] == 0, report


def test_a_slot_that_never_changes_owner_reports_zero_changes():
    """Turnover is the number that says whether a floor of cycles is buildable."""
    rows = [_snapshot("AAA-USDC", "hold", {"AAA-USDC": "ema_cross"}) for _ in range(20)]
    report = census(rows, hours=1.0)

    assert report["slots"]["AAA-USDC"]["changes"] == 0, report["slots"]
    assert report["slot_changes"] == 0, report


def test_a_slot_handover_is_counted_once_not_once_per_snapshot():
    """A repeated owner must not inflate turnover the way op-log rows inflate."""
    rows = [_snapshot("AAA-USDC", "hold", {"AAA-USDC": "ema_cross"}) for _ in range(5)]
    rows += [_snapshot("AAA-USDC", "hold", {"AAA-USDC": "donchian_breakout"}) for _ in range(5)]
    report = census(rows, hours=1.0)

    assert report["slots"]["AAA-USDC"]["changes"] == 1, report["slots"]
    assert report["slots"]["AAA-USDC"]["distinct_owners"] == 2, report["slots"]


def test_hold_share_reaches_one_when_no_cycle_produces_an_entry():
    """The number that says redistributing cycles cannot produce an entry."""
    owners = {"AAA-USDC": "ema_cross", "BBB-USDC": "rsi_reversal@5h"}
    rows = [_snapshot("AAA-USDC", "hold", owners) for _ in range(3)]
    rows += [_snapshot("BBB-USDC", "hold", owners) for _ in range(3)]
    report = census(rows, hours=1.0)

    assert report["hold_share"] == 1.0, report
    assert report["entries"] == 0, report


def _ops_db(rows):
    """An in-memory ``trading_ops`` carrying ``(status, details)`` rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute("create table trading_ops (ts real, status text, details text)")
    conn.executemany(
        "insert into trading_ops values (?,?,?)",
        [(0.0, status, json.dumps(details)) for status, details in rows],
    )
    return conn


def test_a_strategy_named_only_inside_a_dropped_list_still_counts_as_proposed():
    """One refusal row charges several strategies, and each was proposed."""
    conn = _ops_db(
        [
            (
                "entry-predropped-edge-ban",
                {"dropped": [{"strategy_id": "obv_accumulation@1w"},
                             {"strategy_id": "donchian_breakout@1d"}]},
            ),
            ("published", {"bus_actions": [{"strategy_id": "atf_static"}]}),
        ]
    )
    channel = candidate_channel(conn, now=100.0, hours=1.0)

    assert channel["strategies"] == [
        "atf_static",
        "donchian_breakout@1d",
        "obv_accumulation@1w",
    ], channel


def test_a_strategy_in_neither_channel_is_silent_not_proven_starved():
    """Absence from both producers means SILENT, and no more than that.

    This test used to assert the opposite -- that a strategy in neither
    channel had never been offered a cycle. Gale's entry-arbitration rows
    falsified it: 65 of 72 strategies are asked every tick and return
    no_signal, writing no row anywhere. The census can prove a strategy
    produced no visible signal; it cannot prove it was denied a turn, and the
    set below is named accordingly.
    """
    registry = {"ema_cross", "rsi_reversal", "atf_static", "vwap_reversion"}
    report = census([_snapshot("AAA-USDC", "hold", {"AAA-USDC": "ema_cross"})], hours=1.0)
    conn = _ops_db([("published", {"bus_actions": [{"strategy_id": "atf_static"}]})])

    slot_holders = {
        row["strategy"] for row in report["per_strategy"] if row["strategy"] != NO_DIRECTIVE
    }
    proposed = slot_holders | set(candidate_channel(conn, now=100.0, hours=1.0)["strategies"])

    assert proposed == {"ema_cross", "atf_static"}, proposed
    # Silent in this window -- which is a question to take to the arbitration
    # rows, not a verdict about scheduling.
    assert sorted(registry - proposed) == ["rsi_reversal", "vwap_reversion"]


def test_the_trend_reports_symbols_beside_strategies_so_neither_is_quoted_alone():
    """The self-check that caught this script's own overstated conclusion.

    Read over one short window, a shrinking population looks like a permanent
    exclusion. The trend must therefore carry the SYMBOL count next to the
    strategy count in every bucket: a strategy count that falls while symbols
    fall under it has measured the feed, not the allocation.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("create table organism_snapshots (ts real, payload text)")
    conn.execute("create table trading_ops (ts real, status text, details text)")
    # Older bucket: two symbols, two owners. Newer bucket: one of each.
    wide = _snapshot("AAA-USDC", "hold", {"AAA-USDC": "ema_cross", "BBB-USDC": "rsi_reversal"})[1]
    narrow = _snapshot("AAA-USDC", "hold", {"AAA-USDC": "ema_cross"})[1]
    conn.execute("insert into organism_snapshots values (?,?)", (50.0, wide))
    conn.execute("insert into organism_snapshots values (?,?)", (150.0, narrow))

    # now=200s, a 200-second window in two buckets -> [0,100) then [100,200).
    trend = breadth_trend(conn, now=200.0, hours=200.0 / 3600.0, buckets=2)

    # symbols counts SLOTS that existed; slot_holders counts strategies that
    # actually drew a cycle. They are different numbers on purpose -- BBB-USDC
    # has an owner in the older bucket but no decision landed on it, so the
    # breadth fell from 2 symbols to 1 while holders stayed at 1 throughout.
    assert [row["symbols"] for row in trend] == [2, 1], trend
    assert [row["slot_holders"] for row in trend] == [1, 1], trend
    # Oldest first, so a shrinking population reads left to right.
    assert trend[0]["from_hours_ago"] > trend[1]["from_hours_ago"], trend


def test_slot_owners_reads_the_directive_not_the_symbol_order():
    owners = slot_owners(
        {
            "scheduler": [
                {"symbol": "aaa-usdc", "last_directive": {"strategy_id": "ema_cross"}},
                {"symbol": "BBB-USDC", "last_directive": {}},
            ]
        }
    )
    # Symbols are upper-cased so the decision's symbol matches the slot table.
    assert owners == {"AAA-USDC": "ema_cross", "BBB-USDC": NO_DIRECTIVE}, owners
