"""A strategy that publishes but holds no symbol slot gets ZERO cycles.

THE FAILURE THIS PREVENTS. The 2026-09-10 brief reported "decision cycles,
last 6h: atf_static 179, everything else 7 between them" and concluded the
population was starved because one strategy ate the budget. That count came
from ``trading_ops``, which is an append-only op LOG: a strategy that
publishes bus actions writes rows there while receiving no decision cycles at
all, and a refused cycle writes no row. Measured against the cycle table, the
strategy with 179 log rows had ZERO cycles and the real mechanism was
symbol-slot contention.

``scripts/decision_budget_census.py`` exists so that number is not re-derived
from the wrong table. These tests pin the two properties that make it right:
a cycle is attributed to whoever holds THAT SYMBOL'S slot, and a busy
publisher with no slot is credited with nothing.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.decision_budget_census import NO_DIRECTIVE, census, slot_owners


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
