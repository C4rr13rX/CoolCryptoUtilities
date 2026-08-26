#!/usr/bin/env python3
"""
Backfill ghost outcomes that were logged but never reached the strategy ledger.

``services/atf_static_strategy.py`` ran its own ghost cycle and wrote closed
trades to ``trading_ops`` without calling ``StrategyLedger.record()``. The
outcomes are real and already durable -- they simply never reached the gate
that decides graduation, so the ledger sat unchanged while the strategy traded.

This replays them from ``trading_ops`` in timestamp order, which is the same
order ``record()`` would have seen them live. That matters: the ledger tracks
consecutive losses and a running confidence EMA, so replaying out of order
would produce a different (and wrong) state.

Idempotent by high-water mark: only exits newer than the ledger's recorded
``last_ts`` for that strategy are replayed, so running it twice does not
double-count.

    python scripts/backfill_ghost_ledger.py --dry-run
    python scripts/backfill_ghost_ledger.py --yes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_exits(db_path: Path) -> list[tuple[float, str, float]]:
    """(ts, strategy_id, profit) for every logged ghost exit, oldest first."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: list[tuple[float, str, float]] = []
    for row in conn.execute(
        "SELECT ts, details FROM trading_ops "
        "WHERE status = 'ghost-exit' ORDER BY ts"
    ):
        try:
            detail = json.loads(row["details"] or "{}")
        except (TypeError, ValueError):
            continue
        if "profit" not in detail:
            continue
        sid = str(detail.get("strategy_id") or "").strip() or "unclassified"
        try:
            profit = float(detail["profit"])
        except (TypeError, ValueError):
            continue
        out.append((float(row["ts"]), sid, profit))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", default=str(ROOT / "storage" / "trading_cache.db"))
    ap.add_argument("--yes", action="store_true", help="actually write")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dry = args.dry_run or not args.yes
    db_path = Path(args.sqlite)
    if not db_path.exists():
        print(f"trading database not found: {db_path}")
        return 1

    from trading.strategies.ledger import StrategyLedger

    ledger = StrategyLedger()
    data = getattr(ledger, "_data", {})

    exits = load_exits(db_path)
    print("=" * 66)
    print(f"  Ghost ledger backfill — {'DRY RUN' if dry else 'WRITING'}")
    print("=" * 66)
    print(f"  logged ghost exits in trading_ops: {len(exits)}")

    # High-water mark per strategy: anything at or before the ledger's last
    # recorded timestamp is already counted.
    watermark: dict[str, float] = {}
    for sid, entry in data.items():
        watermark[sid] = float((entry.get("ghost") or {}).get("last_ts", 0.0))

    pending = [
        (ts, sid, profit)
        for ts, sid, profit in exits
        if ts > watermark.get(sid, 0.0)
    ]
    print(f"  newer than the ledger's high-water mark: {len(pending)}")

    if not pending:
        print("\n  Nothing to backfill.")
        print("=" * 66)
        return 0

    by_strategy: dict[str, list[float]] = {}
    for _ts, sid, profit in pending:
        by_strategy.setdefault(sid, []).append(profit)

    print()
    for sid, profits in sorted(by_strategy.items()):
        wins = sum(1 for p in profits if p > 0)
        print(f"    {sid:24s} {len(profits):>5} trades "
              f"{wins / len(profits) * 100:5.1f}% win  "
              f"P/L {sum(profits):+.4f}")

    if dry:
        print("\n  re-run with --yes to apply")
        print("=" * 66)
        return 0

    # Oldest first: consecutive-loss counts and the confidence EMA are
    # order-dependent, so replaying out of order would corrupt them.
    for _ts, sid, profit in pending:
        ledger.record(sid, profit=profit, mode="ghost")

    print(f"\n  recorded {len(pending)} outcomes")
    approved = [
        sid for sid, entry in getattr(ledger, "_data", {}).items()
        if entry.get("live_approved")
    ]
    print(f"  live-approved after backfill: {approved or 'none'}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
