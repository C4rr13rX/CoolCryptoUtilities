#!/usr/bin/env python3
"""Give the ATF scout its own ledger identity, and hand ``atf_static`` back
to the executor that can actually spend money.

WHY
---
Link 9 (LIVE) failed with "no live trades yet" while every bot-level gate was
open. Measured 2026-09-02 against the database:

  * the bot reached its live entry gate 264 times in 24h and the swap guard
    PASSED 94 of them -- yet not one became a live trade, because
    ``_strategy_live_approved(directive)`` was False for every directive that
    arrived there (obv_accumulation@5d, rsi_reversal@5d, ...);
  * the ONLY strategy with ``live_approved: true`` is ``atf_static``;
  * and of the 376 closed ``atf_static`` trades in ``trading_ops``, **368
    (97.9%) were executed by services/atf_static_strategy.py's ghost scout**,
    which hardcodes ``wallet="ghost"`` / ``mode="ghost"`` and publishes
    ``live_execution_enabled: False``. It has no live branch at all.

So the graduation was credited to an executor that cannot spend, and the
executors that can spend were never graduated. The strategy allowed to trade
live was structurally incapable of it; that is the root cause, and no amount
of loosening a guard downstream would have produced a live trade.

The two executors share a signal source and nothing else. The scout enters on
its own corroborated quote and exits on an 8% stop, a 1h hold or its target;
the bot enters through the CDCL solver and exits on triggers, a 2% stop,
confidence drops and timed exits. Same idea, different realised P/L. One
ledger id cannot stand for both.

WHAT THIS DOES
--------------
  * ledger + registry: ``atf_static`` -> ``atf_static_scout``, carrying its
    stats intact, with ``live_approved`` cleared. A record that cannot be
    spent must not read as permission to spend: ``approved_ids()`` drives
    ``_refresh_auto_execute`` (which flips real execution on) and picks whose
    ghost book is judged for live, so leaving the flag on a ghost-only
    executor would keep authorising trades on evidence that is not about the
    thing being authorised.
  * ``atf_static`` is then rebuilt from the bot-executed exits ONLY, replayed
    from ``trading_ops`` in timestamp order (the ledger tracks consecutive
    losses and a confidence EMA, so order matters).

The 97.9%/2.1% split is applied wholesale to the historical aggregates rather
than recomputed tick by tick: the registry's lifetime counters predate parts
of ``trading_ops`` and re-deriving them would silently drop the outcomes the
plausibility filter had already rejected. The error this leaves is at most
2.1%, entirely in the scout's column, and zero in the direction that matters
-- nothing unearned is credited to the executor that spends real money.

Idempotent: re-running after the split is a no-op.

    python scripts/split_scout_from_bot_ledger.py --dry-run
    python scripts/split_scout_from_bot_ledger.py --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEDGER = ROOT / "data" / "strategy_ledger.json"
REGISTRY = ROOT / "data" / "strategy_registry.json"
DB = ROOT / "storage" / "trading_cache.db"

BOT_ID = "atf_static"
SCOUT_ID = "atf_static_scout"
SCOUT_SOURCE = "c0d3rv2_atf_static"

DEMOTE_REASON = (
    "ghost-only executor: services/atf_static_strategy.py has no live branch, "
    "so this record can never be spent and must not read as permission to spend"
)


def bot_executed_exits() -> List[Tuple[float, float, str]]:
    """(ts, profit, symbol) for atf_static exits the BOT closed, oldest first.

    The scout stamps every row it writes with ``details.source`` = the module
    constant ``SOURCE``; the bot's exit path does not. That field is the only
    per-row record of which executor took the trade, which is why the two were
    indistinguishable to the ledger.
    """
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: List[Tuple[float, float, str]] = []
    try:
        rows = conn.execute(
            "SELECT ts, symbol, details FROM trading_ops "
            "WHERE status LIKE '%exit%' ORDER BY ts"
        )
        for row in rows:
            try:
                detail = json.loads(row["details"] or "{}")
            except (TypeError, ValueError):
                continue
            if str(detail.get("strategy_id") or "") not in {BOT_ID, SCOUT_ID}:
                continue
            if str(detail.get("source") or "") == SCOUT_SOURCE:
                continue                      # the scout's, not the bot's
            if "profit" not in detail:
                continue
            try:
                profit = float(detail["profit"])
            except (TypeError, ValueError):
                continue
            out.append((float(row["ts"]), profit,
                        str(row["symbol"] or detail.get("symbol") or "")))
    finally:
        conn.close()
    return out


def _blank_mode() -> Dict[str, Any]:
    from trading.strategies.ledger import _blank_mode as blank

    return blank()


def replay_into_ledger_entry(exits: List[Tuple[float, float, str]]) -> Dict[str, Any]:
    """Build a ledger entry from scratch by replaying outcomes in order.

    Mirrors ``StrategyLedger.record`` rather than calling it: ``record()`` also
    writes the lifetime registry, and these outcomes are already counted there.
    Replaying through it would double-count the very numbers this script exists
    to make trustworthy.
    """
    ghost = _blank_mode()
    for ts, profit, _symbol in exits:
        ghost["trades"] = int(ghost["trades"]) + 1
        if profit > 0:
            ghost["wins"] = int(ghost["wins"]) + 1
            ghost["consecutive_losses"] = 0
        else:
            ghost["losses"] = int(ghost["losses"]) + 1
            ghost["consecutive_losses"] = int(ghost["consecutive_losses"]) + 1
        ghost["total_profit"] = float(ghost["total_profit"]) + float(profit)
        ghost["peak_profit"] = max(float(ghost["peak_profit"]), ghost["total_profit"])
        ghost["max_drawdown"] = max(
            float(ghost["max_drawdown"]),
            ghost["peak_profit"] - ghost["total_profit"],
        )
        ghost["last_ts"] = float(ts)
    return {
        "ghost": ghost,
        "live": _blank_mode(),
        "live_approved": False,
        "demotions": 0,
        "demote_reason": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    strategies = registry.get("strategies") or {}

    if SCOUT_ID in ledger or SCOUT_ID in strategies:
        print("already split -- %s exists; nothing to do." % SCOUT_ID)
        return 0

    exits = bot_executed_exits()
    print("bot-executed %s exits found in trading_ops: %d" % (BOT_ID, len(exits)))
    for ts, profit, symbol in exits:
        print("    %s  %-16s %+0.6f"
              % (time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)), symbol, profit))

    old = ledger.get(BOT_ID) or {}
    old_ghost = old.get("ghost") or {}
    print("\nledger %s before: %d trades, %d wins, %+0.4f, live_approved=%s"
          % (BOT_ID, old_ghost.get("trades", 0), old_ghost.get("wins", 0),
             old_ghost.get("total_profit", 0.0), old.get("live_approved")))

    new_entry = replay_into_ledger_entry(exits)
    ng = new_entry["ghost"]
    print("ledger %s after:  %d trades, %d wins, %+0.4f, live_approved=False"
          % (BOT_ID, ng["trades"], ng["wins"], ng["total_profit"]))
    print("ledger %s:        %d trades (was %s), live_approved cleared"
          % (SCOUT_ID, old_ghost.get("trades", 0), BOT_ID))

    if args.dry_run or not args.yes:
        print("\n(dry run -- pass --yes to write)")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(LEDGER, LEDGER.with_suffix(".json.bak-scoutsplit-%s" % stamp))
    shutil.copy2(REGISTRY, REGISTRY.with_suffix(".json.bak-scoutsplit-%s" % stamp))

    if old:
        old["live_approved"] = False
        old["demote_reason"] = DEMOTE_REASON
        old["demoted_ts"] = time.time()
        ledger[SCOUT_ID] = old
    ledger[BOT_ID] = new_entry
    LEDGER.write_text(json.dumps(ledger, indent=1), encoding="utf-8")

    entry = strategies.get(BOT_ID)
    if entry is not None:
        entry = dict(entry)
        entry["strategy_id"] = SCOUT_ID
        entry["name"] = SCOUT_ID
        strategies[SCOUT_ID] = entry
        strategies[BOT_ID] = {
            "strategy_id": BOT_ID,
            "name": BOT_ID,
            "kind": "builtin",
            "genes": {},
            "objective": "",
            "model_id": "",
            "model_name": "(no brain)",
            "metrics": {},
            "commissioned": True,
            "created_at": time.time(),
            "experiments": [],
            "auto_registered": True,
            "lifetime": {"ghost": _registry_ghost(exits)},
        }
        REGISTRY.write_text(json.dumps(registry, indent=1), encoding="utf-8")

    print("\nwritten. backups: *.bak-scoutsplit-%s" % stamp)
    return 0


def _registry_ghost(exits: List[Tuple[float, float, str]]) -> Dict[str, Any]:
    wins = sum(1 for _, p, _ in exits if p > 0)
    gross_win = sum(p for _, p, _ in exits if p > 0)
    gross_loss = sum(-p for _, p, _ in exits if p <= 0)
    symbols: Dict[str, int] = {}
    for _, _, symbol in exits:
        if symbol:
            symbols[symbol] = symbols.get(symbol, 0) + 1
    total = gross_win - gross_loss
    return {
        "trades": len(exits),
        "wins": wins,
        "losses": len(exits) - wins,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "total_profit": total,
        "best": max((p for _, p, _ in exits), default=0.0),
        "worst": min((p for _, p, _ in exits), default=0.0),
        "peak_profit": max(total, 0.0),
        "max_drawdown": 0.0,
        "consecutive_losses": 0,
        "max_consecutive_losses": 0,
        "first_ts": exits[0][0] if exits else 0.0,
        "last_ts": exits[-1][0] if exits else 0.0,
        "symbols": symbols,
    }


if __name__ == "__main__":
    raise SystemExit(main())
