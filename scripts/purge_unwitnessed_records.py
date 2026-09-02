"""Rebuild strategy records the trade log never witnessed.

`scripts/purge_test_artifacts.py` catches ONE fabrication signature: a perfect
win rate at exactly +1.0000 a trade. Fabrications with losses in them walk
straight past it, so this script asks a different and much harder question --

    does the database contain the trades this record claims?

Every ghost exit writes a `trading_ops` row with `status='ghost-exit'` and the
`strategy_id` in its details, and that row is written by the same code path
that later folds the outcome into the registry (trading/bot.py: record the
outcome, THEN record to the ledger). A registry entry claiming trades that
have no exit rows behind them is not a measurement of this account.

Measured 2026-09-02, registry claim vs ghost-exit rows in the database:

    strategy              claimed   in db   witnessed   symbols     profit
    atf_static                250     245       98.0%        11    +2.6684
    obv_accumulation@1w         2       2      100.0%         0    -0.0128
    rsi_reversal@5h            13      13      100.0%         0    -1.9118
    money_button               77       1        1.3%         0    -0.3897
    ema_cross                  24       0        0.0%         0    +0.0000
    volume_spike               21       0        0.0%         0   -29.8200

The bottom three are the fabrications. money_button's is the expensive one:
it is the lane the system most wants evidence for, and its record says
16W/60L at -0.3997, which reads as "this strategy was tried hard and does not
work". It was not tried. The database holds exactly ONE money_button round
trip -- TOAD-USDC, entered 1788207679 at 5.577254599408676e-06, exited 304
seconds later at 5.628224229761183e-06, gross +0.018277, fees -0.013000, net
**+0.005278** -- one profitable 5-minute trade, which is the whole thesis of
the lane.

The other 76 reconcile to the arithmetic of test_ledger_rejects_artifacts.py
running three times against the production registry (its ledger path was
isolated; `record_outcome` takes no path argument and was not). volume_spike's
-29.82 is a 6000x outlier against the largest real trade this stack has ever
booked (+0.41). Both are noise from a test, wearing the costume of evidence.

Refusing to purge is not the safe default here. A fabricated LOSS suppresses a
lane that might work; a fabricated WIN spends real money on one that does not.

The detection is deliberately narrow -- all four must hold:

  * at least 5 claimed trades, AND
  * fewer than 20% of them witnessed by a ghost-exit row, AND
  * no symbols recorded at all (a real record names what it traded), AND
  * the strategy's surviving real trades can be rebuilt from the log

atf_static (98% witnessed, 11 symbols) and every @horizon variant (100%
witnessed) are untouched, as they should be.

The rebuild is not a subtraction of an estimate. Each surviving record is
recomputed from the ghost-exit rows themselves, so what remains is exactly
what the database can prove.

Usage:
    python scripts/purge_unwitnessed_records.py            # report only
    python scripts/purge_unwitnessed_records.py --apply    # write the change
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "strategy_registry.json"
DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

#: Below this share of claimed trades appearing in the trade log, the record
#: is not describing this account's history.
WITNESS_FLOOR = 0.20
#: Too few trades to distinguish a fabrication from an ordinary short history.
MIN_TRADES = 5


def load_exits() -> Dict[str, List[Tuple[float, float, str]]]:
    """Every witnessed ghost exit as {strategy_id: [(ts, profit, symbol)]}."""
    exits: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    conn = sqlite3.connect(DB_PATH, uri=True)
    try:
        rows = conn.execute(
            "SELECT ts, symbol, details FROM trading_ops "
            "WHERE status='ghost-exit' ORDER BY ts"
        )
        for ts, symbol, details in rows:
            try:
                payload = json.loads(details or "{}")
            except Exception:
                continue
            sid = str(payload.get("strategy_id") or "unclassified")
            profit = payload.get("profit")
            if not isinstance(profit, (int, float)):
                continue
            sym = str(payload.get("symbol") or symbol or "").upper()
            exits[sid].append((float(ts), float(profit), sym))
    finally:
        conn.close()
    return exits


def rebuild_ghost(records: List[Tuple[float, float, str]]) -> Dict[str, Any]:
    """Recompute a lifetime ghost record from the trades the log proves.

    Mirrors services.strategy_registry.record_outcome's accumulation exactly,
    replayed in timestamp order, so the rebuilt record is indistinguishable
    from one that had been written a trade at a time.
    """
    stats: Dict[str, Any] = {
        "trades": 0, "wins": 0, "losses": 0,
        "gross_win": 0.0, "gross_loss": 0.0, "total_profit": 0.0,
        "best": 0.0, "worst": 0.0,
        "peak_profit": 0.0, "max_drawdown": 0.0,
        "consecutive_losses": 0, "max_consecutive_losses": 0,
        "first_ts": 0.0, "last_ts": 0.0, "symbols": {},
    }
    for ts, profit, symbol in sorted(records):
        if not stats["trades"]:
            stats["first_ts"] = ts
        stats["last_ts"] = ts
        stats["trades"] += 1
        stats["total_profit"] += profit
        if profit > 0:
            stats["wins"] += 1
            stats["gross_win"] += profit
            stats["consecutive_losses"] = 0
            stats["best"] = max(stats["best"], profit)
        else:
            stats["losses"] += 1
            stats["gross_loss"] += abs(profit)
            stats["consecutive_losses"] += 1
            stats["max_consecutive_losses"] = max(
                stats["max_consecutive_losses"], stats["consecutive_losses"]
            )
            stats["worst"] = min(stats["worst"], profit)
        stats["peak_profit"] = max(stats["peak_profit"], stats["total_profit"])
        stats["max_drawdown"] = max(
            stats["max_drawdown"], stats["peak_profit"] - stats["total_profit"]
        )
        if symbol:
            stats["symbols"][symbol] = int(stats["symbols"].get(symbol, 0)) + 1
    return stats


def judge(sid: str, ghost: Dict[str, Any], witnessed: int) -> Tuple[bool, str]:
    """Return (unwitnessed, reason). Narrow by design: all conditions hold."""
    claimed = int(ghost.get("trades") or 0)
    if claimed < MIN_TRADES:
        return False, "only %d claimed trades; too short to judge" % claimed
    if ghost.get("symbols"):
        return False, "names %d real symbol(s)" % len(ghost["symbols"])
    ratio = witnessed / claimed
    if ratio >= WITNESS_FLOOR:
        return False, "%d/%d trades witnessed (%.0f%%)" % (witnessed, claimed, ratio * 100)
    return True, (
        "claims %d trades, %d in the trade log (%.1f%%), no symbols, total %+.4f"
        % (claimed, witnessed, ratio * 100, float(ghost.get("total_profit") or 0.0))
    )


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    if not REGISTRY.exists():
        print("registry not found: %s" % REGISTRY)
        return 1

    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    strategies = data.get("strategies") or {}
    exits = load_exits()

    purged: List[str] = []
    print("%-24s %8s %8s %9s  %s" % ("strategy", "claimed", "in-db", "witness", "verdict"))
    print("-" * 78)
    for sid in sorted(strategies):
        entry = strategies[sid]
        ghost = (entry.get("lifetime") or {}).get("ghost") or {}
        claimed = int(ghost.get("trades") or 0)
        witnessed = len(exits.get(sid, []))
        unwitnessed, reason = judge(sid, ghost, witnessed)
        ratio = (witnessed / claimed * 100) if claimed else 0.0
        print("%-24s %8d %8d %8.1f%%  %s" % (
            sid, claimed, witnessed, ratio,
            ("PURGE  " if unwitnessed else "keep   ") + reason,
        ))
        if not unwitnessed:
            continue
        purged.append(sid)
        rebuilt = rebuild_ghost(exits.get(sid, []))
        if rebuilt["trades"]:
            entry.setdefault("lifetime", {})["ghost"] = rebuilt
            print("%-24s   -> rebuilt from the log: %d trade(s), %dW/%dL, %+.6f on %s" % (
                "", rebuilt["trades"], rebuilt["wins"], rebuilt["losses"],
                rebuilt["total_profit"], ", ".join(rebuilt["symbols"]) or "-",
            ))
        else:
            (entry.get("lifetime") or {}).pop("ghost", None)
            print("%-24s   -> no witnessed trades at all; ghost record removed" % "")

    if not purged:
        print("\nnothing to purge: every record reconciles to the trade log.")
        return 0

    print("\n%d record(s) rebuilt: %s" % (len(purged), ", ".join(purged)))
    if not apply:
        print("report only. re-run with --apply to write the change.")
        return 0

    backup = REGISTRY.with_suffix(
        ".json.bak-unwitnessed-%s" % time.strftime("%Y%m%d-%H%M%S")
    )
    shutil.copy2(REGISTRY, backup)
    REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("backup: %s" % backup.name)
    print("written: %s" % REGISTRY)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
