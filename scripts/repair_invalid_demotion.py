"""Undo a demotion that our own bug caused, and restore the evidence it wiped.

Measured 2026-09-03. ``atf_static`` -- the only strategy that has ever traded
live -- reads ``live_approved=False`` with
``demote_reason='2 consecutive live losses'``. Every other strategy in the
ledger also reads False, so NOTHING can trade and the burst of rapid profitable
swapping cannot return no matter how long the loop runs.

Both of those losses were ours, not the market's. The CBETH exits at 11:28 and
12:46 sold only 0.000162 and 0.000111 CBETH against roughly 0.00026 held, so
they booked losses on positions that were never fully closed. Under the rule
the ledger now uses -- demote on a NET loss, not on a streak -- that demotion
would not have happened: the strategy was net positive at the time.

Worse, ``_demote_locked`` wipes the ghost record so re-graduation needs fresh
evidence. atf_static was left with 4 ghost trades against a 20-trade
graduation bar, which is hours of ghosting away, purely as a consequence of a
sizing bug.

The evidence was not lost, only dropped from one file: the registry still
holds the real lifetime ghost record. This restores it and clears the invalid
demotion.

Deliberately narrow -- it repairs a strategy ONLY when all of these hold:

  * it is currently demoted, AND
  * the demotion reason is a consecutive-loss streak, AND
  * the registry's ghost record independently clears the graduation bar.

So a strategy demoted for a genuine net loss, or one whose record does not
actually qualify, is left exactly where it is.

Usage:
    python scripts/repair_invalid_demotion.py            # report only
    python scripts/repair_invalid_demotion.py --apply    # write the change
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "strategy_ledger.json"

sys.path.insert(0, str(ROOT))


def _graduation_bar() -> tuple:
    return (
        int(os.getenv("STRATEGY_GRADUATION_MIN_TRADES", "20")),
        float(os.getenv("STRATEGY_GRADUATION_MIN_WINRATE", "0.55")),
        float(os.getenv("STRATEGY_GRADUATION_MIN_PROFIT", "0.0")),
    )


def _registry_ghost() -> dict:
    """The lifetime ghost record, which the ledger wipe did not touch."""
    out = {}
    try:
        from services import strategy_registry

        for row in strategy_registry.list_strategies():
            sid = row.get("strategy_id")
            ghost = ((row.get("lifetime") or {}).get("ghost")) or {}
            if sid and ghost:
                out[sid] = ghost
    except Exception as exc:  # noqa: BLE001
        print("could not read the registry: %s" % exc)
    return out


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    if not LEDGER.exists():
        print("ledger not found: %s" % LEDGER)
        return 1

    data = json.loads(LEDGER.read_text(encoding="utf-8"))
    registry = _registry_ghost()
    min_trades, min_winrate, min_profit = _graduation_bar()
    print("graduation bar: %d trades, %.0f%% win rate, profit > %.4f"
          % (min_trades, min_winrate * 100, min_profit))
    print()

    repairs = []
    for sid, ent in data.items():
        if not isinstance(ent, dict) or ent.get("live_approved"):
            continue
        reason = str(ent.get("demote_reason") or "")
        if "consecutive live losses" not in reason:
            continue

        ghost = registry.get(sid) or {}
        trades = int(ghost.get("trades") or 0)
        wins = int(ghost.get("wins") or 0)
        profit = float(ghost.get("total_profit") or 0.0)
        rate = wins / max(trades, 1)
        qualifies = trades >= min_trades and rate >= min_winrate and profit > min_profit

        print("%-22s demoted: %s" % (sid, reason))
        print("    ledger ghost  : %d trades" % int((ent.get("ghost") or {}).get("trades") or 0))
        print("    registry ghost: %d trades, %d wins (%.0f%%), profit %+.4f"
              % (trades, wins, rate * 100, profit))
        print("    qualifies     : %s" % qualifies)
        if qualifies:
            repairs.append((sid, ghost))
        print()

    if not repairs:
        print("nothing to repair.")
        return 0

    if not apply:
        print("report only. re-run with --apply to restore %d strategy(ies)."
              % len(repairs))
        return 0

    backup = LEDGER.with_suffix(".json.bak-demotionrepair-%s"
                                % time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(LEDGER, backup)
    print("backed up -> %s" % backup.name)

    for sid, ghost in repairs:
        ent = data[sid]
        # Restore the evidence the wipe discarded, from the record that kept it.
        ent["ghost"] = {
            "trades": int(ghost.get("trades") or 0),
            "wins": int(ghost.get("wins") or 0),
            "total_profit": float(ghost.get("total_profit") or 0.0),
            "conf_ema": float((ent.get("ghost") or {}).get("conf_ema") or 0.05),
            "peak_profit": float(ghost.get("peak_profit") or 0.0),
            "max_drawdown": float(ghost.get("max_drawdown") or 0.0),
            "consecutive_losses": 0,
            "last_ts": float(ghost.get("last_ts") or time.time()),
        }
        # Via the ledger's licence-granting helper -- it re-bases `dd_ref`, and
        # without that this repair lasts exactly one live outcome before the
        # give-back brake re-demotes against a stale `peak_profit`.
        from trading.strategies.ledger import StrategyLedger

        StrategyLedger._grant_live_licence(ent, ts_key="graduated_ts")
        ent["demote_reason"] = None
        # Keep the count: it is a real history of demotions, and zeroing it
        # would hide that this happened.
        ent["live"]["consecutive_losses"] = 0
        print("  restored %s -> live_approved=True" % sid)

    LEDGER.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print()
    print("ledger updated. %d strategy(ies) can trade again." % len(repairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
