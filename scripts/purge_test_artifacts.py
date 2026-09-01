"""Remove fabricated strategy records that no market produced.

Four strategies carry records that are not measurements:

    vwap_reversion     15 trades, 15/0, total +15.0000
    rsi_reversal       15 trades, 15/0, total +15.0000
    mean_reversion     15 trades, 15/0, total +15.0000
    momentum_breakout   6 trades,  6/0, total  +6.0000

Every one is exactly +1.0000 per trade with a 100% win rate, `best` 1.0,
`worst` 0.0, an empty `symbols` map, and a first/last timestamp inside the
same six-minute window on 2026-08-27 09:18-09:24. Real trades name the symbol
they traded and do not all return the same round number.

`e3eec69` ("Build bots that CAN trade live, and stop tests fabricating
performance") stopped the tests writing these, but never purged what had
already been written, so they are still in the live registry weeks later.

Why this is worth a script rather than a hand edit: these records feed the
promotion decision. A strategy showing a 100% win rate is exactly the kind
of record that fast-tracks a garbage strategy to live trading with real
money -- the same failure class the ledger artifact guard was written for,
arriving through a different door.

The detection is deliberately narrow. It refuses a record ONLY when every
one of these holds, so a genuinely good strategy is never purged:

  * a perfect win rate over at least 5 trades, AND
  * total profit equal to the trade count to within a cent (the +1.0/trade
    signature), AND
  * no symbols recorded at all.

Usage:
    python scripts/purge_test_artifacts.py            # report only
    python scripts/purge_test_artifacts.py --apply    # write the change
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "strategy_registry.json"


def is_fabricated(ghost: dict) -> tuple:
    """Return (verdict, reason). Narrow by design: all conditions must hold."""
    trades = int(ghost.get("trades") or 0)
    wins = int(ghost.get("wins") or 0)
    losses = int(ghost.get("losses") or 0)
    profit = float(ghost.get("total_profit") or 0.0)
    symbols = ghost.get("symbols") or {}

    if trades < 5:
        return False, "too few trades to judge"
    if losses != 0 or wins != trades:
        return False, "has losses, so not a synthetic perfect record"
    if abs(profit - trades) > 0.01:
        return False, "profit is not the +1.0/trade signature"
    if symbols:
        return False, "names real symbols"
    return True, (
        "%d trades, %d/0 W/L, total %+.4f (exactly +1.0/trade), no symbols"
        % (trades, wins, profit)
    )


def main() -> int:
    apply = "--apply" in sys.argv[1:]

    if not REGISTRY.exists():
        print("registry not found: %s" % REGISTRY)
        return 1

    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    # "strategies" is a dict keyed by strategy id, not a list.
    strategies = data.get("strategies") if isinstance(data, dict) else None
    if not isinstance(strategies, dict):
        print("unexpected registry shape: expected data['strategies'] to be a dict")
        return 1

    doomed, kept = [], []
    for sid, row in strategies.items():
        if not isinstance(row, dict):
            continue
        ghost = ((row.get("lifetime") or {}).get("ghost")) or {}
        bad, why = is_fabricated(ghost)
        (doomed if bad else kept).append((sid, why, ghost))

    print("=== FABRICATED (would be reset) ===")
    if not doomed:
        print("  none")
    for sid, why, _ in doomed:
        print("  %-24s %s" % (sid, why))

    print()
    print("=== KEPT ===")
    for sid, why, ghost in kept:
        t = int(ghost.get("trades") or 0)
        if t:
            print("  %-24s %d trades, %+.4f  (%s)"
                  % (sid, t, float(ghost.get("total_profit") or 0.0), why))

    if not doomed:
        return 0

    if not apply:
        print()
        print("report only. re-run with --apply to reset those records.")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = REGISTRY.with_suffix(".json.bak-purge-%s" % stamp)
    shutil.copy2(REGISTRY, backup)
    print()
    print("backed up -> %s" % backup.name)

    names = {sid for sid, _, _ in doomed}
    for sid, row in strategies.items():
        if sid not in names or not isinstance(row, dict):
            continue
        # Reset to a clean slate rather than deleting the strategy: it is a
        # real strategy whose RECORD is fiction, and it should be free to earn
        # a genuine one.
        lifetime = row.setdefault("lifetime", {})
        lifetime["ghost"] = {
            "trades": 0, "wins": 0, "losses": 0,
            "total_profit": 0.0, "gross_win": 0.0, "gross_loss": 0.0,
            "best": 0.0, "worst": 0.0, "symbols": {},
            "consecutive_losses": 0, "max_consecutive_losses": 0,
            "max_drawdown": 0.0, "peak_profit": 0.0,
        }
        print("  reset %s" % sid)

    REGISTRY.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print()
    print("registry updated. %d record(s) reset." % len(doomed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
