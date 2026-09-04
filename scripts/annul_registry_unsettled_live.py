"""Strike the same four fictions from the SECOND book of record.

``scripts/annul_unsettled_live_exits.py`` removed four live exits that the
chain does not support from ``data/strategy_ledger.json`` and set
``trade_outcomes.status='annulled'`` on their rows. It did not touch
``data/strategy_registry.json``.

There are two books, not one. ``StrategyLedger.record`` mirrors every new
outcome into the registry (trading/strategies/ledger.py, ``mirror_registry``),
so during normal operation they agree. A CORRECTION applied to one of them
makes them disagree silently -- and ``scripts/live_path_check.py``'s link 10
reads the REGISTRY:

    for row in strategy_registry.list_strategies():
        live = ((row.get("lifetime") or {}).get("live")) or {}
        total += float(live.get("total_profit") or 0.0)

so the gate went on reporting

    [FAIL ] 10 PROFIT   live P/L -0.1405 over 7 trades

while the ledger and the chain both said +0.0052 over 3.

RE-VERIFIED AGAINST THE CHAIN, 2026-09-04, base block 0x307e9d1, wallet
0x291c854811e92906a658Fb94Aa511bF919f968ad, via eth_call balanceOf. The four
annulled exits claim to have sold tokens the wallet still holds:

    BSTONK  balanceOf = 360264243225392976659 raw / 1e18
            = 360.264243225393 -- the exact quantity the 19:12:15 exit says
            it sold. It was bought for 0.75 USDC at 17:02:13 in
            0xcd6fb05c92af5077f9be707727c1d57e0ac1dfedd54d9f87e860376b96ea560b
            and has never left the wallet.
    cbBTC   balanceOf = 3709 raw / 1e8 = 0.00003709 -- the 20:03:27 exit says
            it sold 0.00000928 of it; the balance is undiminished.
    cbETH   balanceOf = 373253011411 raw / 1e18 = 0.000000373253011411 -- the
            20:38:53 exit says it sold 0.00015239, four hundred times more
            than the wallet has held since 16:46.

The annulment is a measurement, not a convenience. This applies it to the
registry.

WHAT SURVIVES. The three exits with a settling on-chain sell, in full:

    12:36:59  AERO-USDC   +0.0012394
              0x77f0075e0e6b79e71ba63667efe4aec9a80bb2b6e858400f63305d90ec7a6ed1
    15:27:21  CBETH-USDC  +0.0097748
              0x927834717d12395c1eb3d9148609a2b8142a59403205d1caa4ffd04e68e0e005
    16:38:15  CBETH-USDC  -0.0058591
              0x9ffdd1cfe3f17fbeddb2b3091b49f8f6524b5a0538ae9a2f3aadb56610264e6c

    net +0.005155079857143799, 2W/1L, profit factor 1.925

WHAT THIS DOES NOT DO. It adds nothing. The settled sell
0x166c461a1f6b75da3ad77e8a4fd78080cc0dcb35391b11187c38950e35a377a0 (0.000264
cbETH -> USDC, 16:42:23Z) still has no outcome row and is still NOT inserted:
inventing its entry basis would fabricate a record, and leaving a likely WIN
out is the conservative direction.

The surviving set is READ FROM THE DATABASE (``status='closed'`` AND
``details.mode='live'``), not hardcoded, so this cannot drift from the rows
the annulment actually left standing. The hardcoded numbers below are guards:
if the database has moved on, something has traded and the record must be
re-judged rather than rewritten.

Run:  .venv/Scripts/python.exe -X utf8 scripts/annul_registry_unsettled_live.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import strategy_registry  # noqa: E402

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "trading_cache.db"
STRATEGY_ID = "atf_static"

REASON = (
    "four live exits with no settling ERC-20 Transfer on base were annulled in "
    "the ledger and in trade_outcomes.status on 2026-09-03 but not here; "
    "re-verified against balanceOf on 2026-09-04 (the wallet still holds every "
    "token these exits claim to have sold)"
)

#: What the registry must read BEFORE this runs -- the uncorrected book.
EXPECTED_BEFORE = {"trades": 7, "wins": 3, "losses": 4,
                   "total_profit": -0.14045951062203274}

#: ...and what replaying the settled rows must produce. Cross-checked against
#: data/strategy_ledger.json, which the ledger-side annulment already wrote.
EXPECTED_AFTER = {"trades": 3, "wins": 2, "losses": 1,
                  "total_profit": 0.005155079857143799}

EXPECTED_ANNULLED = 4


def _live_rows(conn: sqlite3.Connection):
    """Every outcome booked live, split by whether the chain supports it."""
    conn.row_factory = sqlite3.Row
    settled, annulled = [], []
    for row in conn.execute("SELECT * FROM trade_outcomes ORDER BY ts"):
        det = json.loads(row["details"] or "{}")
        if str(det.get("mode")) != "live":
            continue
        (settled if str(row["status"]) == "closed" else annulled).append(dict(row))
    return settled, annulled


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the change (default: dry run)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    settled, annulled = _live_rows(conn)

    print(f"database : {DB_PATH}")
    print(f"live outcome rows: {len(settled)} settled, {len(annulled)} annulled")
    for r in annulled:
        det = json.loads(r["details"] or "{}")
        print("  STRUCK  %s  %-13s net=%+.9f  qty=%.12f  %s"
              % (time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(r["ts"])),
                 r["symbol"], r["net_profit"], r["quantity"], r["status"]))
        if "annulled" not in det:
            raise SystemExit(
                f"{r['outcome_id']} is status={r['status']} but carries no "
                "details.annulled audit record; refusing to strike a row whose "
                "reason was never written down"
            )
    outcomes = []
    for r in settled:
        print("  KEEP    %s  %-13s net=%+.9f  qty=%.12f"
              % (time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime(r["ts"])),
                 r["symbol"], r["net_profit"], r["quantity"]))
        outcomes.append({"profit": float(r["net_profit"]),
                         "symbol": str(r["symbol"] or ""),
                         "ts": float(r["ts"])})

    if len(annulled) != EXPECTED_ANNULLED:
        print(f"\nREFUSING: expected {EXPECTED_ANNULLED} annulled live rows, found "
              f"{len(annulled)}. Re-measure against the chain before rewriting.",
              file=sys.stderr)
        return 2

    entry = strategy_registry.get_strategy(STRATEGY_ID) or {}
    live = (entry.get("lifetime") or {}).get("live") or {}
    print(f"\nregistry : {strategy_registry.REGISTRY_PATH}")
    print("  live now  : %d trades %dW/%dL  net %+.9f"
          % (live.get("trades", 0), live.get("wins", 0), live.get("losses", 0),
             live.get("total_profit", 0.0)))

    problems = []
    for key, want in EXPECTED_BEFORE.items():
        got = live.get(key)
        if isinstance(want, float):
            if got is None or abs(float(got) - want) > 1e-12:
                problems.append(f"live.{key} is {got!r}, expected {want!r}")
        elif int(got or 0) != want:
            problems.append(f"live.{key} is {got!r}, expected {want!r}")
    if problems:
        print("\nREFUSING: the registry's live book is not the one measured against "
              "the chain:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("Something has traded since. Re-measure before rewriting.", file=sys.stderr)
        return 2

    total = sum(o["profit"] for o in outcomes)
    wins = sum(1 for o in outcomes if o["profit"] > 0)
    print("  live after: %d trades %dW/%dL  net %+.9f"
          % (len(outcomes), wins, len(outcomes) - wins, total))
    if len(outcomes) != EXPECTED_AFTER["trades"] or abs(total - EXPECTED_AFTER["total_profit"]) > 1e-12:
        print(f"\nREFUSING: the settled rows replay to {len(outcomes)} trades / "
              f"{total!r}, expected {EXPECTED_AFTER['trades']} / "
              f"{EXPECTED_AFTER['total_profit']!r}.", file=sys.stderr)
        return 2

    if not args.apply:
        print("\ndry run -- re-run with --apply")
        return 0

    backup = strategy_registry.REGISTRY_PATH.with_suffix(
        f".json.bak-registryannul-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(strategy_registry.REGISTRY_PATH, backup)
    print(f"\nbacked up registry : {backup}")

    updated = strategy_registry.rebuild_lifetime(
        STRATEGY_ID, mode="live", outcomes=outcomes, reason=REASON,
        struck=[{"ts": float(r["ts"]), "symbol": r["symbol"],
                 "net_profit": float(r["net_profit"]),
                 "quantity": float(r["quantity"])} for r in annulled],
    )
    if updated is None:
        print(f"{STRATEGY_ID} is not in the registry", file=sys.stderr)
        return 1

    live = (strategy_registry.get_strategy(STRATEGY_ID) or {}).get("lifetime", {}).get("live", {})
    print("\nregistry now: " + json.dumps(live, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
