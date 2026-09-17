"""Does the ghost book's evidence survive a POOL-DEPTH admission bar?

[a38cf462]. Pass 426 measured the whole 237-row ghost book bucketed by pool
depth and found the under-$50k bucket was the book's MOST profitable. That
number is about the POOLED book, and the pooled book is not what graduation
reads -- graduation reads ``_tradeable_of(ghost)``, the subset the live lane
could have placed. The two questions are different and only the second one
can change the scoreboard:

    pooled    443 trades  P/L +7.0904   <- what pass 426 bucketed
    tradeable  55 trades  P/L -0.5492   <- what the bar counts

So this script asks the narrower question: of the round trips GRADUATION
COUNTS, how many are in pools too thin to have filled, and does removing them
change the count (n) or the sign. A depth bar that removes profit from a book
nobody spends is housekeeping; a depth bar that changes the tradeable n is a
change to the bar itself.

THE PREDICATE IS IMPORTED, NEVER REIMPLEMENTED -- same discipline as
scripts/tradeable_book.py:73. A second copy of "is this tradeable" is a second
answer to one question.

Depth source is data/pool_depth_snapshot.json (DexScreener, deepest base pool
per symbol). It is a SNAPSHOT and the rows span two weeks; see the CAVEAT
section of the output, which states per row whether depth could be checked
against that row's own timestamp.

Run:  python -X utf8 scripts/ghost_depth_admission_census.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Loading the real env matters: a census that skips it runs whatever literal
# default sits in a getenv call, which is how pass 445 published three numbers
# against a 4x tighter rule than production. See
# [replay-harness-ran-a-different-rule-than-production].
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

from scripts.tradeable_book import (  # noqa: E402
    DEFAULT_DB,
    _tradeable_predicate,
    _win_rate,
    load_rows,
)

SNAPSHOT = ROOT / "data" / "pool_depth_snapshot.json"

# The bars are reported as a LADDER rather than one chosen threshold. Picking a
# single number first and measuring after is how a fitted bar gets published as
# a finding; the ladder shows whether the result depends on where the line is.
BARS = [0.0, 50_000.0, 250_000.0, 1_000_000.0]


def load_depth() -> Dict[str, Optional[float]]:
    raw = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {k.upper(): v for k, v in raw.items() if not k.startswith("_")}


def base_symbol(pair: str) -> str:
    return str(pair or "").upper().split("-")[0].split("/")[0]


def book(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc = {"trades": 0, "wins": 0, "losses": 0, "net": 0.0}
    for r in rows:
        acc["trades"] += 1
        net = float(r.get("net", 0.0) or 0.0)
        acc["net"] += net
        if net > 0:
            acc["wins"] += 1
        elif net < 0:
            acc["losses"] += 1
    acc["win_rate"] = _win_rate(acc)
    return acc


def fmt(acc: Dict[str, Any]) -> str:
    return "%3d trades  %3.0f%% win  P/L %+.4f" % (
        acc["trades"], acc["win_rate"] * 100.0, acc["net"])


def main() -> int:
    days = float(sys.argv[1]) if len(sys.argv) > 1 else 14.3
    now = time.time()
    since = now - days * 86400.0

    is_tradeable = _tradeable_predicate()
    if is_tradeable is None:
        print("ERROR: cannot import trading.strategies.ledger._live_tradeable")
        return 1

    depth = load_depth()
    all_rows = [r for r in load_rows(Path(DEFAULT_DB), since)
                if str(r.get("mode", "")).lower() != "live"]
    # (strategy, symbol), not symbol alone. The ban is per-pair: AERO-USDC is
    # refused FOR atf_static and allowed for everything else, and AERO is the
    # most-traded ghost symbol, so dropping the strategy argument inflates the
    # tradeable population by more than half. Measured this pass: 121 rows
    # symbol-only vs the number below.
    trade_rows = [r for r in all_rows
                  if is_tradeable(r["symbol"], r.get("strategy_id"))]

    print("=" * 72)
    print("GHOST EVIDENCE UNDER A POOL-DEPTH ADMISSION BAR  [a38cf462]")
    print("window %.1f days  |  depth snapshot %s" % (days, SNAPSHOT.name))
    print("=" * 72)
    print()
    print("  POOLED   ghost rows: %s" % fmt(book(all_rows)))
    print("  TRADEABLE ghost rows: %s   <- what graduation counts" % fmt(book(trade_rows)))
    print()

    for label, rows in (("POOLED", all_rows), ("TRADEABLE", trade_rows)):
        print("-" * 72)
        print("  %s book by pool depth" % label)
        print("-" * 72)
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            d = depth.get(base_symbol(r["symbol"]), "MISSING")
            if d == "MISSING":
                key = "NOT SWEPT"
            elif d is None:
                key = "UNRESOLVED (asked, no answer)"
            elif d <= 0:
                key = "NO POOL AT ALL"
            elif d < 50_000:
                key = "under $50k"
            elif d < 1_000_000:
                key = "$50k - $1M"
            else:
                key = "over $1M"
            buckets.setdefault(key, []).append(r)
        order = ["NO POOL AT ALL", "under $50k", "$50k - $1M", "over $1M",
                 "UNRESOLVED (asked, no answer)", "NOT SWEPT"]
        for key in order:
            if key in buckets:
                print("    %-30s %s" % (key, fmt(book(buckets[key]))))
        print()

    print("-" * 72)
    print("  THE BAR APPLIED TO GRADUATION'S OWN POPULATION")
    print("  n is what the 20-trade bar counts; the bar is unchanged at 20.")
    print("-" * 72)
    known = [r for r in trade_rows
             if isinstance(depth.get(base_symbol(r["symbol"])), (int, float))]
    unknown = [r for r in trade_rows if r not in known]
    print("    depth KNOWN for %d of %d tradeable rows (%d unpriced)"
          % (len(known), len(trade_rows), len(unknown)))
    print()
    print("    %-14s %-38s %s" % ("bar", "kept (depth known and >= bar)", "delta n"))
    n_before = len(trade_rows)
    for bar in BARS:
        kept = [r for r in known
                if float(depth[base_symbol(r["symbol"])]) >= bar]
        print("    >= $%-11s %-38s %+d"
              % ("%.0f" % bar, fmt(book(kept)), len(kept) - n_before))
    print()

    # THE SAME LADDER, CENSORED. Six rows of 215 carry +5.36 and set the sign of
    # every bucket above, so the uncensored ladder is not quotable on its own.
    # `trade_outcomes` is a PRE-GUARD source -- it keeps rows the ledger refused
    # at record() time -- so any census reading it measures a dirtier book than
    # graduation does. Printed side by side rather than replacing the
    # uncensored numbers, because the GAP between them is the finding.
    try:
        from services.outcome_plausibility import IMPLAUSIBLE_RET, partition

        censored_all, dropped = partition(all_rows)
        c_tr = [r for r in censored_all
                if is_tradeable(r["symbol"], r.get("strategy_id"))]
        print("-" * 72)
        print("  THE SAME LADDER, CENSORED  (services.outcome_plausibility,")
        print("  IMPLAUSIBLE_RET=%.2f) -- %d of %d rows censored"
              % (IMPLAUSIBLE_RET, len(dropped), len(all_rows)))
        print("-" * 72)
        print("    %-16s %s" % ("all tradeable", fmt(book(c_tr))))
        for bar in BARS:
            kept = [r for r in c_tr
                    if isinstance(depth.get(base_symbol(r["symbol"])), (int, float))
                    and float(depth[base_symbol(r["symbol"])]) >= bar]
            print("    >= $%-11s %s" % ("%.0f" % bar, fmt(book(kept))))
        print()
    except Exception as exc:  # noqa: BLE001
        print("    CENSORED LADDER UNAVAILABLE: %s" % exc)
        print()

    print("-" * 72)
    print("  PER-STRATEGY: does any strategy's tradeable n move?")
    print("-" * 72)
    per: Dict[str, List[Dict[str, Any]]] = {}
    for r in trade_rows:
        per.setdefault(str(r.get("strategy_id") or "unclassified"), []).append(r)
    print("    %-26s %-28s %-28s" % ("strategy", "now (all tradeable)", "with >= $1M pool only"))
    for sid, rs in sorted(per.items(), key=lambda kv: -len(kv[1])):
        deep = [r for r in rs
                if isinstance(depth.get(base_symbol(r["symbol"])), (int, float))
                and float(depth[base_symbol(r["symbol"])]) >= 1_000_000.0]
        print("    %-26s %-28s %-28s" % (sid, fmt(book(rs)), fmt(book(deep))))
    print()

    print("-" * 72)
    print("  CAVEAT: can depth be checked against the ROW'S OWN timestamp?")
    print("-" * 72)
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    taken = snap.get("_taken_utc_approx", "unknown")
    print("    snapshot taken %s; rows span %.1f days before now." % (taken, days))
    print("    DexScreener's search endpoint returns CURRENT liquidity only and")
    print("    market_stream.raw carries no liquidity field (checked: keys are")
    print("    ts/symbol/chain/price/volume/rest/consensus_confidence), so there")
    print("    is NO historical depth anywhere in this repo. Depth attributed to")
    print("    a row older than the snapshot is an INFERENCE, stated as such.")
    print("    NOT MEASURED, and it is the one timestamp-checkable fact there")
    print("    is: DexScreener returns pairCreatedAt, so a pool created AFTER a")
    print("    row's ts would falsify that row outright. Nobody has swept it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
