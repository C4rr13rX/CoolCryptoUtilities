"""Tie every row of the live scoreboard to a settled Base transaction hash.

The goal of this repo is one number: cumulative realised live net P/L, "positive
and still positive as trades accumulate", and its non-negotiable is that a
profit which cannot be tied to a settled on-chain transaction hash did not
happen. Until pass 715 no query in this repo answered that for the scoreboard,
and the reason was a missing JOIN rather than a missing field:

  * ``trade_outcomes.details`` carries no hash under any name. The complete key
    set over all live-mode rows is {reason, mode, remaining_size,
    retained_profit, accounting_version, repriced, annulled, strategy_id}. So
    every instrument that reads ``trade_outcomes`` alone -- and that is all of
    them: the demotion guards, the symbol edge gate, the plausibility censor,
    every per-strategy census any agent has run -- is structurally blind to the
    chain.
  * ``trading_ops`` rows with ``status='live-exit'`` carry ``tx_hash``,
    ``entry_tx_hash``, ``duration_sec``, ``entry_ts``, ``exit_ts``,
    ``route_used``, ``fill_source`` AND ``outcome_id`` -- and ``outcome_id`` is
    the primary key of ``trade_outcomes``. The join is exact and total.

So this script does not add a field. It joins, and it fails loudly when the join
is not total in both directions, because an orphan on either side is worth more
than the total: a live-exit with no outcome is an unrecorded fill, and an
outcome with no live-exit is a row claiming a fill that left no trace.

TRAP, PAID ONCE SO NOBODY PAYS IT AGAIN: do NOT find hashes by matching
``0x[0-9a-fA-F]{64}`` over ``trading_ops``. That matches 54798 of 156406 rows
and almost none are hashes -- the first hit is ``0x04e45aaf...833589fcd6ed...``,
which is Uniswap ``exactInputSingle`` calldata with USDC-on-Base sitting in an
argument slot. Match on the KEY NAME.

Read-only. Touches no production path, needs no restart, cannot break the live
lane. It adds no trade and moves the scoreboard by exactly zero; what it buys is
that the NEXT positive number is provable.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

# Python puts THIS file's directory (scripts/) on sys.path, not the cwd, so
# `import services.outcome_plausibility` fails with ModuleNotFoundError even when
# run from the repo root -- and the censor would then silently report NOT
# APPLIED, which is the failure mode Sage's pass-710 rule exists to catch.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

# The scoreboard counts round trips that SETTLED. `annulled` rows are live-mode
# but were withdrawn, and pooling them would inflate n above the 18 the
# briefing reports.
SETTLED_STATUS = "closed"


def _connect(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(
            f"FAIL: {path} does not exist. data/trading.db, trading_bot.db and "
            "web/db.sqlite3 are 0-byte stubs that silently return nothing -- "
            "the real store is storage/trading_cache.db (TRADING_DB_PATH)."
        )
    if os.path.getsize(path) == 0:
        raise SystemExit(
            f"FAIL: {path} is 0 bytes. That is one of the known stubs, not the "
            "trading store, and every query against it returns nothing while "
            "reporting success."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _json(blob: Optional[str]) -> Dict[str, Any]:
    try:
        parsed = json.loads(blob or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _live_outcomes(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Live-mode settled rows. `mode` is inside details JSON, not a column."""
    rows: List[Dict[str, Any]] = []
    query = (
        "select outcome_id, ts, symbol, gross_profit, fee_cost, net_profit, "
        "status, details from trade_outcomes"
    )
    for outcome_id, ts, symbol, gross, fee, net, status, details in conn.execute(query):
        meta = _json(details)
        if str(meta.get("mode", "")).lower() != "live":
            continue
        if str(status or "").lower() != SETTLED_STATUS:
            continue
        rows.append(
            {
                "outcome_id": str(outcome_id or ""),
                "ts": ts,
                "symbol": symbol,
                "gross": float(gross or 0.0),
                "fee": float(fee or 0.0),
                "net": float(net or 0.0),
                "strategy_id": meta.get("strategy_id") or "(unattributed)",
                "reason": meta.get("reason"),
            }
        )
    return rows


def _live_exits(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    query = "select ts, symbol, details from trading_ops where status='live-exit'"
    for ts, symbol, details in conn.execute(query):
        meta = _json(details)
        rows.append(
            {
                "outcome_id": str(meta.get("outcome_id") or ""),
                "ts": ts,
                "symbol": symbol,
                "tx_hash": str(meta.get("tx_hash") or ""),
                "entry_tx_hash": str(meta.get("entry_tx_hash") or ""),
                "duration_sec": meta.get("duration_sec"),
                "route_used": meta.get("route_used"),
                "fill_source": meta.get("fill_source"),
                "exit_reason": meta.get("exit_reason"),
            }
        )
    return rows


def _censor(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Apply the shared plausibility censor, and report BOTH counts.

    Sage's pass-710 rule: 'I applied the censor' means nothing without the kept
    AND the censored number, so both are always printed -- including when the
    censor is unavailable, which is stated rather than silently skipped. Call
    ``partition`` once over the batch; never ``is_implausible`` in a loop.
    """
    try:
        from services.outcome_plausibility import partition  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return rows, [], f"NOT APPLIED -- services.outcome_plausibility unavailable ({exc!r})"
    try:
        kept, censored = partition(rows)
        return list(kept), list(censored), "APPLIED via services.outcome_plausibility.partition"
    except Exception as exc:  # noqa: BLE001
        return rows, [], f"NOT APPLIED -- partition() raised ({exc!r})"


def _is_hash(value: str) -> bool:
    return len(value) == 66 and value.startswith("0x")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument(
        "--expect-net",
        type=float,
        default=None,
        help="Scoreboard net to reconcile against, e.g. -0.1864. Divergence is "
             "reported, never smoothed: it is either a new settled trade (the "
             "goal) or a book that has drifted, and those must not be confused.",
    )
    args = ap.parse_args(argv)

    conn = _connect(args.db)
    outcomes = _live_outcomes(conn)
    exits = _live_exits(conn)

    by_outcome = {row["outcome_id"]: row for row in outcomes if row["outcome_id"]}
    by_exit = {row["outcome_id"]: row for row in exits if row["outcome_id"]}

    print(f"db: {args.db}")
    print(f"live-mode settled trade_outcomes rows : {len(outcomes)}")
    print(f"trading_ops status='live-exit' rows    : {len(exits)}")
    print(f"live-exit rows carrying a real 66-char tx_hash: "
          f"{sum(1 for r in exits if _is_hash(r['tx_hash']))} of {len(exits)}")
    print()

    # --- the join, asserted total in BOTH directions -----------------------
    outcomes_without_exit = [o for o in outcomes if o["outcome_id"] not in by_exit]
    exits_without_outcome = [e for e in exits if e["outcome_id"] not in by_outcome]
    total = not outcomes_without_exit and not exits_without_outcome

    if total:
        print(f"JOIN TOTAL BOTH WAYS: {len(by_outcome)} outcomes <-> {len(by_exit)} live-exits, "
              "no orphan on either side.")
    else:
        print("JOIN IS NOT TOTAL. The orphans matter more than the total:")
        for o in outcomes_without_exit:
            print(f"  OUTCOME WITH NO live-exit  {o['outcome_id']} {o['symbol']} "
                  f"net={o['net']:+.5f}  <- a row claiming a fill that left no trace")
        for e in exits_without_outcome:
            print(f"  live-exit WITH NO OUTCOME  {e['outcome_id'] or '(no outcome_id)'} "
                  f"{e['symbol']} tx={e['tx_hash'][:14]}  <- an unrecorded fill")
    print()

    # --- per-row provenance ------------------------------------------------
    header = (
        f"{'outcome_id':34} {'symbol':13} {'strategy':18} {'hold_min':>9} "
        f"{'gross':>9} {'fee':>8} {'net':>9}  {'route':14} {'fill':10} entry_tx / exit_tx"
    )
    print(header)
    print("-" * len(header))
    for o in sorted(outcomes, key=lambda r: r["ts"] or 0):
        e = by_exit.get(o["outcome_id"], {})
        dur = e.get("duration_sec")
        hold = f"{float(dur) / 60.0:9.1f}" if dur not in (None, "") else "        ?"
        route = e.get("route_used")
        route = ",".join(route) if isinstance(route, list) else str(route or "?")
        print(
            f"{o['outcome_id'][:34]:34} {str(o['symbol'])[:13]:13} "
            f"{str(o['strategy_id'])[:18]:18} {hold} "
            f"{o['gross']:+9.5f} {o['fee']:8.5f} {o['net']:+9.5f}  "
            f"{route[:14]:14} {str(e.get('fill_source') or '?')[:10]:10} "
            f"{(e.get('entry_tx_hash') or '(none)')[:14]} / {(e.get('tx_hash') or '(none)')[:14]}"
        )
    print()

    # --- the totals, with the censor stated both ways ----------------------
    kept, censored, censor_state = _censor(outcomes)
    print(f"plausibility censor: {censor_state}")
    print(f"  kept {len(kept)} row(s), censored {len(censored)} row(s)")
    if censored:
        for row in censored:
            print(f"  CENSORED {row.get('outcome_id')} {row.get('symbol')} "
                  f"gross={row.get('gross')} -- excluded from the totals below")

    gross = sum(r["gross"] for r in kept)
    fee = sum(r["fee"] for r in kept)
    net = sum(r["net"] for r in kept)
    print()
    print(f"SCOREBOARD over n={len(kept)}:  gross {gross:+.6f}   fee {fee:.6f}   net {net:+.6f}")
    print("  NOTE: gross is NEGATIVE before a single fee is charged, so at a cost "
          "floor of exactly zero this book is still under water. Subtracting cost "
          "cannot make a negative gross positive.")

    # --- hold time: route (c) on the book that counts ----------------------
    holds = [
        float(by_exit[o["outcome_id"]]["duration_sec"]) / 60.0
        for o in kept
        if o["outcome_id"] in by_exit
        and by_exit[o["outcome_id"]].get("duration_sec") not in (None, "")
    ]
    print()
    if holds:
        under = [o for o in kept
                 if o["outcome_id"] in by_exit
                 and by_exit[o["outcome_id"]].get("duration_sec") not in (None, "")
                 and float(by_exit[o["outcome_id"]]["duration_sec"]) < 15 * 60]
        over = [o for o in kept if o not in under]
        print(f"HOLD TIME on {len(holds)} of {len(kept)} rows: "
              f"min {min(holds):.1f} min  p50 {statistics.median(holds):.1f} min  "
              f"max {max(holds):.1f} min")
        print(f"  under 15 min: n={len(under)}  gross {sum(r['gross'] for r in under):+.5f}  "
              f"net {sum(r['net'] for r in under):+.5f}")
        print(f"  15 min plus : n={len(over)}  gross {sum(r['gross'] for r in over):+.5f}  "
              f"net {sum(r['net'] for r in over):+.5f}")
        if not under:
            print("  THE STANDING INSTRUCTION 'the tradeable book is POSITIVE inside "
                  "15 minutes' HAS AN EMPTY POSITIVE BUCKET on the live book: not "
                  "confirmed, not refuted, NO ROWS. Do not cost route (c) off it.")
    else:
        print("HOLD TIME: no joined row carries duration_sec -- report this as "
              "unmeasurable rather than substituting a proxy.")

    # --- reconciliation ----------------------------------------------------
    rc = 0
    if args.expect_net is not None:
        if abs(net - args.expect_net) <= 5e-5:
            print()
            print(f"RECONCILED to the expected scoreboard {args.expect_net:+.4f} (4dp).")
        else:
            print()
            print(f"DIVERGENCE: measured net {net:+.6f} vs expected {args.expect_net:+.4f}. "
                  "STOP and find out which: a NEW SETTLED TRADE (the goal) or a book "
                  "that has drifted. Do not average them.")
            rc = 2
    if not total:
        rc = max(rc, 3)

    print()
    print("WHAT THIS DOES NOT DO: it verifies no hash against the chain. A row that "
          "reconciles internally and does not exist on Base is exactly the failure "
          "this repo has shipped four times, and only an RPC call to Base can rule "
          "it out. Check the largest-magnitude rows first.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
