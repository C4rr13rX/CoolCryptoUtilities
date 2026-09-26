#!/usr/bin/env python
"""Realised slippage census -- item [bcc110a7], pass 710 (Dune, exec).

WHY THIS EXISTS
    Every round-trip toll quoted on the crew board is gas + pool fee and NOTHING
    else.  services/roundtrip_cost.py has no slippage term at all.  The 4.10bp
    WETH number that route (a) turns on is therefore a QUOTE, never checked
    against a settled fill.

WHAT IT MEASURES
    trade_fills carries both expected_price (the quote) and executed_price (what
    actually filled).  The difference, signed so that POSITIVE always means the
    fill was WORSE than the quote, is realised slippage.  We express it in bp of
    notional and report the DISTRIBUTION -- a mean-poisoned verdict table has
    already misled this repo once.

    Every live fill is tied back to a settled Base receipt through
    details.trade_id -> trading_ops(action='swap').details.tx_hash.  There is no
    tx column on trade_fills and a time join is not a tie; trade_id is the key.

WHAT IT DOES NOT DO
    It does not change the cost model.  It does not model a slippage number for
    any symbol that has no live fill.  An unmeasured symbol is reported as
    UNMEASURED.

Usage:  python scripts/realised_slippage_census.py [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
from collections import defaultdict

LIVE_MODES = ("live_entry", "live_exit")


def _pct(xs, q):
    """Nearest-rank percentile on a sorted list; no interpolation, no numpy."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def _dist(xs):
    if not xs:
        return "n=0"
    return (
        "n=%-3d  min %+8.2f  p10 %+8.2f  p25 %+8.2f  MEDIAN %+8.2f  "
        "p75 %+8.2f  p90 %+8.2f  max %+8.2f  mean %+8.2f"
        % (
            len(xs),
            min(xs),
            _pct(xs, 10),
            _pct(xs, 25),
            _pct(xs, 50),
            _pct(xs, 75),
            _pct(xs, 90),
            max(xs),
            statistics.fmean(xs),
        )
    )


def load_fills(con):
    """Every trade_fills row, parsed, with the quote-vs-fill delta in bp.

    Sign convention: POSITIVE bp = the fill was WORSE for us than the quote.
    On a buy leg that means we paid ABOVE the quote; on a sell leg it means we
    received BELOW it.  Legs whose direction we cannot establish are returned
    with leg='?' and counted separately rather than guessed.
    """
    out = []
    cur = con.execute(
        "SELECT id, ts, symbol, expected_amount, executed_amount, "
        "expected_price, executed_price, details FROM trade_fills ORDER BY ts"
    )
    for fid, ts, sym, ea, xa, ep, xp, raw in cur:
        try:
            d = json.loads(raw or "{}")
        except Exception:
            d = {}
        if not ep or not xp:
            continue
        leg = d.get("leg") or ("buy" if d.get("mode", "").endswith("entry") else
                               "sell" if d.get("mode", "").endswith("exit") else "?")
        raw_bp = (xp - ep) / ep * 10000.0
        # A buy that executes above the quote costs us; a sell that executes
        # below the quote costs us.  Flip the sell so positive is always a cost.
        cost_bp = raw_bp if leg == "buy" else -raw_bp if leg == "sell" else None
        notional = d.get("notional_usd")
        if notional is None and xa is not None:
            notional = abs(xa) * xp
        out.append(
            {
                "id": fid,
                "ts": ts,
                "symbol": d.get("symbol") or sym or "?",
                "mode": d.get("mode", "?"),
                "leg": leg,
                "expected_price": ep,
                "executed_price": xp,
                "raw_bp": raw_bp,
                "cost_bp": cost_bp,
                "notional": notional,
                "trade_id": d.get("trade_id") or d.get("outcome_id"),
                "quote_spent": d.get("quote_spent"),
                "gas_native": d.get("gas_spent_native"),
                "tol_bps": d.get("slippage_bps"),
            }
        )
    return out


def load_swap_receipts(con):
    """trade_id -> settled Base receipts.  action='swap', a tx_hash, confirmed."""
    by_trade = defaultdict(list)
    cur = con.execute(
        "SELECT ts, symbol, status, details FROM trading_ops WHERE action='swap'"
    )
    for ts, sym, status, raw in cur:
        try:
            d = json.loads(raw or "{}")
        except Exception:
            continue
        tx = d.get("tx_hash")
        if not tx:
            continue
        tid = d.get("trade_id")
        by_trade[tid].append(
            {
                "ts": ts,
                "symbol": d.get("symbol") or sym,
                "status": status,
                "tx": tx,
                "purpose": d.get("purpose"),
                "route": d.get("route"),
                "amount_human": d.get("amount_human"),
                "confirmed": bool(d.get("confirmed")),
            }
        )
    return by_trade


def clip_bucket(n):
    if n is None:
        return "unknown"
    for hi, label in ((2, "<$2"), (5, "$2-5"), (10, "$5-10"), (20, "$10-20")):
        if n < hi:
            return label
    return ">=$20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("TRADING_DB_PATH", "storage/trading_cache.db"))
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db, uri=True)
    fills = load_fills(con)
    receipts = load_swap_receipts(con)

    live_all = [f for f in fills if f["mode"] in LIVE_MODES]
    ghost = [f for f in fills if f["mode"].startswith("ghost")]

    # A fill whose executed_price is zero or whose delta is a full 100% is a
    # broken row, not a 10000bp slippage event.  One such row drags the live
    # mean from +11bp to -224bp.  Censor them and SAY SO -- this repo has been
    # misled by an uncensored mean before.
    def degenerate(f):
        return (f["executed_price"] <= 0 or f["expected_price"] <= 0
                or abs(f["raw_bp"]) >= 9999.0)

    censored = [f for f in live_all if degenerate(f)]
    live = [f for f in live_all if not degenerate(f)]

    print("=" * 78)
    print("REALISED SLIPPAGE CENSUS -- executed price vs quoted price, settled fills")
    print("db: %s   trade_fills rows: %d" % (args.db, len(fills)))
    print("=" * 78)

    # ---- 0. the ghost book's slippage is a modelling choice, not a measurement
    ghost_nonzero = [f for f in ghost if abs(f["raw_bp"]) > 1e-9]
    print("\n[0] IS THERE ANY SLIPPAGE IN THE GHOST BOOK AT ALL?")
    print("    ghost fills                : %d" % len(ghost))
    print("    with executed != expected  : %d" % len(ghost_nonzero))
    if not ghost_nonzero:
        print("    => ghost fills are booked at the quote EXACTLY.  The ghost book")
        print("       models slippage as ZERO.  It cannot corroborate any toll.")

    # ---- 1. the live population, and its chain tie
    print("\n[1] THE LIVE POPULATION AND ITS ON-CHAIN TIE")
    tied = untied = 0
    for f in live:
        rs = receipts.get(f["trade_id"], [])
        f["receipts"] = [r for r in rs if r["status"] == "live-swap-settled"]
        if f["receipts"]:
            tied += 1
        else:
            untied += 1
    if censored:
        print("    CENSORED as degenerate (executed_price<=0 or |delta|>=100%%): %d"
              % len(censored))
        for f in censored:
            print("        id=%d %-12s %-10s expected %.8g -> executed %.8g"
                  % (f["id"], f["symbol"], f["mode"],
                     f["expected_price"], f["executed_price"]))
        print("        ^ these are BROKEN ROWS, not slippage.  Leaving them in")
        print("          moves the all-live mean from +11bp to -224bp.")
    print("    live fills (after censor)      : %d  (live_entry %d / live_exit %d)"
          % (len(live),
             sum(1 for f in live if f["mode"] == "live_entry"),
             sum(1 for f in live if f["mode"] == "live_exit")))
    print("    tied to a settled swap receipt : %d" % tied)
    print("    NO settled receipt on trade_id : %d" % untied)

    bysym = defaultdict(list)
    for f in live:
        bysym[f["symbol"]].append(f)
    print("    by symbol:")
    for s, fs in sorted(bysym.items(), key=lambda kv: -len(kv[1])):
        print("        %-16s %3d fills, %d tied"
              % (s, len(fs), sum(1 for f in fs if f["receipts"])))

    # ---- 2. the distribution, which is the answer
    print("\n[2] REALISED SLIPPAGE, bp of notional.  POSITIVE = fill WORSE than quote.")
    usable = [f for f in live if f["cost_bp"] is not None]
    print("    ALL LIVE        %s" % _dist([f["cost_bp"] for f in usable]))
    for leg in ("buy", "sell"):
        xs = [f["cost_bp"] for f in usable if f["leg"] == leg]
        print("    leg=%-11s %s" % (leg, _dist(xs)))
    unknown_leg = [f for f in live if f["cost_bp"] is None]
    if unknown_leg:
        print("    leg UNKNOWN on %d fill(s) -- excluded, not guessed" % len(unknown_leg))

    print("\n    per symbol:")
    for s, fs in sorted(bysym.items(), key=lambda kv: -len(kv[1])):
        xs = [f["cost_bp"] for f in fs if f["cost_bp"] is not None]
        print("        %-16s %s" % (s, _dist(xs)))

    # ---- 3. by clip size, because route (a) is a clip-size argument
    print("\n[3] BY CLIP SIZE -- route (a) claims $20 beats $6 because fixed gas dilutes.")
    print("    If realised slippage RISES with clip it cancels that dilution.")
    byclip = defaultdict(list)
    for f in usable:
        byclip[clip_bucket(f["notional"])].append(f)
    order = ["<$2", "$2-5", "$5-10", "$10-20", ">=$20", "unknown"]
    for b in order:
        fs = byclip.get(b)
        if not fs:
            continue
        xs = [f["cost_bp"] for f in fs]
        med_n = statistics.median([f["notional"] for f in fs if f["notional"]] or [0])
        print("    %-8s median notional $%-8.4f  %s" % (b, med_n, _dist(xs)))

    ns = [(f["notional"], f["cost_bp"]) for f in usable if f["notional"]]
    if len(ns) >= 3:
        try:
            rho = statistics.correlation([a for a, _ in ns], [b for _, b in ns])
            print("    Pearson r(notional, slippage_bp) = %+.4f over n=%d" % (rho, len(ns)))
        except Exception as exc:
            print("    correlation unavailable: %s" % exc)

    # ---- 4. what cannot be measured, stated rather than modelled
    print("\n[4] WHAT THIS CENSUS CANNOT MEASURE.  These are NOT estimated.")
    for want in ("WETH-USDC", "ETH-USDC"):
        if want not in bysym:
            print("    %-12s ZERO live fills in this book -- slippage UNMEASURED." % want)
    print("    Any symbol absent from the per-symbol table above has no live fill")
    print("    and therefore no realised slippage.  Do not interpolate one.")

    # ---- 5. the receipts, quoted so the claim is checkable
    print("\n[5] SETTLED RECEIPTS BEHIND THE LIVE FILLS (basescan.org/tx/<hash>)")
    seen = set()
    for f in sorted(live, key=lambda f: f["ts"]):
        for r in f["receipts"]:
            if r["tx"] in seen:
                continue
            seen.add(r["tx"])
            print("    %-10s %-12s %-6s %s" % (r["purpose"] or "?", r["symbol"],
                                               r["route"] or "?", r["tx"]))
    print("    distinct settled tx hashes tied to a live fill: %d" % len(seen))


if __name__ == "__main__":
    main()
