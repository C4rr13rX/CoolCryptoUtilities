#!/usr/bin/env python3
"""
How many dips does the money button actually get offered, and how many of them
are in a pool deep enough to fill?

WHY THIS EXISTS. Two competing explanations for why the money button does not
trade: it is starved of SIGNAL (the gates refuse everything), or it is starved
of TRADEABLE signal (the gates pass plenty, but in pools that cannot absorb a
clip). Those call for opposite fixes -- loosen a threshold, or restrict the
universe -- so guessing between them is expensive.

This counts the supply. It replays the stored tape and applies the money
button's two PRICE-ACTION gates exactly as ``trading/scheduler.py`` computes
them:

  DRAWDOWN -- peak-to-last over MONEY_BUTTON_LOOKBACK_SEC (default 7200s),
    admitted in the band [MONEY_BUTTON_MIN_DRAWDOWN, MONEY_BUTTON_MAX_DRAWDOWN]
    (default 2% to 5%).
  SLOPE -- np.polyfit of log price against MINUTES over
    MONEY_BUTTON_SLOPE_WINDOW_SEC (default 1800s), refused below
    MONEY_BUTTON_SLOPE_FLOOR (default -0.00025 per minute). This is the "don't
    catch a falling knife" guard.

Note the two windows are DIFFERENT LENGTHS -- drawdown over 2h, slope over
30min -- so the pair admits a dip whose fall is older than the last half hour
and refuses one still in progress. That asymmetry is the behaviour worth
measuring, not a bug on its face.

WHAT THIS IS NOT, and the limit matters. Surviving both price-action gates is
NOT an entry. The cost gate and the net-expected-return gate sit downstream of
the slope check and refuse more. This measures the SUPPLY those downstream
gates get to choose from -- a ceiling on opportunity, never a trade count.

THE TRAP IT IS BUILT AGAINST. A dip count summed across symbols hides where
the dips are. This repo has already established that dip supply concentrates in
thin tokens, and a supply number that cannot be filled is not supply. So the
count is reported PER SYMBOL, and a depth table (if given) splits it into the
part that could plausibly fill and the part that could not.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:  # noqa: BLE001
        return default


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", default=None, help="trading store (default TRADING_DB_PATH)")
    ap.add_argument("--depth-json", default=None,
                    help="optional {SYMBOL: liquidity_usd} snapshot to split supply by depth")
    ap.add_argument("--depth-bar", type=float, default=1_000_000.0,
                    help="pool liquidity USD at or above which a dip is called fillable")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    path = args.db or os.environ.get(
        "TRADING_DB_PATH", str(ROOT / "storage" / "trading_cache.db"))
    if not Path(path).exists():
        print("dip supply: no trading store at %s" % path)
        return 2

    lookback = _env_float("MONEY_BUTTON_LOOKBACK_SEC", 7200.0)
    slope_window = _env_float("MONEY_BUTTON_SLOPE_WINDOW_SEC", 1800.0)
    slope_floor = _env_float("MONEY_BUTTON_SLOPE_FLOOR", -0.00025)
    min_dd = _env_float("MONEY_BUTTON_MIN_DRAWDOWN", 0.02)
    max_dd = _env_float("MONEY_BUTTON_MAX_DRAWDOWN", 0.05)

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("select symbol, ts, price from market_stream "
                "where price > 0 order by symbol, ts")
    tapes: dict[str, list] = defaultdict(list)
    for sym, ts, price in cur.fetchall():
        tapes[sym].append((float(ts), float(price)))

    evaluated = dd_pass = knife_kill = both_pass = 0
    per_sym: dict[str, list] = defaultdict(lambda: [0, 0])

    for sym, rows in tapes.items():
        if len(rows) < 20:
            continue
        ts = np.array([r[0] for r in rows])
        px = np.array([r[1] for r in rows])
        for i in range(len(rows)):
            t = ts[i]
            if t - ts[0] < lookback:
                continue  # not enough history for an honest drawdown
            evaluated += 1
            window = px[(ts >= t - lookback) & (ts <= t)]
            if window.size < 3:
                continue
            peak = window.max()
            if peak <= 0:
                continue
            drawdown = (px[i] - peak) / peak
            if drawdown > -min_dd or drawdown < -max_dd:
                continue
            dd_pass += 1
            per_sym[sym][0] += 1

            mask = (ts >= t - slope_window) & (ts <= t)
            st, sp = ts[mask], px[mask]
            if st.size < 6:
                continue  # the real gate skips the slope check here too
            rel_minutes = (st - st[-1]) / 60.0
            if np.allclose(rel_minutes, rel_minutes[0]):
                continue
            slope, _ = np.polyfit(
                rel_minutes, np.log(np.clip(sp, 1e-9, 1e9)), 1)
            if slope < slope_floor:
                knife_kill += 1
            else:
                both_pass += 1
                per_sym[sym][1] += 1

    print("=" * 74)
    print("MONEY BUTTON DIP SUPPLY")
    print("=" * 74)
    print("store: %s" % path)
    print("gates: drawdown %.0f%%-%.0f%% over %.0fs   slope floor %.5f/min over %.0fs"
          % (100 * min_dd, 100 * max_dd, lookback, slope_floor, slope_window))
    print()
    print("  ticks evaluated (carrying full history) : %d" % evaluated)
    print("  passed DRAWDOWN gate                    : %d" % dd_pass)
    print("    refused by the knife guard            : %d" % knife_kill)
    print("    passed BOTH price-action gates        : %d" % both_pass)
    if dd_pass:
        print("  knife guard refusal rate                : %.1f%%"
              % (100.0 * knife_kill / dd_pass))
    print()
    print("Surviving both gates is NOT an entry: the cost and net-expected-return")
    print("gates sit downstream and refuse more. This is a ceiling on opportunity.")
    print()

    depth = {}
    if args.depth_json and Path(args.depth_json).exists():
        try:
            depth = json.loads(Path(args.depth_json).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            depth = {}

    print("DIP SUPPLY BY SYMBOL -- where the opportunity actually is:")
    print()
    head = "    %-18s %7s %9s" % ("symbol", "dips", "survived")
    print(head + ("  %12s" % "pool liq USD" if depth else ""))
    fillable = 0
    for sym, (n, survived) in sorted(per_sym.items(), key=lambda kv: -kv[1][0])[: args.top]:
        line = "    %-18s %7d %9d" % (sym, n, survived)
        if depth:
            base = sym.split("-")[0].upper()
            liq = depth.get(base)
            line += "  %12s" % ("%.0f" % liq if liq is not None else "?")
            if liq is not None and liq >= args.depth_bar:
                fillable += survived
        print(line)
    if depth:
        print()
        print("  admissible dips in pools at or above $%.0f : %d of %d"
              % (args.depth_bar, fillable, both_pass))
        print("  Depth is a SNAPSHOT: a pool can have drained or filled since the")
        print("  tick. Depth attributed to an old tick is an inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
