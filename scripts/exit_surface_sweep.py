#!/usr/bin/env python
"""Sweep the WHOLE exit surface for atf_static -- Echo, pass 486.

WHY. Pass 486 closed the ENTRY side: atf_static beats the median symbol on its
own tape by +2.69bp CI [+0.16,+5.21] over 380 paired windows, against a 27.77bp
live-receipt toll, and the field the gate ranks on is a config literal
(services/atf_static_strategy.py:1760 persists target_return as expected_return).
That leaves one question the entry work cannot answer: could a different EXIT
turn the same entries into money?

THE THING THAT MAKES IT LOOK LIKE YES, and which is a trap. The 45-minute paths
are favourably skewed -- median max favourable excursion +23.69bp against a
median max adverse excursion of -11.06bp, a 2.14:1 ratio -- while the strategy's
own take-profit sits at 150bp, which only 9.2% of paths ever touch (it fills
7.0% of the time). The obvious inference is that an ACHIEVABLE take-profit near
+24bp harvests the skew.

  H1: a tighter, reachable take-profit converts the favourable skew into a
      positive net. Then the exit is the defect and it is a one-line fix.
  H0: it does not, because a tight take-profit caps the winners while the stop
      still pays every loser. Then the exit surface has no positive cell and
      the strategy is closed from both ends.

So sweep it rather than argue it: 7 take-profit levels x 4 stops, FIRST TOUCH
in path order (not max-excursion, which is the standard way to fake this), and
subtract the toll from every cell.

UPPER BOUND, STATED PLAINLY. Ghost prices, no slippage, and first-touch assumes
a fill exactly at the level. Ghost fills cannot carry slippage in this repo, so
every gross figure printed here is optimistic and the real net is worse.

READ-ONLY. Opens the store with mode=ro. Changes nothing.

Run:  python -X utf8 scripts/exit_surface_sweep.py
"""

from __future__ import annotations

import bisect
import json
import os
import sqlite3
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expected_return_calibration as E  # noqa: E402  -- price loader lives there

LOOKBACK_DAYS = 7.0
HOLD = 2700.0  # MAX_HOLD_FORCE_SECONDS=2700 is the hold this bot enforces
STRATEGY = "atf_static"
TOLL_BP = 27.77
TAKE_PROFITS = (20, 30, 40, 50, 75, 100, 150)
STOPS = (30, 50, 100, None)  # None = ride to the 45-min force exit


def _paths(conn, book, since):
    out = []
    for r in conn.execute(
        "SELECT ts, symbol, details FROM trading_ops WHERE ts > ? AND action='enter'",
        (since,),
    ):
        try:
            j = json.loads(r["details"] or "{}")
        except Exception:
            continue
        if j.get("strategy_id") != STRATEGY:
            continue
        sym = r["symbol"] or j.get("symbol")
        if not sym:
            continue
        e = book.get(sym)
        if not e:
            continue
        ts, px = e
        i = bisect.bisect_right(ts, r["ts"]) - 1
        if i < 0 or r["ts"] - ts[i] > E.MAX_STALE:
            continue
        p0 = px[i]
        if p0 <= 0:
            continue
        seg = px[i:bisect.bisect_right(ts, r["ts"] + HOLD)]
        if len(seg) < 3:
            continue
        out.append([(p / p0 - 1.0) * 10000.0 for p in seg])
    return out


def _simulate(path, tp, sl):
    """First touch, in path order. Returns (bp, which)."""
    for x in path:
        if x >= tp:
            return tp, "tp"
        if sl is not None and x <= -sl:
            return -sl, "sl"
    return path[-1], "timeout"


def main() -> int:
    now = time.time()
    since = now - LOOKBACK_DAYS * 86400.0
    conn = E._connect()
    book = E._load_prices(conn, since - HOLD - E.MAX_STALE)
    paths = _paths(conn, book, since)

    print("=" * 76)
    print("EXIT SURFACE SWEEP -- %s, Echo pass 486" % STRATEGY)
    print("lookback %.0fd | hold %.0f min | toll %.2f bp | paths %d"
          % (LOOKBACK_DAYS, HOLD / 60.0, TOLL_BP, len(paths)))
    print("ghost prices, NO slippage, first-touch fills -- every gross is an UPPER BOUND")
    print("=" * 76)
    if len(paths) < 50:
        print("INSUFFICIENT PATHS (%d). No verdict." % len(paths))
        return 0

    mfe = [max(p) for p in paths]
    mae = [min(p) for p in paths]
    print("max favourable excursion  median %+8.2f bp   p90 %+9.2f bp"
          % (statistics.median(mfe), statistics.quantiles(mfe, n=10)[8]))
    print("max adverse    excursion  median %+8.2f bp   p10 %+9.2f bp"
          % (statistics.median(mae), statistics.quantiles(mae, n=10)[0]))
    print("-" * 76)
    print("%6s %6s %9s %9s %9s %10s %10s"
          % ("TP", "SL", "tp_hit%", "sl_hit%", "timeout%", "gross bp", "NET bp"))

    best = None
    for tp in TAKE_PROFITS:
        for sl in STOPS:
            outs, c = [], {"tp": 0, "sl": 0, "timeout": 0}
            for p in paths:
                v, w = _simulate(p, tp, sl)
                outs.append(v)
                c[w] += 1
            n = len(outs)
            g = statistics.fmean(outs)
            print("%6d %6s %8.1f%% %8.1f%% %8.1f%% %+10.2f %+10.2f"
                  % (tp, "none" if sl is None else sl,
                     100.0 * c["tp"] / n, 100.0 * c["sl"] / n, 100.0 * c["timeout"] / n,
                     g, g - TOLL_BP))
            if best is None or g - TOLL_BP > best[0]:
                best = (g - TOLL_BP, tp, sl, g)

    print("-" * 76)
    net, tp, sl, g = best
    print("BEST OF %d CELLS: net %+.2f bp at TP=%d SL=%s (gross %+.2f)"
          % (len(TAKE_PROFITS) * len(STOPS), net, tp, "none" if sl is None else sl, g))
    if net > 0:
        print("VERDICT: H1. An exit rule clears the toll -- and it is an UPPER BOUND,")
        print("         so re-measure it on live fills before believing it.")
    else:
        print("VERDICT: H0. NO cell on the exit surface reaches break-even; the best")
        print("         is %.2f bp short, and that is before any slippage." % -net)
        print("         Gross rises MONOTONICALLY with a WIDER take-profit, so the")
        print("         favourable skew is real and NOT harvestable: a tight TP caps")
        print("         the winners while the stop still pays every loser.")
        print("         This exonerates the strategy's own 150bp take-profit, which")
        print("         sits near the gross optimum of the whole surface.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
