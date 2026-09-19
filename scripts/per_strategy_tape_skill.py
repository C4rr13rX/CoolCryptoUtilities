#!/usr/bin/env python
"""Does ANY strategy beat the tape? -- Echo, pass 486.

WHAT THIS OVERTURNS. Pass 485 concluded that the non-atf strategies persist no
per-tick signal, so testing their skill was an INSTRUMENTATION job ([e1101dc6],
5pt). That is wrong for the decision that matters. trading_ops rows with
action='enter' carry a details JSON containing strategy_id -- and there are
2815 of them in 7 days, not the 59 ghost-entry rows the matched-baseline test
used. Every one is a dated, symbol-tagged entry PROPOSAL attributed to a named
strategy. That is enough to rank strategies on realised forward return today,
with no code change and no restart.

THE QUESTION. For each strategy: do the symbols it proposes, at the moments it
proposes them, outperform the MEDIAN symbol on the same tape in the same
10-minute window?

  H1: at least one strategy clears the 27.77bp live-receipt cost floor.
      Then hand-approve THAT strategy at a minimum clip -- route (a) becomes
      live with a named candidate instead of a guess.
  H0: every strategy ties the tape inside its CI.
      Then no strategy on this roster can pay the toll, and neither cost work
      nor gate work nor a higher evidence rate changes that -- the signal
      itself is the missing ingredient.

WHY PAIRED AGAINST A MEDIAN. This tape drifts and carries eight corrupt price
regimes (one window implies a 7e13x move); an unpaired mean is meaningless
here. Pass 486 measured mean-of-window-MEANS at +167333bp against a
mean-of-window-MEDIANS of +0.94bp on identical rows. Every aggregate below is
a median across symbols within a window, differenced inside that window, so
drift cancels and no single corrupt symbol can carry a verdict.

REPORTED FOR EACH STRATEGY: the paired difference vs tape, its 95% CI, the
share of windows it won, and the minimum detectable effect against the cost
floor -- so a null can be read as "no edge" rather than "no data".

READ-ONLY. Opens the store with mode=ro. Changes nothing.

Run:  python -X utf8 scripts/per_strategy_tape_skill.py
"""

from __future__ import annotations

import bisect
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

LOOKBACK_DAYS = 7.0
HORIZON = 900.0  # 15 min -- the tradeable book is positive inside it
WINDOW = 600.0
MAX_STALE = 600.0
COST_FLOOR_BP = 27.77
MIN_WINDOWS = 5  # below this a CI is theatre; the strategy is reported as thin


def _connect():
    conn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _load_prices(conn, since):
    per = defaultdict(list)
    for r in conn.execute(
        "SELECT symbol, ts, price FROM market_stream WHERE ts > ? AND price > 0 ORDER BY ts",
        (since,),
    ):
        per[r["symbol"]].append((r["ts"], r["price"]))
    return {
        s: ([t for t, _ in rows], [p for _, p in rows])
        for s, rows in ((s, sorted(v, key=lambda x: x[0])) for s, v in per.items())
    }


def _price_at(book, sym, t):
    e = book.get(sym)
    if not e:
        return None
    ts, px = e
    i = bisect.bisect_right(ts, t) - 1
    if i < 0 or t - ts[i] > MAX_STALE:
        return None
    return px[i]


def _fwd(book, sym, t, h):
    p0 = _price_at(book, sym, t)
    p1 = _price_at(book, sym, t + h)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    r = p1 / p0 - 1.0
    return r * 10000.0 if math.isfinite(r) else None


def _ci(xs):
    n = len(xs)
    if n < 2:
        return (statistics.fmean(xs) if xs else float("nan")), float("nan"), float("nan")
    m = statistics.fmean(xs)
    half = 1.96 * statistics.stdev(xs) / math.sqrt(n)
    return m, m - half, m + half


def main() -> int:
    now = time.time()
    since = now - LOOKBACK_DAYS * 86400.0
    conn = _connect()

    print("=" * 84)
    print("PER-STRATEGY SKILL AGAINST THE TAPE -- Echo pass 486")
    print("lookback %.0fd | horizon %.0f min | window %.0fs | cost floor %.2fbp"
          % (LOOKBACK_DAYS, HORIZON / 60.0, WINDOW, COST_FLOOR_BP))
    print("=" * 84)

    book = _load_prices(conn, since - HORIZON - MAX_STALE)

    # ---- entry proposals, attributed ---------------------------------------
    by_strat = defaultdict(list)
    seen_status = defaultdict(lambda: defaultdict(int))
    total, attributed = 0, 0
    for r in conn.execute(
        "SELECT ts, symbol, status, details FROM trading_ops WHERE ts > ? AND action='enter'",
        (since,),
    ):
        total += 1
        d = r["details"]
        if not d:
            continue
        try:
            j = json.loads(d)
        except Exception:
            continue
        sid = j.get("strategy_id")
        if not sid:
            continue
        attributed += 1
        sym = r["symbol"] or j.get("symbol")
        if not sym:
            continue
        by_strat[sid].append((r["ts"], sym))
        seen_status[sid][r["status"] or "?"] += 1

    print("enter rows %d | with strategy_id %d | distinct strategies %d"
          % (total, attributed, len(by_strat)))

    # ---- the tape: median across symbols, per window ------------------------
    windows_needed = set()
    for rows in by_strat.values():
        for t, _ in rows:
            windows_needed.add(int(t // WINDOW))
    tape = {}
    for w in sorted(windows_needed):
        t = w * WINDOW
        vals = [v for sym in book if (v := _fwd(book, sym, t, HORIZON)) is not None]
        if len(vals) >= 3:
            tape[w] = statistics.median(vals)
    print("tape windows priced: %d of %d touched by a proposal"
          % (len(tape), len(windows_needed)))

    # ---- per strategy -------------------------------------------------------
    results = []
    for sid, rows in by_strat.items():
        by_w = defaultdict(list)
        for t, sym in rows:
            v = _fwd(book, sym, t, HORIZON)
            if v is not None:
                by_w[int(t // WINDOW)].append(v)
        diffs, own = [], []
        for w, vals in by_w.items():
            if w not in tape:
                continue
            m = statistics.median(vals)
            diffs.append(m - tape[w])
            own.append(m)
        if not diffs:
            continue
        d, lo, hi = _ci(diffs)
        wins = sum(1 for x in diffs if x > 0)
        mde = (1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))
               if len(diffs) > 1 else float("nan"))
        results.append({
            "sid": sid, "rows": len(rows), "priced": sum(len(v) for v in by_w.values()),
            "windows": len(diffs), "diff": d, "lo": lo, "hi": hi,
            "own": statistics.fmean(own), "wins": wins, "mde": mde,
        })

    results.sort(key=lambda r: (-r["windows"], -r["diff"]))

    print("-" * 84)
    print("%-26s %6s %7s %10s %19s %8s %8s"
          % ("strategy", "props", "windows", "diff(bp)", "95% CI", "win%", "MDE"))
    print("-" * 84)
    for r in results:
        if r["windows"] < MIN_WINDOWS:
            print("%-26s %6d %7d %10.2f %19s %8s %8s   THIN"
                  % (r["sid"][:26], r["rows"], r["windows"], r["diff"], "--", "--", "--"))
            continue
        print("%-26s %6d %7d %+10.2f [%+8.2f,%+8.2f] %7.1f%% %8.2f"
              % (r["sid"][:26], r["rows"], r["windows"], r["diff"],
                 r["lo"], r["hi"], 100.0 * r["wins"] / r["windows"], r["mde"]))

    print("-" * 84)
    testable = [r for r in results if r["windows"] >= MIN_WINDOWS]
    clears = [r for r in testable if r["lo"] > COST_FLOOR_BP]
    positive = [r for r in testable if r["lo"] > 0.0]
    powered = [r for r in testable if r["mde"] < COST_FLOOR_BP]
    print("strategies with >= %d paired windows : %d" % (MIN_WINDOWS, len(testable)))
    print("  CI excludes zero (any edge at all) : %d" % len(positive))
    print("  CI clears the %.2fbp cost floor     : %d" % (COST_FLOOR_BP, len(clears)))
    print("  powered to SEE the cost floor       : %d" % len(powered))
    print()
    if clears:
        print("VERDICT: H1. Named candidates for a hand-approved minimum-clip live lane:")
        for r in clears:
            print("   %-26s %+.2f bp [%+.2f, %+.2f] over %d windows"
                  % (r["sid"], r["diff"], r["lo"], r["hi"], r["windows"]))
    elif positive:
        print("VERDICT: partial. Something beats the tape but NOT the cost floor:")
        for r in positive:
            print("   %-26s %+.2f bp [%+.2f, %+.2f] -- floor is %.2f"
                  % (r["sid"], r["diff"], r["lo"], r["hi"], COST_FLOOR_BP))
        print("   Cost work becomes decisive ONLY for these names.")
    else:
        print("VERDICT: H0. NO strategy on this roster beats the median symbol on")
        print("         its own tape. %d of %d tested strategies are powered to see"
              % (len(powered), len(testable)))
        print("         the cost floor, so this is an edge famine, not a data famine.")
        print("         Cost work, gate work and evidence-rate work all multiply zero.")

    # ---- does the edge concentrate in the CHEAP symbols? --------------------
    # A pooled 27.77bp toll is not the toll on every token: the live-fill
    # census puts AERO near 10.8bp and WETH round trips at 4-11bp, while KTA
    # has been measured up to 74.98%. So a strategy whose edge sits in the
    # cheap names could clear a per-symbol floor it cannot clear pooled. This
    # is the one remaining way route (b), attack the cost floor, stays open.
    if testable:
        top = testable[0]
        sid = top["sid"]
        print("-" * 84)
        print("PER-SYMBOL BREAKDOWN for %s (the only well-powered strategy)" % sid)
        print("-" * 84)
        per_sym = defaultdict(lambda: defaultdict(list))
        for t, sym in by_strat[sid]:
            v = _fwd(book, sym, t, HORIZON)
            if v is not None:
                per_sym[sym][int(t // WINDOW)].append(v)
        rows_out = []
        for sym, by_w in per_sym.items():
            diffs = [statistics.median(vals) - tape[w]
                     for w, vals in by_w.items() if w in tape]
            if len(diffs) < MIN_WINDOWS:
                continue
            d, lo, hi = _ci(diffs)
            rows_out.append((d, lo, hi, sym, len(diffs),
                             sum(1 for x in diffs if x > 0)))
        rows_out.sort(reverse=True)
        print("%-18s %8s %10s %21s %8s" % ("symbol", "windows", "diff(bp)", "95% CI", "win%"))
        for d, lo, hi, sym, n, w in rows_out:
            print("%-18s %8d %+10.2f [%+8.2f,%+8.2f] %7.1f%%"
                  % (sym[:18], n, d, lo, hi, 100.0 * w / n))
        pos = [r for r in rows_out if r[1] > 0.0]
        clr = [r for r in rows_out if r[1] > COST_FLOOR_BP]
        print("  symbols with >= %d windows: %d | CI excludes zero: %d | CI clears %.2fbp: %d"
              % (MIN_WINDOWS, len(rows_out), len(pos), COST_FLOOR_BP, len(clr)))
        if not rows_out:
            print("  NO symbol reaches %d paired windows. The pooled edge is spread thin"
                  % MIN_WINDOWS)
            print("  across many names, so there is no cheap subset to retreat to.")
        elif not pos:
            print("  NO single symbol carries a significant edge. The pooled +%.2fbp is an"
                  % top["diff"])
            print("  aggregate of noise, not a concentration in a tradeable name.")
        else:
            print("  Candidates for a symbol-restricted lane are listed above; check each")
            print("  against its OWN round-trip cost, not the pooled floor.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
