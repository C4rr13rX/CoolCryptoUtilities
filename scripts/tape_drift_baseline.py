#!/usr/bin/env python
"""Is the TAPE itself negative for a long-only bot? -- Echo, pass 486.

WHY THIS AND NOT THE PAIRED TEST AGAIN. Pass 485 compared the accepted arm
against the REFUSED arm and found no difference (paired diff +2.58bp, CI
contains zero) -- but BOTH ARMS WERE NEGATIVE (-2.08bp and -6.88bp). Two
selected pools were compared against each other and nothing was compared
against the population they were drawn from. That leaves the decisive question
open, and it is the one that picks the route:

  HYPOTHESIS  H1: the tape drifts DOWN over 15-45 min on this universe.
                  Then a long-only bot loses by construction, no gate can
                  save it, and the accepted arm at -2.08bp is actually
                  BEATING its own tape. Route: change side or change universe.

  COMPLEMENT  H0: the tape is FLAT (unconditional forward return ~= 0).
                  Then selection is actively HARMFUL -- out of a zero-mean
                  population our entries pick -2.08bp -- and the defect is in
                  candidate generation, not in the gate that filters it.

KILL TEST, pre-registered. Sample EVERY priced symbol on a fixed 10-minute
grid over the same 7 days, compute the same forward return with the same
staleness rule, and put a CI on it.
  - If the unconditional mean is negative and its CI EXCLUDES zero -> H1.
  - If the unconditional CI CONTAINS zero while the accepted arm's does not
    -> H0, and the entry logic is the suspect.
  - If BOTH contain zero the measurement is underpowered and says nothing;
    that outcome is reported as a null, not as H1.

And the paired form, which is the number that actually matters: within each
10-minute window, accepted rows minus the MEDIAN of every priced symbol in
that same window. That is the accepted arm against its own tape with drift
cancelled, and it has never been computed here.

OUTLIERS. This store carries corrupt price regimes (COMP-USDC and 7 others;
one window implies a 7e13x move). Every cross-symbol aggregate below is a
MEDIAN, and the mean is printed beside it so a divergence is visible rather
than silent.

READ-ONLY. Opens the store with mode=ro. Changes nothing.

Run:  python -X utf8 scripts/tape_drift_baseline.py
"""

from __future__ import annotations

import bisect
import math
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

LOOKBACK_DAYS = 7.0
HORIZONS = (900.0, 2700.0)  # 15 min (tradeable book positive inside it), 45 min (.env:169)
WINDOW = 600.0  # same pairing bucket as matched_baseline_test.py
MAX_STALE = 600.0
COST_FLOOR_BP = 27.77  # live-receipt round-trip floor, p482


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
    out = {}
    for sym, rows in per.items():
        rows.sort(key=lambda x: x[0])
        out[sym] = ([t for t, _ in rows], [p for _, p in rows])
    return out


def _price_at(book, sym, t):
    entry = book.get(sym)
    if not entry:
        return None
    ts, px = entry
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
    if not math.isfinite(r):
        return None
    return r * 10000.0


def _ci(xs):
    n = len(xs)
    if n < 2:
        return (statistics.fmean(xs) if xs else float("nan")), float("nan"), float("nan")
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    half = 1.96 * sd / math.sqrt(n)
    return m, m - half, m + half


def _load_accepted(conn, since):
    return [
        (r["ts"], r["symbol"])
        for r in conn.execute(
            "SELECT ts, symbol FROM trading_ops "
            "WHERE ts > ? AND action='enter' AND status='ghost-entry'",
            (since,),
        )
    ]


def main() -> int:
    now = time.time()
    since = now - LOOKBACK_DAYS * 86400.0
    conn = _connect()

    print("=" * 78)
    print("UNCONDITIONAL TAPE DRIFT vs THE ACCEPTED ARM -- Echo pass 486")
    print("lookback %.0fd | grid %.0fs | staleness cap %.0fs | cost floor %.2fbp"
          % (LOOKBACK_DAYS, WINDOW, MAX_STALE, COST_FLOOR_BP))
    print("=" * 78)

    book = _load_prices(conn, since - max(HORIZONS) - MAX_STALE)
    acc = _load_accepted(conn, since)
    print("priced symbols %d | accepted(ghost-entry) rows %d" % (len(book), len(acc)))

    # ---- the fixed grid: every symbol, every 10 minutes ---------------------
    w0 = int(since // WINDOW)
    w1 = int((now - max(HORIZONS)) // WINDOW)
    grid_windows = list(range(w0, w1 + 1))
    print("grid windows in range: %d" % len(grid_windows))

    for h in HORIZONS:
        print("-" * 78)
        print("HORIZON %.0f min" % (h / 60.0))

        # unconditional: one observation per (window), the MEDIAN across symbols
        per_window_med, per_window_mean, n_obs = [], [], 0
        window_med = {}
        for w in grid_windows:
            t = w * WINDOW
            vals = []
            for sym in book:
                v = _fwd(book, sym, t, h)
                if v is not None:
                    vals.append(v)
            if len(vals) < 3:
                continue
            med = statistics.median(vals)
            window_med[w] = med
            per_window_med.append(med)
            per_window_mean.append(statistics.fmean(vals))
            n_obs += len(vals)

        if len(per_window_med) < 2:
            print("  INSUFFICIENT GRID WINDOWS (%d). No verdict." % len(per_window_med))
            continue

        m, lo, hi = _ci(per_window_med)
        mm, mlo, mhi = _ci(per_window_mean)
        print("  UNCONDITIONAL TAPE (median-of-symbols per window)")
        print("    windows %d | symbol-observations %d" % (len(per_window_med), n_obs))
        print("    mean of window medians %+8.2f bp   95%% CI [%+.2f, %+.2f]" % (m, lo, hi))
        print("    mean of window means   %+8.2f bp   95%% CI [%+.2f, %+.2f]  (outlier-exposed)"
              % (mm, mlo, mhi))
        up = sum(1 for x in per_window_med if x > 0)
        print("    windows with a POSITIVE median: %d of %d (%.1f%%)"
              % (up, len(per_window_med), 100.0 * up / len(per_window_med)))
        if lo <= 0.0 <= hi:
            print("    -> tape drift NOT distinguishable from zero at this horizon")
        elif m < 0:
            print("    -> TAPE DRIFTS DOWN. A long-only bot loses before any gate runs.")
        else:
            print("    -> tape drifts UP.")

        # ---- accepted arm vs its own tape, paired within window -------------
        acc_by = defaultdict(list)
        for t, sym in acc:
            v = _fwd(book, sym, t, h)
            if v is not None:
                acc_by[int(t // WINDOW)].append(v)

        diffs, a_side, t_side = [], [], []
        for w in sorted(acc_by):
            if w not in window_med:
                continue
            am = statistics.median(acc_by[w])
            diffs.append(am - window_med[w])
            a_side.append(am)
            t_side.append(window_med[w])

        print("  ACCEPTED ARM vs THE TAPE, paired inside the same 10-min window")
        if len(diffs) < 2:
            print("    INSUFFICIENT PAIRED WINDOWS (%d). No verdict." % len(diffs))
            continue
        d, dlo, dhi = _ci(diffs)
        print("    paired windows %d" % len(diffs))
        print("    accepted median fwd %+8.2f bp" % statistics.fmean(a_side))
        print("    tape     median fwd %+8.2f bp" % statistics.fmean(t_side))
        print("    PAIRED DIFF (acc - tape) %+8.2f bp   95%% CI [%+.2f, %+.2f]"
              % (d, dlo, dhi))
        wins = sum(1 for x in diffs if x > 0)
        print("    windows where accepted beat the tape: %d of %d (%.1f%%)"
              % (wins, len(diffs), 100.0 * wins / len(diffs)))
        mde = 1.96 * (statistics.stdev(diffs) / math.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        print("    minimum detectable effect %.2f bp vs a %.2f bp cost floor"
              % (mde, COST_FLOOR_BP))
        if dlo <= 0.0 <= dhi:
            print("    -> NULL: entry selection adds nothing over picking a symbol at random")
            if mde < COST_FLOOR_BP:
                print("       and the test CAN see the cost floor, so the null is informative")
            else:
                print("       but the test CANNOT see the cost floor -- underpowered, not a result")
        elif d > 0:
            print("    -> entry selection BEATS its own tape by %+.2f bp" % d)
            print("       %s the %.2f bp cost floor"
                  % ("CLEARS" if dlo > COST_FLOOR_BP else "does NOT clear", COST_FLOOR_BP))
        else:
            print("    -> entry selection is WORSE than picking at random. Generation is the bug.")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
