#!/usr/bin/env python
"""Is expected_return calibrated to anything? -- item [296b7dc5], Echo pass 486.

THE FIELD. trading_ops rows with action='enter' carry a details JSON holding
the strategy's own forecast: expected_return, score, confidence, target_price.
2753 of 2813 enter rows in 7 days carry a numeric expected_return. It is a
FRACTION, not bp: min 0.015, median 0.050, max 0.150 -- so the median entry is
admitted on a forecast of +500 bp over the hold.

WHY THIS IS THE DECIDING MEASUREMENT. Pass 486 measured what atf_static
actually delivers against the median symbol on its own tape: +2.69 bp, CI
[+0.16, +5.21], 380 paired windows. The gate admits on a forecast. If the
forecast is two orders of magnitude above the outturn, then no threshold on
this field separates anything, because the field is not measuring the future.

  H1: expected_return carries rank information about realised forward return.
      Then it is a SCALE bug -- rescale it, admit only the top decile, and
      there is a concrete route to an entry rule that clears the 27.77 bp toll.
  H0: no rank information, permutation p > 0.05.
      Then every gate threshold ever tuned on this field was tuned on noise,
      gate work is CLOSED rather than merely unpromising, and the remaining
      route is a different signal -- not a different threshold on this one.

METHOD NOTES, and they are not optional on this store.
  - SPEARMAN, not Pearson. Eight symbols carry corrupt price regimes; pass 486
    measured a mean-of-window-means of +167333 bp against a median of +0.94 bp
    on identical rows. Any moment-based statistic here is void.
  - Permutation p over the rank statistic, 2000 shuffles, so the null needs no
    distributional assumption.
  - The MDE is reported. A test that cannot resolve an effect smaller than the
    27.77 bp toll is underpowered, and is reported as such rather than as a
    null. Pass 485 nearly shipped a false positive on exactly this distinction.

Horizons: 15 min, and 45 min because MAX_HOLD_FORCE_SECONDS=2700 is the hold
this bot actually enforces -- 45 is the fairer test of a hold-horizon forecast.

READ-ONLY. Opens the store with mode=ro. Changes nothing.

Run:  python -X utf8 scripts/expected_return_calibration.py
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")

LOOKBACK_DAYS = 7.0
HORIZONS = (900.0, 2700.0)
MAX_STALE = 600.0
COST_FLOOR_BP = 27.77
SHUFFLES = 2000
SEED = 486  # fixed so a re-run reproduces exactly


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


def _ranks(xs):
    """Average ranks, ties shared -- the store is full of repeated forecasts."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _pearson(a, b):
    n = len(a)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def _spearman(a, b):
    return _pearson(_ranks(a), _ranks(b))


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

    print("=" * 80)
    print("expected_return CALIBRATION -- item [296b7dc5], Echo pass 486")
    print("lookback %.0fd | staleness %.0fs | %d shuffles | seed %d | floor %.2fbp"
          % (LOOKBACK_DAYS, MAX_STALE, SHUFFLES, SEED, COST_FLOOR_BP))
    print("=" * 80)

    book = _load_prices(conn, since - max(HORIZONS) - MAX_STALE)

    rows, no_er = [], 0
    for r in conn.execute(
        "SELECT ts, symbol, details FROM trading_ops WHERE ts > ? AND action='enter'",
        (since,),
    ):
        try:
            j = json.loads(r["details"] or "{}")
        except Exception:
            continue
        v = j.get("expected_return")
        if not isinstance(v, (int, float)):
            no_er += 1
            continue
        sym = r["symbol"] or j.get("symbol")
        if not sym:
            continue
        rows.append((r["ts"], sym, float(v) * 10000.0, j.get("strategy_id") or "?"))

    print("enter rows with numeric expected_return: %d  (without: %d)" % (len(rows), no_er))
    ers = [e for _, _, e, _ in rows]
    print("UNITS: field is a FRACTION. In bp -> min %.1f | median %.1f | max %.1f"
          % (min(ers), statistics.median(ers), max(ers)))

    for h in HORIZONS:
        print("-" * 80)
        print("HORIZON %.0f min" % (h / 60.0))
        pairs = []
        for t, sym, e, sid in rows:
            v = _fwd(book, sym, t, h)
            if v is not None:
                pairs.append((e, v, sid))
        if len(pairs) < 30:
            print("  INSUFFICIENT PRICED ROWS (%d). No verdict." % len(pairs))
            continue
        exp = [p[0] for p in pairs]
        act = [p[1] for p in pairs]
        print("  priced rows %d" % len(pairs))
        print("  mean  expected %+10.2f bp   |  mean  realised %+10.2f bp" %
              (statistics.fmean(exp), statistics.fmean(act)))
        print("  median expected %+10.2f bp   |  median realised %+10.2f bp" %
              (statistics.median(exp), statistics.median(act)))
        me, ma = statistics.fmean(exp), statistics.fmean(act)
        print("  CALIBRATION RATIO realised/expected = %.5f   (forecast is %.1fx the outturn)"
              % (ma / me if me else float("nan"), me / ma if ma else float("inf")))

        rho = _spearman(exp, act)
        rnd = random.Random(SEED)
        shuffled = list(act)
        hits = 0
        for _ in range(SHUFFLES):
            rnd.shuffle(shuffled)
            if abs(_spearman(exp, shuffled)) >= abs(rho):
                hits += 1
        p = (hits + 1) / (SHUFFLES + 1)
        print("  SPEARMAN rho %+.4f   permutation p = %.4f  (%d/%d shuffles >= |rho|)"
              % (rho, p, hits, SHUFFLES))

        # decile table on expected_return
        idx = sorted(range(len(pairs)), key=lambda i: exp[i])
        k = max(1, len(idx) // 10)
        print("  decile of expected_return (low to high):")
        print("    %-6s %6s %12s %12s %20s" % ("decile", "n", "exp mean", "act MEDIAN", "act mean 95% CI"))
        top_ci = None
        for d in range(10):
            chunk = idx[d * k:(d + 1) * k] if d < 9 else idx[9 * k:]
            if not chunk:
                continue
            ce = [exp[i] for i in chunk]
            ca = [act[i] for i in chunk]
            m, lo, hi = _ci(ca)
            print("    %-6d %6d %+12.1f %+12.2f  [%+8.2f,%+8.2f]"
                  % (d + 1, len(chunk), statistics.fmean(ce), statistics.median(ca), lo, hi))
            if d == 9:
                top_ci = (m, lo, hi, len(chunk))

        mde = (1.96 * statistics.stdev(act) / math.sqrt(len(act))) if len(act) > 1 else float("nan")
        print("  MDE on the pooled realised mean: %.2f bp vs a %.2f bp toll -- %s"
              % (mde, COST_FLOOR_BP,
                 "test CAN see the toll" if mde < COST_FLOOR_BP else "UNDERPOWERED, not a null"))

        print("  VERDICT:")
        if p > 0.05:
            print("    H0. expected_return carries NO rank information about realised")
            print("        forward return (p=%.4f). Every gate threshold tuned on this" % p)
            print("        field was tuned on noise. Gate work on it is CLOSED.")
        else:
            print("    H1. expected_return carries rank information (rho %+.4f, p=%.4f)."
                  % (rho, p))
            if top_ci:
                m, lo, hi, n = top_ci
                print("        TOP DECILE realised mean %+.2f bp [%+.2f,%+.2f] over %d rows"
                      % (m, lo, hi, n))
                if lo > COST_FLOOR_BP:
                    print("        -> TOP DECILE CLEARS THE %.2fbp TOLL. This is a route."
                          % COST_FLOOR_BP)
                else:
                    print("        -> top decile does NOT clear the %.2fbp toll. Rank info"
                          % COST_FLOOR_BP)
                    print("           exists but is not worth enough to trade on.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
