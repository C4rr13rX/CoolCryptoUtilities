#!/usr/bin/env python
"""The matched-baseline forward-return test -- item [c9e248aa]'s decisive criterion.

THE QUESTION. The entry gate refuses far more than it accepts. Is it refusing
things that would have LOST, or is it refusing at random?

  If refused candidates do NO WORSE than accepted ones, the gate carries no
  information and IS the bug.
  If refused candidates are genuinely worse, the gate is correct and the defect
  is upstream in candidate generation.

WHY THE PAIRING MATTERS, and why an unpaired forward-return study proves
nothing here: this tape drifts. Accepted and refused rows are not drawn
uniformly over time -- entries cluster in bursts -- so comparing the two pools
directly measures WHEN each pool was sampled, not what the gate knew. Every
comparison below is therefore made WITHIN a time window that contains at least
one accepted row AND one refused row, and only the within-window difference is
aggregated. Tape drift is common to both arms of each pair and cancels.

  accepted : trading_ops action='enter'  status='ghost-entry'
  refused  : trading_ops action='hold'   (every status, and split out per status)
  forward return: market_stream price at the nearest tick at-or-before t,
                  against the nearest tick at-or-before t+h.

Horizons are 15 min (the tradeable book is positive inside 15 min) and 45 min
(MAX_HOLD_FORCE_SECONDS=2700, the hold ceiling this bot actually enforces).

READ-ONLY. Opens the store with mode=ro. Changes nothing.

Run:  python -X utf8 scripts/matched_baseline_test.py
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
HORIZONS = (900.0, 2700.0)  # 15 min, 45 min
WINDOW = 600.0  # pairing bucket: 10 minutes
MAX_STALE = 600.0  # a price tick older than this is not a price for our purpose


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect("file:%s?mode=ro" % DB, uri=True, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def _load_prices(conn, since):
    """symbol -> (sorted ts list, parallel price list)."""
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
    """Nearest tick at or before t, provided it is not staler than MAX_STALE."""
    entry = book.get(sym)
    if not entry:
        return None
    ts, px = entry
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return None
    if t - ts[i] > MAX_STALE:
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
    return r * 10000.0  # basis points


def _load_events(conn, since):
    acc, ref = [], []
    for r in conn.execute(
        "SELECT ts, symbol, status FROM trading_ops "
        "WHERE ts > ? AND action='enter' AND status='ghost-entry'",
        (since,),
    ):
        acc.append((r["ts"], r["symbol"], "ghost-entry"))
    for r in conn.execute(
        "SELECT ts, symbol, status FROM trading_ops WHERE ts > ? AND action='hold'",
        (since,),
    ):
        ref.append((r["ts"], r["symbol"], r["status"] or "?"))
    return acc, ref


def _paired(book, acc, ref, h, ref_filter=None):
    """Within-window paired difference, accepted minus refused, in bp."""
    a_by, r_by = defaultdict(list), defaultdict(list)
    for t, sym, _ in acc:
        v = _fwd(book, sym, t, h)
        if v is not None:
            a_by[int(t // WINDOW)].append(v)
    for t, sym, st in ref:
        if ref_filter and st != ref_filter:
            continue
        v = _fwd(book, sym, t, h)
        if v is not None:
            r_by[int(t // WINDOW)].append(v)

    diffs, a_all, r_all, n_a, n_r = [], [], [], 0, 0
    for w in sorted(set(a_by) & set(r_by)):
        am = statistics.fmean(a_by[w])
        rm = statistics.fmean(r_by[w])
        diffs.append(am - rm)
        a_all.append(am)
        r_all.append(rm)
        n_a += len(a_by[w])
        n_r += len(r_by[w])
    return diffs, a_all, r_all, n_a, n_r


def _ci(xs):
    """Mean and a normal-approx 95% CI on the paired difference."""
    n = len(xs)
    if n < 2:
        return (statistics.fmean(xs) if xs else float("nan")), float("nan"), float("nan")
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs)
    half = 1.96 * sd / math.sqrt(n)
    return m, m - half, m + half


def main() -> int:
    global WINDOW, MAX_STALE  # the sensitivity sweep at the end rebinds these
    now = time.time()
    since = now - LOOKBACK_DAYS * 86400.0
    conn = _connect()

    print("=" * 78)
    print("MATCHED-BASELINE FORWARD-RETURN TEST -- item [c9e248aa]")
    print("lookback %.0f d | pairing window %.0f s | price staleness cap %.0f s"
          % (LOOKBACK_DAYS, WINDOW, MAX_STALE))
    print("=" * 78)

    book = _load_prices(conn, since - max(HORIZONS) - MAX_STALE)
    acc, ref = _load_events(conn, since)
    print("events: accepted(ghost-entry) %d | refused(hold) %d | priced symbols %d"
          % (len(acc), len(ref), len(book)))

    by_status = defaultdict(int)
    for _, _, st in ref:
        by_status[st] += 1

    for h in HORIZONS:
        print("-" * 78)
        print("HORIZON %.0f min -- ALL REFUSALS POOLED" % (h / 60.0))
        diffs, a_all, r_all, n_a, n_r = _paired(book, acc, ref, h)
        if len(diffs) < 2:
            print("  INSUFFICIENT PAIRED WINDOWS (%d). No verdict at this horizon."
                  % len(diffs))
            continue
        m, lo, hi = _ci(diffs)
        print("  paired windows      %d   (accepted rows %d, refused rows %d)"
              % (len(diffs), n_a, n_r))
        print("  accepted mean fwd   %+8.2f bp" % statistics.fmean(a_all))
        print("  refused  mean fwd   %+8.2f bp" % statistics.fmean(r_all))
        print("  PAIRED DIFF (acc-ref) %+8.2f bp   95%% CI [%+.2f, %+.2f]"
              % (m, lo, hi))
        wins = sum(1 for d in diffs if d > 0)
        print("  windows where accepted beat refused: %d of %d (%.1f%%)"
              % (wins, len(diffs), 100.0 * wins / len(diffs)))
        if lo <= 0.0 <= hi:
            print("  VERDICT: CI CONTAINS ZERO. The gate's accept/refuse split carries")
            print("           NO measurable forward-return information at this horizon.")
        elif m > 0:
            print("  VERDICT: accepted genuinely beat refused. The gate is informative.")
        else:
            print("  VERDICT: accepted are WORSE than refused. The gate is INVERTED.")

        print("  per-refusal-reason, same pairing:")
        rows = []
        for st, cnt in sorted(by_status.items(), key=lambda kv: -kv[1]):
            d2, a2, r2, na2, nr2 = _paired(book, acc, ref, h, ref_filter=st)
            if len(d2) < 2:
                rows.append((st, cnt, len(d2), None, None, None))
                continue
            m2, lo2, hi2 = _ci(d2)
            rows.append((st, cnt, len(d2), m2, lo2, hi2))
        for st, cnt, nw, m2, lo2, hi2 in rows:
            if m2 is None:
                print("    %-30s n=%5d  windows=%3d   -- too few windows" % (st, cnt, nw))
            else:
                flag = "ZERO-IN-CI" if lo2 <= 0.0 <= hi2 else ("INFORMATIVE" if m2 > 0 else "INVERTED")
                print("    %-30s n=%5d  windows=%3d   diff %+8.2f bp [%+.1f,%+.1f]  %s"
                      % (st, cnt, nw, m2, lo2, hi2, flag))

    # ------------------------------------------------------------------
    # KILL TESTS ON THIS SCRIPT'S OWN HEADLINE.
    # A null is only worth stating if (a) it survives the arbitrary design
    # constants and (b) the test could have SEEN the effect that matters.
    # ------------------------------------------------------------------
    base_w, base_s = WINDOW, MAX_STALE
    h = HORIZONS[0]

    print("-" * 78)
    print("SENSITIVITY -- does the %0.f min verdict survive the design constants?"
          % (h / 60.0))
    print("  %-10s %-10s %8s %10s %20s" % ("window", "stale", "windows", "diff bp", "95% CI"))
    for w in (300.0, 600.0, 900.0, 1800.0):
        for s in (300.0, 600.0, 1200.0):
            WINDOW, MAX_STALE = w, s
            d, _, _, _, _ = _paired(book, acc, ref, h)
            if len(d) < 2:
                print("  %-10.0f %-10.0f %8d   -- too few" % (w, s, len(d)))
                continue
            m, lo, hi = _ci(d)
            tag = "zero-in-CI" if lo <= 0.0 <= hi else "SIGNIFICANT"
            print("  %-10.0f %-10.0f %8d %+10.2f  [%+7.2f,%+7.2f] %s"
                  % (w, s, len(d), m, lo, hi, tag))
    WINDOW, MAX_STALE = base_w, base_s

    print("-" * 78)
    print("POWER -- inject a KNOWN lift into the accepted arm and see if this")
    print("test detects it. If it cannot detect the cost floor, the null above")
    print("means nothing and this script says so.")
    for lift in (10.0, 20.0, 27.77, 50.0):
        d, _, _, _, _ = _paired(book, acc, ref, h)
        d = [x + lift for x in d]
        m, lo, hi = _ci(d)
        seen = "DETECTED" if lo > 0.0 else "missed"
        print("  injected %+7.2f bp -> diff %+8.2f  CI [%+7.2f,%+7.2f]  %s"
              % (lift, m, lo, hi, seen))
    d0, _, _, _, _ = _paired(book, acc, ref, h)
    m0, lo0, hi0 = _ci(d0)
    mde = hi0 - m0
    print("  minimum detectable effect at 95%%: %+.2f bp" % mde)
    print("  live-receipt round-trip cost floor (p482):   27.77 bp")
    if mde < 27.77:
        print("  => the test CAN resolve an effect the size of the cost floor,")
        print("     and it did not find one. The null is informative.")
    else:
        print("  => the test CANNOT resolve the cost floor. The null is NOT")
        print("     informative and must not be reported as one.")

    conn.close()
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
