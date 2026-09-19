#!/usr/bin/env python
"""Does anything this system ENTERS go up, at ANY horizon, by enough to pay?

Pass 485 established that the entry gate's accept/refuse split carries no
forward-return information: a paired diff of +2.58 bp with a 95% CI of
[-8.79, +13.94] and a minimum detectable effect of 11.37 bp.  That is a
statement about the GATE.  It is not a statement about whether a directional
edge exists at all.

This asks the larger question directly and one-sidedly: take every ghost entry
the system actually made, and ask whether its forward return is positive by
more than the round-trip cost floor, at each of eight horizons.  No pairing
here -- the comparison is against ZERO and against the cost floor, not against
a refused arm, so the question is "does the thing we bought go up enough to pay
for buying it".

The complement is stated so this cannot be read as confirmation: if the mean
forward return is positive and its 95% CI lies entirely above the cost floor at
ANY horizon, a directional edge exists there and the hold time is the lever.
If every horizon's CI contains zero, no horizon is profitable and the defect is
in signal generation, not in hold time and not in any gate.

READ-ONLY.  Run:  python -X utf8 scripts/horizon_skill_sweep.py
"""

from __future__ import annotations

import math
import statistics
import sys

import matched_baseline_test as M

# Live-receipt round-trip cost floor, re-derived pass 482 from actual fills.
COST_FLOOR_BP = 27.77

HORIZONS = (120.0, 300.0, 900.0, 1800.0, 2700.0, 5400.0, 10800.0, 21600.0)


def main() -> int:
    import time

    now = time.time()
    since = now - M.LOOKBACK_DAYS * 86400.0
    conn = M._connect()
    book = M._load_prices(conn, since - max(HORIZONS) - M.MAX_STALE)
    acc, ref = M._load_events(conn, since)

    print("=" * 78)
    print("HORIZON SKILL SWEEP -- does anything we ENTER go up enough to pay?")
    print("lookback %.0f d | %d ghost entries | cost floor %.2f bp"
          % (M.LOOKBACK_DAYS, len(acc), COST_FLOOR_BP))
    print("=" * 78)
    print("%9s %6s %10s %22s %8s  %s"
          % ("horizon", "n", "mean bp", "95% CI", "up-rate", "verdict"))

    any_pays = False
    for h in HORIZONS:
        vals = [v for v in (M._fwd(book, s, t, h) for t, s, _ in acc) if v is not None]
        if len(vals) < 3:
            print("%8.0fm %6d   -- too few priced entries" % (h / 60.0, len(vals)))
            continue
        m = statistics.fmean(vals)
        sd = statistics.stdev(vals)
        half = 1.96 * sd / math.sqrt(len(vals))
        lo, hi = m - half, m + half
        up = 100.0 * sum(1 for v in vals if v > 0) / len(vals)
        if lo > COST_FLOOR_BP:
            verdict = "PAYS -- edge exists here"
            any_pays = True
        elif lo > 0.0:
            verdict = "positive but under the floor"
        elif hi < 0.0:
            verdict = "NEGATIVE"
        else:
            verdict = "zero-in-CI"
        print("%8.0fm %6d %+10.2f  [%+8.2f,%+8.2f] %7.1f%%  %s"
              % (h / 60.0, len(vals), m, lo, hi, up, verdict))

    print("-" * 78)
    # The same sweep over the REFUSED pool, as a scale check: if the refused
    # pool looks identical, the system is not selecting, it is sampling.
    print("refused pool, same horizons (scale check -- is it the same tape?)")
    for h in HORIZONS:
        vals = [v for v in (M._fwd(book, s, t, h) for t, s, _ in ref) if v is not None]
        if len(vals) < 3:
            continue
        m = statistics.fmean(vals)
        sd = statistics.stdev(vals)
        half = 1.96 * sd / math.sqrt(len(vals))
        up = 100.0 * sum(1 for v in vals if v > 0) / len(vals)
        print("%8.0fm %6d %+10.2f  [%+8.2f,%+8.2f] %7.1f%%"
              % (h / 60.0, len(vals), m, m - half, m + half, up))

    print("=" * 78)
    if any_pays:
        print("VERDICT: at least one horizon clears the cost floor on its lower")
        print("         bound. A directional edge exists and hold time IS a lever.")
    else:
        print("VERDICT: NO horizon clears the %.2f bp cost floor on its lower" % COST_FLOOR_BP)
        print("         bound. Hold time is not a lever and no gate is the defect.")
        print("         The deficiency is in signal generation.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
