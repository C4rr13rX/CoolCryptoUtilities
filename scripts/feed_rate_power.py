"""How small a feed-rate change can a 10-minute window actually see?

Item [cee842c9] asks, as its last criterion, for the feed rate "measured
BEFORE and AFTER over a full 10-minute window each" once the model tasks are
moved or bounded.  That criterion is only meaningful if one 10-minute window
resolves the effect being tested.  Measured in the same item (commit bcabb78,
scripts/seq_queue_budget.py), the model category holds 93.36 s per 10 minutes
of the sequential queue but explains roughly 1-2 of the 39 missing ticks per
10 minutes -- so the question is whether a 1-2 tick change is visible against
the feed's own run-to-run spread.

This reads market_stream exactly the way loop_status.py does -- a COUNT of
rows inside a 600-second window -- but over every disjoint 10-minute bin in a
lookback, and reports the spread of those counts.  Nothing is fitted and
nothing is modelled: the numbers are the empirical distribution of the very
statistic the criterion names.

The two-sample detection floor reported at the end is the classic normal
approximation for a difference of means with n windows on each side:

    minimum detectable difference = z * sd * sqrt(2 / n)

at z = 1.96 for 95% confidence.  With n = 1 window on each side that is
2.77 standard deviations, which is the number that decides whether a single
before window and a single after window can close this item.

Read-only.  Opens the trading store in SQLite immutable mode so a running
production process is never blocked or written to.

    python -X utf8 scripts/feed_rate_power.py --hours 24
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
import time
from pathlib import Path

DEFAULT_DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")
BIN_SEC = 600.0  # loop_status.py's window, exactly


def _connect(path: str) -> sqlite3.Connection:
    """Open read-only.  immutable=1 keeps us off production's write lock."""
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def bin_counts(conn: sqlite3.Connection, *, hours: float, now: float | None = None
               ) -> tuple[list[int], float, float]:
    """Rows per disjoint 10-minute bin over the lookback, oldest first.

    Bins are anchored to `now` and counted backwards so the most recent bin
    ends at `now`, which is how loop_status.py's single window is framed.  The
    partial bin at the far end is dropped rather than counted short -- a
    half-length bin holds half the ticks and would widen the spread for a
    reason that has nothing to do with the feed.
    """
    now = float(now if now is not None else time.time())
    n_bins = int((hours * 3600.0) // BIN_SEC)
    start = now - n_bins * BIN_SEC
    rows = conn.execute(
        "SELECT CAST((ts - ?) / ? AS INTEGER) AS b, COUNT(*) "
        "FROM market_stream WHERE ts >= ? AND ts < ? GROUP BY b",
        (start, BIN_SEC, start, now),
    ).fetchall()
    by_bin = {int(b): int(c) for b, c in rows if b is not None}
    return [by_bin.get(i, 0) for i in range(n_bins)], start, now


def summarise(counts: list[int]) -> dict:
    """Spread of the tick counts, plus what a given number of windows can see."""
    n = len(counts)
    if n < 2:
        raise SystemExit("need at least two 10-minute bins; widen --hours")
    mean = statistics.fmean(counts)
    sd = statistics.stdev(counts)
    ordered = sorted(counts)

    def pct(p: float) -> float:
        if not ordered:
            return 0.0
        k = (len(ordered) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    out = {
        "bins": n,
        "mean_ticks_per_10min": round(mean, 2),
        "sd_ticks_per_10min": round(sd, 2),
        "min": ordered[0],
        "p05": round(pct(0.05), 1),
        "median": round(pct(0.50), 1),
        "p95": round(pct(0.95), 1),
        "max": ordered[-1],
        "coefficient_of_variation": round(sd / mean, 3) if mean else None,
        "zero_bins": sum(1 for c in counts if c == 0),
    }
    # z * sd * sqrt(2/n_per_side): the smallest difference in mean ticks that
    # n windows a side could call at 95% confidence.
    out["min_detectable_diff_ticks"] = {
        str(k): round(1.96 * sd * math.sqrt(2.0 / k), 1) for k in (1, 3, 6, 12, 36)
    }
    return out


def windows_needed(sd: float, effect: float) -> int | None:
    """How many 10-minute windows a side to resolve `effect` ticks at 95%."""
    if effect <= 0 or sd <= 0:
        return None
    n = 2.0 * (1.96 * sd / effect) ** 2
    return int(math.ceil(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--effect", type=float, default=2.0,
                    help="ticks/10min the change under test is expected to move "
                         "(default 2, the upper end of the 1-2 measured in bcabb78)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    conn = _connect(args.db)
    try:
        counts, start, now = bin_counts(conn, hours=args.hours)
    finally:
        conn.close()

    summary = summarise(counts)
    need = windows_needed(summary["sd_ticks_per_10min"], args.effect)
    summary["effect_under_test_ticks"] = args.effect
    summary["windows_per_side_needed"] = need
    summary["hours_per_side_needed"] = round(need * BIN_SEC / 3600.0, 1) if need else None
    summary["window_start_epoch"] = round(start, 1)
    summary["window_end_epoch"] = round(now, 1)

    print(f"market_stream, {args.hours:g}h to now, {summary['bins']} disjoint "
          f"{int(BIN_SEC)}s bins")
    print(f"  ticks/10min: mean {summary['mean_ticks_per_10min']} "
          f"sd {summary['sd_ticks_per_10min']} "
          f"(cv {summary['coefficient_of_variation']})")
    print(f"  min {summary['min']}  p05 {summary['p05']}  median {summary['median']}"
          f"  p95 {summary['p95']}  max {summary['max']}  "
          f"zero bins {summary['zero_bins']}")
    print("  smallest difference in mean ticks/10min that N windows a side can call "
          "at 95%:")
    for k, v in summary["min_detectable_diff_ticks"].items():
        print(f"    N={k:>2}: {v}")
    print(f"  effect under test: {args.effect} ticks/10min -> needs {need} windows "
          f"a side ({summary['hours_per_side_needed']}h a side)")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
