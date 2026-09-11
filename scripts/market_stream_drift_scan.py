"""Find an UP window in ``market_stream`` -- the missing control for the horizon table.

WHY THIS EXISTS. Item ``15cc71d4`` measured what a PERFECT direction call would
net at each horizon and found the minutes horizons negative: 5min -0.2152 and
10min -0.1479 in the recent 24h, and -0.2907 / -0.2484 in a disjoint 24h window
72h back. Two windows agreeing is worth much more than one -- but BOTH of those
windows read FLAT-to-DOWN on per-symbol drift (-0.184% and -0.199% median), so
the finding is currently "the minutes horizon does not pay in a flat tape".
That is a weaker claim than the one the loop needs, and the standing
instructions say it in as many words: never report a single direction, because
a long-only rule flatters itself in an up window and that error has already
produced a fake 78% and a fake +0.9067% in this repo.

So before re-running the horizon table anywhere, somebody has to answer a
question no existing script answers: DOES an up window exist in the 373 hours of
tape we hold, and where? This scans for one and prints the ``--end-hours-ago``
you would hand to ``head_vs_realised_census.py`` to land on it.

HOW DRIFT IS MEASURED, and why not first-tick-to-last-tick. A symbol's drift
here is the median price over the LAST tenth of the window against the median
over the FIRST tenth, not a single endpoint pair. This repo's feed has carried
two price regimes under one ticker (``feed-two-regime-census``) and a single
contaminated endpoint tick would manufacture an arbitrarily large drift for that
symbol; a median over a block of ticks cannot be moved by one row. Symbols whose
in-window max/min price ratio exceeds ``--max-price-ratio`` are dropped outright
as denomination contamination and THE DROPPED COUNT IS PRINTED -- a filter that
silently eats rows is how a fake edge gets published here.

The window statistic is the MEDIAN across symbols, not the mean: the mean is set
by whichever microcap doubled, and the census it feeds pools every symbol.

Units: drift is a FRACTION internally and printed as a PERCENT, and the header
says so. Exit code is 1 when no window clears ``--min-drift``, so this is usable
as an acceptance check rather than a transcript to read by eye.

Usage::

    python -X utf8 scripts/market_stream_drift_scan.py
    python -X utf8 scripts/market_stream_drift_scan.py --window-hours 24 --step-hours 6
    python -X utf8 scripts/market_stream_drift_scan.py --min-drift 0.003 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DB = os.path.join("storage", "trading_cache.db")


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise SystemExit(f"no such database: {db_path}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def load_stream(conn: sqlite3.Connection, since: float) -> Dict[str, List[Tuple[float, float]]]:
    by_symbol: Dict[str, List[Tuple[float, float]]] = {}
    rows = conn.execute(
        "SELECT ts, symbol, price FROM market_stream WHERE ts >= ? AND price > 0 ORDER BY ts",
        (since,),
    )
    for ts, symbol, price in rows:
        if not symbol or ts is None or price is None:
            continue
        by_symbol.setdefault(str(symbol), []).append((float(ts), float(price)))
    for pairs in by_symbol.values():
        pairs.sort(key=lambda p: p[0])
    return by_symbol


def _slice(pairs: Sequence[Tuple[float, float]], lo: float, hi: float) -> List[Tuple[float, float]]:
    return [p for p in pairs if lo <= p[0] < hi]


def symbol_drift(
    pairs: Sequence[Tuple[float, float]],
    lo: float,
    hi: float,
    min_ticks: int,
    min_span_frac: float,
    max_price_ratio: float,
) -> Tuple[Optional[float], str]:
    """Median-of-last-tenth over median-of-first-tenth, or (None, reason)."""
    inside = _slice(pairs, lo, hi)
    if len(inside) < min_ticks:
        return None, "thin"
    span = inside[-1][0] - inside[0][0]
    if span < min_span_frac * (hi - lo):
        return None, "short_span"
    prices = [p[1] for p in inside]
    if min(prices) <= 0 or max(prices) / min(prices) > max_price_ratio:
        return None, "price_ratio"
    block = max(2, len(inside) // 10)
    first = statistics.median(prices[:block])
    last = statistics.median(prices[-block:])
    if first <= 0:
        return None, "price_ratio"
    return (last / first) - 1.0, "ok"


def scan(
    by_symbol: Dict[str, List[Tuple[float, float]]],
    now: float,
    window_hours: float,
    step_hours: float,
    max_end_hours_ago: float,
    min_ticks: int,
    min_symbols: int,
    min_span_frac: float,
    max_price_ratio: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    end_hours_ago = 0.0
    while end_hours_ago + window_hours <= max_end_hours_ago:
        hi = now - end_hours_ago * 3600.0
        lo = hi - window_hours * 3600.0
        drifts: List[float] = []
        dropped: Dict[str, int] = {}
        for pairs in by_symbol.values():
            drift, reason = symbol_drift(
                pairs, lo, hi, min_ticks, min_span_frac, max_price_ratio
            )
            if drift is None:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue
            drifts.append(drift)
        if len(drifts) >= min_symbols:
            drifts.sort()
            out.append(
                {
                    "end_hours_ago": end_hours_ago,
                    "symbols": len(drifts),
                    "median_drift": statistics.median(drifts),
                    "mean_drift": statistics.fmean(drifts),
                    "share_up": sum(1 for d in drifts if d > 0) / len(drifts),
                    "p25": drifts[len(drifts) // 4],
                    "p75": drifts[(3 * len(drifts)) // 4],
                    "dropped_price_ratio": dropped.get("price_ratio", 0),
                    "dropped_thin": dropped.get("thin", 0),
                }
            )
        end_hours_ago += step_hours
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan market_stream for an UP window to re-run the horizon table on"
    )
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--step-hours", type=float, default=6.0)
    ap.add_argument(
        "--max-end-hours-ago",
        type=float,
        default=0.0,
        help="how far back to scan; 0 means the whole tape held in market_stream",
    )
    ap.add_argument("--min-ticks", type=int, default=20, help="per symbol, inside the window")
    ap.add_argument("--min-symbols", type=int, default=10, help="a window with fewer is not scored")
    ap.add_argument(
        "--min-span-frac",
        type=float,
        default=0.6,
        help="a symbol's ticks must cover this fraction of the window or its drift is a stale-price artifact",
    )
    ap.add_argument(
        "--max-price-ratio",
        type=float,
        default=50.0,
        help="drop a symbol whose in-window max/min price exceeds this: denomination contamination, not a move",
    )
    ap.add_argument(
        "--min-drift",
        type=float,
        default=0.003,
        help="a window clears as UP when its MEDIAN per-symbol drift exceeds this fraction",
    )
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    conn = _connect(args.db)
    now = time.time()
    oldest = conn.execute("SELECT MIN(ts) FROM market_stream WHERE price > 0").fetchone()[0]
    if oldest is None:
        raise SystemExit("market_stream holds no priced rows")
    span_hours = (now - float(oldest)) / 3600.0
    max_end = args.max_end_hours_ago or span_hours
    by_symbol = load_stream(conn, float(oldest))
    conn.close()

    windows = scan(
        by_symbol,
        now,
        args.window_hours,
        args.step_hours,
        max_end,
        args.min_ticks,
        args.min_symbols,
        args.min_span_frac,
        args.max_price_ratio,
    )
    if not windows:
        print("no window had enough symbols to score")
        return 1

    up = [w for w in windows if w["median_drift"] > args.min_drift]
    ranked = sorted(windows, key=lambda w: w["median_drift"], reverse=True)

    if args.json:
        print(json.dumps({"windows": windows, "up": up, "span_hours": span_hours}, indent=2))
        return 0 if up else 1

    print("=" * 78)
    print(f"MARKET_STREAM DRIFT SCAN -- {args.window_hours:.0f}h windows every {args.step_hours:.0f}h")
    print("=" * 78)
    print(f"  tape span            : {span_hours:.1f}h, {len(by_symbol)} symbols")
    print(f"  windows scored       : {len(windows)}")
    print(f"  UP threshold         : median per-symbol drift > {args.min_drift * 100:+.3f}%")
    print()
    print("  end_hours_ago    n_sym   median%   mean%   share_up   p25%     p75%    dropped(ratio/thin)")
    for w in ranked[: args.top]:
        print(
            f"  {w['end_hours_ago']:>10.0f}   {w['symbols']:>6}  {w['median_drift'] * 100:>+8.3f}"
            f" {w['mean_drift'] * 100:>+8.3f}   {w['share_up'] * 100:>6.1f}%"
            f" {w['p25'] * 100:>+8.3f} {w['p75'] * 100:>+8.3f}"
            f"      {w['dropped_price_ratio']}/{w['dropped_thin']}"
        )
    print()
    best = ranked[0]
    if up:
        print(f"  VERDICT: {len(up)} UP window(s) found. Best is --end-hours-ago {best['end_hours_ago']:.0f}")
        print(f"           median per-symbol drift {best['median_drift'] * 100:+.3f}%, {best['symbols']} symbols.")
        print("           Re-run: python -X utf8 scripts/head_vs_realised_census.py --horizon-table \\")
        print(f"                     --hours {args.window_hours:.0f} --end-hours-ago {best['end_hours_ago']:.0f}")
        return 0
    print(f"  VERDICT: NO UP WINDOW in {span_hours:.0f}h of tape.")
    print(f"           The best window we hold is --end-hours-ago {best['end_hours_ago']:.0f} at"
          f" {best['median_drift'] * 100:+.3f}%,")
    print(f"           which is below the {args.min_drift * 100:+.3f}% bar. The horizon finding cannot be")
    print("           confirmed in an UP tape from stored data; that needs more tape, not more analysis.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
