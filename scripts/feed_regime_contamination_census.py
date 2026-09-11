"""Which symbols carry a SECOND price regime in the live tick table, and which ticks they are.

A strategy that prices its edge as a ratio over the recent window --
`obv_accumulation` uses `(recent_high - last_price) / last_price` -- turns one
bar from a foreign price regime into an unbounded forecast rather than a wrong
one. [2ac8532c]. The bound shipped in `trading/strategies/base.py`
(`STRATEGY_MAX_EXPECTED_RETURN`) stops that forecast from sizing a trade; it does
nothing about the window, which every other statistic those strategies compute
is still reading.

This names the symbols and the ticks. A symbol is flagged when its 7-day
max/min price ratio exceeds `--ratio` (default 100x, i.e. two decades, far
beyond any real move on this feed). Its ticks are then split at the geometric
midpoint between the two extremes and the SMALLER side reported: that is the
minority regime, the foreign one.

    python -X utf8 scripts/feed_regime_contamination_census.py
    python -X utf8 scripts/feed_regime_contamination_census.py --symbol CLANKER-USDC
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "storage" / "trading_cache.db"


def _series(conn: sqlite3.Connection, cut: float):
    by: dict[str, list] = defaultdict(list)
    query = (
        "select symbol, ts, price, raw from market_stream "
        "where ts >= ? and price > 0 order by symbol, ts"
    )
    for symbol, ts, price, raw in conn.execute(query, (cut,)):
        by[str(symbol)].append((float(ts), float(price), raw))
    return by


def _minority_regime(seq: list, ratio: float):
    """(minority ticks, decades apart) for a two-regime series, else (None, 0)."""
    prices = [p for _, p, _ in seq]
    lo, hi = min(prices), max(prices)
    if lo <= 0 or hi / lo <= ratio:
        return None, 0.0
    midpoint = math.sqrt(lo * hi)
    high = [t for t in seq if t[1] >= midpoint]
    low = [t for t in seq if t[1] < midpoint]
    minority = high if len(high) <= len(low) else low
    return minority, math.log10(hi / lo)


def _source_of(raw) -> str:
    try:
        blob = json.loads(raw or "{}") or {}
    except (TypeError, ValueError):
        return ""
    for key in ("source", "pool", "pool_address", "dex"):
        if blob.get(key):
            return str(blob[key])
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--ratio", type=float, default=100.0)
    parser.add_argument("--symbol", default="")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=120)
    by = _series(conn, time.time() - args.days * 86400.0)

    if args.symbol:
        seq = by.get(args.symbol) or []
        minority, decades = _minority_regime(seq, args.ratio)
        print(f"{args.symbol}: {len(seq)} ticks over {args.days:g}d")
        if not minority:
            print("  no second regime at this ratio")
            return 0
        print(f"  {len(minority)} minority-regime ticks, {decades:.1f} decades apart")
        for ts, price, raw in minority:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
            print(f"    {stamp}  price={price:.10g}  {_source_of(raw)[:48]}")
        return 0

    print(f"{'symbol':<16}{'ticks':>6}{'minor':>7}{'minor%':>8}  decades  example")
    flagged = 0
    minority_total = 0
    tick_total = 0
    for symbol, seq in sorted(by.items()):
        tick_total += len(seq)
        minority, decades = _minority_regime(seq, args.ratio)
        if not minority:
            continue
        flagged += 1
        minority_total += len(minority)
        share = 100.0 * len(minority) / len(seq)
        example = minority[0]
        print(f"{symbol:<16}{len(seq):>6}{len(minority):>7}{share:>7.1f}%"
              f"  {decades:>6.1f}  p={example[1]:.6g} {_source_of(example[2])[:28]}")
    print(f"\ncontaminated symbols: {flagged} of {len(by)}")
    print(f"foreign-regime ticks: {minority_total} of {tick_total} "
          f"({100.0 * minority_total / max(tick_total, 1):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
