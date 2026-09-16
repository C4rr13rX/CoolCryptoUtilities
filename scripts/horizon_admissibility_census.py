"""Does the horizon the population PROPOSES fit inside the horizon the feed
can actually carry?

Item b3d087c7. The chaos layer (``web/tradingagent/lattice.py:142``) refuses an
entry when ``proposed_horizon_sec > usable_horizon_sec``, and 4866 of 4875
lattice refusals over seven days are that one comparison. The layer is not the
bug -- it is right, and "unmeasurable is not permission" is right. The question
this script answers is the other side of the comparison, which nobody had
measured: **what usable horizon does the live feed actually support, per
symbol?**

Run:

    python -X utf8 scripts/horizon_admissibility_census.py
    python -X utf8 scripts/horizon_admissibility_census.py --hours 72 --band 300 3600

``--band`` is the operator's asked-for cadence in seconds: trades every 5-30
minutes, stretching to 45-60 when moves are 2-5%. A symbol whose usable horizon
falls below the bottom of that band cannot carry ANY money-button trade, no
matter what the strategy proposes.

Reads ``storage/trading_cache.db`` read-only and calls nothing over the network.
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Below this many ticks ``chaos_profile`` cannot bound a horizon, and the
#: layer refuses on "unmeasurable" rather than on the comparison. Counted
#: separately, because the two have different fixes: more feed vs shorter
#: horizons.
MIN_TICKS = 24


def _db_path() -> Path:
    import os
    configured = os.getenv("TRADING_DB_PATH") or os.getenv("TRADING_CACHE_DB")
    if configured:
        return Path(configured)
    return ROOT / "storage" / "trading_cache.db"


def series(hours: float, db_path: Path) -> Dict[str, List[tuple]]:
    """``{symbol: [(ts, price), ...]}`` from the live stream, oldest first."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True,
                           timeout=10.0)
    try:
        newest = conn.execute("SELECT MAX(ts) FROM market_stream").fetchone()[0]
        if newest is None:
            return {}
        cut = float(newest) - hours * 3600.0
        out: Dict[str, List[tuple]] = defaultdict(list)
        for ts, sym, px in conn.execute(
                "SELECT ts, symbol, price FROM market_stream "
                "WHERE ts >= ? ORDER BY ts", (cut,)):
            if px and float(px) > 0:
                out[sym].append((float(ts), float(px)))
        return dict(out)
    finally:
        conn.close()


def median_bar_sec(points: Sequence[tuple]) -> float:
    gaps = [b[0] - a[0] for a, b in zip(points, points[1:]) if b[0] > a[0]]
    return statistics.median(gaps) if gaps else 0.0


def census(hours: float, band: Sequence[float], db_path: Path) -> Dict:
    from web.tradingagent.chaos import chaos_profile

    lo, hi = float(band[0]), float(band[1])
    rows = []
    unmeasurable = []
    thin = []
    for symbol, points in sorted(series(hours, db_path).items()):
        if len(points) < MIN_TICKS:
            thin.append((symbol, len(points)))
            continue
        bar = median_bar_sec(points)
        prices = [p for _, p in points]
        try:
            profile = chaos_profile(symbol, prices, bar or 60.0)
        except Exception as exc:  # noqa: BLE001 -- a census must not die on one symbol
            unmeasurable.append((symbol, len(points), f"raised {exc!r}"))
            continue
        usable = profile.get("usable_horizon_sec")
        if usable is None:
            unmeasurable.append((symbol, len(points), "horizon unbounded"))
            continue
        rows.append({
            "symbol": symbol,
            "ticks": len(points),
            "bar_sec": bar,
            "usable_sec": float(usable),
            "character": profile.get("character"),
            "carries_band_floor": float(usable) >= lo,
            "carries_band_ceiling": float(usable) >= hi,
        })
    return {"hours": hours, "band": [lo, hi], "rows": rows,
            "unmeasurable": unmeasurable, "thin": thin}


def _pct(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--band", type=float, nargs=2, default=[300.0, 3600.0],
                    help="the cadence band in seconds (default 5 to 60 min)")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    db = Path(args.db) if args.db else _db_path()
    out = census(args.hours, args.band, db)
    rows = out["rows"]
    lo, hi = out["band"]

    print(f"book {db}")
    print(f"window {args.hours:.0f}h   cadence band "
          f"{lo / 60:.0f}-{hi / 60:.0f} min")
    print()
    if not rows:
        print("NO SYMBOL produced a bounded usable horizon. That is the whole "
              "finding: the chaos layer is refusing on 'unmeasurable', not on "
              "the comparison, and the fix is feed density, not shorter "
              "horizons.")
    else:
        usable = [r["usable_sec"] for r in rows]
        print(f"usable_horizon_sec over {len(rows)} symbols that bound one:")
        for q, name in ((0.10, "p10"), (0.50, "p50"), (0.90, "p90")):
            v = _pct(usable, q)
            print(f"  {name}  {v:9.1f} s  = {v / 60:7.2f} min")
        print(f"  min  {min(usable):9.1f} s   max {max(usable):9.1f} s")
        print()
        floor_ok = [r for r in rows if r["carries_band_floor"]]
        ceil_ok = [r for r in rows if r["carries_band_ceiling"]]
        print(f"symbols whose usable horizon reaches the band FLOOR "
              f"({lo / 60:.0f} min): {len(floor_ok)} of {len(rows)}")
        print(f"symbols whose usable horizon reaches the band CEILING "
              f"({hi / 60:.0f} min): {len(ceil_ok)} of {len(rows)}")
        print()
        print(f"{'symbol':<18}{'ticks':>7}{'bar_s':>8}{'usable_min':>12}"
              f"{'>=floor':>9}{'>=ceil':>8}  character")
        for r in sorted(rows, key=lambda r: -r["usable_sec"]):
            print(f"{r['symbol']:<18}{r['ticks']:>7}{r['bar_sec']:>8.0f}"
                  f"{r['usable_sec'] / 60:>12.2f}"
                  f"{'yes' if r['carries_band_floor'] else 'NO':>9}"
                  f"{'yes' if r['carries_band_ceiling'] else 'NO':>8}  "
                  f"{r['character']}")
    print()
    print(f"symbols with an UNBOUNDED horizon (refused as unmeasurable, not on "
          f"the comparison): {len(out['unmeasurable'])}")
    for sym, n, why in out["unmeasurable"][:12]:
        print(f"  {sym:<18} {n:>5} ticks  {why}")
    print(f"symbols below the {MIN_TICKS}-tick floor, not profiled: "
          f"{len(out['thin'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
