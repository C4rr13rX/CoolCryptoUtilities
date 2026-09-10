"""Which streamed symbols carry TWO price regimes -- two assets, or two units.

A symbol whose feed holds two clusters of price is not one asset that moved. It
is either two different tokens published under one ticker, or one token
published in two DENOMINATIONS. Both make every trade in that symbol
unmeasurable, and both are already in this repo's history.

Measured 2026-09-10 over 132 symbols with >= 50 ticks: 47 of them (36%) carry a
second regime of >= 5 ticks. The two clearest kinds:

  TWO ASSETS UNDER ONE TICKER
    COMP-USDC   4468 ticks, median 19.98, and 170 ticks at 42.82-55.34.
                The regimes are separated in TIME (high to 08-27 12:41, low
                from 08-26 16:40 on) and BOTH providers publish BOTH, so it is
                not two sources disagreeing -- the feed changed which asset it
                calls COMP around 08-27.

  ONE ASSET IN TWO DENOMINATIONS
    CBETH-WETH  384 ticks, median 1.1386, and 5 ticks at 2687-2851. The median
                is cbETH priced in WETH; the outliers are cbETH priced in USD.
    EURC-WETH   median 0.00047 against 16 ticks at 0.599-1.160.
    CBETH-CBBTC, SOL-CBBTC, JITOSOL-CBBTC are the same shape.

WHY THIS IS THE GRADUATION PROBLEM AND NOT A FEED TIDY-UP. The two deepest and
worst symbols in the de-contaminated tradeable book are both heavily regime-split:
BASECAT-USDC (1513 of 6339 ticks off-regime; 36 round trips, net -1.9019) and
COMP-USDC (170 of 4468; 16 round trips, net -1.1173). Those two carry most of
the book's losses. A strategy cannot show an edge in a symbol whose price series
is two series interleaved.

WHAT THIS CANNOT DO. It cannot say WHICH regime is the real asset, because
``market_stream`` does not record the pool or contract a tick came from -- the
row is {ts, symbol, chain, price, volume, rest, consensus_confidence} and
nothing more. Attribution needs that field to exist first. This script sizes the
problem and names the symbols; it does not resolve them.

The split is on RATIO, not on standard deviations: these clusters are 2x to
1000x apart, and a sigma-based test on a bimodal series measures the gap between
the modes rather than the spread of either.

Run:  python -X utf8 scripts/feed_regime_census.py
      python -X utf8 scripts/feed_regime_census.py --json --min-ticks 100
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

# A price this many times away from the median is a different regime rather than
# a move. 1.5x is deliberately loose: the real splits measured are 2x (COMP) to
# 2500x (CBETH-WETH), so nothing near the threshold is being judged, and a
# looser bar cannot manufacture a finding out of ordinary volatility.
REGIME_RATIO = 1.5

# Fewer than this many off-regime ticks is a handful of bad prints, not a
# regime. Reported separately rather than dropped silently.
MIN_REGIME_TICKS = 5

# Below this a median is not a stable reference to measure a regime against.
MIN_TICKS = 50


def census(
    *,
    db_path: Optional[Path] = None,
    min_ticks: int = MIN_TICKS,
    ratio: float = REGIME_RATIO,
    min_regime: int = MIN_REGIME_TICKS,
    con: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Every symbol whose feed holds a second price cluster."""
    owned = con is None
    if con is None:
        con = sqlite3.connect(str(db_path or DEFAULT_DB))
    con.row_factory = sqlite3.Row
    try:
        symbols = [r["symbol"] for r in con.execute(
            "SELECT symbol, COUNT(*) c FROM market_stream "
            "WHERE price > 0 GROUP BY symbol HAVING c >= ?", (int(min_ticks),))]
        rows: List[Dict[str, Any]] = []
        for sym in symbols:
            series = [(float(r["ts"]), float(r["price"])) for r in con.execute(
                "SELECT ts, price FROM market_stream "
                "WHERE symbol = ? AND price > 0 ORDER BY ts", (sym,))]
            if len(series) < min_ticks:
                continue
            med = statistics.median([p for _, p in series])
            if med <= 0:
                continue
            far = [(t, p) for t, p in series
                   if p > med * ratio or p < med / ratio]
            if len(far) < min_regime:
                continue
            rows.append({
                "symbol": sym,
                "ticks": len(series),
                "median": med,
                "off_regime": len(far),
                "off_regime_pct": 100.0 * len(far) / len(series),
                "far_min": min(p for _, p in far),
                "far_max": max(p for _, p in far),
                "far_first_ts": min(t for t, _ in far),
                "far_last_ts": max(t for t, _ in far),
            })
    finally:
        if owned:
            con.close()

    rows.sort(key=lambda r: -r["off_regime"])
    return {"symbols_examined": len(symbols), "affected": len(rows),
            "ratio": ratio, "min_ticks": min_ticks,
            "min_regime_ticks": min_regime, "rows": rows}


def render(rep: Dict[str, Any]) -> str:
    out = ["=" * 84,
           "FEED REGIME CENSUS -- symbols whose price series is two series",
           "=" * 84,
           "  %d of %d symbols with >= %d ticks carry a second regime of >= %d "
           "ticks (%.0f%%)" % (rep["affected"], rep["symbols_examined"],
                               rep["min_ticks"], rep["min_regime_ticks"],
                               100.0 * rep["affected"]
                               / max(1, rep["symbols_examined"])),
           "",
           "  %-18s %7s %12s %6s %7s %12s %12s"
           % ("symbol", "ticks", "median", "off", "off%", "off_min",
              "off_max"),
           "  " + "-" * 80]
    for r in rep["rows"]:
        out.append("  %-18s %7d %12.5f %6d %6.1f%% %12.5f %12.5f" % (
            r["symbol"][:18], r["ticks"], r["median"], r["off_regime"],
            r["off_regime_pct"], r["far_min"], r["far_max"]))
    out.append("")
    out.append("  A symbol listed here cannot show an edge: its price series is "
               "two series interleaved.")
    out.append("  WHICH regime is the real asset is NOT answerable here -- "
               "market_stream stores no pool or")
    out.append("  contract per tick. That field has to exist first.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--min-ticks", type=int, default=MIN_TICKS)
    ap.add_argument("--ratio", type=float, default=REGIME_RATIO)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rep = census(db_path=Path(a.db), min_ticks=a.min_ticks, ratio=a.ratio)
    print(json.dumps(rep, indent=2, default=str) if a.json else render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
