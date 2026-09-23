"""Reconcile the TWO tape-density censuses that item [aca39497] carries.

The item's headline says deciles 9-10 of expected_return sit on a THINNER tape:
"median 716 ticks per symbol" against "1242" for deciles 1-8 (pass 486, over a
7-day window). Pass 674's scripts/expected_return_density_probe.py measured the
same axis on the literal-vs-forecast split and got 34 against 28.

34-vs-28 is not 1242-vs-716, and pass 674 filed that gap UNRECONCILED rather
than picking a winner -- correctly, because two readings survive it:

  HYPOTHESIS  the two censuses WEIGHT differently. 1242/716 is ROW-weighted
              (one tick-count per enter ROW, so a symbol emitting 400 rows
              votes 400 times) and 34/28 is SYMBOL-weighted (one value per
              distinct symbol). The item's own text is the tell: 992 rows
              across ELEVEN distinct symbols, 253 rows across FORTY. If this
              is right the gap is pure arithmetic and neither number is wrong.

  COMPLEMENT  the weighting is the same in both and the difference is
              SELECTION. The calibration can only price a row whose symbol has
              a 15-minute forward quote, which selects for dense tape; its 51
              symbols are genuinely denser than the probe's 1100+. If this is
              right, row-weighting will NOT close the gap on its own and only
              restricting to priceable symbols will.

They predict different numbers, so one run separates them. This script counts
ticks ONCE, from the same market_stream table both other scripts read, over one
window, and then reports the median BOTH ways over the same rows. Read-only.

    python -X utf8 scripts/tick_census_weighting_reconcile.py [--days 7]

--days defaults to 7 to match expected_return_calibration.py:57 LOOKBACK_DAYS,
which is the window the 1242/716 figure was measured on. Pass 30 to line it up
with the density probe instead.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the pass-674 probe's own row reader and tick counter rather than
# re-implementing them. If this script parsed `details` or resolved
# market_stream's columns its own way, a difference in the ANSWER could just be
# a difference in the PARSER, and the reconciliation would prove nothing.
from scripts.expected_return_density_probe import (  # noqa: E402
    DEFAULT_DB,
    LOW_CONFIDENCE_FLOOR,
    _connect,
    _rows,
    _target_floor,
    _tick_counts,
)


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _report(
    label: str,
    rows: List[dict],
    ticks: Dict[str, int],
    min_ticks: int,
) -> None:
    """Median tape density for one cohort, computed BOTH ways over one set."""
    kept = [r for r in rows if ticks.get(r["symbol"], 0) >= min_ticks]
    symbols = sorted({r["symbol"] for r in kept})

    # ROW-weighted: one tick-count per enter row. A symbol emitting 400 rows
    # contributes 400 values, so the median lands wherever the ROWS are.
    row_weighted = [float(ticks.get(r["symbol"], 0)) for r in kept]
    # SYMBOL-weighted: one value per distinct symbol, regardless of row count.
    sym_weighted = [float(ticks.get(s, 0)) for s in symbols]

    print(
        f"  {label:<26} rows={len(kept):>6}  symbols={len(symbols):>5}"
        f"   ROW-weighted median={_fmt(_median(row_weighted)):>8}"
        f"   SYMBOL-weighted median={_fmt(_median(sym_weighted)):>6}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--db", default=DEFAULT_DB)
    # The calibration prices a row only if its symbol carries a usable series.
    # This stands in for that selection without re-deriving forward prices.
    ap.add_argument("--priceable-min-ticks", type=int, default=100)
    args = ap.parse_args()

    since = time.time() - args.days * 86400.0
    conn = _connect(args.db)
    try:
        rows = _rows(conn, since)
        if not rows:
            print("  NO ROWS. Widen --days before reading this as a result.")
            return 1
        symbols = sorted({r["symbol"] for r in rows if r["symbol"]})
        ticks = _tick_counts(conn, symbols, since)
    finally:
        conn.close()

    if not ticks:
        print("  NO TICK CENSUS -- cannot reconcile. Stop here.")
        return 1

    # Resolve the floor the same way the p674 probe does -- from the live env,
    # not a hardcoded 0.05 -- so this cannot measure a stale constant.
    literals = {round(LOW_CONFIDENCE_FLOOR, 6), round(_target_floor(), 6)}
    on_literal = [r for r in rows if round(r["expected_return"], 6) in literals]
    forecast = [r for r in rows if round(r["expected_return"], 6) not in literals]

    # The item split at the 8th decile of expected_return, so reproduce that
    # cut too -- it is the split the 1242/716 figure was actually measured on.
    ordered = sorted(rows, key=lambda r: r["expected_return"])
    cut = int(len(ordered) * 0.8)
    low_deciles, top_deciles = ordered[:cut], ordered[cut:]

    print("=" * 78)
    print("TICK-CENSUS RECONCILIATION -- is 1242-vs-716 WEIGHTING or SELECTION?")
    print("=" * 78)
    print(f"  window           : last {args.days:g} days")
    print(f"  db               : {args.db}")
    print(f"  enter rows       : {len(rows):,}   distinct symbols: {len(symbols):,}")
    print(f"  tape table       : market_stream (the same one both censuses read)")

    print("\n--- ALL SYMBOLS (no density floor) ---")
    _report("all enter rows", rows, ticks, 0)
    _report("on a literal", on_literal, ticks, 0)
    _report("carrying a forecast", forecast, ticks, 0)
    _report("deciles 1-8", low_deciles, ticks, 0)
    _report("deciles 9-10", top_deciles, ticks, 0)

    print("\n--- STREAMED ONLY (>=1 tick) -- what the p674 probe reported ---")
    _report("on a literal", on_literal, ticks, 1)
    _report("carrying a forecast", forecast, ticks, 1)
    _report("deciles 1-8", low_deciles, ticks, 1)
    _report("deciles 9-10", top_deciles, ticks, 1)

    print(
        f"\n--- PRICEABLE ONLY (>={args.priceable_min_ticks} ticks) -- stands in"
        " for the calibration's selection ---"
    )
    _report("deciles 1-8", low_deciles, ticks, args.priceable_min_ticks)
    _report("deciles 9-10", top_deciles, ticks, args.priceable_min_ticks)

    print("\n--- HOW TO READ THIS ---")
    print("  The two columns are the SAME rows and the SAME tick counts. Any gap")
    print("  between them is arithmetic alone. If ROW-weighted lands in the")
    print("  hundreds-to-thousands while SYMBOL-weighted stays in the tens, the")
    print("  HYPOTHESIS holds: 1242/716 and 34/28 are the same tape measured two")
    print("  ways and neither refutes the other. If both columns stay in the tens")
    print("  until the density floor is applied, the COMPLEMENT holds and the gap")
    print("  is selection, not weighting.")
    print("  NOTE the window: 1242/716 was measured over 7 days and the p674")
    print("  probe over 30. A LONGER window cannot hold FEWER ticks, so the")
    print("  windows alone can never explain a gap in this direction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
