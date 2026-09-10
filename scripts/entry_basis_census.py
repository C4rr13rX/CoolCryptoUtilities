"""Measure the threshold for the second entry-basis corroboration source.

WHY A SECOND SOURCE EXISTS AT ALL
---------------------------------
``services.entry_price_corroboration`` corroborates an entry price against the
FEED (``market_stream``). That works, and over 209 closed round trips it refuses
4.3% of them. But it cannot judge a symbol the feed has not reached in the
window, and it returns ``corroborated=None`` there -- which the ghost lane, by
design, lets through.

That hole contains the exact row item [d763940a] is about. AERO-USDC entry
1.140000 has NO feed coverage in the two hours before it, so the shipped gate
calls it unjudgeable and the GHOST lane -- the lane that produces every row
graduation reads -- accepts it. Only ``strict=True`` (the live setting) refuses
it. The headline case is not caught in the lane that produced it.

THE SECOND SOURCE, AND WHY IT CANNOT CORROBORATE ITS OWN CONTAMINATION
----------------------------------------------------------------------
The book itself knows what AERO costs. The trade immediately before booked an
entry at 0.436805. Comparing 1.140000 against the symbol's own recently BOOKED
prices catches it with no feed at all.

The trap is obvious and this script exists to avoid falling into it: the
contaminated tick is ALSO booked -- it is trade 1's exit price -- so a
"was this price seen before" rule self-corroborates. The MEDIAN does not. One
bad price among many moves a median by nothing; it takes half the sample to
move it, and contamination here is 6 rows in 209 (2.9%). So the rule is a ratio
against the median of the symbol's other booked prices, not membership.

WHAT THIS SCRIPT PRINTS
-----------------------
For every closed round trip: the ratio ``entry_price / median(other booked
prices for the same symbol within the window)``, the distribution of that
ratio, and how many rows each candidate threshold would refuse. The threshold
belongs to whichever number in that table both catches the contaminated rows
and leaves the harness able to gather evidence.

    python -X utf8 scripts/entry_basis_census.py
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "storage" / "trading_cache.db"

# Candidate refusal factors. A row is refused when the entry price is more than
# FACTOR times the symbol's own median booked price, or less than 1/FACTOR.
CANDIDATES = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 5.0, 10.0)


def _rows(con: sqlite3.Connection) -> List[Tuple[str, float, float, float, float]]:
    out = []
    for sym, ts, ep, xp, net in con.execute(
        "SELECT symbol, ts, entry_price, exit_price, net_profit FROM trade_outcomes "
        "WHERE entry_price IS NOT NULL AND exit_price IS NOT NULL"
    ):
        try:
            e, x = float(ep), float(xp)
        except (TypeError, ValueError):
            continue
        if e <= 0 or x <= 0:
            continue
        out.append((str(sym or ""), float(ts or 0.0), e, x, float(net or 0.0)))
    out.sort(key=lambda r: r[1])
    return out


def basis_ratio(
    rows: List[Tuple[str, float, float, float, float]],
    idx: int,
    window_sec: float,
    min_support: int,
    peers_kind: str = "both",
) -> Tuple[Optional[float], int, Optional[float]]:
    """Ratio of row ``idx``'s entry price to its symbol's own median.

    The median is taken over the OTHER booked prices (entries and exits) of the
    same symbol inside ``window_sec`` either side. Returns
    ``(ratio, support, median)``; ratio is None when support is too thin to
    judge -- unjudgeable is not the same as contaminated.
    """
    sym, ts, entry, _x, _n = rows[idx]
    peers: List[float] = []
    for j, (s2, t2, e2, x2, _n2) in enumerate(rows):
        if j == idx or s2 != sym:
            continue
        if abs(t2 - ts) > window_sec:
            continue
        peers.append(e2)
        if peers_kind == "both":
            # An EXIT price carries the overshoot a limit fill booked, and in
            # the AERO pair the contaminated tick IS an exit. Including exits
            # therefore feeds the contamination into the median that is meant
            # to detect it.
            peers.append(x2)
    if len(peers) < min_support:
        return None, len(peers), None
    med = statistics.median(peers)
    if med <= 0:
        return None, len(peers), None
    return entry / med, len(peers), med


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--window-sec", type=float, default=86400.0,
                    help="how far either side to gather the symbol's own prices")
    ap.add_argument("--min-support", type=int, default=4,
                    help="peer prices needed before a median is trusted")
    ap.add_argument("--peers", choices=("both", "entries"), default="both",
                    help="use the symbol's booked entries only, or entries and exits")
    a = ap.parse_args(argv)

    con = sqlite3.connect(a.db)
    try:
        rows = _rows(con)
    finally:
        con.close()

    print("ENTRY BASIS vs THE SYMBOL'S OWN MEDIAN BOOKED PRICE")
    print("  db            %s" % a.db)
    print("  closed trips  %d" % len(rows))
    print("  window        %.0fs either side, min support %d peer prices (%s)"
          % (a.window_sec, a.min_support, a.peers))
    print()

    ratios: List[Tuple[float, str, float, float, float]] = []
    unjudgeable = 0
    for i in range(len(rows)):
        r, support, med = basis_ratio(rows, i, a.window_sec, a.min_support, a.peers)
        if r is None:
            unjudgeable += 1
            continue
        ratios.append((r, rows[i][0], rows[i][2], med or 0.0, rows[i][4]))

    if not ratios:
        print("  NO judgeable rows -- nothing to threshold.")
        return 1

    vals = sorted(x[0] for x in ratios)
    def pct(p: float) -> float:
        return vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]

    print("  judgeable %d, unjudgeable %d (too few peer prices)"
          % (len(ratios), unjudgeable))
    print("  ratio percentiles: p1 %.4f  p5 %.4f  p50 %.4f  p95 %.4f  p99 %.4f  max %.4f"
          % (pct(0.01), pct(0.05), pct(0.50), pct(0.95), pct(0.99), vals[-1]))
    print()
    print("  factor   refused  share    the rows it refuses")
    print("  " + "-" * 86)
    for f in CANDIDATES:
        bad = [x for x in ratios if x[0] > f or x[0] < 1.0 / f]
        names = ", ".join("%s %.6f (%.2fx)" % (s, e, r)
                          for r, s, e, _m, _n in sorted(bad, key=lambda z: -z[0])[:4])
        print("  %6.2f   %7d  %5.1f%%  %s"
              % (f, len(bad), 100.0 * len(bad) / len(ratios), names or "-"))
    print()

    worst = sorted(ratios, key=lambda z: -max(z[0], 1.0 / z[0] if z[0] > 0 else 0.0))[:10]
    print("  THE ROWS FURTHEST FROM THEIR SYMBOL'S OWN MEDIAN")
    print("  symbol            entry        median      ratio     net")
    print("  " + "-" * 62)
    for r, s, e, m, n in worst:
        print("  %-16s %10.6f  %10.6f  %7.3fx  %+8.4f" % (s, e, m, r, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
