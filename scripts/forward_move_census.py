"""Is the zero median forward move a PROPERTY OF THE MARKET or an ARTIFACT OF A FROZEN TAPE?

Answer, measured p694 over 186 symbols / 168h: NEITHER. Frozen windows are
8.8-11.6%, not the ~49% the feed-darkness number would predict, so the zero
median is not bookkeeping. But the tape moves: median |move| is 10.68 bp at 12
min and 21.28 bp at 30 min, against an 11.00 bp WETH-class toll. The median
SIGNED move is exactly +0.0000% at every horizon AND on the nonzero subset.
The tape is not dead, it is DRIFTLESS -- a symmetric walk with real amplitude.
The prize at deep-pool depth is real and the binding term is DIRECTION, not
cost and not hold length.

Usage:  python -X utf8 scripts/forward_move_census.py

Sage, p694. The money_button tape replay put the median net at -0.700% in every
population at every horizon from 2 to 90 minutes, which is the round trip
exactly -- so the median forward GROSS move is 0.000%.

HYPOTHESIS: the tape genuinely does not move over these horizons, so no cost
reduction and no hold change can make a median trade pay, and only the tail
pays.
COMPLEMENT: the median is zero by BOOKKEEPING -- the feed is dark ~49% of
minutes, and a window whose exit tick is the same print as its entry tick books
an exactly-flat return. Strip the frozen windows and the live tape may move
plenty.

THE MEASUREMENT THAT TELLS THEM APART: the share of forward windows that are
EXACTLY zero (same print, or a repeated price), and the median absolute move on
the windows that are NOT.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from money_button_gate_census import load_ticks  # noqa: E402

HOURS = 168.0
HORIZONS = (5.0, 12.0, 30.0)
#: The deep pairs, read individually. p694 found the POOLED table flatters them:
#: pooled median |move| at 30 min is 21.28 bp but WETH-USDC's own is 14.06 bp
#: and ETH-USDC's 14.79 bp -- the pooled figure is carried by UNI-USDC (86.82)
#: and AERO-USDC (38.25), pairs whose own toll eats the amplitude. Never quote
#: the pooled number as a deep-pair number.
DEEP = ("WETH-USDC", "ETH-USDC", "CBBTC-USDC", "UNI-USDC", "AERO-USDC", "CBETH-USDC")
MIN_CELL = 500   #: below this a per-symbol cell is not reported at all
#: This repo's standing plausibility bar: a forward return over 2-90 minutes
#: larger than 20% is a price-scale artifact, not a move.
PLAUSIBLE_ABS = 0.20
WETH_TOLL = 0.0011   # 11.00 bp, WETH-class depth, from the replay's own footer
CENSUS_TOLL = 0.0070  # 70.00 bp at census fee


def main() -> int:
    ticks = load_ticks(HOURS)
    print("symbols with ticks in %.0fh: %d" % (HOURS, len(ticks)))
    for hold_min in HORIZONS:
        hold = hold_min * 60.0
        moves = []
        frozen_same_print = 0
        for sym, rows in ticks.items():
            n = len(rows)
            if n < 3:
                continue
            j = 0
            for i in range(n):
                t0, p0, _ = rows[i]
                if p0 <= 0:
                    continue
                target = t0 + hold
                if j < i:
                    j = i
                while j < n and rows[j][0] < target:
                    j += 1
                if j >= n:
                    break
                p1 = rows[j][1]
                if p1 <= 0:
                    continue
                if j == i:
                    frozen_same_print += 1
                moves.append((p1 / p0) - 1.0)
        a = np.asarray(moves, dtype=float)
        if not a.size:
            print("h=%.0f  no windows" % hold_min)
            continue
        a = a[np.abs(a) <= 0.20]          # same plausibility bar as the replay
        exact_zero = int((a == 0.0).sum())
        nz = a[a != 0.0]
        print("\n--- horizon %.0f min, n=%d plausible windows ---" % (hold_min, a.size))
        print("  EXACTLY zero move      : %d (%.1f%%)" % (exact_zero, exact_zero / a.size * 100.0))
        print("  median signed move     : %+.4f%%" % (np.median(a) * 100.0))
        print("  median |move|, ALL     : %+.4f%% (%.2f bp)"
              % (np.median(np.abs(a)) * 100.0, np.median(np.abs(a)) * 10000.0))
        if nz.size:
            print("  median |move|, NONZERO : %+.4f%% (%.2f bp)  on n=%d"
                  % (np.median(np.abs(nz)) * 100.0, np.median(np.abs(nz)) * 10000.0, nz.size))
            print("  median signed, NONZERO : %+.4f%%" % (np.median(nz) * 100.0))
        for toll, name in ((WETH_TOLL, "WETH-class 11bp"), (CENSUS_TOLL, "census 70bp")):
            up = float((a > toll).sum()) / a.size * 100.0
            print("  long-only wins vs %-16s: %5.1f%%   (needs >50%% for a median-positive trade)"
                  % (name, up))

    # ------------------------------------------------------------------
    # The same read PER SYMBOL on the deep pairs. This is the one that
    # decides anything: pooling 186 symbols mixes a 11bp toll with a 70bp
    # one, and the amplitude that makes the pooled table look winnable
    # lives on the pairs that cannot pay for it.
    # ------------------------------------------------------------------
    print("")
    print("--- the deep pairs, read one at a time (n>=%d per cell) ---" % MIN_CELL)
    print("  a pair is only tradeable long if its median SIGNED move exceeds its own toll.")
    for sym in DEEP:
        rows = ticks.get(sym) or []
        if len(rows) < 10:
            print("  %-12s ticks=%-6d -- no tape" % (sym, len(rows)))
            continue
        print("  %-12s ticks=%d" % (sym, len(rows)))
        for hold_min in HORIZONS:
            hold = hold_min * 60.0
            mv = []
            j = 0
            n = len(rows)
            for i in range(n):
                t0, p0, _ = rows[i]
                if p0 <= 0:
                    continue
                target = t0 + hold
                if j < i:
                    j = i
                while j < n and rows[j][0] < target:
                    j += 1
                if j >= n:
                    break
                p1 = rows[j][1]
                if p1 > 0:
                    mv.append(p1 / p0 - 1.0)
            a = np.asarray(mv, dtype=float)
            a = a[np.abs(a) <= PLAUSIBLE_ABS] if a.size else a
            if a.size < MIN_CELL:
                print("      h%-3.0f n=%-5d BELOW THE CELL BAR -- not reported"
                      % (hold_min, a.size))
                continue
            print("      h%-3.0f n=%-5d  median SIGNED %+7.2f bp   median |move| %7.2f bp"
                  "   long>11bp %5.1f%%"
                  % (hold_min, a.size, np.median(a) * 10000.0,
                     np.median(np.abs(a)) * 10000.0,
                     float((a > WETH_TOLL).sum()) / a.size * 100.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
