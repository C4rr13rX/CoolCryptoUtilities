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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
