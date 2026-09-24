"""Kill tests for [b4903017] -- the WETH-USDC 30-minute reversion claim.

Sage p686 measured, on market_stream / WETH-USDC / 14 days / non-overlapping
1800s windows / trade AGAINST the trailing 1800s sign:

    n=166, gross mean +4.05bp, MEDIAN +6.60bp, p25 -7.74, p75 +20.80
    net at 4.08bp: mean -0.03bp, MEDIAN +2.52bp, WIN 53.6 pct

This script does not fit anything. It reproduces that measurement and then runs
the pre-registered kill tests from the item's acceptance criteria:

  1. out-of-sample split -- first half vs second half, sign stability
  2. Wilson 95pct binomial CI on the win rate, vs 0.50
  3. medians everywhere, with the top-3 and bottom-3 rows printed individually
  4. tape continuity -- rerun on windows with unbroken tape across the hold
  5. fill-to-fill is NOT priced here; every number is a tick-to-tick UPPER BOUND

Usage:  python scripts/weth_reversion_kill.py [--symbol WETH-USDC] [--days 14]
                                              [--window 1800] [--toll-bp 4.08]
"""

from __future__ import annotations

import argparse
import bisect
import math
import os
import sqlite3
import time
from typing import List, Optional, Sequence, Tuple

DB = os.environ.get("TRADING_DB_PATH", "storage/trading_cache.db")


def _quantile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _wilson(k: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval -- the right one at n=166, unlike the normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def load_ticks(symbol: str, days: int) -> List[Tuple[float, float]]:
    conn = sqlite3.connect(DB)
    cutoff = time.time() - days * 86400.0
    rows = conn.execute(
        "SELECT ts, price FROM market_stream "
        "WHERE symbol=? AND ts>=? AND price>0 ORDER BY ts",
        (symbol, cutoff),
    ).fetchall()
    conn.close()
    return [(float(t), float(p)) for t, p in rows]


class Tape:
    def __init__(self, ticks: Sequence[Tuple[float, float]]):
        self.ts = [t for t, _ in ticks]
        self.px = [p for _, p in ticks]

    def nearest(self, when: float, tol: float) -> Optional[Tuple[float, float]]:
        """Nearest tick to `when`, or None if the closest one is >tol away."""
        if not self.ts:
            return None
        i = bisect.bisect_left(self.ts, when)
        best: Optional[Tuple[float, float]] = None
        for j in (i - 1, i):
            if 0 <= j < len(self.ts):
                gap = abs(self.ts[j] - when)
                if best is None or gap < best[0]:
                    best = (gap, self.px[j])
        if best is None or best[0] > tol:
            return None
        return best  # (gap_seconds, price)

    def max_gap(self, start: float, end: float) -> float:
        """Largest inter-tick gap strictly inside [start, end], seconds.

        Returns inf when there is no tick at all in the interval -- a hold with
        no tape inside it is maximally discontinuous, not conveniently zero.
        """
        lo = bisect.bisect_left(self.ts, start)
        hi = bisect.bisect_right(self.ts, end)
        inner = self.ts[lo:hi]
        if not inner:
            return float("inf")
        marks = [start] + inner + [end]
        return max(marks[k + 1] - marks[k] for k in range(len(marks) - 1))


class Window:
    __slots__ = ("t", "prev_ret", "fwd_ret", "gross", "max_gap", "entry_gap")

    def __init__(self, t, prev_ret, fwd_ret, gross, max_gap, entry_gap):
        self.t = t
        self.prev_ret = prev_ret
        self.fwd_ret = fwd_ret
        self.gross = gross
        self.max_gap = max_gap
        self.entry_gap = entry_gap


def build(tape: Tape, window: float, tol: float) -> List[Window]:
    """Non-overlapping windows on a fixed grid: t-W (look back), t (enter), t+W (exit)."""
    if not tape.ts:
        return []
    start = tape.ts[0] + window
    end = tape.ts[-1] - window
    out: List[Window] = []
    t = start
    while t <= end:
        back = tape.nearest(t - window, tol)
        here = tape.nearest(t, tol)
        fwd = tape.nearest(t + window, tol)
        if back and here and fwd:
            p0, p1, p2 = back[1], here[1], fwd[1]
            prev_ret = p1 / p0 - 1.0
            fwd_ret = p2 / p1 - 1.0
            if prev_ret != 0.0:
                side = -1.0 if prev_ret > 0 else 1.0  # trade AGAINST the trailing sign
                out.append(
                    Window(
                        t=t,
                        prev_ret=prev_ret,
                        fwd_ret=fwd_ret,
                        gross=side * fwd_ret * 1e4,  # bp
                        max_gap=tape.max_gap(t, t + window),
                        entry_gap=here[0],
                    )
                )
        t += window  # NON-overlapping -- the grid steps a full window
    return out


def report(label: str, wins: Sequence[Window], toll_bp: float, show_rows: bool = False) -> None:
    n = len(wins)
    print(f"\n--- {label}  (n={n}) ---")
    if n == 0:
        print("    no windows")
        return
    gross = [w.gross for w in wins]
    net = [g - toll_bp for g in gross]
    k = sum(1 for x in net if x > 0)
    lo, hi = _wilson(k, n)
    gk = sum(1 for x in gross if x > 0)
    glo, ghi = _wilson(gk, n)
    print(f"    GROSS  median {_quantile(gross,0.5):+8.2f}bp   mean {sum(gross)/n:+8.2f}bp"
          f"   p25 {_quantile(gross,0.25):+8.2f}bp   p75 {_quantile(gross,0.75):+8.2f}bp")
    print(f"    NET@{toll_bp:.2f}bp median {_quantile(net,0.5):+8.2f}bp   mean {sum(net)/n:+8.2f}bp"
          f"   p25 {_quantile(net,0.25):+8.2f}bp   p75 {_quantile(net,0.75):+8.2f}bp")
    print(f"    WIN gross {gk}/{n} = {100*gk/n:5.1f}%   Wilson95 [{100*glo:5.1f}%, {100*ghi:5.1f}%]"
          f"   {'EXCLUDES 50' if glo > 0.5 else 'INCLUDES 50 -- not distinguishable from a coin'}")
    print(f"    WIN net   {k}/{n} = {100*k/n:5.1f}%   Wilson95 [{100*lo:5.1f}%, {100*hi:5.1f}%]"
          f"   {'EXCLUDES 50' if lo > 0.5 else 'INCLUDES 50 -- not distinguishable from a coin'}")
    if show_rows:
        ranked = sorted(wins, key=lambda w: w.gross)
        print("    bottom 3 rows (gross bp, prev_ret bp, fwd_ret bp, max tape gap s):")
        for w in ranked[:3]:
            gap = "inf" if w.max_gap == float("inf") else f"{w.max_gap:.0f}"
            print(f"        {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(w.t))}"
                  f"  {w.gross:+9.2f}  {w.prev_ret*1e4:+9.2f}  {w.fwd_ret*1e4:+9.2f}  {gap:>8}")
        print("    top 3 rows:")
        for w in ranked[-3:]:
            gap = "inf" if w.max_gap == float("inf") else f"{w.max_gap:.0f}"
            print(f"        {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(w.t))}"
                  f"  {w.gross:+9.2f}  {w.prev_ret*1e4:+9.2f}  {w.fwd_ret*1e4:+9.2f}  {gap:>8}")
        # how much of the SUM rides on the extremes -- the one-row-book test
        tot = sum(gross)
        if tot != 0:
            top3 = sum(w.gross for w in ranked[-3:])
            print(f"    top 3 rows carry {100*top3/tot:+.1f}% of the gross SUM"
                  f" (sum {tot:+.1f}bp) -- the median is the honest statistic here, not this")


def scan(days: float, window: float, tol: float, gap: float, toll_bp: float,
         top: int) -> int:
    """Is the dark-tape inflation general, or was WETH unlucky?

    For every symbol with enough tape, report the gross median on ALL windows
    against the gross median on the CONTINUOUS-tape subset. If continuous is
    systematically lower, the edge is a stale-price artifact feed-wide.
    """
    conn = sqlite3.connect(DB)
    cutoff = time.time() - days * 86400.0
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM market_stream WHERE ts>=? AND price>0 "
        "GROUP BY symbol HAVING c>=200 ORDER BY c DESC LIMIT ?", (cutoff, top))]
    conn.close()
    print(f"scan: window {window:g}s, tol {tol:g}s, continuous = no gap >{gap:g}s "
          f"inside the hold, toll {toll_bp:.2f}bp, {days:g}d")
    print(f"{'symbol':<16} {'nAll':>5} {'medAll':>9} {'nCont':>6} {'medCont':>9} "
          f"{'delta':>9}  direction")
    lower = higher = 0
    for s in syms:
        ticks = load_ticks(s, days)
        if len(ticks) < 50:
            continue
        tape = Tape(ticks)
        wins = build(tape, window, tol)
        cont = [w for w in wins if w.max_gap <= gap]
        if len(wins) < 10 or len(cont) < 5:
            print(f"{s:<16} {len(wins):>5} {'--':>9} {len(cont):>6} {'--':>9} "
                  f"{'--':>9}  too few windows")
            continue
        ma = _quantile([w.gross for w in wins], 0.5)
        mc = _quantile([w.gross for w in cont], 0.5)
        d = mc - ma
        if d < 0:
            lower += 1
        else:
            higher += 1
        print(f"{s:<16} {len(wins):>5} {ma:>+8.2f}bp {len(cont):>6} {mc:>+8.2f}bp "
              f"{d:>+8.2f}bp  {'CONTINUOUS IS LOWER' if d < 0 else 'continuous is higher'}")
    tot = lower + higher
    print(f"\n    continuous-tape median LOWER on {lower} of {tot} symbols, "
          f"higher on {higher}")
    if tot:
        print("    the complement (dark-tape inflation is WETH-specific) "
              f"{'DIES' if lower > higher else 'SURVIVES'} on this count")
    print("    NOTE: a median over non-overlapping windows is a direction-free "
          "move statistic here --")
    print("    the sign of each row is the reversion rule's, so read the DELTA, "
          "not the level.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true",
                    help="sweep every symbol: all-windows median vs continuous-tape median")
    ap.add_argument("--top", type=int, default=14, help="symbols to scan, by tick count")
    ap.add_argument("--symbol", default="WETH-USDC")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--window", type=float, default=1800.0)
    ap.add_argument("--toll-bp", type=float, default=4.08)
    ap.add_argument("--tol", type=float, default=120.0,
                    help="max seconds a tick may be from the intended timestamp")
    ap.add_argument("--gap", type=float, default=300.0,
                    help="max inter-tick gap allowed inside the hold for CONTINUOUS")
    args = ap.parse_args()

    if args.scan:
        return scan(args.days, args.window, args.tol, args.gap, args.toll_bp, args.top)

    ticks = load_ticks(args.symbol, args.days)
    print(f"{args.symbol}: {len(ticks)} ticks over {args.days:g}d from {DB}")
    if len(ticks) < 10:
        print("not enough tape")
        return 1
    span = (ticks[-1][0] - ticks[0][0]) / 86400.0
    print(f"tape spans {span:.2f} days, "
          f"{time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(ticks[0][0]))} .. "
          f"{time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(ticks[-1][0]))}")

    tape = Tape(ticks)
    wins = build(tape, args.window, args.tol)
    grid = int((ticks[-1][0] - ticks[0][0] - 2 * args.window) // args.window) + 1
    print(f"grid slots {grid}, usable windows {len(wins)} "
          f"({100*len(wins)/grid:.1f}% of the grid had tape at all three points"
          f" within {args.tol:g}s)")

    print("\n================ TEST 0: REPRODUCE THE CLAIM (all windows) ================")
    report("ALL WINDOWS", wins, args.toll_bp, show_rows=True)

    print("\n================ TEST 1: OUT-OF-SAMPLE SPLIT ================")
    mid = (ticks[0][0] + ticks[-1][0]) / 2.0
    first = [w for w in wins if w.t < mid]
    second = [w for w in wins if w.t >= mid]
    report("FIRST HALF", first, args.toll_bp)
    report("SECOND HALF", second, args.toll_bp)
    if first and second:
        m1 = _quantile([w.gross for w in first], 0.5)
        m2 = _quantile([w.gross for w in second], 0.5)
        same = (m1 > 0) == (m2 > 0)
        print(f"\n    gross median first {m1:+.2f}bp  second {m2:+.2f}bp  -> "
              f"{'SIGN HOLDS' if same else 'SIGN FLIPS -- the pre-registered kill fired'}")

    print("\n================ TEST 4: TAPE CONTINUITY ================")
    cont = [w for w in wins if w.max_gap <= args.gap]
    dark = len(wins) - len(cont)
    print(f"    windows with NO inter-tick gap over {args.gap:g}s across the whole hold: "
          f"{len(cont)}/{len(wins)}  ({dark} discontinuous)")
    gaps = sorted(w.max_gap for w in wins if w.max_gap != float("inf"))
    if gaps:
        print(f"    max-gap-inside-hold distribution: median {_quantile(gaps,0.5):.0f}s  "
              f"p75 {_quantile(gaps,0.75):.0f}s  p95 {_quantile(gaps,0.95):.0f}s  "
              f"max {gaps[-1]:.0f}s")
    n_inf = sum(1 for w in wins if w.max_gap == float("inf"))
    print(f"    windows with ZERO ticks inside the hold: {n_inf}")
    report(f"CONTINUOUS TAPE ONLY (max gap <= {args.gap:g}s)", cont, args.toll_bp, show_rows=True)

    print("\n================ TEST 5: FILL-TO-FILL ================")
    print("    NOT PRICED. Every number above is TICK-TO-TICK: it assumes both legs fill")
    print("    at the tick price with zero slippage and zero fill risk. It is therefore an")
    print("    UPPER BOUND. Live slippage measured on this repo's own fills is ~14-21bp")
    print("    (the-entry-leg-is-the-whole-slippage-toll, fee-cost-is-round-trip-gas-and-")
    print("    nothing-else) and WETH has ZERO live fills, so its slippage is unmeasured at")
    print("    any clip (weth-has-never-been-filled-live).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
