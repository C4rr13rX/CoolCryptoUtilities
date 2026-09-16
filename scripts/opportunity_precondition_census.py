"""Does the money_button DIP rule ever see its precondition?

scripts/money_button_dip_census.py measures the dip PRICE rule and says so
explicitly: it is an UPPER BOUND, because one gate is not replayed --

    trading/scheduler.py:1665-1668
        require_opportunity = MONEY_BUTTON_REQUIRE_OPPORTUNITY (default "1")
        opportunity = self._opportunity_bias.get(symbol)
        if require_opportunity and (not opportunity or opportunity.kind != "buy-low"):
            return None

This script replays that gate, so the two numbers can be read together.

THE SUPPLY SIDE, and why it is suspected of starving the rule:

  * trading/bot.py:3641 is the ONLY producer. Once per processed sample it
    calls OpportunityTracker.evaluate(symbol, price_history) and, on a
    signal, scheduler.record_opportunity(signal).

  * trading/opportunity.py:51-53 debounces:

        existing = self._last_signal.get(symbol)
        if existing and existing.kind == kind and abs(existing.zscore - zscore) < 0.2:
            return None

    _last_signal is written ONLY when a signal is emitted, so the reference
    point never decays. A symbol sitting in a sustained dip holds its
    z-score roughly still -- which is exactly what a mean-reverting 2-5%
    drawdown looks like -- so the tracker emits ONCE and then goes quiet
    for the rest of the dip.

  * trading/scheduler.py:1505-1506 DELETES the bias once it is 1800s old.

Put together: 30 minutes into a dip the consumer's copy is deleted and the
producer refuses to re-supply it. The rule is then dead on that symbol for
the remainder of the very window it was built to trade.

WHAT IS COUNTED. For every tick of every symbol this replays, in lockstep:

    (a) the dip PRICE condition  -- drawdown off the 2h peak inside
        [MONEY_BUTTON_MIN_DRAWDOWN, MONEY_BUTTON_MAX_DRAWDOWN], >= 8
        samples in the lookback, the scheduler's own defaults;
    (b) whether a FRESH (< bias_ttl) buy-low bias exists at that instant,
        under the CURRENT debounce;
    (c) the same under a TIME-BOUNDED debounce -- identical except that a
        same-kind signal is re-emitted when the last one is older than the
        consumer's TTL.

The headline is the count of ticks where (a) holds: under (b) the rule can
fire, under "not (b)" it is refused for a reason that has nothing to do
with the price. (c) is the counterfactual for the proposed change.

The baseline uses the REAL OpportunityTracker class, not a copy of it.

Usage:
    python -X utf8 scripts/opportunity_precondition_census.py
    python -X utf8 scripts/opportunity_precondition_census.py --hours 168 --window 240
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.opportunity import OpportunityTracker  # noqa: E402

DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")


def load_ticks(hours: float) -> Dict[str, List[Tuple[float, float]]]:
    conn = sqlite3.connect(DB_PATH, uri=True)
    try:
        newest = list(conn.execute("SELECT MAX(ts) FROM market_stream"))[0][0]
        if not newest:
            return {}
        cutoff = float(newest) - hours * 3600.0
        rows = conn.execute(
            "SELECT symbol, ts, price FROM market_stream "
            "WHERE ts >= ? AND price > 0 ORDER BY symbol, ts",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for symbol, ts, price in rows:
        out[str(symbol)].append((float(ts), float(price)))
    return out


def dip_condition(window: List[Tuple[float, float]], lo: float, hi: float) -> bool:
    """The scheduler's own drawdown test, scheduler.py:1737-1745."""
    prices = [p for _, p in window if p > 0.0]
    if len(prices) < 8:
        return False
    peak = max(prices)
    if peak <= 0.0:
        return False
    drawdown = (prices[-1] - peak) / peak
    return not (drawdown > -lo or drawdown < -hi)


def zscore_of(series: np.ndarray, min_points: int, threshold: float) -> Optional[Tuple[str, float]]:
    """The tracker's arithmetic, with no debounce -- what COULD be emitted."""
    if series.size < min_points:
        return None
    mean_price = float(np.mean(series))
    std_price = float(np.std(series))
    if std_price < 1e-9:
        return None
    current = float(series[-1])
    z = (current - mean_price) / std_price
    if abs(z) < threshold:
        return None
    return ("buy-low" if z < 0 else "sell-high", float(z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--window", type=int, default=120, help="bot._brain_window: price history depth")
    ap.add_argument("--bias-ttl", type=float, default=1800.0, help="scheduler.py:1505 expiry")
    args = ap.parse_args()

    lo = float(os.getenv("MONEY_BUTTON_MIN_DRAWDOWN", "0.02"))
    hi = float(os.getenv("MONEY_BUTTON_MAX_DRAWDOWN", "0.05"))
    lookback = max(600.0, min(24 * 3600.0, float(os.getenv("MONEY_BUTTON_LOOKBACK_SEC", str(2 * 3600)))))

    probe = OpportunityTracker()
    min_points, threshold = probe.min_points, probe.zscore_threshold

    ticks = load_ticks(args.hours)
    if not ticks:
        print("no ticks in market_stream")
        return 1

    dip_ticks = 0           # (a) the price rule admits this tick
    armed_now = 0           # (b) ... and a fresh bias exists under the CURRENT debounce
    armed_fixed = 0         # (c) ... and one exists under a TIME-BOUNDED debounce
    emits_now = 0
    emits_fixed = 0
    suppressed_same_kind = 0    # debounce refusals while the consumer had NOTHING
    per_symbol: Dict[str, Tuple[int, int, int]] = {}

    for symbol, rows in sorted(ticks.items()):
        if len(rows) < min_points:
            continue
        hist: deque = deque(maxlen=args.window)
        look: deque = deque()
        tracker = OpportunityTracker()          # the real debounce
        bias_now: Optional[Tuple[str, float]] = None      # (kind, emitted_ts)
        bias_fixed: Optional[Tuple[str, float]] = None
        last_fixed: Optional[Tuple[str, float, float]] = None   # (kind, z, ts)
        s_dip = s_now = s_fixed = 0

        for ts, price in rows:
            hist.append(price)
            look.append((ts, price))
            while look and look[0][0] < ts - lookback:
                look.popleft()

            series = np.asarray(list(hist), dtype=np.float64)

            # --- producer, current code: the real class decides ---
            sig = tracker.evaluate(symbol, series)
            if sig is not None:
                emits_now += 1
                bias_now = (sig.kind, ts)

            # --- producer, time-bounded debounce ---
            cand = zscore_of(series, min_points, threshold)
            if cand is not None:
                kind, z = cand
                stale = last_fixed is None or (ts - last_fixed[2]) >= args.bias_ttl
                moved = last_fixed is None or last_fixed[0] != kind or abs(last_fixed[1] - z) >= 0.2
                if stale or moved:
                    emits_fixed += 1
                    last_fixed = (kind, z, ts)
                    bias_fixed = (kind, ts)
                elif bias_now is None or (ts - bias_now[1]) > args.bias_ttl:
                    # the debounce refused while the consumer was holding nothing
                    suppressed_same_kind += 1

            # --- consumer, scheduler.py:1505: expire the bias ---
            if bias_now is not None and (ts - bias_now[1]) > args.bias_ttl:
                bias_now = None
            if bias_fixed is not None and (ts - bias_fixed[1]) > args.bias_ttl:
                bias_fixed = None

            if dip_condition(list(look), lo, hi):
                s_dip += 1
                if bias_now is not None and bias_now[0] == "buy-low":
                    s_now += 1
                if bias_fixed is not None and bias_fixed[0] == "buy-low":
                    s_fixed += 1

        dip_ticks += s_dip
        armed_now += s_now
        armed_fixed += s_fixed
        if s_dip:
            per_symbol[symbol] = (s_dip, s_now, s_fixed)

    print("=" * 74)
    print("MONEY_BUTTON PRECONDITION CENSUS -- does the dip rule ever see a buy-low bias?")
    print("=" * 74)
    print(f"  window {args.hours:.0f}h   symbols {len(ticks)}   brain_window {args.window}   bias TTL {args.bias_ttl:.0f}s")
    print(f"  tracker: min_points {min_points}, |z| threshold {threshold}")
    print(f"  dip rule: drawdown in [-{hi:.0%}, -{lo:.0%}] off the {lookback/3600:.0f}h peak")
    print()
    print(f"  ticks where the DIP PRICE RULE admits          : {dip_ticks}")
    pct_now = 100.0 * armed_now / dip_ticks if dip_ticks else 0.0
    pct_fix = 100.0 * armed_fixed / dip_ticks if dip_ticks else 0.0
    print(f"  ... and a fresh buy-low bias exists  (TODAY)   : {armed_now}  ({pct_now:.1f}%)")
    print(f"  ... and one exists (TIME-BOUNDED DEBOUNCE)     : {armed_fixed}  ({pct_fix:.1f}%)")
    print()
    print(f"  opportunity signals emitted, today             : {emits_now}")
    print(f"  opportunity signals emitted, time-bounded      : {emits_fixed}")
    print(f"  debounce refusals while consumer held NOTHING  : {suppressed_same_kind}")
    print()
    if dip_ticks:
        print("  top symbols by dip ticks   (dip / armed today / armed fixed)")
        for sym, (d, n, f) in sorted(per_symbol.items(), key=lambda kv: -kv[1][0])[:12]:
            print(f"    {sym:<18} {d:>6} {n:>8} {f:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
