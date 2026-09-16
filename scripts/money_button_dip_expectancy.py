"""Is there money in the money button? Priced on stored ticks, per token.

The supply question is settled: scripts/opportunity_precondition_census.py
shows the dip rule's precondition is available about 4.2 times an hour.
This script asks the only question that follows -- when the rule fires, does
the trade PAY, after the round trip that token actually costs on chain.

WHAT IS SIMULATED, and it is the rule's own exit, not a favourable one:

    entry   the tick at which the scheduler's dip test admits
            (drawdown off the 2h peak inside [MIN, MAX], >= 8 samples)
    target  take_profit_return = max(fee + MIN_NET_RETURN,
                                     |drawdown| * REVERSION_FRACTION)
            capped at the conservative reversion target the rule computes
            from the window MEDIAN (scheduler.py:1757-1762, 1799-1806)
    exit    whichever comes FIRST -- the target being touched, or the
            directive's own horizon elapsing (MONEY_BUTTON_HORIZON, 30m),
            in which case the trade is closed at the price then, win or
            lose. No stop is simulated, so a loser runs the full horizon:
            that is the PESSIMISTIC side of the assumption, and the
            optimistic side is that a touch is treated as a fill.

WHAT IS CHARGED. The per-token round trip measured on chain and recorded
in project memory as "the cost wall is a TOKEN choice, not a venue fee" --
both legs, gas and slippage included. This is the term that decides the
sign, which is why it is per token and not a single venue fee.

WHAT IS NOT MODELLED, said plainly rather than buried: the opportunity
precondition (this prices the price rule alone, so read it against the
45.6% arming rate, not as a per-hour trade count), the live risk gates,
position caps, and the fact that a touch inside a 13-second bar is not a
guaranteed fill. It is an expectancy estimate on real stored prices, not
a backtest of the executor.

Usage:
    python -X utf8 scripts/money_button_dip_expectancy.py
    python -X utf8 scripts/money_button_dip_expectancy.py --hours 168 --horizon-sec 1800
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

# Full round trips (both legs, gas + slippage), measured on chain.
TOKEN_ROUND_TRIP = {
    "WETH": 0.0011,
    "ETH": 0.0011,
    "CBETH": 0.0030,
    "USDC": 0.0011,
    "AERO": 0.0435,
    "KTA": 0.7498,
}
DEFAULT_ROUND_TRIP = 0.0070


def round_trip_for(symbol: str) -> float:
    base = symbol.split("-")[0].upper()
    return TOKEN_ROUND_TRIP.get(base, DEFAULT_ROUND_TRIP)


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


#: The set the 168h census found POSITIVE after cost, plus the majors whose
#: round trip is in the same class. Named here rather than inferred from the
#: result, so the sweep cannot select its own universe from its own outcome.
DEEP_POOLS = (
    "ETH-USDT", "ETH-USDC", "WETH-USDT", "WETH-USDC", "CBETH-USDC",
    "WBTC-USDC", "SOL-USDC", "LINK-USDC", "LDO-USDC", "AAVE-USDC",
    "OP-USDC", "CRV-USDC", "CAKE-USDC",
)


def simulate(
    ticks: Dict[str, List[Tuple[float, float]]],
    *,
    lo: float,
    hi: float,
    min_net: float,
    rev_frac: float,
    lookback: float,
    horizon_sec: float,
) -> Dict[str, List[float]]:
    per_symbol: Dict[str, List[float]] = defaultdict(list)
    for symbol, rows in ticks.items():
        fee = round_trip_for(symbol)
        n = len(rows)
        busy_until = 0.0
        for i in range(n):
            ts, price = rows[i]
            if ts < busy_until:
                continue
            j = i
            while j > 0 and rows[j - 1][0] >= ts - lookback:
                j -= 1
            window = [p for _, p in rows[j:i + 1] if p > 0.0]
            if len(window) < 8:
                continue
            peak = max(window)
            if peak <= 0.0:
                continue
            drawdown = (price - peak) / peak
            if drawdown > -lo or drawdown < -hi:
                continue

            median_price = float(np.median(np.asarray(window, dtype=np.float64)))
            reversion_target = min(max(median_price, price), peak)
            expected_return = (reversion_target - price) / max(price, 1e-9)
            if expected_return <= 0.0:
                continue
            if expected_return - fee < min_net:
                continue
            tp = max(fee + min_net, abs(drawdown) * rev_frac)
            tp = min(tp, expected_return)
            if tp - fee < min_net:
                continue
            target = price * (1.0 + tp)

            gross = None
            k = i + 1
            while k < n and rows[k][0] <= ts + horizon_sec:
                if rows[k][1] >= target:
                    gross = tp                      # target touched: the rule's own exit
                    break
                k += 1
            if gross is None:
                last_idx = min(k, n - 1)
                if last_idx <= i:
                    continue
                gross = (rows[last_idx][1] - price) / max(price, 1e-9)
            per_symbol[symbol].append(gross - fee)
            busy_until = ts + horizon_sec
    return per_symbol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--horizon-sec", type=float, default=1800.0)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the dip floor on the deep-pool universe, loading ticks once")
    ap.add_argument("--floors", default="0.0025,0.005,0.01,0.02,0.05")
    args = ap.parse_args()

    lo = float(os.getenv("MONEY_BUTTON_MIN_DRAWDOWN", "0.02"))
    hi = float(os.getenv("MONEY_BUTTON_MAX_DRAWDOWN", "0.05"))
    lookback = max(600.0, min(24 * 3600.0, float(os.getenv("MONEY_BUTTON_LOOKBACK_SEC", str(2 * 3600)))))
    min_net = float(os.getenv("MONEY_BUTTON_MIN_NET_RETURN", "0.005"))
    rev_frac = float(os.getenv("MONEY_BUTTON_REVERSION_FRACTION", "0.55"))

    ticks = load_ticks(args.hours)
    if not ticks:
        print("no ticks in market_stream")
        return 1

    if args.sweep:
        return sweep(ticks, args, hi=hi, rev_frac=rev_frac, lookback=lookback)

    per_symbol = simulate(ticks, lo=lo, hi=hi, min_net=min_net, rev_frac=rev_frac,
                          lookback=lookback, horizon_sec=args.horizon_sec)
    all_trades = [r for rs in per_symbol.values() for r in rs]
    if not all_trades:
        print("the dip rule admitted no priceable entry in this window")
        return 0

    def block(name: str, rs: List[float]) -> str:
        arr = np.asarray(rs, dtype=np.float64)
        wins = int((arr > 0).sum())
        return (f"    {name:<18} {len(arr):>5} {100.0*wins/len(arr):>5.0f}% "
                f"{float(arr.mean())*100:>+8.3f}% {float(arr.sum())*100:>+9.2f}%")

    print("=" * 78)
    print("MONEY BUTTON DIP -- EXPECTANCY PER ROUND TRIP, NET OF THE TOKEN'S OWN COST")
    print("=" * 78)
    print(f"  window {args.hours:.0f}h   horizon {args.horizon_sec/60:.0f}m   "
          f"drawdown [-{hi:.0%}, -{lo:.0%}]   min net {min_net:.2%}   reversion {rev_frac:.0%}")
    print()
    print(f"    {'symbol':<18} {'trades':>5} {'win':>6} {'mean net':>9} {'total net':>10}")
    print(block("ALL", all_trades))
    print()
    ranked = sorted(per_symbol.items(), key=lambda kv: -float(np.sum(kv[1])))
    for sym, rs in ranked[:14]:
        print(block(sym, rs))
    print()
    losers = [kv for kv in ranked if float(np.sum(kv[1])) < 0]
    if losers:
        print("  worst:")
        for sym, rs in losers[-5:]:
            print(block(sym, rs))
    return 0


def sweep(ticks, args, *, hi: float, rev_frac: float, lookback: float) -> int:
    """[3c411c08] -- the affordable dip is a function of pool depth.

    One constant (2-5%) is applied today to tokens whose round trip spans
    WETH 0.11% to KTA 74.98%. Sweeping the FLOOR on the deep-pool universe
    asks whether a threshold exists that is both profitable after cost and
    frequent enough to answer the cadence ask. Trades/day is printed beside
    the mean on every row because either number alone is misleading.

    MIN_NET_RETURN is scaled with the floor rather than held at 0.5%: left
    at its default it would itself veto every small-dip entry, and the
    sweep would be measuring that veto instead of the threshold.
    """
    deep = {s: r for s, r in ticks.items() if s in DEEP_POOLS}
    days = max(args.hours / 24.0, 1e-9)
    print("=" * 86)
    print("DIP FLOOR SWEEP -- DEEP POOLS ONLY  [3c411c08]")
    print("=" * 86)
    print(f"  window {args.hours:.0f}h ({days:.1f}d)   horizon {args.horizon_sec/60:.0f}m   "
          f"ceiling -{hi:.0%}   universe {len(deep)} of {len(ticks)} symbols")
    print(f"  {', '.join(sorted(deep))}")
    print()
    print(f"    {'floor':>7} {'min_net':>8} {'trades':>7} {'per day':>8} {'win':>6} {'mean net':>10} {'total net':>11}")
    best = None
    for raw in args.floors.split(","):
        floor = float(raw)
        mn = max(0.0005, floor * 0.25)
        per_symbol = simulate(deep, lo=floor, hi=hi, min_net=mn, rev_frac=rev_frac,
                              lookback=lookback, horizon_sec=args.horizon_sec)
        rs = [r for v in per_symbol.values() for r in v]
        if not rs:
            print(f"    {floor:>6.2%} {mn:>7.2%} {0:>7} {0.0:>8.2f} {'--':>6} {'--':>10} {'--':>11}")
            continue
        arr = np.asarray(rs, dtype=np.float64)
        total = float(arr.sum())
        row = (floor, mn, len(arr), len(arr) / days, 100.0 * float((arr > 0).sum()) / len(arr),
               float(arr.mean()) * 100.0, total * 100.0)
        print(f"    {row[0]:>6.2%} {row[1]:>7.2%} {row[2]:>7} {row[3]:>8.2f} {row[4]:>5.0f}% "
              f"{row[5]:>+9.3f}% {row[6]:>+10.2f}%")
        if best is None or total > best[1]:
            best = (row, total, per_symbol)
    print()
    if best is None:
        print("  NO floor produced a priceable trade on the deep-pool universe.")
        return 0
    row, total, per_symbol = best
    print(f"  BEST TOTAL over {args.hours:.0f}h: floor {row[0]:.2%}  "
          f"-> {row[2]} trades ({row[3]:.2f}/day), {row[4]:.0f}% win, mean {row[5]:+.3f}%, total {row[6]:+.2f}%")
    print(f"  per-trade mean after cost is {'POSITIVE' if row[5] > 0 else 'NEGATIVE'}.")
    if row[2] < 30:
        print(f"  SAMPLE {row[2]} < 30: this is a LEAD REQUIRING MORE DATA, not an edge.")
    outliers = {s: v for s, v in per_symbol.items() if any(abs(x) > 0.2 for x in v)}
    if outliers:
        print("  outliers left in (|return| > 20%), reported not filtered:")
        for s, v in outliers.items():
            print(f"    {s:<14} {['%+.1f%%' % (x*100) for x in v if abs(x) > 0.2]}")
    print("  per symbol at the best floor:")
    for s, v in sorted(per_symbol.items(), key=lambda kv: -float(np.sum(kv[1]))):
        a = np.asarray(v, dtype=np.float64)
        print(f"    {s:<14} {len(a):>4} {100.0*float((a>0).sum())/len(a):>5.0f}% "
              f"{float(a.mean())*100:>+8.3f}% {float(a.sum())*100:>+9.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
