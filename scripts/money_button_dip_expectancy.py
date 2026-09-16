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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--horizon-sec", type=float, default=1800.0)
    ap.add_argument("--cooldown-sec", type=float, default=1800.0,
                    help="one open position per symbol: ignore entries inside a live trade")
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
            while k < n and rows[k][0] <= ts + args.horizon_sec:
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
            busy_until = ts + args.horizon_sec

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


if __name__ == "__main__":
    raise SystemExit(main())
