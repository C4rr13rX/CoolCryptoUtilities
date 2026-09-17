"""Census of the DIP rule -- the money_button the OPERATOR described.

There are two different strategies in this repo both carrying
``strategy_id = "money_button"``, and they trade in opposite directions:

    trading/strategies/money_button.py:278   MOMENTUM. Refuses to enter
        unless the 5/10/30-minute returns are ALL positive. Buys strength.
        This is what scripts/money_button_gate_census.py measures.

    trading/scheduler.py:1743                DIP. Enters on a 2-5% drawdown
        off the 2-hour peak and targets a partial reversion to the median.
        Buys weakness. This is the one wired to the live candidate list
        (scheduler.py:794, 808) and the one the operator described.

The momentum census reports ~2 fires in 168h and is quoted as "money_button
does not fire". That number is about the OTHER strategy. This script gives
the dip rule its own census so the two are never again read as one.

No gate is loosened here: every threshold below is the scheduler's own
default, read from the same environment variable with the same fallback.

ONE GATE IS NOT REPLAYED, and it makes this an UPPER BOUND rather than a
prediction: scheduler.py:1666-1669 also requires
``self._opportunity_bias[symbol].kind == "buy-low"`` when
MONEY_BUTTON_REQUIRE_OPPORTUNITY=1 (the default). That bias is live
scheduler state and is not reconstructible from stored ticks, so the fire
counts here are what the PRICE RULE alone admits. Production fires at most
this often and probably less.

Usage:
    python -X utf8 scripts/money_button_dip_census.py
    python -X utf8 scripts/money_button_dip_census.py --hours 168 --fee 0.0035
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# [cae445dc] LOAD PRODUCTION'S ENVIRONMENT BEFORE READING A SINGLE THRESHOLD.
#
# This script used to import no repo module, so services/env_loader never ran,
# so every os.getenv below returned its DEFAULT. services/env_loader.py:377
# sets MONEY_BUTTON_MIN_DRAWDOWN to 0.005 -- so this census replayed a 2% dip
# floor while production admits at 0.5%, a rule 4x deeper than the one on the
# money path, and nothing in the output said so.
#
# MEASURED pass 442: MIN_DRAWDOWN is the ONLY one of this script's eight
# parameters that disagreed. env_loader leaves MAX_DRAWDOWN, MIN_NET_RETURN,
# REVERSION_FRACTION, SLOPE_FLOOR, SLOPE_WINDOW_SEC, LOOKBACK_SEC and
# TRADE_FEE_RATE unset, so those defaults already matched. One parameter --
# but the one that decides how many candidates EXIST.
#
# The resolved values are printed in the header below. A census that does not
# say which rule it replayed is not evidence.
try:
    import services.env_loader  # noqa: F401  (import side effect: loads .env)
    _ENV_LOADED = True
except Exception as exc:  # pragma: no cover - diagnostic path
    _ENV_LOADED = False
    _ENV_ERROR = str(exc)

DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

# Per-token round-trip cost, measured on chain and recorded in the project
# memory as "the cost wall is a TOKEN choice, not a venue fee". These are
# full round trips (both legs, gas + slippage), not one-way fees.
TOKEN_ROUND_TRIP = {
    "WETH": 0.0011,
    "ETH": 0.0011,
    "CBETH": 0.0030,
    "USDC": 0.0011,
    "AERO": 0.0435,
    "KTA": 0.7498,
}
DEFAULT_ROUND_TRIP = 0.0070


def base_of(symbol: str) -> str:
    return symbol.split("-", 1)[0].upper()


def round_trip_for(symbol: str) -> float:
    return TOKEN_ROUND_TRIP.get(base_of(symbol), DEFAULT_ROUND_TRIP)


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
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee the GATE uses; default TRADE_FEE_RATE or 0.003")
    args = ap.parse_args()

    # Every threshold is the scheduler's own default (scheduler.py:1672-1793).
    fee = args.fee if args.fee is not None else float(os.getenv("TRADE_FEE_RATE", "0.003"))
    lookback = max(600.0, min(24 * 3600.0, float(os.getenv("MONEY_BUTTON_LOOKBACK_SEC", str(2 * 3600)))))
    min_dd = float(os.getenv("MONEY_BUTTON_MIN_DRAWDOWN", "0.02"))
    max_dd = float(os.getenv("MONEY_BUTTON_MAX_DRAWDOWN", "0.05"))
    min_net = float(os.getenv("MONEY_BUTTON_MIN_NET_RETURN", "0.005"))
    slope_win = max(300.0, min(4 * 3600.0, float(os.getenv("MONEY_BUTTON_SLOPE_WINDOW_SEC", "1800"))))
    slope_floor = float(os.getenv("MONEY_BUTTON_SLOPE_FLOOR", "-0.00025"))
    rev_frac = max(0.1, min(1.0, float(os.getenv("MONEY_BUTTON_REVERSION_FRACTION", "0.55"))))
    horizon_sec = 30.0 * 60.0  # MONEY_BUTTON_HORIZON default "30m"

    ticks = load_ticks(args.hours)
    census: Dict[str, int] = defaultdict(int)
    fires: List[dict] = []
    evaluated = 0

    for symbol, series in ticks.items():
        times = [t for t, _ in series]
        prices = [p for _, p in series]
        n = len(series)
        for i in range(n):
            evaluated += 1
            now_ts, last_price = series[i]
            lo = now_ts - lookback
            j = i
            while j > 0 and times[j - 1] >= lo:
                j -= 1
            recent = prices[j:i + 1]
            if len(recent) < 8:
                census["too_few_samples"] += 1
                continue
            peak = max(recent)
            if peak <= 0:
                census["no_price"] += 1
                continue
            drawdown = (last_price - peak) / peak
            if drawdown > -min_dd:
                census["dip_too_shallow"] += 1
                continue
            if drawdown < -max_dd:
                census["dip_too_deep"] += 1
                continue
            median_price = float(np.median(np.asarray(recent, dtype=np.float64)))
            target = min(max(median_price, last_price), peak)
            expected = (target - last_price) / max(last_price, 1e-9)
            if expected <= 0.0:
                census["no_reversion_room"] += 1
                continue
            if expected - fee < min_net:
                census["edge_below_cost"] += 1
                continue
            # slope guard (scheduler.py:1766-1786)
            slo = now_ts - slope_win
            k = i
            while k > 0 and times[k - 1] >= slo:
                k -= 1
            sl_t = times[k:i + 1]
            sl_p = prices[k:i + 1]
            if len(sl_t) >= 6:
                rel = (np.asarray(sl_t) - sl_t[-1]) / 60.0
                lp = np.log(np.clip(np.asarray(sl_p), 1e-9, 1e9))
                slope = 0.0
                if not np.allclose(rel, rel[0]):
                    with np.errstate(all="ignore"):
                        slope = float(np.polyfit(rel, lp, 1)[0])
                if slope < slope_floor:
                    census["falling_knife"] += 1
                    continue
            tp = min(max(fee + min_net, abs(drawdown) * rev_frac), expected)
            if tp - fee < min_net:
                census["target_below_cost"] += 1
                continue
            census["(candidate produced)"] += 1
            fires.append({
                "symbol": symbol, "i": i, "ts": now_ts, "entry": last_price,
                "target": last_price * (1.0 + tp), "tp": tp, "dd": drawdown,
            })

    # ---- forward outcome: hold until target or 30m horizon -------------
    resolved: List[dict] = []
    for f in fires:
        series = ticks[f["symbol"]]
        entry, target = f["entry"], f["target"]
        exit_price, hit, held = None, False, 0.0
        for t2, p2 in series[f["i"] + 1:]:
            held = t2 - f["ts"]
            if p2 >= target:
                exit_price, hit = target, True
                break
            if held >= horizon_sec:
                exit_price = p2
                break
        if exit_price is None:
            continue
        gross = (exit_price - entry) / entry
        rt = round_trip_for(f["symbol"])
        resolved.append({**f, "gross": gross, "net": gross - rt,
                         "hit": hit, "held": held, "rt": rt})

    print("money_button DIP census -- the rule at trading/scheduler.py:1743")
    print("%.0fh of stored ticks, gate fee %.4f, dip window %.0f%%-%.0f%%, horizon 30m"
          % (args.hours, fee, min_dd * 100, max_dd * 100))
    print("%d symbols, %d evaluations" % (len(ticks), evaluated))

    # [cae445dc] SAY WHICH RULE THIS REPLAYED. A number without its parameters
    # is not reproducible, and this script silently replayed a 2% dip floor
    # against a production that runs 0.5% until pass 442.
    print("\nRESOLVED THRESHOLDS -- this is the rule that was replayed:")
    if _ENV_LOADED:
        print("  production environment: LOADED (services/env_loader)")
    else:
        print("  production environment: *** NOT LOADED (%s) ***" % _ENV_ERROR)
        print("  every value below is an os.getenv DEFAULT and may not be production's.")
    for name, value, env_key in (
        ("min_drawdown", min_dd, "MONEY_BUTTON_MIN_DRAWDOWN"),
        ("max_drawdown", max_dd, "MONEY_BUTTON_MAX_DRAWDOWN"),
        ("min_net_return", min_net, "MONEY_BUTTON_MIN_NET_RETURN"),
        ("reversion_fraction", rev_frac, "MONEY_BUTTON_REVERSION_FRACTION"),
        ("slope_floor", slope_floor, "MONEY_BUTTON_SLOPE_FLOOR"),
        ("slope_window_sec", slope_win, "MONEY_BUTTON_SLOPE_WINDOW_SEC"),
        ("lookback_sec", lookback, "MONEY_BUTTON_LOOKBACK_SEC"),
        ("fee_rate", fee, "TRADE_FEE_RATE"),
    ):
        origin = "from env" if os.getenv(env_key) is not None else "DEFAULT (env unset)"
        print("  %-20s %-12s %s" % (name, ("%g" % value), origin))

    print("\nNOTE: the MONEY_BUTTON_REQUIRE_OPPORTUNITY buy-low bias gate is NOT")
    print("      replayed (live scheduler state) -- these are UPPER BOUNDS.\n")

    print("%-26s %9s %8s" % ("gate", "declines", "share"))
    print("-" * 46)
    for gate, count in sorted(census.items(), key=lambda kv: -kv[1]):
        print("%-26s %9d %7.2f%%" % (gate, count, 100.0 * count / max(evaluated, 1)))

    fired = census.get("(candidate produced)", 0)
    print("\nfired: %d of %d evaluations (%.3f%%)" % (fired, evaluated, 100.0 * fired / max(evaluated, 1)))

    per_sym = defaultdict(int)
    for f in fires:
        per_sym[f["symbol"]] += 1
    if per_sym:
        print("\ntop symbols by fire count:")
        for sym, c in sorted(per_sym.items(), key=lambda kv: -kv[1])[:12]:
            print("  %-22s %5d   round-trip cost %.2f%%" % (sym, c, round_trip_for(sym) * 100))

    if not resolved:
        print("\nno fire resolved inside the recorded data.")
        return 0

    print("\n--- outcome, held to target or 30m, net of the REAL per-token round trip ---")
    wins = [r for r in resolved if r["net"] > 0]
    nets = [r["net"] for r in resolved]
    print("%d resolved, %dW/%dL (%.1f%% win), mean net %+.4f%%, total net %+.4f%% of notional"
          % (len(resolved), len(wins), len(resolved) - len(wins),
             100.0 * len(wins) / len(resolved),
             100.0 * statistics.fmean(nets), 100.0 * sum(nets)))
    print("    target hit: %d of %d (%.1f%%); mean gross %+.4f%%"
          % (sum(1 for r in resolved if r["hit"]), len(resolved),
             100.0 * sum(1 for r in resolved if r["hit"]) / len(resolved),
             100.0 * statistics.fmean([r["gross"] for r in resolved])))
    print("    median hold %.1f min" % (statistics.median([r["held"] for r in resolved]) / 60.0))

    print("\n--- the same signal, split by TOKEN (cost is a token choice) ---")
    print("%-22s %6s %6s %10s %10s %9s" % ("symbol", "n", "win%", "mean gross", "mean net", "cost"))
    print("-" * 70)
    by_sym = defaultdict(list)
    for r in resolved:
        by_sym[r["symbol"]].append(r)
    rows = sorted(by_sym.items(), key=lambda kv: -sum(x["net"] for x in kv[1]))
    for sym, rs in rows[:15]:
        w = sum(1 for x in rs if x["net"] > 0)
        print("%-22s %6d %5.0f%% %+9.3f%% %+9.3f%% %8.2f%%"
              % (sym, len(rs), 100.0 * w / len(rs),
                 100.0 * statistics.fmean([x["gross"] for x in rs]),
                 100.0 * statistics.fmean([x["net"] for x in rs]),
                 rs[0]["rt"] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
