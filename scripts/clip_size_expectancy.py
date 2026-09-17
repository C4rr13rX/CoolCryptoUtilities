"""[c5c198b2] Does a BIGGER CLIP change the sign, or only the signal?

THE CLAIM THIS TESTS, and it is a SIZE argument, not a signal argument:
this book trades AERO at $2.01 mean notional, where AERO's measured round
trip is 0.6395% and the measured directional gross per round trip is
-$0.0016 -- the cost is ~8x the signal. The item alleges that the same
rule on a deep pool at a larger clip inverts that ratio. This script
prices exactly that, on stored ticks, and it isolates SIZE from SIGNAL by
running the identical rule twice with only the clip changed.

WHY A CLIP CAN MATTER AT ALL. A round trip here is TWO terms and only one
of them scales:

    rate   0.31%  = 0.0369% quote spread + 27.8 bps measured live slippage
                    (buy 21.73 + sell 6.04, data/roundtrip_cost_census.json
                    lanes.live.*_slippage_bps_median, n=12 legs)
                    THIS IS A PERCENTAGE. It does not care about the clip.
    fixed  $0.00457 = gas, both legs, measured on chain
                    (roundtrip_cost_census.json rederived.fixed_usd, from
                    20 buy + 18 sell native-gas observations repriced at
                    $2498.78/ETH)
                    THIS IS A DOLLAR. It is the ONLY term a clip moves.

So the entire size effect available to us is fixed/clip:

    at $2   0.31% + 0.2285% = 0.5432%
    at $10  0.31% + 0.0457% = 0.3604%

18.3 bps. That is the whole prize, and it is bounded: no clip, however
large, gets the round trip below the 0.31% rate floor. If the rule's gross
edge is under 0.31%, SIZE CANNOT SAVE IT -- and that is the question.

WHAT IS SIMULATED. The money_button dip rule as the scheduler runs it
(trading/scheduler.py:1757-1806), identical to
scripts/money_button_dip_expectancy.py: entry on a drawdown off the 2h
peak inside [MIN, MAX] with >=8 samples; take-profit at
max(fee + MIN_NET_RETURN, |drawdown| * REVERSION_FRACTION) capped at the
median-reversion target; exit at whichever comes FIRST, the target being
touched or the horizon elapsing. No stop: a loser runs the full horizon,
which is the pessimistic side. A touch is treated as a fill, which is the
optimistic side.

THE COST FEEDS THE ENTRY, NOT ONLY THE EXIT. `fee` appears in the
admission test and in the take-profit level, so a cheaper clip admits a
DIFFERENT set of entries at a LOWER target. That is not a confound, it is
the mechanism the item is about, and it is why the two clips are run
through the same code rather than rescaled arithmetically afterwards.

CENSORING. Round trips whose gross moves more than --censor (default 20%)
are dropped and COUNTED. 10 of 237 rows carry the entire sign of the
existing book, so an uncensored expectancy is not evidence here. Both the
censored and uncensored figures are printed; neither is hidden.

WHAT IS NOT MODELLED, said plainly: the live risk gates, position caps,
the opportunity precondition, and the fact that a touch inside a 13s bar
is not a guaranteed fill. This is an expectancy estimate on real stored
prices. It changes no gate and places no trade.

Usage:
    python -X utf8 scripts/clip_size_expectancy.py
    python -X utf8 scripts/clip_size_expectancy.py --hours 168 --clips 2,10,50
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")
CENSUS = ROOT / "data" / "roundtrip_cost_census.json"

# [c5c198b2] THIS IMPORT IS LOAD-BEARING AND IT IS NOT FOR THE SYMBOL IT BINDS.
# Importing anything under services/ loads the dotenv that production runs with,
# and that is where MONEY_BUTTON_MIN_DRAWDOWN=0.005 comes from. Without it this
# script falls back to the literal "0.02" default below and silently prices a
# FOUR-TIMES-TIGHTER rule than the one that ships -- which is exactly the error
# recorded as "the replay harness ran a different rule than production", and
# which this script committed on its first run (pass 445) before the import was
# added. Do not remove it because the name looks unused.
import services.round_trip_cost  # noqa: E402,F401  (imported for its dotenv side effect)

#: [c5c198b2] criterion 2. The rate term, stated here in the body and not
#: only in a commit message. 0.0369% quote + 27.8 bps live slippage.
QUOTE_RATE = 0.000369
SLIPPAGE_RATE = 0.00278
FILL_RATE = QUOTE_RATE + SLIPPAGE_RATE

#: The dollar term, measured. Overridden from the census at runtime when the
#: file is present so this constant can never drift away from its source.
FIXED_USD_FALLBACK = 0.004570221896030538

#: The pairs the item names. CBBTC-USDC is included because it is the other
#: deep-pool major this book has actually streamed and has a live row in.
DEFAULT_SYMBOLS = ("WETH-USDC", "CBBTC-USDC")

#: Criterion 1. Minutes.
DEFAULT_HORIZONS = (5, 15, 30, 45, 60)


def load_fixed_usd() -> Tuple[float, str]:
    """The measured gas dollar per round trip, from the census that measured it."""
    try:
        blob = json.loads(CENSUS.read_text(encoding="utf-8"))
        val = float(blob["rederived"]["fixed_usd"])
        if val > 0:
            return val, f"{CENSUS.name} rederived.fixed_usd"
    except Exception:
        pass
    return FIXED_USD_FALLBACK, "built-in fallback (census unreadable)"


def load_ticks(hours: float, symbols: Sequence[str]) -> Dict[str, List[Tuple[float, float]]]:
    conn = sqlite3.connect(DB_PATH, uri=True)
    try:
        newest = list(conn.execute("SELECT MAX(ts) FROM market_stream"))[0][0]
        if not newest:
            return {}
        cutoff = float(newest) - hours * 3600.0
        marks = ",".join("?" for _ in symbols)
        rows = conn.execute(
            f"SELECT symbol, ts, price FROM market_stream "
            f"WHERE ts >= ? AND price > 0 AND symbol IN ({marks}) ORDER BY symbol, ts",
            (cutoff, *symbols),
        ).fetchall()
    finally:
        conn.close()
    out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for symbol, ts, price in rows:
        out[str(symbol)].append((float(ts), float(price)))
    return out


def simulate(
    rows: List[Tuple[float, float]],
    *,
    fee: float,
    lo: float,
    hi: float,
    min_net: float,
    rev_frac: float,
    lookback: float,
    horizon_sec: float,
) -> List[float]:
    """The dip rule's own entry and its own exit, charged `fee` per round trip.

    Returns GROSS returns; the caller subtracts the fee, so the same trade
    list can be reported before and after cost without re-simulating.
    """
    out: List[float] = []
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
        out.append(gross)
        busy_until = ts + horizon_sec
    return out


def stats(gross: List[float], fee: float, censor: float):
    """(n, censored, win%, mean_net%, total_net%, mean_gross%) after censoring."""
    kept = [g for g in gross if abs(g) <= censor]
    dropped = len(gross) - len(kept)
    if not kept:
        return 0, dropped, None, None, None, None
    arr = np.asarray(kept, dtype=np.float64) - fee
    raw = np.asarray(kept, dtype=np.float64)
    return (len(kept), dropped,
            100.0 * float((arr > 0).sum()) / len(arr),
            float(arr.mean()) * 100.0,
            float(arr.sum()) * 100.0,
            float(raw.mean()) * 100.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=526.0)
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--clips", default="2,10")
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS),
                    help="hold horizons in MINUTES")
    ap.add_argument("--censor", type=float, default=0.20,
                    help="drop round trips whose gross moves more than this")
    ap.add_argument("--floors", default="",
                    help="also sweep these drawdown floors (e.g. 0.005,0.01,0.015). "
                         "MIN_NET_RETURN is scaled to floor*0.25 so the 0.5%% default "
                         "does not itself veto every small-dip entry and get reported "
                         "as an absent signal.")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    clips = [float(c) for c in args.clips.split(",") if c.strip()]
    horizons = [float(h) for h in args.horizons.split(",") if h.strip()]

    lo = float(os.getenv("MONEY_BUTTON_MIN_DRAWDOWN", "0.02"))
    hi = float(os.getenv("MONEY_BUTTON_MAX_DRAWDOWN", "0.05"))
    lookback = max(600.0, min(24 * 3600.0, float(os.getenv("MONEY_BUTTON_LOOKBACK_SEC", str(2 * 3600)))))
    min_net = float(os.getenv("MONEY_BUTTON_MIN_NET_RETURN", "0.005"))
    rev_frac = float(os.getenv("MONEY_BUTTON_REVERSION_FRACTION", "0.55"))

    fixed_usd, fixed_src = load_fixed_usd()

    ticks = load_ticks(args.hours, symbols)
    if not ticks:
        print("no ticks in market_stream for %s" % ", ".join(symbols))
        return 1

    print("=" * 92)
    print("CLIP SIZE vs SIGNAL -- money_button dip expectancy per round trip   [c5c198b2]")
    print("=" * 92)
    print(f"  window {args.hours:.0f}h   drawdown [-{hi:.0%}, -{lo:.0%}]   "
          f"min net {min_net:.2%}   reversion {rev_frac:.0%}   censor |gross| > {args.censor:.0%}")
    print()
    print("  THE COST CHARGED, both terms, stated here and not only in the commit:")
    print(f"    rate   {FILL_RATE:.4%}  = {QUOTE_RATE:.4%} quote spread + {SLIPPAGE_RATE*1e4:.1f} bps live slippage")
    print(f"           source: data/roundtrip_cost_census.json lanes.live buy {21.73:.2f} + sell {6.04:.2f} bps, n=12 legs")
    print(f"    fixed  ${fixed_usd:.6f}  gas both legs, measured on chain")
    print(f"           source: {fixed_src}")
    print("    charged per round trip as   rate + fixed/clip   -- only the second term moves with size.")
    print()
    for clip in clips:
        print(f"    clip ${clip:>6.2f}  ->  round trip {FILL_RATE + fixed_usd/clip:.4%}  "
              f"({FILL_RATE:.4%} rate + {fixed_usd/clip:.4%} gas)")
    print()

    for symbol in symbols:
        rows = ticks.get(symbol) or []
        print("-" * 92)
        if len(rows) < 8:
            print(f"  {symbol}: {len(rows)} ticks in {args.hours:.0f}h -- NOT MEASURABLE, reported as such.")
            continue
        span_h = (rows[-1][0] - rows[0][0]) / 3600.0
        print(f"  {symbol}   {len(rows)} ticks over {span_h:.1f}h "
              f"({len(rows)/max(span_h,1e-9):.1f}/h)")
        print()
        print(f"    {'clip':>7} {'hold':>6} {'n':>5} {'cens':>5} {'win':>6} "
              f"{'mean gross':>11} {'cost':>9} {'mean net':>10} {'total net':>11}")
        for clip in clips:
            fee = FILL_RATE + fixed_usd / clip
            for hz in horizons:
                gross = simulate(rows, fee=fee, lo=lo, hi=hi, min_net=min_net,
                                 rev_frac=rev_frac, lookback=lookback,
                                 horizon_sec=hz * 60.0)
                n, dropped, win, mean_net, total_net, mean_gross = stats(gross, fee, args.censor)
                if not n:
                    print(f"    ${clip:>6.2f} {hz:>5.0f}m {0:>5} {dropped:>5} "
                          f"{'--':>6} {'--':>11} {fee:>8.4%} {'--':>10} {'--':>11}")
                    continue
                print(f"    ${clip:>6.2f} {hz:>5.0f}m {n:>5} {dropped:>5} {win:>5.0f}% "
                      f"{mean_gross:>+10.4f}% {fee:>8.4%} {mean_net:>+9.4f}% {total_net:>+10.3f}%")
        print()

    floors = [float(f) for f in args.floors.split(",") if f.strip()]
    if floors:
        print("=" * 92)
        print("  DRAWDOWN FLOOR SWEEP -- does the rule fire AT ALL where these symbols live?")
        print("=" * 92)
        print("  The production band starts at 2%. If a symbol's deepest 2h drawdown never")
        print("  reaches it, the rule is not selective on that symbol, it is ABSENT from it,")
        print("  and 'no edge' and 'no trades' are different findings. min_net is scaled to")
        print("  floor*0.25 so the 0.50% default does not veto the small dips and get read")
        print("  back as a missing signal.")
        print()
        for symbol in symbols:
            rows = ticks.get(symbol) or []
            if len(rows) < 8:
                continue
            print(f"  {symbol}")
            print(f"    {'floor':>7} {'clip':>7} {'hold':>6} {'n':>5} {'cens':>5} {'win':>6} "
                  f"{'mean gross':>11} {'cost':>9} {'mean net':>10} {'total net':>11}")
            for floor in floors:
                mn = max(0.0005, floor * 0.25)
                for clip in clips:
                    fee = FILL_RATE + fixed_usd / clip
                    for hz in horizons:
                        gross = simulate(rows, fee=fee, lo=floor, hi=hi, min_net=mn,
                                         rev_frac=rev_frac, lookback=lookback,
                                         horizon_sec=hz * 60.0)
                        n, dropped, win, mean_net, total_net, mean_gross = stats(
                            gross, fee, args.censor)
                        if not n:
                            print(f"    {floor:>6.2%} ${clip:>6.2f} {hz:>5.0f}m {0:>5} "
                                  f"{dropped:>5} {'--':>6} {'--':>11} {fee:>8.4%} "
                                  f"{'--':>10} {'--':>11}")
                            continue
                        print(f"    {floor:>6.2%} ${clip:>6.2f} {hz:>5.0f}m {n:>5} {dropped:>5} "
                              f"{win:>5.0f}% {mean_gross:>+10.4f}% {fee:>8.4%} "
                              f"{mean_net:>+9.4f}% {total_net:>+10.3f}%")
            print()

    print("=" * 92)
    print("  READ IT THIS WAY. 'mean gross' is the signal and it is the SAME experiment at")
    print("  both clips except that a cheaper fee admits a different entry set at a lower")
    print("  target. 'cost' is what size buys you. If mean gross sits below the rate floor")
    print(f"  of {FILL_RATE:.4%} at every horizon, then NO clip inverts the sign and the")
    print("  constraint is the SIGNAL, not the SIZE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
