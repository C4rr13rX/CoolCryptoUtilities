"""Where does the money_button lane actually die?

The registry said money_button was 16W/60L at -0.3997: tried hard, does not
work. That record was fabricated (see scripts/purge_unwitnessed_records.py);
the database holds exactly ONE money_button round trip, and it won. So the
real question was never "does it lose" -- it is "why does it almost never
fire", and nothing in the system could answer that, because a strategy that
returns None returns nothing to look at.

This replays the real strategy object over the ticks the feed actually
recorded, and counts which gate declined. No gate is re-implemented here: it
calls MoneyButtonStrategy.evaluate and reads the reason it recorded, so this
census cannot drift away from the code it measures.

Read the output as a diagnosis of the FEED first and the EDGE second. The
gates fall in two groups:

    too_few_samples, window_too_short, feed_frozen
        The lane never got to look at the market. These are feed-density
        failures wearing a strategy's clothes -- if they dominate, tuning the
        cost gate would change nothing except how much money is lost.

    momentum_not_aligned, move_exhausted, slope_not_positive,
    projection_not_positive, edge_below_cost, volume_faded,
    confidence_below_floor
        The lane looked and declined. `edge_below_cost` dominating is the
        honest "the edge is not there at this cost" answer, and the correct
        response to it is to say so, not to lower the bar.

Usage:
    python scripts/money_button_gate_census.py                # 7 days
    python scripts/money_button_gate_census.py --hours 24
    python scripts/money_button_gate_census.py --fee 0.0065   # live fee rate
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB_PATH = "file:%s?mode=ro" % (ROOT / "storage" / "trading_cache.db")

FEED_GATES = {"no_price", "too_few_samples", "window_too_short", "feed_frozen"}


class _State:
    """Stands in for RouteState.

    The gates read only `.samples`, but `make_candidate` reads the token
    fields -- it refuses any pair that is not stable-quoted, because a
    base/base pair carries a token ratio rather than a USD price. Leaving
    them unset made every genuine fire come back as a decline with no reason
    attached, which is exactly the blind spot this census exists to remove.
    """

    __slots__ = ("samples", "symbol", "base_token", "quote_token")

    def __init__(self, samples, symbol: str):
        self.samples = samples
        self.symbol = symbol
        base, _, quote = symbol.partition("-")
        self.base_token = base
        self.quote_token = quote or "USDC"


def load_ticks(hours: float) -> Dict[str, List[Tuple[float, float, float]]]:
    conn = sqlite3.connect(DB_PATH, uri=True)
    try:
        newest = list(conn.execute("SELECT MAX(ts) FROM market_stream"))[0][0]
        if not newest:
            return {}
        cutoff = float(newest) - hours * 3600.0
        rows = conn.execute(
            "SELECT symbol, ts, price, COALESCE(volume, 0.0) FROM market_stream "
            "WHERE ts > ? AND price > 0 ORDER BY symbol, ts",
            (cutoff,),
        )
        by_symbol: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        for symbol, ts, price, volume in rows:
            by_symbol[str(symbol).upper()].append((float(ts), float(price), float(volume)))
        return dict(by_symbol)
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--fee", type=float, default=None,
                    help="one-way fee rate; default reads TRADE_FEE_RATE or 0.003")
    args = ap.parse_args()

    import os
    from trading.strategies.base import StrategyContext
    from trading.strategies.money_button import MoneyButtonStrategy

    fee = args.fee if args.fee is not None else float(os.getenv("TRADE_FEE_RATE", "0.003"))
    strat = MoneyButtonStrategy()
    lookback = strat.LOOKBACK_SEC

    by_symbol = load_ticks(args.hours)
    if not by_symbol:
        print("no ticks in the window; nothing to census.")
        return 0

    hold_sec = float(os.getenv("MONEY_BUTTON_HOLD_MIN", "12")) * 60.0
    slippage = float(os.getenv("MONEY_BUTTON_SLIPPAGE", "0.001"))
    round_trip = 2.0 * fee + slippage

    census: Counter = Counter()
    fires: Counter = Counter()
    per_symbol_reached: Counter = Counter()
    evaluated = 0
    #: (symbol, entry_ts, entry_price, exit_ts, exit_price) for every fire
    #: whose holding period is fully inside the window, so the trade can be
    #: scored instead of guessed at.
    scored: List[Tuple[str, float, float, float, float]] = []
    unresolved = 0

    for symbol, ticks in by_symbol.items():
        # Walk forward: at each tick the strategy sees the window it would
        # have seen live.
        start = 0
        for end in range(len(ticks)):
            cutoff = ticks[end][0] - lookback
            while ticks[start][0] < cutoff:
                start += 1
            window = ticks[start:end + 1]
            ctx = StrategyContext(
                chain="base",
                last_price=window[-1][1],
                last_volume=window[-1][2],
                fee_rate=fee,
                available_quote=10.0,
                available_base=0.0,
            )
            result = strat.evaluate(_State(window, symbol), ctx)
            evaluated += 1
            if result is None:
                census[strat.last_decline or "unknown"] += 1
                if strat.last_decline not in FEED_GATES:
                    per_symbol_reached[symbol] += 1
            else:
                fires[symbol] += 1
                per_symbol_reached[symbol] += 1
                entry_ts, entry_price = ticks[end][0], ticks[end][1]
                exit_idx = next(
                    (j for j in range(end + 1, len(ticks))
                     if ticks[j][0] >= entry_ts + hold_sec),
                    None,
                )
                if exit_idx is None:
                    # The hold runs past the end of the recorded data. Counting
                    # it as flat would invent a result; leave it unscored.
                    unresolved += 1
                else:
                    scored.append((symbol, entry_ts, entry_price,
                                   ticks[exit_idx][0], ticks[exit_idx][1]))

    total_fires = sum(fires.values())
    print("money_button gate census -- %.0fh of stored ticks, one-way fee %.4f"
          % (args.hours, fee))
    print("%d symbols, %d ticks, %d evaluations\n" % (
        len(by_symbol), sum(len(v) for v in by_symbol.values()), evaluated))

    feed_blocked = sum(n for g, n in census.items() if g in FEED_GATES)
    print("%-26s %9s %8s   %s" % ("gate", "declines", "share", "kind"))
    print("-" * 62)
    for gate, n in census.most_common():
        print("%-26s %9d %7.2f%%   %s" % (
            gate, n, n / evaluated * 100,
            "FEED  -- never looked at the market" if gate in FEED_GATES
            else "EDGE  -- looked and declined",
        ))
    print("%-26s %9d %7.2f%%   FIRED" % ("(candidate produced)", total_fires,
                                         total_fires / evaluated * 100))
    print()
    reached = evaluated - feed_blocked
    print("evaluations that got past the feed gates: %d of %d (%.2f%%)"
          % (reached, evaluated, reached / evaluated * 100))
    if reached:
        print("of those, fired: %d (%.2f%%)" % (total_fires, total_fires / reached * 100))
    else:
        print("of those, fired: n/a -- the lane never saw a usable window.")

    if fires:
        print("\nsymbols that produced a candidate:")
        for symbol, n in fires.most_common(15):
            print("  %-22s %d" % (symbol, n))

    if not scored:
        print("\nno fire had a full holding period inside the window; "
              "nothing can be scored.")
        return 0

    # Overlapping fires are the same move counted repeatedly: at ~7-minute
    # sample gaps a 12-minute hold spans two prints, so one trend books two or
    # three "trades" a real bot could never have taken concurrently. Both
    # numbers are reported -- the raw one is what the gate did, the sequential
    # one is what an account would have experienced.
    def summarise(rows, label):
        nets = [(px_out / px_in - 1.0) - round_trip for _s, _t, px_in, _te, px_out in rows]
        wins = sum(1 for n in nets if n > 0)
        total = sum(nets)
        print("\n%s: %d trade(s), %dW/%dL, mean %+.4f%%, total %+.4f%% of notional"
              % (label, len(nets), wins, len(nets) - wins,
                 total / len(nets) * 100, total * 100))
        gross = [(px_out / px_in - 1.0) for _s, _t, px_in, _te, px_out in rows]
        print("    gross before the %.2f%% round trip: mean %+.4f%%, %d of %d up"
              % (round_trip * 100, sum(gross) / len(gross) * 100,
                 sum(1 for g in gross if g > 0), len(gross)))

    print("\n--- outcome of every fire, held %.0f min then exited at the next tick ---"
          % (hold_sec / 60.0))
    print("(a fixed-hold proxy: the live bot also exits on stops and confidence"
          " drops, so this is the lane's raw signal, not its realised P&L)")
    if unresolved:
        print("%d fire(s) unresolved -- their hold runs past the recorded data."
              % unresolved)
    summarise(scored, "raw (overlapping fires counted separately)")

    sequential = []
    busy_until: Dict[str, float] = defaultdict(float)
    for row in sorted(scored, key=lambda r: r[1]):
        symbol, entry_ts = row[0], row[1]
        if entry_ts < busy_until[symbol]:
            continue
        sequential.append(row)
        busy_until[symbol] = row[3]
    summarise(sequential, "sequential (no re-entry while still holding)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
