"""[c75f9b1c] Run the SYMBOL EDGE GATE's own verdict on the FIRING arm alone.

The gate cannot do this itself: services/symbol_edge_gate.py:_load_book selects
symbol/net/gross/entry_price/quantity/details from trade_outcomes and NOTHING in
that row set records whether the strategy's entry condition was true. This script
reconstructs the missing dimension from the price tape and re-asks the gate's own
question -- using the gate's own functions, not a re-derivation of its arithmetic.

Arm proxy: donchian breakout on 1h bars resampled from market_stream, the same
instrument Sage and Echo used at p696/p705. FIRING = entry price strictly above
the highest close of the previous LB bars.

Usage: python scripts/arm_split_census.py [SYMBOL] [LOOKBACK]
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from services import symbol_edge_gate as G  # noqa: E402

SYMBOL = (sys.argv[1] if len(sys.argv) > 1 else "AERO-USDC").upper()
LOOKBACK = int(sys.argv[2]) if len(sys.argv) > 2 else 24
# The gate's PAIR book is keyed (strategy_id, symbol); pooling every strategy
# on the symbol is a different population and answers a different question.
STRATEGY = sys.argv[3] if len(sys.argv) > 3 else "atf_static"
BAR = 3600.0

conn = sqlite3.connect(f"file:{G.DB_PATH}?mode=ro", uri=True)


def _json(blob):
    try:
        return json.loads(blob or "{}")
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- population
# Entry fills, keyed by trade_id. Parse the JSON -- `details LIKE '%_entry%'`
# matches every exit row too, because exits carry an `entry_price` key.
entries = {}
for ts, det in conn.execute(
    "SELECT ts, details FROM trade_fills WHERE symbol = ? ORDER BY ts", (SYMBOL,)
):
    d = _json(det)
    mode = str(d.get("mode") or "")
    tid = d.get("trade_id")
    if not mode.endswith("_entry") or not tid:
        continue
    entries.setdefault(str(tid), (float(ts), mode))

# Outcomes, non-annulled, AGGREGATED per trade_id -- the extra rows are
# SCALE-OUTS (p707), so sum the legs rather than dropping them.
legs = defaultdict(list)
for tid, net, gross, ep, qty, det in conn.execute(
    "SELECT trade_id, net_profit, gross_profit, entry_price, quantity, details "
    "FROM trade_outcomes WHERE symbol = ? AND status = 'closed' "
    "AND net_profit IS NOT NULL",
    (SYMBOL,),
):
    d = _json(det)
    if d.get("annulled") or str(d.get("status") or "") == "annulled":
        continue
    if not tid:
        continue
    legs[str(tid)].append((net, gross, ep, qty, G._strategy_of(det)))

# ---------------------------------------------------------------- price tape
ticks = list(conn.execute(
    "SELECT ts, price FROM market_stream WHERE symbol = ? AND price > 0 ORDER BY ts",
    (SYMBOL,),
))
bars = {}
for ts, px in ticks:
    bars[int(float(ts) // BAR)] = float(px)   # last tick in the bar = close
bar_keys = sorted(bars)
# Forward-fill across the holes. Every atf_static entry sits INSIDE the tape
# window (measured: 0 before start, 0 after end) -- what kills coverage is that
# market_stream is sparse, so demanding LOOKBACK consecutive populated bars
# throws away 10 of 14 entries. Carrying the last close forward is the
# conservative direction for a breakout test: a filled bar repeats the prior
# high, so it can only make FIRING harder, never spuriously easier.
filled = {}
if bar_keys:
    last = bars[bar_keys[0]]
    for k in range(bar_keys[0], bar_keys[-1] + 1):
        last = bars.get(k, last)
        filled[k] = last
FILL_LO = bar_keys[0] if bar_keys else 0
FILL_HI = bar_keys[-1] if bar_keys else -1


def arm(ts: float):
    """True = FIRING (breakout), False = SILENT, None = no tape to judge on."""
    b = int(ts // BAR)
    if b - LOOKBACK < FILL_LO or b > FILL_HI:
        return None
    prior = [filled[k] for k in range(b - LOOKBACK, b)]
    px = filled.get(b)
    if px is None or not prior:
        return None
    return px > max(prior)


# ---------------------------------------------------------------- build book
def trips_for(pred):
    out = []
    for tid, (ts, mode) in entries.items():
        if tid not in legs:
            continue
        a = arm(ts)
        if not pred(a):
            continue
        if STRATEGY != "*" and not any(s == STRATEGY for *_x, s in legs[tid]):
            continue
        gross_sum = 0.0
        qty_sum = 0.0
        ep = None
        for net, gross, e, q, _s in legs[tid]:
            try:
                gross_sum += float(net if gross is None else gross)
                qty_sum += float(q)
                ep = float(e) if ep is None else ep
            except (TypeError, ValueError):
                continue
        if ep is None or qty_sum <= 0:
            continue
        notional = ep * qty_sum
        if not (notional >= G.MIN_NOTIONAL):
            continue
        out.append((ts, G.Trip(ret=gross_sum / notional, gross=gross_sum,
                               notional=notional)))
    return sorted(out)


def report(label, rows):
    trips = [t for _ts, t in rows]
    n = len(trips)
    if not n:
        print(f"  {label:<26} n=0")
        return
    mean = sum(t.ret for t in trips) / n
    flat = G.ROUND_TRIP_COST
    costs = [G._trip_cost_fraction(t, flat) for t in trips]
    cost = sum(costs) / n
    excess = [t.ret - c for t, c in zip(trips, costs)]
    tstat = G._t_statistic(excess)
    wins = sum(1 for t, c in zip(trips, costs) if t.ret > c)
    v = G._verdict(trips)
    print(f"  {label:<26} n={n:<3} mean={mean*100:+.3f}%  cost={cost*100:.3f}%  "
          f"t={tstat:+.2f}  win>cost={wins}/{n} ({wins/n*100:.0f}%)  "
          f"gross_sum={sum(t.gross for t in trips):+.5f}")
    print(f"  {'':26} GATE VERDICT: {'REFUSE -- ' + v[1] if v else 'ADMIT (no refusal)'}")


judged = trips_for(lambda a: a is not None)
firing = trips_for(lambda a: a is True)
silent = trips_for(lambda a: a is False)
pooled = trips_for(lambda a: True)

print(f"=== {SYMBOL}  donchian 1h LB={LOOKBACK}  "
      f"MIN_SAMPLES={G.MIN_SAMPLES} ROUND_TRIP_COST={G.ROUND_TRIP_COST}")
print(f"entry fills with trade_id: {len(entries)}   joined to outcomes: "
      f"{len(pooled)}   judgeable on tape: {len(judged)}   "
      f"no tape: {len(pooled) - len(judged)}")
print()
report("POOLED (what gate sees)", pooled)
report("FIRING arm", firing)
report("SILENT arm", silent)

# ------------------------------------------------- disjoint halves (criterion 3)
print()
print("DISJOINT HALVES of the tape -- does the 65/33 win-rate split survive?")
if judged:
    mid = (min(ts for ts, _ in judged) + max(ts for ts, _ in judged)) / 2.0
    for half, lo, hi in (("first half", -math.inf, mid), ("second half", mid, math.inf)):
        f = [t for ts, t in firing if lo <= ts < hi]
        s = [t for ts, t in silent if lo <= ts < hi]
        def wr(trips):
            if not trips:
                return "n=0"
            w = sum(1 for t in trips if t.ret > G._trip_cost_fraction(t, G.ROUND_TRIP_COST))
            return f"{w}/{len(trips)} = {w/len(trips)*100:.0f}%"
        print(f"  {half:<12} FIRING {wr(f):<16} SILENT {wr(s):<16} "
              f"gap {'n/a' if not f or not s else f'{(sum(1 for t in f if t.ret>G._trip_cost_fraction(t,G.ROUND_TRIP_COST))/len(f) - sum(1 for t in s if t.ret>G._trip_cost_fraction(t,G.ROUND_TRIP_COST))/len(s))*100:+.0f} pts'}")
conn.close()
