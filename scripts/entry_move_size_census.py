#!/usr/bin/env python
"""Does the entry conjunction test HOW FAR the price is expected to move?

[4d3310e7]. The model-long entry test at ``trading/bot.py`` reads
``direction_prob``, ``exit_conf``, ``net_margin`` and ``delta``. Three of those
are about WHETHER; the fourth, ``delta`` -- which IS the model's forward
expected return (``price_mu``) -- is read only for its SIGN (``delta >= 0.0``).

So a correctly predicted move that is smaller than the round trip is admitted.
The measured median absolute 15-minute return on this feed is 0.2233% against a
round trip of 0.3187% + $0.004047/notional, so that is most ticks.

This script replays the conjunction over recorded ``organism_snapshots``
decision cycles -- the same fields ``_interpret_predictions`` put into
``decision`` -- and reports:

  * how many cycles the CURRENT conjunction admits,
  * how many the conjunction admits with the size conjunct added,
  * and, on the admitted subsets, the realised forward net return per trade
    after charging each cycle its own measured round-trip rate.

The size threshold is not a round number. It is the round-trip rate the cycle
itself was charged, recovered from the snapshot as
``net_margin - net_margin_after_fees`` (bot.py sets
``net_margin_after_fees = margin - entry_fees``), which is exactly
``services.roundtrip_cost.roundtrip_cost_rate(notional)``.

Usage:
    python scripts/entry_move_size_census.py [--limit N] [--mult 1.0]
                                             [--horizon 900 --horizon 1800]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.roundtrip_cost import DEFAULT_FIXED_USD, DEFAULT_RATE  # noqa: E402

DB_PATH = os.getenv(
    "TRADING_DB_PATH", str(PROJECT_ROOT / "storage" / "trading_cache.db")
)

# The conjunction's own constants, read from the same place bot.py reads them.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE_REQUIRED", "0.9"))
MIN_NET_MARGIN = float(os.getenv("MIN_NET_MARGIN", "0.0001"))
SMALL_PROFIT_FLOOR = float(os.getenv("SMALL_PROFIT_FLOOR_USD", "0.02"))


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=120)


def load_cycles(conn: sqlite3.Connection, limit: int) -> List[Dict[str, float]]:
    """Decision cycles, newest `limit` of them, as plain dicts.

    json_extract keeps the parse in C: the payloads carry organism_graph and
    process_clusters too, and json.loads on all of them reads gigabytes.
    """
    sql = """
        SELECT ts,
               json_extract(payload, '$.decision.symbol'),
               json_extract(payload, '$.sample.price'),
               json_extract(payload, '$.decision.direction_prob'),
               json_extract(payload, '$.decision.exit_confidence'),
               json_extract(payload, '$.decision.expected_delta'),
               json_extract(payload, '$.decision.net_margin'),
               json_extract(payload, '$.decision.net_margin_after_fees'),
               json_extract(payload, '$.decision.action'),
               json_extract(payload, '$.pipeline.decision_threshold')
        FROM organism_snapshots
        ORDER BY ts DESC
        LIMIT ?
    """
    rows: List[Dict[str, float]] = []
    for r in conn.execute(sql, (limit,)):
        if r[1] is None or r[6] is None or r[7] is None:
            continue
        rows.append(
            {
                "ts": float(r[0]),
                "symbol": str(r[1]),
                "price": float(r[2] or 0.0),
                "direction_prob": float(r[3] or 0.0),
                "exit_conf": float(r[4] or 0.0),
                "delta": float(r[5] or 0.0),
                "margin": float(r[6]),
                "margin_after_fees": float(r[7]),
                "action": str(r[8] or ""),
                "decision_threshold": float(r[9] or 0.0),
            }
        )
    rows.reverse()
    return rows


def load_prices(conn: sqlite3.Connection) -> Dict[str, Tuple[List[float], List[float]]]:
    """Per-symbol (ts, price) series from market_stream, ascending in ts."""
    series: Dict[str, Tuple[List[float], List[float]]] = {}
    sql = "SELECT symbol, ts, price FROM market_stream WHERE price > 0 ORDER BY symbol, ts"
    for sym, ts, price in conn.execute(sql):
        key = str(sym)
        tss, ps = series.setdefault(key, ([], []))
        tss.append(float(ts))
        ps.append(float(price))
    return series


def forward_return(
    series: Dict[str, Tuple[List[float], List[float]]],
    symbol: str,
    ts: float,
    price: float,
    horizon_sec: float,
    tolerance_sec: float,
) -> float | None:
    """Realised return from `price` to the first tick at least `horizon_sec` later.

    Returns None when no tick lands inside [horizon, horizon + tolerance]; a
    gap in the feed is not a zero return and must not be scored as one.
    """
    if price <= 0:
        return None
    pair = series.get(symbol)
    if not pair:
        return None
    tss, ps = pair
    idx = bisect.bisect_left(tss, ts + horizon_sec)
    if idx >= len(tss):
        return None
    if tss[idx] > ts + horizon_sec + tolerance_sec:
        return None
    later = ps[idx]
    if later <= 0:
        return None
    return (later - price) / price


def notional_from_rate(rate: float) -> float:
    """Invert rate = DEFAULT_RATE + DEFAULT_FIXED_USD / notional."""
    excess = rate - DEFAULT_RATE
    if excess <= 0:
        return 0.0
    return DEFAULT_FIXED_USD / excess


def admits(row: Dict[str, float], *, min_move_mult: float | None) -> bool:
    """The model-long conjunction. `min_move_mult=None` is today's behaviour."""
    enter_threshold = max(row["decision_threshold"], MIN_CONFIDENCE)
    enter_threshold = max(0.5, min(0.99, enter_threshold))
    entry_fees = max(0.0, row["margin"] - row["margin_after_fees"])
    min_margin_gate = max(max(entry_fees * 1.5, MIN_NET_MARGIN), entry_fees)
    notional = notional_from_rate(entry_fees)
    expected_profit_units = max(0.0, row["margin"] - entry_fees) * notional
    if not (
        row["direction_prob"] >= enter_threshold
        and row["exit_conf"] >= enter_threshold
        and row["margin"] >= min_margin_gate
        and row["margin_after_fees"] >= MIN_NET_MARGIN
        and expected_profit_units >= SMALL_PROFIT_FLOOR
    ):
        return False
    if min_move_mult is None:
        return row["delta"] >= 0.0
    return row["delta"] >= entry_fees * min_move_mult


def score(
    rows: List[Dict[str, float]],
    series: Dict[str, Tuple[List[float], List[float]]],
    horizon_sec: float,
    tolerance_sec: float,
) -> Dict[str, float]:
    """Per-trade realised net return after the cycle's own round-trip rate."""
    nets: List[float] = []
    grosses: List[float] = []
    for row in rows:
        fwd = forward_return(
            series, row["symbol"], row["ts"], row["price"], horizon_sec, tolerance_sec
        )
        if fwd is None:
            continue
        cost = max(0.0, row["margin"] - row["margin_after_fees"])
        grosses.append(fwd)
        nets.append(fwd - cost)
    if not nets:
        return {"n": 0}
    return {
        "n": len(nets),
        "mean_gross": statistics.fmean(grosses),
        "mean_net": statistics.fmean(nets),
        "median_net": statistics.median(nets),
        "win_rate": sum(1 for x in nets if x > 0) / len(nets),
    }


def fmt(stats: Dict[str, float]) -> str:
    if not stats.get("n"):
        return "n=0 (no forward tick inside tolerance)"
    return (
        f"n={stats['n']:5d}  mean gross {stats['mean_gross']*100:+.4f}%  "
        f"mean NET {stats['mean_net']*100:+.4f}%  median NET "
        f"{stats['median_net']*100:+.4f}%  win {stats['win_rate']*100:.1f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--mult", type=float, action="append", default=None)
    ap.add_argument("--horizon", type=float, action="append", default=None)
    ap.add_argument("--tolerance", type=float, default=600.0)
    args = ap.parse_args()
    mults = args.mult or [1.0]
    horizons = args.horizon or [900.0, 1800.0]

    conn = _connect()
    rows = load_cycles(conn, args.limit)
    series = load_prices(conn)
    conn.close()
    if not rows:
        print("no decision cycles found", file=sys.stderr)
        return 2

    span_h = (rows[-1]["ts"] - rows[0]["ts"]) / 3600.0
    print(f"cycles            : {len(rows)} over {span_h:.1f}h")
    print(f"symbols           : {len({r['symbol'] for r in rows})}")
    print(f"price series       : {len(series)} symbols from market_stream")
    print(f"enter_threshold    : max(decision_threshold, MIN_CONFIDENCE={MIN_CONFIDENCE})")
    rates = [max(0.0, r["margin"] - r["margin_after_fees"]) for r in rows]
    live_rates = [x for x in rates if x > 0]
    if live_rates:
        print(
            f"round-trip rate    : median {statistics.median(live_rates)*100:.4f}% "
            f"(= {DEFAULT_RATE*100:.4f}% + ${DEFAULT_FIXED_USD}/notional; "
            f"implied clip ${notional_from_rate(statistics.median(live_rates)):.2f})"
        )

    # How far does the tape actually move? This is the number the conjunct exists
    # for, measured on this corpus rather than quoted from the item.
    for horizon in horizons:
        moves = []
        for row in rows:
            fwd = forward_return(
                series, row["symbol"], row["ts"], row["price"], horizon, args.tolerance
            )
            if fwd is not None:
                moves.append(abs(fwd))
        if moves:
            med = statistics.median(moves)
            cost = statistics.median(live_rates) if live_rates else DEFAULT_RATE
            clears = sum(1 for m in moves if m > cost) / len(moves)
            print(
                f"|move| @{horizon/60:.0f}m     : median {med*100:.4f}% on n={len(moves)}; "
                f"{clears*100:.1f}% of ticks clear the {cost*100:.4f}% round trip"
            )

    # IS `delta` ON THE SAME SCALE AS THE TAPE? A size test on a forward
    # expected return is only as good as that return's calibration, and a head
    # whose magnitude is an order of magnitude too large makes any cost-derived
    # floor vacuous. This is the number that says whether the conjunct can bite.
    for horizon in horizons:
        pairs = []
        for row in rows:
            fwd = forward_return(
                series, row["symbol"], row["ts"], row["price"], horizon, args.tolerance
            )
            if fwd is not None:
                pairs.append((abs(row["delta"]), abs(fwd)))
        if pairs:
            med_pred = statistics.median([p for p, _ in pairs])
            med_real = statistics.median([r for _, r in pairs])
            ratio = (med_pred / med_real) if med_real > 0 else float("inf")
            over = sum(1 for p, r in pairs if p > r) / len(pairs)
            print(
                f"delta vs |move|@{horizon/60:.0f}m: median |delta| {med_pred*100:.4f}% "
                f"vs median realised {med_real*100:.4f}% -- {ratio:.1f}x too large; "
                f"|delta| overstates the move on {over*100:.1f}% of n={len(pairs)}"
            )

    before = [r for r in rows if admits(r, min_move_mult=None)]
    print()
    print(f"ADMITTED BEFORE (delta >= 0.0)              : {len(before)}")
    for mult in mults:
        after = [r for r in rows if admits(r, min_move_mult=mult)]
        drop = (1 - len(after) / len(before)) * 100 if before else float("nan")
        print(
            f"ADMITTED AFTER  (delta >= {mult:g} x round trip) : {len(after)}"
            f"   ({drop:.1f}% fewer)"
        )
    print()
    for horizon in horizons:
        print(f"-- forward horizon {horizon/60:.0f} min, tolerance {args.tolerance/60:.0f} min")
        print(f"   before          : {fmt(score(before, series, horizon, args.tolerance))}")
        for mult in mults:
            after = [r for r in rows if admits(r, min_move_mult=mult)]
            print(
                f"   after x{mult:g}        : "
                f"{fmt(score(after, series, horizon, args.tolerance))}"
            )
        allrows = score(rows, series, horizon, args.tolerance)
        print(f"   every cycle     : {fmt(allrows)}   <- buy-every-tick baseline")

    # The conjunct in isolation, over every cycle rather than the admitted few:
    # the sign test against the size test, so the strictness is legible even
    # when the rest of the conjunction admits almost nothing.
    sign_ok = sum(1 for r in rows if r["delta"] >= 0.0)
    print()
    print(f"delta >= 0.0 alone  : {sign_ok} of {len(rows)} cycles "
          f"({sign_ok/len(rows)*100:.1f}%)")
    for mult in mults:
        size_ok = sum(
            1
            for r in rows
            if r["delta"] >= max(0.0, r["margin"] - r["margin_after_fees"]) * mult
        )
        print(
            f"delta >= {mult:g}x cost alone: {size_ok} of {len(rows)} cycles "
            f"({size_ok/len(rows)*100:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
