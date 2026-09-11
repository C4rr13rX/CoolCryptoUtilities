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
               json_extract(payload, '$.pipeline.decision_threshold'),
               json_extract(payload, '$.decision.entry_fee_rate'),
               json_extract(payload, '$.brain.volatility_rel')
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
                # THE RATE THE BRANCH CHARGES, NOT THE RATE AT THE CLIP.
                # `net_margin_after_fees` is set at bot.py:6300 as
                # `margin - fees`, where `fees` is priced at the live clip with
                # no notional hint. The entry branch then re-prices the round
                # trip for the notional it is about to spend (bot.py:6901-6905)
                # and records it as `entry_fee_rate`. Measured on this corpus
                # the clip rate is 0.38615% on EVERY row while the recorded
                # entry rate runs 0.3923%..0.8583% -- up to 2.2x higher -- so
                # recovering the cost floor by subtraction understates the bar
                # on most cycles and makes a size-aware threshold read as a
                # flat one. Prefer the recorded rate; subtract only when the
                # row predates it.
                "entry_fee_rate": (
                    float(r[10]) if r[10] is not None else None
                ),
                # THE ONLY SIZE ESTIMATE ON THIS CORPUS THAT IS ON THE TAPE'S
                # SCALE. bot.py:3544 computes it as the standard deviation of
                # the per-tick FRACTIONAL change over the last <=20 ticks, so
                # it is dimensionless like the round-trip rate is. None means
                # the row predates its publication or the symbol had <4 ticks
                # of history; such a row cannot be scored and is not guessed at.
                "volatility_rel": (
                    float(r[11]) if r[11] is not None else None
                ),
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


def cost_rate(row: Dict[str, float]) -> float:
    """The round-trip rate THIS cycle's entry branch was charged, as a fraction.

    bot.py re-prices the round trip for the notional the entry is about to
    spend and records it as ``entry_fee_rate``; ``net_margin_after_fees`` in
    the same payload was computed earlier against the flat clip rate. Prefer
    the recorded entry rate and fall back to the subtraction only for rows
    that predate it, for the reason set out where the column is loaded.
    """
    recorded = row.get("entry_fee_rate")
    if recorded is not None and float(recorded) > 0:
        return float(recorded)
    return max(0.0, row["margin"] - row["margin_after_fees"])


def conjuncts(row: Dict[str, float]) -> List[Tuple[str, bool]]:
    """Every NON-delta conjunct of the model-long entry test, named, in order.

    Split out of ``admits`` so the funnel can say WHICH term refuses a cycle.
    A conjunction that admits almost nothing cannot be used to measure any one
    of its own terms, and the only way to know that is to count them apart.
    """
    enter_threshold = max(row["decision_threshold"], MIN_CONFIDENCE)
    enter_threshold = max(0.5, min(0.99, enter_threshold))
    entry_fees = cost_rate(row)
    min_margin_gate = max(max(entry_fees * 1.5, MIN_NET_MARGIN), entry_fees)
    notional = notional_from_rate(entry_fees)
    expected_profit_units = max(0.0, row["margin"] - entry_fees) * notional
    return [
        ("direction_prob >= enter_threshold", row["direction_prob"] >= enter_threshold),
        ("exit_conf >= enter_threshold", row["exit_conf"] >= enter_threshold),
        ("net_margin >= min_margin_gate", row["margin"] >= min_margin_gate),
        ("net_margin_after_fees >= MIN_NET_MARGIN", row["margin"] - entry_fees >= MIN_NET_MARGIN),
        ("expected_profit_units >= SMALL_PROFIT_FLOOR", expected_profit_units >= SMALL_PROFIT_FLOOR),
    ]


def admits(
    row: Dict[str, float],
    *,
    min_move_mult: float | None,
    require_vol: bool = False,
) -> bool:
    """The model-long conjunction.

    ``min_move_mult=None`` is the pre-709c505 behaviour, ``delta >= 0.0``: the
    sign test. A float applies the shipped ``delta >= c(N) * mult``.
    ``require_vol`` adds the conjunct this pass is about -- the symbol's own
    recent relative volatility must itself clear the round trip.

    A row with no recorded ``volatility_rel`` is REFUSED under ``require_vol``
    rather than admitted: an entry whose expected move cannot be estimated has
    not been shown to clear its cost, and defaulting an unmeasurable size to
    "big enough" is the loosening this item exists to prevent.
    """
    entry_fees = cost_rate(row)
    if not all(ok for _, ok in conjuncts(row)):
        return False
    if min_move_mult is None:
        ok = row["delta"] >= 0.0
        mult = 1.0
    else:
        ok = row["delta"] >= entry_fees * min_move_mult
        mult = min_move_mult
    if not ok:
        return False
    if not require_vol:
        return True
    vol = row.get("volatility_rel")
    return vol is not None and float(vol) >= entry_fees * mult


def funnel(rows: List[Dict[str, float]], mults: List[float]) -> None:
    """WHICH conjunct refuses, and can the delta term be measured at all?

    [4d3310e7] criterion 3 asks for entries admitted BEFORE and AFTER the size
    conjunct over the same cycle window, and says that if the count does not
    FALL the conjunct is not doing what it claims. That test is only readable
    when enough cycles REACH the delta term: a before/after of 2 against 2
    licenses neither verdict, exactly as a per-trade mean on one trade does.

    So this prints three things:

      * each non-delta conjunct's own pass rate, which names the binding term;
      * the MARGINAL DENOMINATOR -- cycles passing every non-delta conjunct,
        i.e. the cycles on which the delta term is the deciding one;
      * the before/after counts inside that denominator.

    Everything here is a count over recorded cycles. Nothing is loosened and
    no threshold is read from anywhere other than the conjunction's own env.
    """
    n = len(rows)
    print()
    print("-- FUNNEL: which conjunct refuses the cycle (each term on its own)")
    names = [name for name, _ in conjuncts(rows[0])]
    passes = [0] * len(names)
    reach: List[Dict[str, float]] = []
    for row in rows:
        terms = conjuncts(row)
        for i, (_, ok) in enumerate(terms):
            if ok:
                passes[i] += 1
        if all(ok for _, ok in terms):
            reach.append(row)
    for name, count in zip(names, passes):
        print(f"   {name:46s} {count:6d} of {n}  ({count / n * 100:6.3f}%)")
    print()
    print(
        f"   MARGINAL DENOMINATOR (all non-delta conjuncts pass): {len(reach)} of {n}"
    )
    if not reach:
        print("   the delta term decides ZERO cycles: before/after is unmeasurable here")
        return
    before = sum(1 for r in reach if r["delta"] >= 0.0)
    print(f"     of those, delta >= 0.0 (before)      : {before}")
    for mult in mults:
        after = sum(1 for r in reach if r["delta"] >= cost_rate(r) * mult)
        vol_after = sum(
            1
            for r in reach
            if r["delta"] >= cost_rate(r) * mult
            and r.get("volatility_rel") is not None
            and float(r["volatility_rel"]) >= cost_rate(r) * mult
        )
        print(f"     of those, delta >= {mult:g}x cost (after) : {after}")
        print(f"     of those, AND vol_rel >= {mult:g}x cost   : {vol_after}")
    print(
        "   SMALLEST FALL THIS DENOMINATOR CAN SHOW: "
        f"1 cycle = {100.0 / len(reach):.1f} percentage points"
    )
    deltas = sorted(r["delta"] for r in reach)
    fees = sorted(cost_rate(r) for r in reach)
    print(
        f"   delta on those cycles: median {statistics.median(deltas) * 100:.2f}% "
        f"against a median round trip of {statistics.median(fees) * 100:.4f}% "
        f"-- {statistics.median(deltas) / statistics.median(fees):.0f}x the bar"
    )
    vols = sorted(
        float(r["volatility_rel"]) for r in reach if r.get("volatility_rel") is not None
    )
    if vols:
        print(
            f"   vol_rel on those cycles: median {statistics.median(vols) * 100:.4f}% "
            f"against the same {statistics.median(fees) * 100:.4f}% round trip "
            f"-- {statistics.median(vols) / statistics.median(fees):.2f}x the bar "
            f"(n={len(vols)} of {len(reach)} carry it)"
        )


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
        cost = cost_rate(row)
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
    ap.add_argument(
        "--funnel-only",
        action="store_true",
        help="print the per-conjunct funnel and stop: counts only, no forward "
        "returns, so it answers 'can the delta term be measured' cheaply.",
    )
    args = ap.parse_args()
    mults = args.mult or [1.0]
    horizons = args.horizon or [900.0, 1800.0]

    conn = _connect()
    rows = load_cycles(conn, args.limit)
    if not rows:
        conn.close()
        print("no decision cycles found", file=sys.stderr)
        return 2
    if args.funnel_only:
        conn.close()
        span = (rows[-1]["ts"] - rows[0]["ts"]) / 3600.0
        print(f"cycles            : {len(rows)} over {span:.1f}h")
        funnel(rows, mults)
        return 0
    series = load_prices(conn)
    conn.close()

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

    # THE SAME CALIBRATION TEST, ON THE OTHER CANDIDATE SIZE ESTIMATE.
    # A size conjunct is only as good as the calibration of the quantity it
    # tests, and the test above is the one that was never run on `delta` before
    # 709c505 shipped it. `volatility_rel` (bot.py:3544) is the standard
    # deviation of the per-tick FRACTIONAL change over the symbol's last <=20
    # ticks -- dimensionless, computed from data available AT ENTRY, and an
    # estimate of a move that has not happened yet, so it is forward-looking
    # and is not the realised move. Whether it is the right SCALE is not
    # assumed here: it is measured against the same realised tape below.
    for horizon in horizons:
        pairs = []
        for row in rows:
            vol = row.get("volatility_rel")
            if vol is None:
                continue
            fwd = forward_return(
                series, row["symbol"], row["ts"], row["price"], horizon, args.tolerance
            )
            if fwd is not None:
                pairs.append((float(vol), abs(fwd)))
        if pairs:
            med_pred = statistics.median([p for p, _ in pairs])
            med_real = statistics.median([r for _, r in pairs])
            ratio = (med_pred / med_real) if med_real > 0 else float("inf")
            over = sum(1 for p, r in pairs if p > r) / len(pairs)
            print(
                f"vol_rel vs |move|@{horizon/60:.0f}m: median vol_rel "
                f"{med_pred*100:.4f}% vs median realised {med_real*100:.4f}% "
                f"-- ratio {ratio:.2f}; overstates the move on {over*100:.1f}% "
                f"of n={len(pairs)}"
            )

    funnel(rows, mults)

    before = [r for r in rows if admits(r, min_move_mult=None)]
    print()
    print(f"ADMITTED BEFORE (delta >= 0.0)              : {len(before)}")
    for mult in mults:
        after = [r for r in rows if admits(r, min_move_mult=mult)]
        with_vol = [
            r for r in rows if admits(r, min_move_mult=mult, require_vol=True)
        ]
        drop = (1 - len(after) / len(before)) * 100 if before else float("nan")
        vdrop = (1 - len(with_vol) / len(before)) * 100 if before else float("nan")
        print(
            f"ADMITTED AFTER  (delta >= {mult:g} x round trip) : {len(after)}"
            f"   ({drop:.1f}% fewer)"
        )
        print(
            f"ADMITTED AFTER  (+ vol_rel >= {mult:g} x round trip): {len(with_vol)}"
            f"   ({vdrop:.1f}% fewer)"
        )
    print()
    for horizon in horizons:
        print(f"-- forward horizon {horizon/60:.0f} min, tolerance {args.tolerance/60:.0f} min")
        print(f"   before          : {fmt(score(before, series, horizon, args.tolerance))}")
        for mult in mults:
            after = [r for r in rows if admits(r, min_move_mult=mult)]
            with_vol = [
                r for r in rows if admits(r, min_move_mult=mult, require_vol=True)
            ]
            print(
                f"   after x{mult:g}        : "
                f"{fmt(score(after, series, horizon, args.tolerance))}"
            )
            print(
                f"   after x{mult:g} +vol   : "
                f"{fmt(score(with_vol, series, horizon, args.tolerance))}"
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
        size_ok = sum(1 for r in rows if r["delta"] >= cost_rate(r) * mult)
        print(
            f"delta >= {mult:g}x cost alone: {size_ok} of {len(rows)} cycles "
            f"({size_ok/len(rows)*100:.1f}%)"
        )
        vol_ok = sum(
            1
            for r in rows
            if r.get("volatility_rel") is not None
            and float(r["volatility_rel"]) >= cost_rate(r) * mult
        )
        print(
            f"vol_rel >= {mult:g}x cost alone: {vol_ok} of {len(rows)} cycles "
            f"({vol_ok/len(rows)*100:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
