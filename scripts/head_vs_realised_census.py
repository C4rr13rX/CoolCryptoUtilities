"""Is the collapsed prediction head BROKEN, or is it correctly describing a flat tape?

The operator asked this exactly (2026-09-10 09:42, on item ``618d4c4b``)::

    A head that reads "no edge anywhere" for 12h is either broken or correctly
    describing a flat tape. Those need opposite responses and the difference is
    measurable: compare the head against realised forward returns over the same
    window. If the tape genuinely offered nothing, say so plainly and the
    correct action is to wait, not to loosen the test.

Both responses are expensive if taken against the wrong diagnosis. "Fix the
head" against a genuinely flat tape burns passes on a component that is right;
"wait" against a broken head leaves the ghost lane dark indefinitely. Nothing in
the collapse evidence so far distinguishes them, because every number measured
to date -- ``direction_prob`` p50 0.7833 -> 0.0317, ``net_margin`` MAX negative
across ~3350 cycles, 1572 of 1574 cycles holding -- is a statement about the
head's OUTPUT and none is a statement about the market it was describing.

WHAT THIS MEASURES. Every ``organism_snapshots`` row carries
``payload['prediction']['direction_prob']`` and ``payload['sample']`` naming the
symbol and the tick it was formed on. ``market_stream`` independently carries
the realised price series. So for each snapshot we can ask what the price
actually did over the following N minutes and score the head's call against it:

  * TAPE  -- the realised forward return distribution. If the tape is flat, the
    median |return| collapses toward zero and the head reading "no edge" is
    CORRECT. Reported per era so the pre-collapse window is the control.
  * HEAD  -- directional hit rate of ``direction_prob`` against the realised
    sign, scored ONLY against the majority-class baseline. A head that calls
    DOWN on everything scores 100% in a down tape and that is not skill; the
    honest baseline is "always call the more common realised direction", and
    the head has to beat THAT to be informative.

READ THE VERDICT LINE, NOT THE HIT RATE. Three outcomes, and they are the three
different next actions:

  MARKET   -- tape flat AND head at/below baseline. The head is describing a
              tape with nothing in it. Correct action is to wait, not to loosen
              the entry test.
  MODEL    -- tape NOT flat (there were moves to catch) AND head at/below
              baseline. The head is broken: it is reading "no edge" while the
              tape offered edge. Fix the head.
  INFORMATIVE -- head beats baseline. The head still carries signal and the
              collapse is a CALIBRATION problem, not a content one: the ranking
              survives while the absolute level does not.

UNITS AND CONTAMINATION, because this repo has shipped both bugs. Returns are
FRACTIONS, not percents, everywhere in this file; only the printed table
multiplies by 100 and it says so in the header. Base and forward price are both
read from ``market_stream`` -- never mixing the snapshot's own ``sample.price``
with a stream price, because the feed has carried two denominations under one
ticker (``feed-denomination-contamination``) and a cross-source ratio would
manufacture a return that never happened. Rows whose implied return exceeds
``--max-abs-return`` are dropped as implausible and the DROPPED COUNT IS
PRINTED; a filter that silently eats rows is how a fake edge gets published.

Usage::

    python -X utf8 scripts/head_vs_realised_census.py
    python -X utf8 scripts/head_vs_realised_census.py --hours 24 --horizon-min 15
    python -X utf8 scripts/head_vs_realised_census.py --split-hours 12
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DB = os.path.join("storage", "trading_cache.db")

# A prediction block that reads exactly (0.5, 0.0) is bot.py's no-model
# sentinel, not a prediction. Counting it reports a collapsed head sitting at
# its own ceiling. See memory: no-prediction-sentinel-fakes-a-healthy-head.
SENTINEL_DIRECTION_PROB = 0.5
SENTINEL_NET_MARGIN = 0.0


def _connect(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        raise SystemExit(f"no such database: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_stream(conn: sqlite3.Connection, since: float) -> Dict[str, Tuple[List[float], List[float]]]:
    """Per-symbol (sorted ts list, aligned price list) from ``market_stream``."""
    by_symbol: Dict[str, List[Tuple[float, float]]] = {}
    rows = conn.execute(
        "SELECT ts, symbol, price FROM market_stream WHERE ts >= ? AND price > 0 ORDER BY ts",
        (since,),
    )
    for ts, symbol, price in rows:
        if not symbol or ts is None or price is None:
            continue
        by_symbol.setdefault(str(symbol), []).append((float(ts), float(price)))
    out: Dict[str, Tuple[List[float], List[float]]] = {}
    for symbol, pairs in by_symbol.items():
        pairs.sort(key=lambda p: p[0])
        out[symbol] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out


def price_at(
    series: Tuple[List[float], List[float]],
    target_ts: float,
    tolerance_sec: float,
) -> Optional[float]:
    """Price of the stream tick NEAREST ``target_ts``, or None outside tolerance.

    Nearest rather than last-before: the feed's per-symbol cadence is irregular
    (see ``market-data-parquet-migration``), and a last-before rule silently
    reaches back an unbounded distance whenever a symbol goes quiet, which turns
    a stale price into a fabricated forward return.
    """
    times, prices = series
    if not times:
        return None
    idx = bisect.bisect_left(times, target_ts)
    best: Optional[Tuple[float, float]] = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(times):
            gap = abs(times[cand] - target_ts)
            if best is None or gap < best[0]:
                best = (gap, prices[cand])
    if best is None or best[0] > tolerance_sec:
        return None
    return best[1]


def load_predictions(conn: sqlite3.Connection, since: float) -> List[Dict[str, Any]]:
    """Snapshots carrying a real prediction and an identifiable sample tick."""
    out: List[Dict[str, Any]] = []
    for ts, payload in conn.execute(
        "SELECT ts, payload FROM organism_snapshots WHERE ts >= ? ORDER BY ts", (since,)
    ):
        try:
            blob = json.loads(payload)
        except (TypeError, ValueError):
            continue
        prediction = blob.get("prediction")
        sample = blob.get("sample")
        if not isinstance(prediction, dict) or not isinstance(sample, dict):
            continue
        symbol = sample.get("symbol")
        if not symbol:
            continue
        dp = prediction.get("direction_prob")
        nm = prediction.get("net_margin")
        if dp is None:
            continue
        dp = float(dp)
        nm = float(nm) if nm is not None else None
        if dp == SENTINEL_DIRECTION_PROB and nm == SENTINEL_NET_MARGIN:
            continue  # bot.py's no-model row, not a prediction
        out.append(
            {
                "ts": float(sample.get("ts") or ts),
                "symbol": str(symbol),
                "direction_prob": dp,
                "net_margin": nm,
            }
        )
    return out


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def score_era(rows: List[Dict[str, Any]], flat_bp: float) -> Dict[str, Any]:
    """Tape statistics and head hit rate versus the majority-class baseline."""
    n = len(rows)
    if not n:
        return {"n": 0}
    returns = sorted(r["realised"] for r in rows)
    abs_returns = sorted(abs(r["realised"]) for r in rows)

    ups = sum(1 for r in rows if r["realised"] > 0)
    downs = sum(1 for r in rows if r["realised"] < 0)
    flats = n - ups - downs
    # Honest baseline: always call the more common realised direction.
    majority = max(ups, downs) / n if n else float("nan")

    # The head's call. Ties at exactly 0.5 are an abstention, not an up-call;
    # scoring them as either direction hands the head free accuracy.
    scored = [r for r in rows if r["direction_prob"] != 0.5 and r["realised"] != 0.0]
    hits = sum(
        1
        for r in scored
        if (r["direction_prob"] > 0.5) == (r["realised"] > 0.0)
    )
    hit_rate = hits / len(scored) if scored else float("nan")

    # A hit rate above baseline is not an edge until it is above SAMPLING NOISE.
    # 2724 coin flips have a standard error near 0.0096, so a +0.0115 "beat" is
    # 1.2 SE and means nothing. Report the lower 95% bound and require IT to
    # clear the baseline before any caller is allowed to call this informative.
    if scored:
        se = (hit_rate * (1.0 - hit_rate) / len(scored)) ** 0.5
        lower95 = hit_rate - 1.96 * se
        z = (hit_rate - majority) / se if se > 0 else 0.0
    else:
        se = lower95 = z = float("nan")

    median_abs = _quantile(abs_returns, 0.5)
    return {
        "hit_se": se,
        "hit_lower95": lower95,
        "z_vs_majority": z,
        "significant": bool(scored) and lower95 > majority,
        "n": n,
        "scored": len(scored),
        "dp_p50": _quantile(sorted(r["direction_prob"] for r in rows), 0.5),
        "dp_max": max(r["direction_prob"] for r in rows),
        "ret_p50": _quantile(returns, 0.5),
        "abs_ret_p50": median_abs,
        "abs_ret_p90": _quantile(abs_returns, 0.9),
        "up_frac": ups / n,
        "down_frac": downs / n,
        "flat_frac": flats / n,
        "majority": majority,
        "hit_rate": hit_rate,
        "edge": (hit_rate - majority) if scored else float("nan"),
        "tape_flat": median_abs < (flat_bp / 10000.0),
    }


def verdict(era: Dict[str, Any], min_scored: int) -> Tuple[str, str]:
    """MARKET / MODEL / INFORMATIVE / INSUFFICIENT, plus the reason."""
    if era.get("scored", 0) < min_scored:
        return (
            "INSUFFICIENT",
            f"only {era.get('scored', 0)} scoreable rows (need {min_scored}); "
            "widen --hours or shorten --horizon-min",
        )
    edge = era["edge"]
    if era.get("significant"):
        return (
            "INFORMATIVE",
            f"head hit rate {era['hit_rate']:.4f} beats the majority baseline "
            f"{era['majority']:.4f} by {edge:+.4f}, and its 95% lower bound "
            f"{era['hit_lower95']:.4f} is still above baseline (z={era['z_vs_majority']:+.2f}) "
            "-- content survives, so the collapse is in the LEVEL (calibration), "
            "not the ranking",
        )
    noise = (
        f"head hit rate {era['hit_rate']:.4f} vs majority baseline "
        f"{era['majority']:.4f} is {edge:+.4f}, z={era['z_vs_majority']:+.2f} -- "
        "inside sampling noise, so the head carries NO measurable direction signal"
    )
    if era["tape_flat"]:
        return (
            "MARKET",
            f"median |forward return| {era['abs_ret_p50'] * 100:.4f}% is below the flat "
            f"threshold, and {noise}. The tape offered nothing; WAIT, do not loosen "
            "the entry test",
        )
    return (
        "MODEL",
        f"median |forward return| {era['abs_ret_p50'] * 100:.4f}% says the tape DID move, "
        f"and {noise}. It is reading 'no edge' over a tape that moved; the head is the "
        "defect, not the market",
    )


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "     -"
    try:
        if value != value:  # NaN
            return "   nan"
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--hours", type=float, default=24.0, help="total lookback")
    ap.add_argument(
        "--split-hours",
        type=float,
        default=12.0,
        help="rows newer than this many hours are the POST-collapse era",
    )
    ap.add_argument(
        "--horizon-min",
        type=float,
        default=15.0,
        help="forward return horizon in minutes (trades here resolve in minutes)",
    )
    ap.add_argument(
        "--tolerance-sec",
        type=float,
        default=120.0,
        help="how far from the target timestamp a stream tick may sit",
    )
    ap.add_argument(
        "--max-abs-return",
        type=float,
        default=0.20,
        help="drop rows implying a larger move than this fraction (contamination guard)",
    )
    ap.add_argument(
        "--flat-bp",
        type=float,
        default=10.0,
        help="median |forward return| below this many basis points counts as a FLAT tape",
    )
    ap.add_argument("--min-scored", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    now = time.time()
    since = now - args.hours * 3600.0
    horizon = args.horizon_min * 60.0

    conn = _connect(args.db)
    try:
        stream = load_stream(conn, since - args.tolerance_sec)
        preds = load_predictions(conn, since)
    finally:
        conn.close()

    matched: List[Dict[str, Any]] = []
    no_symbol = no_base = no_forward = implausible = 0
    for row in preds:
        series = stream.get(row["symbol"])
        if series is None:
            no_symbol += 1
            continue
        base = price_at(series, row["ts"], args.tolerance_sec)
        if base is None or base <= 0:
            no_base += 1
            continue
        fwd = price_at(series, row["ts"] + horizon, args.tolerance_sec)
        if fwd is None or fwd <= 0:
            no_forward += 1
            continue
        realised = (fwd - base) / base
        if abs(realised) > args.max_abs_return:
            implausible += 1
            continue
        row = dict(row)
        row["realised"] = realised
        matched.append(row)

    boundary = now - args.split_hours * 3600.0
    pre = [r for r in matched if r["ts"] < boundary]
    post = [r for r in matched if r["ts"] >= boundary]

    eras = {
        "pre_collapse": score_era(pre, args.flat_bp),
        "post_collapse": score_era(post, args.flat_bp),
    }
    kind, reason = verdict(eras["post_collapse"], args.min_scored)

    result = {
        "generated_ts": now,
        "hours": args.hours,
        "split_hours": args.split_hours,
        "horizon_min": args.horizon_min,
        "predictions_seen": len(preds),
        "matched": len(matched),
        "dropped": {
            "symbol_absent_from_stream": no_symbol,
            "no_base_price": no_base,
            "no_forward_price": no_forward,
            "implausible_return": implausible,
        },
        "eras": eras,
        "verdict": kind,
        "reason": reason,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print("=" * 78)
    print("HEAD vs REALISED FORWARD RETURNS -- is the collapse the MODEL or the MARKET?")
    print("=" * 78)
    print(
        f"  window {args.hours:.0f}h, split at -{args.split_hours:.0f}h, "
        f"forward horizon {args.horizon_min:.0f} min, tolerance {args.tolerance_sec:.0f}s"
    )
    print(
        f"  predictions with a real head: {len(preds)}   matched to a forward price: {len(matched)}"
    )
    print(
        "  dropped: "
        f"symbol not in stream {no_symbol}, no base {no_base}, "
        f"no forward {no_forward}, implausible |ret|>{args.max_abs_return:.0%} {implausible}"
    )
    print()
    print("  returns are PERCENT in this table; the head columns are probabilities")
    print(
        "  era             n  scored   dp_p50  dp_max   |ret|p50  |ret|p90   up%  down%"
        "   base   head    edge"
    )
    for name in ("pre_collapse", "post_collapse"):
        era = eras[name]
        if not era.get("n"):
            print(f"  {name:<14} {0:>5}   -- no rows --")
            continue
        print(
            f"  {name:<14} {era['n']:>5} {era['scored']:>7}  "
            f"{_fmt(era['dp_p50'])} {_fmt(era['dp_max'])}  "
            f"{era['abs_ret_p50'] * 100:>8.4f}  {era['abs_ret_p90'] * 100:>8.4f}  "
            f"{era['up_frac'] * 100:>4.1f}  {era['down_frac'] * 100:>5.1f}  "
            f"{_fmt(era['majority'])} {_fmt(era['hit_rate'])} {_fmt(era['edge'], '+.4f')}"
        )
    print()
    # Direction is worthless if the move is smaller than the round trip. The
    # cost is the measured one from receipts (0.004047 fixed + 0.3187% of
    # notional), never a flat 0.65%. At the status command's $23.18 deployable
    # the fixed leg alone dominates, so quote both.
    post = eras["post_collapse"]
    if post.get("n"):
        pct_cost = 0.3187
        print(
            f"  COST FLOOR: median |{args.horizon_min:.0f}min return| is "
            f"{post['abs_ret_p50'] * 100:.4f}% against a {pct_cost:.4f}% proportional "
            f"round trip (plus 0.004047 fixed)."
        )
        payers = sum(1 for r in matched if r["ts"] >= boundary and abs(r["realised"]) * 100 > pct_cost)
        print(
            f"  {payers} of {post['n']} post-collapse ticks moved further than the "
            f"proportional cost alone ({payers / post['n'] * 100:.1f}%) -- a perfect "
            "direction call on the rest still loses money."
        )
        print()
    print(f"  VERDICT: {kind}")
    print(f"  {reason}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
