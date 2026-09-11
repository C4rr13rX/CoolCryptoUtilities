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


def buy_rule_profile(
    rows: List[Dict[str, Any]],
    pct_cost: float,
    clip: float,
    fixed_cost: float,
    top_frac: float = 0.10,
) -> Dict[str, Any]:
    """Score the head as a BUY-LOW rule and as a SELL-HIGH rule, not on accuracy.

    Operator direction, 2026-09-10: exact/directional accuracy is dominated by
    outcomes nobody trades. The question is whether the head is accurate
    SPECIFICALLY about when we can buy low and sell high, so the numbers here
    are precision on the actionable call and net per trade against the
    do-it-every-bar baseline.

    PRECISION, not hit rate. A directional hit on a move smaller than the round
    trip is a LOSS, so a "correct" call that does not clear the cost floor is
    scored as a miss. floor = pct_cost/100 + fixed_cost/clip, both legs of the
    measured receipt cost (0.3187% of notional + 0.004047 fixed), and the fixed
    leg is amortised over the clip because that is the only part clip moves.

    BASELINE IS THE MONEY RULE, NEVER 0.5 AND NEVER THE MAJORITY CLASS. For the
    buy side it is "buy every bar" in the SAME rows; for the sell side it is
    the unconditional rate of a fall worth exiting. A rule that clears the
    floor less often than buying blind has negative value however accurate it
    looks.

    THE SELL SIDE IS MEASURED HERE BECAUSE IT IS MEASURED NOWHERE ELSE. The
    experiment harness is long-only, so a correct DOWN call is an abstention
    and scores zero -- but a correct DOWN call on a position we HOLD is worth
    money as an exit. Scored against forward returns, not by shorting: an exit
    is right when the price fell by more than the ONE leg it costs to leave.

    Also scored: the same rule at the head's TOP DECILE of direction_prob. The
    level and the ordering are separate properties (see rank_profile) and only
    a rank threshold exploits the ordering, so if the ordering is the only
    thing with skill this is where it has to show up as money.
    """
    n = len(rows)
    if not n:
        return {"n": 0}
    floor = pct_cost / 100.0 + (fixed_cost / clip if clip > 0 else 0.0)
    one_leg = floor / 2.0

    def _side(group: List[Dict[str, Any]], want_up: bool) -> Dict[str, Any]:
        if not group:
            return {"n": 0, "precision": float("nan"), "net_per_trade": float("nan")}
        if want_up:
            paid = sum(1 for r in group if r["realised"] > floor)
        else:
            paid = sum(1 for r in group if r["realised"] < -one_leg)
        gross = sum(r["realised"] for r in group) / len(group)
        return {
            "n": len(group),
            "precision": paid / len(group),
            "gross_per_trade": gross,
            # A long entry pays the whole round trip; an exit pays one leg.
            "net_per_trade": (gross - floor) if want_up else (-gross - one_leg),
        }

    ups = [r for r in rows if r["direction_prob"] > 0.5]
    downs = [r for r in rows if r["direction_prob"] < 0.5]
    ordered = sorted(rows, key=lambda r: r["direction_prob"], reverse=True)
    k = max(1, int(round(len(ordered) * top_frac)))
    top = ordered[:k]

    return {
        "n": n,
        "floor": floor,
        "one_leg": one_leg,
        "clip": clip,
        "buy": _side(ups, True),
        "buy_baseline": _side(rows, True),          # buy every bar
        "buy_top_decile": _side(top, True),         # the ORDERING as a money rule
        "top_decile_n": k,
        "top_decile_dp_min": top[-1]["direction_prob"] if top else float("nan"),
        "sell": _side(downs, False),
        "sell_baseline": _side(rows, False),        # exit every bar
    }


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


def rank_profile(rows: List[Dict[str, Any]], buckets: int = 5) -> Dict[str, Any]:
    """Does a HIGHER direction_prob actually mean a higher realised return?

    This is the question a calibrator cannot dodge. Calibration is a monotone
    remap of the head's output: it moves the LEVEL and preserves the ORDER by
    construction. So if the ranking carries information, recalibrating can
    recover a usable head; if the ranking carries none, there is nothing for a
    calibrator to rescue and the defect is in the model's content.

    READ ``auc``, NOT ``monotonic``, FOR THE VERDICT. This is a correction to
    an earlier version of this file, which reported only the quintile profile
    and drew "the order carries nothing" from its non-monotonicity. THAT
    INFERENCE IS WRONG: a noisy-but-informative ranking is routinely
    non-monotonic across five buckets, so ``monotonic`` is a shape description
    and ``auc`` is the aggregate that decides it. 0.5 is no ranking skill.

    Measured 2026-09-10 over 15-min forward returns:

        pre_collapse   AUC 0.4076   quintiles 50.3 52.7 31.9 37.9 43.0
        post_collapse  AUC 0.5264   quintiles 42.1 50.5 38.5 41.2 53.6

    Both are non-monotonic and they are NOT the same finding. The pre-collapse
    head -- the dp_max 0.9795 state everyone has been trying to restore -- is
    genuinely INVERTED: more confidence, less likely to be right, top decile
    mean return -0.1020% against the bottom decile's -0.0363%. The
    post-collapse head sits weakly ABOVE chance. So restoring the old level is
    not a fix, and recalibrating the new one is not obviously hopeless -- but
    neither is tradeable, because the skill does not survive the round trip
    cost at any percentile.
    """
    if not rows:
        return {"n": 0}
    ordered = sorted(rows, key=lambda r: r["direction_prob"])
    n = len(ordered)
    k = max(n // 10, 1)

    def up_rate(group: List[Dict[str, Any]]) -> float:
        return sum(1 for r in group if r["realised"] > 0) / len(group) if group else float("nan")

    def mean_ret(group: List[Dict[str, Any]]) -> float:
        return sum(r["realised"] for r in group) / len(group) if group else float("nan")

    # Mann-Whitney AUC: the probability that a randomly chosen row that went UP
    # was given a higher direction_prob than a randomly chosen row that went
    # DOWN. Baseline-invariant, so unlike a hit rate it is not flattered by a
    # head that calls DOWN on everything in a down tape. Ties share their rank.
    pos = [r["direction_prob"] for r in ordered if r["realised"] > 0]
    neg = [r["direction_prob"] for r in ordered if r["realised"] < 0]
    if pos and neg:
        values = sorted(pos + neg)
        rank_of: Dict[float, float] = {}
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[j + 1] == values[i]:
                j += 1
            rank_of[values[i]] = (i + j) / 2.0 + 1.0
            i = j + 1
        rank_sum = sum(rank_of[v] for v in pos)
        auc = (rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
        # Hanley-McNeil standard error, the usual approximation.
        q1 = auc / (2.0 - auc)
        q2 = 2.0 * auc * auc / (1.0 + auc)
        auc_se = (
            (
                auc * (1 - auc)
                + (len(pos) - 1) * (q1 - auc * auc)
                + (len(neg) - 1) * (q2 - auc * auc)
            )
            / (len(pos) * len(neg))
        ) ** 0.5
    else:
        auc = auc_se = float("nan")

    bottom, top = ordered[:k], ordered[-k:]
    quintiles = [ordered[i * n // buckets : (i + 1) * n // buckets] for i in range(buckets)]
    quintiles = [q for q in quintiles if q]
    return {
        "n": n,
        "auc": auc,
        "auc_se": auc_se,
        "auc_beats_chance": bool(auc == auc and (auc - 1.96 * auc_se) > 0.5),
        "auc_inverted": bool(auc == auc and (auc + 1.96 * auc_se) < 0.5),
        "dp_min": ordered[0]["direction_prob"],
        "dp_max": ordered[-1]["direction_prob"],
        "bottom_up": up_rate(bottom),
        "top_up": up_rate(top),
        "bottom_mean_ret": mean_ret(bottom),
        "top_mean_ret": mean_ret(top),
        "up_spread": up_rate(top) - up_rate(bottom),
        "ret_spread": mean_ret(top) - mean_ret(bottom),
        "quintile_up": [up_rate(q) for q in quintiles],
        # Monotone means every step up in confidence raises the realised
        # up-rate. Anything else is a head whose order carries no information.
        "monotonic": all(
            up_rate(quintiles[i]) <= up_rate(quintiles[i + 1])
            for i in range(len(quintiles) - 1)
        ),
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


# Measured from receipts, never a flat 0.65%: 0.3187% of notional plus a
# 0.004047 fixed leg per round trip.
ROUND_TRIP_PCT = 0.3187
ROUND_TRIP_FIXED = 0.004047


def total_cost_pct(pct_cost: float, fixed_cost: float, clip: float) -> float:
    """Round-trip cost as a PERCENT of notional, both legs included.

    The proportional leg is already a percent of notional. The fixed leg is
    DOLLARS and only becomes a percent once it is divided by the clip, which is
    why the clip belongs in this arithmetic and why a table that quotes
    "0.3187% + 0.004047 fixed" in its header and then subtracts only 0.3187 is
    under-billing every row. At the $5 clip this system actually trades, the
    fixed leg is another 0.0809% -- a quarter as large again as the whole
    proportional cost.
    """
    if clip <= 0:
        raise ValueError(f"clip must be positive, got {clip}")
    return pct_cost + (fixed_cost / clip) * 100.0


def window_regime(
    preds: List[Dict[str, Any]],
    stream: Dict[str, Tuple[List[float], List[float]]],
    lo: float,
    hi: float,
    max_abs_return: float,
) -> Dict[str, Any]:
    """UP / DOWN / FLAT for the window, over the symbols the sample actually uses.

    Classifying over EVERY symbol in ``market_stream`` reads FLAT on every window
    in this database, because most symbols there hold a seed price and never tick
    (memory: ``frozen-feed-dexscreener``). The regime that matters is the one the
    sampled ticks were drawn from, so this measures per-symbol first-to-last
    drift over the prediction symbols only, drops drifts beyond the same
    contamination guard the table uses, and reports the MEDIAN -- the mean over
    this feed reaches 8e7 percent on a single bad row.
    """
    wanted = {row["symbol"] for row in preds}
    drifts: List[float] = []
    for symbol in wanted:
        series = stream.get(symbol)
        if series is None:
            continue
        times, prices = series
        inside = [
            (t, p) for t, p in zip(times, prices) if lo <= t <= hi and p > 0
        ]
        if len(inside) < 10:
            continue
        drift = (inside[-1][1] - inside[0][1]) / inside[0][1]
        if abs(drift) > max_abs_return:
            continue
        drifts.append(drift * 100.0)
    if len(drifts) < 5:
        return {"regime": "UNKNOWN", "median_drift_pct": None, "symbols": len(drifts)}
    drifts.sort()
    median = _quantile(drifts, 0.5)
    if median > 0.3:
        regime = "UP"
    elif median < -0.3:
        regime = "DOWN"
    else:
        regime = "FLAT"
    return {"regime": regime, "median_drift_pct": median, "symbols": len(drifts)}


def tape_rows(
    stream: Dict[str, Tuple[List[float], List[float]]],
    lo: float,
    hi: float,
) -> List[Dict[str, Any]]:
    """Every priced ``market_stream`` tick in the window, as (symbol, ts) rows.

    The horizon table and ``window_regime`` both want "a set of (symbol, ts)
    points to price forward from". They were handed prediction snapshots because
    that is what the rest of this census works on -- but neither reads a
    prediction field, so the snapshot dependency bought nothing and cost every
    window where the head was silent. Rows carry no ``direction_prob``: anything
    that scores the HEAD must keep using ``preds``, and a caller that mixes them
    up gets a KeyError rather than a silent wrong answer.
    """
    rows: List[Dict[str, Any]] = []
    for symbol, (times, _prices) in stream.items():
        left = bisect.bisect_left(times, lo)
        right = bisect.bisect_left(times, hi)
        for ts in times[left:right]:
            rows.append({"symbol": symbol, "ts": ts})
    rows.sort(key=lambda r: r["ts"])
    return rows


def _print_horizon_table(
    preds: List[Dict[str, Any]],
    stream: Dict[str, Tuple[List[float], List[float]]],
    args: Any,
    lo: float,
    hi: float,
) -> int:
    """Can ANY horizon clear the round trip, even with a perfect direction call?

    The last column is a CEILING NOBODY CAN REACH: it assumes the direction is
    called correctly on every tick and the whole move is captured, then pays the
    cost once. Where that ceiling is negative, no head and no strategy can make
    the horizon pay -- the move is smaller than the toll.

    Measured 2026-09-10 over 24h it printed a negative ceiling at 5 and 10
    minutes, and the MEDIAN tick cleared the cost at no horizon under ~45 min.
    That is a direct finding against the "single-digit minutes" target.

    THE COST IS BOTH LEGS. Until pass 111 this billed only the proportional
    0.3187% and quoted the fixed leg in its header without ever charging it,
    which made every ceiling 0.0809 points too generous at the $5 clip and
    printed 15min as +0.0198 when the honest number is negative. It now charges
    ``total_cost_pct`` and honours --clip / --pct-cost / --fixed-cost, which the
    table previously accepted and ignored.

    ITS ROWS ARE TICKS, NOT PREDICTIONS -- see ``--tape-rows``. Everything this
    table computes is |forward return| from ``market_stream`` against a cost; it
    reads nothing whatsoever out of a prediction. But it was fed ``preds``, the
    snapshot rows, so it could only be measured where the head happened to have
    been emitting non-sentinel predictions. Measured pass 112: the 24h window
    ending 168h ago is an UP window (+5.572% median per-symbol drift) and the
    table printed NO ROWS AT ALL there -- n blank at every horizon and "0
    symbols, too few to classify" -- because no usable snapshot exists that far
    back. That is why "neither measured window is an UP window" survived three
    passes: it was never a choice of window, it was the instrument gating a
    pure-tape measurement behind the model's own history.
    """
    horizons = (5.0, 10.0, 15.0, 30.0, 60.0, 120.0)
    cost = total_cost_pct(args.pct_cost, args.fixed_cost, args.clip)
    regime = window_regime(preds, stream, lo, hi, args.max_abs_return)
    print("=" * 78)
    print("HORIZON vs COST FLOOR -- what a PERFECT direction call nets at each horizon")
    print("=" * 78)
    print(
        f"  cost basis {args.pct_cost}% of notional + {args.fixed_cost} fixed over a "
        f"${args.clip:.2f} clip = {cost:.4f}% ALL IN, measured from receipts"
    )
    drift = regime["median_drift_pct"]
    print(
        f"  window: {(hi - lo) / 3600.0:.1f}h ending {(time.time() - hi) / 3600.0:.1f}h ago"
        f"   REGIME {regime['regime']}"
        + (
            f" (median per-symbol drift {drift:+.3f}% over {regime['symbols']} symbols)"
            if drift is not None
            else f" ({regime['symbols']} symbols -- too few to classify)"
        )
    )
    print("  the last column is an unreachable CEILING: perfect direction, full capture")
    print()
    print(
        "  horizon      n   median|ret|%   %ticks>cost   mean|ret|%   perfect-oracle net%"
    )
    for horizon_min in horizons:
        moves: List[float] = []
        for row in preds:
            series = stream.get(row["symbol"])
            if series is None:
                continue
            base = price_at(series, row["ts"], args.tolerance_sec)
            fwd = price_at(series, row["ts"] + horizon_min * 60.0, args.tolerance_sec)
            if not base or not fwd or base <= 0:
                continue
            realised = (fwd - base) / base
            if abs(realised) > args.max_abs_return:
                continue
            moves.append(abs(realised) * 100.0)
        if not moves:
            continue
        moves.sort()
        n = len(moves)
        mean = sum(moves) / n
        over = sum(1 for v in moves if v > cost) / n * 100.0
        print(
            f"  {horizon_min:>5.0f}min {n:>6}   {_quantile(moves, 0.5):>10.4f}   "
            f"{over:>10.1f}%   {mean:>9.4f}   {mean - cost:>+16.4f}"
        )
    print()
    print("  A NEGATIVE ceiling means the horizon cannot pay at any skill level.")
    print("  Compare the MEDIAN column against the cost too -- the mean is skew-inflated")
    print("  by a few large movers, so a positive ceiling can still lose on the typical tick.")
    print("  ONE TAPE IS NOT A LAW: re-run with --end-hours-ago to land on a different")
    print("  regime and check the sign of the 5 and 10 minute rows there too.")
    print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--hours", type=float, default=24.0, help="total lookback")
    ap.add_argument(
        "--end-hours-ago",
        type=float,
        default=0.0,
        help=(
            "end the window this many hours before now instead of at now; the only "
            "way to land the census on a SECOND, DIFFERENT regime rather than "
            "re-reading the same rolling tape"
        ),
    )
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
    ap.add_argument(
        "--clip",
        type=float,
        default=5.0,
        help="notional per trade in USD; the 0.004047 fixed cost leg amortises over it",
    )
    ap.add_argument(
        "--pct-cost",
        type=float,
        default=0.3187,
        help="measured proportional round-trip cost, PERCENT of notional",
    )
    ap.add_argument(
        "--fixed-cost",
        type=float,
        default=0.004047,
        help="measured fixed round-trip cost in USD",
    )
    ap.add_argument(
        "--horizon-table",
        action="store_true",
        help="sweep horizons and print what a PERFECT direction call would net at each",
    )
    ap.add_argument(
        "--tape-rows",
        action="store_true",
        help=(
            "build the horizon table's rows from market_stream ticks rather than "
            "prediction snapshots. The table reads no prediction field, so this is "
            "the only way to land it on a window where the head was silent -- which "
            "is every window older than the current rolling one"
        ),
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    now = time.time()
    if args.end_hours_ago < 0:
        raise SystemExit("--end-hours-ago must not be negative")
    until = now - args.end_hours_ago * 3600.0
    since = until - args.hours * 3600.0
    horizon = args.horizon_min * 60.0

    conn = _connect(args.db)
    try:
        # The stream must reach PAST the window end far enough to price the
        # longest forward horizon, or every tick near the end loses its forward
        # price and the window silently measures only its own first half.
        stream = load_stream(conn, since - args.tolerance_sec)
        # --tape-rows needs no prediction, and loading them is not free: parsing
        # every organism_snapshots payload in a 24h window took ~7 minutes on
        # this box, which is a quarter of an agent pass spent on rows that are
        # then discarded. Skip it rather than pay it.
        preds = (
            []
            if (args.horizon_table and args.tape_rows)
            else [row for row in load_predictions(conn, since) if row["ts"] <= until]
        )
    finally:
        conn.close()

    if args.horizon_table:
        rows = preds
        if args.tape_rows:
            # The table reads no prediction field, so the honest row source for
            # it is the tape itself. Without this the table can only be measured
            # where the head was emitting predictions, which on this box is the
            # recent flat-to-down window and nowhere else.
            rows = tape_rows(stream, since, until)
            print(
                f"  ROWS FROM THE TAPE: {len(rows)} market_stream ticks in the window, "
                f"not {len(preds)} prediction snapshots"
            )
        return _print_horizon_table(rows, stream, args, since, until)

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

    # Relative to the WINDOW END, not to now: with --end-hours-ago the split
    # would otherwise fall outside the window entirely and put every row in one
    # era while still printing two.
    boundary = until - args.split_hours * 3600.0
    pre = [r for r in matched if r["ts"] < boundary]
    post = [r for r in matched if r["ts"] >= boundary]

    eras = {
        "pre_collapse": score_era(pre, args.flat_bp),
        "post_collapse": score_era(post, args.flat_bp),
    }
    ranks = {"pre_collapse": rank_profile(pre), "post_collapse": rank_profile(post)}
    money = {
        "pre_collapse": buy_rule_profile(pre, args.pct_cost, args.clip, args.fixed_cost),
        "post_collapse": buy_rule_profile(post, args.pct_cost, args.clip, args.fixed_cost),
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
        "rank_profile": ranks,
        "money_rule": money,
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
    print("  RANKING -- can a CALIBRATOR fix this? Calibration is a monotone remap: it")
    print("  moves the level and preserves the order. If the order carries nothing,")
    print("  there is nothing for a calibrator to rescue.")
    print("     era            realised up% by direction_prob quintile (low -> high)   monotone")
    for name in ("pre_collapse", "post_collapse"):
        rp = ranks[name]
        if not rp.get("n"):
            continue
        cells = "  ".join(f"{v * 100:5.1f}" for v in rp["quintile_up"])
        if rp["auc_inverted"]:
            call = "INVERTED -- more confidence, MORE wrong"
        elif rp["auc_beats_chance"]:
            call = "above chance (weak ranking skill)"
        else:
            call = "indistinguishable from chance"
        print(f"     {name:<14} {cells}      {'YES' if rp['monotonic'] else 'NO'}")
        print(
            f"       AUC {rp['auc']:.4f} +/- {rp['auc_se']:.4f} -- {call}. "
            "AUC decides this, not the monotone flag."
        )
        print(
            f"       top decile mean return {rp['top_mean_ret'] * 100:+.4f}% vs bottom "
            f"{rp['bottom_mean_ret'] * 100:+.4f}%  (spread {rp['ret_spread'] * 100:+.4f}%)"
        )
    print()
    # Direction is worthless if the move is smaller than the round trip. The
    # cost is the measured one from receipts (0.004047 fixed + 0.3187% of
    # notional), never a flat 0.65%. At the status command's $23.18 deployable
    # the fixed leg alone dominates, so quote both.
    post = eras["post_collapse"]
    if post.get("n"):
        # Honour the flags rather than the constants, and print the ALL-IN
        # count beside the proportional one: "moved further than the rate"
        # is a weaker statement than "cleared the round trip", and quoting
        # only the first is how the horizon table published a positive
        # 15-minute row that is really negative.
        pct_cost = args.pct_cost
        all_in = total_cost_pct(args.pct_cost, args.fixed_cost, args.clip)
        print(
            f"  COST FLOOR: median |{args.horizon_min:.0f}min return| is "
            f"{post['abs_ret_p50'] * 100:.4f}% against a {pct_cost:.4f}% proportional "
            f"round trip, {all_in:.4f}% ALL IN over the ${args.clip:.2f} clip."
        )
        post_rows = [r for r in matched if r["ts"] >= boundary]
        payers = sum(1 for r in post_rows if abs(r["realised"]) * 100 > pct_cost)
        payers_all_in = sum(1 for r in post_rows if abs(r["realised"]) * 100 > all_in)
        print(
            f"  {payers} of {post['n']} post-collapse ticks moved further than the "
            f"proportional cost alone ({payers / post['n'] * 100:.1f}%), and only "
            f"{payers_all_in} cleared the ALL-IN cost "
            f"({payers_all_in / post['n'] * 100:.1f}%) -- a perfect "
            "direction call on the rest still loses money."
        )
        print("  BUY LOW / SELL HIGH -- the head as a MONEY rule, not as an accuracy score.")
    print("  Precision counts a call as right only if the move CLEARED THE COST FLOOR;")
    print("  a directional hit smaller than the round trip is a loss. Baselines are the")
    print("  do-it-every-bar money rules in the SAME rows, never 0.5 and never the")
    print(f"  majority class. clip ${args.clip:.2f} -> floor {money['post_collapse'].get('floor', float('nan')) * 100:.4f}% round trip,")
    print(f"  {money['post_collapse'].get('one_leg', float('nan')) * 100:.4f}% one leg (exit).")
    print("     era             rule                  n   precision   net/trade   vs baseline")
    for name in ("pre_collapse", "post_collapse"):
        mr = money[name]
        if not mr.get("n"):
            continue
        for label, key, base_key in (
            ("head says UP (buy)", "buy", "buy_baseline"),
            ("top decile dp (buy)", "buy_top_decile", "buy_baseline"),
            ("buy EVERY bar", "buy_baseline", None),
            ("head says DOWN (exit)", "sell", "sell_baseline"),
            ("exit EVERY bar", "sell_baseline", None),
        ):
            cell = mr[key]
            if not cell.get("n"):
                continue
            base = mr[base_key] if base_key else None
            delta = (
                f"{(cell['precision'] - base['precision']) * 100:+7.2f}pp"
                if base and base.get("n")
                else "     -- "
            )
            print(
                f"     {name:<14} {label:<20} {cell['n']:>5}    {cell['precision'] * 100:6.2f}%   "
                f"{cell['net_per_trade'] * 100:+8.4f}%   {delta}"
            )
    print()
    print("  READ THIS AND NOT THE HIT RATE. A rule whose precision is at or below its")
    print("  every-bar baseline places no better trades however accurate it scores, and")
    print("  a NEGATIVE net/trade loses money on every one it places. The top-decile row")
    print("  is the only one that exploits the ORDERING; the head-says-UP row is the")
    print("  LEVEL. They are separate properties and they can disagree.")
    print()
    print(f"  VERDICT: {kind}")
    print(f"  {reason}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
