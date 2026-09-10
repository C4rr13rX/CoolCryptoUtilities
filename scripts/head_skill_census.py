"""Does the prediction head actually PREDICT, or is it merely confident?

THE LEVEL OF ``direction_prob`` IS NOT ITS SKILL, AND HERE THEY MOVED IN
OPPOSITE DIRECTIONS.

Backlog item 618d4c4b was filed as "the prediction head collapsed ~12h ago:
direction_prob p50 0.7833 -> 0.0317", with an acceptance criterion asking for
the p50 to be restored above 0.5. Measured 2026-09-10 13:40 by this script
over 6200 predictions joined to the realised tape, that criterion would have
re-opened the ghost lane onto the worst signal in the window::

                       last 6h ("collapsed")   12-24h ago ("healthy")
    direction_prob p50        0.1049                   0.4900
    price_mu p50             -0.1902                  -1.0627
    AUC vs realised  5m       0.5594                   0.4009
    AUC vs realised 15m       0.5950                   0.3834
    AUC vs realised 30m       0.5607                   0.3542

An AUC of 0.35-0.40, holding at every horizon, is a head that is reliably
WRONG about direction -- worse than a coin, not merely uninformative. The 141
ticks that cleared both scheduler floors in that 24h window were all drawn
from it. The "collapsed" head is the only one of the two with any skill at
all.

WHY THE BROKEN HEAD LOOKED BULLISH. ``price_mu`` p50 of -1.0627 is the model
predicting a -106% price move, which no market delivers; it is the foreign-row
saturation that ``trading/data_loader.sanitize_model_price_window`` repairs.
That saturation drove ``direction_prob`` UP. Cleaning the served window is
what dropped the level -- and it is what raised the skill. Level fell, skill
rose, because the level was contamination.

So a census that reads ``direction_prob`` alone cannot tell a recovering head
from a poisoned one. This one joins every prediction to what the tape actually
did next and scores the ORDERING, which is the only question that decides
whether an entry gate should ever fire.

WHAT AUC MEANS HERE. Probability that a randomly chosen prediction before an
UP move carried a higher ``direction_prob`` than one before a DOWN move. 0.5
is no information. Below 0.5 is an inverted head. The standard error is
roughly ``0.5 / sqrt(min(n_up, n_down))``, printed beside each row, so a
reader can tell 0.56 on 700 samples from 0.56 on 12.

READ IT BEFORE RESTORING A HEAD LEVEL. A number recovering toward 0.5 from
below is not automatically progress, and this repo has already paid for
treating confidence as correctness once (the brain's confidence gate: +0.030
train, -0.002 held-out).

Usage::

    python -X utf8 scripts/head_skill_census.py [--hours 26] [--split 6]
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sqlite3
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_DB = os.path.join("storage", "trading_cache.db")

#: Horizons scored, in seconds. One horizon can flatter a head by accident --
#: a 5m read is dominated by the spread on this feed -- so the verdict below
#: only calls a head inverted when EVERY horizon agrees. 5/15/30 minutes is
#: also the band the loop trades in ("round trips resolving in single-digit to
#: tens of minutes").
HORIZONS_SEC: Tuple[int, ...] = (300, 900, 1800)

#: A realised move larger than this is a feed artifact, not a price. The same
#: bound the tradeable-book work uses: a 50% move inside 30 minutes on these
#: pairs has always been a denomination flip or a foreign row, and leaving one
#: in moves an AUC by more than the effect being measured.
_MAX_PLAUSIBLE_ABS_RETURN = 0.5

#: ``bot.py::_summarise_predictions`` seeds its summary with exit_conf 0.5,
#: direction_prob 0.5, net_margin 0.0 and only overwrites what the model
#: actually returned. A row still carrying that exact pair is the
#: NO-PREDICTION SENTINEL -- no head ran -- and counting it as a prediction
#: reports a collapsed head sitting at its neutral ceiling.
def _is_no_prediction_sentinel(pred: Dict[str, Any]) -> bool:
    return pred.get("direction_prob") == 0.5 and pred.get("net_margin") in (0.0, None)


def load_price_series(
    db_path: str = DEFAULT_DB, *, hours: float, now: Optional[float] = None
) -> Dict[str, List[Tuple[float, float]]]:
    """Every symbol's price track in the window, ascending by ts."""
    cutoff = (now if now is not None else time.time()) - float(hours) * 3600.0
    series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT symbol, ts, price FROM market_stream WHERE ts > ? "
            "ORDER BY symbol, ts",
            (float(cutoff),),
        )
        for symbol, ts, price in rows:
            if symbol and price and price > 0:
                series[str(symbol)].append((float(ts), float(price)))
    finally:
        conn.close()
    return dict(series)


def forward_return(
    series: Dict[str, List[Tuple[float, float]]],
    symbol: str,
    ts: float,
    horizon_sec: float,
) -> Optional[float]:
    """What the tape did over ``horizon_sec`` after ``ts``, or None.

    Anchored on the first quote at or after ``ts`` rather than on ``ts``
    itself, so a prediction made between two ticks is scored against a price
    it could actually have traded at.
    """
    track = series.get(symbol)
    if not track:
        return None
    start = bisect.bisect_left(track, (ts,))
    if start >= len(track):
        return None
    anchor_ts, anchor_px = track[start]
    end = start
    while end < len(track) and track[end][0] - anchor_ts < horizon_sec:
        end += 1
    if end >= len(track):
        return None
    ret = (track[end][1] - anchor_px) / anchor_px
    if abs(ret) > _MAX_PLAUSIBLE_ABS_RETURN:
        return None
    return ret


def auc(scored: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, int, int]]:
    """Rank-order skill of a score against the sign of a realised return.

    Returns ``(auc, n_up, n_down)``, or None when either class is empty --
    an AUC needs both, and a window where the tape only ever rose says
    nothing about a head's ordering.
    """
    ups = sorted(score for score, ret in scored if ret > 0)
    downs = sorted(score for score, ret in scored if ret < 0)
    if not ups or not downs:
        return None
    wins = 0.0
    for score in ups:
        lo = bisect.bisect_left(downs, score)
        hi = bisect.bisect_right(downs, score)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(ups) * len(downs)), len(ups), len(downs)


def auc_stderr(n_up: int, n_down: int) -> float:
    """The crude ``0.5 / sqrt(min(n))`` bound, so a reader can size an AUC.

    Deliberately not the Hanley-McNeil estimator: this is printed next to a
    number that decides whether to trust a head at all, and an approximation
    that is always conservative is the right one to hand a reader.
    """
    smaller = min(int(n_up), int(n_down))
    return 0.5 / math.sqrt(smaller) if smaller > 0 else float("nan")


def load_predictions(
    db_path: str = DEFAULT_DB, *, hours: float, now: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Every real prediction in the window, tagged with its symbol.

    ``organism_snapshots`` carries the head under ``payload['prediction']``
    and the symbol it was made for under ``payload['sample']['symbol']`` --
    the prediction block itself has no symbol, so a census that reads only
    that block cannot join to the tape at all.
    """
    cutoff = (now if now is not None else time.time()) - float(hours) * 3600.0
    out: List[Dict[str, Any]] = []
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ts, payload FROM organism_snapshots WHERE ts > ? ORDER BY ts",
            (float(cutoff),),
        )
        for ts, payload in rows:
            try:
                blob = json.loads(payload)
            except (TypeError, ValueError):
                continue
            pred = blob.get("prediction")
            sample = blob.get("sample")
            if not isinstance(pred, dict) or not isinstance(sample, dict):
                continue
            symbol = sample.get("symbol")
            if not symbol or "direction_prob" not in pred:
                continue
            if _is_no_prediction_sentinel(pred):
                continue
            out.append(
                {
                    "ts": float(ts),
                    "symbol": str(symbol),
                    "direction_prob": float(pred["direction_prob"]),
                    "direction_prob_raw": (
                        float(pred["direction_prob_raw"])
                        if pred.get("direction_prob_raw") is not None
                        else float(pred["direction_prob"])
                    ),
                    "price_mu": pred.get("price_mu"),
                }
            )
    finally:
        conn.close()
    return out


def score_window(
    preds: Sequence[Dict[str, Any]],
    series: Dict[str, List[Tuple[float, float]]],
    *,
    field: str = "direction_prob_raw",
    horizons: Sequence[int] = HORIZONS_SEC,
) -> Dict[str, Any]:
    """Level AND skill for one window. Both, because either alone misleads."""
    levels = [p[field] for p in preds]
    mus = [float(p["price_mu"]) for p in preds if p.get("price_mu") is not None]
    result: Dict[str, Any] = {
        "n": len(preds),
        "level_p50": statistics.median(levels) if levels else float("nan"),
        "price_mu_p50": statistics.median(mus) if mus else float("nan"),
        "horizons": {},
    }
    for horizon in horizons:
        scored = []
        for pred in preds:
            ret = forward_return(series, pred["symbol"], pred["ts"], horizon)
            if ret is not None:
                scored.append((pred[field], ret))
        measured = auc(scored)
        if measured is None:
            result["horizons"][horizon] = None
            continue
        value, n_up, n_down = measured
        result["horizons"][horizon] = {
            "auc": value,
            "n": len(scored),
            "n_up": n_up,
            "n_down": n_down,
            "stderr": auc_stderr(n_up, n_down),
            "realised_p50": statistics.median([r for _, r in scored]),
        }
    return result


def verdict(window: Dict[str, Any]) -> str:
    """INVERTED / SKILLED / NO INFORMATION, from the horizons that agree.

    A head is only called inverted when every scored horizon puts it more
    than one standard error below 0.5, and only called skilled on the mirror
    condition. Anything else is no information -- which is the honest reading
    of almost every head this repo has measured.
    """
    scored = [h for h in window["horizons"].values() if h]
    if not scored:
        return "UNSCORED (no horizon had both an up and a down move)"
    if all(h["auc"] < 0.5 - h["stderr"] for h in scored):
        return "INVERTED -- this head is reliably WRONG about direction"
    if all(h["auc"] > 0.5 + h["stderr"] for h in scored):
        return "SKILLED -- ordering beats a coin at every horizon"
    return "NO INFORMATION -- ordering is inside noise at some horizon"


#: The round trip's FIXED leg, in dollars, from receipts. It does not shrink
#: with the trade, so as a FRACTION of notional it is whatever ``fixed / clip``
#: happens to be -- 0.0405% on a $10 clip and 0.4047% on a $1 one. A census
#: that charges only the 0.3187% rate is under-billing every entry it scores,
#: and under-billing is how a rule that loses money reads as an edge.
ROUND_TRIP_FIXED_USD = 0.004047

#: ``LIVE_MIN_CLIP_USD`` at trading/pipeline.py:5427. The clip the live lane
#: would actually place, and therefore the clip the fixed leg amortises over.
DEFAULT_CLIP_USD = 10.0

#: The percentile family swept by ``--sweep``. A single decile is ONE POINT in
#: this family: "the top 10% does not pay" is not the same claim as "no rank
#: threshold pays", and the acceptance criterion on item 618d4c4b asks the
#: second. Tightening the threshold is the standard rescue for a weak-but-real
#: ordering -- if the head's AUC 0.56-0.60 carries any tradeable signal, it
#: shows up as the net improving monotonically toward the tight end.
SWEEP_PERCENTILES: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50)


def total_cost_fraction(
    *, notional_rate: float = 0.003187, clip_usd: float = DEFAULT_CLIP_USD
) -> float:
    """The whole round-trip cost as a fraction of notional.

    BOTH LEGS, because this repo has already shipped a fee in the wrong
    currency and a rate subtracted from a dollar amount. The fixed leg is
    dollars and the rate is a fraction; they only add after the fixed leg is
    divided by the clip it is charged against.
    """
    if clip_usd <= 0:
        raise ValueError("clip_usd must be positive: the fixed leg is amortised over it")
    return float(notional_rate) + ROUND_TRIP_FIXED_USD / float(clip_usd)


def bucket_scored_returns(
    preds: Sequence[Dict[str, Any]],
    series: Dict[str, List[Tuple[float, float]]],
    *,
    now: float,
    hours: float,
    bucket_sec: float = 2 * 3600.0,
    horizon_sec: int = 900,
    field: str = "direction_prob_raw",
) -> Dict[int, List[Tuple[float, float]]]:
    """``{bucket_index: [(head_score, realised_return), ...]}``.

    Split out so the percentile sweep joins predictions to the tape ONCE
    rather than once per threshold. The join is the expensive half and doing
    it seven times would also invite the two arms to drift apart -- every
    threshold in the sweep has to be scored on identical rows for the
    comparison between them to mean anything.
    """
    buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for pred in preds:
        age = now - pred["ts"]
        if age < 0 or age >= hours * 3600.0:
            continue
        ret = forward_return(series, pred["symbol"], pred["ts"], horizon_sec)
        if ret is not None:
            buckets[int(age // bucket_sec)].append((pred[field], ret))
    return buckets


def regime_split(
    preds: Sequence[Dict[str, Any]],
    series: Dict[str, List[Tuple[float, float]]],
    *,
    now: float,
    hours: float,
    bucket_sec: float = 2 * 3600.0,
    horizon_sec: int = 900,
    decile: float = 0.10,
    cost: float = 0.003187,
    field: str = "direction_prob_raw",
    min_rows: int = 80,
) -> Dict[str, Any]:
    """Would entering on the head's top decile have paid, window by window?

    THIS IS THE DEFAULT OUTPUT BECAUSE POOLING IS THE TRAP.

    Twice in one pass on 2026-09-10 a pooled read produced an edge that this
    split killed. Excluding the over-concentrated symbol DRB-USDC, the pooled
    top decile came out at +0.1042% net of cost at 15m and +0.2931% at 30m,
    both beating the buy-every-bar baseline -- a clean, shippable-looking
    result. Split into twelve 2h windows, the SAME data gave::

        UP windows   (7)   mean top-decile net -0.0725%   2 of 7 positive
        DOWN windows (5)   mean top-decile net -0.5982%   0 of 5 positive

    The pooled positive was one recent 2h window (+0.4460%, 62.7% up-share)
    carrying the average. A long-only rule flatters itself in an up window,
    and this repo has already shipped a fake 78% and a fake +0.9067% that way.

    Each window is classified by ITS OWN realised all-bar mean rather than by
    a global regime label, so the split cannot be gamed by choosing where the
    boundary falls. What matters in the output is the COUNT of net-positive
    windows in each regime, not the mean across them -- one window with a big
    number is exactly what the mean hides.
    """
    buckets = bucket_scored_returns(
        preds, series, now=now, hours=hours, bucket_sec=bucket_sec,
        horizon_sec=horizon_sec, field=field,
    )
    windows: List[Dict[str, Any]] = []
    for index in sorted(buckets):
        rows = buckets[index]
        if len(rows) < min_rows:
            continue
        ups = sum(1 for _, ret in rows if ret > 0)
        downs = sum(1 for _, ret in rows if ret < 0)
        all_bar = statistics.mean([ret for _, ret in rows])
        ordered = sorted(rows, key=lambda item: -item[0])
        take = max(1, int(len(ordered) * decile))
        top_mean = statistics.mean([ret for _, ret in ordered[:take]])
        # THE MIRROR, and it is the cheapest honest test of the ordering claim.
        # If the head's AUC carries real information, the bars it scores
        # LOWEST must fall relative to the bars it scores highest. A spread
        # near zero says the AUC is ordering noise; a NEGATIVE spread says the
        # ordering is backwards at the tails whatever the pooled AUC reads.
        bottom_mean = statistics.mean([ret for _, ret in ordered[-take:]])
        windows.append(
            {
                "hours_ago": (index + 1) * bucket_sec / 3600.0,
                "n": len(rows),
                "up_share": ups / max(1, ups + downs),
                "all_bar_mean": all_bar,
                "top_mean": top_mean,
                "top_net": top_mean - cost,
                "bottom_mean": bottom_mean,
                "tail_spread": top_mean - bottom_mean,
                "regime": "UP" if all_bar > 0 else "DOWN",
            }
        )

    summary: Dict[str, Any] = {"windows": windows, "regimes": {}}
    for regime in ("UP", "DOWN"):
        subset = [w for w in windows if w["regime"] == regime]
        summary["regimes"][regime] = {
            "n_windows": len(subset),
            "n_positive": sum(1 for w in subset if w["top_net"] > 0),
            "mean_net": statistics.mean([w["top_net"] for w in subset]) if subset else float("nan"),
        }
    return summary


def regime_verdict(summary: Dict[str, Any]) -> str:
    """An edge must hold in an UP window AND a DOWN window, or it is not one.

    The standing bar for this loop: "never report a single window -- a
    long-only rule flatters itself in an up window". A majority of windows
    positive in BOTH regimes is the weakest claim worth making here.
    """
    up = summary["regimes"].get("UP", {})
    down = summary["regimes"].get("DOWN", {})
    if not up.get("n_windows") or not down.get("n_windows"):
        return "UNPROVEN -- the window carries only one regime, so no claim is possible"
    up_ok = up["n_positive"] * 2 > up["n_windows"]
    down_ok = down["n_positive"] * 2 > down["n_windows"]
    if up_ok and down_ok:
        return "EDGE HOLDS IN BOTH REGIMES -- the weakest claim worth making"
    return (
        "NO EDGE -- positive in "
        f"{up['n_positive']}/{up['n_windows']} up and "
        f"{down['n_positive']}/{down['n_windows']} down windows"
    )


def percentile_sweep(
    preds: Sequence[Dict[str, Any]],
    series: Dict[str, List[Tuple[float, float]]],
    *,
    now: float,
    hours: float,
    horizon_sec: int = 900,
    cost: float,
    field: str = "direction_prob_raw",
    percentiles: Sequence[float] = SWEEP_PERCENTILES,
    bucket_sec: float = 2 * 3600.0,
    min_rows: int = 80,
) -> Dict[str, Any]:
    """Does ANY rank threshold on the head extract a tradeable ordering?

    THE QUESTION ITEM 618d4c4b ASKS, AND WHY ONE DECILE CANNOT ANSWER IT.

    The head's LEVEL is miscalibrated against ``SCHEDULER_MIN_DIRECTION_PROB``
    = 0.6 -- a p50 of 0.11 clears no absolute floor -- while its ORDERING
    scored AUC 0.56-0.60 against the realised tape over the same rows. A rank
    threshold is the obvious way to spend an ordering without touching a
    level: enter on the top N% of the head's scores in the window, whatever
    number those scores happen to be.

    ``net_margin >= 0`` IS UNTOUCHED BY ALL OF THIS. That is the cost test,
    not the direction test, and it keeps its floor: this sweep asks only
    whether direction can be ranked, and every threshold below is still
    charged the full round trip.

    A weak-but-real ordering has a signature: the net improves as the
    threshold tightens, because a tighter cut keeps a higher-quality slice.
    An ordering that is noise does the opposite -- the net wanders and the
    variance explodes as the sample shrinks. The verdict reads the whole
    family rather than the best member of it, because the best of seven
    thresholds is the maximum of seven noisy draws and picking it is how a
    sweep manufactures an edge.
    """
    buckets = bucket_scored_returns(
        preds, series, now=now, hours=hours, bucket_sec=bucket_sec,
        horizon_sec=horizon_sec, field=field,
    )
    usable = {index: rows for index, rows in buckets.items() if len(rows) >= min_rows}

    rungs: List[Dict[str, Any]] = []
    for pct in percentiles:
        per_regime: Dict[str, List[float]] = {"UP": [], "DOWN": []}
        for rows in usable.values():
            all_bar = statistics.mean([ret for _, ret in rows])
            ordered = sorted(rows, key=lambda item: -item[0])
            take = max(1, int(len(ordered) * pct))
            net = statistics.mean([ret for _, ret in ordered[:take]]) - cost
            per_regime["UP" if all_bar > 0 else "DOWN"].append(net)
        rung: Dict[str, Any] = {"pct": pct, "regimes": {}}
        for regime, nets in per_regime.items():
            rung["regimes"][regime] = {
                "n_windows": len(nets),
                "n_positive": sum(1 for net in nets if net > 0),
                "mean_net": statistics.mean(nets) if nets else float("nan"),
            }
        up, down = rung["regimes"]["UP"], rung["regimes"]["DOWN"]
        rung["clears_both"] = bool(
            up["n_windows"]
            and down["n_windows"]
            and up["n_positive"] * 2 > up["n_windows"]
            and down["n_positive"] * 2 > down["n_windows"]
        )
        rungs.append(rung)
    return {"rungs": rungs, "n_windows": len(usable), "cost": cost}


def sweep_verdict(sweep: Dict[str, Any]) -> str:
    """NO RANK THRESHOLD PAYS, or which ones do -- never "the best one".

    Naming the winning rung of a sweep is the same error as reporting a
    single window: seven thresholds scored on one dataset produce a maximum
    whether or not any signal is present. So the verdict counts how many
    rungs clear BOTH regimes and says plainly when the answer is none.
    """
    clearing = [rung for rung in sweep["rungs"] if rung["clears_both"]]
    if not clearing:
        return (
            "NO RANK THRESHOLD PAYS -- none of the "
            f"{len(sweep['rungs'])} percentile cuts is net-positive in a majority "
            "of UP windows AND a majority of DOWN windows. The head's ordering "
            "does not survive the round-trip cost at any tightness, so a "
            "percentile entry gate would open the lane onto a losing rule."
        )
    names = ", ".join(f"top {rung['pct'] * 100:g}%" for rung in clearing)
    return (
        f"{len(clearing)} of {len(sweep['rungs'])} thresholds clear both regimes "
        f"({names}). Treat this as a HYPOTHESIS, not an edge: the cuts were "
        "chosen after seeing the data and need a held-out window before any "
        "gate is built on one."
    )


#: Horizons scanned by ``--grid``. Chosen to bracket the point where a
#: PERFECT oracle stops losing: ``scripts/head_vs_realised_census.py
#: --horizon-table`` puts the ceiling at -0.1219% at 5min, -0.0353% at 10min
#: and +0.0197% at 15min, rising to +1.2531% at 120min. Below 15min no skill
#: can pay, so a head failing there proves nothing about the head. This grid
#: asks the only remaining question: does OUR head pay where headroom exists?
GRID_HORIZONS_SEC: Tuple[int, ...] = (900, 1800, 3600, 7200)


def horizon_threshold_grid(
    preds: Sequence[Dict[str, Any]],
    series: Dict[str, List[Tuple[float, float]]],
    *,
    now: float,
    hours: float,
    cost: float,
    field: str = "direction_prob_raw",
    horizons: Sequence[int] = GRID_HORIZONS_SEC,
    percentiles: Sequence[float] = SWEEP_PERCENTILES,
    bucket_sec: float = 2 * 3600.0,
    min_rows: int = 80,
) -> Dict[str, Any]:
    """Every (horizon, rank threshold) cell, and how many clear both regimes.

    WHY THIS IS SCANNED RATHER THAN CHOSEN. A 15m sweep alone cannot settle
    the head, because a perfect oracle also loses at 15m-and-below on this
    feed -- the cost floor eats the move. Failing where nothing can succeed is
    not evidence about the head. Longer horizons are where a real ordering
    would show, so the honest test scans them.

    AND WHY THE COUNT IS THE ANSWER, NOT THE BEST CELL. Scanning N cells and
    reporting the winner is the purest form of the mistake this file exists to
    prevent: the maximum of N noisy draws rises with N whether or not any
    signal is present. Each cell is a coin-flip-ish pair of majority tests, so
    a grid of this size yields several clearing cells FROM NOISE ALONE. The
    verdict therefore compares the observed count against that chance
    expectation instead of pointing at a cell.
    """
    cells: List[Dict[str, Any]] = []
    for horizon in horizons:
        sweep = percentile_sweep(
            preds, series, now=now, hours=hours, horizon_sec=horizon, cost=cost,
            field=field, percentiles=percentiles, bucket_sec=bucket_sec,
            min_rows=min_rows,
        )
        for rung in sweep["rungs"]:
            cells.append(
                {
                    "horizon_sec": horizon,
                    "pct": rung["pct"],
                    "clears_both": rung["clears_both"],
                    "up": rung["regimes"]["UP"],
                    "down": rung["regimes"]["DOWN"],
                }
            )
    clearing = [cell for cell in cells if cell["clears_both"]]
    # Each cell must win a majority of UP windows AND a majority of DOWN
    # windows. Treating a window as a fair coin makes each majority test
    # roughly 1/2, so a cell clears by chance about 1/4 of the time. That is
    # the bar an observed count has to BEAT, not merely reach.
    expected_by_chance = 0.25 * len(cells)
    return {
        "cells": cells,
        "n_cells": len(cells),
        "n_clearing": len(clearing),
        "expected_by_chance": expected_by_chance,
        "cost": cost,
    }


def grid_verdict(grid: Dict[str, Any]) -> str:
    """Beats chance, or does not. Never "the best cell was +x%"."""
    observed, expected = grid["n_clearing"], grid["expected_by_chance"]
    if observed == 0:
        return (
            f"NO CELL PAYS -- 0 of {grid['n_cells']} (horizon, threshold) "
            f"combinations clear both regimes, against {expected:.1f} expected "
            "from chance alone. The head has no tradeable direction signal at "
            "any horizon or tightness scanned."
        )
    if observed <= expected:
        return (
            f"INDISTINGUISHABLE FROM CHANCE -- {observed} of {grid['n_cells']} "
            f"cells clear both regimes, against {expected:.1f} expected from "
            "chance alone. Naming the best of them would be reporting the "
            "maximum of a noisy scan as an edge."
        )
    return (
        f"ABOVE CHANCE -- {observed} of {grid['n_cells']} cells clear both "
        f"regimes against {expected:.1f} expected. This is a HYPOTHESIS and not "
        "an edge: the cells were chosen after seeing the data and need a "
        "held-out window before any gate is built on one."
    )


def render_grid(grid: Dict[str, Any]) -> str:
    horizons = sorted({cell["horizon_sec"] for cell in grid["cells"]})
    percentiles = sorted({cell["pct"] for cell in grid["cells"]})
    lines = [
        f"HORIZON x RANK-THRESHOLD GRID, net of {grid['cost'] * 100:.4f}% round-trip cost",
        "  cells show UP/DOWN windows net-positive; * marks a cell clearing BOTH",
        "  threshold  " + "".join(f"{h // 60:>14d}m" for h in horizons),
    ]
    lookup = {(cell["horizon_sec"], cell["pct"]): cell for cell in grid["cells"]}
    for pct in percentiles:
        row = f"  top {pct * 100:5.1f}%  "
        for horizon in horizons:
            cell = lookup.get((horizon, pct))
            if cell is None:
                row += f"{'--':>15s}"
                continue
            mark = "*" if cell["clears_both"] else " "
            row += (
                f"{cell['up']['n_positive']}/{cell['up']['n_windows']}"
                f",{cell['down']['n_positive']}/{cell['down']['n_windows']}{mark}"
            ).rjust(15)
        lines.append(row)
    lines.append("")
    lines.append(f"  VERDICT: {grid_verdict(grid)}")
    return "\n".join(lines)


def render_sweep(sweep: Dict[str, Any], *, horizon_sec: int) -> str:
    lines = [
        f"RANK-THRESHOLD SWEEP, {horizon_sec // 60}m horizon, "
        f"net of {sweep['cost'] * 100:.4f}% total round-trip cost",
        f"  {sweep['n_windows']} windows scored. Does entering on the head's top N% pay?",
        "  threshold      UP windows net-positive        DOWN windows net-positive",
    ]
    for rung in sweep["rungs"]:
        up, down = rung["regimes"]["UP"], rung["regimes"]["DOWN"]
        lines.append(
            f"  top {rung['pct'] * 100:5.1f}%      "
            f"{up['n_positive']}/{up['n_windows']}  mean {up['mean_net'] * 100:+8.4f}%      "
            f"{down['n_positive']}/{down['n_windows']}  mean {down['mean_net'] * 100:+8.4f}%"
        )
    lines.append("")
    lines.append(f"  VERDICT: {sweep_verdict(sweep)}")
    return "\n".join(lines)


def render_regimes(summary: Dict[str, Any], *, horizon_sec: int, cost: float) -> str:
    lines = [
        f"TOP-DECILE ENTRY, {horizon_sec // 60}m horizon, net of {cost * 100:.4f}% round-trip cost",
        "  window   up-share   all-bar    top-dec   bot-dec    SPREAD       NET   regime",
    ]
    for win in summary["windows"]:
        lines.append(
            f"  -{win['hours_ago']:4.0f}h    {win['up_share'] * 100:5.1f}%   "
            f"{win['all_bar_mean'] * 100:+.4f}%   {win['top_mean'] * 100:+.4f}%   "
            f"{win['bottom_mean'] * 100:+.4f}%   {win['tail_spread'] * 100:+.4f}%   "
            f"{win['top_net'] * 100:+.4f}%   {win['regime']}"
        )
    lines.append("")
    for regime in ("UP", "DOWN"):
        slot = summary["regimes"][regime]
        if not slot["n_windows"]:
            continue
        lines.append(
            f"  {regime:5s} windows: {slot['n_positive']}/{slot['n_windows']} net-positive"
            f"   mean net {slot['mean_net'] * 100:+.4f}%"
        )
    lines.append(f"  VERDICT: {regime_verdict(summary)}")
    return "\n".join(lines)


def render(windows: Sequence[Tuple[str, Dict[str, Any]]]) -> str:
    lines: List[str] = []
    for label, win in windows:
        lines.append(
            f"{label}   n={win['n']}   direction_prob p50={win['level_p50']:.4f}"
            f"   price_mu p50={win['price_mu_p50']:.4f}"
        )
        for horizon in sorted(win["horizons"]):
            slot = win["horizons"][horizon]
            if not slot:
                lines.append(f"    {horizon // 60:3d}m   unscored")
                continue
            lines.append(
                f"    {horizon // 60:3d}m   AUC {slot['auc']:.4f} +/- {slot['stderr']:.4f}"
                f"   n={slot['n']:5d} (up {slot['n_up']} / down {slot['n_down']})"
                f"   realised p50 {slot['realised_p50'] * 100:+.4f}%"
            )
        lines.append(f"    VERDICT: {verdict(win)}")
        lines.append("")
    lines.append(
        "A HEAD'S LEVEL IS NOT ITS SKILL. Restoring direction_prob toward 0.5 is "
        "progress only if the AUC goes WITH it; on 2026-09-10 the level and the "
        "skill moved in opposite directions, because the level was foreign-row "
        "saturation rather than opinion."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--hours", type=float, default=26.0)
    parser.add_argument(
        "--split",
        type=float,
        default=6.0,
        help="hours ago separating the RECENT window from the EARLIER one",
    )
    parser.add_argument("--field", default="direction_prob_raw")
    parser.add_argument(
        "--regime-horizon", type=int, default=900,
        help="horizon in seconds for the regime split (default 900)",
    )
    parser.add_argument(
        "--cost", type=float, default=0.003187,
        help="round-trip cost RATE as a fraction of notional, from receipts. "
             "The fixed $0.004047 leg is added on top, amortised over --clip",
    )
    parser.add_argument(
        "--clip", type=float, default=DEFAULT_CLIP_USD,
        help="clip in dollars the fixed $0.004047 leg is charged against "
             f"(default {DEFAULT_CLIP_USD:g}, LIVE_MIN_CLIP_USD). A smaller clip "
             "makes every entry more expensive, not less",
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="symbol to drop, repeatable; excluding one still has to survive "
             "the regime split, which is where DRB-USDC's apparent edge died",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="scan every (horizon, rank threshold) cell. Slower, and the "
             "answer it gives is a COUNT against chance, never a best cell",
    )
    args = parser.parse_args(argv)

    now = time.time()
    series = load_price_series(args.db, hours=args.hours, now=now)
    preds = load_predictions(args.db, hours=args.hours, now=now)
    if args.exclude:
        dropped = set(args.exclude)
        preds = [p for p in preds if p["symbol"] not in dropped]
    recent = [p for p in preds if now - p["ts"] < args.split * 3600.0]
    earlier = [p for p in preds if now - p["ts"] >= args.split * 3600.0]

    print(f"{len(preds)} real predictions over {args.hours:g}h, "
          f"{len(series)} symbols on the tape\n")
    print(
        render(
            [
                (f"LAST {args.split:g}h", score_window(recent, series, field=args.field)),
                (f"EARLIER (>{args.split:g}h ago)", score_window(earlier, series, field=args.field)),
            ]
        )
    )
    print()
    # PRINTED EVERY RUN, NOT BEHIND A FLAG. The AUCs above are pooled, and a
    # pooled read on this feed is the trap -- it produced two apparent edges
    # on 2026-09-10 that this split killed. A reader who sees only the block
    # above will believe the first one.
    # BOTH LEGS OF THE ROUND TRIP. Charging only the 0.3187% rate under-bills
    # every entry scored below by $0.004047, which on a $10 clip is another
    # 0.0405% and on a $1 clip is another 0.4047% -- larger than the effect
    # being measured. Printed so a reader can see which cost produced the
    # verdict rather than having to assume one.
    cost = total_cost_fraction(notional_rate=args.cost, clip_usd=args.clip)
    print(
        f"ROUND-TRIP COST {cost * 100:.4f}% of notional = {args.cost * 100:.4f}% rate "
        f"+ ${ROUND_TRIP_FIXED_USD:g} fixed amortised over a ${args.clip:g} clip\n"
    )
    print(
        render_regimes(
            regime_split(
                preds, series, now=now, hours=args.hours,
                horizon_sec=args.regime_horizon, cost=cost, field=args.field,
            ),
            horizon_sec=args.regime_horizon,
            cost=cost,
        )
    )
    print()
    # THE ACCEPTANCE CRITERION ON 618d4c4b, ANSWERED EVERY RUN. "The top decile
    # does not pay" is one point; "no rank threshold pays" is the claim that
    # decides whether a percentile entry gate should be built at all.
    print(
        render_sweep(
            percentile_sweep(
                preds, series, now=now, hours=args.hours,
                horizon_sec=args.regime_horizon, cost=cost, field=args.field,
            ),
            horizon_sec=args.regime_horizon,
        )
    )
    if args.grid:
        print()
        print(
            render_grid(
                horizon_threshold_grid(
                    preds, series, now=now, hours=args.hours, cost=cost,
                    field=args.field,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
