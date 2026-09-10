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
    buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for pred in preds:
        age = now - pred["ts"]
        if age < 0 or age >= hours * 3600.0:
            continue
        ret = forward_return(series, pred["symbol"], pred["ts"], horizon_sec)
        if ret is not None:
            buckets[int(age // bucket_sec)].append((pred[field], ret))

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
        windows.append(
            {
                "hours_ago": (index + 1) * bucket_sec / 3600.0,
                "n": len(rows),
                "up_share": ups / max(1, ups + downs),
                "all_bar_mean": all_bar,
                "top_mean": top_mean,
                "top_net": top_mean - cost,
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


def render_regimes(summary: Dict[str, Any], *, horizon_sec: int, cost: float) -> str:
    lines = [
        f"TOP-DECILE ENTRY, {horizon_sec // 60}m horizon, net of {cost * 100:.4f}% notional cost",
        "  window   up-share   all-bar    top-decile      NET   regime",
    ]
    for win in summary["windows"]:
        lines.append(
            f"  -{win['hours_ago']:4.0f}h    {win['up_share'] * 100:5.1f}%   "
            f"{win['all_bar_mean'] * 100:+.4f}%   {win['top_mean'] * 100:+.4f}%   "
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
        help="round-trip cost as a fraction of notional, from receipts",
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="symbol to drop, repeatable; excluding one still has to survive "
             "the regime split, which is where DRB-USDC's apparent edge died",
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
    print(
        render_regimes(
            regime_split(
                preds, series, now=now, hours=args.hours,
                horizon_sec=args.regime_horizon, cost=args.cost, field=args.field,
            ),
            horizon_sec=args.regime_horizon,
            cost=args.cost,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
