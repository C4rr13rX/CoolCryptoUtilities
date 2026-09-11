#!/usr/bin/env python3
"""Where in the 30-240 minute band does the L1 motif rule stop losing money?

WHY THIS EXISTS, AND WHAT IT IS NOT RE-RUNNING. Pass 114 (45a2959) measured the
L1 motif -> trough rule over 102 base corpora held at a 3600s median cadence and
got a clean, well-powered NEGATIVE: DOWN +0.10pp on 194 trades, UP -1.97pp on 6.
That result is not in dispute and this harness does not re-run it. What it did
not settle is its own SCOPE. On a 3600s cadence ``--horizon 12`` is TWELVE
HOURS, so pass 114 ruled out a twelve-hour rule and ruled out nothing between
thirty minutes and four hours -- which is the band that matters, because
[15cc71d4] measured that the share of ticks whose realised move outruns the
0.3592% live cost floor is 17.8% at 5 minutes, 53.2% at 60 and 80.3% at 240.
The cost floor and the hit rate cross somewhere in that band.

AND THE BYPRODUCT THAT MAY BE WORTH MORE THAN THE EDGE: 83 of those 102 windows
called ZERO trades. An 81% abstention rate means the headline was a 19-window
estimate wearing a 102-window label. A rule that declines four windows in five
is not a weak predictor, it is a NARROW one -- and if the windows it fires in
share a property measurable BEFORE the window is scored, that property is a
selector. Abstention is free, so a selector is worth more than the edge
estimate. This harness therefore measures two things in one process:

  1. the held-out net per trade at each horizon, UP and DOWN separately,
     against buy-every-bar in the same window, with the trade count and a
     standard error, because a difference smaller than its own error bar is
     not a result;
  2. whether the firing windows differ from the abstaining ones on realised
     volatility, up-rate or bar count -- fitted on a DISCOVERY half of the
     corpora and checked on a HELD-OUT half that had no say in picking it.

A SUB-BAR HORIZON IS THE TRAP THIS FILE IS NAMED AFTER. Thirty minutes on a
3600s cadence is HALF A BAR. It cannot be asked, it rounds to one bar, and a
report that then writes ``horizon_minutes: 30`` has labelled a sixty-minute
measurement as a thirty-minute one. Every row here records the RESOLVED
minutes beside the asked ones, and a horizon whose rounding moved it by more
than a tenth is marked ``sub_bar: true`` and reported as a duplicate of the
arm it collapsed into rather than as its own answer.

    python -X utf8 scripts/omen_l1_horizon_sweep.py --chain base --cadence 3600

NO NODE IS CONTACTED. Production runs :8090 and this never opens a socket.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    LOOKBACK_BARS, OMEN_TROUGH, ROUND_TRIP_COST,
)
from trading.omen_layers import relative_bands  # noqa: E402
from scripts.omen_experiment import (  # noqa: E402
    horizon_bars, validate_report_horizon,
)
from scripts.omen_layer_probe import (  # noqa: E402
    build_layer_frames, label_skew, load_bars,
)
from scripts.omen_scorable_windows import median_spacing  # noqa: E402

#: The horizons the backlog asked for, in MINUTES of wall clock. Bars are not
#: comparable across a corpus spanning 166s to 345600s bars, and this repo has
#: already written two reports that both said "h12" about 33 minutes and 48
#: days ([4d0b539b]).
DEFAULT_MINUTES = (30.0, 60.0, 120.0, 240.0)

#: Properties of a window that can be read BEFORE it is scored. Every one is
#: computed from the TRAIN slice only: a selector fitted on anything the test
#: window knows is not a selector, it is the leak that makes the whole
#: abstention question meaningless.
PROPERTIES = ("realised_vol", "up_rate", "bar_count", "mean_abs_return",
              "train_drift")


# ---------------------------------------------------------------------------
# Scoring. This mirrors omen_layer_probe.heldout_edge step for step, and keeps
# the per-trade returns that function aggregates away -- a mean with no sample
# behind it cannot carry a standard error, and an edge quoted without one is
# how a +0.10pp inside its own noise gets read as an edge.
# tests/test_an_l1_negative_names_the_horizon_it_measured.py pins the two
# against each other on a real corpus, so this copy cannot drift.
# ---------------------------------------------------------------------------


def window_properties(bars: Sequence[Mapping[str, Any]],
                      start: int, stop: int) -> Dict[str, float]:
    """Pre-scoring properties of ``bars[start:stop]``.

    Nothing here looks past ``stop``. ``stop`` is the TRAIN stop, so every
    number is available to a caller standing at the moment the rule would be
    fitted, which is the only kind of property a selector may use.
    """
    closes = [float(b["close"]) for b in bars[start:stop] if b.get("close")]
    if len(closes) < 3:
        return {k: 0.0 for k in PROPERTIES}
    rets = [(b - a) / a for a, b in zip(closes, closes[1:]) if a]
    if not rets:
        return {k: 0.0 for k in PROPERTIES}
    return {
        "realised_vol": statistics.pstdev(rets),
        "up_rate": sum(1 for r in rets if r > 0) / len(rets),
        # The corpus's own length, not the window's: the sweep slices a fixed
        # window off the end, so this is the only bar count that varies and
        # the only one a selector could key on.
        "bar_count": float(len(bars)),
        "mean_abs_return": sum(abs(r) for r in rets) / len(rets),
        "train_drift": (closes[-1] - closes[0]) / closes[0] if closes[0] else 0.0,
    }


def score_window(bars: Sequence[Mapping[str, Any]], symbol: str, chain: str,
                 horizon: int, train: int, test: int,
                 min_lift: float, min_support: int,
                 margin: float) -> Dict[str, Any]:
    """One corpus, one horizon: what the frozen rule called and what it made.

    Returns the same three headline numbers ``heldout_edge`` returns plus the
    per-trade forward returns behind them, so the caller can pool trades
    across windows and put an error bar on the pooled mean.
    """
    train_stop = len(bars) - test - horizon - 1
    train_start = max(LOOKBACK_BARS, train_stop - train)
    test_start = train_stop + horizon       # purge: no train future reaches a test bar
    test_stop = len(bars) - horizon - 1
    if train_stop <= train_start or test_stop <= test_start:
        return {"error": "window does not fit"}

    def rows_for(start: int, stop: int,
                 bands: Optional[Mapping[str, Any]] = None
                 ) -> List[Dict[str, Any]]:
        out = build_layer_frames(bars, symbol, chain, horizon, start, stop,
                                 bands=bands, margin=margin)
        return [r for r in out
                if r.get("_index") is not None and r.get("_label")]

    # Bands fitted on TRAIN and frozen, exactly as heldout_edge does it: cut
    # points taken over the whole corpus would put the test window's own
    # distribution inside the frame it is scored on.
    seed = rows_for(train_start, train_stop)
    bands = relative_bands(seed) or None

    tr = rows_for(train_start, train_stop, bands)
    te = rows_for(test_start, test_stop, bands)
    if not tr or not te:
        return {"error": "empty train or test window"}

    fit = label_skew(tr, "L1_cooccurrence", min_support=min_support)
    buyable = {g["frame"] for g in fit.get("groups", [])
               if g["lift"] >= min_lift}
    called = [r for r in te if r["L1_cooccurrence"] in buyable]

    called_rets = [float(r["_forward"]) - ROUND_TRIP_COST for r in called]
    test_rets = [float(r["_forward"]) - ROUND_TRIP_COST for r in te]

    def mean(xs: Sequence[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "train_window": [train_start, train_stop],
        "test_window": [test_start, test_stop],
        "train_n": len(tr), "test_n": len(te),
        "buyable_motifs": sorted(buyable),
        "called_n": len(called),
        "called_net": mean(called_rets),
        "baseline_net": mean(test_rets),
        "called_trough_rate": (
            sum(1 for r in called if r["_label"] == OMEN_TROUGH) / len(called)
            if called else 0.0),
        "called_returns": called_rets,
        "test_returns": test_rets,
        "properties": window_properties(bars, train_start, train_stop),
    }


# ---------------------------------------------------------------------------
# Statistics. Two of them, and both exist because a number here has already
# been over-read: a +0.10pp difference on 194 trades, and an 81% abstention
# rate hiding behind a 102-window denominator.
# ---------------------------------------------------------------------------


def pooled_edge(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Trade-weighted per-trade net against buy-every-bar, with its error bar.

    The standard error of the DIFFERENCE is taken as
    ``sqrt(se_called^2 + se_baseline^2)``, which treats the two samples as
    independent when in fact the called trades are a SUBSET of the baseline
    bars. The overlap makes the true error smaller, so this is the
    conservative direction -- and an edge that cannot clear a conservative
    error bar is not an edge.
    """
    fired = [r for r in rows if r.get("called_n", 0) > 0]
    called: List[float] = [x for r in fired
                           for x in r.get("called_returns", ())]

    def se(xs: Sequence[float]) -> float:
        if len(xs) < 2:
            return 0.0
        return statistics.stdev(xs) / math.sqrt(len(xs))

    # A rule that called nothing has NO per-trade net. Reporting 0.0 against a
    # negative baseline would manufacture an edge out of an abstention, which
    # is the single easiest way to invent a positive in this repo.
    if not called:
        return {"corpora": len(rows), "fired": 0, "trades": 0,
                "net": None, "baseline": None, "edge_pp": None,
                "se_pp": None, "z": None}
    # The baseline is the buy-every-bar net over the SAME windows that fired:
    # pooling in windows the rule declined would score it against a market it
    # never traded.
    base = [x for r in fired for x in r.get("test_returns", ())]
    net = sum(called) / len(called)
    bnet = sum(base) / len(base) if base else 0.0
    edge = net - bnet
    se_diff = math.sqrt(se(called) ** 2 + se(base) ** 2)
    return {
        "corpora": len(rows), "fired": len(fired), "trades": len(called),
        "baseline_bars": len(base),
        "net": net, "baseline": bnet,
        "edge_pp": 100.0 * edge,
        "se_pp": 100.0 * se_diff,
        "z": (edge / se_diff) if se_diff else None,
    }


def auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """P(a firing window scores higher than an abstaining one), ties at 0.5.

    Rank-based on purpose: a property measured in price units and one measured
    as a rate cannot be compared by their means, and 0.5 is exactly "this
    property tells you nothing about whether the window will fire".
    """
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for n in negative:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(positive) * len(negative))


def best_threshold(rows: Sequence[Mapping[str, Any]], prop: str
                   ) -> Optional[Dict[str, Any]]:
    """The cut on ``prop`` that best separates fired from abstained.

    Youden's J over every value present, in both directions, so a property
    that predicts firing when LOW is found as readily as one that predicts it
    when high. Fitted on the discovery split only.
    """
    vals = [(float(r["properties"][prop]), r["called_n"] > 0) for r in rows
            if "properties" in r]
    if not vals:
        return None
    fired = sum(1 for _, f in vals if f)
    quiet = len(vals) - fired
    if not fired or not quiet:
        return None
    best: Optional[Dict[str, Any]] = None
    for cut, _ in vals:
        for direction in (1, -1):
            tp = sum(1 for v, f in vals if f and direction * v >= direction * cut)
            fp = sum(1 for v, f in vals if not f and direction * v >= direction * cut)
            j = tp / fired - fp / quiet
            if best is None or j > best["youden_j"]:
                best = {"property": prop, "cut": cut, "direction": direction,
                        "youden_j": j, "tp": tp, "fp": fp}
    return best


def apply_threshold(rows: Sequence[Mapping[str, Any]],
                    rule: Mapping[str, Any]) -> Dict[str, Any]:
    """Does the fitted cut predict firing on corpora it never saw?"""
    prop, cut, direction = rule["property"], rule["cut"], rule["direction"]
    picked = [direction * float(r["properties"][prop]) >= direction * cut
              for r in rows]
    inside = [r for r, keep in zip(rows, picked) if keep]
    outside = [r for r, keep in zip(rows, picked) if not keep]

    def rate(subset: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not subset:
            return None
        return sum(1 for r in subset if r["called_n"] > 0) / len(subset)

    overall = rate(rows)
    selected = rate(inside)
    return {
        "n": len(rows), "selected_n": len(inside),
        "fire_rate_selected": selected,
        "fire_rate_rejected": rate(outside),
        "fire_rate_overall": overall,
        "lift": (selected / overall) if (selected is not None and overall)
                else None,
    }


# ---------------------------------------------------------------------------


def collect_corpora(chain: str, cadence: float, min_bars: int,
                    limit: Optional[int]) -> Tuple[List[Dict[str, Any]],
                                                   List[Dict[str, Any]]]:
    """Every corpus on ``chain`` whose MEDIAN bar spacing is ``cadence``.

    A corpus that cannot be read is NAMED rather than dropped: a sweep that
    quietly loses corpora reports a smaller denominator as if it were the
    population, which is the error the abstention line exists to prevent.
    """
    root = Path("data/historical_ohlcv") / chain
    kept: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if limit is not None and len(kept) >= limit:
            break
        try:
            bars = load_bars(path)
        except Exception as exc:
            failed.append({"corpus": path.name, "error": type(exc).__name__})
            continue
        if len(bars) < min_bars or median_spacing(bars) != cadence:
            continue
        kept.append({"corpus": path.name, "symbol": path.stem.split("_", 1)[-1],
                     "bars": bars})
    return kept, failed


def sweep_horizon(corpora: Sequence[Mapping[str, Any]], chain: str,
                  cadence: float, asked_minutes: float,
                  args: argparse.Namespace) -> Dict[str, Any]:
    """One horizon, every corpus, one pass."""
    bars_ahead = horizon_bars(asked_minutes, int(cadence))
    resolved_minutes = bars_ahead * cadence / 60.0
    window = args.train + args.test + bars_ahead + 60

    rows: List[Dict[str, Any]] = []
    for entry in corpora:
        bars = entry["bars"]
        if len(bars) < window:
            continue
        try:
            scored = score_window(bars[-window:], entry["symbol"], chain,
                                  bars_ahead, args.train, args.test,
                                  args.min_lift, args.min_support,
                                  args.hysteresis)
        except Exception as exc:
            rows.append({"corpus": entry["corpus"], "error": type(exc).__name__})
            continue
        if "error" in scored:
            rows.append({"corpus": entry["corpus"], "error": scored["error"]})
            continue
        scored["corpus"] = entry["corpus"]
        scored["symbol"] = entry["symbol"]
        rows.append(scored)

    scored_rows = [r for r in rows if "called_n" in r]
    # UP and DOWN are decided by WHICH WAY THE MARKET WENT in the held-out
    # window -- the mean forward return BEFORE the round-trip cost.
    #
    # THE BUG THIS AVOIDS, MEASURED. Pass 114 split on the buy-every-bar net
    # AFTER cost, and at a one-bar horizon the cost is larger than almost any
    # realised move: the first run of this sweep put 102 of 102 windows in
    # DOWN at 60m and 120m, so the both-windows rule -- the only scoreboard
    # this loop has -- could not be evaluated at all in the band it exists to
    # measure. "The market fell" and "a round trip in it lost money" are
    # different statements and only the first one names a window.
    up = [r for r in scored_rows
          if r["baseline_net"] + ROUND_TRIP_COST > 0]
    down = [r for r in scored_rows
            if r["baseline_net"] + ROUND_TRIP_COST <= 0]

    report = {
        "asked_minutes": asked_minutes,
        "horizon_bars": bars_ahead,
        "horizon_minutes": resolved_minutes,
        "bar_seconds": int(cadence),
        # A horizon the cadence cannot express is not a measurement of that
        # horizon, and saying so is the entire point of this field.
        "sub_bar": abs(resolved_minutes - asked_minutes) > 0.1 * asked_minutes,
        "corpora_scored": len(scored_rows),
        "corpora_failed": len(rows) - len(scored_rows),
        "abstained": sum(1 for r in scored_rows if r["called_n"] == 0),
        "UP": pooled_edge(up),
        "DOWN": pooled_edge(down),
        "BOTH": pooled_edge(scored_rows),
        "rows": [{k: v for k, v in r.items()
                  if k not in ("called_returns", "test_returns")}
                 for r in rows],
        "_scored": scored_rows,
    }
    validate_report_horizon(report)
    return report


def separation(scored_rows: Sequence[Mapping[str, Any]],
               split_every: int = 2) -> Dict[str, Any]:
    """Do the firing windows differ from the abstaining ones, before scoring?

    The corpora are split deterministically into DISCOVERY and HELD-OUT halves
    by their position in sorted order. The property and its cut are chosen on
    discovery alone; the held-out half is the only number that answers the
    question, because a threshold fitted and read on one sample always
    separates it.
    """
    rows = [r for r in scored_rows if "properties" in r]
    fired = [r for r in rows if r["called_n"] > 0]
    quiet = [r for r in rows if r["called_n"] == 0]
    aucs = {}
    for prop in PROPERTIES:
        aucs[prop] = auc([float(r["properties"][prop]) for r in fired],
                         [float(r["properties"][prop]) for r in quiet])

    discovery = [r for i, r in enumerate(rows) if i % split_every == 0]
    heldout = [r for i, r in enumerate(rows) if i % split_every != 0]
    candidates = [c for c in (best_threshold(discovery, p) for p in PROPERTIES)
                  if c]
    best = max(candidates, key=lambda c: c["youden_j"]) if candidates else None
    checked = apply_threshold(heldout, best) if best and heldout else None
    return {
        "fired": len(fired), "abstained": len(quiet),
        "auc": aucs,
        "discovery_n": len(discovery), "heldout_n": len(heldout),
        "selector": best,
        "heldout_check": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--cadence", type=float, default=3600.0,
                        help="keep only corpora whose MEDIAN bar spacing is "
                             "this many seconds, so a horizon in minutes "
                             "means the same thing on every one of them")
    parser.add_argument("--minutes", default=",".join(
        "%g" % m for m in DEFAULT_MINUTES),
        help="comma-separated horizons in MINUTES of wall clock")
    parser.add_argument("--train", type=int, default=350)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--hysteresis", type=float, default=0.50)
    parser.add_argument("--min-lift", type=float, default=1.3)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    minutes = [float(m) for m in args.minutes.split(",") if m.strip()]
    biggest = max(horizon_bars(m, int(args.cadence)) for m in minutes)
    corpora, failed = collect_corpora(
        args.chain, args.cadence, args.train + args.test + biggest + 60,
        args.limit)
    if not corpora:
        print("no corpus in data/historical_ohlcv/%s at cadence %ss"
              % (args.chain, args.cadence))
        return 2

    print("chain %s  cadence %ss  corpora %d  (unreadable: %d)"
          % (args.chain, args.cadence, len(corpora), len(failed)))
    print("train %d / purge = horizon / test %d, bands fitted on TRAIN and "
          "frozen, hysteresis %.2f, NO node contacted"
          % (args.train, args.test, args.hysteresis))
    print("")
    print("  %9s %6s %6s %7s %7s %10s %10s %9s %8s"
          % ("horizon", "bars", "window", "corpora", "fired", "per-trade",
             "baseline", "EDGE", "z"))

    horizons: List[Dict[str, Any]] = []
    for asked in minutes:
        report = sweep_horizon(corpora, args.chain, args.cadence, asked, args)
        horizons.append(report)
        tag = "%.0fm" % asked + ("*" if report["sub_bar"] else "")
        for name in ("UP", "DOWN"):
            block = report[name]
            if block["net"] is None:
                print("  %9s %6d %6s %7d %7d %10s %10s %9s %8s"
                      % (tag, report["horizon_bars"], name, block["corpora"],
                         0, "--", "--", "no trades", "--"))
                continue
            print("  %9s %6d %6s %7d %7d %9.4f%% %9.4f%% %7.2fpp %8s"
                  % (tag, report["horizon_bars"], name, block["corpora"],
                     block["fired"], 100 * block["net"],
                     100 * block["baseline"], block["edge_pp"],
                     "%.2f" % block["z"] if block["z"] is not None else "--"))
        print("  %9s %6s %6s abstained %d of %d (%.0f%%), %d trades, se %s"
              % ("", "", "", report["abstained"], report["corpora_scored"],
                 100.0 * report["abstained"] / max(1, report["corpora_scored"]),
                 report["BOTH"]["trades"],
                 "%.2fpp" % report["BOTH"]["se_pp"]
                 if report["BOTH"]["se_pp"] is not None else "--"))

    sub = [h for h in horizons if h["sub_bar"]]
    if sub:
        print("\n  * %s is SUB-BAR on a %gs cadence and rounded to %s. It is "
              "the same measurement as the arm it collapsed into, not its own "
              "answer." % (", ".join("%gm" % h["asked_minutes"] for h in sub),
                           args.cadence,
                           ", ".join("%gm" % h["horizon_minutes"] for h in sub)))

    # --- the selector question, at EVERY horizon ---------------------------
    # Not just at the richest: a property that separates firing windows at one
    # horizon and nowhere else is a coincidence found by looking four times,
    # and the only way to see that is to print all four.
    print("\nSELECTOR: do the firing windows differ from the abstaining ones "
          "BEFORE they are scored?")
    print("  AUC 0.500 means the property says nothing about whether the "
          "window will fire.")
    print("\n  %9s %7s %10s %s"
          % ("horizon", "fired", "abstained",
             " ".join("%16s" % p for p in PROPERTIES)))
    for report in horizons:
        sep = separation(report["_scored"])
        report["separation"] = sep
        print("  %9s %7d %10d %s"
              % ("%gm" % report["horizon_minutes"], sep["fired"],
                 sep["abstained"],
                 " ".join("%16s" % ("%.3f" % sep["auc"][p]
                                    if sep["auc"][p] is not None else "--")
                          for p in PROPERTIES)))

    for report in horizons:
        sep = report["separation"]
        rule, check = sep["selector"], sep["heldout_check"]
        if not rule:
            print("\n  %gm: no cut could be fitted (one side of the split had "
                  "no firing window)." % report["horizon_minutes"])
            continue
        print("\n  %gm: best cut on DISCOVERY (%d corpora) is %s %s %.6g, "
              "Youden J %.3f"
              % (report["horizon_minutes"], sep["discovery_n"],
                 rule["property"], ">=" if rule["direction"] > 0 else "<=",
                 rule["cut"], rule["youden_j"]))
        if not check:
            continue
        print("      HELD-OUT (%d corpora it never saw): selected %d, fire "
              "rate %s inside vs %s outside, %s overall -- lift %s"
              % (check["n"], check["selected_n"],
                 "%.0f%%" % (100 * check["fire_rate_selected"])
                 if check["fire_rate_selected"] is not None else "--",
                 "%.0f%%" % (100 * check["fire_rate_rejected"])
                 if check["fire_rate_rejected"] is not None else "--",
                 "%.0f%%" % (100 * check["fire_rate_overall"])
                 if check["fire_rate_overall"] is not None else "--",
                 "%.2fx" % check["lift"] if check["lift"] else "--"))

    payload = {
        "chain": args.chain, "cadence": args.cadence,
        "train": args.train, "test": args.test,
        "hysteresis": args.hysteresis, "min_lift": args.min_lift,
        "min_support": args.min_support,
        "round_trip_cost": ROUND_TRIP_COST,
        "corpora": len(corpora), "unreadable": len(failed),
        "horizons": [{k: v for k, v in h.items() if k != "_scored"}
                     for h in horizons],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
        print("\njson -> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
