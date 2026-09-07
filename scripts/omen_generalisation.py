"""Do the omen features carry forward information OUT OF SAMPLE, net of cost?

The 2026-09-07 omen run reported 100% train recall beside 26.6% held-out
accuracy against a 31.2% majority class, and buy omens that lost 0.94bp per
trade relative to buying every bar. The offline cause, measured before this
script existed: the substrate frames produce **2725 distinct signatures out of
2725 training samples**. Every key unique means perfect memorisation and zero
generalisation, and no amount of substrate tuning changes that -- it is a
property of the representation, visible without training anything.

This script asks the question that decides whether the omen idea can pay at
all, with no brain and no fabric involved, so the answer arrives in minutes
instead of a 15-minute retrain per configuration:

    at what bin coarseness do similar bars share a bin, and does the forward
    return conditioned on those bins beat buying indiscriminately, after the
    round-trip cost, on data the estimator never saw?

Protocol
--------
Three disjoint chronological windows per symbol, each separated by a full
horizon so no window's future overlaps the next:

    TRAIN   fit bin edges (labels never read) and the per-bin forward means
    VALID   choose the score threshold
    TEST    report -- touched once, by a threshold chosen without it

Controls, because a held-out number alone has fooled this repo before:

  * EVERY-BAR baseline on the identical test bars. A long-only rule in an up
    window flatters itself; the only meaningful comparison is against taking
    every bar in the SAME window.
  * SHUFFLED-LABEL control. The whole pipeline re-run with forward returns
    permuted. Whatever it "finds" there is what the method invents from noise,
    and a real result has to clear it.
  * NON-OVERLAPPING P/L beside per-trade expectancy. Buying at t and t+1 while
    holding 12 bars double-counts one move; a single-slot strategy cannot take
    both, so the sequential number is the one a live lane would earn.
  * MANY SYMBOLS. One window's direction is not evidence.

Usage
-----
  python -X utf8 scripts/omen_generalisation.py --symbols 12 --bins 2,3,4,5,8,20
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_features import (  # noqa: E402
    FEATURE_NAMES, LOOKBACK_BARS, BinTable, digitize, features, fit_bins,
    signature,
)

try:
    from services.symbol_edge_gate import ROUND_TRIP_COST  # noqa: E402
except Exception:  # pragma: no cover - import-order safety only
    ROUND_TRIP_COST = 0.0065


# --- corpus ---------------------------------------------------------------

def load_bars(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close") and b.get("timestamp")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def bar_seconds(bars: Sequence[Mapping[str, Any]]) -> int:
    gaps = [int(bars[i]["timestamp"]) - int(bars[i - 1]["timestamp"])
            for i in range(1, min(len(bars), 400))]
    gaps = [g for g in gaps if g > 0]
    return int(sorted(gaps)[len(gaps) // 2]) if gaps else 3600


def horizon_bars(minutes: float, cadence_seconds: int) -> int:
    """A wall-clock horizon in this symbol's own bars, never fewer than one.

    The horizon is the thing the strategy promises -- "I will be out in twenty
    minutes" -- and it is a wall-clock promise. Expressing it in bars makes it
    mean six different things across a corpus whose cadences run 300s to 3600s,
    which is how a single run came to average a 120-minute forecast on cbBTC
    with a 720-minute forecast on SHIB.

    Rounds to nearest so a 10-minute horizon on 600s bars is 1 bar rather than
    0; a zero-bar horizon would compare a bar's close against itself and report
    a flawless, free, entirely fictional edge.
    """
    if cadence_seconds <= 0:
        raise ValueError("cadence must be positive")
    return max(1, int(round(minutes * 60.0 / cadence_seconds)))


def extract_once(bars: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Every scalar row for the corpus, computed once.

    The features read only the PAST, so they do not depend on the horizon or
    on the bin table -- computing them per (horizon x bins x shuffle) setting
    repeated the most expensive step in the script dozens of times over
    identical inputs. They are built once here and the horizon is attached
    afterwards by ``attach_forward``.
    """
    rows: List[Dict[str, Any]] = []
    for index in range(LOOKBACK_BARS, len(bars)):
        try:
            rows.append({"index": index, "x": features(bars, index)})
        except (ValueError, IndexError):
            continue
    return rows


def attach_forward(rows: Sequence[Mapping[str, Any]],
                   bars: Sequence[Mapping[str, Any]],
                   horizon: int, start: int, stop: int) -> List[Dict[str, Any]]:
    """Rows in ``[start, stop)`` carrying their forward return at ``horizon``.

    A bar whose future is not in the corpus is DROPPED, never labelled: a
    missing future is not a flat move.
    """
    out: List[Dict[str, Any]] = []
    limit = len(bars) - horizon
    for row in rows:
        index = row["index"]
        if index < start or index >= min(stop, limit):
            continue
        entry = float(bars[index]["close"])
        future = float(bars[index + horizon]["close"])
        if entry <= 0 or not math.isfinite(future):
            continue
        out.append({"index": index, "x": row["x"],
                    "forward": (future - entry) / entry})
    return out


# --- the estimator --------------------------------------------------------

class BinnedForward:
    """E[forward | bin], one table per feature, shrunk toward the base rate.

    Deliberately the simplest estimator that can generalise: each feature's
    bin holds the mean forward return of the training bars that landed in it,
    pulled toward the global mean in proportion to how little evidence the bin
    has (``prior_strength`` bars' worth). A bin seen twice therefore says
    almost nothing, and a bin seen four hundred times says almost all of what
    it measured. Without that shrinkage the rarest bin wins every ranking and
    the model is a lookup table for outliers.

    The score is the MEAN of the per-feature deviations, not the sum: the 27
    features are heavily correlated (``r24`` and ``d24`` see one move twice),
    and summing correlated evidence inflates confidence in exact proportion to
    the redundancy.
    """

    def __init__(self, table: BinTable, prior_strength: float = 50.0) -> None:
        self.table = table
        self.prior_strength = float(prior_strength)
        self.base = 0.0
        self.deviation: Dict[str, Dict[int, float]] = {}
        self.counts: Dict[str, Counter] = {}

    def fit(self, rows: Sequence[Mapping[str, Any]],
            forwards: Sequence[float]) -> "BinnedForward":
        if len(rows) != len(forwards):
            raise ValueError("rows and forwards must be the same length")
        if not rows:
            raise ValueError("cannot fit on an empty training set")
        self.base = sum(forwards) / len(forwards)
        totals: Dict[str, Dict[int, List[float]]] = {n: {} for n in FEATURE_NAMES}
        for row, forward in zip(rows, forwards):
            binned = row["bins"]
            for name in FEATURE_NAMES:
                bucket = totals[name].setdefault(binned[name], [0.0, 0.0])
                bucket[0] += forward
                bucket[1] += 1.0
        self.deviation = {}
        self.counts = {}
        for name in FEATURE_NAMES:
            self.deviation[name] = {}
            self.counts[name] = Counter()
            for index, (total, count) in totals[name].items():
                shrunk = ((total + self.prior_strength * self.base)
                          / (count + self.prior_strength))
                self.deviation[name][index] = shrunk - self.base
                self.counts[name][index] = int(count)
        return self

    def score(self, row: Mapping[str, Any]) -> float:
        """Expected forward return for this bar, as a fraction."""
        binned = row["bins"]
        total = 0.0
        for name in FEATURE_NAMES:
            total += self.deviation.get(name, {}).get(binned[name], 0.0)
        return self.base + total / len(FEATURE_NAMES)


# --- scoring --------------------------------------------------------------

def sequential_pnl(rows: Sequence[Mapping[str, Any]], taken: Sequence[bool],
                   horizon: int, cost: float) -> Tuple[int, float]:
    """P/L a single-slot lane would actually earn: no overlapping holds.

    Per-trade expectancy over overlapping entries counts one move many times.
    A live lane holding one position at a time cannot take a signal while the
    previous trade is open, so this walks forward and skips those.
    """
    total, trades, free_at = 0.0, 0, -1
    for row, take in zip(rows, taken):
        if not take or row["index"] < free_at:
            continue
        total += row["forward"] - cost
        trades += 1
        free_at = row["index"] + horizon
    return trades, total


#: Score quantiles the tail profile reports, as "keep the top q fraction".
#: A rule that gets to CHOOSE when to fire is not described by the mean over
#: every bar it admits; it is described by what it earns at the tightest cut
#: it can still trade. ``choose_threshold`` only searches to the 95th
#: percentile and refuses any cut holding fewer than ``--min-trades`` bars, so
#: the top 1% has never appeared in an omen number.
TAIL_QUANTILES: Tuple[float, ...] = (0.50, 0.25, 0.10, 0.05, 0.02, 0.01)


def tail_profile(rows: Sequence[Mapping[str, Any]], scores: Sequence[float],
                 cost: float) -> Dict[float, Tuple[int, float, int]]:
    """Out-of-sample forward return by score quantile, per symbol.

    Returns ``{quantile: (bars, summed_forward, bars_clearing_cost)}``. Sums
    rather than means so the caller can aggregate symbols by bar count without
    letting a symbol that contributed nine bars outvote one that contributed
    nine hundred.

    Scores are ranked WITHIN the symbol: the score is a shrunk deviation from
    that symbol's own training base rate, so a score of 0.002 means something
    different on SHIB than on cbBTC and a pooled ranking would compare them.
    """
    if not rows:
        return {}
    ordered = sorted(zip(scores, (row["forward"] for row in rows)),
                     key=lambda pair: pair[0], reverse=True)
    out: Dict[float, Tuple[int, float, int]] = {}
    for quantile in TAIL_QUANTILES:
        keep = int(len(ordered) * quantile)
        if keep < 1:
            continue
        head = ordered[:keep]
        out[quantile] = (keep,
                         sum(forward for _, forward in head),
                         sum(1 for _, forward in head if forward > cost))
    return out


def evaluate(model: BinnedForward, rows: Sequence[Mapping[str, Any]],
             threshold: float, horizon: int, cost: float,
             scores: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    if scores is None:
        scores = [model.score(row) for row in rows]
    taken = [score > threshold for score in scores]
    net = [row["forward"] - cost for row, take in zip(rows, taken) if take]
    seq_trades, seq_total = sequential_pnl(rows, taken, horizon, cost)
    every = [row["forward"] - cost for row in rows]
    return {
        "trades": len(net),
        "per_trade": (sum(net) / len(net)) if net else 0.0,
        "hit_rate": (sum(1 for v in net if v > 0) / len(net)) if net else 0.0,
        "sequential_trades": seq_trades,
        "sequential_total": seq_total,
        "sequential_per_trade": (seq_total / seq_trades) if seq_trades else 0.0,
        "every_bar_per_trade": (sum(every) / len(every)) if every else 0.0,
        "every_bar_bars": len(every),
    }


def choose_threshold(rows: Sequence[Mapping[str, Any]],
                     scores: Sequence[float],
                     cost: float, min_trades: int) -> Optional[float]:
    """The validation threshold: best per-trade net that still trades enough.

    ``min_trades`` is not tuning -- it refuses a threshold whose evidence is a
    handful of bars, which is how a validation split gets fitted as hard as a
    test split. Returns ``None`` when nothing clears it, and ``None`` means
    "this configuration does not trade", not "take everything".

    Scores are computed by the caller and passed in: an earlier version
    re-scored the whole validation set inside this loop, once per candidate
    threshold, which made the sweep 19x slower than it needed to be for an
    identical answer.
    """
    if not scores:
        return None
    # Sort rows by score ONCE, then walk the candidate cut points from the
    # most permissive to the tightest, accumulating the net as trades drop
    # out. Every candidate's mean is then O(1) rather than a fresh pass.
    ordered = sorted(zip(scores, (row["forward"] - cost for row in rows)),
                     key=lambda pair: pair[0])
    suffix_total, suffix_count = 0.0, 0
    means: List[Tuple[float, float, int]] = []
    for score, net in reversed(ordered):
        suffix_total += net
        suffix_count += 1
        means.append((score, suffix_total / suffix_count, suffix_count))

    ranked = sorted(scores)
    best_threshold, best_value = None, None
    for k in range(5, 96, 5):
        candidate = ranked[min(len(ranked) - 1, (k * len(ranked)) // 100)]
        # Rows strictly above the candidate: the suffix of `ordered`.
        kept = [entry for entry in means if entry[0] > candidate]
        if not kept:
            continue
        value, count = kept[-1][1], kept[-1][2]
        if count < min_trades:
            continue
        if best_value is None or value > best_value:
            best_threshold, best_value = candidate, value
    return best_threshold


# --- one symbol, one coarseness ------------------------------------------

def run_symbol(bars: Sequence[Mapping[str, Any]],
               cached: Sequence[Mapping[str, Any]], horizon: int, bins: int,
               cost: float, rng: random.Random, prior: float,
               min_trades: int, shuffle: bool) -> Optional[Dict[str, Any]]:
    """Fit on train, threshold on valid, report on test. One number each."""
    last = len(bars) - horizon - 1
    span = (last - LOOKBACK_BARS - 2 * horizon) // 3
    if span < 200:
        return None
    train_start = LOOKBACK_BARS
    train_stop = train_start + span
    valid_start = train_stop + horizon
    valid_stop = valid_start + span
    test_start = valid_stop + horizon
    test_stop = min(last, test_start + span)

    train = attach_forward(cached, bars, horizon, train_start, train_stop)
    valid = attach_forward(cached, bars, horizon, valid_start, valid_stop)
    test = attach_forward(cached, bars, horizon, test_start, test_stop)
    if len(train) < 200 or len(valid) < 100 or len(test) < 100:
        return None

    if shuffle:
        # Permute the futures WITHIN each window. Breaking the bar->future
        # pairing while leaving both marginal distributions intact is what
        # makes this a control: anything the pipeline still reports is
        # manufactured by the pipeline. The bin table is fitted on features
        # only, so permuting futures cannot move a bin edge -- the control
        # differs from the real run in exactly one thing.
        for window in (train, valid, test):
            forwards = [row["forward"] for row in window]
            rng.shuffle(forwards)
            for row, forward in zip(window, forwards):
                row["forward"] = forward

    table = fit_bins([row["x"] for row in train], bins)
    for window in (train, valid, test):
        for row in window:
            row["bins"] = digitize(row["x"], table)

    distinct = len({signature(row["bins"]) for row in train})
    model = BinnedForward(table, prior_strength=prior).fit(
        train, [row["forward"] for row in train])
    valid_scores = [model.score(row) for row in valid]
    test_scores = [model.score(row) for row in test]
    # The tail is measured whether or not a threshold cleared the trade floor:
    # "this configuration does not trade" is a statement about the threshold
    # search, not about whether the score ranks the forward move.
    tail = tail_profile(test, test_scores, cost)
    threshold = choose_threshold(valid, valid_scores, cost, min_trades)
    if threshold is None:
        return {"traded": False, "distinct_ratio": distinct / len(train),
                "train_rows": len(train), "test_rows": len(test), "tail": tail}

    result = evaluate(model, test, threshold, horizon, cost, scores=test_scores)
    result.update({
        "traded": True,
        "threshold": threshold,
        "distinct_ratio": distinct / len(train),
        "train_rows": len(train), "valid_rows": len(valid),
        "test_rows": len(test),
        "tail": tail,
    })
    return result


# --- main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default="data/historical_ohlcv/base")
    parser.add_argument("--symbols", type=int, default=12)
    parser.add_argument("--horizons", default="120",
                        help="comma-separated forecast horizons in MINUTES of "
                             "wall clock, converted to bars per symbol. Bars, "
                             "the old unit, are not comparable across this "
                             "corpus: it mixes 300s and 3600s cadences, so one "
                             "'--horizons 12' run forecast 120 minutes on "
                             "cbBTC and 720 on SHIB and averaged them into one "
                             "row. The lattice's chaos layer measures a usable "
                             "horizon per symbol (~230 min on AERO-USDC); a "
                             "horizon past it is a forecast the data cannot "
                             "support.")
    parser.add_argument("--max-bar-seconds", type=int, default=None,
                        help="skip symbols coarser than this cadence. A "
                             "10-minute forecast cannot be measured on hourly "
                             "bars, and the corpus is 63%% hourly by file "
                             "count while the LARGEST files -- which is how "
                             "symbols get picked -- are hourly too.")
    parser.add_argument("--min-bar-seconds", type=int, default=None,
                        help="skip symbols finer than this cadence.")
    parser.add_argument("--bins", default="2,3,4,5,8,20")
    parser.add_argument("--prior", type=float, default=50.0)
    parser.add_argument("--min-trades", type=int, default=25)
    parser.add_argument("--min-bars", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cost", type=float, default=None)
    parser.add_argument("--report-dir", default="data/brain_experiments")
    args = parser.parse_args()

    cost = ROUND_TRIP_COST if args.cost is None else args.cost
    bin_settings = [int(v) for v in args.bins.split(",") if v.strip()]
    horizon_minutes = [int(v) for v in args.horizons.split(",") if v.strip()]

    corpus_dir = ROOT / args.corpus_dir
    candidates = sorted(p for p in corpus_dir.glob("*.json"))
    # (symbol, bars, bar_seconds). The cadence travels WITH the symbol because
    # the horizon is a wall-clock quantity and every symbol converts it itself.
    chosen: List[Tuple[str, List[Dict[str, Any]], int]] = []
    seen_symbols = set()
    skipped_cadence = 0
    for path in sorted(candidates, key=lambda p: -p.stat().st_size):
        if len(chosen) >= args.symbols:
            break
        symbol = path.stem.split("_", 1)[-1].upper()
        if symbol in seen_symbols:
            continue  # one window per market; the same pair twice is not two tests
        try:
            bars = load_bars(path)
        except Exception:
            continue
        if len(bars) < args.min_bars:
            continue
        cadence = bar_seconds(bars)
        if args.max_bar_seconds is not None and cadence > args.max_bar_seconds:
            skipped_cadence += 1
            continue
        if args.min_bar_seconds is not None and cadence < args.min_bar_seconds:
            skipped_cadence += 1
            continue
        seen_symbols.add(symbol)
        chosen.append((symbol, bars, cadence))

    if not chosen:
        print(f"no corpus in {corpus_dir} with >= {args.min_bars} bars"
              + (f" at the requested cadence ({skipped_cadence} skipped)"
                 if skipped_cadence else ""))
        return 2

    print(f"round-trip cost {cost:.4%}, horizons {horizon_minutes} MINUTES, "
          f"bins {bin_settings}"
          + (f", {skipped_cadence} symbols skipped on cadence"
             if skipped_cadence else ""))
    print(f"{len(chosen)} symbols: " + ", ".join(
        f"{s}({len(b)}@{c}s)" for s, b, c in chosen))
    for minutes in horizon_minutes:
        spread = sorted({horizon_bars(minutes, c) for _, _, c in chosen})
        print(f"  {minutes:>5} min -> {spread} bars across the set")

    started = time.time()
    cache: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, bars, _cadence in chosen:
        cache[symbol] = extract_once(bars)
    print(f"features: {sum(len(v) for v in cache.values())} rows in "
          f"{time.time() - started:.0f}s\n", flush=True)

    report: Dict[str, Any] = {
        "cost": cost, "horizon_minutes": horizon_minutes, "bins": bin_settings,
        "prior_strength": args.prior, "min_trades": args.min_trades,
        "symbols": [s for s, _, _ in chosen],
        "bar_seconds": {s: c for s, _, c in chosen}, "settings": [],
    }

    header = (f"{'minutes':>8} {'bins':>5} {'shuf':>5} {'uniq-key':>9} "
              f"{'syms':>5} {'trades':>7} {'per-trade':>10} {'every-bar':>10} "
              f"{'EDGE':>9} {'seq/trade':>10} {'won':>7}")
    print(header)
    print("-" * len(header))

    for minutes in horizon_minutes:
        for bins in bin_settings:
            for shuffle in (False, True):
                rows_out, distinct_ratios = [], []
                tails: Dict[float, List[float]] = {}
                for symbol, bars, cadence in chosen:
                    horizon = horizon_bars(minutes, cadence)
                    result = run_symbol(bars, cache[symbol], horizon, bins,
                                        cost, random.Random(args.seed),
                                        args.prior, args.min_trades, shuffle)
                    if result is None:
                        continue
                    distinct_ratios.append(result["distinct_ratio"])
                    for quantile, (n, total, clearing) in result.get("tail", {}).items():
                        bucket = tails.setdefault(quantile, [0.0, 0.0, 0.0])
                        bucket[0] += n
                        bucket[1] += total
                        bucket[2] += clearing
                    if result.get("traded"):
                        rows_out.append(result)
                tail_rows = {
                    q: {"bars": int(v[0]),
                        "mean_forward": v[1] / v[0] if v[0] else 0.0,
                        "clearing_cost": v[2] / v[0] if v[0] else 0.0}
                    for q, v in sorted(tails.items(), reverse=True) if v[0]
                }
                if not rows_out:
                    print(f"{minutes:>8} {bins:>5} {str(shuffle):>5} "
                          f"{'-':>9} {0:>5}  nothing cleared the "
                          f"{args.min_trades}-trade floor", flush=True)
                    report["settings"].append({
                        "horizon_minutes": minutes, "bins": bins,
                        "shuffled": shuffle, "traded": False, "tail": tail_rows,
                    })
                    continue
                trades = sum(r["trades"] for r in rows_out)
                # Trade-weighted, so a symbol that fired twice cannot outvote
                # one that fired four hundred times.
                per_trade = sum(r["per_trade"] * r["trades"]
                                for r in rows_out) / max(1, trades)
                every_bars = sum(r["every_bar_bars"] for r in rows_out)
                every = sum(r["every_bar_per_trade"] * r["every_bar_bars"]
                            for r in rows_out) / max(1, every_bars)
                seq_trades = sum(r["sequential_trades"] for r in rows_out)
                seq_total = sum(r["sequential_total"] for r in rows_out)
                won = sum(1 for r in rows_out
                          if r["per_trade"] > r["every_bar_per_trade"])
                uniq = sum(distinct_ratios) / len(distinct_ratios)
                print(f"{minutes:>8} {bins:>5} {str(shuffle):>5} {uniq:>8.1%} "
                      f"{len(rows_out):>5} {trades:>7} {per_trade:>+9.4%} "
                      f"{every:>+9.4%} {per_trade - every:>+8.4%} "
                      f"{(seq_total / seq_trades if seq_trades else 0.0):>+9.4%} "
                      f"{won:>3}/{len(rows_out):<3}", flush=True)
                if tail_rows:
                    # GROSS forward return by score quantile, cost NOT
                    # subtracted: the question this answers is whether the
                    # score ranks the size of the move, and subtracting a
                    # constant from every row cannot change a ranking.
                    #
                    # Printed for the SHUFFLED control too. A tail is a
                    # selection, and on a fat-tailed return distribution a
                    # selection of 1% has a wide sampling error -- a lift that
                    # the control reproduces is sampling noise wearing the
                    # shape of a finding. This crew has already been fooled by
                    # exactly that class of number, so the control has to be
                    # visible on the same screen, not inferred from the EDGE
                    # column beside it.
                    print(f"     {'tail-SHUF' if shuffle else 'tail     '}  "
                          + "  ".join(
                        f"top{q:.0%}:{v['mean_forward']:+.4%}"
                        f"/{v['clearing_cost']:.0%}pay(n={v['bars']})"
                        for q, v in tail_rows.items()), flush=True)
                report["settings"].append({
                    "horizon_minutes": minutes, "bins": bins,
                    "shuffled": shuffle, "traded": True,
                    "distinct_key_ratio": uniq, "symbols_traded": len(rows_out),
                    "trades": trades, "per_trade": per_trade,
                    "every_bar_per_trade": every, "edge": per_trade - every,
                    "sequential_trades": seq_trades,
                    "sequential_per_trade": (seq_total / seq_trades) if seq_trades else 0.0,
                    "symbols_beating_every_bar": won,
                    "tail": tail_rows,
                })

    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = report_dir / f"omen-generalisation-{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
