"""Score the SELL-HIGH half beside the buy-low half, held out, UP and DOWN.

The gap this closes
-------------------
Every omen report in this repo scores buying. ``buy_omens``,
``buy_hit_rate``, ``buy_net_per_trade`` -- and nothing about the other half
of the book. Pass 114's true-label ceiling (data/brain_experiments/
TROUGH-BASE-RATE-AND-CEILING-pass114.md) made the omission expensive to keep:
over the last 208 bars of 150 eligible 3600s corpora the DOWN window's crest
half is the BIGGER of the two (+1.8884%/trade against +1.6396% for the buy
half), so the half nobody measures is the half that is worth more exactly
where the long-only half struggles.

That ceiling is BY CONSTRUCTION -- its caller was the labels, so its
precision is 100% and it says nothing about whether anything can reach it.
This script is the other end: a predictor that sees only the past, fitted on
a training window and scored on a held-out one, reporting BOTH halves.

The predictor, and why it is this one
-------------------------------------
``label_omen`` calls a bar a trough when the forward move clears the cost AND
the bar sits in the bottom band of its recent range; a crest is the mirror.
The range position is computable at the bar -- the forward move is not. So
the honest causal predictor is: take the range position, and choose the two
bands from the TRAINING window alone.

  buy  (trough call): range position <= b_low
  sell (crest call):  range position >= b_high

``b_low`` and ``b_high`` are fitted independently on the train window by
picking the band with the best per-trade net there, subject to a minimum
train count so a one-trade band cannot win. Neither half is given the other's
threshold, because the point of the item is that the two halves may be
different instruments.

This is deliberately a WEAK predictor. It is not trying to be the brain; it
is the floor the brain has to beat, measured on both halves at one cost and
one horizon. Nothing here queries a node.

Pooling, and the readability floor
----------------------------------
A 208-bar held-out window cannot carry 30 trades on both halves, so the cells
are pooled across corpora by ``omen_scoreboard.pool_scoreboards`` -- totals
added, divided once. Any cell under ``READABLE_TRADES`` renders as UNREADABLE
with its n and is never quoted as an edge, pooled or not.

Usage
-----
  python -X utf8 scripts/omen_both_halves.py
  python -X utf8 scripts/omen_both_halves.py --limit 40 --test 208
  python -X utf8 scripts/omen_both_halves.py --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.omen_experiment import (  # noqa: E402
    DEFAULT_HORIZON_MINUTES, MAX_CORPUS_BAR_SECONDS, bar_seconds, horizon_bars,
    load_bars, window_regime,
)
from trading.omen_brain import (  # noqa: E402
    COST_MULTIPLE, OMEN_CREST, OMEN_MURK, OMEN_TROUGH, RANGE_WINDOW,
    ROUND_TRIP_COST,
)
from trading.omen_scoreboard import (  # noqa: E402
    READABLE_TRADES, money_scoreboard, pool_scoreboards, render_scoreboard,
)

#: Held-out window, in bars. 208 is pass 114's measured readability floor:
#: at the shipped threshold the median corpus labels 14.46% of bars trough, so
#: 30 trough labels needs ~208 bars AT PERFECT RECALL. Below it a per-trade
#: net on one window is one trade wearing a percentage sign.
DEFAULT_TEST_BARS = 208
#: Training window. Large enough that a band fitted on it is fitted on
#: hundreds of trades, not tens.
DEFAULT_TRAIN_BARS = 600
#: Bands swept when fitting. Coarse on purpose: a fine grid over a few hundred
#: train trades fits noise, and the held-out number then measures the grid.
LOW_BANDS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
HIGH_BANDS: Tuple[float, ...] = (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)
#: A band that fired fewer times than this on TRAIN is not a fit, it is a
#: coincidence, and it must not be carried into the held-out window.
MIN_TRAIN_CALLS = 20
#: Shortest corpus worth touching: lookback + train + horizon + test + horizon.
MIN_BARS = RANGE_WINDOW + DEFAULT_TRAIN_BARS + DEFAULT_TEST_BARS + 64


def range_position(bars: Sequence[Mapping[str, Any]], index: int,
                   window: int = RANGE_WINDOW) -> Optional[float]:
    """Where this bar's close sits in the last ``window`` closes, 0..1.

    Uses bars up to and INCLUDING ``index`` and nothing after it. That is the
    whole point: the labeller may look forward, a predictor may not.
    """
    if index < 0 or index >= len(bars):
        return None
    start = max(0, index - window + 1)
    closes = []
    for bar in bars[start:index + 1]:
        try:
            closes.append(float(bar["close"]))
        except (KeyError, TypeError, ValueError):
            return None
    if not closes:
        return None
    low, high = min(closes), max(closes)
    if high == low:
        # A dead-flat window has no low and no high to be at. Calling it a
        # trough would buy a frozen feed.
        return None
    return (closes[-1] - low) / (high - low)


def calls_for(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
              b_low: float, b_high: float) -> List[Tuple[int, str]]:
    """(index, predicted omen) for EVERY bar in the range.

    Every bar is emitted, murk included, so the scoreboard's every-bar
    baseline is computed over the whole window rather than over the bars the
    rule happened to like.
    """
    out: List[Tuple[int, str]] = []
    for index in range(max(start, 0), min(stop, len(bars))):
        position = range_position(bars, index)
        if position is None:
            out.append((index, OMEN_MURK))
        elif position <= b_low:
            out.append((index, OMEN_TROUGH))
        elif position >= b_high:
            out.append((index, OMEN_CREST))
        else:
            out.append((index, OMEN_MURK))
    return out


def fit_bands(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
              horizon: int, cost: float, multiple: float) -> Dict[str, Any]:
    """Pick b_low and b_high on the TRAIN window, independently per half.

    Each half is scored by ``money_scoreboard`` -- the same function that
    scores the held-out window -- so the fit and the test are the same game
    at the same cost. Returns the chosen bands and the train cells they won
    with, because a held-out number whose train number is invisible cannot be
    read as generalisation or as overfitting.
    """
    best_low: Dict[str, Any] = {"band": None, "net_per_trade": None, "n": 0}
    for band in LOW_BANDS:
        board = money_scoreboard(
            bars, calls_for(bars, start, stop, band, 2.0),
            horizon_bars=horizon, cost=cost, multiple=multiple)
        cell = board["buy"]
        if cell["n"] < MIN_TRAIN_CALLS:
            continue
        if best_low["net_per_trade"] is None or \
                cell["net_per_trade"] > best_low["net_per_trade"]:
            best_low = {"band": band, "net_per_trade": cell["net_per_trade"],
                        "n": cell["n"]}
    best_high: Dict[str, Any] = {"band": None, "net_per_trade": None, "n": 0}
    for band in HIGH_BANDS:
        board = money_scoreboard(
            bars, calls_for(bars, start, stop, -1.0, band),
            horizon_bars=horizon, cost=cost, multiple=multiple)
        cell = board["sell"]
        if cell["n"] < MIN_TRAIN_CALLS:
            continue
        if best_high["net_per_trade"] is None or \
                cell["net_per_trade"] > best_high["net_per_trade"]:
            best_high = {"band": band, "net_per_trade": cell["net_per_trade"],
                         "n": cell["n"]}
    return {"low": best_low, "high": best_high}


def pick_window(bars: Sequence[Mapping[str, Any]], horizon: int, test: int,
                train: int, want: str) -> Optional[Dict[str, int]]:
    """The most extreme UP (or DOWN) held-out window this corpus admits.

    Scanned rather than assumed. Pass 116 learned that hard-wiring the
    held-out window to the corpus END scored ONE regime in every report ever
    written here, so the window is chosen by measuring the candidates.
    """
    need_before = RANGE_WINDOW + train + horizon
    last_stop = len(bars) - horizon - 1
    best: Optional[Dict[str, int]] = None
    best_rate: Optional[float] = None
    stop = last_stop
    while stop - test - need_before >= 0:
        regime = window_regime(bars, stop - test, stop, horizon)
        if regime["regime"] == want:
            rate = regime["up_rate"]
            better = (best_rate is None
                      or (rate > best_rate if want == "UP" else rate < best_rate))
            if better:
                best_rate = rate
                best = {"test_start": stop - test, "test_stop": stop,
                        "train_stop": stop - test - horizon,
                        "train_start": stop - test - horizon - train,
                        "up_rate": regime["up_rate"],
                        "mean_forward": regime["mean_forward"],
                        "zero_share": regime["zero_share"]}
        # Step by a quarter window: a one-bar step re-measures the same window
        # 200 times and turns a 40-corpus run into a coffee break.
        stop -= max(1, test // 4)
    return best


def score_corpus(path: Path, horizon_minutes: float, test: int, train: int,
                 cadence_filter: Optional[int], cost: float,
                 multiple: float) -> Optional[Dict[str, Any]]:
    """One corpus, both regimes, both halves. None when it is not eligible."""
    try:
        bars = load_bars(path)
    except (json.JSONDecodeError, OSError, ValueError, AttributeError):
        return None
    if len(bars) < MIN_BARS:
        return None
    cadence = bar_seconds(bars)
    if cadence > MAX_CORPUS_BAR_SECONDS:
        return None
    if cadence_filter is not None and cadence != cadence_filter:
        return None
    horizon = horizon_bars(horizon_minutes, cadence)
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        # A corpus outside the repo (a tmp_path in a test, a mounted archive)
        # is still a corpus. Naming it absolutely beats refusing to score it.
        name = str(path)
    out: Dict[str, Any] = {"path": name,
                           "bar_seconds": cadence, "bars": len(bars),
                           "horizon_bars": horizon,
                           "horizon_minutes": horizon_minutes}
    found = False
    for want in ("UP", "DOWN"):
        window = pick_window(bars, horizon, test, train, want)
        if window is None:
            out[want] = None
            continue
        bands = fit_bands(bars, window["train_start"], window["train_stop"],
                          horizon, cost, multiple)
        if bands["low"]["band"] is None and bands["high"]["band"] is None:
            out[want] = None
            continue
        board = money_scoreboard(
            bars,
            calls_for(bars, window["test_start"], window["test_stop"],
                      bands["low"]["band"] if bands["low"]["band"] is not None else -1.0,
                      bands["high"]["band"] if bands["high"]["band"] is not None else 2.0),
            horizon_bars=horizon, cost=cost, multiple=multiple)
        out[want] = {"window": window, "bands": bands, "board": board}
        found = True
    return out if found else None


def corpus_paths(root: Path, limit: Optional[int]) -> List[Path]:
    paths = sorted(root.rglob("*.json"))
    return paths[:limit] if limit else paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/historical_ohlcv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--test", type=int, default=DEFAULT_TEST_BARS)
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN_BARS)
    parser.add_argument("--horizon-minutes", type=float,
                        default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--cadence", type=int, default=3600,
                        help="only score corpora at this measured cadence; "
                             "0 means every eligible cadence")
    parser.add_argument("--cost", type=float, default=ROUND_TRIP_COST)
    parser.add_argument("--multiple", type=float, default=COST_MULTIPLE)
    parser.add_argument("--report", action="store_true",
                        help="write the JSON artifact to data/brain_experiments")
    args = parser.parse_args(argv)

    root = (ROOT / args.root) if not Path(args.root).is_absolute() else Path(args.root)
    cadence_filter = args.cadence or None
    paths = corpus_paths(root, args.limit)
    print(f"omen_both_halves: {len(paths)} candidate corpora under {root}")
    print(f"  horizon {args.horizon_minutes:.0f} minutes, held-out {args.test} "
          f"bars, train {args.train} bars, cost {args.cost:.4%} x{args.multiple}")

    per_corpus: List[Dict[str, Any]] = []
    for path in paths:
        scored = score_corpus(path, args.horizon_minutes, args.test, args.train,
                              cadence_filter, args.cost, args.multiple)
        if scored is not None:
            per_corpus.append(scored)
    print(f"  {len(per_corpus)} eligible corpora scored")

    pooled: Dict[str, Any] = {}
    for want in ("UP", "DOWN"):
        boards = [c[want]["board"] for c in per_corpus if c.get(want)]
        # Boards from different cadences carry different horizon_bars for the
        # same wall-clock horizon, and pool_scoreboards refuses to mix them.
        # Pool per horizon_bars and keep the largest group, naming the rest.
        by_horizon: Dict[int, List[Dict[str, Any]]] = {}
        for board in boards:
            by_horizon.setdefault(int(board["horizon_bars"]), []).append(board)
        if not by_horizon:
            pooled[want] = None
            print(f"\n{want}: no corpus admitted a {want} held-out window")
            continue
        main_h = max(by_horizon, key=lambda h: len(by_horizon[h]))
        dropped = sum(len(v) for h, v in by_horizon.items() if h != main_h)
        board = pool_scoreboards(by_horizon[main_h])
        board["corpora"] = len(by_horizon[main_h])
        board["corpora_dropped_other_horizon"] = dropped
        pooled[want] = board
        print(f"\n{want} window -- {board['corpora']} corpora pooled at "
              f"horizon {main_h} bars"
              + (f" ({dropped} dropped at another bar-horizon)" if dropped else ""))
        print(render_scoreboard(board, title=f"{want} held-out"))

    verdict = _verdict(pooled)
    print("\n" + verdict)

    artifact = {
        "script": "scripts/omen_both_halves.py",
        "horizon_minutes": args.horizon_minutes,
        "horizon_bars": (pooled.get("UP") or pooled.get("DOWN") or {}).get("horizon_bars"),
        "bar_seconds": cadence_filter,
        "test_bars": args.test,
        "train_bars": args.train,
        "round_trip_cost": args.cost,
        "cost_multiple": args.multiple,
        "readable_trades_floor": READABLE_TRADES,
        "corpora_scored": len(per_corpus),
        "pooled": pooled,
        "per_corpus": per_corpus,
        "verdict": verdict,
    }
    if args.report:
        out = ROOT / "data" / "brain_experiments" / "omen-both-halves-pass118.json"
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


def _verdict(pooled: Mapping[str, Any]) -> str:
    """Say plainly which cells may be quoted and which may not."""
    lines = ["VERDICT"]
    for want in ("UP", "DOWN"):
        board = pooled.get(want)
        if not board:
            lines.append(f"  {want}: no window -- nothing measured")
            continue
        for half, key in (("buy (trough)", "buy"), ("sell (crest)", "sell")):
            cell = board[key]
            if not cell["readable"]:
                lines.append(f"  {want} {half}: UNREADABLE, n={cell['n']} "
                             f"below the {READABLE_TRADES}-trade floor -- "
                             f"not an edge, not quotable")
                continue
            baseline = board["every_bar_net_per_trade"]
            ref = baseline if key == "buy" else (
                -baseline if baseline is not None else None)
            delta = (cell["net_per_trade"] - ref) if ref is not None else None
            lines.append(
                f"  {want} {half}: n={cell['n']} "
                f"{cell['net_per_trade']:+.4%} per trade against an every-bar "
                f"{ref:+.4%}" + (f" -- {delta:+.4%} edge" if delta is not None else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
