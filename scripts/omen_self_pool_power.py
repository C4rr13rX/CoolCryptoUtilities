"""How many trades can a REGIME-PURE held-out window actually place?

Item [5ec44914]. The self-pool dilution verdict rests on 19 trades across four
cells (7, 1, 2, 9) and the operator's power arithmetic says each arm needs
roughly 60. The obvious reply is "use a longer held-out window" -- and that
reply is wrong in a way nobody has written down, which is what this script
measures.

A held-out window has to be TWO things at once and they pull apart:

  * POWERED -- long enough that the arm places enough trough trades for a
    per-trade mean to have a standard error.
  * REGIME-PURE -- an UP window and a DOWN window, separately, because a
    long-only rule flatters itself in an up window and that error has already
    produced a fake 78% and a fake +0.9067% in this repo. Gale's windows
    separated 53.3% up-rate from 11.7% ONLY because they were 60 bars long.

Stretch the window for power and the up-rate walks back to the corpus mean, at
which point there is no UP window and no DOWN window, only the market. So the
question this script answers is not "how long a window" but:

    For each corpus, what is the LONGEST window that still clears the purity
    bar, how many trough omens does it contain, and how many corpora must be
    POOLED before each arm reaches n=60?

Nothing here talks to a node. It is arithmetic over the corpora, so it costs no
node time and it is decisive before any fabric is trained -- the standing rule
about sizing an experiment against the budget before launching it.

Usage
-----
  python -X utf8 scripts/omen_self_pool_power.py --horizon-minutes 720
  python -X utf8 scripts/omen_self_pool_power.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import (  # noqa: E402
    OMEN_TROUGH, ROUND_TRIP_COST, label_omen,
)

#: A corpus coarser than four hours is a sparse or broken download rather than
#: a timeframe anybody chose -- the same cut scripts/omen_experiment.py takes.
MAX_CORPUS_BAR_SECONDS = 14400

#: Gale's pass-111 windows measured 53.3% up-rate (UP) and 11.7% (DOWN) over 60
#: bars. These are those numbers rounded OUTWARD, so a window that clears them
#: is at least as pure as the pair the dilution verdict was read off. Purity is
#: the FORWARD-return up-rate, which is what the report quoted.
UP_PURITY = 0.50
DOWN_PURITY = 0.15

#: Below this a window is not a window -- it is a handful of bars whose up-rate
#: is pure by accident. Gale's were 60.
MIN_WINDOW_BARS = 60

#: The share of held-out bars on which the arm actually emitted a trough omen,
#: measured in pass 111: 7/60 and 2/60 with the self pools queried, 1/60 and
#: 9/60 without. The arm cannot trade a trough it does not call, so the trades
#: a window can deliver is its trough COUNT times this, not its bar count.
FIRE_RATES = {"pass111 best cell (9/60)": 9.0 / 60.0,
              "pass111 UP with self (7/60)": 7.0 / 60.0,
              "pass111 DOWN with self (2/60)": 2.0 / 60.0,
              "pass111 worst cell (1/60)": 1.0 / 60.0}

#: The item's floor: below this an arm reports its achieved n and declines to
#: give a per-trade verdict.
TARGET_TRADES = 60


def load_bars(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def bar_seconds(bars: Sequence[Mapping[str, Any]]) -> int:
    gaps = [int(bars[i]["timestamp"]) - int(bars[i - 1]["timestamp"])
            for i in range(1, min(len(bars), 400))]
    gaps = [g for g in gaps if g > 0]
    return int(sorted(gaps)[len(gaps) // 2]) if gaps else 3600


def horizon_bars(minutes: float, cadence_seconds: int) -> int:
    """Never fewer than one bar: a zero-bar horizon compares a close against
    itself and reports a flawless, free, entirely fictional edge."""
    if cadence_seconds <= 0:
        raise ValueError("cadence must be positive")
    return max(1, int(round(float(minutes) * 60.0 / cadence_seconds)))


def forward_returns(bars: Sequence[Mapping[str, Any]], h: int) -> List[float | None]:
    out: List[float | None] = []
    for i in range(len(bars)):
        j = i + h
        if j >= len(bars):
            out.append(None)
            continue
        entry = float(bars[i]["close"])
        if entry == 0:
            out.append(None)
            continue
        out.append((float(bars[j]["close"]) - entry) / entry)
    return out


def trough_flags(bars: Sequence[Mapping[str, Any]], h: int) -> List[bool]:
    """The TRUE trough label, from the same function the experiment trains on."""
    return [label_omen(bars, i, horizon_bars=h) == OMEN_TROUGH
            for i in range(len(bars))]


#: Window lengths tried, shortest first. A grid rather than every length: the
#: answer wanted is an ORDER OF MAGNITUDE ("60 bars or 2000?"), and scanning
#: every start at every length on a 22k-bar corpus is 480M window evaluations,
#: which does not fit a pass. Each length is swept over EVERY start, so the
#: grid loses resolution in the length, never a window.
LENGTH_GRID = (60, 80, 100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400,
               3600, 5000)


def longest_pure_window(rates: Sequence[float | None], *, want_up: bool,
                        purity: float, min_bars: int) -> Dict[str, Any] | None:
    """The longest window on ``LENGTH_GRID`` whose forward up-rate clears
    ``purity``, scanning every start at each length.

    Purity is NOT monotone in length -- a window that fails at 200 bars can
    pass at 400 by reaching into a different stretch of tape -- so every grid
    length is tried rather than stopping at the first failure.
    """
    n = len(rates)
    best: Dict[str, Any] | None = None
    # Prefix counts, so any window's up-rate is O(1).
    up = [0] * (n + 1)
    known = [0] * (n + 1)
    for i, r in enumerate(rates):
        up[i + 1] = up[i] + (1 if (r is not None and r > 0) else 0)
        known[i + 1] = known[i] + (1 if r is not None else 0)

    for length in LENGTH_GRID:
        if length < min_bars or length > n:
            continue
        for start in range(0, n - length + 1):
            end = start + length
            k = known[end] - known[start]
            if k < min_bars:
                continue
            rate = (up[end] - up[start]) / k
            if (rate >= purity) if want_up else (rate <= purity):
                best = {"start": start, "end": end, "bars": length,
                        "up_rate": rate}
                break  # this length is reachable; try a longer one
    return best


def census_corpus(path: Path, minutes: float) -> Dict[str, Any] | None:
    bars = load_bars(path)
    if len(bars) < 400:
        return None
    cadence = bar_seconds(bars)
    if cadence > MAX_CORPUS_BAR_SECONDS:
        return None
    h = horizon_bars(minutes, cadence)
    rates = forward_returns(bars, h)
    troughs = trough_flags(bars, h)

    # The PATH, not the name. `0002_LINK-WETH.json` exists under arbitrum and
    # under polygon, so a plan that names only the file is a command that
    # raises FileNotFoundError -- and worse, two different chains' windows
    # would look like one corpus repeated and be pooled as if independent.
    row: Dict[str, Any] = {"corpus": path.as_posix(), "chain": path.parent.name,
                           "bars": len(bars),
                           "bar_seconds": cadence, "horizon_bars": h,
                           "horizon_minutes": minutes}
    for tag, want_up, purity in (("up", True, UP_PURITY),
                                 ("down", False, DOWN_PURITY)):
        win = longest_pure_window(rates, want_up=want_up, purity=purity,
                                  min_bars=MIN_WINDOW_BARS)
        if win is None:
            row[tag] = None
            continue
        n_trough = sum(troughs[win["start"]:win["end"]])
        win["troughs"] = n_trough
        win["trough_rate"] = n_trough / max(1, win["bars"])
        row[tag] = win
    return row


#: ``scripts/omen_experiment.plan_windows`` refuses a train window that starts
#: before the lookback, and refuses ``train_end + horizon > test_start`` because
#: a training sample whose future reaches into the held-out window leaks the
#: answer. A planned window that ignores either is a command that raises rather
#: than an arm that runs, so both are checked HERE, before any node time.
LOOKBACK_BARS = 60
TRAIN_BARS = 600


def probe_plan(row: Mapping[str, Any], tag: str, *, train: int = TRAIN_BARS,
               lookback: int = LOOKBACK_BARS) -> Dict[str, Any] | None:
    """The exact ``omen_query_path_probe`` window for one corpus's pure window.

    Returns ``None`` when the corpus cannot carry the arm -- which is a
    RESULT, not a failure: a pure DOWN window sitting at bar 80 has no room
    for a 600-bar training set in front of it, and pooling it in anyway would
    either crash or silently shrink the training set and make its cell
    incomparable with the others.
    """
    win = row.get(tag)
    if not win:
        return None
    h = int(row["horizon_bars"])
    train_end = int(win["start"]) - h
    train_start = train_end - train
    if train_start < lookback:
        return None
    return {"corpus": row["corpus"], "chain": row.get("chain", "base"),
            "regime": tag,
            "test": int(win["bars"]), "test_end": int(win["end"]),
            "train": train, "train_end": train_end,
            "horizon": h, "up_rate": win["up_rate"],
            "troughs": win["troughs"]}


def print_plan(rows: Sequence[Mapping[str, Any]], tag: str, *,
               target: int = TARGET_TRADES) -> None:
    plans = [p for p in (probe_plan(r, tag) for r in rows) if p]
    plans.sort(key=lambda p: -p["troughs"])
    # DEDUPE BY TRADED PAIR, and this is a power correction rather than tidying.
    # LINK-WETH appears as five files across five chains with the same 17
    # troughs; LINK against WETH is the same two assets whichever chain quotes
    # it, so pooling all five multiplies n by five and adds no independent
    # tape. An n inflated that way is exactly the defect this item exists to
    # stop, so only the richest window per pair survives.
    seen: Dict[str, bool] = {}
    deduped = []
    for p in plans:
        pair = Path(p["corpus"]).stem.split("_", 1)[-1]
        if pair in seen:
            continue
        seen[pair] = True
        deduped.append(p)
    dropped = len(plans) - len(deduped)
    plans = deduped
    if dropped:
        print(f"  ({dropped} windows dropped as repeats of a pair already "
              f"counted -- same two assets, different chain)")
    if not plans:
        print(f"{tag.upper()} PLAN: no corpus has a pure window with "
              f"{TRAIN_BARS} training bars in front of it")
        return
    print(f"{tag.upper()} PLAN -- {len(plans)} of {len(rows)} corpora can "
          f"carry a {TRAIN_BARS}-bar training set before the window")
    running = 0
    for rank, p in enumerate(plans[:8], 1):
        running += p["troughs"]
        print(f"  {rank}. {p['corpus']:<28} test={p['test']:<5} "
              f"test_end={p['test_end']:<6} train_end={p['train_end']:<6} "
              f"h={p['horizon']:<3} up_rate={p['up_rate']:.3f} "
              f"troughs={p['troughs']:<4} cumulative={running}")
        print(f"     OMEN_META_COLLECTIONS=1 "
              f"OMEN_BRAIN_ENDPOINT=http://127.0.0.1:8091 \\\n"
              f"       python -X utf8 scripts/omen_query_path_probe.py \\\n"
              f"       --corpus {p['corpus']} --chain {p['chain']} \\\n"
              f"       --horizon {p['horizon']} --train {p['train']} "
              f"--test {p['test']} \\\n"
              f"       --train-end {p['train_end']} --test-end {p['test_end']} \\\n"
              f"       --query-a temporal,geometry,cross \\\n"
              f"       --query-b temporal,geometry,cross,"
              f"self_outcome,self_error_run")
        if running >= target and rank < len(plans):
            print(f"     ^ pooling the top {rank} clears {target} TRUE troughs")
            break
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true",
                    help="emit probe-ready --test/--test-end/--train-end for "
                         "each pure window, pooled until the trade floor is met")
    ap.add_argument("--corpus", default=None,
                    help="one corpus file; default is every file under "
                         "data/historical_ohlcv whose cadence is tradeable")
    ap.add_argument("--horizon-minutes", type=float, default=720.0,
                    help="wall-clock horizon; 720 is 12 bars on the 3600s "
                         "corpora the pass-111 arm used")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many corpora (0 = all)")
    ap.add_argument("--json", default=None, help="write the rows here")
    args = ap.parse_args()

    if args.corpus:
        paths = [Path(args.corpus)]
    else:
        paths = sorted(Path("data/historical_ohlcv").rglob("*.json"))
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for path in paths:
        if args.limit and len(rows) >= args.limit:
            break
        try:
            row = census_corpus(path, args.horizon_minutes)
        except Exception as exc:  # a broken download is data, not a crash
            print(f"  skip {path.name}: {type(exc).__name__} {exc}")
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    print(f"round-trip cost {ROUND_TRIP_COST:.4f}  "
          f"horizon {args.horizon_minutes:.0f} min  "
          f"purity UP>={UP_PURITY:.2f} DOWN<={DOWN_PURITY:.2f}  "
          f"min window {MIN_WINDOW_BARS} bars")
    print(f"corpora measured {len(rows)}, skipped {skipped}")
    print()

    for tag in ("up", "down"):
        wins = [r[tag] for r in rows if r.get(tag)]
        if not wins:
            print(f"{tag.upper()}: no corpus has a pure window of "
                  f"{MIN_WINDOW_BARS}+ bars")
            continue
        lengths = sorted(w["bars"] for w in wins)
        troughs = sorted(w["troughs"] for w in wins)
        total_troughs = sum(w["troughs"] for w in wins)
        print(f"{tag.upper()} -- {len(wins)} corpora have a pure window")
        print(f"  longest window bars: med {lengths[len(lengths)//2]} "
              f"max {lengths[-1]}")
        print(f"  TRUE troughs in it : med {troughs[len(troughs)//2]} "
              f"max {troughs[-1]}  total across corpora {total_troughs}")
        best = max(wins, key=lambda w: w["troughs"])
        for name, rate in FIRE_RATES.items():
            # The arm trades a trough only when it CALLS one. Scale the
            # window's bar count by the measured call rate, not its trough
            # count -- the node fires on bars, right or wrong.
            single = best["bars"] * rate
            need = TARGET_TRADES / rate
            pooled = need / max(1.0, float(best["bars"]))
            print(f"  at {name:<28} best single window places "
                  f"{single:5.1f} trades; n={TARGET_TRADES} needs "
                  f"{need:7.0f} held-out bars = {pooled:5.1f} such windows")
        print()

    if args.plan:
        for tag in ("down", "up"):
            print_plan(rows, tag)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
