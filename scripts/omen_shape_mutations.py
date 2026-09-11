#!/usr/bin/env python
"""Chart-shape MUTATIONS: teach 'same shape', and prove the LABEL survives.

WHY THIS EXISTS. The fabric recalls 0.975 and generalises at chance (held-out
exact 0.2850 against a 0.3025 majority). That gap is memorisation: the frame at
one instant is a near-unique key, so the only thing the substrate can learn is
the instant. Mutations attack it directly -- train several DIFFERENT frames on
the SAME label, so the only thing they share is the shape.

THE TRAP, AND WHY IT IS SETTLED BY ARITHMETIC RATHER THAN BY ARGUMENT. A
mutation that changes the label is poison: it teaches the fabric to answer the
same for genuinely different futures. ``label_omen`` reads exactly three things

    entry        = close[index]
    future       = close[index + horizon]
    position     = where entry sits in [min, max] of close[index-23 .. index]

with ``RANGE_WINDOW = 24`` and ``LOOKBACK_BARS = 168``. So a mutation confined
to bars ``[index-167, index-24]`` -- the DEEP prefix -- cannot move any of the
three. The label survives by CONSTRUCTION, not by assumption, and the census
below checks it empirically anyway (flip rate must read exactly 0.0000).

That same arithmetic condemns the two mutations the item asked to be justified
or dropped, and both are implemented here so the refusal carries a number:

  * AMPLITUDE SCALING is DROPPED. Scaling deviations about the anchor leaves
    position-in-range invariant (it is a ratio), so the label does not flip --
    and that is exactly what makes it poison. The label's FIRST test is
    ``abs(forward) < threshold``, an absolute move against an absolute cost.
    Amplitude is the one quantity that decides murk from trough, and scaling
    it while holding the label teaches the fabric to ignore it. The census
    prints which frame slots it moves: they are the magnitude slots.
  * INVERSION is DROPPED. Mirroring the deep prefix is label-safe (it is deep),
    but mirroring the whole window flips at_low into at_high, i.e. trough into
    crest. Keeping the label is a lie; flipping it assumes this tape is
    up/down symmetric, which no measurement here supports.

THREE GATES, all node-free, all decided before any node time is spent:

  1. LABEL FLIP RATE -- must be 0.0000 or the mutation is poison.
  2. NO-OP RATE -- a mutation whose frame tuple equals the base frame tuple
     buys nothing; it is the same training pair taught twice. High no-op rate
     means the encoder's bands swallowed the mutation.
  3. CROSS-LABEL COLLISION -- a mutated frame that lands on a frame already
     carrying a DIFFERENT label lowers the recall ceiling. It is the cost of
     the mutation and it is reported, not hidden.

Usage:
    python scripts/omen_shape_mutations.py census \
        --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
        --start 841 --stop 1441 --samples 120

Exit code is nonzero when an ADMITTED mutation flips a label -- so this is an
acceptance test, not a transcript.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trading.omen_brain import (  # noqa: E402
    COLLECTIONS, LOOKBACK_BARS, RANGE_WINDOW, build_collections, label_omen,
    label_regime,
)

#: Bars strictly older than this many back from the anchor cannot touch the
#: label. Derived from the labeller, never hard-coded: if RANGE_WINDOW moves,
#: the safe prefix moves with it.
DEEP_EDGE = RANGE_WINDOW


def _closes(window: Sequence[Mapping[str, Any]]) -> List[float]:
    return [float(b["close"]) for b in window]


def _rescale_bar(bar: Mapping[str, Any], old_close: float,
                 new_close: float) -> Dict[str, Any]:
    """Move a bar to a new close, carrying its OHLC geometry with it.

    The frame reads body and wick as ratios of the bar's own range, so a bar
    whose close moves without its high/low moving is a different CANDLE, not
    the same candle at a different price. Everything is shifted by the same
    delta and the high/low are widened only enough to keep the candle valid.
    """
    delta = new_close - old_close
    out = dict(bar)
    out["close"] = new_close
    out["open"] = float(bar["open"]) + delta
    high = float(bar["high"]) + delta
    low = float(bar["low"]) + delta
    out["high"] = max(high, new_close, out["open"])
    out["low"] = min(low, new_close, out["open"])
    return out


def _apply_closes(window: List[Dict[str, Any]], closes: Sequence[float],
                  lo: int, hi: int) -> List[Dict[str, Any]]:
    out = list(window)
    for i in range(lo, hi):
        out[i] = _rescale_bar(window[i], float(window[i]["close"]), closes[i])
    return out


# --------------------------------------------------------------------------
# The mutations. Each takes the 169-bar lookback window (anchor LAST) plus the
# horizon tail, and returns a new window. ``deep`` mutations touch only
# indices [0, len - 1 - DEEP_EDGE) -- provably label-safe.
# --------------------------------------------------------------------------

def _deep_span(window: Sequence[Mapping[str, Any]], anchor: int) -> tuple:
    """[lo, hi) of bars that cannot touch the label."""
    return 0, max(0, anchor - DEEP_EDGE + 1)


def mut_deep_jitter(window, anchor, rng, strength):
    """Gaussian noise on the deep prefix, scaled to that prefix's own step size.

    LABEL SURVIVES: every touched bar is more than RANGE_WINDOW back, so it
    enters neither the entry price, the future price, nor the min/max that
    position-in-range is taken over.
    """
    lo, hi = _deep_span(window, anchor)
    closes = _closes(window)
    steps = [abs(closes[i] - closes[i - 1]) for i in range(max(1, lo), hi)]
    scale = (sum(steps) / len(steps)) if steps else 0.0
    moved = list(closes)
    for i in range(lo, hi):
        moved[i] = max(1e-12, closes[i] + rng.gauss(0.0, strength * scale))
    return _apply_closes(list(window), moved, lo, hi)


def mut_deep_flatten(window, anchor, rng, strength):
    """Truncation: replace the OLDEST share of the deep prefix with a flat run.

    This is the item's 'truncated' variant expressed without changing the frame
    length -- a short frame is a different byte string, so padding would create
    a second atom for the same situation (build_collections says so at :483).

    LABEL SURVIVES: deep-only, as above.
    """
    lo, hi = _deep_span(window, anchor)
    closes = _closes(window)
    cut = lo + int((hi - lo) * strength)
    if cut <= lo:
        return list(window)
    level = closes[cut]
    moved = list(closes)
    for i in range(lo, cut):
        moved[i] = level
    return _apply_closes(list(window), moved, lo, cut)


def mut_deep_dilate(window, anchor, rng, strength):
    """Time stretch: resample the deep prefix so the same shape runs slower.

    The prefix is read at a stride of ``strength`` and linearly interpolated
    back onto the same number of slots, so the shape is preserved and its
    SPEED is not. This is the mutation that most directly says 'same shape,
    different tempo'.

    LABEL SURVIVES: deep-only, as above.
    """
    lo, hi = _deep_span(window, anchor)
    n = hi - lo
    if n < 4:
        return list(window)
    closes = _closes(window)
    src = closes[lo:hi]
    moved = list(closes)
    for k in range(n):
        # Read position: the tail of the prefix is pinned so the join to the
        # untouched recent bars stays continuous.
        pos = (n - 1) - (n - 1 - k) * strength
        if pos < 0:
            pos = 0.0
        left = int(math.floor(pos))
        right = min(n - 1, left + 1)
        frac = pos - left
        moved[lo + k] = src[left] * (1.0 - frac) + src[right] * frac
    return _apply_closes(list(window), moved, lo, hi)


def mut_amplitude(window, anchor, rng, strength):
    """DROPPED -- implemented so the refusal carries a number, not an opinion.

    Scales every close's deviation from the anchor. Position-in-range is a
    ratio so it is invariant and the label does NOT flip -- which is the trap:
    the label's first test is abs(forward) against an absolute cost, and this
    mutation moves exactly the magnitude the test reads.
    """
    closes = _closes(window)
    now = closes[anchor]
    moved = list(closes)
    for i in range(0, anchor):
        moved[i] = now * (1.0 + strength * (closes[i] / now - 1.0))
    return _apply_closes(list(window), moved, 0, anchor)


def mut_invert(window, anchor, rng, strength):
    """DROPPED -- mirrors the WHOLE lookback about the anchor close.

    Kept for the census because it is the clearest demonstration of a poison
    mutation: it flips at_low into at_high, i.e. trough into crest, while the
    caller would be holding the label constant.
    """
    closes = _closes(window)
    now = closes[anchor]
    moved = list(closes)
    for i in range(0, anchor):
        moved[i] = max(1e-12, 2.0 * now - closes[i])
    return _apply_closes(list(window), moved, 0, anchor)


#: name -> (fn, default strength, admitted, why the label survives / does not)
MUTATIONS: Dict[str, tuple] = {
    "deep_jitter": (mut_deep_jitter, 1.0, True,
                    "deep-only: touched bars are >RANGE_WINDOW back, so entry, "
                    "future and position-in-range are all untouched"),
    "deep_flatten": (mut_deep_flatten, 0.5, True,
                     "deep-only: truncation of the oldest half of the safe "
                     "prefix, frame length preserved"),
    "deep_dilate": (mut_deep_dilate, 0.6, True,
                    "deep-only: same shape at a different tempo, joined "
                    "continuously to the untouched recent bars"),
    "amplitude": (mut_amplitude, 1.6, False,
                  "DROPPED: label does not flip, and that is the trap -- it "
                  "moves the magnitude the cost threshold is taken on"),
    "invert": (mut_invert, 1.0, False,
               "DROPPED: mirrors position-in-range, so trough becomes crest; "
               "holding the label is a lie and flipping it assumes a symmetry "
               "this tape has never been shown to have"),
}

ADMITTED = tuple(k for k, v in MUTATIONS.items() if v[2])


def mutated_sample(bars: Sequence[Mapping[str, Any]], index: int, *,
                   horizon: int, symbol: str, chain: str, kind: str,
                   rng: random.Random, strength: float | None = None):
    """One mutated (frames, label) pair, plus what the label WOULD have been.

    Returns ``None`` when the base bar is unlabelable. The returned ``label``
    is always the BASE label -- that is the whole point -- and ``mutated_label``
    is what the labeller says about the mutated series, so a caller can refuse
    any pair where they disagree.
    """
    fn, default_strength, _admitted, _why = MUTATIONS[kind]
    strength = default_strength if strength is None else strength
    lo = index - LOOKBACK_BARS
    hi = index + horizon + 1
    if lo < 0 or hi > len(bars):
        return None
    window = [dict(b) for b in bars[lo:hi]]
    anchor = LOOKBACK_BARS
    base_label = label_omen(window, anchor, horizon_bars=horizon)
    if base_label is None:
        return None
    moved = fn(window, anchor, rng, strength)
    try:
        frames = build_collections(moved, anchor, horizon_bars=horizon,
                                   symbol=symbol, chain=chain)
        base_frames = build_collections(window, anchor, horizon_bars=horizon,
                                        symbol=symbol, chain=chain)
    except (ValueError, IndexError):
        return None
    return {
        "index": index,
        "kind": kind,
        "frames": frames,
        "base_frames": base_frames,
        "label": base_label,
        "mutated_label": label_omen(moved, anchor, horizon_bars=horizon),
        "regime": label_regime(moved, anchor),
        "ts": int(bars[index]["timestamp"]),
        "price": float(bars[index]["close"]),
        "forward": (float(bars[index + horizon]["close"])
                    - float(bars[index]["close"])) / float(bars[index]["close"]),
    }


def _token_diff(a: str, b: str) -> float:
    at, bt = a.split(), b.split()
    if not at:
        return 0.0
    n = max(len(at), len(bt))
    same = sum(1 for i in range(min(len(at), len(bt))) if at[i] == bt[i])
    return 1.0 - same / n


def census(bars, *, symbol: str, chain: str, horizon: int, start: int,
           stop: int, samples: int, rng: random.Random,
           strengths: Mapping[str, float]) -> Dict[str, Any]:
    """The three gates, per mutation, over a sample of anchors."""
    anchors = [i for i in range(max(start, LOOKBACK_BARS),
                                min(stop, len(bars) - horizon))]
    if len(anchors) > samples:
        anchors = rng.sample(anchors, samples)
    anchors.sort()

    # Every base frame tuple in the window, so a mutated frame can be checked
    # for landing on a DIFFERENT label's key.
    base_keys: Dict[tuple, set] = {}
    base_by_index: Dict[int, dict] = {}
    for i in anchors:
        s = mutated_sample(bars, i, horizon=horizon, symbol=symbol, chain=chain,
                           kind="deep_jitter", rng=random.Random(0),
                           strength=0.0)
        if s is None:
            continue
        base_by_index[i] = s
        base_keys.setdefault(tuple(sorted(s["base_frames"].items())),
                             set()).add(s["label"])

    out: Dict[str, Any] = {"anchors": len(base_by_index), "mutations": {}}
    for kind in MUTATIONS:
        fn_strength = strengths.get(kind)
        flips = 0
        noops = 0
        collisions = 0
        total = 0
        slot_moved: Counter = Counter()
        divergence: List[float] = []
        for i in sorted(base_by_index):
            s = mutated_sample(bars, i, horizon=horizon, symbol=symbol,
                               chain=chain, kind=kind, rng=rng,
                               strength=fn_strength)
            if s is None:
                continue
            total += 1
            if s["mutated_label"] != s["label"]:
                flips += 1
            key = tuple(sorted(s["frames"].items()))
            if key == tuple(sorted(s["base_frames"].items())):
                noops += 1
            labels = base_keys.get(key)
            if labels and labels != {s["label"]}:
                collisions += 1
            moved_slots = 0
            for name, frame in s["frames"].items():
                d = _token_diff(s["base_frames"][name], frame)
                if d > 0:
                    slot_moved[name] += 1
                    moved_slots += 1
            divergence.append(moved_slots / max(1, len(s["frames"])))
        admitted = MUTATIONS[kind][2]
        out["mutations"][kind] = {
            "admitted": admitted,
            "why": MUTATIONS[kind][3],
            "strength": (fn_strength if fn_strength is not None
                         else MUTATIONS[kind][1]),
            "samples": total,
            "label_flip_rate": flips / max(1, total),
            "noop_rate": noops / max(1, total),
            "cross_label_collision_rate": collisions / max(1, total),
            "mean_slots_moved": (sum(divergence) / max(1, len(divergence))),
            "slots_moved": dict(slot_moved.most_common()),
        }
    return out


def run_arm(args) -> int:
    """Base pairs, optionally plus admitted mutations, scored held-out.

    ONE WINDOW PER INVOCATION, deliberately. The second window must reuse the
    SAME fabric via --skip-train: this node gave 89.2% and 93.6% on identical
    inputs thirty-four minutes apart, so two windows measured on two fabrics
    are not comparable and the UP/DOWN rule would be decoration.
    """
    import time as _time

    from scripts.omen_experiment import (bar_seconds, balance, build_samples,
                                         fabric_census, fabric_is_empty,
                                         load_bars, plan_windows,
                                         window_regime)
    from trading.omen_brain import ROUND_TRIP_COST, OmenBrain

    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = load_bars(path)
    cadence = bar_seconds(bars)
    rng = random.Random(args.seed)

    refused = [k for k in args.mutate if not MUTATIONS[k][2]]
    if refused:
        for kind in refused:
            print(f"REFUSING {kind}: {MUTATIONS[kind][3]}")
        return 2
    kinds = list(args.mutate)

    plan = plan_windows(len(bars), args.train, args.test, args.horizon,
                        train_end=args.train_end, test_end=args.test_end)
    train_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                  plan["train_start"], plan["train_stop"])
    test_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                 plan["test_start"], plan["test_stop"])
    regime = window_regime(bars, plan["test_start"], plan["test_stop"],
                           args.horizon)
    balanced = balance(train_samples, rng)

    pairs = [{"frames": s["frames"], "label": s["label"],
              "regime": s["regime"], "kind": "base"} for s in balanced]
    poisoned = 0
    for kind in kinds:
        for sample in balanced:
            mutant = mutated_sample(bars, sample["index"],
                                    horizon=args.horizon, symbol=symbol,
                                    chain=args.chain, kind=kind, rng=rng)
            if mutant is None:
                continue
            # Belt and braces. The arithmetic says a deep mutation cannot move
            # the label; if one ever does, the pair is DROPPED rather than
            # taught, and the count is reported.
            if mutant["mutated_label"] != mutant["label"]:
                poisoned += 1
                continue
            pairs.append({"frames": mutant["frames"], "label": mutant["label"],
                          "regime": mutant["regime"], "kind": kind})
    # NEGATIVE CONTROL. The mutated arm differs from base in TWO ways: the
    # mutations, and the pair count (300 -> 900). --repeat teaches the SAME
    # base pairs that many times, matching the count while adding no new
    # information. If repetition alone narrows the gap, the mutation result is
    # explained by volume and not by invariance, and it must be withdrawn.
    if args.repeat > 1:
        pairs = pairs * args.repeat
    rng.shuffle(pairs)

    arm_name = ("base+" + "+".join(kinds)) if kinds else "base"
    if args.repeat > 1:
        arm_name += f" x{args.repeat} (repetition control)"
    print(f"ARM {arm_name}")
    print(f"  train [{plan['train_start']}, {plan['train_stop']}) -> "
          f"{len(balanced)} base samples -> {len(pairs)} pairs "
          f"({len(kinds)} mutations, {poisoned} poisoned dropped)")
    print(f"  test  [{plan['test_start']}, {plan['test_stop']}) -> "
          f"{len(test_samples)} samples, window is {regime['regime']} "
          f"(up-rate {regime['up_rate']:.1%}, mean forward "
          f"{regime['mean_forward']:+.4%})")
    print(f"  train label mix: {dict(Counter(p['label'] for p in pairs))}")
    if args.dry_run:
        print("  --dry-run: nothing sent to the node")
        return 0

    brain = OmenBrain(endpoint=args.endpoint)
    if not brain.supports_multi():
        print("FAIL: node has no /brain/predict/multi -- stale binary or port")
        return 3
    before = fabric_census(args.endpoint)
    print(f"  fabric before: neurons={before.get('total_neurons')} "
          f"concepts={before.get('total_concepts')} @ {before.get('endpoint')}")
    if not args.skip_train and not fabric_is_empty(before):
        print("REFUSING TO TRAIN: this fabric is not clean. Whatever it "
              "learned before is inside every number this run would report. "
              "Start a node on a FRESH brain dir, or pass --skip-train to "
              "re-measure the fabric already there.")
        return 5

    started = _time.time()
    if args.skip_train:
        print("  skipping training -- re-measuring the fabric on the node")
    else:
        for count, pair in enumerate(pairs, 1):
            brain.train(pair["frames"], pair["label"], pair["regime"])
            if count % 200 == 0:
                print(f"    trained {count}/{len(pairs)} "
                      f"({count / max(1e-9, _time.time() - started):.1f}/s)")
        print(f"  trained {brain.trained_pairs} pairs, {brain.failed_pairs} "
              f"failed, in {(_time.time() - started) / 60:.1f} min")

    def ask(frames, ts, price, regime_name):
        return brain.predict(frames, symbol=symbol, chain=args.chain,
                             as_of_ts=ts, price=price,
                             horizon_bars=args.horizon, bar_seconds=cadence,
                             regime=regime_name)

    # RECALL over BASE pairs only. Reproducing a mutation is not what we want
    # and scoring it would inflate the one number the item forbids leading on.
    base_pairs = [p for p in pairs if p["kind"] == "base"]
    recall_set = rng.sample(base_pairs, min(args.recall_sample, len(base_pairs)))
    hits = answered = 0
    for pair in recall_set:
        omen = ask(pair["frames"], 0, 1.0, pair["regime"])
        if omen.verdict == "split":
            continue
        answered += 1
        hits += omen.omen == pair["label"]
    recall = hits / max(1, answered)

    exact = admitted = 0
    trades: List[float] = []
    buy_calls = buy_true_trough = 0
    sell_calls = sell_paid = 0
    for sample in test_samples:
        omen = ask(sample["frames"], sample["ts"], sample["price"],
                   sample["regime"])
        if omen.verdict == "admitted":
            admitted += 1
            exact += omen.omen == sample["label"]
        if omen.is_actionable and omen.action == "buy":
            buy_calls += 1
            buy_true_trough += sample["label"] == "trough"
            trades.append(sample["forward"] - ROUND_TRIP_COST)
        if omen.is_actionable and omen.action == "sell":
            # The live lane is long-only so a crest is an abstention, not a
            # short -- but a crest that correctly calls a fall is worth money
            # as an EXIT, and that half is otherwise measured nowhere.
            sell_calls += 1
            sell_paid += sample["forward"] < 0
    truth = Counter(s["label"] for s in test_samples)
    majority = max(truth.values()) / max(1, len(test_samples))
    every_bar = [s["forward"] - ROUND_TRIP_COST for s in test_samples]
    heldout = exact / max(1, len(test_samples))

    result = {
        "arm": arm_name, "corpus": str(path), "symbol": symbol,
        "horizon": args.horizon,
        "train_window": [plan["train_start"], plan["train_stop"]],
        "test_window": [plan["test_start"], plan["test_stop"]],
        "heldout_regime": regime["regime"],
        "heldout_window_up_rate": regime["up_rate"],
        "base_samples": len(balanced), "trained_pairs": len(pairs),
        "mutations": kinds, "poisoned_dropped": poisoned,
        "train_recall": recall, "recall_answered": answered,
        "heldout_exact_accuracy": heldout,
        "heldout_admitted": admitted, "heldout_total": len(test_samples),
        "majority_baseline": majority,
        "recall_generalisation_gap": recall - heldout,
        "buy_omens": buy_calls,
        "trough_precision": buy_true_trough / max(1, buy_calls),
        "buy_net_per_trade": (sum(trades) / len(trades)) if trades else None,
        "every_bar_net_per_trade": sum(every_bar) / max(1, len(every_bar)),
        "crest_omens": sell_calls,
        "crest_precision": sell_paid / max(1, sell_calls),
        "skipped_training": args.skip_train,
        "fabric_before": before,
    }
    print(f"\n  train_recall              {recall:.4f} ({answered} answered)")
    print(f"  held-out exact            {heldout:.4f} vs majority {majority:.4f}")
    print(f"  RECALL-GENERALISATION GAP {recall - heldout:+.4f}  "
          f"<- the number this item moves")
    print(f"  buy omens {buy_calls}, trough precision "
          f"{result['trough_precision']:.4f}, net/trade "
          + (f"{result['buy_net_per_trade']:+.4%}" if trades else "n/a")
          + f" vs every-bar {result['every_bar_net_per_trade']:+.4%}")
    print(f"  crest omens {sell_calls}, crest precision "
          f"{result['crest_precision']:.4f}")
    # POWER, stated beside the money number rather than left for a reader to
    # notice. Per-trade absolute returns on this feed run 2-3%, so detecting
    # 1.0pp needs roughly 63 trades per arm at sd=2% and ~141 at sd=3%. A
    # per-trade mean on a handful of trades is one number wearing a percent
    # sign, and recording it as a verdict stops anyone looking again.
    if buy_calls < 28:
        print(f"  POWER: {buy_calls} buy omens is below the most generous "
              f"detectable-effect floor (~28 trades for 1.5pp at sd=2%). The "
              f"per-trade net above licenses NO verdict either way -- the "
              f"powered comparison in this arm is the held-out accuracy over "
              f"{len(test_samples)} bars, not the money line.")
    result["buy_power_sufficient"] = buy_calls >= 28
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2))
        print(f"  wrote {args.report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["census", "arm"])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--start", type=int, default=LOOKBACK_BARS)
    parser.add_argument("--stop", type=int, default=10**9)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strength", action="append", default=[],
                        metavar="KIND=VALUE",
                        help="override a mutation's strength, repeatable")
    parser.add_argument("--report", default=None,
                        help="write the result JSON here")
    parser.add_argument("--mutate", action="append", default=[],
                        choices=list(MUTATIONS),
                        help="arm mode: add this mutation of every base pair")
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--train-end", type=int, default=None)
    parser.add_argument("--test-end", type=int, default=None)
    parser.add_argument("--recall-sample", type=int, default=100)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="re-measure the fabric already on the node -- how "
                             "the second window is made comparable to the first")
    parser.add_argument("--repeat", type=int, default=1,
                        help="negative control: teach the base pairs N times, "
                             "matching a mutated arm's pair count with no new "
                             "information")
    parser.add_argument("--dry-run", action="store_true",
                        help="size the arm and send the node nothing")
    args = parser.parse_args()

    strengths: Dict[str, float] = {}
    for pair in args.strength:
        kind, _, value = pair.partition("=")
        if kind not in MUTATIONS:
            print(f"unknown mutation {kind!r}; known: {list(MUTATIONS)}")
            return 2
        strengths[kind] = float(value)

    if args.mode == "arm":
        return run_arm(args)

    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = json.loads(path.read_text())
    if isinstance(bars, dict):
        bars = bars.get("bars", bars.get("data", []))
    rng = random.Random(args.seed)

    print(f"corpus {path.name}: {len(bars)} bars, symbol {symbol}")
    print(f"labeller reads entry, entry+{args.horizon}, and min/max over the "
          f"last {RANGE_WINDOW} bars; lookback is {LOOKBACK_BARS}, so bars "
          f"[anchor-{LOOKBACK_BARS}, anchor-{RANGE_WINDOW}) are LABEL-SAFE "
          f"by construction")
    result = census(bars, symbol=symbol, chain=args.chain,
                    horizon=args.horizon, start=args.start, stop=args.stop,
                    samples=args.samples, rng=rng, strengths=strengths)
    print(f"\nanchors {result['anchors']} in [{args.start}, {args.stop})")
    print(f"{'mutation':>14} {'adm':>4} {'flip':>7} {'no-op':>7} "
          f"{'xlabel':>7} {'slots':>7}  live slots")
    bad: List[str] = []
    for kind, row in result["mutations"].items():
        live = ", ".join(f"{k}:{v}" for k, v in
                         list(row["slots_moved"].items())[:4]) or "NONE"
        print(f"{kind:>14} {'Y' if row['admitted'] else 'n':>4} "
              f"{row['label_flip_rate']:>6.1%} {row['noop_rate']:>6.1%} "
              f"{row['cross_label_collision_rate']:>6.1%} "
              f"{row['mean_slots_moved']:>6.1%}  {live}")
        if row["admitted"] and row["label_flip_rate"] > 0:
            bad.append(kind)
        if row["admitted"] and row["noop_rate"] >= 0.5:
            bad.append(f"{kind} (no-op {row['noop_rate']:.0%})")

    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.report}")

    if bad:
        print(f"\nFAIL: admitted mutations that are not usable: {bad}")
        print("  a flip rate above zero means the mutation is POISON -- it "
              "keeps a label the future no longer supports.")
        print("  a no-op rate at or above 50% means the encoder's bands "
              "swallowed the mutation and the pair buys nothing.")
        return 1
    print("\nOK: every admitted mutation preserves the label and moves the frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
