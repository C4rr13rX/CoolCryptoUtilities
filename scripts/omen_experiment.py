"""Measure the omen brain: does it PRODUCE perfectly, and does it PAY?

Four numbers, in the order they matter. The first three are controls; a
held-out score without them has already fooled this repo once (a "78%
directional accuracy" that was the market's own down-drift, inverted,
because the model was answering the same thing to everything).

  1. TRAIN RECALL      -- re-predict frames the brain was trained on. This
                          is the "produce perfectly" bar. Anything below
                          ~100% means the substrate is losing bindings and
                          no held-out number from it can be trusted.
  2. GARBAGE CONTROL   -- predict on frames built from noise. A healthy
                          brain returns few or no confident actionable
                          omens here, and more than one distinct token
                          across the whole set.
  3. HELD-OUT          -- exact-label accuracy on bars strictly after the
                          training window, against TWO baselines: the
                          majority class, and the market's own up-rate.
  4. NET-OF-COST P/L   -- act on every admitted ``trough`` (buy, exit
                          ``horizon`` bars later), charge ROUND_TRIP_COST on
                          every round trip, and report the total. This is
                          the only line that is about money.

Usage
-----
  python -X utf8 scripts/omen_experiment.py --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
      --train 3000 --test 600 --horizon 12

The node it talks to defaults to OMEN_BRAIN_ENDPOINT (127.0.0.1:8091) --
its own process with its own brain dir, so this never touches the fabric
the live regime signal reads on :8090.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import (  # noqa: E402
    COLLECTIONS, LOOKBACK_BARS, OMEN_ACTIONS, OMEN_CREST, OMEN_LABELS,
    OMEN_MURK, OMEN_TROUGH, PREDICT_COLLECTIONS, ROUND_TRIP_COST, OmenBrain,
    build_collections, collection_distinctness, discriminating_collections,
    label_omen, label_regime, omen_threshold,
)


def _pct(values: Sequence[float]) -> str:
    """min / median / p90 / max, so a floor can be read off the gap."""
    if not values:
        return "n/a"
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    return (f"min {ordered[0]:.3f} med {at(0.5):.3f} "
            f"p90 {at(0.9):.3f} max {ordered[-1]:.3f}")


def load_bars(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def bar_seconds(bars: Sequence[Mapping[str, Any]]) -> int:
    """The corpus's own cadence, measured -- never assumed to be hourly."""
    gaps = [int(bars[i]["timestamp"]) - int(bars[i - 1]["timestamp"])
            for i in range(1, min(len(bars), 400))]
    gaps = [g for g in gaps if g > 0]
    return int(sorted(gaps)[len(gaps) // 2]) if gaps else 3600


def build_samples(bars, symbol, chain, horizon, start, stop):
    """Frames + true label for every bar in [start, stop) that has both."""
    samples = []
    for index in range(max(start, LOOKBACK_BARS), stop):
        label = label_omen(bars, index, horizon_bars=horizon)
        if label is None:
            continue
        try:
            frames = build_collections(bars, index, horizon_bars=horizon,
                                       symbol=symbol, chain=chain)
        except (ValueError, IndexError):
            continue
        samples.append({
            "index": index,
            "frames": frames,
            "label": label,
            "regime": label_regime(bars, index),
            "ts": int(bars[index]["timestamp"]),
            "price": float(bars[index]["close"]),
            "forward": (float(bars[index + horizon]["close"])
                        - float(bars[index]["close"])) / float(bars[index]["close"]),
        })
    return samples


def balance(samples, rng):
    """Undersample the majority so decode cannot collapse to one attractor.

    Measured 2026-07: unbalanced training at scale made every prediction
    "steady". ``murk`` is the majority here for the same structural reason,
    so it is capped at the mean count of the actionable labels.
    """
    by_label: Dict[str, List[dict]] = {}
    for sample in samples:
        by_label.setdefault(sample["label"], []).append(sample)
    actionable = [len(v) for k, v in by_label.items() if k != OMEN_MURK]
    if not actionable:
        return samples
    cap = max(1, int(sum(actionable) / len(actionable)))
    out: List[dict] = []
    for label, group in by_label.items():
        if label == OMEN_MURK and len(group) > cap:
            # Prefer recent murk bars -- the recent regime is the one we
            # will be trading in.
            out.extend(group[-cap:])
        else:
            out.extend(group)
    rng.shuffle(out)
    return out


def garbage_frames(rng, count):
    """Frames with the right SHAPE and meaningless content.

    Same prefixes and key names as a real frame, random bucket values. A
    brain that answers these as confidently as real ones is pattern-free.
    """
    out = []
    for _ in range(count):
        frames = {}
        for collection in COLLECTIONS:
            tokens = " ".join(
                f"k{i}={rng.choice('udz')}{rng.randrange(25)}" for i in range(6))
            frames[collection.name] = f"{collection.prefix} {tokens}"
        out.append(frames)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=12,
                        help="bars ahead the omen is about")
    parser.add_argument("--recall-sample", type=int, default=200)
    parser.add_argument("--garbage", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--report-dir", default="data/brain_experiments")
    parser.add_argument("--skip-train", action="store_true",
                        help="re-measure an already-trained fabric; the "
                             "sample split is deterministic in --seed so the "
                             "recall set is the same one it was taught")
    parser.add_argument("--query-collections", default=None,
                        help="comma-separated override of the collections a "
                             "PREDICTION fires (default: measured)")
    parser.add_argument("--consensus", action="store_true",
                        help="require CONSENSUS_QUERIES to agree; abstains "
                             "with verdict='split' when they do not")
    parser.add_argument("--guess-regime", action="store_true",
                        help="let stage 1 guess the regime instead of "
                             "computing it -- the pre-2026-09-07 behaviour, "
                             "kept so the fix stays falsifiable")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = load_bars(path)
    cadence = bar_seconds(bars)
    threshold = omen_threshold()

    print(f"corpus {path.name}: {len(bars)} bars, {cadence}s cadence, "
          f"symbol {symbol}")
    print(f"horizon {args.horizon} bars = {args.horizon * cadence / 60:.0f} min; "
          f"omen threshold {threshold:.4%} "
          f"(round trip {ROUND_TRIP_COST:.4%})")

    # Chronological split: train strictly before test, and leave a horizon
    # gap so no training sample's FUTURE overlaps a test bar.
    need = args.train + args.horizon + args.test + LOOKBACK_BARS + args.horizon
    if len(bars) < need:
        print(f"corpus too short: need {need} bars, have {len(bars)}")
        return 2
    test_stop = len(bars) - args.horizon - 1
    test_start = test_stop - args.test
    train_stop = test_start - args.horizon
    train_start = train_stop - args.train

    train_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                  train_start, train_stop)
    test_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                 test_start, test_stop)
    print(f"train bars [{train_start}, {train_stop}) -> {len(train_samples)} samples")
    print(f"test  bars [{test_start}, {test_stop}) -> {len(test_samples)} samples")
    print("train label mix :", dict(Counter(s['label'] for s in train_samples)))
    print("test  label mix :", dict(Counter(s['label'] for s in test_samples)))

    brain = OmenBrain(endpoint=args.endpoint)
    if not brain.supports_multi():
        print("FAIL: node has no /brain/predict/multi -- stale binary or wrong port")
        return 3

    balanced = balance(train_samples, rng)
    print(f"balanced train   : {len(balanced)} samples, "
          f"{dict(Counter(s['label'] for s in balanced))}")

    # 0 -- THE DILUTION LAW. Which collections can discriminate at all on
    # THIS corpus, measured rather than inherited. A query fires the sharp
    # ones only; see the table above PREDICT_COLLECTIONS in trading/omen_brain.
    frame_sets = [s["frames"] for s in balanced]
    distinctness = collection_distinctness(frame_sets)
    measured_query = discriminating_collections(frame_sets)
    print("0. DISTINCTNESS  : " + " ".join(
        f"{name}={distinctness[name]:.3f}"
        for name in sorted(distinctness, key=lambda k: -distinctness[k])))
    print(f"   measured query: {measured_query}   default: {PREDICT_COLLECTIONS}")
    query = (tuple(n.strip() for n in args.query_collections.split(","))
             if args.query_collections else None)
    if query:
        print(f"   OVERRIDE      : {query}")

    # Frame collisions cap recall no matter how good the substrate is: two
    # identical frame tuples carrying different labels cannot both be
    # reproduced. Reported so a recall number is never blamed on the brain
    # when it belongs to the corpus.
    keyed: Dict[tuple, set] = {}
    for sample in balanced:
        keyed.setdefault(tuple(sorted(sample["frames"].items())), set()).add(
            sample["label"])
    conflicts = sum(1 for labels in keyed.values() if len(labels) > 1)
    print(f"   collisions    : {len(balanced) - len(keyed)} duplicate tuples, "
          f"{conflicts} with conflicting labels "
          f"(recall ceiling {1.0 - conflicts / max(1, len(balanced)):.1%})")

    started = time.time()
    if args.skip_train:
        print("skipping training -- re-measuring the fabric already on the node")
        train_secs = 0.0
    else:
        for count, sample in enumerate(balanced, 1):
            brain.train(sample["frames"], sample["label"], sample["regime"])
            if count % 250 == 0:
                rate = count / max(1e-9, time.time() - started)
                print(f"  trained {count}/{len(balanced)} ({rate:.1f}/s, "
                      f"{brain.failed_pairs} failed)")
        train_secs = time.time() - started
        print(f"trained {brain.trained_pairs} pairs, {brain.failed_pairs} failed, "
              f"in {train_secs / 60:.1f} min")

    def predict(sample):
        return brain.predict(
            sample["frames"], symbol=symbol, chain=args.chain,
            as_of_ts=sample["ts"], price=sample["price"],
            horizon_bars=args.horizon, bar_seconds=cadence,
            # The regime is arithmetic here: we hold the bars. --guess-regime
            # restores the old chain so the two can be compared in one run.
            regime=None if args.guess_regime else sample["regime"],
            query_collections=query, consensus=args.consensus)

    # 1 -- TRAIN RECALL. Does it reproduce what it was taught?
    recall_set = rng.sample(balanced, min(args.recall_sample, len(balanced)))
    recall_hits = 0
    recall_split = 0
    recall_misses: List[Dict[str, str]] = []
    for sample in recall_set:
        omen = predict(sample)
        if omen.verdict == "split":
            recall_split += 1
            continue
        if omen.omen == sample["label"] and omen.verdict == "admitted":
            recall_hits += 1
        elif len(recall_misses) < 12:
            recall_misses.append({"expected": sample["label"],
                                  "got": omen.omen, "verdict": omen.verdict})
    # Recall is scored over the asks the brain ANSWERED. An abstention is
    # neither a hit nor a miss, and the abstention rate is printed beside it
    # so a high recall bought by refusing everything cannot hide.
    answered = len(recall_set) - recall_split
    recall = recall_hits / max(1, answered)
    print(f"\n1. TRAIN RECALL  : {recall:.1%} ({recall_hits}/{answered} answered"
          + (f", {recall_split} split/abstained of {len(recall_set)}"
             if recall_split else "") + ")")
    if recall_misses:
        print("   misses       :", recall_misses[:6])

    # 2 -- GARBAGE CONTROL.
    garbage_labels: List[str] = []
    garbage_conf: List[float] = []
    garbage_actionable = 0
    for frames in garbage_frames(rng, args.garbage):
        # No regime is supplied: noise has no bars to compute one from, and
        # this is the one caller that genuinely cannot.
        omen = brain.predict(frames, symbol="garbage", chain=args.chain,
                             as_of_ts=0, price=1.0, horizon_bars=args.horizon,
                             bar_seconds=cadence, query_collections=query)
        garbage_labels.append(f"{omen.omen}/{omen.verdict}")
        garbage_conf.append(omen.confidence)
        if omen.is_actionable:
            garbage_actionable += 1
    print(f"2. GARBAGE       : {len(set(garbage_labels))} distinct answers "
          f"of {len(garbage_labels)}, {garbage_actionable} actionable, "
          f"conf {_pct(garbage_conf)}")

    # 3 -- HELD-OUT, against two baselines.
    predicted: List[str] = []
    exact = 0
    trades: List[float] = []
    admitted = 0
    real_conf: List[float] = []
    #: Every buy omen, paired with its confidence, so the floor that would
    #: have been best is READ OFF the run instead of guessed at.
    buys_by_conf: List[tuple] = []
    for sample in test_samples:
        omen = predict(sample)
        predicted.append(omen.omen if omen.verdict == "admitted" else "__hold__")
        real_conf.append(omen.confidence)
        if omen.verdict == "admitted":
            admitted += 1
            if omen.omen == sample["label"]:
                exact += 1
        # 4 -- money. Only a BUY-LOW omen opens a position; the live lane is
        # long-only, so a crest is an abstention, not a short.
        if omen.is_actionable and omen.action == "buy":
            trades.append(sample["forward"] - ROUND_TRIP_COST)
            buys_by_conf.append((omen.confidence, sample["forward"] - ROUND_TRIP_COST))

    truth = Counter(s["label"] for s in test_samples)
    majority = max(truth.values()) / max(1, len(test_samples))
    up_rate = sum(1 for s in test_samples if s["forward"] > 0) / max(1, len(test_samples))
    accuracy = exact / max(1, admitted)
    print(f"3. HELD-OUT      : {accuracy:.1%} exact on {admitted} admitted "
          f"of {len(test_samples)}")
    print(f"   baselines     : majority class {majority:.1%}, market up-rate {up_rate:.1%}")
    print(f"   predicted mix : {dict(Counter(predicted))}")

    # 4 -- money, net of cost.
    total = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    hit = wins / max(1, len(trades))
    buy_and_hold = [s["forward"] - ROUND_TRIP_COST for s in test_samples]
    print(f"4. NET OF COST   : {len(trades)} buy omens, {hit:.1%} paid, "
          f"total {total:+.4f} ({total / max(1, len(trades)):+.4%} per trade)")
    print(f"   every-bar buy : {len(buy_and_hold)} trades, "
          f"{sum(buy_and_hold) / max(1, len(buy_and_hold)):+.4%} per trade")
    print(f"   real conf     : {_pct(real_conf)}")

    # Confidence sweep -- what a floor would have done. Reported, never
    # auto-applied: a floor picked to maximise the number it is scored on is
    # not a floor, it is a fit.
    sweep = []
    for floor in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        kept = [pnl for conf, pnl in buys_by_conf if conf >= floor]
        sweep.append({
            "floor": floor, "trades": len(kept),
            "per_trade": (sum(kept) / len(kept)) if kept else 0.0,
            "hit_rate": (sum(1 for p in kept if p > 0) / len(kept)) if kept else 0.0,
        })
    print("   conf sweep    : " + " | ".join(
        f"{s['floor']:.2f}->{s['trades']}t {s['per_trade']:+.3%}" for s in sweep))

    report = {
        "corpus": str(path), "symbol": symbol, "bars": len(bars),
        "bar_seconds": cadence, "horizon_bars": args.horizon,
        "round_trip_cost": ROUND_TRIP_COST, "omen_threshold": threshold,
        "train_window": [train_start, train_stop],
        "test_window": [test_start, test_stop],
        "trained_pairs": brain.trained_pairs, "failed_pairs": brain.failed_pairs,
        "train_seconds": round(train_secs, 1),
        "skipped_training": bool(args.skip_train),
        "collection_distinctness": distinctness,
        "measured_query_collections": list(measured_query),
        "query_collections": list(query or PREDICT_COLLECTIONS),
        "regime_source": "stage1_guess" if args.guess_regime else "computed",
        "conflicting_frame_tuples": conflicts,
        "recall_ceiling": 1.0 - conflicts / max(1, len(balanced)),
        "consensus": bool(args.consensus),
        "train_recall": recall, "recall_sample": len(recall_set),
        "recall_answered": answered, "recall_abstained": recall_split,
        "recall_misses": recall_misses,
        "garbage_distinct": len(set(garbage_labels)),
        "garbage_total": len(garbage_labels),
        "garbage_actionable": garbage_actionable,
        "garbage_confidence": sorted(garbage_conf),
        "real_confidence_percentiles": _pct(real_conf),
        "confidence_sweep": sweep,
        "heldout_admitted": admitted, "heldout_total": len(test_samples),
        "heldout_exact_accuracy": accuracy,
        "baseline_majority": majority, "baseline_up_rate": up_rate,
        "predicted_mix": dict(Counter(predicted)),
        "true_mix": dict(truth),
        "buy_omens": len(trades), "buy_hit_rate": hit,
        "buy_net_total": total,
        "buy_net_per_trade": total / max(1, len(trades)),
        "every_bar_net_per_trade": (sum(buy_and_hold) / max(1, len(buy_and_hold))),
    }
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = report_dir / f"omen-{symbol}-h{args.horizon}-{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
