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


class WindowError(ValueError):
    """The requested train/test split does not fit the corpus."""


def plan_windows(total_bars: int, train: int, test: int, horizon: int,
                 train_end: int | None = None,
                 test_end: int | None = None) -> Dict[str, int]:
    """Bar indices for the train and held-out windows.

    Pure arithmetic, kept out of ``main`` so the one property this whole
    two-window protocol rests on can be tested without a node: with
    ``train_end`` pinned, moving ``test_end`` must NOT move the train
    window. Two held-out windows measured against DIFFERENT training data
    are not two measurements of one fabric, they are two experiments, and
    comparing them says nothing -- which is the trap the default derivation
    walks into, because it computes ``train_stop`` from ``test_start``.

    A full ``horizon`` gap sits between train_stop and test_start so no
    training sample's future overlaps a held-out bar.
    """
    test_stop = (total_bars - horizon - 1) if test_end is None else int(test_end)
    test_stop = min(test_stop, total_bars - horizon - 1)
    test_start = test_stop - test
    if train_end is None:
        train_stop = test_start - horizon
    else:
        train_stop = int(train_end)
        if train_stop + horizon > test_start:
            raise WindowError(
                f"train_end {train_stop} + horizon {horizon} overruns "
                f"test_start {test_start}: the training window's future would "
                f"overlap the held-out window, which leaks the answer")
    train_start = train_stop - train
    if train_start < LOOKBACK_BARS:
        raise WindowError(
            f"train window starts at {train_start}, before the {LOOKBACK_BARS}-bar "
            f"lookback: need {LOOKBACK_BARS + train + horizon + test + horizon} "
            f"bars before this test window, have {total_bars}")
    return {"train_start": train_start, "train_stop": train_stop,
            "test_start": test_start, "test_stop": test_stop}


def window_regime(bars: Sequence[Mapping[str, Any]], start: int, stop: int,
                  horizon: int) -> Dict[str, Any]:
    """Up-rate and drift of a bar range, so a window's DIRECTION is stated.

    The standing rule here is that a single window is never evidence: a
    long-only rule flatters itself in an up window, and that error has
    already produced a fake 78% and a fake +0.9067% in this repo. So every
    window a run touches reports whether it was UP or DOWN, measured, and
    the report carries it.
    """
    forwards = []
    for index in range(max(start, 0), min(stop, len(bars) - horizon)):
        close = float(bars[index]["close"])
        if close <= 0:
            continue
        forwards.append((float(bars[index + horizon]["close"]) - close) / close)
    if not forwards:
        return {"bars": 0, "up_rate": 0.0, "mean_forward": 0.0,
                "zero_share": 0.0, "regime": "EMPTY"}
    # Bars that did not MOVE are neither up nor down, and they must not be
    # counted as down. A frozen feed republishing one price gives every bar
    # a forward of exactly 0.0; scoring those as "not up" reads up_rate 0.0
    # and would name a dead feed the DOWN window. This repo has shipped
    # frozen feeds before -- 82 of 94 symbols once held a seed price -- so
    # the zero share is measured, reported, and gates the verdict.
    moved = [f for f in forwards if f != 0.0]
    zero_share = 1.0 - len(moved) / len(forwards)
    up_rate = (sum(1 for f in moved if f > 0) / len(moved)) if moved else 0.0
    if zero_share > 0.5:
        # More than half the window did not move: that is not a direction,
        # and it is a reason to distrust the corpus slice.
        regime = "FLAT"
    else:
        # 50% +/- 2 points is not a direction either, it is a coin. Naming
        # that band FLAT stops a 50.4% window being called "the UP window".
        regime = "UP" if up_rate > 0.52 else "DOWN" if up_rate < 0.48 else "FLAT"
    return {"bars": len(forwards), "up_rate": up_rate, "zero_share": zero_share,
            "mean_forward": sum(forwards) / len(forwards), "regime": regime,
            "ts_start": int(bars[max(start, 0)]["timestamp"]),
            "ts_stop": int(bars[min(stop, len(bars)) - 1]["timestamp"])}


def backpressure_probe(endpoint: str | None) -> Dict[str, Any]:
    """Ask the node whether it will actually LEARN, before we spend an hour.

    Measured 2026-09-10 pass 108: a node with a 4096 MB consolidation floor
    on a box with 2903 MB free answers /health OK, returns a full plausible
    /brain/stats, and serves /brain/observe in 0.16s -- while REFUSING every
    supervised binding. total_binding froze at 415 while the training loop
    still looked alive.

    The client retries a refused sample WIZARD_BACKPRESSURE_RETRIES times
    (default 30) at 2s across two stages, so a run under backpressure does
    not fail fast: it takes up to 120 SECONDS PER SAMPLE and reports
    failed_pairs at the end of a window that has already closed. Nothing was
    dishonest about that accounting -- it was just far too late to act on.

    A result measured under backpressure is not a weak result, it is not a
    result: the fabric never learned the samples the report says it taught.
    So this is a hard stop, not a warning.
    """
    from http.client import HTTPConnection  # local: only the preflight needs it
    from urllib.parse import urlparse

    target = endpoint or os.getenv("OMEN_BRAIN_ENDPOINT") or "http://127.0.0.1:8091"
    parsed = urlparse(target if "//" in target else f"http://{target}")
    try:
        conn = HTTPConnection(parsed.hostname or "127.0.0.1",
                              parsed.port or 80, timeout=15)
        # An empty stream list is a no-op binding: it asks the node's ingest
        # gate the question without teaching it anything.
        conn.request("POST", "/brain/consolidate/multi",
                     json.dumps({"streams": [], "outcome_pool": 0,
                                 "outcome_frame": ""}),
                     {"Content-Type": "application/json"})
        reply = json.loads(conn.getresponse().read() or b"{}")
    except Exception as exc:  # noqa: BLE001 -- any failure here is unknown, not OK
        return {"reachable": False, "error": str(exc)}
    return {"reachable": True, "backpressure": bool(reply.get("backpressure")),
            "available_mb": reply.get("available_mb"),
            "floor_mb": reply.get("floor_mb"),
            "retry_after_ms": reply.get("retry_after_ms")}


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
    parser.add_argument("--train-end", type=int, default=None,
                        help="PIN the last bar of the training window. Use with "
                             "--test-end to measure a second held-out window on "
                             "the SAME fabric: without it the train window is "
                             "derived from the test window, so moving the test "
                             "window silently retrains on different data")
    parser.add_argument("--test-end", type=int, default=None,
                        help="last bar of the held-out window (default: end of "
                             "corpus). This is how an UP window and a DOWN "
                             "window are each measured")
    parser.add_argument("--list-windows", action="store_true",
                        help="census the corpus for candidate held-out windows "
                             "with their up-rate and drift, then exit -- so an "
                             "UP and a DOWN window are PICKED from measurement "
                             "rather than hoped for")
    parser.add_argument("--ignore-backpressure", action="store_true",
                        help="train even when the node says it will not "
                             "consolidate. The result is NOT a measurement -- "
                             "the fabric never learns the samples the report "
                             "says it was taught")
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

    # A census of where the UP and DOWN windows actually ARE, so the two
    # windows the standing rule demands are picked from measurement.
    if args.list_windows:
        earliest = LOOKBACK_BARS + args.train + args.horizon + args.test
        latest = len(bars) - args.horizon - 1
        if latest < earliest:
            print(f"corpus too short for any window: need {earliest} bars "
                  f"before the first candidate, have {len(bars)}")
            return 2
        print(f"\ncandidate held-out windows of {args.test} bars "
              f"(train {args.train} + {args.horizon}-bar purge before each):")
        print(f"{'test_end':>9} {'regime':>7} {'up_rate':>8} {'mean_fwd':>10}  window")
        step = max(1, args.test // 2)
        for end in range(latest, earliest - 1, -step):
            info = window_regime(bars, end - args.test, end, args.horizon)
            if not info["bars"]:
                continue
            print(f"{end:>9} {info['regime']:>7} {info['up_rate']:>7.1%} "
                  f"{info['mean_forward']:>+9.4%}  "
                  f"[{end - args.test}, {end})")
        print("\nPick one UP and one DOWN end, then measure BOTH on ONE fabric:")
        print(f"  run 1: --train-end <T> --test-end <UP>")
        print(f"  run 2: --train-end <T> --test-end <DOWN> --skip-train")
        print("  <T> must satisfy T + horizon <= min(UP, DOWN) - test, so the "
              "fabric never saw either window.")
        return 0

    # Chronological split: train strictly before test, and leave a horizon
    # gap so no training sample's FUTURE overlaps a test bar.
    try:
        plan = plan_windows(len(bars), args.train, args.test, args.horizon,
                            train_end=args.train_end, test_end=args.test_end)
    except WindowError as exc:
        print(f"cannot plan windows: {exc}")
        return 2
    train_start = plan["train_start"]
    train_stop = plan["train_stop"]
    test_start = plan["test_start"]
    test_stop = plan["test_stop"]

    train_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                  train_start, train_stop)
    test_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                 test_start, test_stop)
    heldout_regime = window_regime(bars, test_start, test_stop, args.horizon)
    print(f"train bars [{train_start}, {train_stop}) -> {len(train_samples)} samples"
          + ("  (train_end PINNED)" if args.train_end is not None else ""))
    print(f"test  bars [{test_start}, {test_stop}) -> {len(test_samples)} samples")
    print(f"HELD-OUT WINDOW IS {heldout_regime['regime']}: up-rate "
          f"{heldout_regime['up_rate']:.1%}, mean forward "
          f"{heldout_regime['mean_forward']:+.4%} over "
          f"{heldout_regime['bars']} bars. ONE WINDOW IS NOT EVIDENCE -- "
          f"a long-only rule flatters itself in an UP window.")
    print("train label mix :", dict(Counter(s['label'] for s in train_samples)))
    print("test  label mix :", dict(Counter(s['label'] for s in test_samples)))

    brain = OmenBrain(endpoint=args.endpoint)
    if not brain.supports_multi():
        print("FAIL: node has no /brain/predict/multi -- stale binary or wrong port")
        return 3

    # PREFLIGHT. A node that will not consolidate makes a training run a very
    # slow no-op, and the failure is invisible until the run ends.
    if not args.skip_train:
        gate = backpressure_probe(args.endpoint)
        if gate.get("backpressure"):
            print(f"\nREFUSING TO TRAIN: the node is applying ingest backpressure.")
            print(f"  available {gate.get('available_mb')} MB against a "
                  f"{gate.get('floor_mb')} MB consolidation floor")
            print("  It answers /health, /brain/stats and /brain/observe "
                  "normally and still binds NOTHING.")
            print("  Free memory first -- kill omen nodes you are not using -- "
                  "then re-run. A run started now would take up to 120s per "
                  "sample and teach the fabric nothing.")
            print("  --ignore-backpressure overrides this, and the result is "
                  "not a measurement.")
            if not args.ignore_backpressure:
                return 4
        elif not gate.get("reachable"):
            print(f"\nWARNING: could not probe the node's ingest gate "
                  f"({gate.get('error')}). Proceeding, but if training stalls "
                  f"this is the first thing to check.")

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
    # THE MEASURED SET MUST BE THE SET THAT FIRES. Until 2026-09-10 this line
    # computed measured_query, PRINTED it, and then passed None -- so every run
    # reported the measured set and fired the hard-coded PREDICT_COLLECTIONS.
    # That silently defeats the dilution law this script exists to apply, and
    # it makes any experiment that CHANGES which collections discriminate
    # unfalsifiable: the query set never moves, so the arms cannot differ.
    # It is why the pass-108 relation arm came back byte-for-byte identical to
    # the flat arm over 180 held-out predictions.
    if args.query_collections:
        query = tuple(n.strip() for n in args.query_collections.split(","))
        print(f"   OVERRIDE      : {query}")
    else:
        query = tuple(measured_query)
        print(f"   FIRING        : {query} (measured on this corpus)")

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
        "train_end_pinned": args.train_end is not None,
        "heldout_regime": heldout_regime["regime"],
        "heldout_window_up_rate": heldout_regime["up_rate"],
        "heldout_window_mean_forward": heldout_regime["mean_forward"],
        "heldout_ts_range": [heldout_regime.get("ts_start"),
                             heldout_regime.get("ts_stop")],
        "trained_pairs": brain.trained_pairs, "failed_pairs": brain.failed_pairs,
        "train_seconds": round(train_secs, 1),
        "skipped_training": bool(args.skip_train),
        "collection_distinctness": distinctness,
        "measured_query_collections": list(measured_query),
        # What ACTUALLY fired, not what the default would have been.
        "query_collections": list(query),
        "query_source": "override" if args.query_collections else "measured",
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
    # The regime goes in the FILENAME: two windows measured on one fabric
    # produce two reports, and a reader must not have to open them to see
    # which is the UP one.
    out = (report_dir /
           f"omen-{symbol}-h{args.horizon}-{heldout_regime['regime']}-{stamp}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
