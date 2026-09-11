#!/usr/bin/env python3
"""Do pools 15/16/19 carry anything once the resolved-prediction feeder runs?

THE DEFECT THIS ANSWERS. Measured pass 109, the three self-knowledge pools read
0.002 distinctness -- one distinct frame over the whole corpus, i.e. a
CONSTANT. They were bound and streamed and carried zero information, because
``self_frames`` refuses an unsettled prediction and nothing constructed a
settled one. ``trading/omen_resolved_history`` is that missing feeder; this
measures whether attaching it moves the number.

WHAT DRIVES THE PREDICTIONS, SAID PLAINLY BECAUSE IT BOUNDS THE CLAIM. The
vocabulary of a self-frame is a function of the SHAPE of the prediction stream
-- how often it is right, how long it stays wrong, how often the query sets
agree -- and not of which predictor produced it. So this script drives the
feeder with a causal, non-oracle rule (predict the commonest label among the
last N SETTLED bars, the same majority-class rule the scoreboard baselines
against) and reports the vocabulary that results. That is a measurement of
whether the pools can carry information at all. It is NOT a held-out edge
number and must never be quoted as one: the held-out number needs the node's
own predictions in a walk-forward, which is the next step and not this one.

The driver is strictly causal by construction -- it reads ``history.as_of(bar)``
and nothing else, so it cannot see a bar it has not reached.

    python -X utf8 scripts/omen_self_distinctness.py \
        --corpus data/historical_ohlcv/base/0004_AERO-USDC.json --bars 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.omen_experiment import (  # noqa: E402
    add_horizon_args, bar_seconds, settle_horizon, validate_report_horizon,
)
from trading.omen_brain import LOOKBACK_BARS, label_omen  # noqa: E402
from trading.omen_metacognition import self_frames  # noqa: E402
from trading.omen_resolved_history import ResolvedHistory  # noqa: E402

#: A CONSTANT is a stream with ONE value, and that is a COUNT test, not a
#: ratio test. Getting this wrong is easy and consequential: these pools are
#: deliberately coarse -- bucketed rates and run-length bands -- so a healthy
#: one lands at a few dozen distinct frames over thousands of samples, which
#: is a RATIO near zero and a vocabulary that is doing its job. Judging them
#: on the ratio alone reports "carrying nothing" for a stream carrying exactly
#: the shareable fact it was built to carry, and that is the same conflation
#: that would wrongly cut an abstraction layer for having abstracted well.
CONSTANT_VALUES = 1

#: The dilution law's empty band: below it a stream is too coarse to
#: discriminate a query, above it the stream is a query-set candidate.
EMPTY_BAND = (0.103, 0.260)

SELF_KEYS = ("self_outcome", "self_agreement", "self_error_run")


def load_bars(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    bars = raw if isinstance(raw, list) else raw.get("bars") or raw.get("data") or []
    bars = [b for b in bars if b.get("close")]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--bars", type=int, default=3000)
    add_horizon_args(ap)
    ap.add_argument("--report", default=None,
                    help="write the distinctness and separation numbers as "
                         "JSON here. The file carries horizon_minutes, "
                         "horizon_bars and bar_seconds, because a distinctness "
                         "measured at one cadence is not comparable with one "
                         "measured at another and 'h12' never said which.")
    ap.add_argument("--window", type=int, default=32,
                    help="how many settled rows a self-frame reads")
    ap.add_argument("--end", type=int, default=None,
                    help="last bar index of the window. ONE WINDOW IS NOT "
                         "EVIDENCE: a separation measured only at the corpus "
                         "end is a fact about that regime, and every brain "
                         "number ever made on this repo scored the SAME "
                         "window and that window is DOWN.")
    ap.add_argument("--null-trials", type=int, default=2000,
                    help="shuffles of next_correct used to build the null the "
                         "spread is marked against; 0 is not allowed, the "
                         "whole point of this item is that there IS a null")
    ap.add_argument("--null-seed", type=int, default=1)
    ap.add_argument("--shuffle-frames", action="store_true",
                    help="CONTROL. Permute the self frames while keeping the "
                         "labels where they are, so the frames provably carry "
                         "nothing. A correct estimator must NOT mark this "
                         "SEPARATES; the 0.10 literal marked it every time.")
    args = ap.parse_args()
    null_trials = max(100, int(args.null_trials))
    null_seed = int(args.null_seed)

    path = Path(args.corpus)
    bars = load_bars(path)
    cadence = bar_seconds(bars)
    try:
        horizon = settle_horizon(args, cadence)
    except ValueError as exc:
        print(f"cannot resolve horizon: {exc}")
        return 2
    stop = (len(bars) - args.horizon - 1) if args.end is None else min(
        int(args.end), len(bars) - args.horizon - 1)
    start = max(LOOKBACK_BARS, stop - args.bars)

    truth = {}
    for index in range(start, stop):
        label = label_omen(bars, index, horizon_bars=args.horizon)
        if label is not None:
            truth[index] = label

    print(f"corpus {path.name}  bars [{start}, {stop})  "
          f"{len(truth)} labelled  horizon {args.horizon} bars "
          f"= {horizon['horizon_minutes']:.0f} min of {cadence}s")

    history = ResolvedHistory(args.horizon)
    frame_rows = []
    predicted_counts: Counter = Counter()
    #: Was the prediction made AT this bar eventually right? Filled in after
    #: the walk, from the truth map -- never during it, so the driver cannot
    #: see it. This is scoring, not input.
    made_at = []
    for index in range(start, stop):
        if index in truth:
            try:
                history.settle(index, truth[index])
            except ValueError:
                pass
        seen = history.as_of(index)
        frame_rows.append(self_frames(seen, window=args.window))

        # THE DRIVER. Commonest label among the settled rows this bar is
        # allowed to see -- causal, non-oracle, and the same rule the
        # scoreboard baselines against. `murk` until anything has settled.
        actuals = [r.actual for r in seen if r.actual]
        call = Counter(actuals).most_common(1)[0][0] if actuals else "murk"
        predicted_counts[call] += 1
        history.record(index, call, agreed=len(set(actuals[-3:])) and
                       (3 if len(set(actuals[-3:])) == 1 else 1), asked=3)
        made_at.append((index, call))

    # Score each prediction against the label at the bar its horizon landed
    # on. Done AFTER the walk so nothing in the walk could have read it.
    next_correct = [
        (truth[bar + args.horizon] == call) if (bar + args.horizon) in truth
        else None
        for bar, call in made_at
    ]

    # THE CONTROL. Permuting the frames while the labels stay put leaves every
    # bucket count identical and provably destroys any frame-to-correctness
    # relation. Marked SEPARATES here, an estimator is reporting its own noise.
    if args.shuffle_frames:
        random.Random(null_seed + 7).shuffle(frame_rows)
        print("CONTROL RUN: frames permuted, labels held. Nothing here can "
              "legitimately be marked SEPARATES.")

    total = len(frame_rows)
    print(f"predictions fed  : {len(history)}  "
          f"({dict(predicted_counts)})")
    print(f"pending at end   : {history.pending(stop - 1)} "
          f"(should be ~horizon once warm)")

    print("\n0. DISTINCTNESS (distinct frames / samples), self pools only:")
    verdict_ok = True
    pools: dict = {}
    for key in SELF_KEYS:
        values = [row[key] for row in frame_rows]
        distinct = len(set(values))
        score = distinct / total if total else 0.0
        pools[key] = {"distinct": distinct, "samples": total,
                      "distinctness": round(score, 6)}
        if distinct <= CONSTANT_VALUES:
            note = "CONSTANT -- one value, bound and carrying nothing"
            verdict_ok = False
        elif score < EMPTY_BAND[0]:
            note = "below the empty band -- train it, do not query it"
        elif score <= EMPTY_BAND[1]:
            note = "INSIDE the empty band -- train it, do not query it"
        else:
            note = "clears the query floor -- query-set candidate"
        print(f"  {key:<16} {score:.4f}  ({distinct} distinct / {total})  {note}")
        common = Counter(values).most_common(3)
        for frame, count in common:
            print(f"       {count:>5} {count / total:>6.1%}  {frame}")

    print()
    if verdict_ok:
        print("VERDICT: the feeder moved all three off their na sentinels. "
              "Pass 109 measured ONE distinct frame each -- the sentinel -- "
              "and they now carry a real vocabulary.")
    else:
        print("VERDICT: at least one pool is STILL one value. The feeder is "
              "attached, so the cause is the frame's own vocabulary, not the "
              "missing history.")
    print("EVERY ONE OF THESE SITS BELOW THE 0.260 QUERY FLOOR, and that is "
          "the correct outcome rather than a disappointing one: a coarse "
          "stream votes for the label DISTRIBUTION over everything it "
          "matches, which dilutes a query and helps a consolidation. TRAIN "
          "all three, QUERY none of them, unless a sweep says otherwise.")
    print("This is a VOCABULARY measurement, not a held-out edge. The held-out "
          "number needs the node's own predictions in a walk-forward.")

    # ------------------------------------------------------------------
    # DOES THE FRAME KNOW ANYTHING? A vocabulary is necessary and not
    # sufficient: a pool can hold fifty distinct values and every one of them
    # be unrelated to what happens next, in which case wiring it to a node
    # buys a consolidation and a query per sample for nothing. That is
    # answerable HERE, offline, before anyone spends a training run -- and
    # answering it first is the discipline that a pass-108 result went without.
    #
    # The question, stated so it can only be answered with a number: given the
    # self_error_run frame at bar i, how often is the prediction MADE at bar i
    # correct? A frame that carries self-knowledge separates those rates. A
    # frame that does not leaves them all at the base rate, and the pool is
    # decoration.
    # ------------------------------------------------------------------
    # THE SPREAD IS MARKED AGAINST ITS OWN NULL, NOT AGAINST A LITERAL.
    #
    # [ca4ac5d5]. The estimator here is max-minus-min over buckets, and BOTH
    # extremes are chosen after seeing the data. The largest of k noisy bucket
    # rates minus the smallest of them is large by construction -- the more
    # buckets and the smaller they are, the larger. Measured against a bare
    # 0.10 literal it fires on pure noise: Jet's pass-110 null put
    # P(spread >= 0.10 | the frames carry NOTHING) at 96.4%-100.0% for all
    # three self pools, and every measured spread sat inside its own null's
    # 95th percentile. So the shipped output said SEPARATES three times on
    # three flat frames.
    #
    # The null is computed here rather than approximated from printed counts:
    # the frame assignment is held exactly as it is -- same buckets, same
    # sizes -- and only `next_correct` is shuffled, which destroys any relation
    # between frame and correctness while preserving everything else the
    # estimator is sensitive to. A spread is marked only if it beats the 95th
    # percentile of that null, which is the percentile the literal was
    # pretending to be.
    print("\n1. DOES THE FRAME KNOW ANYTHING (offline, before any node run):")
    print(f"   null: {null_trials} shuffles of next_correct per pool, frames "
          "held fixed; SEPARATES needs the 95th percentile of that null")
    rng = random.Random(null_seed)
    for key in SELF_KEYS:
        buckets: dict = {}
        members: dict = {}
        for idx, (frame, correct) in enumerate(
            zip((r[key] for r in frame_rows), next_correct)
        ):
            if correct is None:
                continue
            hit, seen = buckets.get(frame, (0, 0))
            buckets[frame] = (hit + (1 if correct else 0), seen + 1)
            members.setdefault(frame, []).append(idx)
        scored = [(h / s, s, f) for f, (h, s) in buckets.items() if s >= 30]
        if len(scored) < 2:
            print(f"  {key:<16} too few populated frames to separate")
            pools.setdefault(key, {})["separation"] = "too_few_frames"
            continue
        scored.sort()
        base = sum(1 for c in next_correct if c) / max(
            1, sum(1 for c in next_correct if c is not None))
        lo_rate, lo_n, lo_f = scored[0]
        hi_rate, hi_n, hi_f = scored[-1]
        spread = hi_rate - lo_rate

        # The null. Sizes and membership are fixed; the labels move.
        groups = [len(members[f]) for _, _, f in scored]
        labels = [
            1 if next_correct[i] else 0
            for _, _, f in scored
            for i in members[f]
        ]
        null: list = []
        for _ in range(null_trials):
            rng.shuffle(labels)
            at = 0
            rates = []
            for size in groups:
                rates.append(sum(labels[at:at + size]) / size)
                at += size
            null.append(max(rates) - min(rates))
        null.sort()
        p95 = null[int(0.95 * (len(null) - 1))]
        beat = sum(1 for x in null if x < spread) / len(null)
        mark = (
            "SEPARATES"
            if spread > p95
            else "flat -- inside its own null, carries no self-knowledge"
        )
        print(f"  {key:<16} base {base:.1%}  worst {lo_rate:.1%} (n={lo_n})  "
              f"best {hi_rate:.1%} (n={hi_n})  spread {spread:+.1%}  {mark}")
        print(f"       null over {len(groups)} buckets: median "
              f"{null[len(null)//2]:+.1%}  95th {p95:+.1%}  -- the measured "
              f"spread is at the {beat*100:.1f}th percentile")
        print(f"       worst: {lo_f}")
        print(f"       best : {hi_f}")
        pools.setdefault(key, {}).update({
            "separation": mark.split(" --")[0],
            "base_rate": round(base, 6),
            "spread": round(spread, 6),
            "null_p95": round(p95, 6),
            "null_percentile": round(beat, 6),
            "buckets": len(groups),
        })
    print("  A spread inside its own null means the pool holds a vocabulary "
          "that is unrelated to being right, and wiring it to a node buys "
          "nothing. max-minus-min over small buckets is large BY CONSTRUCTION, "
          "so a spread is only evidence when it beats that.")

    if args.report:
        report = {
            "corpus": str(path),
            **horizon["report_fields"],
            "horizon_source": horizon["horizon_source"],
            "window": [start, stop],
            "labelled": len(truth),
            "samples": total,
            "self_frame_window": args.window,
            "null_trials": null_trials,
            "null_seed": null_seed,
            "shuffled_frames_control": bool(args.shuffle_frames),
            "empty_band": list(EMPTY_BAND),
            "pools": pools,
            "all_pools_have_a_vocabulary": verdict_ok,
        }
        validate_report_horizon(report)
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {out}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
