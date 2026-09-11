"""Prove that changing the QUERY SET moves a held-out prediction.

WHY THIS SCRIPT EXISTS. In pass 108 an association experiment measured a
with-relations arm against a flat arm and got BYTE-FOR-BYTE identical output
over 180 held-out predictions. That looked like "the relation pools are
redundant". It was not. ``scripts/omen_experiment.py`` computed the measured
query set, PRINTED it, and then passed ``None`` into ``predict``, so
``omen_brain.predict`` fell back to the hard-coded ``PREDICT_COLLECTIONS``.
Both arms fired the same three streams against fabrics whose extra pools were
bound and never read. The experiment reported one query set and fired another.

That bug is fixed (2026-09-10). This script is the ACCEPTANCE CRITERION for
the fix, and the one every future association experiment must clear before its
number is trusted: *a query-set change must move at least one held-out
prediction.* If it moves none, the pool is not being read, and whatever the
arm reports is a measurement of nothing.

It is deliberately NOT a skill measurement. It says nothing about whether the
brain is right -- only whether the query path is live. A prediction that
changes may well change from one wrong answer to another; that is still proof
the stream was read. Skill is ``omen_experiment.py``'s job.

METHOD. Train ONE fabric, then predict the SAME held-out samples twice --
once under query set A, once under query set B -- and diff. Two reads of one
fabric, so nothing here is confounded by run-to-run training variance (the
same fabric and samples have given 89.2% and 93.6% thirty-four minutes apart;
that variance is why the two arms must share a fabric).

Usage -- prove the relation pools 12/13/14 are read at all:

    set OMEN_RELATION_COLLECTIONS=1
    python -X utf8 scripts/omen_query_path_probe.py \
        --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
        --train 600 --test 120 \
        --query-a temporal,geometry,cross \
        --query-b temporal,geometry,cross,rel_move_vol,rel_shape_flow,rel_trend_noise

A negative control is built in: ``--query-a`` against itself must move ZERO
predictions. If an identical query set moves predictions, the node is
non-deterministic and NO diff from this script means anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.omen_experiment import (  # noqa: E402
    add_horizon_args,
    bar_seconds,
    build_samples,
    load_bars,
    plan_windows,
    settle_horizon,
    validate_report_horizon,
)
from trading.omen_brain import COLLECTIONS, OmenBrain  # noqa: E402


def _predict_all(brain, samples, names, cadence, horizon, symbol, chain):
    """Fire one query set over every held-out sample. Returns labels."""
    out = []
    for sample in samples:
        omen = brain.predict(
            sample["frames"],
            symbol=symbol,
            chain=chain,
            as_of_ts=sample["ts"],
            price=sample["price"],
            horizon_bars=horizon,
            bar_seconds=cadence,
            regime=sample["regime"],
            query_collections=names,
        )
        support = getattr(omen, "support", None) or {}
        out.append({
            "index": sample["index"],
            "verdict": getattr(omen, "verdict", None),
            "raw": support.get("raw_answer"),
            "fired": support.get("collections_fired"),
            "asked": tuple(support.get("query_collections") or ()),
        })
    return out


def _diff(left, right):
    moved = [(a, b) for a, b in zip(left, right)
             if (a["verdict"], a["raw"]) != (b["verdict"], b["raw"])]
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--test", type=int, default=120)
    add_horizon_args(parser)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--query-a", required=True,
                        help="comma-separated collection names")
    parser.add_argument("--query-b", required=True)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-end", type=int, default=None)
    parser.add_argument("--test-end", type=int, default=None)
    parser.add_argument("--report", default=None,
                        help="write the verdict as JSON here")
    args = parser.parse_args()

    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = load_bars(path)
    cadence = bar_seconds(bars)
    try:
        horizon = settle_horizon(args, cadence)
    except ValueError as exc:
        print(f"cannot resolve horizon: {exc}")
        return 2

    known = {c.name for c in COLLECTIONS}
    query_a = tuple(n.strip() for n in args.query_a.split(","))
    query_b = tuple(n.strip() for n in args.query_b.split(","))
    for name, which in ((query_a, "--query-a"), (query_b, "--query-b")):
        unknown = [n for n in name if n not in known]
        if unknown:
            # A silent fallback is exactly the failure this script exists to
            # catch, so refuse rather than measure the default by accident.
            print(f"FAIL: {which} names collections this build does not have: "
                  f"{unknown}. Known: {sorted(known)}. "
                  f"Relation pools need OMEN_RELATION_COLLECTIONS=1.")
            return 2

    plan = plan_windows(len(bars), args.train, args.test, args.horizon,
                        train_end=args.train_end, test_end=args.test_end)
    train_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                  plan["train_start"], plan["train_stop"])
    test_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                 plan["test_start"], plan["test_stop"])
    print(f"corpus {path.name}: {len(bars)} bars, {cadence}s cadence")
    print(f"train [{plan['train_start']}, {plan['train_stop']}) -> "
          f"{len(train_samples)} samples")
    print(f"test  [{plan['test_start']}, {plan['test_stop']}) -> "
          f"{len(test_samples)} samples")
    print(f"collections this build streams: {sorted(known)}")
    print(f"A = {query_a}")
    print(f"B = {query_b}")

    brain = OmenBrain(endpoint=args.endpoint)
    if not brain.supports_multi():
        print("FAIL: node has no /brain/predict/multi -- stale binary or port")
        return 3

    if not args.skip_train:
        print(f"\ntraining {len(train_samples)} samples (ONE epoch)...")
        taught = 0
        for sample in train_samples:
            if brain.train(sample["frames"], sample["label"],
                           sample["regime"]):
                taught += 1
        print(f"taught {taught}/{len(train_samples)}")
        if not taught:
            print("FAIL: the node consolidated NOTHING -- check the RAM floor "
                  "and the identity's pool ids before reading any diff below.")
            return 4

    print("\ncontrol   : A vs A (must move ZERO)")
    a1 = _predict_all(brain, test_samples, query_a, cadence, args.horizon,
                       symbol, args.chain)
    a2 = _predict_all(brain, test_samples, query_a, cadence, args.horizon,
                       symbol, args.chain)
    control = _diff(a1, a2)
    print(f"  moved {len(control)}/{len(a1)}")

    print("\ntreatment : A vs B")
    b1 = _predict_all(brain, test_samples, query_b, cadence, args.horizon,
                       symbol, args.chain)
    treatment = _diff(a1, b1)
    print(f"  moved {len(treatment)}/{len(a1)}")

    fired_a = Counter(r["fired"] for r in a1)
    fired_b = Counter(r["fired"] for r in b1)
    print(f"\nstreams fired per prediction: A {dict(fired_a)}  B {dict(fired_b)}")
    print(f"A verdicts: {dict(Counter(r['verdict'] for r in a1))}")
    print(f"B verdicts: {dict(Counter(r['verdict'] for r in b1))}")

    if control:
        verdict = "INCONCLUSIVE"
        note = ("the node is non-deterministic: an IDENTICAL query set moved "
                f"{len(control)} predictions, so the A-vs-B diff cannot be "
                "attributed to the query set")
    elif treatment:
        verdict = "QUERY PATH LIVE"
        note = (f"changing the query set moved {len(treatment)} of {len(a1)} "
                "held-out predictions, so the extra streams are READ")
    else:
        verdict = "QUERY PATH DEAD"
        note = ("a different query set moved ZERO of "
                f"{len(a1)} held-out predictions. Either the extra streams "
                "are genuinely redundant or they are not being read -- and "
                "until this reads LIVE, no association result is evidence")
    print(f"\nVERDICT: {verdict}\n  {note}")

    if args.report:
        report = {
            "corpus": str(path),
            **horizon["report_fields"],
            "horizon_source": horizon["horizon_source"],
            "train": len(train_samples),
            "test": len(test_samples),
            "query_a": list(query_a),
            "query_b": list(query_b),
            "control_moved": len(control),
            "treatment_moved": len(treatment),
            "streams_fired_a": {str(k): v for k, v in fired_a.items()},
            "streams_fired_b": {str(k): v for k, v in fired_b.items()},
            "verdict": verdict,
            "note": note,
        }
        validate_report_horizon(report)
        Path(args.report).write_text(json.dumps(report, indent=2),
                                     encoding="utf-8")
        print(f"wrote {args.report}")

    return 0 if verdict == "QUERY PATH LIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
