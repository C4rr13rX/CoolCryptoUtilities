"""Does ABSTAINING ON DISAGREEMENT pay? One fabric, one held-out run, scored twice.

The asymmetry this exists to test, both numbers already measured on this repo:

  * CONFIDENCE is worthless as a correctness gate -- +0.030 correlation on
    train, -0.002 held out. Cove's pass-107 run read median confidence 0.970
    on a run scoring 0.2850 exact against a 0.3025 majority baseline. A number
    that is 0.97 when you are wrong is not a signal.
  * AGREEMENT between QUERY SETS is strong -- 99.4% correct when the sets are
    unanimous against 73.3% when they split.

Agreement is computed client-side inside ``OmenBrain.predict(consensus=True)``
and thrown away after each call. This script keeps it and asks the only
question that matters: **does gating on it change PER-TRADE NET?**

THE HONEST TRAP, and it is why every number below is printed with its n:
selecting the agreeing subset SHRINKS the sample. A higher accuracy on 40 of
400 answers is not an edge, it is a smaller sample. Abstention only pays if
the answers it KEEPS clear the round trip -- ROUND_TRIP_COST, charged on
every buy.

Both arms are scored from ONE pass over the held-out set. Each sample is
fired once per query-set member, and the member sets are exactly the ones
``omen_brain`` would use under ``consensus=True``: the measured discriminating
set first (it is the primary answer), then ``CONSENSUS_QUERIES``. So the two
arms are not two runs 34 minutes apart -- they are the SAME queries against
the SAME fabric, and the only difference is which answers are kept.

  ALL   -- every admitted answer from the primary query set. This is exactly
           what a plain non-consensus run returns.
  AGREE -- the subset where every member decoded the SAME label.

THE QUERY PATH MUST FIRE, and this script proves it rather than assuming it.
If the members never disagree, either the extra collections are redundant or
the query path is not firing them -- the pass-108 failure. ``disagreement_rate``
is reported first for that reason, and a rate of exactly 0.0 fails the run
with exit 3 instead of reporting a meaningless "agreement pays" result.

Scoreboard order is the operator's, not exact-accuracy-first:
  (a) per-trade net on TROUGH omens against buy-every-bar in the same window
  (b) TROUGH PRECISION -- of the bars called trough, what share paid the round
      trip. That is "accurate about buying low" as a number.
  (c) CREST PRECISION against forward returns. The live lane is long-only so a
      crest is an abstention and the sell-high half is measured NOWHERE else;
      scored here as an EXIT signal, never by shorting.
  (d) exact accuracy, REPORTED BUT DEMOTED -- a sanity check that the fabric
      learned anything, never the thing optimised.

Usage
-----
  OMEN_BRAIN_ENDPOINT=127.0.0.1:8093 python -X utf8 \
      scripts/omen_agreement_census.py \
      --corpus data/historical_ohlcv/base/0004_AERO-USDC.json \
      --train 1500 --test 200 --horizon-minutes 720 --seed 7 --label DOWN

The horizon is asked in MINUTES and converted with this corpus's own measured
cadence; --horizon still takes bars as an explicit override so an old run
reproduces exactly, and the report records both units either way.

Never point it at 127.0.0.1:8090 -- that is production's fabric.
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
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trading.omen_brain import (  # noqa: E402
    CONSENSUS_QUERIES, OMEN_CREST, OMEN_TROUGH, ROUND_TRIP_COST, OmenBrain,
    collection_distinctness, discriminating_collections,
)
from omen_experiment import (  # noqa: E402
    WindowError, add_horizon_args, backpressure_probe, balance, bar_seconds,
    build_samples, load_bars, plan_windows, settle_horizon,
    validate_report_horizon, window_regime,
)


def _pct(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    at = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {"min": round(ordered[0], 4), "median": round(at(0.5), 4),
            "p90": round(at(0.9), 4), "max": round(ordered[-1], 4)}


def score(rows: Sequence[Dict[str, Any]], name: str) -> Dict[str, Any]:
    """Score one arm. Every count here is reported beside its own n.

    ``rows`` are the PRIMARY query set's answers for the samples this arm
    keeps. Money comes first; exact accuracy is last and is a control.
    """
    admitted = [r for r in rows if r["verdict"] == "admitted"]
    exact = sum(1 for r in admitted if r["label"] == r["truth"])

    # (a) money. Long-only: only a buy-low omen opens a position.
    trades = [r["forward"] - ROUND_TRIP_COST
              for r in rows if r["actionable"] and r["action"] == "buy"]
    total = sum(trades)

    # (b) trough precision -- of the bars CALLED trough, what share paid the
    # round trip. Not "was the label right", but "did the trade pay", which is
    # the thing the money cares about.
    troughs = [r for r in admitted if r["label"] == OMEN_TROUGH]
    trough_paid = sum(1 for r in troughs if r["forward"] - ROUND_TRIP_COST > 0)

    # (c) crest precision -- a crest that correctly calls a fall is worth
    # money as an EXIT on a held position. Scored against the forward return,
    # never by shorting.
    crests = [r for r in admitted if r["label"] == OMEN_CREST]
    crest_right = sum(1 for r in crests if r["forward"] < 0)

    return {
        "arm": name,
        "n_samples": len(rows),
        "n_admitted": len(admitted),
        # (a)
        "buy_omens": len(trades),
        "buy_net_total": round(total, 6),
        "buy_net_per_trade": (total / len(trades)) if trades else 0.0,
        "buy_hit_rate": (sum(1 for t in trades if t > 0) / len(trades)) if trades else 0.0,
        # (b)
        "trough_called": len(troughs),
        "trough_precision_paid": (trough_paid / len(troughs)) if troughs else 0.0,
        # (c)
        "crest_called": len(crests),
        "crest_precision_fell": (crest_right / len(crests)) if crests else 0.0,
        # (d) demoted control
        "exact_accuracy": (exact / len(admitted)) if admitted else 0.0,
        "predicted_mix": dict(Counter(
            r["label"] if r["verdict"] == "admitted" else "__hold__" for r in rows)),
        "confidence": _pct([r["confidence"] for r in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--test", type=int, default=200)
    add_horizon_args(parser)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chain", default="base")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--skip-train", action="store_true",
                        help="re-measure a fabric already on the node; the "
                             "split is deterministic in --seed")
    parser.add_argument("--train-end", type=int, default=None)
    parser.add_argument("--test-end", type=int, default=None)
    parser.add_argument("--label", default="",
                        help="a note for the report filename, e.g. UP or DOWN. "
                             "The regime is MEASURED, not taken from this.")
    parser.add_argument("--report-dir", default="data/brain_experiments")
    parser.add_argument("--ignore-backpressure", action="store_true")
    args = parser.parse_args()

    endpoint = args.endpoint or os.getenv("OMEN_BRAIN_ENDPOINT") or "127.0.0.1:8093"
    if "8090" in endpoint:
        print("REFUSING: 127.0.0.1:8090 is PRODUCTION's fabric. Use another node.")
        return 2

    rng = random.Random(args.seed)
    path = Path(args.corpus)
    symbol = path.stem.split("_", 1)[-1]
    bars = load_bars(path)
    cadence = bar_seconds(bars)
    print(f"corpus {path.name}: {len(bars)} bars, {cadence}s cadence, {symbol}")
    print(f"endpoint {endpoint}; round trip {ROUND_TRIP_COST:.4%}")

    # Settled once, in both units, before anything is planned or sampled: every
    # args.horizon below this line is bars, and the report carries the minutes.
    try:
        horizon = settle_horizon(args, cadence)
    except ValueError as exc:
        print(f"cannot resolve horizon: {exc}")
        return 2

    try:
        plan = plan_windows(len(bars), args.train, args.test, args.horizon,
                            train_end=args.train_end, test_end=args.test_end)
    except WindowError as exc:
        print(f"cannot plan windows: {exc}")
        return 2

    train_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                  plan["train_start"], plan["train_stop"])
    test_samples = build_samples(bars, symbol, args.chain, args.horizon,
                                 plan["test_start"], plan["test_stop"])
    regime = window_regime(bars, plan["test_start"], plan["test_stop"], args.horizon)
    print(f"train [{plan['train_start']}, {plan['train_stop']}) -> "
          f"{len(train_samples)}; test [{plan['test_start']}, "
          f"{plan['test_stop']}) -> {len(test_samples)}")
    print(f"HELD-OUT WINDOW IS {regime['regime']}: up-rate {regime['up_rate']:.1%}, "
          f"mean forward {regime['mean_forward']:+.4%} over {regime['bars']} bars. "
          f"ONE WINDOW IS NOT EVIDENCE.")

    brain = OmenBrain(endpoint=endpoint)
    # supports_multi is the reachability test that matters: a node answering
    # /health but lacking /brain/predict/multi returns a transport_error hold
    # for every sample, which would score as 100% abstention rather than as a
    # dead node.
    if not brain.supports_multi():
        print(f"node {endpoint} lacks /brain/predict/multi -- nothing to measure.")
        return 2
    gate = backpressure_probe(endpoint)
    if gate.get("reachable") and gate.get("backpressure") and not args.ignore_backpressure:
        print("node is under backpressure; free memory and re-run.")
        return 4

    balanced = balance(train_samples, rng)
    distinctness = collection_distinctness([s["frames"] for s in balanced])
    measured = tuple(discriminating_collections([s["frames"] for s in balanced]))
    print("distinctness : " + " ".join(
        f"{k}={distinctness[k]:.3f}" for k in sorted(distinctness, key=lambda k: -distinctness[k])))

    # The member sets are EXACTLY what omen_brain fires under consensus=True:
    # the measured set first (it is the primary answer, so ALL and AGREE agree
    # on WHAT was predicted and differ only on whether it is kept), then the
    # consensus queries that are not already it.
    #
    # Deduplicated by SET, not by tuple. ``discriminating_collections`` returns
    # the measured set in distinctness order, so ('geometry','temporal','cross')
    # is not equal to ('temporal','geometry','cross') as a tuple while firing
    # exactly the same query -- omen_brain's own consensus construction compares
    # tuples and so fires its primary member twice. A member that is the primary
    # in a different order cannot disagree with it, so counting it inflates the
    # agreement rate with a query that agrees by construction.
    seen = {frozenset(measured)}
    members: List[tuple] = [measured]
    for candidate in CONSENSUS_QUERIES:
        if frozenset(candidate) not in seen:
            seen.add(frozenset(candidate))
            members.append(tuple(candidate))
    print(f"members ({len(members)}, deduped by set): primary={measured}")
    for extra in members[1:]:
        print(f"              {extra}")
    # THE NEGATIVE CONTROL, fired on every sample: the primary query set again.
    # A vs A must move NOTHING. If it does, the node is not deterministic and
    # every disagreement below is run-to-run variance rather than a query-set
    # difference -- which is the only reading under which the AGREE arm means
    # anything at all.
    print(f"  control     : {measured} refired (A vs A must never disagree)")

    if not args.skip_train:
        started = time.time()
        for count, sample in enumerate(balanced, 1):
            brain.train(sample["frames"], sample["label"], sample["regime"])
            if count % 250 == 0:
                print(f"  trained {count}/{len(balanced)} "
                      f"({count / max(1e-9, time.time() - started):.1f}/s, "
                      f"{brain.failed_pairs} failed)")
        print(f"trained {brain.trained_pairs} pairs, {brain.failed_pairs} failed, "
              f"in {(time.time() - started) / 60:.1f} min")
    else:
        print("skipping training -- measuring the fabric already on the node")

    # ONE pass over the held-out set. Every member fires on every sample, so
    # the two arms are the same queries against the same fabric.
    primary_rows: List[Dict[str, Any]] = []
    agree_rows: List[Dict[str, Any]] = []
    disagreements = 0
    control_disagreements = 0
    member_only_labels: List[List[Optional[str]]] = []
    started = time.time()
    for index, sample in enumerate(test_samples, 1):
        labels: List[Optional[str]] = []
        row: Optional[Dict[str, Any]] = None
        for position, member in enumerate(list(members) + [measured]):
            omen = brain.predict(
                sample["frames"], symbol=symbol, chain=args.chain,
                as_of_ts=sample["ts"], price=sample["price"],
                horizon_bars=args.horizon, bar_seconds=cadence,
                regime=sample["regime"], query_collections=list(member),
                consensus=False)
            decoded = omen.omen if omen.verdict == "admitted" else None
            labels.append(decoded)
            if position == 0:
                row = {"ts": sample["ts"], "truth": sample["label"],
                       "forward": sample["forward"], "label": omen.omen,
                       "verdict": omen.verdict, "confidence": omen.confidence,
                       "actionable": bool(omen.is_actionable),
                       "action": omen.action}
        # The last fire is the A-vs-A control, not a member. It is scored
        # separately and never counted as agreement.
        control = labels.pop()
        member_only_labels.append(labels)
        assert row is not None
        primary_rows.append(row)
        if control != labels[0]:
            control_disagreements += 1
        # Unanimity, defined exactly as omen_brain defines it: the primary
        # decoded something and every member decoded the same thing.
        head = labels[0]
        if head is not None and all(m == head for m in labels):
            agree_rows.append(row)
        else:
            disagreements += 1
        if index % 50 == 0:
            print(f"  scored {index}/{len(test_samples)} "
                  f"({index / max(1e-9, time.time() - started):.1f}/s, "
                  f"{disagreements} split)")

    disagreement_rate = disagreements / max(1, len(test_samples))
    print(f"\n0. QUERY PATH    : {disagreements}/{len(test_samples)} samples "
          f"split across {len(members)} query sets "
          f"({disagreement_rate:.1%} disagreement)")
    print(f"   A-vs-A control: {control_disagreements}/{len(test_samples)} "
          f"moved (must be 0; anything else is node non-determinism and the "
          f"split rate above is not a query-set effect)")

    all_arm = score(primary_rows, "ALL")
    agree_arm = score(agree_rows, "AGREE")

    truth = Counter(s["label"] for s in test_samples)
    majority = max(truth.values()) / max(1, len(test_samples))
    every_bar = [s["forward"] - ROUND_TRIP_COST for s in test_samples]
    every_bar_per_trade = sum(every_bar) / max(1, len(every_bar))

    for arm in (all_arm, agree_arm):
        print(f"\n{arm['arm']:<6} n={arm['n_samples']} kept, "
              f"{arm['n_admitted']} admitted")
        print(f"  (a) money    : {arm['buy_omens']} buy omens, "
              f"{arm['buy_net_per_trade']:+.4%} per trade, "
              f"total {arm['buy_net_total']:+.4f}, "
              f"{arm['buy_hit_rate']:.1%} paid")
        print(f"  (b) trough   : {arm['trough_called']} called, "
              f"{arm['trough_precision_paid']:.1%} paid the round trip")
        print(f"  (c) crest    : {arm['crest_called']} called, "
              f"{arm['crest_precision_fell']:.1%} fell")
        print(f"  (d) exact    : {arm['exact_accuracy']:.1%} (control, not the target)")

    print(f"\nbaselines      : majority class {majority:.1%}, "
          f"buy-every-bar {every_bar_per_trade:+.4%} per trade over "
          f"{len(every_bar)} bars")

    # THE VERDICT, stated in the script's own output so a reader cannot take a
    # subset accuracy for an edge. Abstention pays only if what it KEEPS
    # clears the round trip AND beats buying every bar.
    delta = agree_arm["buy_net_per_trade"] - all_arm["buy_net_per_trade"]
    kept_pays = (agree_arm["buy_omens"] > 0
                 and agree_arm["buy_net_per_trade"] > 0.0
                 and agree_arm["buy_net_per_trade"] > every_bar_per_trade)
    if agree_arm["buy_omens"] == 0:
        verdict = ("AGREEMENT-GATING PLACES NO TRADES: it abstains on every buy "
                   "omen, so there is no per-trade net to compare. Not an edge.")
    elif kept_pays:
        verdict = (f"AGREEMENT-GATING KEEPS A PAYING SUBSET: "
                   f"{agree_arm['buy_net_per_trade']:+.4%} per trade on "
                   f"{agree_arm['buy_omens']} trades against "
                   f"{all_arm['buy_net_per_trade']:+.4%} on "
                   f"{all_arm['buy_omens']}. ONE WINDOW IS NOT EVIDENCE -- "
                   f"it must hold in the other window too.")
    else:
        verdict = (f"AGREEMENT-GATING DOES NOT PAY HERE: the kept subset runs "
                   f"{agree_arm['buy_net_per_trade']:+.4%} per trade over "
                   f"{agree_arm['buy_omens']} trades, which does not clear the "
                   f"{ROUND_TRIP_COST:.4%} round trip against buy-every-bar at "
                   f"{every_bar_per_trade:+.4%}. A smaller sample is not an edge.")
    print(f"\nVERDICT: {verdict}")

    report = {
        "corpus": str(path), "symbol": symbol, "bars": len(bars),
        **horizon["report_fields"],
        "horizon_source": horizon["horizon_source"],
        "round_trip_cost": ROUND_TRIP_COST,
        "train_window": [plan["train_start"], plan["train_stop"]],
        "test_window": [plan["test_start"], plan["test_stop"]],
        "heldout_regime": regime["regime"],
        "heldout_window_up_rate": regime["up_rate"],
        "heldout_window_mean_forward": regime["mean_forward"],
        "heldout_ts_range": [regime.get("ts_start"), regime.get("ts_stop")],
        "endpoint": endpoint,
        "skipped_training": bool(args.skip_train),
        "trained_pairs": brain.trained_pairs, "failed_pairs": brain.failed_pairs,
        "collection_distinctness": distinctness,
        "measured_query_collections": list(measured),
        "member_query_sets": [list(m) for m in members],
        "query_path_disagreements": disagreements,
        "query_path_disagreement_rate": disagreement_rate,
        "control_a_vs_a_disagreements": control_disagreements,
        "arms": {"all": all_arm, "agree": agree_arm},
        "agree_minus_all_per_trade": delta,
        "baseline_majority": majority,
        "baseline_every_bar_per_trade": every_bar_per_trade,
        "true_mix": dict(truth),
        "verdict": verdict,
    }
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.label}" if args.label else ""
    # Checked BEFORE the write: a report that cannot say what horizon it
    # measured must never reach data/brain_experiments/, because the next
    # reader cannot tell it from a run at a different cadence.
    validate_report_horizon(report)
    out = (report_dir /
           f"agreement-{symbol}-h{horizon['horizon_minutes']:.0f}m"
           f"{horizon['horizon_bars']}b-{regime['regime']}{tag}-{stamp}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report -> {out}")

    # A run where no two query sets ever disagree measured NOTHING about
    # agreement: either the extra collections are redundant or the query path
    # is not firing them. That is the pass-108 failure and it must not be
    # reported as "agreement pays".
    if disagreements == 0:
        print("\nFAIL: the query sets NEVER disagreed. Agreement was not "
              "measured -- prove the query path fires before trusting any "
              "number from it (scripts/omen_query_path_probe.py).")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
