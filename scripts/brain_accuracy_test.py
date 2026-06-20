"""brain_accuracy_test — walk a historical OHLCV corpus, train the brain on
(features_at_t -> realised_outcome_at_t+h) pairs, then evaluate held-out
accuracy.

Test design:
  1. Load the 26k-bar 3-year hourly WETH-USDC corpus from
     data/historical_ohlcv/base/9000_WETH-USDC.json.
  2. Split 80/20 train/test by time (last 20% held out — no look-ahead).
  3. Train phase: for each train bar at index t, build features_text from
     (price, computed momentum, computed volatility), compute outcome
     bucket from realised return over `horizon` bars forward, and POST
     both to brain via brain_bridge.observe_outcome.  Substrate forms
     cross-pool bindings on each pair.
  4. Test phase: for each test bar, query brain_bridge.query_confidence
     against the same features; compare brain's decoded answer +
     confidence to the actual outcome bucket.
  5. Report: per-bucket accuracy, directional accuracy (up vs down),
     confidence calibration, confusion matrix.

This is the substrate's "did you actually learn anything" test. The
brain has had OHLCV streamed into POOL_TEXT for weeks by brain_feeder,
but it has never seen labelled outcomes — so this is also the first
real supervised training pass the brain bridge will deliver.

Usage:
    python scripts/brain_accuracy_test.py \
      --train-n 5000 --test-n 1000 --horizon 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.brain_bridge import (
    BrainBridge, features_text, outcome_text,
)

CORPUS_PATH = ROOT / "data" / "historical_ohlcv" / "base" / "9000_WETH-USDC.json"


def _load_corpus() -> List[Dict]:
    with open(CORPUS_PATH, "r") as fh:
        return json.load(fh)


def _features_for_bar(bars: List[Dict], i: int) -> str:
    """Stable canonical features at bar index i. Same encoding as the
    live bot: bucketed price + momentum + volatility."""
    bar = bars[i]
    prev_close = bars[i - 1]["close"] if i > 0 else bar["open"]
    close = bar["close"]
    momentum = (close - prev_close) / prev_close if prev_close > 0 else 0.0
    vol = (bar["high"] - bar["low"]) / bar["open"] if bar["open"] > 0 else 0.0
    return features_text(
        side="buy",  # direction doesn't matter for the encoder; both buy/sell
                    # observations share the same atom set on (p, m, v)
        symbol="WETH-USDC",
        chain="base",
        price=close,
        momentum=momentum,
        # reuse `spread_bps` channel as volatility bucket
        spread_bps=vol * 10000.0,  # vol fraction -> bps
        confidence=None,
    )


def _outcome_for_bar(bars: List[Dict], i: int, horizon: int) -> str:
    """Realised return from bar i to bar i+horizon, bucketed via
    brain_bridge.outcome_text (5 levels)."""
    if i + horizon >= len(bars):
        return ""
    p0 = bars[i]["close"]
    p1 = bars[i + horizon]["close"]
    pnl_pct = (p1 - p0) / p0 if p0 > 0 else 0.0
    return outcome_text(pnl_pct)


def _outcome_label(text: str) -> str:
    """Strip the 'outcome ' prefix that outcome_text adds, so we can
    compare against brain's decoded answer."""
    if text.startswith("outcome "):
        return text[len("outcome "):]
    return text


def _parse_answer_bucket(answer: str) -> str:
    """Brain's decoded answer for POOL_ACTION may come back as something
    like 'outcome win_big' or partial. Extract the bucket if present;
    otherwise return the raw string for the confusion matrix."""
    if not answer:
        return "(none)"
    a = answer.strip().lower()
    for label in ("win_big", "loss_big", "win", "flat", "loss"):
        if label in a:
            return label
    return f"(other:{a[:20]})"


def train_phase(bars: List[Dict], train_n: int, horizon: int,
                bridge: BrainBridge, status_every: int = 500) -> Dict:
    """Push the first `train_n` (features, outcome) pairs to the brain."""
    t0 = time.time()
    pushed = 0
    failures = 0
    outcome_counts: Dict[str, int] = {}
    last_status = t0
    for i in range(1, min(train_n + 1, len(bars) - horizon - 1)):
        feats = _features_for_bar(bars, i)
        out = _outcome_for_bar(bars, i, horizon)
        if not out:
            continue
        ok = bridge.observe_outcome(feats, out)
        if ok:
            pushed += 1
            outcome_counts[out] = outcome_counts.get(out, 0) + 1
        else:
            failures += 1
        if (pushed + failures) % status_every == 0:
            now = time.time()
            rate = (pushed + failures) / (now - t0)
            print(f"  trained {pushed + failures}/{train_n}  "
                  f"({rate:.1f} obs/s, fails={failures})")
            last_status = now
    elapsed = time.time() - t0
    print(f"\nTRAIN: pushed={pushed} fails={failures} "
          f"elapsed={elapsed:.1f}s rate={pushed/elapsed:.1f}/s")
    print(f"  outcome distribution: {outcome_counts}")
    return {
        "pushed": pushed, "fails": failures,
        "outcome_counts": outcome_counts, "elapsed": elapsed,
    }


def test_phase(bars: List[Dict], train_n: int, test_n: int, horizon: int,
               bridge: BrainBridge) -> Dict:
    """For each test bar, query brain confidence and compare to actual."""
    t0 = time.time()
    rows: List[Dict] = []
    confusion: Dict[Tuple[str, str], int] = {}
    correct = 0
    directional_correct = 0
    directional_total = 0
    confidence_sum = 0.0
    confidence_correct_sum = 0.0
    confidence_wrong_sum = 0.0

    start = train_n + 1
    end = min(start + test_n, len(bars) - horizon - 1)
    queries = 0

    for i in range(start, end):
        feats = _features_for_bar(bars, i)
        actual = _outcome_label(_outcome_for_bar(bars, i, horizon))
        if not actual:
            continue
        answer, conf = bridge.query_confidence(feats)
        pred = _parse_answer_bucket(answer)
        confidence_sum += conf
        if pred == actual:
            correct += 1
            confidence_correct_sum += conf
        else:
            confidence_wrong_sum += conf

        # Directional accuracy: was the brain's bucket on the same side
        # of zero as the actual outcome?
        def _is_up(label: str) -> int:
            if label in ("win_big", "win"): return 1
            if label in ("loss_big", "loss"): return -1
            return 0
        a_dir = _is_up(actual)
        p_dir = _is_up(pred)
        if a_dir != 0 and p_dir != 0:
            directional_total += 1
            if a_dir == p_dir:
                directional_correct += 1

        confusion[(actual, pred)] = confusion.get((actual, pred), 0) + 1
        rows.append({
            "i": i, "features": feats[:60], "actual": actual,
            "predicted": pred, "confidence": conf,
        })
        queries += 1
        if queries % 200 == 0:
            print(f"  tested {queries}/{test_n}  "
                  f"acc={correct/queries:.3f} dir={directional_correct/max(directional_total,1):.3f}")

    elapsed = time.time() - t0
    n = max(queries, 1)
    acc = correct / n
    dir_acc = directional_correct / max(directional_total, 1)
    mean_conf = confidence_sum / n
    mean_conf_right = confidence_correct_sum / max(correct, 1)
    mean_conf_wrong = confidence_wrong_sum / max(n - correct, 1)

    print(f"\nTEST: queries={queries} elapsed={elapsed:.1f}s")
    print(f"  exact-bucket accuracy: {acc:.3f}  ({correct}/{n})")
    print(f"  directional accuracy:  {dir_acc:.3f}  "
          f"({directional_correct}/{directional_total})")
    print(f"  mean confidence:       {mean_conf:.3f}")
    print(f"  mean conf when right:  {mean_conf_right:.3f}")
    print(f"  mean conf when wrong:  {mean_conf_wrong:.3f}")

    # 5x5 confusion matrix (actual rows, predicted cols)
    buckets = ["win_big", "win", "flat", "loss", "loss_big"]
    print("\nConfusion (actual v / predicted ->):")
    header = "  actual\\pred  " + "".join(f"{b:>10s}" for b in buckets) + "      other"
    print(header)
    for a in buckets:
        row = f"  {a:11s}  " + "".join(
            f"{confusion.get((a, p), 0):>10d}" for p in buckets
        )
        others = sum(c for (act, pred), c in confusion.items()
                     if act == a and pred not in buckets)
        row += f"  {others:>10d}"
        print(row)

    return {
        "queries": queries, "correct": correct,
        "exact_acc": acc, "directional_acc": dir_acc,
        "directional_n": directional_total,
        "mean_confidence": mean_conf,
        "mean_conf_right": mean_conf_right,
        "mean_conf_wrong": mean_conf_wrong,
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Brain accuracy test on historical OHLCV")
    ap.add_argument("--train-n", type=int, default=2000,
                    help="Bars to push into brain as training")
    ap.add_argument("--test-n", type=int, default=500,
                    help="Held-out bars to evaluate prediction accuracy on")
    ap.add_argument("--horizon", type=int, default=1,
                    help="Forward bars for outcome (1 = next bar)")
    ap.add_argument("--brain", default="http://127.0.0.1:8090",
                    help="Brain HTTP endpoint")
    ap.add_argument("--out", default="data/brain_accuracy_report.json",
                    help="Where to write the JSON report")
    args = ap.parse_args()

    print(f"Loading corpus from {CORPUS_PATH}...")
    bars = _load_corpus()
    print(f"  loaded {len(bars)} bars, "
          f"span {bars[0]['timestamp']} -> {bars[-1]['timestamp']}")

    bridge = BrainBridge(endpoint=args.brain)
    if not bridge._ensure():
        print("ERROR: brain endpoint unreachable", file=sys.stderr)
        return 1

    print(f"\n=== TRAIN PHASE ===")
    train_stats = train_phase(bars, args.train_n, args.horizon, bridge)

    print(f"\n=== TEST PHASE ===")
    test_stats = test_phase(bars, args.train_n, args.test_n, args.horizon, bridge)

    report = {
        "corpus": str(CORPUS_PATH),
        "bars_total": len(bars),
        "train_n": args.train_n,
        "test_n": args.test_n,
        "horizon": args.horizon,
        "train": train_stats,
        "test": test_stats,
        "ran_at": time.time(),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
