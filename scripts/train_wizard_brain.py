"""Train and honestly evaluate a W1z4rD-node brain on real market history.

The brain ships default-off for a reason: untrained, it churned ~842 losing
trades (0 wins, -$8.37) on noise. Measured again 2026-08-27 it answered every
input -- including fabricated symbols like ZZZZNOTREAL-USDC -- with the same
two answers at confidence 0.0000. It had learned nothing.

Naive scoring hides this. A brain that always says "down" scores ~70% on a
market that fell 70% of the time, which looks like skill and is not. So this
script always reports:

  * a CONTROL on garbage inputs   (a real brain must not answer these the same)
  * the always-up / always-down BASELINE (skill must beat the majority class)
  * a CHRONOLOGICAL split          (train on the past, test on the future only)

Usage:
    python scripts/train_wizard_brain.py --train --epochs 2
    python scripts/train_wizard_brain.py --evaluate
"""

from __future__ import annotations

import argparse
import collections
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.brain_bridge import (  # noqa: E402
    get_bridge,
    features_text,
    outcome_text,
    parse_outcome,
)

DB = "file:storage/trading_cache.db?mode=ro"
HORIZON = 5      # ticks ahead the label looks
LOOKBACK = 5     # ticks back the momentum feature uses


def load_series():
    c = sqlite3.connect(DB, uri=True)
    rows = collections.defaultdict(list)
    for sym, ts, px in c.execute(
        "SELECT symbol, ts, price FROM market_stream WHERE price > 0 ORDER BY ts"
    ):
        rows[sym].append((float(ts), float(px)))
    return rows


def build_samples():
    """(symbol, ts, price, momentum, forward_return) from corroborated ticks."""
    samples = []
    for sym, s in load_series().items():
        if len(s) < LOOKBACK + HORIZON + 2:
            continue
        for i in range(LOOKBACK, len(s) - HORIZON):
            px = s[i][1]
            prev = s[i - LOOKBACK][1]
            fut = s[i + HORIZON][1]
            if px <= 0 or prev <= 0:
                continue
            samples.append((sym, s[i][0], px, (px / prev) - 1.0, (fut / px) - 1.0))
    samples.sort(key=lambda r: r[1])       # chronological
    return samples


def feats(sym, px, mom):
    return features_text(
        side="enter", symbol=sym, chain="base",
        price=px, momentum=mom, confidence=0.5,
    )


def control(bridge):
    """Garbage in: a brain that learned something must not answer these alike."""
    fakes = [
        ("ZZZZNOTREAL-USDC", 1.0, 0.0), ("QQQFAKE-USDC", 1e9, 0.99),
        ("XXXX-USDC", 1e-9, -0.99), ("AAAA-USDC", 42.0, 0.5),
        ("BBBB-USDC", 0.001, -0.5), ("CCCC-USDC", 100.0, 0.0),
        ("DDDD-USDC", 7.77, 0.2), ("EEEE-USDC", 3.3, -0.2),
    ]
    answers, confs = collections.Counter(), []
    for sym, px, mom in fakes:
        ans, conf = bridge.predict_outcome(feats(sym, px, mom))
        answers[ans] += 1
        confs.append(conf)
    return len(answers), (max(confs) if confs else 0.0)


def evaluate(bridge, samples, label):
    answers, confs = collections.Counter(), []
    correct = total = 0
    for sym, _ts, px, mom, fut in samples:
        ans, conf = bridge.predict_outcome(feats(sym, px, mom))
        answers[ans] += 1
        confs.append(conf)
        parsed = parse_outcome(ans)
        if parsed is None:
            continue
        # Score ONLY directional calls. Counting "steady" as a down-call is
        # what inflated an untrained brain to a fake 86%.
        if parsed == "flat":
            continue
        pred_up = parsed in ("win", "win_big")
        total += 1
        if pred_up == (fut > 0):
            correct += 1

    ups = sum(1 for r in samples if r[4] > 0)
    n = len(samples)
    base_up = 100.0 * ups / n if n else 0.0
    baseline = max(base_up, 100.0 - base_up)

    print("\n--- %s (n=%d) ---" % (label, n))
    print("  distinct answers : %d  %s" % (len(answers), dict(list(answers.items())[:5])))
    print("  confidence       : min=%.4f max=%.4f" % (
        min(confs) if confs else 0.0, max(confs) if confs else 0.0))
    if total:
        acc = 100.0 * correct / total
        print("  directional acc  : %d/%d = %.1f%%  (only non-flat calls)" % (correct, total, acc))
        print("  BASELINE (majority class) = %.1f%%" % baseline)
        print("  VERDICT: %s" % (
            "beats baseline by %.1f pts" % (acc - baseline) if acc > baseline
            else "NO SKILL -- at or below the majority class"))
    else:
        print("  directional acc  : no directional calls made")
        print("  BASELINE (majority class) = %.1f%%" % baseline)
        print("  VERDICT: NO SKILL -- brain never commits to a direction")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--split", type=float, default=0.7, help="chronological train fraction")
    ap.add_argument("--limit", type=int, default=4000)
    args = ap.parse_args()

    bridge = get_bridge()
    if not bridge._ensure():
        print("brain node unreachable")
        return 1

    samples = build_samples()
    if len(samples) < 50:
        print("not enough corroborated ticks to train on (%d)" % len(samples))
        return 1
    samples = samples[-args.limit:]
    cut = int(len(samples) * args.split)
    train, test = samples[:cut], samples[cut:]
    print("samples=%d  train=%d  test=%d (chronological, no overlap)"
          % (len(samples), len(train), len(test)))

    n_ans, max_conf = control(bridge)
    print("\nCONTROL (garbage inputs): distinct answers=%d max_conf=%.4f" % (n_ans, max_conf))
    if n_ans <= 2 and max_conf == 0.0:
        print("  -> brain is currently UNTRAINED (garbage and real data look alike)")

    if args.evaluate and not args.train:
        evaluate(bridge, test, "EVAL (held-out future)")
        return 0

    if args.train:
        print("\ntraining %d epoch(s) on %d samples..." % (args.epochs, len(train)))
        ok = fail = 0
        for epoch in range(args.epochs):
            for i, (sym, _ts, px, mom, fut) in enumerate(train):
                if bridge.train_binding(feats(sym, px, mom), outcome_text(fut * 100.0)):
                    ok += 1
                else:
                    fail += 1
                if i and i % 500 == 0:
                    print("   epoch %d: %d/%d bound (%d failed)" % (epoch + 1, ok, len(train), fail))
        print("  bound=%d failed=%d" % (ok, fail))

        n_ans2, max_conf2 = control(bridge)
        print("\nCONTROL after training: distinct answers=%d max_conf=%.4f" % (n_ans2, max_conf2))
        evaluate(bridge, test, "EVAL after training (held-out future)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
