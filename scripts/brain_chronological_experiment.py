"""Small, read-only OHLCV(+optional news) Wizard brain experiment.

The corpus and news inputs are never modified.  Only the requested JSON report is
written.  Run this against a dedicated/fresh Wizard instance when strict state
isolation from prior training is required.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.brain_bridge import (
    OUTCOME_TOKENS as LABEL_TOKEN,
    BrainBridge,
    outcome_text,
    parse_outcome,
)

LABELS = ("win_big", "win", "flat", "loss", "loss_big")
# Byte-disjoint outcome tokens live in trading.brain_bridge.OUTCOME_TOKENS
# (single source of truth shared with the live trading path).
TOKEN_LABEL = {v: k for k, v in LABEL_TOKEN.items()}


def timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        value = float(value)
        return value / 1000.0 if value > 10_000_000_000 else value
    text = str(value).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)\
        .timestamp() if "+" not in text[10:] else datetime.fromisoformat(text).timestamp()


def load_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        obj = json.loads(raw)
        rows = obj if isinstance(obj, list) else obj.get("articles", obj.get("rows", []))
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a JSON list, JSONL, or articles/rows list")
    return rows


def load_news(path: Path | None) -> list[tuple[float, str]]:
    if path is None:
        return []
    result: list[tuple[float, str]] = []
    for row in load_records(path):
        raw_time = next((row.get(k) for k in
                         ("published_at", "published", "timestamp", "date", "created_at")
                         if row.get(k) is not None), None)
        if raw_time is None:
            continue
        title = str(row.get("title") or row.get("headline") or row.get("text") or "")
        result.append((timestamp(raw_time), title))
    return sorted(result)


def label(text: str | None) -> str | None:
    value = (text or "").lower()
    for token, canonical in TOKEN_LABEL.items():
        if token in value:
            return canonical
    for candidate in LABELS:  # legacy answers from older brains
        if candidate in value:
            return candidate
    return None


def actual_label(bars: list[dict[str, Any]], index: int, horizon: int) -> str:
    p0, p1 = float(bars[index]["close"]), float(bars[index + horizon]["close"])
    # outcome_text now emits disjoint tokens; parse back to canonical label.
    return parse_outcome(outcome_text((p1 - p0) / p0 if p0 else 0.0)) or "flat"


def _bucket(value: float, edges: Iterable[float]) -> int:
    return bisect.bisect_right(list(edges), value)


_RET_EDGES = (-.08, -.05, -.03, -.02, -.012, -.008, -.005, -.003, -.002, -.001,
              .001, .002, .003, .005, .008, .012, .02, .03, .05, .08)


def features(bars: list[dict[str, Any]], index: int, symbol: str, chain: str,
             news: list[tuple[float, str]], news_lookback: float) -> str:
    """Causal features: every referenced bar/news item is timestamp <= bar t.

    Enriched (2026-07): seven return windows at 20-bucket resolution,
    intra-bar range, volume ratio, 24-bar realized volatility, position in
    the 7-day range, and hour/day-of-week time tokens.  On the WETH-USDC
    hourly corpus this makes every training situation unique (recall
    ceiling 1.0) while staying compositional — situations share categorical
    atoms, so partial overlap still generalizes.
    """
    close = float(bars[index]["close"])
    parts = [f"market {symbol.lower()} {chain.lower()}"]
    for window in (1, 2, 3, 6, 12, 24, 48):
        prior = float(bars[max(0, index - window)]["close"])
        ret = (close - prior) / prior if prior else 0.0
        parts.append(f"r{window}=b{_bucket(ret, _RET_EDGES)}")
    bar = bars[index]
    rng = (float(bar["high"]) - float(bar["low"])) / max(float(bar["open"]), 1e-12)
    parts.append(f"range=b{_bucket(rng, (.0005, .001, .002, .003, .0045, .006, .008, .01, .015, .02, .03, .05))}")
    # DEX OHLCV exports carry net_volume/buy_volume rather than volume.
    volumes = [abs(float(x.get("volume") or x.get("net_volume") or 0.0))
               for x in bars[max(0, index - 23):index + 1]]
    median = statistics.median(volumes) if volumes else 0.0
    ratio = volumes[-1] / median if median else 1.0
    parts.append(f"volume=b{_bucket(ratio, (.3, .5, .7, .9, 1.1, 1.4, 2, 3, 5, 10))}")
    rets = []
    for k in range(max(1, index - 23), index + 1):
        p_prev = float(bars[k - 1]["close"])
        if p_prev:
            rets.append((float(bars[k]["close"]) - p_prev) / p_prev)
    vol24 = statistics.pstdev(rets) if len(rets) > 2 else 0.0
    parts.append(f"vol24=b{_bucket(vol24, (.001, .002, .003, .0045, .006, .009, .014, .02))}")
    window_bars = bars[max(0, index - 167):index + 1]
    lo = min(float(x["low"]) for x in window_bars)
    hi = max(float(x["high"]) for x in window_bars)
    pos = (close - lo) / (hi - lo) if hi > lo else 0.5
    parts.append(f"pos7d=b{int(pos * 8)}")
    dt = datetime.fromtimestamp(timestamp(bar["timestamp"]), tz=timezone.utc)
    parts.append(f"hour={dt.hour}")
    parts.append(f"dow={dt.weekday()}")
    if news:
        now = timestamp(bar["timestamp"])
        eligible = [title for ts, title in news if now - news_lookback <= ts <= now]
        words = Counter(re.findall(r"[a-z0-9]{3,}", " ".join(eligible).lower()))
        parts.append(f"news_count=b{_bucket(len(eligible), (0, 1, 3, 10, 30))}")
        parts.extend(f"news={word}" for word, _ in words.most_common(5))
    return " ".join(parts)


def calibration(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        selected = [r for r in rows if lo <= r["confidence"] <= hi and (i == bins - 1 or r["confidence"] < hi)]
        if not selected:
            continue
        conf = statistics.fmean(r["confidence"] for r in selected)
        acc = statistics.fmean(float(r["correct"]) for r in selected)
        ece += len(selected) / max(len(rows), 1) * abs(acc - conf)
        groups.append({"min": lo, "max": hi, "n": len(selected), "mean_confidence": conf, "accuracy": acc})
    return {"ece": ece, "bins": groups}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[round((len(values) - 1) * q)]


def run(bars: list[dict[str, Any]], bridge: BrainBridge, *, train_n: int, test_n: int,
        horizon: int, symbol: str, chain: str, news: list[tuple[float, str]] | None = None,
        news_lookback_hours: float = 24.0, recall_n: int = 0,
        epochs: int = 1, balance: bool = False) -> dict[str, Any]:
    news = news or []
    if horizon < 1 or train_n < 1 or test_n < 1:
        raise ValueError("horizon, train_n, and test_n must be positive")
    if len(bars) < train_n + horizon + test_n + 1:
        raise ValueError("corpus is too short for requested train/gap/test windows")
    lookback = news_lookback_hours * 3600
    # Purge `horizon` bars: no training target may land in the test period.
    train_indices: list[int] = list(range(1, train_n + 1))
    test_start = train_n + horizon + 1
    test_indices = range(test_start, test_start + test_n)
    train_labels = [actual_label(bars, i, horizon) for i in train_indices]
    if balance:
        # Undersample dominant classes so no label's binding mass drowns
        # the others at decode time (unbalanced 8k training collapsed all
        # predictions to the majority attractor). Cap the three common
        # classes at their minimum count, keep every rare *_big instance,
        # and prefer the most recent examples of each class.
        by_label: dict[str, list[int]] = {}
        for i, lab in zip(train_indices, train_labels):
            by_label.setdefault(lab, []).append(i)
        common = [lab for lab in ("win", "loss", "flat") if lab in by_label]
        cap = min(len(by_label[lab]) for lab in common) if common else 0
        selected: list[int] = []
        for lab, idxs in by_label.items():
            selected.extend(idxs if lab.endswith("_big") else idxs[-cap:])
        train_indices = sorted(selected)
        train_labels = [actual_label(bars, i, horizon) for i in train_indices]
    failures = 0
    started = time.perf_counter()
    for _epoch in range(max(1, epochs)):
        for i, target in zip(train_indices, train_labels):
            # Supervised surface: /brain/consolidate binds features→outcome
            # explicitly (observe_outcome's Hebbian co-firing is the live-path
            # alternative, but consolidate is the measurable training contract).
            if not bridge.train_binding(features(bars, i, symbol, chain, news, lookback),
                                        f"outcome {LABEL_TOKEN[target]}"):
                failures += 1
    training_seconds = time.perf_counter() - started
    majority = Counter(train_labels).most_common(1)[0][0]
    # Train-set recall: re-query a sample of the exact trained features.
    # This measures whether the substrate faithfully stored and retrieves
    # what it was shown — the gate for "the fabric itself works" — before
    # any question of generalization to unseen bars.
    recall: dict[str, Any] | None = None
    if recall_n > 0:
        sample = list(train_indices)[:: max(1, len(train_indices) // recall_n)][:recall_n]
        hits = 0
        answered = 0
        for i in sample:
            answer, _conf = bridge.predict_outcome(
                features(bars, i, symbol, chain, news, lookback))
            predicted = label(answer)
            if predicted is not None:
                answered += 1
                if predicted == actual_label(bars, i, horizon):
                    hits += 1
        recall = {
            "sampled": len(sample),
            "answered": answered,
            "coverage": answered / max(len(sample), 1),
            "recall_accuracy": hits / max(answered, 1),
        }
    rows: list[dict[str, Any]] = []
    for i in test_indices:
        actual = actual_label(bars, i, horizon)
        query = features(bars, i, symbol, chain, news, lookback)
        t0 = time.perf_counter()
        # Read-only /brain/predict: test features never enter the
        # learning moment, so held-out evaluation stays leak-free.
        answer, confidence = bridge.predict_outcome(query)
        latency = time.perf_counter() - t0
        predicted = label(answer)
        previous = float(bars[i - 1]["close"])
        momentum = "win" if float(bars[i]["close"]) >= previous else "loss"
        rows.append({"index": i, "timestamp": bars[i]["timestamp"], "actual": actual,
                     "predicted": predicted, "confidence": max(0.0, min(1.0, confidence)),
                     "correct": predicted == actual, "latency_seconds": latency,
                     "majority_correct": majority == actual, "momentum_direction": momentum})
    covered = [r for r in rows if r["predicted"] is not None]
    directional = lambda x: 1 if x in ("win", "win_big") else -1 if x in ("loss", "loss_big") else 0
    comparable = [r for r in covered if directional(r["actual"]) and directional(r["predicted"])]
    actual_directional = [r for r in rows if directional(r["actual"])]
    latencies = [r["latency_seconds"] for r in rows]
    return {
        "split": {"train_n": train_n, "purge_gap_bars": horizon, "test_n": test_n,
                  "horizon_bars": horizon, "test_start_index": test_start},
        "training": {"attempted": len(train_indices), "failures": failures,
                     "seconds": training_seconds, "balanced": balance,
                     "label_counts": dict(Counter(train_labels)),
                     "train_recall": recall},
        "brain": {"coverage": len(covered) / len(rows), "covered_n": len(covered),
                  "exact_accuracy_covered": sum(r["correct"] for r in covered) / max(len(covered), 1),
                  "exact_accuracy_all": sum(r["correct"] for r in rows) / len(rows),
                  "directional_accuracy": sum(directional(r["actual"]) == directional(r["predicted"]) for r in comparable) / max(len(comparable), 1),
                  "directional_n": len(comparable), "calibration": calibration(covered)},
        "baselines": {"train_majority_label": majority,
                      "train_majority_exact_accuracy": sum(r["majority_correct"] for r in rows) / len(rows),
                      "one_bar_momentum_directional_accuracy": sum(directional(r["actual"]) == directional(r["momentum_direction"]) for r in actual_directional) / max(len(actual_directional), 1),
                      "directional_n": len(actual_directional)},
        "latency_seconds": {"mean": statistics.fmean(latencies), "p50": percentile(latencies, .5),
                            "p95": percentile(latencies, .95), "max": max(latencies)},
        "news": {"enabled": bool(news), "records": len(news), "lookback_hours": news_lookback_hours},
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--news", type=Path, help="optional JSON/JSONL timestamped news export")
    parser.add_argument("--brain", default="http://127.0.0.1:8090")
    parser.add_argument("--symbol", default="WETH-USDC")
    parser.add_argument("--chain", default="base")
    parser.add_argument("--train-n", type=int, default=500)
    parser.add_argument("--test-n", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--news-lookback-hours", type=float, default=24.0)
    parser.add_argument("--recall-n", type=int, default=0,
                        help="re-query N sampled train features to measure substrate recall")
    parser.add_argument("--epochs", type=int, default=1,
                        help="consolidation passes over the training window")
    parser.add_argument("--balance", action="store_true",
                        help="undersample dominant labels so binding mass stays comparable")
    parser.add_argument("--start-index", type=int, default=0,
                        help="skip the first N bars (e.g. to align windows with news coverage)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bars = sorted(load_records(args.corpus), key=lambda row: timestamp(row["timestamp"]))
    if args.start_index > 0:
        bars = bars[args.start_index:]
    bridge = BrainBridge(endpoint=args.brain)
    if not bridge._ensure():
        print(f"Wizard endpoint unreachable: {args.brain}", file=sys.stderr)
        return 2
    report = run(bars, bridge, train_n=args.train_n, test_n=args.test_n,
                 horizon=args.horizon, symbol=args.symbol, chain=args.chain,
                 news=load_news(args.news), news_lookback_hours=args.news_lookback_hours,
                 recall_n=args.recall_n, epochs=args.epochs, balance=args.balance)
    report["inputs"] = {"corpus": str(args.corpus.resolve()), "news": str(args.news.resolve()) if args.news else None,
                        "brain": args.brain, "symbol": args.symbol, "chain": args.chain}
    report["ran_at"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("training", "brain", "baselines", "latency_seconds")}, indent=2))
    print(f"Report: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
