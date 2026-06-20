"""brain_history_supervisor — walk the 3-year historical OHLCV corpus
and push (features_at_t → outcome_at_t+horizon) bindings into the brain.

Why: brain_feeder currently pushes raw OHLCV bytes into POOL_TEXT only.
The substrate forms perception concepts but has no labelled supervision
for price-pattern → return mapping. This loop provides that supervision
by treating historical bars as ground truth: features at bar t, actual
forward return at bar t+h, formatted via brain_bridge.features_text +
outcome_text exactly as the live bot would emit them.

After enough samples (target ~5000+ per pair), the brain's binding pool
will hold tentative bindings (features-fingerprint -> outcome-fingerprint)
that the live trading loop can query at decision time.

Usage:
    python scripts/brain_history_supervisor.py \
        --chain base \
        --pairs WETH-USDC,WETH-USDT,WBTC-USDC,LINK-USDC \
        --horizon 1 \
        --start 0 --limit 5000 \
        --rate-limit-sec 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.brain_bridge import (
    BrainBridge, features_text, outcome_text, POOL_TEXT, POOL_ACTION,
)

HIST_ROOT = ROOT / "data" / "historical_ohlcv"


def _find_pair_file(chain: str, pair: str) -> Optional[Path]:
    """data/historical_ohlcv/{chain}/9000_{pair}.json — falls back to
    other prefixes if the 9000_ file is absent."""
    chain_dir = HIST_ROOT / chain.lower()
    if not chain_dir.exists():
        return None
    primary = chain_dir / f"9000_{pair}.json"
    if primary.exists():
        return primary
    for p in chain_dir.glob(f"*_{pair}.json"):
        return p
    return None


def _bar_features(bars: List[Dict], i: int, pair: str, chain: str) -> str:
    """Same encoder as the live bot — keeps train/test/live consistent."""
    bar = bars[i]
    prev_close = bars[i - 1]["close"] if i > 0 else bar["open"]
    close = bar["close"]
    momentum = (close - prev_close) / prev_close if prev_close > 0 else 0.0
    vol = (bar["high"] - bar["low"]) / bar["open"] if bar["open"] > 0 else 0.0
    return features_text(
        side="buy", symbol=pair, chain=chain,
        price=close, momentum=momentum,
        spread_bps=vol * 10000.0, confidence=None,
    )


def _bar_outcome(bars: List[Dict], i: int, horizon: int) -> Optional[str]:
    if i + horizon >= len(bars):
        return None
    p0 = bars[i]["close"]
    p1 = bars[i + horizon]["close"]
    pnl_pct = (p1 - p0) / p0 if p0 > 0 else 0.0
    return outcome_text(pnl_pct)


def supervise_pair(
    bridge: BrainBridge,
    chain: str,
    pair: str,
    *,
    horizon: int,
    start: int,
    limit: int,
    rate_limit_sec: float,
    status_every: int = 100,
) -> Dict:
    path = _find_pair_file(chain, pair)
    if path is None:
        print(f"[skip] {chain}/{pair}: no corpus file")
        return {"pushed": 0, "failed": 0}
    print(f"[load] {path.name}")
    bars = json.loads(path.read_text())
    print(f"  bars={len(bars)} horizon={horizon} start={start} limit={limit}")

    end = min(start + limit, len(bars) - horizon - 1)
    pushed = 0
    failed = 0
    outcome_counts: Dict[str, int] = {}
    t0 = time.time()

    for i in range(max(start, 1), end):
        feats = _bar_features(bars, i, pair, chain)
        out = _bar_outcome(bars, i, horizon)
        if not out:
            continue
        ok = bridge.observe_outcome(feats, out)
        if ok:
            pushed += 1
            outcome_counts[out] = outcome_counts.get(out, 0) + 1
        else:
            failed += 1
        if (pushed + failed) % status_every == 0:
            elapsed = time.time() - t0
            rate = (pushed + failed) / max(elapsed, 0.001)
            print(f"  {pair}: {pushed + failed}/{end-start} "
                  f"pushed={pushed} fail={failed} {rate:.2f}/s "
                  f"eta={int((end-start - (pushed+failed))/max(rate, 0.001))}s")
        if rate_limit_sec > 0:
            time.sleep(rate_limit_sec)

    elapsed = time.time() - t0
    print(f"[done] {pair}: pushed={pushed} failed={failed} "
          f"elapsed={elapsed:.0f}s outcomes={outcome_counts}")
    return {
        "pushed": pushed, "failed": failed,
        "outcome_counts": outcome_counts,
        "elapsed_sec": elapsed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Push historical OHLCV → brain as supervised bindings")
    ap.add_argument("--chain", default="base")
    ap.add_argument("--pairs", default="WETH-USDC",
                    help="comma-separated pair list")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--start", type=int, default=0,
                    help="bar index to start from (after warmup)")
    ap.add_argument("--limit", type=int, default=5000,
                    help="max bars per pair")
    ap.add_argument("--rate-limit-sec", type=float, default=0.0)
    ap.add_argument("--brain", default="http://127.0.0.1:8090")
    ap.add_argument("--report", default="data/brain_history_report.json")
    args = ap.parse_args()

    bridge = BrainBridge(endpoint=args.brain)
    if not bridge._ensure():
        print("ERROR: brain endpoint unreachable", file=sys.stderr)
        return 1

    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    summaries: Dict[str, Dict] = {}
    overall_t0 = time.time()
    for pair in pairs:
        summaries[pair] = supervise_pair(
            bridge, args.chain, pair,
            horizon=args.horizon, start=args.start,
            limit=args.limit, rate_limit_sec=args.rate_limit_sec,
        )

    overall_elapsed = time.time() - overall_t0
    out = {
        "chain": args.chain, "pairs": pairs, "horizon": args.horizon,
        "limit_per_pair": args.limit, "summaries": summaries,
        "total_elapsed_sec": overall_elapsed, "ran_at": time.time(),
    }
    out_path = ROOT / args.report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nTotal pushed: "
          f"{sum(s['pushed'] for s in summaries.values())} in {overall_elapsed:.0f}s")
    print(f"Report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
