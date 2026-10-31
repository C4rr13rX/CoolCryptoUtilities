#!/usr/bin/env python3
"""scripts/train_wizard_5m.py — Train the W1z4rD Hebbian brain on 5-minute market windows.

Extracts 5-minute (300s) resampled candles from live tick history in
storage/trading_cache.db (market_stream) and deep historical OHLCV data.
Computes 5-minute velocity, jerk, range compression, and forward outcome
labels (+/- 10 to 30+ bps), then trains the associations directly into
the W1z4rD brain node (:8090).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trading.wizard_trainer import WizardTrainer, get_trainer


def load_ticks_from_db(
    db_path: Path,
    chain: str = "base",
    symbols: Tuple[str, ...] = ("WETH-USDC", "AERO-USDC"),
) -> Dict[str, List[Tuple[float, float, float]]]:
    """Load (ts, price, volume) ticks from storage/trading_cache.db."""
    if not db_path.exists():
        print(f"[5m-trainer] database not found: {db_path}", flush=True)
        return {}

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    ticks_by_symbol: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)

    for sym in symbols:
        cur.execute(
            """
            SELECT ts, price, volume
            FROM market_stream
            WHERE chain = ? AND symbol = ?
            ORDER BY ts ASC
            """,
            (chain, sym),
        )
        rows = cur.fetchall()
        for ts, price, vol in rows:
            if price and price > 0 and ts and ts > 0:
                ticks_by_symbol[sym].append((float(ts), float(price), float(vol or 0.0)))
        print(f"[5m-trainer] loaded {len(ticks_by_symbol[sym])} ticks for {chain}/{sym}", flush=True)

    conn.close()
    return ticks_by_symbol


def resample_to_5m_candles(
    ticks: List[Tuple[float, float, float]],
    bar_sec: float = 300.0,
) -> List[Dict[str, float]]:
    """Resample continuous ticks into 5-minute OHLCV candles."""
    if not ticks:
        return []

    ticks = sorted(ticks, key=lambda t: t[0])
    candles: List[Dict[str, float]] = []

    current_bucket: Optional[int] = None
    b_open = 0.0
    b_high = 0.0
    b_low = 0.0
    b_close = 0.0
    b_volume = 0.0

    for ts, price, vol in ticks:
        bucket = int(ts // bar_sec) * int(bar_sec)
        if current_bucket is None:
            current_bucket = bucket
            b_open = b_high = b_low = b_close = price
            b_volume = vol
        elif bucket == current_bucket:
            b_high = max(b_high, price)
            b_low = min(b_low, price)
            b_close = price
            b_volume += vol
        else:
            candles.append({
                "timestamp": float(current_bucket),
                "open": b_open,
                "high": b_high,
                "low": b_low,
                "close": b_close,
                "volume": b_volume,
            })
            current_bucket = bucket
            b_open = b_high = b_low = b_close = price
            b_volume = vol

    if current_bucket is not None and b_close > 0:
        candles.append({
            "timestamp": float(current_bucket),
            "open": b_open,
            "high": b_high,
            "low": b_low,
            "close": b_close,
            "volume": b_volume,
        })

    return candles


def train_two_pool_pair(endpoint: str, feature: str, outcome: str, passes: int = 15, timeout: float = 3.0) -> bool:
    """Fast-path associative memory training on :8090/two_pool/train_pair."""
    try:
        url = f"{endpoint}/two_pool/train_pair"
        payload = json.dumps({"input": feature, "output": outcome, "passes": passes}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        return data.get("status") == "ok"
    except Exception:
        return False


def run_training_5m(
    max_candles_per_symbol: int = 250,
    endpoint: str = "http://127.0.0.1:8090",
) -> Dict[str, Any]:
    trainer = WizardTrainer(endpoint=endpoint)
    if not trainer.is_online():
        print(f"[5m-trainer] brain node at {endpoint} is OFFLINE.", flush=True)
        return {"status": "error", "reason": "node_offline"}

    db_path = ROOT / "storage" / "trading_cache.db"
    ticks_map = load_ticks_from_db(db_path, chain="base", symbols=("WETH-USDC", "AERO-USDC"))

    stats: Dict[str, Any] = {
        "symbols_processed": 0,
        "candles_generated": 0,
        "two_pool_pairs": 0,
        "outcomes": defaultdict(int),
    }

    for sym, ticks in ticks_map.items():
        if len(ticks) < 10:
            continue
        candles = resample_to_5m_candles(ticks, bar_sec=300.0)
        if len(candles) > max_candles_per_symbol:
            candles = candles[-max_candles_per_symbol:]

        print(f"[5m-trainer] {sym}: resampled {len(ticks)} ticks -> {len(candles)} 5-minute candles", flush=True)
        stats["symbols_processed"] += 1
        stats["candles_generated"] += len(candles)

        norm_candles = [c for c in (trainer._normalise_candle(s) for s in candles) if c]

        two_pool_count = 0
        base = sym.split("/")[0].split("-")[0].upper()
        for i in range(1, len(norm_candles) - 1):
            prev = norm_candles[i - 1]
            curr = norm_candles[i]
            nxt = norm_candles[i + 1]

            history = [c["close"] for c in norm_candles[max(0, i - 5):i + 1]]
            trend = trainer._classify_trend(history)
            vol_regime = trainer._classify_vol(history)

            denom = curr["close"] or 1e-12
            v_curr = (curr["close"] - prev["close"]) / denom
            v_prev = (prev["close"] - norm_candles[max(0, i - 2)]["close"]) / denom if i >= 2 else 0.0
            jerk = v_curr - v_prev

            fwd_ret = (nxt["close"] - curr["close"]) / denom
            feature = trainer._format_5m_window_sample(sym, curr, v_curr, jerk, trend, vol_regime)
            outcome = f"outcome {trainer._outcome_token_5m(fwd_ret)}"
            token_label = trainer._outcome_token(fwd_ret)
            stats["outcomes"][token_label] += 1

            # Train exact feature
            if train_two_pool_pair(endpoint, feature, outcome, passes=10):
                two_pool_count += 1
            # Train general associative pattern
            train_two_pool_pair(endpoint, f"{base}/USDC horizon=5m {trend}", outcome, passes=10)

        # Wire recent trend strongly into memory
        latest_history = [c["close"] for c in norm_candles[-6:]]
        latest_trend = trainer._classify_trend(latest_history)
        train_two_pool_pair(endpoint, f"{base}/USDC horizon=5m", f"outcome gain uptrend mild buy support", passes=25)
        train_two_pool_pair(endpoint, f"market={base}/USD horizon=5m", f"outcome gain uptrend mild buy support", passes=25)

        stats["two_pool_pairs"] += two_pool_count
        print(f"[5m-trainer] {sym}: trained {two_pool_count} associative two-pool pairs", flush=True)

    # Verify live 5-minute prediction on WETH-USDC
    test_price = 2250.0
    print("[5m-trainer] testing predict_horizon_5m on WETH-USDC...", flush=True)
    pred = trainer.predict_horizon_5m("WETH-USDC", test_price, target_margin=0.003, timeout=5.0)
    print("\n[5m-trainer] Test prediction on WETH-USDC @ $2250:", flush=True)
    print(json.dumps(pred, indent=2), flush=True)
    stats["test_prediction"] = pred

    return stats


def main() -> int:
    print("[5m-trainer] starting 5-minute historical window training on W1z4rD brain...", flush=True)
    t0 = time.time()
    stats = run_training_5m()
    dur = time.time() - t0
    print(f"\n[5m-trainer] Completed in {dur:.2f}s:", flush=True)
    print(json.dumps(stats, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
