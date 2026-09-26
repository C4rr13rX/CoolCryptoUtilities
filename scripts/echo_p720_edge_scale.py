"""Echo p720: how far BELOW zero is money_button's conservative_edge?

Kite p719 measured `conservative_edge` as identically 0.0000% over 5842
endpoints in three bar-gap tiers. That number is CLAMPED -- money_button.py:316
is `max(0.0, projected*decay - 0.5*volatility)` -- so the table cannot show
whether the term misses by a hair or by a mile, and that distinction decides
whether the fix is a constant or a redesign.

This recomputes the SAME quantity UNCLAMPED, and separately reports the two
halves so the scale mismatch is visible as a ratio rather than a verdict.

Read-only. Touches no production file and no other agent's claim.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOLD_MIN = 12.0
DECAY = 0.5
LOOKBACK_SEC = 120.0 * 60.0
REQUIRED_EDGE = 0.0286  # 2.86% floor as built today


def log_slope_per_min(ts: np.ndarray, prices: np.ndarray) -> float:
    if prices.size < 4:
        return 0.0
    rel_min = (ts - ts[-1]) / 60.0
    if np.allclose(rel_min, rel_min[0]):
        return 0.0
    safe = np.clip(prices, 1e-12, None)
    try:
        with np.errstate(all="ignore"):
            slope, _ = np.polyfit(rel_min, np.log(safe), 1)
        return float(slope) if math.isfinite(slope) else 0.0
    except Exception:
        return 0.0


def ret_over(ts: np.ndarray, prices: np.ndarray, seconds: float) -> float:
    if prices.size < 2:
        return 0.0
    cutoff = ts[-1] - seconds
    idx = np.searchsorted(ts, cutoff, side="left")
    if idx >= prices.size - 1:
        return 0.0
    base = float(prices[idx])
    if base <= 0:
        return 0.0
    return float(prices[-1]) / base - 1.0


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    rows = raw if isinstance(raw, list) else (raw.get("candles") or raw.get("bars") or [])
    ts, px = [], []
    for r in rows:
        if isinstance(r, dict):
            t = r.get("timestamp") or r.get("ts") or r.get("time") or r.get("t")
            c = r.get("close") or r.get("c") or r.get("price")
        elif isinstance(r, (list, tuple)) and len(r) >= 5:
            t, c = r[0], r[4]
        else:
            continue
        try:
            t = float(t)
            c = float(c)
        except Exception:
            continue
        if t > 1e11:
            t /= 1000.0
        if c > 0:
            ts.append(t)
            px.append(c)
    if len(ts) < 8:
        return None
    a = np.asarray(ts, dtype=np.float64)
    p = np.asarray(px, dtype=np.float64)
    o = np.argsort(a)
    return a[o], p[o]


def main():
    files = sorted(glob.glob("data/historical_ohlcv/base/*.json"))
    tiers = {"dense <=6min": [], "mid 6-20min": [], "hourly >20min": []}
    n_total = 0
    for path in files:
        got = load(path)
        if got is None:
            continue
        ts, px = got
        gaps = np.diff(ts)
        gaps = gaps[gaps > 0]
        if gaps.size < 4:
            continue
        med_gap = float(np.median(gaps))
        gap_min = med_gap / 60.0
        tier = (
            "dense <=6min" if gap_min <= 6.0
            else "mid 6-20min" if gap_min <= 20.0
            else "hourly >20min"
        )
        # Walk endpoints exactly as the strategy sees them: a trailing window.
        step = max(1, px.size // 40)
        for end in range(8, px.size, step):
            w_ts, w_px = ts[: end + 1], px[: end + 1]
            cut = w_ts[-1] - LOOKBACK_SEC
            keep = w_ts >= cut
            w_ts, w_px = w_ts[keep], w_px[keep]
            if w_px.size < 4:
                continue
            unit = max(5.0 * 60.0, 2.0 * med_gap)
            if unit * 6.0 > LOOKBACK_SEC:
                continue
            r5 = ret_over(w_ts, w_px, unit)
            r10 = ret_over(w_ts, w_px, unit * 2.0)
            r30 = ret_over(w_ts, w_px, unit * 6.0)
            slope = log_slope_per_min(w_ts, w_px)
            projected = float(np.expm1(slope * HOLD_MIN))
            steps = np.diff(np.log(np.clip(w_px, 1e-12, None)))
            vol = float(np.std(steps)) if steps.size else 0.0
            raw = projected * DECAY - 0.5 * vol
            momentum_ok = (r5 > 0.0 and r10 > 0.0 and r30 > 0.0
                           and r5 >= r10 * 0.35 and slope > 0.0 and projected > 0.0)
            tiers[tier].append((raw, projected * DECAY, 0.5 * vol, momentum_ok))
            n_total += 1

    print(f"files={len(files)}  endpoints={n_total}\n")
    hdr = (f"{'bar tier':<16}{'n':>7}{'n_mom_ok':>10}"
           f"{'UNCLAMPED max':>15}{'median':>11}"
           f"{'med signal':>12}{'med penalty':>13}{'ratio':>8}{'pass 2.86%':>12}{'pass 0%':>9}")
    print(hdr)
    print("-" * len(hdr))
    for tier, rows in tiers.items():
        if not rows:
            continue
        raws = np.array([r[0] for r in rows])
        sig = np.array([r[1] for r in rows])
        pen = np.array([r[2] for r in rows])
        mom = np.array([r[3] for r in rows])
        clamped = np.maximum(0.0, raws)
        # Restrict the ratio to endpoints that actually reach line 316 in prod.
        m = mom if mom.any() else np.ones_like(mom, dtype=bool)
        med_sig = float(np.median(sig[m]))
        med_pen = float(np.median(pen[m]))
        ratio = (med_pen / med_sig) if med_sig > 0 else float("inf")
        print(f"{tier:<16}{len(rows):>7}{int(mom.sum()):>10}"
              f"{raws.max()*100:>14.4f}%{np.median(raws)*100:>10.4f}%"
              f"{med_sig*100:>11.4f}%{med_pen*100:>12.4f}%{ratio:>8.1f}"
              f"{(clamped >= REQUIRED_EDGE).mean()*100:>11.2f}%"
              f"{(raws > 0).mean()*100:>8.2f}%")

    allraw = np.array([r[0] for t in tiers.values() for r in t])
    allmom = np.array([r[3] for t in tiers.values() for r in t])
    print(f"\nALL: n={allraw.size}  unclamped max={allraw.max()*100:.4f}%  "
          f"positive={int((allraw > 0).sum())}  "
          f"momentum-eligible={int(allmom.sum())}")
    if allmom.any():
        sub = allraw[allmom]
        print(f"AMONG MOMENTUM-ELIGIBLE ONLY: n={sub.size} "
              f"max={sub.max()*100:.4f}%  positive={int((sub > 0).sum())} "
              f"({(sub > 0).mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
