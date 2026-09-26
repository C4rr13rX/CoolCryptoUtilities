"""Echo p720: does money_button's projected edge survive contact with the future?

The last surviving lever on `[d7c71792]` was mine: the volatility penalty at
money_button.py:316 is horizon-BLIND -- `projected = expm1(slope*hold)` grows
with the hold while `0.5*std(per-bar log returns)` does not -- so a longer hold
should lift signal past cost for free.

PRE-REGISTERED COMPLEMENT: the slope is trailing and mean-reverts, so a longer
hold inflates `projected` without raising REALISED return, and the extra
candidates are fictitious.

SEPARATING TEST (this file): for the same endpoints, measure the ACTUAL forward
return over each horizon and compare how it scales against how the projection
scales. If projected scales ~linearly in the hold and realised does not, the
lever is fake.

RESULT: the lever is fake, and the finding is larger than the lever -- momentum-
eligible endpoints have NEGATIVE mean forward return at every horizon tested,
and the subset clearing the cost floor is WORSE than the average. Forward return
here is gross, with no cost subtracted, so the cost floor cannot be the cause.

Read-only. Touches no production file.
"""
from __future__ import annotations

import glob
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join(_HERE, "echo_p720_edge_scale.py")

_g: dict = {"__name__": "notmain", "__file__": _BASE}
with open(_BASE, "r", encoding="utf-8") as _fh:
    exec(_fh.read().replace('if __name__ == "__main__":\n    main()', ""), _g)

load = _g["load"]
ret_over = _g["ret_over"]
log_slope_per_min = _g["log_slope_per_min"]

DECAY = 0.5
LOOKBACK_SEC = 120.0 * 60.0
HORIZONS = [12.0, 15.0, 30.0, 60.0, 120.0]
MEASURED_FLOOR = 0.004582  # Dune p719 all-in realised cost at the $6 live clip


def collect():
    """Momentum-eligible endpoints only -- the ones production actually reaches."""
    rec = []
    for path in sorted(glob.glob("data/historical_ohlcv/base/*.json")):
        got = load(path)
        if got is None:
            continue
        ts, px = got
        gaps = np.diff(ts)
        gaps = gaps[gaps > 0]
        if gaps.size < 4:
            continue
        med_gap = float(np.median(gaps))
        step = max(1, px.size // 40)
        for end in range(8, px.size, step):
            w_ts, w_px = ts[: end + 1], px[: end + 1]
            keep = w_ts >= w_ts[-1] - LOOKBACK_SEC
            w_ts, w_px = w_ts[keep], w_px[keep]
            if w_px.size < 4:
                continue
            unit = max(5.0 * 60.0, 2.0 * med_gap)
            if unit * 6.0 > LOOKBACK_SEC:
                continue  # money_button.py:234 window_exceeds_lookback
            r5 = ret_over(w_ts, w_px, unit)
            r10 = ret_over(w_ts, w_px, unit * 2.0)
            r30 = ret_over(w_ts, w_px, unit * 6.0)
            slope = log_slope_per_min(w_ts, w_px)
            if not (r5 > 0 and r10 > 0 and r30 > 0 and r5 >= r10 * 0.35 and slope > 0):
                continue
            steps = np.diff(np.log(np.clip(w_px, 1e-12, None)))
            vol = float(np.std(steps)) if steps.size else 0.0
            p0, t0 = float(px[end]), float(ts[end])
            fwd = {}
            for H in HORIZONS:
                j = np.searchsorted(ts, t0 + H * 60.0, side="left")
                fwd[H] = (float(px[j]) / p0 - 1.0) if j < px.size else None
            rec.append((slope, vol, fwd))
    return rec


def main():
    rec = collect()
    print(f"momentum-eligible endpoints with forward data: n={len(rec)}\n")
    hdr = (f"{'hold':>6}{'proj*decay':>13}{'penalty':>10}{'n pass floor':>14}"
           f"{'share':>8}{'REALISED fwd':>14}{'fwd ALL':>10}{'fwd>0':>8}")
    print(hdr)
    print("-" * len(hdr))
    scales = []
    for H in HORIZONS:
        passing, allf, sigs, pens = [], [], [], []
        for slope, vol, fwd in rec:
            f = fwd[H]
            if f is None:
                continue
            sig = float(np.expm1(slope * H)) * DECAY
            pen = 0.5 * vol
            sigs.append(sig)
            pens.append(pen)
            allf.append(f)
            if max(0.0, sig - pen) >= MEASURED_FLOOR:
                passing.append(f)
        allf = np.asarray(allf)
        n = len(passing)
        fm = float(np.mean(passing)) * 100 if n else float("nan")
        up = (np.asarray(passing) > 0).mean() * 100 if n else float("nan")
        print(f"{H:>6.0f}{np.median(sigs)*100:>12.4f}%{np.median(pens)*100:>9.4f}%"
              f"{n:>14}{n/max(1,allf.size)*100:>7.2f}%{fm:>13.4f}%"
              f"{float(np.mean(allf))*100:>9.4f}%{up:>7.1f}%")
        scales.append((H, float(np.median(sigs)), float(np.mean(allf)), allf.size))

    print("\nKILL TEST -- projected vs realised scaling in the hold:")
    b_sig, b_real = scales[0][1], scales[0][2]
    for H, sig, real, n in scales:
        print(f"  hold {H:>5.0f} min: projected x{sig/b_sig:>6.2f}   "
              f"realised x{(real/b_real) if b_real else float('nan'):>6.2f}   "
              f"(realised mean {real*100:+.4f}%, n={n})")
    print("\nVERDICT: projection scales ~10x to a 120-min hold while realised mean "
          "return\nstays NEGATIVE and gets worse. The horizon lever is fictitious, and "
          "the\nmomentum-eligible set has no gross forward edge to charge a cost against.")


if __name__ == "__main__":
    main()
