"""Settle [bb32b5c7]: "the probability calibrator pushed the head DOWN in 854 of
855 cycles (99.9%), one-directional: justify it or remove it".

READ-ONLY. Touches no db, no network, no model artifact. It is pure arithmetic
over the Platt fit that trading/bot.py:6215 records, because that is all the
question needs -- the claim is about a TRANSFORM, and a transform can be settled
analytically over its whole domain rather than sampled over 855 cycles.

WHAT THIS PRINTS, and the three things it establishes:

  1. The one-way push is REAL and is not 854 of 855 -- it is 20001 of 20001,
     i.e. ALL of them. With a negative Platt intercept and a scale near 1,
     sigmoid(S*l + O) < sigmoid(l) for every probability in the clipped domain.
     So "99.9% one-directional" understates it, and the 1 exception in the
     operator's sample cannot be a property of a fixed calibrator: it has to
     come from the EMA at trading/pipeline.py:1239 moving the parameters, or
     from a cycle where use_calibration was False.

  2. That push lives in ``direction_prob_calibrated``, which is WRITE-ONLY on
     the production path -- trading/bot.py:6238 writes it and nothing outside
     tests/ reads it. The DECISION variable is ``direction_prob``, which since
     113d862 (2026-09-07) is the RE-CENTRED column, and the re-centred column
     is exactly symmetric: 10000 points move up, 10000 move down, one is fixed.

  3. Removing the calibrator is a PROVABLE no-op at the 0.5 neutral test and a
     0.16pp shift at the 0.58 entry gate, because all three columns are
     strictly monotone images of each other. The same monotonicity means
     AUC(raw) == AUC(calibrated) == AUC(centred) exactly, so the calibrator
     cannot be what makes or breaks the ordering skill that [821de434] and
     [2a503d95] are arguing about.

Run:  python scripts/calibrator_direction_census.py
"""

from __future__ import annotations

import math

# The fit trading/bot.py:6215 records, measured 2026-09-07 over the 1324
# production evaluations where graph_confidence was exactly 1.0:
#     logit_out = 0.9806 * logit_in - 2.0784   (median |resid| 0.048)
SCALE = 0.9806
OFFSET = -2.0784

# bot.py:6196 clips the raw probability to this band before taking the logit,
# so this -- not (0, 1) -- is the domain the transform is ever asked about.
EPS = 1e-6
GRID = 20001


def columns(raw: float) -> tuple[float, float]:
    """Return (calibrated, centred) exactly as trading/bot.py:6196-6244 does."""
    logit = math.log(raw / (1.0 - raw))
    scaled = max(-30.0, min(30.0, logit * SCALE + OFFSET))
    calibrated = 1.0 / (1.0 + math.exp(-scaled))
    neutral_logit = max(-30.0, min(30.0, OFFSET))
    centred = 1.0 / (1.0 + math.exp(-(scaled - neutral_logit)))
    return calibrated, centred


def raw_needed_for(threshold: float) -> float:
    """The raw probability that clears ``centred >= threshold``."""
    return 1.0 / (1.0 + math.exp(-(math.log(threshold / (1.0 - threshold)) / SCALE)))


def main() -> None:
    grid = [EPS + i * (1.0 - 2.0 * EPS) / (GRID - 1) for i in range(GRID)]
    cal, cen = zip(*(columns(r) for r in grid))

    print("Platt fit in force (trading/bot.py:6215): logit_out = %.4f*logit_in %+.4f"
          % (SCALE, OFFSET))
    print("calibrator's own no-information point sigmoid(offset) = %.6f"
          % (1.0 / (1.0 + math.exp(-OFFSET))))
    print()

    print("%-9s %-11s %-11s %-11s" % ("RAW", "CALIBRATED", "CENTRED", "cal - raw"))
    for r in (0.02, 0.05, 0.1237, 0.25, 0.40, 0.4999, 0.5226, 0.70, 0.906):
        c, k = columns(r)
        print("%-9.4f %-11.6f %-11.6f %+-11.6f" % (r, c, k, c - r))
    print()

    down_cal = sum(1 for r, c in zip(grid, cal) if c < r)
    down_cen = sum(1 for r, k in zip(grid, cen) if k < r)
    up_cen = sum(1 for r, k in zip(grid, cen) if k > r)

    print("(1) THE ONE-WAY PUSH, over %d points spanning the clipped domain" % GRID)
    print("    calibrated < raw : %d / %d  (%.1f%%)" % (down_cal, GRID, 100.0 * down_cal / GRID))
    print("    => not 854 of 855. It is ALL of them, and it is a theorem, not a sample:")
    print("       offset < 0 with scale ~ 1 sends every point down. That is what a")
    print("       BASE RATE below 50% looks like on a logit scale, so the push is")
    print("       JUSTIFIED as calibration -- it is only a bias if the base rate is wrong.")
    print()

    print("(2) THE DECISION VARIABLE IS THE CENTRED COLUMN, and it is symmetric")
    print("    centred < raw : %d / %d" % (down_cen, GRID))
    print("    centred > raw : %d / %d" % (up_cen, GRID))
    print("    => the money path sees no one-directional push at all.")
    print()

    print("(3) WHAT REMOVING THE CALIBRATOR WOULD DO")
    mono_cal = all(cal[i] < cal[i + 1] for i in range(GRID - 1))
    mono_cen = all(cen[i] < cen[i + 1] for i in range(GRID - 1))
    print("    calibrated strictly increasing in raw : %s" % mono_cal)
    print("    centred    strictly increasing in raw : %s" % mono_cen)
    print("    => AUC(raw) == AUC(calibrated) == AUC(centred) EXACTLY (AUC is rank-only),")
    print("       so the calibrator cannot create or destroy the head's ordering skill.")
    for t in (0.50, 0.58, 0.60):
        need = raw_needed_for(t)
        print("    gate centred >= %.2f  <=>  raw >= %.6f   (shift %+.4f pp)"
              % (t, need, (need - t) * 100.0))
    print("    => at the 0.5 neutral test removal admits EXACTLY ZERO extra entries.")


if __name__ == "__main__":
    main()
