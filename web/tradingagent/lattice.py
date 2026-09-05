"""The model lattice: layers that constrain each other rather than vote.

WHAT THIS IS FOR
----------------
There are six mathematical views of the same trades in this package --
algebra, calculus, probability, statistics, game theory, chaos -- and until
now each produced a number that a human read separately. Averaging them would
be worse than useless: they measure incompatible things, and a mean of a
Hurst exponent and a Kelly fraction is not a quantity.

They are useful to each other in a different way. Each layer answers a
question that CONSTRAINS the next:

    chaos       -> how far ahead is prediction possible at all?
    calculus    -> which way is it moving, and is that accelerating?
    statistics  -> is the effect distinguishable from noise?
    probability -> what is the expected value net of what it costs?
    game theory -> if this is real, why has nobody else taken it?
    algebra     -> does the arithmetic close?

A signal must survive every layer in order. That is not a committee vote --
it is a chain of necessary conditions, and any single failure is fatal
regardless of how strong the others look. A 90% win rate on a horizon longer
than the Lyapunov time is not a strong signal; it is a measurement of the
past that says nothing about the next trade.

THE FEEDBACK LOOPS
------------------
Constraint runs downward, evidence runs back up:

  * chaos sets the horizon; when realised outcomes keep beating the horizon
    it predicted, the horizon estimate is wrong and gets widened.
  * game theory sets the adverse-selection discount; when fills come in
    better than it expected, the discount was too harsh and relaxes.
  * probability sets the size; when realised variance exceeds what it
    assumed, size shrinks.

Each loop is driven by REALISED outcomes, never by another model's opinion.
A loop closed on a model's own forecast is a model agreeing with itself, and
the whole point is to be corrected by the market.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

#: How far a layer's calibration may drift from its default before the
#: adjustment is treated as a bug in the layer rather than a fact about the
#: market. A model that has doubled its own permissiveness is not learning.
MAX_CALIBRATION_DRIFT = 2.0


class LayerVerdict:
    """One layer's answer: pass, plus what it constrains for the next."""

    def __init__(self, layer: str, passed: bool, reason: str,
                 constraints: Optional[Dict[str, Any]] = None,
                 evidence: Optional[Dict[str, Any]] = None) -> None:
        self.layer = layer
        self.passed = bool(passed)
        self.reason = str(reason)
        self.constraints = dict(constraints or {})
        self.evidence = dict(evidence or {})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "passed": self.passed,
            "reason": self.reason,
            "constraints": self.constraints,
            "evidence": self.evidence,
        }


class Calibration:
    """What the layers have learned from being wrong.

    Persisted between passes so the lattice is shaped by outcomes rather than
    re-derived from defaults every time. Every field is a multiplier on a
    default, bounded by MAX_CALIBRATION_DRIFT in both directions -- a layer
    that wants to move ten times further than its default is broken, not
    informed.
    """

    def __init__(self, **kw: float) -> None:
        self.horizon_scale: float = float(kw.get("horizon_scale", 1.0))
        self.adverse_scale: float = float(kw.get("adverse_scale", 1.0))
        self.size_scale: float = float(kw.get("size_scale", 1.0))

    def clamp(self) -> None:
        lo, hi = 1.0 / MAX_CALIBRATION_DRIFT, MAX_CALIBRATION_DRIFT
        self.horizon_scale = min(hi, max(lo, self.horizon_scale))
        self.adverse_scale = min(hi, max(lo, self.adverse_scale))
        self.size_scale = min(hi, max(lo, self.size_scale))

    def as_dict(self) -> Dict[str, float]:
        return {
            "horizon_scale": round(self.horizon_scale, 4),
            "adverse_scale": round(self.adverse_scale, 4),
            "size_scale": round(self.size_scale, 4),
        }


def evaluate_signal(*, symbol: str, prices: Sequence[float], bar_sec: float,
                    proposed_horizon_sec: float, expected_return: float,
                    round_trip_cost: float, notional_usd: float,
                    realised: Optional[Sequence[float]] = None,
                    calibration: Optional[Calibration] = None,
                    ) -> Dict[str, Any]:
    """Run one candidate signal down the whole lattice.

    Returns every layer's verdict, in order, and whether the signal survived.
    A rejected signal reports WHICH layer rejected it and why -- a lattice
    that only says no is no more useful than the flat list it replaced.
    """
    from .chaos import chaos_profile

    cal = calibration or Calibration()
    cal.clamp()
    verdicts: List[LayerVerdict] = []

    # ---- 1. CHAOS: is prediction possible at this horizon at all? --------
    profile = chaos_profile(symbol, prices, bar_sec)
    usable = profile.get("usable_horizon_sec")
    if usable is not None:
        usable = float(usable) * cal.horizon_scale

    if usable is None:
        # Unmeasurable is not permission. A horizon nobody can bound is one
        # nobody should commit capital across.
        verdicts.append(LayerVerdict(
            "chaos", False,
            f"{symbol}: horizon unmeasurable from {len(prices)} samples, so "
            f"a {proposed_horizon_sec / 60:.0f} min forecast cannot be "
            f"justified",
            evidence=profile))
        return _finish(symbol, verdicts, cal)

    if proposed_horizon_sec > usable:
        verdicts.append(LayerVerdict(
            "chaos", False,
            f"{symbol}: information decays after {usable / 60:.1f} min but "
            f"the signal looks {proposed_horizon_sec / 60:.1f} min ahead -- "
            f"{proposed_horizon_sec / usable:.1f}x past where prediction is "
            f"possible",
            evidence=profile))
        return _finish(symbol, verdicts, cal)

    verdicts.append(LayerVerdict(
        "chaos", True,
        f"{symbol} is {profile['character']} with a {usable / 60:.1f} min "
        f"usable horizon; the {proposed_horizon_sec / 60:.1f} min signal fits "
        f"inside it",
        constraints={"max_horizon_sec": usable,
                     "character": profile["character"]},
        evidence=profile))

    # ---- 2. CALCULUS: direction and whether it is still building ---------
    slope, accel = _derivatives(prices)
    if slope is None:
        verdicts.append(LayerVerdict(
            "calculus", False,
            f"{symbol}: too few points to establish a direction"))
        return _finish(symbol, verdicts, cal)

    # A signal fighting its own second derivative is late: the move is
    # decelerating toward reversal.
    fading = (expected_return > 0 and accel is not None and accel < 0) or \
             (expected_return < 0 and accel is not None and accel > 0)
    verdicts.append(LayerVerdict(
        "calculus", True,
        f"slope {slope:+.6g}/bar, acceleration {accel:+.6g}"
        + ("; the move is DECELERATING, so the edge is late" if fading else ""),
        constraints={"decelerating": bool(fading)},
        evidence={"slope": slope, "acceleration": accel}))

    # ---- 3. STATISTICS: distinguishable from noise? ----------------------
    if realised is not None and len(realised) >= 8:
        mean = statistics.mean(realised)
        try:
            sd = statistics.stdev(realised)
        except statistics.StatisticsError:
            sd = 0.0
        t = mean / (sd / math.sqrt(len(realised))) if sd > 0 else 0.0
        if t <= 1.0:
            verdicts.append(LayerVerdict(
                "statistics", False,
                f"realised mean {mean:+.5f} over {len(realised)} trades has "
                f"t={t:+.2f}; indistinguishable from noise"))
            return _finish(symbol, verdicts, cal)
        verdicts.append(LayerVerdict(
            "statistics", True,
            f"realised mean {mean:+.5f} over {len(realised)} trades, t={t:+.2f}",
            evidence={"t": t, "n": len(realised), "mean": mean}))
    else:
        verdicts.append(LayerVerdict(
            "statistics", True,
            "no realised sample yet; this signal is a hypothesis, and the "
            "size layer below will treat it as one",
            constraints={"unproven": True}))

    # ---- 4. PROBABILITY: expected value net of cost ----------------------
    net_edge = float(expected_return) - float(round_trip_cost)
    if net_edge <= 0:
        verdicts.append(LayerVerdict(
            "probability", False,
            f"expected {expected_return:+.4%} against a {round_trip_cost:.4%} "
            f"round trip is {net_edge:+.4%} -- a loss the model is confident "
            f"about"))
        return _finish(symbol, verdicts, cal)
    verdicts.append(LayerVerdict(
        "probability", True,
        f"expected {expected_return:+.4%} clears the {round_trip_cost:.4%} "
        f"round trip by {net_edge:+.4%}",
        constraints={"net_edge": net_edge},
        evidence={"net_edge": net_edge}))

    # ---- 5. GAME THEORY: why has nobody else taken this? ----------------
    # Governs the chaos layer, per the design: a pattern that is real AND
    # reachable by faster participants is one we arrive at last. The shorter
    # the usable horizon, the more this matters.
    adverse = _adverse_selection_discount(usable, notional_usd) * cal.adverse_scale
    survived = net_edge - adverse
    if survived <= 0:
        verdicts.append(LayerVerdict(
            "game_theory", False,
            f"a {usable / 60:.1f} min horizon is short enough that faster "
            f"participants reach it first; the {adverse:.4%} adverse-selection "
            f"discount eats the {net_edge:+.4%} edge",
            evidence={"adverse_discount": adverse}))
        return _finish(symbol, verdicts, cal)
    verdicts.append(LayerVerdict(
        "game_theory", True,
        f"edge survives a {adverse:.4%} adverse-selection discount with "
        f"{survived:+.4%} left",
        constraints={"survived_edge": survived},
        evidence={"adverse_discount": adverse, "survived_edge": survived}))

    # ---- 6. ALGEBRA: does the arithmetic close? -------------------------
    expected_pl = survived * float(notional_usd)
    if not math.isfinite(expected_pl):
        verdicts.append(LayerVerdict(
            "algebra", False, "expected P/L is not a finite number"))
        return _finish(symbol, verdicts, cal)
    verdicts.append(LayerVerdict(
        "algebra", True,
        f"${notional_usd:.4f} at {survived:+.4%} is {expected_pl:+.6f} expected",
        evidence={"expected_pl_usd": expected_pl}))

    return _finish(symbol, verdicts, cal)


def _finish(symbol: str, verdicts: List[LayerVerdict],
            cal: Calibration) -> Dict[str, Any]:
    failed = [v for v in verdicts if not v.passed]
    return {
        "symbol": symbol,
        "passed": not failed,
        "stopped_at": failed[0].layer if failed else None,
        "reason": failed[0].reason if failed else "survived every layer",
        "layers": [v.as_dict() for v in verdicts],
        "calibration": cal.as_dict(),
    }


def _derivatives(prices: Sequence[float]) -> tuple:
    """First and second derivative of the recent series, per bar."""
    series = [float(p) for p in prices if p is not None and float(p) > 0]
    if len(series) < 6:
        return None, None
    window = series[-16:] if len(series) >= 16 else series
    firsts = [window[i] - window[i - 1] for i in range(1, len(window))]
    if not firsts:
        return None, None
    slope = statistics.mean(firsts)
    if len(firsts) < 3:
        return slope, None
    seconds = [firsts[i] - firsts[i - 1] for i in range(1, len(firsts))]
    accel = statistics.mean(seconds) if seconds else None
    return slope, accel


def _adverse_selection_discount(usable_horizon_sec: float,
                                notional_usd: float) -> float:
    """How much of an edge a faster participant takes before we act.

    Scales with how SHORT the usable horizon is: a pattern that decays in
    thirty seconds is one only the fastest can harvest, and we are not the
    fastest. A pattern that persists for an hour is reachable.

    Small size helps here rather than hurting: a $0.75 clip does not move a
    pool, so it is not worth anyone's while to front-run specifically.
    """
    if usable_horizon_sec <= 0:
        return 1.0
    # 0.5% at a 1-minute horizon, decaying toward ~0.05% by an hour.
    base = 0.005 * math.exp(-usable_horizon_sec / 1800.0)
    # Trades large enough to move a thin pool attract attention.
    size_penalty = min(0.004, max(0.0, (float(notional_usd) - 50.0) / 50000.0))
    return base + size_penalty


def update_calibration(cal: Calibration, *,
                       predicted_horizon_sec: Optional[float] = None,
                       realised_hold_sec: Optional[float] = None,
                       expected_edge: Optional[float] = None,
                       realised_edge: Optional[float] = None,
                       ) -> Dict[str, Any]:
    """Close the feedback loops from REALISED outcomes.

    Only realised numbers move calibration. A loop driven by another model's
    forecast is a model agreeing with itself.
    """
    notes: List[str] = []

    # Horizon loop: outcomes that keep resolving well past the predicted
    # horizon mean the horizon estimate is too tight.
    if predicted_horizon_sec and realised_hold_sec and predicted_horizon_sec > 0:
        ratio = float(realised_hold_sec) / float(predicted_horizon_sec)
        if ratio > 1.5:
            cal.horizon_scale *= 1.1
            notes.append(
                f"positions resolved {ratio:.1f}x past the predicted horizon; "
                f"widening it")
        elif ratio < 0.5:
            cal.horizon_scale *= 0.95
            notes.append(
                f"positions resolved in {ratio:.1f}x the predicted horizon; "
                f"tightening it")

    # Adverse-selection loop: fills better than expected mean the discount
    # was too harsh and is refusing tradeable edges.
    if expected_edge is not None and realised_edge is not None:
        if realised_edge > expected_edge:
            cal.adverse_scale *= 0.95
            notes.append(
                f"realised {realised_edge:+.4%} beat the expected "
                f"{expected_edge:+.4%}; relaxing the adverse discount")
        elif realised_edge < 0 < expected_edge:
            cal.adverse_scale *= 1.1
            notes.append(
                f"expected {expected_edge:+.4%} but realised "
                f"{realised_edge:+.4%}; the discount was too generous")

    cal.clamp()
    return {"calibration": cal.as_dict(), "notes": notes}
