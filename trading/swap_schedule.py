"""Turn per-horizon forecasts into a SCHEDULE of swaps, not a choice for now.

THE GAP THIS FILLS
------------------
``BusScheduler.evaluate`` loops per symbol and asks the CDCL solver "which of
these directives is best on this tick". Every candidate is for one symbol at
the current moment, and ``CDCLTradingSolver`` has no time axis at all -- grep
it for ``resolve_ts`` and there is nothing.

Meanwhile ``RouteState.pending_predictions`` already holds exactly the right
data, 2048 entries deep::

    {"label": "1h", "resolve_ts": ..., "predicted_return": 0.031,
     "start_price": 2.4471}

"at time T this symbol will have returned X%". But ``_resolve_predictions``
only touches those entries once ``resolve_ts`` has PASSED, and feeds them to
``self.accuracy.record(...)``. The forecasts are graded after the fact and
then discarded. Nothing ever schedules against them.

So the model's whole opinion about the FUTURE -- which is the thing worth
having -- was being used only to score the model.

WHAT A SCHEDULE IS
------------------
A leg is (symbol, action, expected execution time, expected return). A
schedule is a sequence of legs that:

  * does not spend the same dollar twice -- two legs whose holding periods
    overlap need two lots of capital, and the wallet has one;
  * clears round-trip cost per leg, so a leg that cannot pay for its own gas
    is not scheduled however confident the forecast;
  * carries a GUARD per leg: the price move that says the prediction was
    wrong, and what to do about it;
  * and is recomputed whenever conditions change, because a plan built on a
    forecast is only as good as the forecast still is.

The SAT/UNSAT machinery in ``trading/cdcl_solver.py`` is the right tool and
does not need changing -- it is fed a candidate list and returns a choice or
an UNSAT certificate. This module builds the candidate list from FUTURE-dated
forecasts and enforces the schedule-level constraints the per-tick path never
had to think about.

WHAT THIS DOES NOT DO
---------------------
It does not execute. It produces a plan; the bot's existing entry and exit
paths remain the only things that touch money, and every existing guard still
applies to each leg when its time comes. A schedule is an intention, and an
intention that skipped the swap guard would be a way of laundering a trade
past the checks.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.logging_utils import log_message


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


#: The clamp inside BusScheduler._build_signals:
#:     expected_return = float(np.clip(expected_return, -5.0, 5.0))
#: A forecast landing on exactly this value did not predict +500% -- the
#: extrapolation overflowed and was truncated.
_FORECAST_CLAMP = 5.0


def _is_clamped(predicted: float) -> bool:
    """Did this forecast hit the clamp rather than predict anything?

    The forecast is exp(intercept + slope * future_minutes) fitted on log
    price, so it compounds with the horizon. A slope of 0.05% per minute --
    ordinary noise on a thin token -- extrapolates to +7000% over three days
    and is cut to exactly 5.0. Observed in production 2026-09-04 16:42 as
    "BASECAT-USDC@3d +500.0%".

    This is NOT a size test. A large forecast is allowed to be large; the
    system should act on a genuine 300% opportunity. What is refused is the
    specific value that means the arithmetic ran off the end.
    """
    return abs(abs(float(predicted)) - _FORECAST_CLAMP) < 1e-9


def _max_extrapolation_ratio() -> float:
    """How far past its own evidence a forecast may be projected.

    The fit is a straight line through a window of recent log prices,
    extended `future_minutes` forward. Projecting 4320 minutes (3d) from a
    60-minute window is a 72x extrapolation: the fit has no information at
    that range, and whatever slope the last hour happened to have is simply
    compounded until it dominates.

    24x is one day of projection per hour of evidence. Beyond that the
    number reflects the window's noise rather than the market's direction.
    Set SCHEDULE_MAX_EXTRAPOLATION=0 to disable.
    """
    return _env_float("SCHEDULE_MAX_EXTRAPOLATION", 24.0)


try:
    from services.symbol_edge_gate import refusal_reason as _symbol_edge_refusal
except Exception:  # noqa: BLE001 - a missing gate must not stop planning
    def _symbol_edge_refusal(_symbol: str):  # type: ignore[misc]
        return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class ScheduledLeg:
    """One planned swap, with the guard that says it went wrong."""

    symbol: str
    action: str                      # "enter" | "exit"
    horizon: str                     # the forecast label this came from
    execute_after_ts: float          # not before this
    expires_ts: float                # a forecast is not valid forever
    expected_return: float           # what the model predicts, as a fraction
    entry_price: float               # the price the forecast was made from
    target_price: float              # where it says the price will be
    notional_usd: float              # capital this leg needs
    strategy_id: str = ""
    confidence: float = 0.0

    #: The move that falsifies this leg. Not a stop-loss -- a stop is about
    #: risk, this is about the FORECAST being wrong, and they can differ: a
    #: prediction of +3% that delivers -0.5% has failed without hitting any
    #: sane stop.
    invalidate_below: float = 0.0
    invalidate_above: float = 0.0

    #: Legs that only make sense if this one happened (a hop's second half).
    depends_on: Optional[str] = None
    leg_id: str = ""

    def is_ripe(self, now: float) -> bool:
        return self.execute_after_ts <= now < self.expires_ts

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_ts

    def is_invalidated(self, price: float) -> bool:
        """Has the market already proven this forecast wrong?"""
        if price <= 0:
            return False
        if self.invalidate_below > 0 and price <= self.invalidate_below:
            return True
        if self.invalidate_above > 0 and price >= self.invalidate_above:
            return True
        return False


@dataclass
class Schedule:
    """A plan, plus why each rejected leg was rejected."""

    legs: List[ScheduledLeg] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    built_at: float = 0.0
    capital_usd: float = 0.0

    def ripe(self, now: Optional[float] = None) -> List[ScheduledLeg]:
        moment = float(now if now is not None else time.time())
        return [leg for leg in self.legs if leg.is_ripe(moment)]

    def committed_usd(self, now: Optional[float] = None) -> float:
        """Capital this plan has already promised to legs still in flight."""
        moment = float(now if now is not None else time.time())
        return sum(leg.notional_usd for leg in self.legs if not leg.is_expired(moment))


def predictions_to_candidates(
    symbol: str,
    pending: Iterable[Dict[str, Any]],
    *,
    now: Optional[float] = None,
    min_return: float = 0.0,
) -> List[Dict[str, Any]]:
    """Turn a RouteState's unresolved predictions into schedulable candidates.

    Only predictions that have NOT yet resolved are of any use here: once
    ``resolve_ts`` has passed the forecast is history, and history belongs to
    the accuracy tracker rather than to a plan.
    """
    moment = float(now if now is not None else time.time())
    out: List[Dict[str, Any]] = []
    for entry in pending or []:
        if not isinstance(entry, dict):
            continue
        try:
            resolve_ts = float(entry.get("resolve_ts") or 0.0)
            predicted = float(entry.get("predicted_return") or 0.0)
            start_price = float(entry.get("start_price") or 0.0)
        except (TypeError, ValueError):
            continue
        if resolve_ts <= moment:
            continue                      # already history
        if start_price <= 0 or not math.isfinite(predicted):
            continue
        if predicted <= min_return:
            continue                      # not worth planning around

        # REFUSE THE ARTIFACT, NOT THE OPPORTUNITY.
        #
        # An earlier version of this capped forecasts at 200% on the grounds
        # that nothing bigger had ever been observed. That was wrong: it would
        # have refused a genuine 300% opportunity for being unprecedented,
        # which is precisely the thing worth acting on.
        #
        # What actually went wrong in production is narrower. The forecast is
        # exp(intercept + slope * future_minutes) fitted on log price, so it
        # compounds with the horizon, and _build_signals clamps the result to
        # +/-5.0. "BASECAT-USDC@3d +500.0%" was not a prediction of +500% --
        # it was the clamp constant, reached because a 0.05%/minute drift
        # compounds to +7000% over three days.
        #
        # So the test is whether the number MEANS anything, not whether it is
        # big:
        if _is_clamped(predicted):
            continue

        # ...and whether the fit had any information at that range. A straight
        # line through the last hour, projected three days forward, is a 72x
        # extrapolation: it reports the window's noise, compounded.
        max_ratio = _max_extrapolation_ratio()
        if max_ratio > 0:
            horizon_sec = max(0.0, resolve_ts - moment)
            window_sec = float(entry.get("fit_window_sec") or 0.0)
            if window_sec > 0 and horizon_sec / window_sec > max_ratio:
                continue

        # Symbols the book has proven we lose on are not planned around
        # either. The gate refuses them at entry, so a leg scheduled on one
        # can only ever be refused later -- it would hold capital in the plan
        # and buy nothing. BASECAT-USDC, the symbol in that same +500% leg,
        # is exactly this case: 37 closed round trips at mean -0.0517.
        if _symbol_edge_refusal(str(symbol)):
            continue
        out.append({
            "symbol": symbol,
            "label": str(entry.get("label") or ""),
            "resolve_ts": resolve_ts,
            "predicted_return": predicted,
            "start_price": start_price,
            "target_price": start_price * (1.0 + predicted),
        })
    return out


def build_schedule(
    candidates: Sequence[Dict[str, Any]],
    *,
    capital_usd: float,
    clip_usd: float,
    roundtrip_cost_rate: float,
    now: Optional[float] = None,
    max_legs: Optional[int] = None,
    invalidate_fraction: Optional[float] = None,
) -> Schedule:
    """Choose which forecasts to actually plan around.

    Greedy by expected return per unit of capital-time, which is the right
    ordering for a capital constraint: a leg that ties up the same dollar for
    twice as long has to earn twice as much to be worth the same.

    Every rejection is recorded. A scheduler that silently drops a leg is
    indistinguishable from one that never saw it, and that silence is exactly
    how the existing predictions ended up feeding nothing.
    """
    moment = float(now if now is not None else time.time())
    max_legs = _env_int("SCHEDULE_MAX_LEGS", 6) if max_legs is None else max_legs
    invalidate_fraction = (
        _env_float("SCHEDULE_INVALIDATE_FRACTION", 0.5)
        if invalidate_fraction is None else invalidate_fraction
    )

    schedule = Schedule(built_at=moment, capital_usd=float(capital_usd))
    if clip_usd <= 0 or capital_usd <= 0:
        schedule.rejected.append({
            "reason": "no capital to schedule",
            "capital_usd": capital_usd,
            "clip_usd": clip_usd,
        })
        return schedule

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for cand in candidates:
        predicted = float(cand.get("predicted_return") or 0.0)
        resolve_ts = float(cand.get("resolve_ts") or 0.0)
        hold_sec = max(1.0, resolve_ts - moment)

        # A leg must clear what the round trip costs. Confidence in a forecast
        # is not the same as the forecast being worth acting on: +0.1% at a
        # 0.75% round-trip cost is a loss the model is sure about.
        if predicted <= roundtrip_cost_rate:
            schedule.rejected.append({
                "symbol": cand.get("symbol"),
                "label": cand.get("label"),
                "reason": (f"predicted {predicted:.4%} does not clear the "
                           f"{roundtrip_cost_rate:.4%} round trip"),
            })
            continue

        # Return per dollar per hour held: what makes two legs comparable when
        # they tie up the same capital for different lengths of time.
        density = (predicted - roundtrip_cost_rate) / (hold_sec / 3600.0)
        scored.append((density, cand))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    committed = 0.0
    for density, cand in scored:
        if len(schedule.legs) >= max_legs:
            schedule.rejected.append({
                "symbol": cand.get("symbol"),
                "label": cand.get("label"),
                "reason": f"schedule already holds {max_legs} legs",
            })
            continue

        notional = min(float(clip_usd), float(capital_usd) - committed)
        if notional < float(clip_usd) * 0.5:
            # THE SAME DOLLAR CANNOT FUND TWO OVERLAPPING LEGS. This is the
            # constraint the per-tick path never had to think about, because
            # it only ever considered one trade at a time.
            schedule.rejected.append({
                "symbol": cand.get("symbol"),
                "label": cand.get("label"),
                "reason": (f"only ${max(0.0, capital_usd - committed):.4f} of "
                           f"${capital_usd:.4f} is uncommitted; a leg needs "
                           f"${clip_usd:.4f}"),
            })
            continue

        # One leg per symbol. Two forecasts for the same token at different
        # horizons are the same bet, not two.
        if any(leg.symbol == cand.get("symbol") for leg in schedule.legs):
            schedule.rejected.append({
                "symbol": cand.get("symbol"),
                "label": cand.get("label"),
                "reason": "another horizon already schedules this symbol",
            })
            continue

        start_price = float(cand.get("start_price") or 0.0)
        predicted = float(cand.get("predicted_return") or 0.0)
        resolve_ts = float(cand.get("resolve_ts") or 0.0)

        # The guard. A forecast of +3% that has instead fallen by half the
        # predicted move has been refuted; waiting for a stop-loss to confirm
        # it costs the difference.
        invalidate_below = start_price * (1.0 - predicted * invalidate_fraction)

        leg = ScheduledLeg(
            symbol=str(cand.get("symbol") or ""),
            action="enter",
            horizon=str(cand.get("label") or ""),
            execute_after_ts=moment,
            expires_ts=resolve_ts,
            expected_return=predicted,
            entry_price=start_price,
            target_price=float(cand.get("target_price") or 0.0),
            notional_usd=float(notional),
            strategy_id=str(cand.get("strategy_id") or ""),
            confidence=float(cand.get("confidence") or 0.0),
            invalidate_below=invalidate_below,
            leg_id=f"{cand.get('symbol')}:{cand.get('label')}:{int(resolve_ts)}",
        )
        schedule.legs.append(leg)
        committed += notional

    return schedule


def recalculate(
    schedule: Schedule,
    prices: Dict[str, float],
    *,
    now: Optional[float] = None,
) -> Tuple[Schedule, List[Dict[str, Any]]]:
    """Drop legs the market has overtaken. Returns the surviving plan.

    This is the "recalculate with changing conditions" half. A schedule is a
    statement about the future, and the future keeps arriving: a leg whose
    forecast has expired, or whose guard price has been breached, is not a
    plan any more -- it is a stale opinion holding capital.
    """
    moment = float(now if now is not None else time.time())
    survivors: List[ScheduledLeg] = []
    dropped: List[Dict[str, Any]] = []

    for leg in schedule.legs:
        price = float(prices.get(leg.symbol, 0.0) or 0.0)
        if leg.is_expired(moment):
            dropped.append({
                "leg_id": leg.leg_id,
                "symbol": leg.symbol,
                "reason": f"forecast horizon {leg.horizon} elapsed without execution",
            })
            continue
        if leg.is_invalidated(price):
            dropped.append({
                "leg_id": leg.leg_id,
                "symbol": leg.symbol,
                "reason": (f"price {price:.8g} breached the guard "
                           f"{leg.invalidate_below:.8g}; the forecast of "
                           f"{leg.expected_return:+.2%} is refuted"),
            })
            continue
        survivors.append(leg)

    # Legs whose dependency is gone go with it: the second half of a hop makes
    # no sense once the first half will not happen.
    surviving_ids = {leg.leg_id for leg in survivors}
    kept: List[ScheduledLeg] = []
    for leg in survivors:
        if leg.depends_on and leg.depends_on not in surviving_ids:
            dropped.append({
                "leg_id": leg.leg_id,
                "symbol": leg.symbol,
                "reason": f"depends on {leg.depends_on}, which was dropped",
            })
            continue
        kept.append(leg)

    if dropped:
        log_message(
            "bus-scheduler",
            "schedule recalculated: dropped %d of %d leg(s) -- %s"
            % (len(dropped), len(schedule.legs),
               "; ".join(f"{d['symbol']}: {d['reason']}" for d in dropped[:3])),
            severity="info",
        )

    schedule.legs = kept
    return schedule, dropped
