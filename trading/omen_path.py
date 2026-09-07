"""What actually happens to a trade opened at a bar -- the PATH, not the endpoint.

``trading.omen_brain.label_omen`` answers one question: where is the close
``horizon_bars`` from now, relative to the round-trip cost. That is not the
question a trade asks. A trade has a take-profit and a stop, and it ends the
moment one of them is touched -- which is usually long before the horizon and
sometimes on the wrong side of a move that finishes green.

Two failures follow from labelling on the endpoint alone, and both of them
teach the substrate the opposite of what pays:

  * A bar that runs +2x cost within three bars and hands it all back by bar
    twelve is labelled ``murk``. That is a completed, profitable, fast round
    trip -- exactly what this system is for -- and the brain is being taught
    to sit on its hands through it.

  * A bar that finishes +1% at bar twelve after visiting -3% at bar four is
    labelled a buy. Any stop this system would actually set was hit at bar
    four, so the "win" the brain learns is a realised loss.

So this module labels the *path*. It walks forward bar by bar from the entry,
asks which barrier price reaches first, and reports the outcome, the realised
return net of nothing (the caller charges cost), and how many bars it took.
No horizon-endpoint arithmetic anywhere.

Units, stated once because this repo has shipped a fee in the wrong currency:
every ``take``/``stop``/return in this module is a SIGNED FRACTION of the
entry price (0.0065 == 0.65%), never basis points, never dollars, never a
percentage number like 0.65. ``bars_held`` is a bar count; the caller
multiplies by its own cadence to get minutes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

try:  # pragma: no cover - exercised by whichever import path resolves
    from trading.omen_brain import ROUND_TRIP_COST, COST_MULTIPLE
except Exception:  # pragma: no cover
    ROUND_TRIP_COST = float(os.getenv("SYMBOL_EDGE_ROUND_TRIP_COST", "0.0065"))
    COST_MULTIPLE = float(os.getenv("OMEN_COST_MULTIPLE", "1.5"))

#: How wide the stop sits relative to the take-profit. 1.0 is symmetric.
#: Below 1.0 the stop is tighter than the target, which is what a
#: cost-covering system wants: the loop's own measurement (see
#: `tight-stop-beats-the-selector`) is that 0.10% beat 0.30% on the live path.
STOP_RATIO = float(os.getenv("OMEN_STOP_RATIO", "1.0"))

#: Outcome tokens. These are the labels a path-trained brain predicts.
PATH_WIN = "win"      # take-profit touched first
PATH_STOP = "stop"    # stop touched first
PATH_FLAT = "flat"    # neither touched inside the horizon
PATH_OUTCOMES = (PATH_WIN, PATH_STOP, PATH_FLAT)

#: Directions a path can be walked.
LONG = "long"
SHORT = "short"


def _f(bar: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        value = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _high(bar: Mapping[str, Any]) -> Optional[float]:
    """Bar high, falling back to close so a corpus without OHLC still walks."""
    return _f(bar, "high") or _f(bar, "close")


def _low(bar: Mapping[str, Any]) -> Optional[float]:
    return _f(bar, "low") or _f(bar, "close")


def _close(bar: Mapping[str, Any]) -> Optional[float]:
    return _f(bar, "close")


@dataclass(frozen=True)
class PathOutcome:
    """What a trade opened at ``index`` did before it ended.

    ``ret`` is the signed fraction realised by the trade BEFORE cost -- the
    caller subtracts the round trip. ``bars_held`` is how long it was open,
    which is the number that decides whether this system trades on the scale
    of minutes or sits in a position for half a day.

    ``ambiguous`` is true when the deciding bar touched BOTH barriers, so the
    order inside that bar is unknown. Those are counted as ``stop`` here --
    the pessimistic reading -- and the count is reported separately so nobody
    mistakes an assumption for a measurement.
    """

    outcome: str
    ret: float
    bars_held: int
    ambiguous: bool
    direction: str = LONG

    @property
    def is_win(self) -> bool:
        return self.outcome == PATH_WIN

    def net(self, cost: float = ROUND_TRIP_COST) -> float:
        """Return after the round trip is charged. Cost is always positive."""
        return self.ret - abs(cost)


def walk_path(
    bars: Sequence[Mapping[str, Any]],
    index: int,
    *,
    horizon_bars: int,
    take: Optional[float] = None,
    stop: Optional[float] = None,
    direction: str = LONG,
) -> Optional[PathOutcome]:
    """Walk forward from ``index`` and report which barrier price hit first.

    Returns ``None`` when the horizon runs off the end of the corpus -- a
    missing future is not a flat outcome, and labelling it flat would teach
    the brain that the end of the file is a reason not to trade.

    The entry fills at ``close[index]``. A win fills at exactly the take
    barrier (a resting limit; the bar traded through it). A stop fills at
    exactly the stop barrier, which is optimistic on a gap -- ``ambiguous``
    marks the bars where that optimism could matter.
    """
    if index < 0 or horizon_bars <= 0:
        return None
    last = index + horizon_bars
    if last >= len(bars):
        return None
    entry = _close(bars[index])
    if entry is None:
        return None

    take_frac = abs(take if take is not None else omen_take())
    stop_frac = abs(stop if stop is not None else take_frac * STOP_RATIO)
    if take_frac <= 0 or stop_frac <= 0:
        return None

    if direction == LONG:
        take_price = entry * (1.0 + take_frac)
        stop_price = entry * (1.0 - stop_frac)
    elif direction == SHORT:
        take_price = entry * (1.0 - take_frac)
        stop_price = entry * (1.0 + stop_frac)
    else:
        raise ValueError(f"direction must be {LONG!r} or {SHORT!r}, got {direction!r}")

    for offset in range(1, horizon_bars + 1):
        bar = bars[index + offset]
        high, low = _high(bar), _low(bar)
        if high is None or low is None:
            continue
        if direction == LONG:
            hit_take = high >= take_price
            hit_stop = low <= stop_price
        else:
            hit_take = low <= take_price
            hit_stop = high >= stop_price
        if not (hit_take or hit_stop):
            continue
        ambiguous = bool(hit_take and hit_stop)
        # Both barriers inside one bar: the order is unknowable from OHLC, so
        # take the loss. Optimism here would manufacture an edge out of a
        # missing column.
        if hit_stop or ambiguous:
            return PathOutcome(PATH_STOP, -stop_frac, offset, ambiguous, direction)
        return PathOutcome(PATH_WIN, take_frac, offset, False, direction)

    # Neither barrier inside the horizon: the position is closed at the
    # horizon close, because a trade that never ends is not a trade.
    exit_close = _close(bars[last])
    if exit_close is None:
        return None
    raw = (exit_close - entry) / entry
    ret = raw if direction == LONG else -raw
    return PathOutcome(PATH_FLAT, ret, horizon_bars, False, direction)


def omen_take(cost: float = ROUND_TRIP_COST,
              multiple: float = COST_MULTIPLE) -> float:
    """The take-profit an omen aims at, as a fraction. Never zero.

    Identical in spirit to ``omen_brain.omen_threshold``: the target must
    clear the round trip by a margin, because a move that merely equals the
    fee is a flat trade dressed as a win.
    """
    return abs(cost) * abs(multiple)


def break_even_win_rate(take: float, stop: float,
                        cost: float = ROUND_TRIP_COST) -> float:
    """The conditional win rate a (take, stop) pair needs to return zero.

    Acting N times, you win ``p*N`` of them for ``take - cost`` each and lose
    ``(1-p)*N`` for ``stop + cost`` each. Setting the sum to zero:

        p (take - cost) = (1 - p) (stop + cost)
        p (take + stop) = stop + cost
        p*              = (stop + cost) / (take + stop)

    The cost is charged on BOTH outcomes, which is the term every naive
    version of this drops. With take == stop == 0.975% and a 0.65% round
    trip, p* is 83.3%: a win nets +0.325% and a loss nets -1.625%.
    """
    width = abs(take) + abs(stop)
    if width <= 0:
        return 1.0
    return min(1.0, (abs(stop) + abs(cost)) / width)


def indiscriminate_win_rate(take: float, stop: float) -> float:
    """The win rate you get by entering at random, with no skill at all.

    A price with no drift is a martingale, so the probability of touching
    ``+take`` before ``-stop`` is ``stop / (take + stop)`` -- the optimal
    stopping identity, and it holds whatever the volatility. Measured on ten
    five-minute corpora at take == stop it comes out at 49.80%, against the
    50.00% this returns.

    The consequence is the one that matters: a barrier scheme has EXACTLY
    ZERO pre-cost expectancy. ``p0 * take - (1 - p0) * stop == 0`` for every
    pair. Barriers alone can never pay; only choosing WHICH bars to enter
    can. A finite horizon pushes the realised rate below this, because paths
    that would have reached the far barrier time out first, so treat it as
    the optimistic bound.
    """
    width = abs(take) + abs(stop)
    if width <= 0:
        return 0.0
    return abs(stop) / width


def skill_required(take: float, stop: float,
                   cost: float = ROUND_TRIP_COST) -> float:
    """Win-rate points the selector must add over entering at random.

        p* - p0 = (stop + cost)/(take + stop) - stop/(take + stop)
                = cost / (take + stop)

    So the skill a predictor must supply is the round-trip cost divided by
    the BARRIER WIDTH, and nothing else -- not the volatility, not the
    horizon, not the symbol. Two readings of that, both load-bearing:

      * Tight barriers are expensive. At the omen's own width (take = stop =
        0.975%) and a 0.65% round trip, the brain must add 25 points of
        conditional win rate. Held-out it has demonstrated none.
      * Widening the barriers is the only lever that does not require the
        cost to fall -- and it buys time, so it trades directly against this
        loop's mandate to close inside tens of minutes. That trade-off is
        real and this function is where it is priced.
    """
    width = abs(take) + abs(stop)
    if width <= 0:
        return 1.0
    return abs(cost) / width


def max_bearable_cost(win_rate: float, take: float, stop: float) -> float:
    """The largest round trip a scheme can pay at a demonstrated win rate.

    Inverting ``break_even_win_rate``: ``c = p (take + stop) - stop``. A
    negative result means the scheme loses money at ZERO fees, so no cost
    reduction can rescue it.
    """
    return float(win_rate) * (abs(take) + abs(stop)) - abs(stop)


def barriers_are_payable(
    take: float,
    stop: float,
    demonstrated_win_rate: float,
    cost: float = ROUND_TRIP_COST,
    margin: float = 0.0,
) -> tuple[bool, str]:
    """Can a selector with this much demonstrated skill pay for this scheme?

    Returns ``(payable, reason)``. The reason is always populated, including
    on success, so a caller can log why it armed as readily as why it did
    not -- a guard that only speaks when it refuses is a guard nobody can
    audit.

    ``demonstrated_win_rate`` must be a CONDITIONAL win rate measured on
    held-out bars the selector actually chose. Passing a training-set number
    here re-answers the question with the answer, and this repo has already
    shipped one edge that was the training set talking.
    """
    needed = break_even_win_rate(take, stop, cost) + abs(margin)
    if not (0.0 <= demonstrated_win_rate <= 1.0):
        return False, (f"demonstrated win rate {demonstrated_win_rate!r} is not a "
                       f"fraction in [0, 1]")
    if abs(take) <= abs(cost):
        return False, (f"take {abs(take):.4%} does not exceed the round trip "
                       f"{abs(cost):.4%}, so even a win loses money")
    if demonstrated_win_rate < needed:
        return False, (
            f"take {abs(take):.4%} / stop {abs(stop):.4%} at cost "
            f"{abs(cost):.4%} needs a {needed:.2%} win rate; the selector has "
            f"demonstrated {demonstrated_win_rate:.2%}. It must add "
            f"{skill_required(take, stop, cost):.2%} over indiscriminate "
            f"entry ({indiscriminate_win_rate(take, stop):.2%}) and has "
            f"added {demonstrated_win_rate - indiscriminate_win_rate(take, stop):+.2%}")
    return True, (
        f"take {abs(take):.4%} / stop {abs(stop):.4%} at cost {abs(cost):.4%} "
        f"needs {needed:.2%}; selector demonstrated {demonstrated_win_rate:.2%}")


def label_path(
    bars: Sequence[Mapping[str, Any]],
    index: int,
    *,
    horizon_bars: int,
    take: Optional[float] = None,
    stop: Optional[float] = None,
) -> Optional[str]:
    """The path label at ``index``: ``win``, ``stop`` or ``flat``.

    This is the training target. Unlike the endpoint label it is a direct
    statement about a trade -- "a long opened here reached its target before
    its stop" -- so a brain that reproduces it is stating something the
    strategy can act on without a translation step.
    """
    outcome = walk_path(bars, index, horizon_bars=horizon_bars,
                        take=take, stop=stop)
    return outcome.outcome if outcome is not None else None
