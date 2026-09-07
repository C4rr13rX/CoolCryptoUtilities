"""Chaos-theoretic structure in price series, gated by game theory.

WHY CHAOS THEORY BELONGS HERE
-----------------------------
A random walk and a deterministic chaotic system look identical to the
statistics already in this package: both wander, both have fat tails, both
defeat a linear fit. They differ in one respect that matters enormously for
trading -- a chaotic system is *predictable at short horizons and only short
horizons*, and the horizon has a measurable length.

That length is the Lyapunov time: how long before two nearby states diverge
beyond usefulness. If a symbol's Lyapunov time is four minutes, a four-minute
forecast on it can carry real information and a four-hour forecast cannot,
however good the model. This package has been making forecasts at horizons
nobody checked were meaningful -- @1w variants reading 4h bars and projecting
240 hours out -- and that is the check.

WHY GAME THEORY GATES IT
------------------------
Chaos measures are seductive and easy to over-read. A positive Lyapunov
exponent on 40 noisy ticks is not evidence of deterministic chaos; it is
evidence of 40 noisy ticks. Left ungoverned, these metrics would licence
trading on any symbol whose noise happened to look structured, which is the
most expensive kind of false positive available.

So game theory holds the gate. Before a chaos signal is allowed to propagate
into a strategy, it must survive the adversarial question this package
already asks elsewhere: if this pattern were real and visible, would someone
faster have taken it already? A short Lyapunov time on a liquid pair is a
pattern a faster participant harvests before we can act. A short Lyapunov
time on a thin pair we can actually reach is worth something. The chaos layer
proposes; the game-theory layer decides what is allowed to reach a decision.

The two are deliberately not merged. Chaos measurement must stay honest about
what the series says, and the decision about whether to ACT on it must stay
honest about who else is looking -- keeping them separate is what stops
either from quietly justifying the other.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Minimum series length before any chaos statistic is reported at all.
#: Below this, every measure here returns noise with a confident face on it.
MIN_SERIES = 64


def _returns(prices: Sequence[float]) -> List[float]:
    """Log returns. Chaos measures operate on returns, not levels."""
    out: List[float] = []
    for i in range(1, len(prices)):
        a, b = float(prices[i - 1]), float(prices[i])
        if a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def hurst_exponent(prices: Sequence[float]) -> Optional[float]:
    """Rescaled-range Hurst exponent, or None when it cannot be measured.

    H = 0.5 is a random walk: increments are independent and there is nothing
    to predict. H > 0.5 is persistent (a move tends to continue); H < 0.5 is
    mean-reverting (a move tends to reverse). Both are tradeable in opposite
    ways, and mistaking one for the other is worse than trading neither.

    None means "not measurable", never 0.5. A short or degenerate series must
    not be reported as a confident random walk -- that is a claim, and we do
    not have the evidence for it.
    """
    series = [float(p) for p in prices if p is not None and float(p) > 0]
    if len(series) < MIN_SERIES:
        return None

    rets = _returns(series)
    if len(rets) < 32:
        return None

    # R/S across several window sizes; the slope in log-log space is H.
    sizes: List[int] = []
    n = 8
    while n <= len(rets) // 2:
        sizes.append(n)
        n *= 2
    if len(sizes) < 3:
        return None

    xs: List[float] = []
    ys: List[float] = []
    for size in sizes:
        rescaled: List[float] = []
        for start in range(0, len(rets) - size + 1, size):
            chunk = rets[start:start + size]
            if len(chunk) < size:
                continue
            mean = statistics.mean(chunk)
            deviations = [c - mean for c in chunk]
            cumulative: List[float] = []
            running = 0.0
            for d in deviations:
                running += d
                cumulative.append(running)
            spread = max(cumulative) - min(cumulative)
            try:
                sd = statistics.stdev(chunk)
            except statistics.StatisticsError:
                continue
            if sd <= 0 or spread <= 0:
                continue
            rescaled.append(spread / sd)
        if rescaled:
            xs.append(math.log(size))
            ys.append(math.log(statistics.mean(rescaled)))

    if len(xs) < 3:
        return None

    # Least squares slope.
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    hurst = num / den
    # Outside [0, 1] the estimate is numerically broken, not informative.
    return hurst if 0.0 <= hurst <= 1.0 else None


def lyapunov_horizon_sec(prices: Sequence[float], bar_sec: float
                         ) -> Optional[float]:
    """How many seconds ahead this series carries information, or None.

    Estimates the largest Lyapunov exponent by Rosenstein's method in spirit:
    find pairs of nearby states and measure how fast they separate. A positive
    exponent means nearby states diverge exponentially, and 1/exponent is the
    timescale over which a forecast decays to noise.

    THIS IS THE NUMBER THAT SAYS HOW FAR OUT A FORECAST MAY LOOK. A horizon
    longer than this is not a bolder prediction, it is arithmetic on noise.
    """
    series = [float(p) for p in prices if p is not None and float(p) > 0]
    if len(series) < MIN_SERIES or bar_sec <= 0:
        return None

    rets = _returns(series)
    if len(rets) < 32:
        return None

    # Embed in 2 dimensions: state i is (r[i], r[i+1]).
    states = [(rets[i], rets[i + 1]) for i in range(len(rets) - 1)]
    if len(states) < 16:
        return None

    divergences: List[float] = []
    horizon = min(8, len(states) // 4)
    if horizon < 2:
        return None

    for i in range(len(states) - horizon):
        # Nearest neighbour that is not a temporal neighbour -- otherwise we
        # measure the series' own smoothness rather than its dynamics.
        best_j = None
        best_d = float("inf")
        for j in range(len(states) - horizon):
            if abs(i - j) < 4:
                continue
            d = math.dist(states[i], states[j])
            if 0 < d < best_d:
                best_d, best_j = d, j
        if best_j is None or best_d <= 0:
            continue
        later = math.dist(states[i + horizon], states[best_j + horizon])
        if later > 0:
            divergences.append(math.log(later / best_d))

    if len(divergences) < 8:
        return None

    # Average divergence per step, converted to an exponent per second.
    per_step = statistics.mean(divergences) / horizon
    if per_step <= 0:
        # Non-positive exponent: states converge or hold. Not chaotic, and
        # not something this function can put a horizon on.
        return None
    exponent_per_sec = per_step / bar_sec
    if exponent_per_sec <= 0:
        return None
    usable = 1.0 / exponent_per_sec

    # A decay timescale cannot be longer than the data it was estimated from.
    # `per_step` is a mean of log ratios: on a near-flat series -- a stablecoin
    # pair, a frozen feed -- those ratios cancel to a value indistinguishable
    # from zero, and 1/tiny is astronomical rather than informative. Measured
    # 2026-09-07 on the live stream: EURC-USDC returned 5.963e16 s of "usable
    # horizon" from a 17980 s window, 3.3e12 times its own observation span.
    #
    # That number is not merely wrong, it DISABLES THE GATE. The chaos layer
    # refuses when `proposed_horizon_sec > usable`, so an unbounded estimate
    # passes every horizon anyone asks for, on exactly the flattest symbols --
    # the same "flat is unmeasurable, not calm" mistake the swap guard's
    # frozen-feed clause was written to fix. An unmeasurable quantity that
    # defaults to the permissive value is the losing shape
    # `services.profit_logic_audit` was built to catch.
    #
    # Returning None is not a weaker answer here: the caller already treats
    # None as "unmeasurable is not permission" and refuses.
    observed_span = len(series) * bar_sec
    if usable > observed_span:
        return None
    return usable


def _return_autocorrelation(prices: Sequence[float], lag: int = 1
                            ) -> Optional[float]:
    """Lag-1 autocorrelation of log returns, or None if unmeasurable.

    Negative means a move tends to be followed by a move the other way --
    mean reversion, stated directly rather than inferred from a scaling
    exponent. Positive means momentum.
    """
    rets = _returns([float(p) for p in prices if p is not None and float(p) > 0])
    if len(rets) < 32:
        return None
    mean = statistics.mean(rets)
    denominator = sum((r - mean) ** 2 for r in rets)
    if denominator <= 0:
        return None
    numerator = sum((rets[i] - mean) * (rets[i - lag] - mean)
                    for i in range(lag, len(rets)))
    value = numerator / denominator
    return value if -1.0 <= value <= 1.0 else None


def chaos_profile(symbol: str, prices: Sequence[float], bar_sec: float
                  ) -> Dict[str, Any]:
    """Everything the chaos layer can say about one symbol.

    Every field may be None, and None always means "not measurable here"
    rather than a neutral default. A caller that cannot tell those apart will
    trade on the absence of evidence.
    """
    hurst = hurst_exponent(prices)
    horizon = lyapunov_horizon_sec(prices, bar_sec)

    # R/S ALONE IS NOT ENOUGH TO CALL MEAN REVERSION.
    #
    # Measured against synthetic series with known character: R/S correctly
    # separated a trending series (H=0.599) from a random walk (H=0.564), but
    # reported a genuinely mean-reverting series as H=0.501 -- a random walk.
    # The estimator runs on log RETURNS, and mean reversion in the level
    # series shows up only weakly there.
    #
    # Return autocorrelation measures it directly: a negative lag-1
    # correlation IS "a move tends to reverse", which is the definition. It is
    # used as a second witness rather than a replacement, because
    # autocorrelation says nothing about persistence at longer lags, which is
    # what R/S is good at.
    #
    # Where they disagree, the more conservative reading wins: calling a
    # random walk "tradeable" costs money, and calling a tradeable series
    # "random" costs only an opportunity.
    autocorr = _return_autocorrelation(prices)

    character = "unmeasurable"
    if hurst is not None:
        if hurst > 0.58 and (autocorr is None or autocorr > -0.05):
            character = "persistent"        # trends continue: momentum
        elif hurst < 0.42 or (autocorr is not None and autocorr < -0.08):
            character = "mean-reverting"    # moves reverse: fade them
        else:
            character = "random-walk"       # nothing to predict

    return {
        "symbol": symbol,
        "samples": len(prices),
        "hurst": hurst,
        "return_autocorrelation": autocorr,
        "character": character,
        "lyapunov_horizon_sec": horizon,
        "usable_horizon_sec": horizon,
        "detail": _describe(symbol, hurst, character, horizon, len(prices),
                            autocorr),
    }


def _describe(symbol: str, hurst: Optional[float], character: str,
              horizon: Optional[float], samples: int,
              autocorr: Optional[float] = None) -> str:
    if hurst is None:
        return (f"{symbol}: {samples} samples is too few to measure structure; "
                f"no chaos claim is made")
    parts = [f"{symbol}: H={hurst:.3f} ({character})"]
    if autocorr is not None:
        parts.append(f"return autocorrelation {autocorr:+.3f}")
    if horizon is not None:
        parts.append(
            f"information decays after ~{horizon / 60:.1f} min, so a forecast "
            f"longer than that is arithmetic on noise")
    else:
        parts.append("no positive divergence rate; horizon unmeasurable")
    return "; ".join(parts)
