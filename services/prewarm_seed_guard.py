"""Refuse a historical prewarm seed that cannot be the same price series
as the live feed it is about to be spliced into.

WHY THIS EXISTS.  ``trading/bot.py::_prewarm_buffer_from_history`` seeds
``self._buffer`` with historical OHLCV closes shaped exactly like live ticks,
so the model's 60-bar window can span BOTH the seed and the live feed.  When
the two sit at different price scales the seam between them is a log return of
8-10 -- the single-foreign-row case that drags ``organism_snapshots``
``prediction.price_mu`` from -0.17 to -1.8 (proved in b966158,
``scripts/model_window_probe.py``).  One foreign row saturates the whole
window: the model does not see "one odd bar", it sees a distribution whose
scale is set entirely by that bar.

TWO INDEPENDENT WAYS A SEED CAN BE FOREIGN, and both were measured on this
box (Gale, pass 105, 35 live symbols against 626 prewarm files):

  * WRONG SCALE.  PUMP-USDC's newest prewarm file has a median close of
    0.0041 while the live feed reads 1.0220e-07 -- a log ratio of -10.600.
    That is not a price move, it is a different quantity.
  * STALE.  The offending files were 19.9 and 20.2 days old, and their rows
    are appended with their ORIGINAL timestamps into a buffer the rest of the
    bot reads as live ticks.

The verdict is a pure function of numbers so the census
(``scripts/prewarm_seed_census.py``) and the guard test can call exactly what
the bot calls -- a guard measured by a re-implementation is not measured.

DELIBERATELY NOT REFUSED: a seed with no live reference at all.  The prewarm
exists precisely to cover the cold start where no live tick has arrived yet;
refusing there would switch the feature off for every genuinely-new symbol
rather than fixing a seam.  Those are reported as ``no_live_reference`` so
they are counted rather than hidden.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

# A seed whose median close differs from the live median by more than this in
# log space is refused.  0.6931 is a factor of two.
#
# MEASURED 2026-09-10 by scripts/prewarm_seed_census.py over 33 live base
# symbols: the eight that resolve to a FRESH file (<= 1.02 days) all sit at
# |log ratio| <= 0.0802, and the four largest offenders read +10.5995
# (PUMP-USDC), +1.0075 (ALIGN-USDC), +0.7993 (TIBBIR-USDC, resolved to
# 0024_TIBBIR-VIRTUAL.json) and +0.7883 (VIRTUAL-USDC).  The empty band on
# scale alone is 0.0802 to 0.3586.
#
# The threshold sits at a factor of two rather than at the top of the clean
# set BECAUSE THIS GUARD IS NOT A PRICE-MOVE GUARD.  It is asking "is this the
# same series?", and a genuinely volatile symbol can move 30% against a
# three-day-old seed without anything being wrong.  Three rows land between
# the band and the threshold -- ETH-USDC -0.4329, DRB-USDC -0.4164, VVV-USDC
# -0.3586 -- and every one of them is caught by the AGE rule instead (8.3 to
# 94.6 days).  Say that plainly rather than claiming the scale rule catches
# them: at 0.6931 it does not.
DEFAULT_MAX_LOG_RATIO = 0.6931

# Bars older than this are refused.  MEASURED: of the 19 live symbols that
# resolve to a prewarm file at all, 11 are older than three days and they run
# 5.4, 8.3, 9.5, 13.4 (x3), 20.1, 20.3, 20.6 and 94.6 days.  Their rows are
# appended with their ORIGINAL timestamps into a buffer the rest of the bot
# reads as live ticks.  A seed stands in for the recent tape; at three days it
# is a different market regime even when the scale still matches, and the
# 94.6-day ETH-USDC row is not a seed at all.
DEFAULT_MAX_AGE_DAYS = 3.0

_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class SeedVerdict:
    """``ok`` says whether the seed may be spliced into the live window."""

    ok: bool
    reason: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def log_line(self, symbol: str, source: str = "") -> str:
        """One line carrying BOTH numbers, because a refusal that does not
        name what it compared cannot be audited later."""
        d = self.detail
        bits = [f"[bot prewarm] {symbol}: {'ACCEPT' if self.ok else 'REFUSED'} {self.reason}"]
        if d.get("seed_median") is not None:
            bits.append(f"seed_median={d['seed_median']:.10g}")
        if d.get("live_median") is not None:
            bits.append(f"live_median={d['live_median']:.10g}")
        if d.get("log_ratio") is not None:
            bits.append(
                f"log_ratio={d['log_ratio']:+.4f} (max {d.get('max_log_ratio', DEFAULT_MAX_LOG_RATIO):.4f})"
            )
        if d.get("age_days") is not None:
            bits.append(
                f"age_days={d['age_days']:.2f} (max {d.get('max_age_days', DEFAULT_MAX_AGE_DAYS):.2f})"
            )
        if source:
            bits.append(f"file={source}")
        return " ".join(bits)


def _median(values: Sequence[float]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and float(v) > 0)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def median_price(values: Sequence[float]) -> Optional[float]:
    """Median of the positive entries, or None.

    A median rather than a mean or a single sample: ``market_stream``
    interleaves sources and a source publishing a different denomination makes
    consecutive ticks alternate between right and wrong, so one lookup is a
    coin flip (see ``TradingDatabase.recent_market_prices``).
    """
    return _median(values)


def max_log_ratio() -> float:
    try:
        return abs(float(os.getenv("BOT_PREWARM_MAX_LOG_RATIO", DEFAULT_MAX_LOG_RATIO)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_LOG_RATIO


def max_age_days() -> float:
    try:
        return abs(float(os.getenv("BOT_PREWARM_MAX_AGE_DAYS", DEFAULT_MAX_AGE_DAYS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_AGE_DAYS


def seed_verdict(
    *,
    seed_median: Optional[float],
    live_median: Optional[float],
    newest_bar_ts: Optional[float],
    now: float,
    max_log_ratio_: Optional[float] = None,
    max_age_days_: Optional[float] = None,
) -> SeedVerdict:
    """Decide whether a prewarm seed may be spliced into the live window.

    Age is checked BEFORE scale.  A 20-day-old file whose scale happens to
    match is still refused, and reporting the age is more useful to whoever
    reads the log than reporting that the scale was fine.
    """
    lim_ratio = float(max_log_ratio() if max_log_ratio_ is None else max_log_ratio_)
    lim_age = float(max_age_days() if max_age_days_ is None else max_age_days_)
    detail: Dict[str, Any] = {
        "seed_median": None if seed_median is None else float(seed_median),
        "live_median": None if live_median is None else float(live_median),
        "max_log_ratio": lim_ratio,
        "max_age_days": lim_age,
    }

    if seed_median is None or float(seed_median) <= 0:
        return SeedVerdict(False, "seed_has_no_positive_close", detail)

    # BOTH measurements are computed before EITHER is judged, so a refusal on
    # one always carries the other in its detail.  A log line that says
    # "too old" and omits the scale sends the next reader back to re-measure.
    if newest_bar_ts is not None and float(newest_bar_ts) > 0:
        detail["age_days"] = (float(now) - float(newest_bar_ts)) / _SECONDS_PER_DAY
    else:
        detail["age_days"] = None
    if live_median is not None and float(live_median) > 0:
        detail["log_ratio"] = math.log(float(seed_median) / float(live_median))

    age_days = detail["age_days"]
    if age_days is not None:
        if age_days > lim_age:
            return SeedVerdict(False, "seed_too_old", detail)
        if age_days < -1.0:
            # A bar stamped more than a day in the future is not a fresh seed,
            # it is a corrupt or differently-scaled timestamp column.
            return SeedVerdict(False, "seed_timestamp_in_future", detail)

    if detail.get("log_ratio") is None:
        return SeedVerdict(True, "no_live_reference", detail)
    if abs(detail["log_ratio"]) > lim_ratio:
        return SeedVerdict(False, "seed_price_scale_mismatch", detail)
    return SeedVerdict(True, "within_tolerance", detail)
