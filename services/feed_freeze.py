"""Is this price window a market, or one number repeated?

WHY THIS FILE EXISTS
--------------------
Measured 2026-09-23 (pass 690, item e44d880d) on ``storage/trading_cache.db``:
15 of 266 ``trade_outcomes`` rows carrying both prices have
``entry_price == exit_price`` EXACTLY. All 15 are ``wallet='ghost'``. Their
``gross_profit`` sums to exactly +0.0000 and their ``net_profit`` to -0.2473 --
pure cost, no move. They were held 13.4 minutes to 19.3 hours, so they are not
harmless no-op round trips; the price under them never changed while they were
open.

The cause is NOT a cache inside the bot, and that matters because it decides
where the fix goes. Counted directly on ``market_stream`` over each window:

    PUMP-USDC    09-10 20:30 -> 09-11 12:45 UTC   153 rows, 2 distinct prices,
                                                  125 rows exactly 1.022046762e-07
    JITOSOL-USDC 09-07 00:30 -> 09-07 02:50 UTC    65 rows, 1 distinct price
    DRB-USDC     09-16 03:15 -> 09-16 04:05 UTC    10 rows, 1 distinct price

153 ticks over 16.2 hours is one every 6.4 minutes. The feed was not quiet --
it was loud and identical. The ``raw`` column names the publisher:

    {"symbol": "PUMP-USDC", "price": 1.022046762e-07, "volume": 0.0,
     "rest": "geckoterminal", "consensus_confidence": 1.0}

``volume`` 0.0 on every one. A REST pool endpoint returns the LAST TRADE price
for a pool nothing is trading, and the poller republishes it each cycle as a
fresh observation (``trading/data_stream.py:4296`` geckoterminal,
``trading/data_stream.py:4338`` dexscreener).

The shipped guard cannot see this. ``trading/data_stream.py:1424`` drops
invented ticks by SOURCE LABEL -- ``_SYNTHETIC_SOURCES`` is fallback / offline /
bootstrap / reference / snapshot. geckoterminal and dexscreener are real
venues, so a frozen quote from one walks straight through. That guard filters
who said it; this defect is what they said.

WHAT IS ALREADY WRITTEN, AND WHERE
----------------------------------
The value-level check exists twice, and neither instance is on the generic
ghost entry path:

  * ``trading/strategies/money_button.py:145`` -- distinct-price floor,
    ``max(3, ceil(n/3))``, declines ``feed_frozen``. Correct and already tuned
    to the 10-sample window. It runs inside ONE strategy, and that strategy
    fires 0 trades a day. Its own comment states this finding verbatim:
    "when a feed cannot price one it repeats a seed value forever ... Every
    such trade exits at exactly its entry and drags the win rate toward zero".
  * ``trading/swap_validator.py:565`` and ``:609`` -- span-aware ``vol_frozen``,
    on the LIVE swap path only.

This module is neither a new idea nor a replacement for those two. It is the
same two clauses in one place a caller on the ghost entry path can reach,
so that the other 43 strategies stop opening round trips on a price nobody
checked for movement. It deliberately keeps money_button's constants, so
lifting it does not silently re-tune a guard that was tuned against live data.

NOT THE SAME FAULT AS CONTAMINATED ENTRY PRICES. ``services/
entry_price_corroboration.py`` guards a DIFFERENT defect: an entry price the
tape never corroborated, with an exit that is exact. That one stops at
2026-09-04 and already has a guard. This one is entry == exit with a price that
did not move, it runs to 2026-09-22, and it has no guard.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

__all__ = ["FreezeVerdict", "window_is_frozen"]

#: money_button's shipped floor, kept verbatim so lifting this does not
#: re-tune a constant that was measured against live data. See
#: ``trading/strategies/money_button.py:145``: at the 10-sample window the
#: strategies actually run on, a fixed 6 demanded 60% distinct prices and
#: blocked every symbol (AERO 2 distinct of 7, CBBTC 5 of 6). A third of the
#: window still rejects a seed value replayed forever, without demanding that
#: an on-chain quote change on every single poll.
MIN_DISTINCT_FLOOR = 3
DISTINCT_FRACTION = 3.0

#: swap_validator's span clause (``trading/swap_validator.py:565``). One price
#: repeated is only evidence of a freeze once enough of them have arrived over
#: enough wall-clock that a live pair would have moved. Below these, "frozen"
#: and "quiet" are not distinguishable and this module abstains.
FROZEN_MIN_RETURNS = 4
FROZEN_MIN_SPAN_SEC = 300.0


class FreezeVerdict(tuple):
    """``(frozen, reason, diagnostics)``.

    ``frozen`` is True only when the window is positively shown not to move.
    An undecidable window returns False with reason ``"undecidable"`` -- it
    abstains rather than refusing, because refusing on too little data is how
    a guard bans the only approved strategy.
    """

    __slots__ = ()

    def __new__(cls, frozen: bool, reason: str, diag: Dict[str, float]) -> "FreezeVerdict":
        return super().__new__(cls, (bool(frozen), str(reason), dict(diag)))

    @property
    def frozen(self) -> bool:
        return self[0]

    @property
    def reason(self) -> str:
        return self[1]

    @property
    def diagnostics(self) -> Dict[str, float]:
        return self[2]


def window_is_frozen(
    prices: Sequence[float],
    timestamps: Optional[Sequence[float]] = None,
) -> FreezeVerdict:
    """Decide whether ``prices`` is a market or one number repeated.

    Two clauses, in the order the two shipped implementations apply them:

      1. SPAN CLAUSE (swap_validator). Enough prints, spanning enough
         wall-clock, and ``max == min``. This is the clause that catches the
         measured rows: JITOSOL's 65 identical prints over 2.3 hours, DRB's 10
         over 50 minutes, PUMP's 125 over 16 hours.
      2. DISTINCT CLAUSE (money_button). Fewer than ``max(3, ceil(n/3))``
         distinct values. This catches a window that moved ONCE in an hour and
         is otherwise a seed value -- PUMP's full window holds 2 distinct
         prices across 153 prints and is frozen by this clause even though
         ``max != min``.

    ``timestamps`` is optional. Without it the span clause cannot run and only
    the distinct clause applies; a caller that has the stamps should pass them,
    because the span clause is the one that can tell a frozen feed from a
    genuinely calm minute.
    """
    clean = [float(p) for p in prices if p is not None and math.isfinite(float(p)) and float(p) > 0.0]
    n = len(clean)
    diag: Dict[str, float] = {"n": float(n)}
    if n == 0:
        # No price at all is a different fault with a different remedy, and
        # saying "frozen" here would hide it. The caller must refuse on the
        # absent price, not on this verdict.
        return FreezeVerdict(False, "no_price", diag)

    distinct = len(set(clean))
    diag["distinct"] = float(distinct)

    span: Optional[float] = None
    if timestamps is not None:
        stamps = [float(t) for t in timestamps if t is not None and math.isfinite(float(t))]
        if len(stamps) >= 2:
            span = float(max(stamps) - min(stamps))
            diag["span_sec"] = span

    # 1. Span clause. Needs prints, wall-clock, and a genuinely single value.
    if (
        span is not None
        and float(n - 1) >= FROZEN_MIN_RETURNS
        and span >= FROZEN_MIN_SPAN_SEC
        and max(clean) == min(clean)
    ):
        return FreezeVerdict(True, "feed_frozen", diag)

    # 2. Distinct clause. money_button's floor, unchanged -- but gated on the
    # same span the clause above requires, and that gate is NOT cosmetic.
    #
    # money_button runs this on a fixed-cadence window whose span is implied by
    # its own sampler, so it never needed a wall-clock term. Lifted somewhere a
    # caller chooses the window, it does: four identical prints ten seconds
    # apart is an on-chain pair between trades, and calling that "frozen" is
    # the failure mode where a guard refuses a working feed. Caught by
    # test_a_calm_minute_is_not_called_frozen, which failed on the first
    # version of this file.
    #
    # With no timestamps the caller has opted out of the wall-clock term and
    # gets money_button's behaviour exactly.
    min_distinct = max(MIN_DISTINCT_FLOOR, int(math.ceil(n / DISTINCT_FRACTION)))
    diag["min_distinct"] = float(min_distinct)
    if span is not None and span < FROZEN_MIN_SPAN_SEC:
        return FreezeVerdict(False, "undecidable", diag)
    if n < min_distinct:
        # The window is shorter than its own floor. There is nothing to
        # measure and abstaining is the honest answer.
        return FreezeVerdict(False, "undecidable", diag)
    if distinct < min_distinct:
        return FreezeVerdict(True, "feed_frozen", diag)

    return FreezeVerdict(False, "ok", diag)
