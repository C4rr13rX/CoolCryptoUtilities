"""One formula for what a round trip costs, and one clip it is priced at.

Every lane that simulates or judges a round trip needs the same two numbers:
the notional a trade is opened at, and what opening and closing it costs. When
each lane carried its own copy, they drifted -- and the drift is invisible,
because the wrong number is still a number.

Measured 2026-09-06 on the 5-day ghost book (242 paired round trips):
``services/atf_static_strategy.py`` wrote ``details["profit"] = (mark/entry)-1``
-- a FRACTION -- into the same column ``trading/bot.py`` writes USD into, and
charged no fee at all. 104 of the 242 trades (43%) were scout rows, so the book
the live gate reads was 43% fractions summed with 57% dollars:

    strategy              n    profit==return_pct    unit
    atf_static_scout    104         104/104          FRACTION
    atf_static           45           7/45           USD
    (29 others)          93           0/93           USD

Both halves of that mattered. Denominating the scout's 104 trades in USD at the
live clip and charging them the round trip they never paid moves every statistic
the live gate reads:

    net           +3.578  ->  +11.512
    expectancy   +0.0148  ->  +0.0476   USD/trade
    payoff ratio   1.081  ->    2.698
    loss rate      0.331  ->    0.512   <- fees turn flat trades into losses
    longest loss streak 7 ->       13

The loss rate and the streak got WORSE, which is the point: the fractional unit
was reporting a book as safe by never charging it for the trades it lost money
on. 33.1% of all 242 exits closed at a return smaller than one round trip --
+0.0597 of gross return between them, -$1.4951 net at a $6 clip.

The cost model is measured from this account's settled base receipts rather than
assumed, and is deliberately SIZE-DEPENDENT: the fixed part does not shrink, so
a $0.75 trade and a $6.00 trade do not pay the same percentage.

    cost_usd = ROUNDTRIP_FEE_FIXED_USD + ROUNDTRIP_FEE_RATE * notional
"""

from __future__ import annotations

import os

#: Fixed USD per round trip -- base gas for the two swaps. Measured from
#: settled receipts, not assumed.
DEFAULT_FIXED_USD = 0.004047

#: Proportional part of a round trip -- both legs' DEX fee plus spread.
DEFAULT_RATE = 0.003187


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value == value and value not in (float("inf"), float("-inf")) else default


def ghost_clip_usd() -> float:
    """USD notional a SIMULATED round trip is priced at.

    A ghost book run at a different size from the live lane prices a different
    game. ``LIVE_MIN_CLIP_USD`` is the floor the capital plan publishes as
    ``min_clip_usd`` and the size a real entry is raised to, so it is the honest
    proxy for a lane that cannot reach the plan object itself.

    Never returns zero: a simulated trade priced at nothing books its return as
    a bare fraction again, which is the exact bug this module exists to close.
    """
    for name, default in (("LIVE_MIN_CLIP_USD", 0.0), ("GHOST_MIN_TRADE_USD", 0.0)):
        clip = _env_float(name, default)
        if clip > 0.0:
            return clip
    return 2.0


def roundtrip_cost_usd(notional_usd: float | None = None) -> float:
    """USD cost of opening AND closing one position of this size."""
    notional = notional_usd if notional_usd is not None else ghost_clip_usd()
    try:
        notional = float(notional)
    except (TypeError, ValueError):
        notional = ghost_clip_usd()
    if not (notional > 0.0):
        notional = ghost_clip_usd()
    fixed = _env_float("ROUNDTRIP_FEE_FIXED_USD", DEFAULT_FIXED_USD)
    rate = _env_float("ROUNDTRIP_FEE_RATE", DEFAULT_RATE)
    return max(0.0, fixed + rate * notional)


def roundtrip_cost_rate(notional_usd: float | None = None) -> float:
    """The same cost as a FRACTION of notional -- what a return must clear.

    Floored at the proportional rate: a very large notional would otherwise
    amortise the fixed part toward zero and imply a round trip could be nearly
    free, which no DEX offers.
    """
    notional = notional_usd if notional_usd is not None else ghost_clip_usd()
    try:
        notional = float(notional)
    except (TypeError, ValueError):
        notional = ghost_clip_usd()
    if not (notional > 0.0):
        notional = ghost_clip_usd()
    rate = _env_float("ROUNDTRIP_FEE_RATE", DEFAULT_RATE)
    return max(roundtrip_cost_usd(notional) / notional, rate)


#: The worst round-trip cost rate a live entry is allowed to be placed at.
#:
#: This is a ceiling on ``roundtrip_cost_rate(notional)``, not on the cost in
#: dollars, because the thing that kills a small clip is the RATE: the fixed
#: leg does not shrink with the trade.
DEFAULT_MAX_COST_RATE = 0.005


def min_viable_notional_usd() -> float:
    """Smallest live notional whose round trip can still be paid for.

    ``roundtrip_cost_rate(n) = FIXED/n + RATE`` falls as ``n`` grows, so there
    is a notional below which the fixed leg alone eats more than a trade can
    plausibly return. Solving ``FIXED/n + RATE = MAX_RATE`` for ``n`` gives it::

        n = FIXED / (MAX_RATE - RATE) = 0.004047 / (0.005 - 0.003187) = $2.23

    Measured against the only live evidence this account has -- all 18 live
    round trips it has ever settled, replayed against this module's own cost
    model at a range of clips:

        clip     gross$      cost$       net$
        $ 0.75   +0.0770     0.1159     -0.0389
        $ 1.00   +0.1026     0.1302     -0.0276
        $ 2.00   +0.2053     0.1876     +0.0177
        $ 6.00   +0.6158     0.4170     +0.1987
        $19.66   +2.0177     1.2007     +0.8170

    The same trades, the same direction calls, the same fills -- only the size
    differs, and the sign of the book flips between $1 and $2. Those 18 trades
    ran at a median notional of $0.750 and a maximum of $3.00 against a plan
    that published ``min_clip_usd`` $6.00, so the book was decided by the clip
    rather than by the market. Their gross return averaged +0.5702%/trade,
    which puts empirical break-even at ``FIXED/(0.005702 - RATE)`` = $1.61.
    The $2.23 this returns sits above that and below the $6.00 clip the plan
    authorises, so it refuses the losing region without touching a normal
    entry.

    This is a FLOOR TO REFUSE AT, never a size to raise to. A clip is shrunk by
    real constraints -- the wallet, ``live_capital_cap_usd`` headroom -- and
    the honest response to "we can only afford a quarter of the sanctioned
    clip" is to not take the trade, because the fixed gas leg is spent whether
    or not the position is big enough to repay it. Raising the size instead
    would spend money the cap was written to protect.

    Returns 0.0 when the configured ceiling is at or below the proportional
    rate: no notional can satisfy it, and a floor that refuses every entry is
    the same as being switched off. The caller keeps its existing behaviour.
    """
    rate = _env_float("ROUNDTRIP_FEE_RATE", DEFAULT_RATE)
    max_rate = _env_float("LIVE_MAX_ROUNDTRIP_COST_RATE", DEFAULT_MAX_COST_RATE)
    fixed = _env_float("ROUNDTRIP_FEE_FIXED_USD", DEFAULT_FIXED_USD)
    headroom = max_rate - rate
    if not (headroom > 0.0) or not (fixed > 0.0):
        return 0.0
    return fixed / headroom


def net_profit_usd(return_pct: float, notional_usd: float | None = None) -> float:
    """A fractional return, denominated in USD and charged its round trip.

    This is the single conversion between the two units the ghost book was
    mixing. A trade that went nowhere returns a NEGATIVE number here, because a
    trade that went nowhere loses exactly one round trip.
    """
    notional = notional_usd if notional_usd is not None else ghost_clip_usd()
    try:
        notional = float(notional)
    except (TypeError, ValueError):
        notional = ghost_clip_usd()
    if not (notional > 0.0):
        notional = ghost_clip_usd()
    try:
        ret = float(return_pct)
    except (TypeError, ValueError):
        ret = 0.0
    return ret * notional - roundtrip_cost_usd(notional)
