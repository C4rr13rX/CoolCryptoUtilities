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

...and, since 2026-09-17, deliberately SYMBOL-DEPENDENT as well.

``DEFAULT_RATE`` -- the proportional half -- is one number standing in for
"both legs' DEX fee plus spread" across every token on Base. It is not close
on any of them. ``data/round_trip_quote_census_*.json`` quoted nine symbols
USDC->token->USDC through the Uniswap V3 quoter across a nine-point clip
grid. Pool cost (fee + price impact) at a $2 clip:

    symbol    pool cost   fee tier
    WETH         2.01bp       100
    AERO        13.03bp       500
    CBBTC       14.79bp       500
    AAVE        60.26bp      3000
    CBETH       67.12bp      3000
    VIRTUAL     67.62bp       500
    VVV        212.47bp     10000
    DRB       1507.62bp      3000
    COMP      3301.53bp     10000

A factor of 1,642 between the cheapest and the dearest, against a constant
31.87bp. Because ``roundtrip_cost_rate`` is wired in as an ENTRY GATE -- a
leg is refused unless its predicted return clears it -- that constant is
wrong in BOTH directions at once: it over-charges WETH, the only symbol this
account can round-trip for two basis points, by 16x, and under-charges COMP
by 104x. No tuning of a single number fixes that; raising it to admit COMP
would refuse everything, lowering it to stop over-charging WETH would admit
a token that costs a third of the trade to enter and leave.

GAS IS NOT IN THE CENSUS RATES, and it is not a rate. The census file says so
in ``THESE_ARE_QUOTES_NOT_FILLS.not_modelled``: what it quotes is pool cost
only. The fixed leg above IS the gas -- ``DEFAULT_FIXED_USD`` 0.004047 is the
same median round trip the census recorded in ``measured_gas_context`` -- so
the per-symbol cost is the census pool cost with the SAME fixed leg added:

    cost_rate(n, symbol) = FIXED_USD / n + pool_cost_pct(symbol, n)

which is exactly the old formula with the token-blind proportional term
replaced by the measured one. Dropping the fixed leg instead would have made
every symbol look free at a small clip, where gas alone is 405bp at $0.10.

And these are QUOTES, NOT FILLS. The live lane measured 13.5bp of round-trip
slippage (27.8bp at a $0.75 clip) that no quote here carries, and MEV is not
modelled at all. Every per-symbol number this module returns is therefore a
FLOOR on the cost -- the best case -- which makes the gate it feeds
permissive, never over-tight, in the one dimension it is still blind to.

THE CENSUS COVERS A FIFTH OF THE TAPE. Measured read-only against
``storage/trading_cache.db`` market_stream, 24h to 2026-09-17:

    streamed symbols     61     ticks 6969
    priced by census      4 ( 6.6%)   1435 ticks (20.6%)
    unpriced             57 (93.4%)   5534 ticks (79.4%)

The overlap is exactly WETH-USDC, WETH-USDT, AERO-USDC and VIRTUAL-USDC. Six
of the nine census symbols -- AAVE, CBBTC, CBETH, COMP, DRB, VVV -- are not
streamed at all, so their mispricing is LATENT: the gate is not currently
admitting a COMP leg because no COMP leg is offered. The live, active effect
of this change is the over-charged half: WETH and AERO, 20.6% of ticks.

WHICH IS WHY THE UNPRICED PATH RETURNS TODAY'S NUMBER AND DOES NOT REFUSE.
Fail it closed and 93.4% of the streamed universe stops being tradeable, in a
diff that reads as a cost fix. Unpriced is UNCHANGED, and merely LABELLED:
``roundtrip_cost_rate_detail`` reports ``source="unpriced_default"``, the
symbol is logged once, and ``unpriced_symbols()`` counts them. Unchanged and
labelled is correct; unchanged and silent is the defect this repo already
shipped once as ``round-trip-cost-measured-its-own-default``.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

#: Fixed USD per round trip -- base gas for the two swaps. Measured from
#: settled receipts, not assumed.
DEFAULT_FIXED_USD = 0.004047

#: Proportional part of a round trip -- both legs' DEX fee plus spread.
#:
#: TOKEN-BLIND, and only correct for a token nobody has measured. Reached
#: only on the ``unpriced_default`` path below; anything the census priced
#: uses its own number instead.
DEFAULT_RATE = 0.003187

#: ``source`` values returned by :func:`roundtrip_cost_rate_detail`.
SOURCE_CENSUS = "census"
SOURCE_UNPRICED = "unpriced_default"

_CENSUS_DIR = Path(__file__).resolve().parents[1] / "data"
_CENSUS_GLOB = "round_trip_quote_census_*.json"

_CENSUS_LOCK = threading.Lock()
_CENSUS_TABLE: dict[str, tuple[tuple[float, float], ...]] | None = None
_CENSUS_PATH = ""
_UNPRICED_SEEN: set[str] = set()


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


def _resolve_notional(notional_usd: float | None) -> float:
    """The size to price at, with every bad input falling back to the clip."""
    notional = notional_usd if notional_usd is not None else ghost_clip_usd()
    try:
        notional = float(notional)
    except (TypeError, ValueError):
        notional = ghost_clip_usd()
    if not (notional > 0.0):
        notional = ghost_clip_usd()
    return notional


def normalise_symbol(symbol: object) -> str:
    """Census key for whatever spelling a caller happens to hold.

    Callers carry ``"WETH-USDC"``, ``"weth"``, ``"AERO/USDC"``; the census is
    keyed by the bare base ticker. A lookup that missed on punctuation would
    silently demote a measured symbol to the unpriced default, which is the
    failure this whole change exists to stop -- so the normalisation is shared
    rather than re-implemented per call site.
    """
    if symbol is None:
        return ""
    try:
        text = str(symbol)
    except Exception:  # noqa: BLE001
        return ""
    text = text.strip().upper()
    for sep in ("-", "/", "_", ":"):
        idx = text.find(sep)
        if idx > 0:
            text = text[:idx]
    return text.strip()


def _load_census() -> dict[str, tuple[tuple[float, float], ...]]:
    """Per-symbol pool cost curves, read ONCE from the newest census.

    ``{SYMBOL: ((clip_usd, cost_fraction), ...)}`` sorted by clip. The file
    stores PERCENT (``0.0201`` is 2.01bp); this stores the fraction every
    other number in this module is denominated in.

    A missing, unreadable or malformed file yields an EMPTY table, which sends
    every symbol down the recorded ``unpriced_default`` path. It must never
    raise: this sits under an entry gate, and an exception here would stop
    trading rather than mis-price it.
    """
    global _CENSUS_TABLE, _CENSUS_PATH
    table = _CENSUS_TABLE
    if table is not None:
        return table
    with _CENSUS_LOCK:
        if _CENSUS_TABLE is not None:
            return _CENSUS_TABLE
        built: dict[str, tuple[tuple[float, float], ...]] = {}
        path = ""
        try:
            files = sorted(_CENSUS_DIR.glob(_CENSUS_GLOB))
        except Exception:  # noqa: BLE001
            files = []
        if files:
            # Filenames are ``..._<ISO8601>Z.json``, so lexical order IS
            # chronological order and the last one is the newest.
            newest = files[-1]
            try:
                blob = json.loads(newest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                blob = {}
            quoted = blob.get("quoted") if isinstance(blob, dict) else None
            if isinstance(quoted, dict):
                path = str(newest)
                for raw_symbol, record in quoted.items():
                    key = normalise_symbol(raw_symbol)
                    if not key or not isinstance(record, dict):
                        continue
                    points: list[tuple[float, float]] = []
                    for point in record.get("points") or ():
                        try:
                            clip = float(point[0])
                            percent = float(point[1])
                        except (TypeError, ValueError, IndexError):
                            continue
                        # NaN, negative cost and a zero clip are all readings
                        # a cost model cannot use. A negative round trip is
                        # the exact artifact the census retracted in pass 452.
                        if not (clip > 0.0) or percent != percent or percent < 0.0:
                            continue
                        points.append((clip, percent / 100.0))
                    if points:
                        points.sort(key=lambda pair: pair[0])
                        built[key] = tuple(points)
        _CENSUS_PATH = path
        _CENSUS_TABLE = built
    return _CENSUS_TABLE


def census_path() -> str:
    """Which file priced this process, or "" if nothing did."""
    _load_census()
    return _CENSUS_PATH


def census_symbols() -> tuple[str, ...]:
    """Every symbol that has a measured round trip. Everything else defaults."""
    return tuple(sorted(_load_census()))


def _census_enabled() -> bool:
    """``ROUNDTRIP_COST_CENSUS=0`` returns the whole module to the constant.

    A kill switch, because this number gates live entries: if the census turns
    out to be wrong about a symbol, the operator needs one env var to fall
    back to the behaviour that has been running, not a code change.
    """
    raw = os.getenv("ROUNDTRIP_COST_CENSUS")
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def _interpolate(points: tuple[tuple[float, float], ...], notional: float) -> float:
    """Cost at this clip, linear between the two nearest quoted clips.

    ALWAYS interpolated at the caller's actual notional. A symbol is never
    collapsed to one representative figure, because the curves do not share a
    direction: ``FIXED/n + RATE`` can only FALL with size, and VIRTUAL RISES
    9x across the grid (31.7bp at $0.75 to 291.6bp at $10) while AERO falls
    (24.3bp to 10.3bp) and WETH is flat. One figure per symbol would be wrong
    by up to 4.6x on VIRTUAL depending which clip it happened to be sampled
    at -- the same defect as one figure per book, one level down.

    CLAMPED at both ends, never extrapolated. Below the smallest quoted clip
    it returns the smallest clip's cost, above the largest the largest's. The
    grid stops at $10 and the dear symbols are steeply convex, so extending
    VIRTUAL's or COMP's slope past the last quote would invent a number no
    pool was ever asked for. Clamping is per SYMBOL, not global: AERO's grid
    ends at $6 and WETH's at $10.
    """
    if notional <= points[0][0]:
        return points[0][1]
    if notional >= points[-1][0]:
        return points[-1][1]
    for index in range(1, len(points)):
        high_clip, high_cost = points[index]
        if notional <= high_clip:
            low_clip, low_cost = points[index - 1]
            span = high_clip - low_clip
            if not (span > 0.0):
                return high_cost
            return low_cost + (high_cost - low_cost) * (notional - low_clip) / span
    return points[-1][1]


def _warn_unpriced(symbol: str) -> None:
    """Say out loud, once per symbol, that this one is not measured.

    The defect this guards is one this repo has already shipped: a round-trip
    cost that fell back to its own default and was then read downstream as a
    measurement (``round-trip-cost-measured-its-own-default``). Falling back
    is fine. Being INDISTINGUISHABLE from a measured symbol is not.
    """
    if symbol in _UNPRICED_SEEN:
        return
    _UNPRICED_SEEN.add(symbol)
    message = (
        "round-trip cost for %s is NOT measured -- no row in %s. Charging the "
        "token-blind default %.4f%% + $%.6f fixed. This is a DEFAULT, not a "
        "price: the census spans 2.01bp (WETH) to 3301bp (COMP), so the "
        "default is wrong for this symbol by an unknown factor in an unknown "
        "direction." % (
            symbol,
            _CENSUS_PATH or "(no census file found)",
            _env_float("ROUNDTRIP_FEE_RATE", DEFAULT_RATE) * 100.0,
            _env_float("ROUNDTRIP_FEE_FIXED_USD", DEFAULT_FIXED_USD),
        )
    )
    try:
        from services.logging_utils import log_message

        log_message("roundtrip-cost", message, severity="warning")
    except Exception:  # noqa: BLE001 - a costing call must not die in its logger
        try:
            import logging

            logging.getLogger("roundtrip_cost").warning(message)
        except Exception:  # noqa: BLE001
            pass


def unpriced_symbols() -> tuple[str, ...]:
    """Every symbol this process has had to charge the default for."""
    return tuple(sorted(_UNPRICED_SEEN))


def roundtrip_cost_breakdown(
    notional_usd: float | None = None,
    symbol: object = None,
) -> dict:
    """Every component of the rate, so a reader can audit the number.

    ``gas_rate`` is the fixed leg amortised over this notional; ``pool_rate``
    is the DEX fee plus price impact -- measured per symbol when the census
    has it, and the token-blind constant when it does not. They are reported
    separately because they are not comparable: one is a dollar, one is a
    percentage, and the repo has conflated them before.
    """
    notional = _resolve_notional(notional_usd)
    key = normalise_symbol(symbol)
    fixed = _env_float("ROUNDTRIP_FEE_FIXED_USD", DEFAULT_FIXED_USD)
    gas_rate = fixed / notional

    pool_rate = None
    source = SOURCE_UNPRICED
    if key and _census_enabled():
        points = _load_census().get(key)
        if points:
            pool_rate = _interpolate(points, notional)
            source = SOURCE_CENSUS
        else:
            # Warned only HERE: a symbol the census skipped. Not when the
            # operator has switched pricing off with ROUNDTRIP_COST_CENSUS=0,
            # which is a deliberate choice rather than a gap in the data, and
            # not when no symbol was passed at all, which is every legacy
            # caller and would bury the real gaps in noise.
            _warn_unpriced(key)
    if pool_rate is None:
        pool_rate = _env_float("ROUNDTRIP_FEE_RATE", DEFAULT_RATE)

    return {
        "symbol": key,
        "notional_usd": notional,
        "gas_rate": max(0.0, gas_rate),
        "pool_rate": max(0.0, pool_rate),
        "rate": max(0.0, gas_rate) + max(0.0, pool_rate),
        "source": source,
        "census_path": _CENSUS_PATH,
        "quotes_not_fills": source == SOURCE_CENSUS,
    }


def roundtrip_cost_rate_detail(
    notional_usd: float | None = None,
    symbol: object = None,
) -> tuple[float, str]:
    """``(rate, source)`` -- the cost AND where the cost came from.

    ``source`` is ``"census"`` when this symbol's round trip was measured, and
    ``"unpriced_default"`` when it was not. A caller that cannot tell those
    apart is reading a constant as a measurement.
    """
    parts = roundtrip_cost_breakdown(notional_usd, symbol)
    return float(parts["rate"]), str(parts["source"])


def roundtrip_cost_usd(
    notional_usd: float | None = None,
    symbol: object = None,
) -> float:
    """USD cost of opening AND closing one position of this size.

    With a ``symbol`` the census has priced, this is the measured cost for
    THAT token. Without one it is ``fixed + rate * notional`` to the bit --
    the same arithmetic, in the same order, as before this function grew a
    second parameter.
    """
    parts = roundtrip_cost_breakdown(notional_usd, symbol)
    fixed = _env_float("ROUNDTRIP_FEE_FIXED_USD", DEFAULT_FIXED_USD)
    return max(0.0, fixed + float(parts["pool_rate"]) * float(parts["notional_usd"]))


def roundtrip_cost_rate(
    notional_usd: float | None = None,
    symbol: object = None,
) -> float:
    """The cost as a FRACTION of notional -- what a return must clear.

    ``symbol`` is optional and the one-argument call is byte-for-byte what it
    was: no symbol means no measurement, which means the token-blind default,
    which means the number every existing caller already gets. Supply a symbol
    the census priced and the proportional half becomes that symbol's own
    measured pool cost instead.

    NOT floored at ``DEFAULT_RATE`` on the census path. That floor exists so a
    huge notional cannot amortise the fixed gas leg toward an implausibly free
    round trip -- a guard against a number nobody measured. On a symbol
    somebody DID measure the floor is the bug: it would charge WETH 31.87bp
    when the pool charges 2.01bp, which is precisely the over-charge this
    function was changed to stop. The floor still applies to the unpriced
    default, where it is still the only thing standing between a big clip and
    a free-lunch estimate.
    """
    notional = _resolve_notional(notional_usd)
    parts = roundtrip_cost_breakdown(notional, symbol)
    if parts["source"] == SOURCE_CENSUS:
        return float(parts["rate"])
    return max(float(parts["rate"]), _env_float("ROUNDTRIP_FEE_RATE", DEFAULT_RATE))


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


def net_profit_usd(
    return_pct: float,
    notional_usd: float | None = None,
    symbol: object = None,
) -> float:
    """A fractional return, denominated in USD and charged its round trip.

    This is the single conversion between the two units the ghost book was
    mixing. A trade that went nowhere returns a NEGATIVE number here, because a
    trade that went nowhere loses exactly one round trip.

    ``symbol`` is accepted so this stays the same shape as the rest of the
    module, but NO caller passes it yet and this pass deliberately did not
    change one: re-pricing what the ghost book BOOKS is a different change
    from re-pricing what the gates REFUSE, and mixing per-symbol and
    token-blind rows inside one book is how the unit bug got in last time.
    """
    notional = _resolve_notional(notional_usd)
    try:
        ret = float(return_pct)
    except (TypeError, ValueError):
        ret = 0.0
    return ret * notional - roundtrip_cost_usd(notional, symbol)
