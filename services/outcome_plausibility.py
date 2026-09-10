"""The ledger's implausibility test, applied to ``trade_outcomes`` reads.

WHY THIS EXISTS
---------------
``trading/strategies/ledger.py::record`` REFUSES an outcome whose shape says it
did not happen -- ``_is_implausible`` -- so the graduation bar never reads one.
``trade_outcomes`` is append-only and has no such guard: every row the ledger
rejected on the way in is still sitting in the money table, and every all-time
per-symbol query reads it as a real fill.

The cost of that is not theoretical. Measured 2026-09-10 over all 210 closed
rows:

    AERO-USDC  2026-08-26 06:27  entry 0.436805 -> exit 1.140000  +160.99%
                                 gross +3.2196   net +3.2066
    AERO-USDC  2026-08-26 16:58  entry 1.140000 -> exit 0.513839   -54.93%
                                 gross -1.0985   net -1.1115

AERO trades near 0.44. The feed printed 1.14, the book took a fake +161% win on
it, AND THEN OPENED THE NEXT POSITION AT THAT FICTIONAL BASIS and booked a
real-looking -55% stop as the price "fell" back. Two passes measured AERO in
opposite directions purely by choosing a window that did or did not contain
those two rows, and the memory ``aero-profit-is-one-implausible-row`` records
the conclusion "AERO is the one symbol we can spend on that pays" -- which is
the opposite of what the de-contaminated book says.

SIX ROWS OF 210 CARRY THE SIGN OF THE WHOLE TABLE
-------------------------------------------------
All-time, the filter below catches exactly six rows, in three contaminated
PAIRS -- a fake win and the real-looking stop taken from its fictional basis:

    AERO-USDC     +160.99%  gross +3.2196   |  AERO-USDC   -54.93%  -1.0985
    AAVE-USDC     +174.16%  gross +3.4830   |  COMP-USDC   -55.33%  -1.1065
    UNI-USDC      +122.89%  gross +0.7196   |  BASELINE    +57.94%  +0.2021

TWO ARMS, AND WHY BOTH
----------------------
``is_implausible`` is the OR of two tests, because the artifact shows up in two
different units and neither arm sees the other's rows:

  * THE RATIO ARM (``IMPLAUSIBLE_RET``). Scale-free, and the one that actually
    catches the contamination: every strategy in this book targets +5% and
    stops at 2-4%, so a booked |return| over 50% is not a fill any of them
    could have produced. It is SYMMETRIC BY NECESSITY, not by preference --
    dropping only the fake wins and keeping the real-looking stops they caused
    would be laundering the record in the other direction. Of the six rows,
    four are positive (+7.62) and two are negative (-2.21).
  * THE DOLLAR ARM, delegated to ``ledger._is_implausible`` so the read and the
    write cannot drift apart. It bounds an outcome in DOLLARS against an
    absolute cap and the strategy's own recent scale, and it is deliberately
    asymmetric: a large LOSS is believable and is filtered only when the
    strategy has enough history to say it is out of character. That asymmetry
    is documented in place in the ledger and must not be re-litigated here --
    this module CALLS that function, it does not restate its rule.

Delegating rather than copying is the point. ``scripts/tradeable_book.py``
carried its own copy of the tradeability predicate until 2026-09-10 and the two
drifted the moment the ledger's changed; a second copy of a rule is a second
answer to one question.

WHAT THIS IS NOT
----------------
It is not a P/L improver. Applied to the all-time book it makes every headline
WORSE, which is the check working:

    AERO-USDC all-time   gross +2.0252 -> -0.0959    net +1.4907 -> -0.6044

A filter that improved the record it judges would be laundering it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

#: A booked return this far from zero is not a fill any strategy here could
#: have produced. Every strategy targets +5% and stops at 2-4%, so the
#: reachable band is narrow and 50% is far outside it with room to spare.
#: Kept here rather than in ``scripts/`` so the reports and any future caller
#: read ONE number.
IMPLAUSIBLE_RET = 0.50

_RATIO = "repricing_return"
_DOLLARS = "outsized_profit"
_UNPRICED = "unpriced_leg"


def _get(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` off a dict, a ``sqlite3.Row`` or anything mapping-like."""
    if isinstance(row, Mapping):
        return row.get(key, default)
    try:
        return row[key]
    except Exception:  # noqa: BLE001
        return getattr(row, key, default)


def _f(row: Any, *keys: str) -> float:
    """First of ``keys`` present on ``row``, as a float, else 0.0."""
    for k in keys:
        v = _get(row, k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def booked_return(row: Any) -> Optional[float]:
    """``exit/entry - 1`` for one closed row, or None when a leg is unpriced.

    None rather than 0.0 on purpose: a row with no entry price is UNJUDGEABLE
    by the ratio arm, and returning 0.0 would silently pass it as plausible.
    """
    entry = _f(row, "entry_price")
    exit_px = _f(row, "exit_price")
    if entry <= 0.0 or exit_px <= 0.0:
        return None
    return exit_px / entry - 1.0


def strategy_scales(rows: Iterable[Any]) -> Dict[str, float]:
    """Average |profit| per strategy over ``rows`` -- the dollar arm's yardstick.

    This is the read-side equivalent of ``ledger._recent_scale``, which divides
    a strategy's total_profit by its trade count. Computed from the rows being
    read so the test does not need the live ledger to be loadable, and so a
    report over an arbitrary window judges each strategy against ITS OWN scale
    in that window rather than against a global constant.

    A strategy with no attributable rows gets no entry, which leaves
    ``relative_to=None`` and hands the decision to the ledger's absolute bound
    -- the conservative direction, and the one the ledger itself takes.
    """
    totals: Dict[str, List[float]] = {}
    for row in rows:
        sid = str(_get(row, "strategy_id") or "").strip()
        if not sid or sid == "unclassified":
            continue
        totals.setdefault(sid, []).append(abs(_f(row, "gross", "gross_profit")))
    return {sid: (sum(v) / len(v)) for sid, v in totals.items() if v}


def implausible_reason(
    row: Any,
    *,
    scales: Optional[Mapping[str, float]] = None,
    max_ret: Optional[float] = None,
) -> Optional[str]:
    """Why this row cannot be counted, or None if it can.

    Named reasons rather than a bare bool so a report can print WHICH arm
    refused a row. A number that moved without a reason beside it is how a
    filter becomes folklore.

    ``max_ret`` overrides ``IMPLAUSIBLE_RET`` for the ratio arm. It exists
    because the default is calibrated ABOVE most of the rows the ledger already
    calls fabricated, and pretending otherwise would be the worse failure. See
    ``SWEEP`` and ``sweep``.
    """
    cap = IMPLAUSIBLE_RET if max_ret is None else float(max_ret)
    ret = booked_return(row)
    if ret is not None and abs(ret) > cap:
        return _RATIO

    profit = _f(row, "gross", "gross_profit")
    relative_to: Optional[float] = None
    if scales:
        sid = str(_get(row, "strategy_id") or "").strip()
        if sid:
            got = scales.get(sid)
            if got is not None and got > 0.0:
                relative_to = float(got)
    if relative_to is None:
        # THE DOLLAR ARM DOES NOT RUN WITHOUT A SCALE TO JUDGE AGAINST.
        #
        # In the ledger, ``relative_to=None`` means "a strategy with no track
        # record", and there the absolute bound rightly governs: the write path
        # always knows whose outcome it is. On the READ side it means something
        # different -- the row carries no attributable strategy_id -- and
        # falling through to the absolute bound there is a HARSHER rule than the
        # write path ever applied. It refuses any unattributed row over $2.00
        # gross, and at the clips this book trades ($2-6 notional) a +2% move on
        # a $100 position is an ordinary trade, not an artifact.
        #
        # Caught by ``test_an_unattributed_row_is_excluded_because_no_strategy
        # _can_spend_it``, which asserts such a row is dropped as UNATTRIBUTED
        # rather than reclassified as implausible -- two different findings that
        # a report must not conflate.
        #
        # The ratio arm has already run and is the one that catches the
        # contamination, so nothing is let through that the six measured
        # artifact rows depend on.
        return None
    try:
        from trading.strategies.ledger import _is_implausible
    except Exception:  # noqa: BLE001
        # UNJUDGEABLE BY THE DOLLAR ARM, and said so rather than assumed clean.
        # The ratio arm above has already run and is the one that catches the
        # contamination; a missing ledger must not turn this into a no-op that
        # quietly reports a "de-contaminated" figure that is the raw one.
        return None
    return _DOLLARS if _is_implausible(profit, relative_to=relative_to) else None


def is_implausible(
    row: Any,
    *,
    scales: Optional[Mapping[str, float]] = None,
    max_ret: Optional[float] = None,
) -> bool:
    """Would the ledger have refused to record this outcome?"""
    return implausible_reason(row, scales=scales, max_ret=max_ret) is not None


def partition(
    rows: Iterable[Any], *, max_ret: Optional[float] = None
) -> Tuple[List[Any], List[Any]]:
    """Split ``rows`` into (countable, refused), judging each in ITS OWN read.

    Materialises the input because the dollar arm needs the whole population to
    know each strategy's scale. Callers get both halves: a filter that hides
    what it dropped cannot be audited, and this repo has shipped a report whose
    entire edge was the rows it silently kept.
    """
    materialised = list(rows)
    scales = strategy_scales(materialised)
    keep: List[Any] = []
    drop: List[Any] = []
    for row in materialised:
        bad = is_implausible(row, scales=scales, max_ret=max_ret)
        (drop if bad else keep).append(row)
    return keep, drop


#: Ratio thresholds the reports walk, loosest first. Not a menu of options to
#: pick from -- every caller reports ALL of them, because the one number a
#: reader inherits is the one that decides the verdict without being argued.
SWEEP = (0.50, 0.25, 0.15, 0.10, 0.05)


def sweep(rows: Iterable[Any],
          thresholds: Iterable[float] = SWEEP) -> Dict[str, Any]:
    """The surviving book's gross at each ratio threshold, and where it flips.

    WHY THIS IS REPORTED RATHER THAN A THRESHOLD BEING CHOSEN
    --------------------------------------------------------
    ``IMPLAUSIBLE_RET`` is 0.50, and measured 2026-09-10 over the 7-day ghost
    book (125 closed rows with positive notional, gross +2.3846) it is
    CALIBRATED ABOVE ALMOST EVERY ROW IT IS MEANT TO CATCH:

        |ret| <= 50%   123 rows   gross +1.4629   POSITIVE
        |ret| <= 25%   122 rows   gross +1.2744   POSITIVE
        |ret| <= 15%   117 rows   gross -0.2626   NEGATIVE  <- the sign flips
        |ret| <= 10%   ...        gross -0.1779   NEGATIVE
        |ret| <=  5%   ...        gross -0.5761   NEGATIVE

    EIGHT rows carry +2.6472 of a +2.3846 book and the 50% cap catches TWO of
    them. The six it misses are the shape the ledger ALREADY calls fabricated --
    BSTONK-USDC +25.35% and +17.28% (both atf_static, and literally the two rows
    [c4f16946] names as the +1.0201 of fabricated fills the re-arm rule reads),
    BSTONK +23.68%, +17.87%, +17.83%, BASECAT +17.31%. Five of the eight are one
    symbol.

    So 0.50 is kept as the DEFAULT -- it is the tested constant, and lowering it
    on this evidence alone would be fitting a threshold to a window -- and the
    whole curve is printed beside every verdict, so a reader sees the sign flip
    at 15% instead of inheriting one number.

    THE RIGHT FIX IS NOT A SMALLER NUMBER. An overshoot is named by filling past
    ITS OWN limit, not by being large: ``tradeable_book.clamped_gross``
    already does that by delegating to ``trading.bot.limit_exit_fill_price``,
    and annulling the rows is [db76611a] / [c4f16946]. This function exists so
    no report claims a clean book on a threshold that never touched those rows.
    """
    materialised = list(rows)
    scales = strategy_scales(materialised)
    steps: List[Dict[str, Any]] = []
    for cap in thresholds:
        keep = [r for r in materialised
                if not is_implausible(r, scales=scales, max_ret=cap)]
        steps.append({
            "max_ret": float(cap),
            "kept": len(keep),
            "dropped": len(materialised) - len(keep),
            "gross": sum(_f(r, "gross", "gross_profit") for r in keep),
            "net": sum(_f(r, "net", "net_profit") for r in keep),
        })
    # The loosest threshold at which the surviving book is still positive, and
    # the first at which it is not. When those differ, the verdict is a choice
    # of threshold rather than a property of the book, and the report must say so.
    positive = [s for s in steps if s["gross"] > 0.0]
    negative = [s for s in steps if s["gross"] <= 0.0]
    return {
        "steps": steps,
        "default": IMPLAUSIBLE_RET,
        "flips_at": negative[0]["max_ret"] if negative else None,
        "verdict_is_threshold_dependent": bool(positive and negative),
    }


def summarise(rows: Iterable[Any]) -> Dict[str, Any]:
    """Counts and dollar effect of the filter over ``rows``, for reporting."""
    keep, drop = partition(rows)
    scales = strategy_scales(list(keep) + list(drop))
    reasons: Dict[str, int] = {}
    for row in drop:
        why = implausible_reason(row, scales=scales) or "unknown"
        reasons[why] = reasons.get(why, 0) + 1
    return {
        "kept": len(keep),
        "dropped": len(drop),
        "reasons": reasons,
        "gross_dropped": sum(_f(r, "gross", "gross_profit") for r in drop),
        "net_dropped": sum(_f(r, "net", "net_profit") for r in drop),
        "implausible_ret": IMPLAUSIBLE_RET,
    }
