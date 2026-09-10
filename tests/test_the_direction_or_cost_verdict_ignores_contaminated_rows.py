"""Two overshoot rows must not be able to flip the direction-or-cost verdict.

``scripts/tradeable_book.py`` is the tool that names the wall, and it printed

    "The book picks correctly and pays it away: a POSITIVE gross edge means
     this is a cost problem, not a direction problem."

off a raw +0.2625% of notional over 109 live-tradeable trips. That +0.2625%
is TWO rows -- UNI-USDC +122.89% (entry 2.859, exit 6.3723) and BASELINE-USDC
+57.94% -- both of them limit exits that booked the tick which CROSSED their
target rather than the target. That is the gap between two samples, not a
fill. De-contaminated the same book is NEGATIVE, which is the opposite
verdict, and two passes of cost-model work were aimed at it.

The file already had the arithmetic to see this: ``clamped_gross`` re-prices a
limit exit at its own limit, and ``IMPLAUSIBLE_RET`` drops what is still
absurd afterwards. Both were used only by ``symbol_edge``; the headline
verdict went on reading the raw gross. Wiring existing arithmetic into the
number people actually read is the fix, and this test is what pins it.

Each test builds its own rows, so none of them depends on the state of the
database.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tradeable_book import collect as book, render  # noqa: E402

FEE = 0.003187


def _row(symbol, sid, entry, exit_px, qty, reason="take_profit_limit", ts=1.0):
    gross = (exit_px - entry) * qty
    notional = entry * qty
    fees = notional * FEE
    return {
        "symbol": symbol, "strategy_id": sid, "mode": "ghost",
        "reason": reason, "entry_price": entry, "exit_price": exit_px,
        "quantity": qty, "gross": gross, "fees": fees,
        "net": gross - fees, "notional": notional, "ts": ts,
    }


def _losers(n, sid="atf_static"):
    """n small, honest, attributed losing round trips."""
    return [_row("AERO-USDC", sid, 1.0, 0.995, 10.0, reason="stop_loss", ts=float(i))
            for i in range(n)]


def _all_tradeable(_symbol):
    return True


def test_one_overshoot_row_cannot_make_a_losing_book_look_profitable():
    """The regression, reproduced: 20 losers + 1 UNI-shaped overshoot."""
    rows = _losers(20) + [_row("UNI-USDC", "atf_static", 2.859, 6.3723, 5.0)]
    r = book(rows=rows, is_tradeable=_all_tradeable)

    # The RAW book is dragged positive by the single overshoot...
    assert r["tradeable"]["rates"]["gross_pct"] > 0
    # ...and the de-contaminated one is not fooled.
    assert r["sane"]["rates"]["gross_pct"] < 0
    assert r["sane_dropped"]["clamped"] >= 1

    text = render(r)
    assert "Gross is NEGATIVE" in text
    assert "OPPOSITE verdict" in text
    assert "cost problem, not a direction problem" not in text


def test_an_unattributed_row_is_excluded_because_no_strategy_can_spend_it():
    """`unclassified` carried +0.5404 of the live-tradeable book at 86% win."""
    rows = _losers(10) + [
        _row("AERO-USDC", "unclassified", 1.0, 1.02, 100.0, reason="stop_loss")]
    r = book(rows=rows, is_tradeable=_all_tradeable)
    assert r["sane_dropped"]["unattributed"] == 1
    assert r["sane"]["trades"] == 10
    assert r["sane"]["rates"]["gross_pct"] < 0


def test_a_clean_profitable_book_still_reads_as_a_cost_problem():
    """The de-contamination must not simply always say NEGATIVE.

    A book of honest, attributed, non-overshooting winners whose gross beats
    the variable floor has to survive every filter -- otherwise the report
    would be a constant, not a measurement.
    """
    rows = [_row("AERO-USDC", "atf_static", 1.0, 1.02, 10.0,
                 reason="take_profit_limit", ts=float(i)) for i in range(20)]
    r = book(rows=rows, is_tradeable=_all_tradeable)
    assert r["sane"]["trades"] == 20
    assert r["sane_dropped"] == {"implausible": 0, "unattributed": 0, "clamped": 0}
    assert r["sane"]["rates"]["gross_pct"] > 0

    text = render(r)
    assert "cost problem, not a direction problem" in text
    # No flip, so no note about the raw book disagreeing.
    assert "OPPOSITE verdict" not in text


def test_a_stop_loss_is_never_repriced_into_a_smaller_loss():
    """A stop is a MARKET order (1135a79). Clamping it would invent money."""
    rows = _losers(5) + [
        _row("AERO-USDC", "atf_static", 1.0, 0.80, 10.0, reason="stop_loss")]
    r = book(rows=rows, is_tradeable=_all_tradeable)
    assert r["sane_dropped"]["clamped"] == 0
    assert r["sane"]["trades"] == 6
    # The -20% row is fully carried into the de-contaminated net.
    assert r["sane"]["net"] < -1.9


def test_an_empty_book_is_unjudgeable_rather_than_positive():
    """Zero attributable trips must not read as a passing verdict."""
    rows = [_row("AERO-USDC", "unclassified", 1.0, 1.5, 10.0)]
    r = book(rows=rows, is_tradeable=_all_tradeable)
    assert r["sane"]["trades"] == 0
    text = render(r)
    assert "UNJUDGEABLE" in text
    assert "cost problem, not a direction problem" not in text
