"""An edge that one round trip supplies is not an edge, and must not print as one.

The same shape has now been found three times in this repo: AERO's +161%
repricing row, BSTONK carrying 100% of a positive sign on 12% of the volume,
and -- measured 2026-09-10 -- the live-tradeable book's whole +0.2625% of
notional coming from a single UNI-USDC round trip that gained +122.89% on a
$0.59 notional.

It survives a symbol-admission rule, and correctly so: a rule with a derived
minimum sample cannot judge a symbol with one round trip, because one round
trip is not evidence. The failure is not in the rule, it is in the REPORT
making the claim on the rule's behalf. With that one row the admitted book
reads +0.4711% and "clears the variable floor, profitable above a $2.66
clip"; without it, -0.0119% and below the floor. The encouraging half is the
one that gets quoted onto a board and into a commit message.

So the report carries a leave-one-out beside every edge it prints, and it
refuses to offer the clip curve for an edge that does not survive it -- a
clip curve says "this is a cost problem, spend more per trip", which is
expensive advice to take from one row.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import tradeable_book as tb  # noqa: E402


def _row(symbol: str, gross: float, notional: float) -> dict:
    """One closed ghost round trip. Fees are the receipts model at that size."""
    fees = tb.COST_FIXED + tb.COST_VARIABLE * notional
    return {"symbol": symbol, "strategy_id": "s", "mode": "ghost",
            "net": gross - fees, "gross": gross, "fees": fees,
            "notional": notional, "ts": 0.0}


def _one_row_carries_it() -> list:
    """60 trips slightly negative on gross, plus one +122.89% on $0.59.

    The measured shape: the rest of the book is BELOW zero, so the single row
    supplies more than 100% of the edge -- it is carrying the sign, not just
    the size. The aggregate still reads as a book comfortably above the
    variable floor, which is the trap.
    """
    rows = [_row("STEADY-USDC", -0.0001, 2.0) for _ in range(60)]
    rows.append(_row("UNI-USDC", 0.7196, 0.5856))
    return rows


def test_the_leave_one_out_names_the_row_and_the_share_it_supplies():
    rows = _one_row_carries_it()
    jk = tb._jackknife(rows)
    assert jk["applies"]
    assert jk["symbol"] == "UNI-USDC"
    # A single row supplying MORE than the whole edge means the rest of the
    # book is negative and the row is carrying the sign, not just the size.
    assert jk["share_of_edge"] > 100.0
    assert jk["gross_pct_without"] < tb.COST_VARIABLE * 100.0


def test_a_one_row_edge_is_not_offered_a_clip_curve():
    rows = _one_row_carries_it()
    book = {"trades": 0, "wins": 0, "losses": 0, "net": 0.0, "gross": 0.0,
            "fees": 0.0, "notional": 0.0}
    for r in rows:
        tb._add(book, r["net"], r["gross"], r["fees"], r["notional"])
    book["win_rate"] = tb._win_rate(book)
    book["rates"] = tb._rates(book)
    book["jackknife"] = tb._jackknife(rows)

    # The aggregate genuinely looks like an edge above the floor. That is the
    # trap: without the leave-one-out this is the number that gets reported.
    assert book["rates"]["gross_pct"] > book["rates"]["variable_floor_pct"]

    text = tb.render_rule({
        "days": 7.0, "fitted_on_window": len(rows), "fitted_on_older": 0,
        "baseline": book,
        "in_sample": {"refused": {}, "book": book},
        "out_of_sample": {"refused": {}, "book": book},
    })
    assert "ONE ROW" in text
    assert "UNI-USDC" in text
    assert "profitable above a" not in text, (
        "a clip curve was offered for an edge that one row supplies -- that is "
        "the report telling the next pass to spend more per trip on nothing"
    )


def test_a_real_edge_still_gets_its_clip_curve():
    """The check must not swallow a book that survives losing its best row."""
    good = tb.COST_VARIABLE * 3.0
    rows = [_row("REAL-USDC", good * 2.0, 2.0) for _ in range(40)]
    book = {"trades": 0, "wins": 0, "losses": 0, "net": 0.0, "gross": 0.0,
            "fees": 0.0, "notional": 0.0}
    for r in rows:
        tb._add(book, r["net"], r["gross"], r["fees"], r["notional"])
    book["win_rate"] = tb._win_rate(book)
    book["rates"] = tb._rates(book)
    book["jackknife"] = tb._jackknife(rows)
    assert book["jackknife"]["gross_pct_without"] > \
        book["rates"]["variable_floor_pct"]

    text = tb.render_rule({
        "days": 7.0, "fitted_on_window": len(rows), "fitted_on_older": 0,
        "baseline": book,
        "in_sample": {"refused": {}, "book": book},
        "out_of_sample": {"refused": {}, "book": book},
    })
    assert "profitable above a" in text
    assert "ONE ROW" not in text
