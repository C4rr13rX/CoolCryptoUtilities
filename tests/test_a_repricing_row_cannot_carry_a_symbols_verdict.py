"""A +161% single-tick repricing row must not decide whether a symbol pays.

THE FAILURE THIS PREVENTS
-------------------------
``trading/strategies/ledger.py::record`` refuses an outcome whose shape says it
did not happen. ``trade_outcomes`` is append-only and has no such guard, so
every row the ledger rejected on the way in is still in the money table and
every all-time per-symbol query read it as a real fill.

Measured on the real book 2026-09-10, AERO-USDC over 52 closed rows:

    booked                      gross +2.0252   net +1.4907
    artifact pair dropped       gross -0.0959   net -0.6044

The pair is one event: entry 0.436805 -> exit 1.140000 (+160.99%, a fake win on
a misprinted tick) and then entry 1.140000 -> exit 0.513839 (-54.93%, a real-
looking stop taken from that fictional basis). On the booked numbers AERO is
"the one symbol we can spend on that pays"; on the filtered ones it is a loser.
Two passes reached opposite conclusions about it purely by window choice.

WHY THIS TEST GOES RED WITHOUT THE FILTER
-----------------------------------------
Every assertion below is paired: the BOOKED figure is asserted positive and the
REPORTED figure asserted negative, on the same rows. Remove the filter and the
two become the same number, so the second half of each pair fails. A test that
only asserted "negative" would pass against a book that happened to be negative
for unrelated reasons.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# The real AERO event, to the prices and quantities in the book.
_FAKE_WIN = {
    "symbol": "AERO-USDC",
    "strategy_id": "atf_static",
    "entry_price": 0.436805,
    "exit_price": 1.140000,
    "quantity": 4.5786,
    "gross_profit": 3.2196,
    "net_profit": 3.2066,
    "fee_cost": 0.0130,
    "reason": "time_take_profit",
}

# The SAME event's second half: its entry IS the first row's contaminated exit.
_PAIRED_STOP = {
    "symbol": "AERO-USDC",
    "strategy_id": "atf_static",
    "entry_price": 1.140000,
    "exit_price": 0.513839,
    "quantity": 1.7546,
    "gross_profit": -1.0985,
    "net_profit": -1.1115,
    "fee_cost": 0.0130,
    "reason": "stop_loss",
}


def _ordinary(n: int, *, gross: float) -> list:
    """``n`` unremarkable AERO round trips, inside the band a 5% target allows."""
    out = []
    for i in range(n):
        entry = 0.44
        ret = gross / (entry * 1.0)
        out.append({
            "symbol": "AERO-USDC",
            "strategy_id": "atf_static",
            "entry_price": entry,
            "exit_price": entry * (1.0 + ret),
            "quantity": 1.0,
            "gross_profit": gross,
            "net_profit": gross - 0.013,
            "fee_cost": 0.013,
            "reason": "stale_loser" if gross < 0 else "take_profit",
        })
    return out


def test_the_ledgers_own_test_refuses_both_halves_of_the_repricing_pair():
    """The filter is SYMMETRIC, because the artifact has two halves.

    Dropping only the fake win would be cherry-picking in the other direction:
    the -54.93% stop is not a strategy losing money, it is a position measured
    from a price that did not exist when it was opened.
    """
    from services.outcome_plausibility import (
        implausible_reason, is_implausible, partition, strategy_scales,
    )

    rows = [_FAKE_WIN, _PAIRED_STOP] + _ordinary(20, gross=-0.01)
    scales = strategy_scales(rows)

    assert is_implausible(_FAKE_WIN, scales=scales), (
        "+160.99% on one tick is outside every strategy's 5% target band"
    )
    assert is_implausible(_PAIRED_STOP, scales=scales), (
        "-54.93% from the fake win's exit price is the same event, not a trade"
    )
    assert implausible_reason(_FAKE_WIN, scales=scales) == "repricing_return"

    # And an ordinary trip survives. A filter that refused these would be
    # laundering the record, which is the failure mode the ledger documents.
    keep, drop = partition(rows)
    assert len(drop) == 2, "exactly the pair, not the ordinary book"
    assert len(keep) == 20


def test_a_plus_161_percent_row_cannot_carry_aeros_all_time_verdict():
    """The per-symbol edge table must report AERO negative, not positive.

    This is the acceptance criterion of [b9295f16] in the units the item names:
    the symbol's all-time gross flips sign when the rows the ledger already
    rejected stop being counted.
    """
    from scripts.tradeable_symbol_edge import collect

    # A book that LOSES everywhere except the artifact pair, which is the real
    # AERO shape: 50 small negatives plus one +3.2196 fake win.
    rows = [_FAKE_WIN, _PAIRED_STOP] + _ordinary(50, gross=-0.04)
    rep = collect(rows=rows)
    aero = [d for d in rep["symbols"] if d["symbol"] == "AERO-USDC"]
    assert aero, "AERO-USDC must still appear in the table; it is not hidden"
    d = aero[0]

    # THE PAIR THAT MAKES THIS TEST FAIL WITHOUT THE FILTER.
    assert d["gross_booked"] > 0.0, (
        "precondition: the BOOKED book is positive, so a report that does not "
        "filter reaches the opposite verdict -- got %+.4f" % d["gross_booked"]
    )
    assert d["gross"] < 0.0, (
        "AERO's all-time gross must be reported NEGATIVE once the repricing "
        "rows are dropped -- got %+.4f" % d["gross"]
    )
    assert d["net_booked"] < 0.0, "and so must its net"
    assert d["implausible"] == 2, "both halves of the pair counted as artifacts"
    assert d["sign_flipped"] is True, (
        "the report must SAY the verdict was being carried by an artifact row, "
        "not just quietly return a different number"
    )
    # The artifact supplied more than the whole booked edge.
    assert d["artifact_share_of_gross"] > 100.0


def test_both_reports_read_one_implausibility_rule():
    """``tradeable_book`` and ``tradeable_symbol_edge`` cannot drift apart.

    The rule was a literal inside ``scripts/tradeable_book.py`` while the
    all-time per-symbol table had no filter at all. A second copy of a rule is
    a second answer to one question, and this repo has already shipped that
    shape once with the tradeability predicate.
    """
    import scripts.tradeable_book as tb
    from services import outcome_plausibility as op

    assert tb.IMPLAUSIBLE_RET is op.IMPLAUSIBLE_RET
    assert tb._row_is_implausible is op.is_implausible

    # And the all-time table actually calls it, rather than importing it and
    # reading the raw rows anyway.
    from scripts.tradeable_symbol_edge import collect
    clean = collect(rows=_ordinary(5, gross=0.02))
    assert clean["rows_dropped"] == 0
    contaminated = collect(rows=_ordinary(5, gross=0.02) + [_FAKE_WIN])
    assert contaminated["rows_dropped"] == 1
    assert abs(contaminated["gross_dropped"] - 3.2196) < 1e-9


def test_the_filter_makes_the_record_worse_not_better():
    """The direction is the check working.

    A plausibility filter that improved the book it judges would be laundering
    it. The ledger documents this from the other side at its "only outsized
    GAINS are filtered" note; pinning the direction here means a future change
    that quietly starts dropping losses fails rather than flatters.
    """
    from scripts.tradeable_symbol_edge import collect

    rows = [_FAKE_WIN, _PAIRED_STOP] + _ordinary(50, gross=-0.04)
    rep = collect(rows=rows)
    d = [s for s in rep["symbols"] if s["symbol"] == "AERO-USDC"][0]
    assert d["gross"] < d["gross_booked"]
    assert rep["gross_dropped"] > 0.0, (
        "the dropped rows are net POSITIVE in aggregate: the filter removes "
        "fiction that flattered the book"
    )
