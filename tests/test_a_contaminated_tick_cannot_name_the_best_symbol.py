"""The per-symbol tradeable table must not be led by a single bad tick.

Two failures, both of which this repo has actually shipped, and both of which
the FIRST version of ``symbol_edge`` reproduced before these tests existed:

1. RANKED BY RAW NET, a ONE-trip symbol came top. Measured 2026-09-10 on the
   real book, AAVE-USDC held a single row of entry 129.485 -> exit 354.990
   (+174.16%, net +3.4700) and was reported as "BEST TRADEABLE SYMBOL" of a
   book whose whole de-contaminated total is +0.3035. AAVE's other two round
   trips that window sit at 131.50 and 128.38 -- the feed printed another
   asset's price. Naming that symbol best is how a losing symbol gets built on.

2. THE CONTAMINATED PRICE BECOMES THE NEXT TRADE'S ENTRY BASIS, so filtering
   only the fake WIN leaves the matching fake LOSS in the book and reports a
   number that is wrong in the other direction. The real pair:

       AERO  entry 0.436805 -> exit 1.140000  +160.99%  net +3.2066
       AERO  entry 1.140000 -> exit 0.513839   -54.93%  net -1.1115

   The second entry IS the first exit. A filter that is not symmetric on |ret|
   would keep the -1.1115 and drop the +3.2066.

Both are tested against ``symbol_edge`` with hand-built rows, so they fail
against a ranking with no depth floor and against a one-sided filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tradeable_book import (  # noqa: E402
    IMPLAUSIBLE_RET, MIN_RANK_TRIPS, symbol_edge,
)


def _row(symbol, entry, exit_px, *, strategy="s1", qty=1.0, reason="",
         fees=0.0):
    """One closed ghost round trip, priced consistently from its two legs.

    gross and net are DERIVED from the legs rather than passed in, so a test
    cannot accidentally assert on a number that disagrees with its own prices.
    """
    gross = (exit_px - entry) * qty
    return {
        "symbol": symbol, "strategy_id": strategy, "mode": "ghost",
        "reason": reason, "entry_price": entry, "exit_price": exit_px,
        "quantity": qty, "net": gross - fees, "gross": gross, "fees": fees,
        "notional": entry * qty, "ts": 0.0,
    }


def _all_tradeable(_symbol):
    return True


def test_a_one_trip_contaminated_symbol_is_not_named_best():
    """A single +174% row must not outrank a symbol with a real book."""
    rows = [_row("AAVE-USDC", 129.485, 354.990, reason="time_take_profit:1.7416")]
    # A genuine, boring, mildly profitable book with real depth.
    for _ in range(MIN_RANK_TRIPS):
        rows.append(_row("AERO-USDC", 1.0, 1.01))

    rep = symbol_edge(rows=rows, is_tradeable=_all_tradeable)
    best = rep["best_symbol"]

    assert best is not None, "a symbol with depth exists and must be rankable"
    assert best["symbol"] == "AERO-USDC", (
        "the one-trip +174%% row was named best: %r" % (best["symbol"],))

    aave = [s for s in rep["symbols"] if s["symbol"] == "AAVE-USDC"][0]
    assert aave["rankable"] is False, "a 1-trip symbol must not be rankable"
    assert aave["implausible"] == 1, "the +174% row must be flagged"
    # And its fake profit must not reach the headline number.
    assert aave["net_sane"] == 0.0, (
        "the contaminated row still contributes %r to net_sane"
        % (aave["net_sane"],))


def test_the_contaminated_exit_that_becomes_the_next_entry_is_dropped_too():
    """Both halves of one contamination event go, or the sign is wrong."""
    win = _row("AERO-USDC", 0.436805, 1.140000, reason="time_take_profit:1.6099")
    # The SAME price, now used as an entry basis -- this is the real pair.
    loss = _row("AERO-USDC", 1.140000, 0.513839, reason="stop_loss:-0.5493")
    assert win["net"] > 0 and loss["net"] < 0, "fixture must hold both signs"

    rep = symbol_edge(rows=[win, loss], is_tradeable=_all_tradeable)
    aero = [s for s in rep["symbols"] if s["symbol"] == "AERO-USDC"][0]

    assert aero["implausible"] == 2, (
        "expected BOTH halves flagged, got %d -- a one-sided filter keeps the "
        "fake loss and reports a book that is too pessimistic"
        % aero["implausible"])
    assert aero["trips_sane"] == 0
    assert aero["net_sane"] == 0.0
    assert rep["total_net_sane"] == 0.0, (
        "one contamination event leaked %r into the headline"
        % (rep["total_net_sane"],))


def test_an_ordinary_sized_loss_is_kept():
    """The filter is for contamination, not for losses it dislikes.

    A stop is a market order and genuinely fills through its level, so a real
    -4% stop must survive. Dropping ordinary losers would flatter the book,
    which is the same class of error in the opposite direction.
    """
    rows = [_row("CBETH-USDC", 100.0, 96.0, reason="stop_loss:-0.0400")]
    rep = symbol_edge(rows=rows, is_tradeable=_all_tradeable)
    sym = rep["symbols"][0]

    assert sym["implausible"] == 0, "an ordinary -4% stop was dropped"
    assert sym["trips_sane"] == 1
    assert sym["net_sane"] < 0, "the loss must still count against the book"
    assert abs(IMPLAUSIBLE_RET - 0.50) < 1e-9, (
        "threshold moved; re-read the note in tradeable_book.py before "
        "changing what counts as a contaminated print")


def test_the_bar_is_judged_on_the_de_contaminated_book():
    """20/55%/positive must be read off sane trips, not booked ones.

    Built so the strategy clears the bar ONLY if the contaminated row counts:
    19 real trips at 53% (below both legs of the bar) plus one fake +174% win.
    """
    rows = [_row("X-USDC", 1.0, 1.01) for _ in range(10)]          # 10 wins
    rows += [_row("X-USDC", 1.0, 0.99) for _ in range(9)]          # 9 losses
    rows.append(_row("X-USDC", 1.0, 2.75, reason="time_take_profit:1.7500"))

    rep = symbol_edge(rows=rows, is_tradeable=_all_tradeable)
    cell = rep["grid"][0]

    assert cell["book"]["trades"] == 20, "fixture should book 20 round trips"
    assert cell["trips_sane"] == 19, "the fake row must not count toward depth"
    assert cell["clears_bar"] is False, (
        "a strategy cleared 20/55%% on 19 real trips plus one contaminated "
        "print: %d sane trips, %.3f win rate, net %+.4f"
        % (cell["trips_sane"], cell["win_rate_sane"], cell["net_sane"]))
    assert not rep["strategies_clearing_bar"]
