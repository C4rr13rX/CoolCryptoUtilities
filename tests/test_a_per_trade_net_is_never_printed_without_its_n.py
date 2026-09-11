"""A per-trade net that appears without its trade count is how n=1 got quoted.

The failure this prevents, measured 2026-09-10 from the four pass-111 reports
in data/brain_experiments: ``buy_net_per_trade`` +0.0031 at ``buy_hit_rate``
1.0 was the only positive cell in the table and it rested on ONE trade, because
``buy_omens`` was a different key in a 50-key JSON and nobody quoting the net
read it. These tests assert the contract line by line: every rendered line that
carries a percentage also carries an ``n=``, and a cell below the readability
floor is rendered as UNREADABLE rather than as a number.

They also pin the arithmetic of the sell half, which had no test at all because
it had no implementation: a crest call is scored on the move it AVOIDS, so a
price that falls further than the round trip is money kept, at the same
threshold as the buy half.
"""
from __future__ import annotations

import re

import pytest

from trading.omen_brain import OMEN_CREST, OMEN_MURK, OMEN_TROUGH
from trading.omen_scoreboard import (
    READABLE_TRADES, forward_return, money_scoreboard, render_scoreboard,
)

#: Any percentage in rendered output. The contract is that no line matching
#: this may lack an ``n=``.
PERCENT = re.compile(r"[-+]?\d+\.\d+%")


def _bars(closes):
    return [{"timestamp": 3600 * i, "close": c} for i, c in enumerate(closes)]


def test_a_per_trade_net_is_never_printed_without_its_n():
    # One winning trough call and nothing else: the exact shape of the pass-111
    # cell that got quoted.
    bars = _bars([100.0] * 5 + [110.0] * 5)
    board = money_scoreboard(bars, [(0, OMEN_TROUGH)], horizon_bars=5)

    assert board["buy_omens"] == 1
    text = render_scoreboard(board)
    for line in text.splitlines():
        if PERCENT.search(line) and "per trade" in line:
            assert "n=" in line, f"per-trade net with no trade count: {line!r}"


def test_a_cell_below_the_readability_floor_says_unreadable():
    bars = _bars([100.0] * 5 + [110.0] * 5)
    board = money_scoreboard(bars, [(0, OMEN_TROUGH)], horizon_bars=5)

    assert board["buy"]["readable"] is False
    assert board["readable"] is False
    line = [l for l in render_scoreboard(board).splitlines() if "buy" in l][0]
    assert "UNREADABLE" in line
    assert "n=1" in line
    assert str(READABLE_TRADES) in line


def test_a_cell_at_the_floor_is_readable_and_still_carries_its_n():
    # 40 calls, alternating winner and loser, all at the same horizon.
    closes = []
    calls = []
    for i in range(40):
        closes.extend([100.0, 110.0 if i % 2 == 0 else 95.0])
    bars = _bars(closes)
    calls = [(2 * i, OMEN_TROUGH) for i in range(40)]
    board = money_scoreboard(bars, calls, horizon_bars=1)

    assert board["buy_omens"] == 40
    assert board["buy"]["readable"] is True
    line = [l for l in render_scoreboard(board).splitlines() if "buy" in l][0]
    assert "UNREADABLE" not in line
    assert "n=40" in line


def test_an_empty_cell_reports_none_not_a_break_even_zero():
    # max(1, n) in the old report wrote 0.0000% for "no calls at all", which
    # reads as a measured break-even. It must read as nothing measured.
    bars = _bars([100.0] * 10)
    board = money_scoreboard(bars, [(0, OMEN_MURK)], horizon_bars=5)

    assert board["buy_omens"] == 0
    assert board["buy_net_per_trade"] is None
    assert board["trough_precision"] is None
    line = [l for l in render_scoreboard(board).splitlines() if "buy" in l][0]
    assert "NO CALLS" in line
    assert not PERCENT.search(line)


def test_the_sell_half_is_scored_on_the_move_it_avoids():
    # Price falls 10%: a crest call keeps that move less the round trip.
    bars = _bars([100.0] * 5 + [90.0] * 5)
    board = money_scoreboard(bars, [(0, OMEN_CREST)], horizon_bars=5,
                             cost=0.0065)

    assert board["crest_omens"] == 1
    assert board["crest_net_per_trade"] == pytest.approx(0.10 - 0.0065)
    assert board["crest_precision"] == pytest.approx(1.0)
    # And a crest call on a RISING price is a miss, not a win.
    bars_up = _bars([100.0] * 5 + [110.0] * 5)
    miss = money_scoreboard(bars_up, [(0, OMEN_CREST)], horizon_bars=5,
                            cost=0.0065)
    assert miss["crest_precision"] == pytest.approx(0.0)
    assert miss["crest_net_per_trade"] == pytest.approx(-0.10 - 0.0065)


def test_a_bar_with_no_future_is_dropped_and_counted_not_scored_as_flat():
    # Padding the tail with zeros turns a losing tail into a break-even one.
    bars = _bars([100.0] * 6)
    board = money_scoreboard(bars, [(0, OMEN_TROUGH), (5, OMEN_TROUGH)],
                             horizon_bars=5)

    assert board["dropped_no_future"] == 1
    assert board["scored_bars"] == 1
    assert board["buy_omens"] == 1
    assert forward_return(bars, 5, 5) is None


def test_the_baseline_is_scored_on_exactly_the_bars_that_were_scored():
    # An every-bar baseline measured over a different bar set than the omens
    # is not a baseline; it is two experiments compared.
    bars = _bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    calls = [(0, OMEN_TROUGH), (1, OMEN_MURK), (2, OMEN_CREST)]
    board = money_scoreboard(bars, calls, horizon_bars=3)

    assert board["every_bar_n"] == board["scored_bars"] == 3
