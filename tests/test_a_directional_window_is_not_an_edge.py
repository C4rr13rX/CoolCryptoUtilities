"""A per-trade percentage in a directional window is mostly the window.

The failure this prevents
-------------------------
Measured pass 118 over 167 held-out 3600s corpora
(data/brain_experiments/BOTH-HALVES-pass118-iris.md): the DOWN window's crest
half read **+1.2379% per trade** and every one of those percent signs is
real -- and it is 1.19 points WORSE than selling every bar in the same
window, which pays +2.4246% for no skill at all. Pass 114's true-label
ceiling read +1.8884% on the same half and it was quoted as headroom for
exactly this reason: the scoreboard published the LEVEL and left the EDGE
for the reader to derive.

So the edge is a field now, on both halves, with the sell half measured
against the MIRRORED baseline -- what a crest call competes with is selling
every bar, not buying every bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import OMEN_CREST, OMEN_MURK, OMEN_TROUGH  # noqa: E402
from trading.omen_scoreboard import (  # noqa: E402
    READABLE_TRADES, money_scoreboard, pool_scoreboards, render_scoreboard,
)


def _falling_bars(n=400, step=-0.01):
    """A DOWN window: every bar falls 1%, so selling every bar pays."""
    closes, price = [], 100.0
    for _ in range(n):
        closes.append(price)
        price *= (1.0 + step)
    return [{"timestamp": 1_000_000 + i * 3600, "close": c}
            for i, c in enumerate(closes)]


def test_a_crest_cell_in_a_down_window_reports_a_negative_edge():
    """The exact shape that was quoted as headroom, reduced to one assert.

    Every bar falls, so both the crest calls and the baseline are positive.
    The cell is NOT an edge: the crest subset does no better than the whole
    window, so the edge must be zero or below and must never be published as
    a bare positive.
    """
    bars = _falling_bars()
    # Call a crest on every third bar, murk elsewhere. The subset has no
    # information in it -- every bar falls identically.
    calls = [(i, OMEN_CREST if i % 3 == 0 else OMEN_MURK)
             for i in range(24, 380)]
    board = money_scoreboard(bars, calls, horizon_bars=12)

    assert board["crest_net_per_trade"] > 0, "the level is positive"
    assert board["every_bar_net_per_trade"] < 0, "buying every bar loses here"
    # ...and the edge, against the mirrored baseline, is not.
    assert board["crest_edge_vs_baseline"] == pytest.approx(0.0, abs=1e-9)

    rendered = render_scoreboard(board)
    sell_line = [l for l in rendered.splitlines() if "sell (crest)" in l][0]
    assert "EDGE" in sell_line, f"no edge published: {sell_line!r}"
    assert "EDGE +1" not in sell_line and "EDGE +2" not in sell_line


def test_a_crest_call_that_beats_the_window_reports_a_positive_edge():
    """The field is not hard-wired to zero: real selection shows up."""
    bars = _falling_bars(n=400, step=-0.005)
    # Steepen a handful of bars, and call a crest on exactly those.
    for index in range(100, 160):
        bars[index + 12]["close"] = bars[index]["close"] * 0.80
    calls = [(i, OMEN_CREST if 100 <= i < 160 else OMEN_MURK)
             for i in range(24, 380)]
    board = money_scoreboard(bars, calls, horizon_bars=12)
    assert board["crest_edge_vs_baseline"] > 0.05
    assert board["crest_omens"] == 60 and board["sell"]["readable"]


def test_the_buy_edge_is_measured_against_the_unmirrored_baseline():
    """Buying competes with buying every bar, selling with selling it."""
    bars = _falling_bars(n=400, step=+0.01)  # an UP window
    calls = [(i, OMEN_TROUGH if i % 2 == 0 else OMEN_MURK)
             for i in range(24, 380)]
    board = money_scoreboard(bars, calls, horizon_bars=12)
    baseline = board["every_bar_net_per_trade"]
    assert baseline > 0
    assert board["buy_edge_vs_baseline"] == pytest.approx(
        board["buy_net_per_trade"] - baseline)
    # And the two halves do not share a baseline.
    assert board["crest_edge_vs_baseline"] is None, "no crest calls were made"


def test_an_unreadable_cell_publishes_no_edge_at_all():
    """The most quotable-looking number on the least supported cell."""
    bars = _falling_bars()
    calls = [(i, OMEN_CREST if i < 29 else OMEN_MURK) for i in range(24, 380)]
    board = money_scoreboard(bars, calls, horizon_bars=12)
    assert board["crest_omens"] < READABLE_TRADES
    sell_line = [l for l in render_scoreboard(board).splitlines()
                 if "sell (crest)" in l][0]
    assert "UNREADABLE" in sell_line and "n=" in sell_line
    assert "EDGE" not in sell_line


def test_the_edge_survives_pooling_and_uses_the_pooled_baseline():
    """Pooled boards get a pooled baseline, not the first board's."""
    def _board(every_n, every_mean):
        return {
            "horizon_bars": 12, "round_trip_cost": 0.0065,
            "omen_threshold": 0.00975,
            "readable_trades_floor": READABLE_TRADES,
            "scored_bars": every_n, "dropped_no_future": 0,
            "buy": {"n": 40, "paid": 20, "net_total": 0.40,
                    "net_per_trade": 0.01, "precision_paid": 0.5,
                    "readable": True},
            "sell": {"n": 40, "paid": 20, "net_total": 0.40,
                     "net_per_trade": 0.01, "precision_paid": 0.5,
                     "readable": True},
            "every_bar_n": every_n, "every_bar_net_per_trade": every_mean,
        }

    pooled = pool_scoreboards([_board(100, 0.05), _board(300, -0.01)])
    # Pooled baseline is the count-weighted mean: (100*0.05 + 300*-0.01)/400.
    assert pooled["every_bar_net_per_trade"] == pytest.approx(0.005)
    assert pooled["buy_edge_vs_baseline"] == pytest.approx(0.01 - 0.005)
    # Selling every bar is -baseline MINUS two round trips, not -baseline.
    assert pooled["sell_every_bar_net_per_trade"] == pytest.approx(
        -0.005 - 2 * 0.0065)
    assert pooled["crest_edge_vs_baseline"] == pytest.approx(
        0.01 - (-0.005 - 2 * 0.0065))
