"""Twelve bars and a hundred and forty-four can be the same question.

The failure this prevents
-------------------------
``omen_trough_census`` pooled 3600s corpora (720 minutes = 12 bars) with 300s
corpora (720 minutes = 144 bars) through a hand-rolled ``_pool`` that never
looked at the horizon. That pool is gone; the shared
``omen_scoreboard.pool_scoreboards`` does it. But the shared one first
refused the census outright, because all it could see was that the BAR
horizons disagreed -- and bars are not the question. Minutes are.

So a board carries its wall-clock horizon now, pooling is judged on minutes,
and the pooled board names every bar-horizon it spanned rather than wearing
the first board's number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trading.omen_brain import OMEN_TROUGH  # noqa: E402
from trading.omen_scoreboard import money_scoreboard, pool_scoreboards  # noqa: E402


def _board(horizon_bars, horizon_minutes, cost=0.0065):
    return {
        "horizon_bars": horizon_bars, "horizon_minutes": horizon_minutes,
        "bar_seconds": None, "round_trip_cost": cost,
        "omen_threshold": cost * 1.5, "readable_trades_floor": 30,
        "scored_bars": 100, "dropped_no_future": 0,
        "buy": {"n": 40, "paid": 20, "net_total": 0.4, "net_per_trade": 0.01,
                "precision_paid": 0.5, "readable": True},
        "sell": {"n": 40, "paid": 20, "net_total": 0.4, "net_per_trade": 0.01,
                 "precision_paid": 0.5, "readable": True},
        "every_bar_n": 100, "every_bar_net_per_trade": 0.002,
    }


def test_two_cadences_pool_when_they_ask_the_same_wall_clock_horizon():
    pooled = pool_scoreboards([_board(12, 720.0), _board(144, 720.0)])
    assert pooled["buy_omens"] == 80
    assert pooled["horizon_minutes"] == 720.0
    # And the pool says which bar-horizons it spanned, with a count each.
    assert pooled["horizon_bars_spanned"] == {12: 1, 144: 1}


def test_two_wall_clock_horizons_are_still_refused():
    with pytest.raises(ValueError, match="wall-clock"):
        pool_scoreboards([_board(12, 720.0), _board(12, 60.0)])
    with pytest.raises(ValueError, match="cost"):
        pool_scoreboards([_board(12, 720.0), _board(12, 720.0, cost=0.01)])


def test_boards_with_no_minutes_fall_back_to_the_strict_bar_test():
    """An unknown horizon cannot be checked, so it is not waved through."""
    with pytest.raises(ValueError, match="horizon_minutes"):
        pool_scoreboards([_board(12, None), _board(144, None)])
    # Same bars and no minutes is still poolable -- that is the old contract.
    assert pool_scoreboards([_board(12, None), _board(12, None)])["buy_omens"] == 80


def test_money_scoreboard_records_the_minutes_the_caller_asked_for():
    bars = [{"timestamp": 1_000_000 + i * 3600, "close": 100.0 + i}
            for i in range(60)]
    board = money_scoreboard(bars, [(0, OMEN_TROUGH)], horizon_bars=12,
                             horizon_minutes=720.0, bar_seconds=3600)
    assert board["horizon_minutes"] == 720.0 and board["bar_seconds"] == 3600
    # Never guessed: a caller that does not say gets None, not a fabrication.
    quiet = money_scoreboard(bars, [(0, OMEN_TROUGH)], horizon_bars=12)
    assert quiet["horizon_minutes"] is None and quiet["bar_seconds"] is None
