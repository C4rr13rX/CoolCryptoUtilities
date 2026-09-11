"""A held-out window must be POWERED and REGIME-PURE, and those pull apart.

Item [5ec44914]. The pass-111 self-pool verdict rested on 19 trades, and the
instinctive fix -- lengthen the held-out window -- silently destroys the thing
the window was for: stretch it far enough to place 60 trades and its up-rate
walks back to the corpus mean, so it is no longer a DOWN window and the
both-windows rule is no longer being applied.

These tests pin the sizing instrument that says so, at the seams where it could
quietly lie:

  * a horizon that rounds to zero bars would compare a close against itself and
    report a flawless, free, entirely fictional edge;
  * a window sweep that returned a window failing its own purity bound would
    let an arm call the market a DOWN window;
  * a sweep that stopped at the first failing length would under-report how
    long a pure window can be, because purity is NOT monotone in length.
"""
from __future__ import annotations

import pytest

from scripts.omen_self_pool_power import (
    LENGTH_GRID, LOOKBACK_BARS, TRAIN_BARS, horizon_bars, longest_pure_window,
    probe_plan,
)


def _row(corpus="data/historical_ohlcv/arbitrum/0002_LINK-WETH.json",
         chain="arbitrum", *, start, bars, horizon=12, troughs=17):
    return {"corpus": corpus, "chain": chain, "horizon_bars": horizon,
            "down": {"start": start, "end": start + bars, "bars": bars,
                     "up_rate": 0.15, "troughs": troughs}}


def test_a_horizon_never_rounds_down_to_a_zero_bar_forecast():
    # 10 minutes on 3600s bars is 0.167 bars. Truncating gives 0, and a 0-bar
    # horizon prices a bar against itself: a free, perfect, fictional edge.
    assert horizon_bars(10, 3600) == 1
    assert horizon_bars(720, 3600) == 12
    assert horizon_bars(720, 73) == 592
    with pytest.raises(ValueError):
        horizon_bars(720, 0)


def test_a_returned_window_actually_clears_its_own_purity_bound():
    # 200 bars up, then 200 bars down. A 300-bar window must reach 100 bars
    # back into the up stretch (up-rate 0.333) and cannot clear, so 200 is the
    # longest reachable length -- and it is allowed to start at 170, borrowing
    # 30 up bars, because 30/200 is exactly the bound. That is the contract:
    # the BOUND holds, not some guess about where the window starts.
    rates = [0.01] * 200 + [-0.01] * 200
    win = longest_pure_window(rates, want_up=False, purity=0.15, min_bars=60)
    assert win is not None
    assert win["up_rate"] <= 0.15
    assert win["bars"] == 200
    assert win["bars"] in LENGTH_GRID
    # And the window it names is really that long.
    assert win["end"] - win["start"] == win["bars"]


def test_no_pure_down_window_is_reported_when_the_tape_only_goes_up():
    rates = [0.01] * 400
    assert longest_pure_window(rates, want_up=False, purity=0.15,
                               min_bars=60) is None


def test_the_sweep_does_not_stop_at_the_first_length_that_fails():
    # Purity is not monotone in length. Bars 0-79 are 50% up, so an 80-bar
    # window starting at 0 FAILS a <=0.15 bound; a 150-bar window starting at
    # 80 passes. A sweep that abandoned the search at the first failing length
    # would return the 60-bar answer and under-report the reachable window.
    rates = ([0.01, -0.01] * 40) + ([-0.01] * 320)
    win = longest_pure_window(rates, want_up=False, purity=0.15, min_bars=60)
    assert win is not None
    assert win["bars"] >= 300, (
        "the longest pure DOWN window here is the 320-bar down stretch; "
        f"the sweep stopped at {win['bars']}")


def test_a_down_window_long_enough_to_place_sixty_trades_is_no_longer_down():
    # The finding itself, as an assertion: 1800 bars are needed at the DOWN
    # arm's measured 2/60 fire rate, and on a tape whose down stretch is only
    # 320 bars long no window of that size can clear the bound.
    rates = ([0.01, -0.01] * 40) + ([-0.01] * 320) + ([0.01] * 2000)
    needed_bars = int(round(60 / (2 / 60)))
    assert needed_bars == 1800
    win = longest_pure_window(rates, want_up=False, purity=0.15, min_bars=60)
    assert win is not None
    assert win["bars"] < needed_bars, (
        "if a pure DOWN window this long existed, the single-corpus arm would "
        "be powered and pooling would be unnecessary")


def test_the_plan_names_the_chain_so_the_command_it_prints_can_actually_run():
    # 0002_LINK-WETH.json exists under arbitrum AND under polygon. A plan that
    # carried only the file NAME printed a command that died on
    # FileNotFoundError before a single bar was read -- and that is the mild
    # failure. The severe one is below: two chains' windows look like one
    # corpus repeated.
    plan = probe_plan(_row(start=3000, bars=200), "down")
    assert plan is not None
    assert plan["corpus"].endswith(
        "data/historical_ohlcv/arbitrum/0002_LINK-WETH.json")
    assert plan["chain"] == "arbitrum"


def test_a_window_with_no_room_for_its_training_set_is_refused_not_shrunk():
    # A pure DOWN window at bar 300 cannot carry 600 training bars plus the
    # lookback in front of it. Returning it anyway would either raise inside
    # plan_windows or silently train on a shorter set, and a cell trained on
    # less data is not comparable with the cells it is pooled against.
    assert probe_plan(_row(start=300, bars=200), "down") is None
    ok = probe_plan(_row(start=TRAIN_BARS + LOOKBACK_BARS + 12, bars=200),
                    "down")
    assert ok is not None
    # And the training window it names never reaches into the held-out window:
    # train_end plus one horizon must still land at or before the window start.
    assert ok["train_end"] + ok["horizon"] <= TRAIN_BARS + LOOKBACK_BARS + 12


def test_the_same_pair_on_two_chains_is_not_two_independent_windows(capsys):
    # LINK against WETH is the same two assets whichever chain quotes it.
    # Pooling five chains' copies multiplies n by five and adds no independent
    # tape -- an n inflated exactly the way this item exists to stop.
    from scripts.omen_self_pool_power import print_plan
    rows = [_row(corpus=f"data/historical_ohlcv/{c}/0002_LINK-WETH.json",
                 chain=c, start=3000, bars=200)
            for c in ("arbitrum", "base", "optimism", "polygon", "ethereum")]
    print_plan(rows, "down")
    out = capsys.readouterr().out
    assert "4 windows dropped as repeats of a pair already counted" in out
    # One surviving window, so the cumulative trough count is 17 and not 85.
    assert "cumulative=17" in out
    assert "cumulative=85" not in out
