"""The perfect-oracle ceiling must charge BOTH cost legs, not just the percentage.

THE BUG THIS PREVENTS. ``head_vs_realised_census.py --horizon-table`` printed a
header reading "cost basis 0.3187% of notional + 0.004047 fixed, measured from
receipts" and then computed its two cost-bearing columns as ``mean - 0.3187``
and ``count(v > 0.3187)``. The fixed leg was quoted and never charged. At the
$5 clip this system actually trades, that leg is another 0.0809% of notional --
a quarter as much again as the whole proportional cost -- so every row of the
published table was 0.0809 points too generous, and the 15-minute row was
published as ``+0.0198`` when the all-in number is negative.

It also accepted ``--clip``, ``--pct-cost`` and ``--fixed-cost`` and ignored all
three in the table path, so an operator checking "what if the cost halved?"
would have been shown the unchanged default table and believed it.

The tests below pin the arithmetic and the behaviour: a tape whose typical move
lands BETWEEN the proportional cost and the all-in cost must read NEGATIVE.
Against the old code every one of them fails.
"""

from __future__ import annotations

import argparse
import re

import pytest

from scripts.head_vs_realised_census import (
    _print_horizon_table,
    main,
    total_cost_pct,
    window_regime,
)

# The measured receipt cost this repo bills a round trip at, and the clip the
# live lane actually sends. Keep these literal: the point of the test is that
# the two legs combine, not that they match some constant elsewhere.
PCT_COST = 0.3187
FIXED_COST = 0.004047
CLIP = 5.0
ALL_IN = 0.3187 + (0.004047 / 5.0) * 100.0  # 0.39964


def _args(**overrides):
    base = dict(
        pct_cost=PCT_COST,
        fixed_cost=FIXED_COST,
        clip=CLIP,
        max_abs_return=0.20,
        tolerance_sec=120.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _sawtooth(start_ts: float, bars: int, step_sec: float, swing: float):
    """A stream whose forward return over exactly ``step_sec`` is ~``swing``.

    Prices alternate between ``p`` and ``p * (1 + swing)``, so a tick's forward
    price one step out is always the other rail and the |return| is the swing on
    the way up and swing/(1+swing) on the way down.
    """
    times = [start_ts + i * step_sec for i in range(bars)]
    prices = [100.0 * (1.0 + swing) if i % 2 else 100.0 for i in range(bars)]
    return times, prices


def _five_minute_row(capsys) -> tuple[float, float, float]:
    """(median, pct_over_cost, oracle_net) parsed from the printed 5min row."""
    text = capsys.readouterr().out
    match = re.search(
        r"^\s*5min\s+(\d+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([+-][\d.]+)\s*$",
        text,
        re.MULTILINE,
    )
    assert match is not None, f"no 5min row in:\n{text}"
    return float(match.group(2)), float(match.group(3)), float(match.group(5))


def test_the_fixed_leg_is_charged_and_the_clip_is_what_amortises_it():
    """0.004047 over a $5 clip is 0.0809% of notional, not zero."""
    assert total_cost_pct(PCT_COST, FIXED_COST, CLIP) == pytest.approx(ALL_IN)
    assert total_cost_pct(PCT_COST, FIXED_COST, CLIP) > PCT_COST

    # 25x the capital buys back only the fixed leg, never the proportional one.
    big = total_cost_pct(PCT_COST, FIXED_COST, 125.0)
    assert big < total_cost_pct(PCT_COST, FIXED_COST, CLIP)
    assert big > PCT_COST  # it amortises toward the rate, it never goes below it

    with pytest.raises(ValueError):
        total_cost_pct(PCT_COST, FIXED_COST, 0.0)


def test_a_move_between_the_two_cost_legs_reads_NEGATIVE_not_positive(capsys):
    """The failure the old table shipped: 0.35% is above 0.3187% and below 0.3996%."""
    swing = 0.0035  # 0.35%: clears the rate, does NOT clear the all-in cost
    times, prices = _sawtooth(1_000_000.0, 80, 300.0, swing)
    stream = {"TEST": (times, prices)}
    preds = [{"symbol": "TEST", "ts": t} for t in times[:-1]]

    _print_horizon_table(preds, stream, _args(), times[0], times[-1])
    median, pct_over, oracle_net = _five_minute_row(capsys)

    # The tape itself is unchanged -- only the bill is.
    assert median == pytest.approx(0.349, abs=0.01)
    # Under the old rule this printed ~+0.031 and 100% of ticks "over cost".
    assert oracle_net < 0.0, "a move under the all-in cost must not read positive"
    assert oracle_net == pytest.approx(median - ALL_IN, abs=0.01)
    assert pct_over == 0.0, "no tick clears a cost floor above the whole swing"


def test_the_table_honours_clip_and_cost_flags_instead_of_module_constants(capsys):
    """--clip / --pct-cost / --fixed-cost were accepted and silently ignored."""
    swing = 0.0035
    times, prices = _sawtooth(1_000_000.0, 80, 300.0, swing)
    stream = {"TEST": (times, prices)}
    preds = [{"symbol": "TEST", "ts": t} for t in times[:-1]]

    # A clip 25x larger amortises the fixed leg almost away; the SAME tape must
    # now clear, which is only possible if the flag reaches the arithmetic.
    _print_horizon_table(preds, stream, _args(clip=125.0), times[0], times[-1])
    _, pct_over_big_clip, net_big_clip = _five_minute_row(capsys)

    _print_horizon_table(preds, stream, _args(), times[0], times[-1])
    _, pct_over_small_clip, net_small_clip = _five_minute_row(capsys)

    assert net_big_clip > net_small_clip
    assert pct_over_big_clip == 100.0 and pct_over_small_clip == 0.0

    # And halving the proportional rate has to move it too.
    _print_horizon_table(preds, stream, _args(pct_cost=0.15), times[0], times[-1])
    _, _, net_cheap = _five_minute_row(capsys)
    assert net_cheap > net_small_clip


def test_the_header_states_the_all_in_cost_and_the_window_regime(capsys):
    times, prices = _sawtooth(1_000_000.0, 80, 300.0, 0.0035)
    stream = {"TEST": (times, prices)}
    preds = [{"symbol": "TEST", "ts": t} for t in times[:-1]]

    _print_horizon_table(preds, stream, _args(), times[0], times[-1])
    text = capsys.readouterr().out
    assert "ALL IN" in text
    assert f"{ALL_IN:.4f}%" in text
    assert "REGIME" in text
    # A reader must be told the sign was checked on one tape only.
    assert "--end-hours-ago" in text


def test_the_regime_is_measured_over_the_SAMPLED_symbols_not_the_frozen_feed():
    """Most market_stream symbols hold a seed price; averaging them reads FLAT."""
    rising_t = [1_000_000.0 + i * 300.0 for i in range(40)]
    rising_p = [100.0 * (1.0 + 0.0005 * i) for i in range(40)]
    frozen_p = [100.0] * 40
    movers = [f"MOVER{i}" for i in range(6)]
    stream = {
        **{name: (rising_t, rising_p) for name in movers},
        **{f"FROZEN{i}": (rising_t, frozen_p) for i in range(50)},
    }
    preds = [{"symbol": name, "ts": t} for name in movers for t in rising_t]

    verdict = window_regime(preds, stream, rising_t[0], rising_t[-1], 0.20)
    assert verdict["regime"] == "UP"
    assert verdict["symbols"] == 6, "frozen symbols nobody sampled must not vote"
    # Over the whole feed the same window reads FLAT: 50 seed prices drown 6 movers.
    everyone = [{"symbol": name, "ts": rising_t[0]} for name in stream]
    assert window_regime(everyone, stream, rising_t[0], rising_t[-1], 0.20)["regime"] == "FLAT"

    # An implausible drift is dropped by the same guard the table uses, so a
    # single 1e6 row cannot manufacture a regime the way a mean over it would.
    for name in movers:
        stream[name] = (rising_t, [100.0] * 39 + [1e6])
    assert window_regime(preds, stream, rising_t[0], rising_t[-1], 0.20)["regime"] == "UNKNOWN"


def test_a_second_regime_window_is_selectable_at_all():
    """Without --end-hours-ago the census can only ever re-read the same tape."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--end-hours-ago", "-1"])
    assert "end-hours-ago" in str(excinfo.value)
