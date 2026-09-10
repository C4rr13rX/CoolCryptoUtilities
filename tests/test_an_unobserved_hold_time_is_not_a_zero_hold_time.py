"""The hold-time split must not manufacture its own headline.

scripts/hold_time_edge reports that the tradeable book is positive inside 15
minutes and loses everything outside it. Two bookkeeping slips would produce
that headline whether or not it were true, and this pins both:

  1. COUNTING AN UNOBSERVED HOLD AS AN INSTANT ONE. A trip whose recovered path
     is empty (the entry was the last tick before the exit) has a hold time of
     UNKNOWN, not zero. Bucketing it at 0 minutes drops it straight into the
     "under 5 min" bucket -- the one carrying the positive gross -- on no
     evidence at all. It must be excluded from both sides of the horizon split.

  2. DIVIDING THE LOSS SHARE BY THE NET. ``loss_share`` is each bucket's share
     of the total LOSS, so it must divide by the sum of the negative buckets. If
     it divided by the net book (which is near zero and can flip sign) the
     shares would explode or invert.

Also pinned: the bucket edges are half-open, so a trip lands in exactly one, and
the horizon split is drawn at STALE_EXIT_MINS rather than at a literal 15.
"""

from __future__ import annotations

import pytest

from scripts.hold_time_edge import STALE_EXIT_MINS, hold_time_edge


def _r(held, ticks, gross, notional=5.0, symbol="T-USDC"):
    return {"symbol": symbol, "held_mins": held, "ticks": ticks,
            "cgross": gross, "notional": notional, "ts": 1000.0,
            "strategy_id": "s"}


def test_an_unobserved_hold_is_excluded_from_the_horizon_split_not_called_instant():
    # The 0-length path carries a big WIN. If it were treated as a 0-minute
    # hold it would land inside the horizon and inflate exactly the number
    # this report exists to state.
    rows = [_r(0.0, 0, +9.0),
            _r(3.0, 4, +0.10),
            _r(40.0, 4, -0.50)]
    rep = hold_time_edge(rows=rows)
    assert rep["within_horizon"]["trips"] == 1
    assert rep["within_horizon"]["gross"] == pytest.approx(+0.10)
    assert rep["outlived_horizon"]["trips"] == 1
    # The +9.0 appears in neither side.
    total = (rep["within_horizon"]["gross"] + rep["outlived_horizon"]["gross"])
    assert total == pytest.approx(-0.40)


def test_loss_share_divides_by_the_total_loss_not_by_the_net_book():
    # Net book is +0.10 - 0.30 - 0.10 = -0.30, but total LOSS is -0.40.
    rows = [_r(3.0, 4, +0.10), _r(40.0, 4, -0.30), _r(300.0, 9, -0.10)]
    rep = hold_time_edge(rows=rows)
    shares = {b["label"]: b["loss_share"] for b in rep["buckets"]}
    assert shares["15-60 min"] == pytest.approx(75.0)
    assert shares["over 4 hours"] == pytest.approx(25.0)
    # A winning bucket contributes no loss share.
    assert shares["under 5 min"] == pytest.approx(0.0)


def test_a_trip_lands_in_exactly_one_bucket_at_every_edge():
    edges = [0.0, 5.0, 15.0, 60.0, 240.0]
    rows = [_r(e, 1, -0.01) for e in edges] + [_r(1e6, 1, -0.01)]
    rep = hold_time_edge(rows=rows)
    assert sum(b["trips"] for b in rep["buckets"]) == len(rows)


def test_the_horizon_split_follows_stale_exit_mins():
    just_inside = STALE_EXIT_MINS - 0.01
    just_outside = STALE_EXIT_MINS + 0.01
    rep = hold_time_edge(rows=[_r(just_inside, 3, +0.20),
                               _r(just_outside, 3, -0.20)])
    assert rep["within_horizon"]["gross"] == pytest.approx(+0.20)
    assert rep["outlived_horizon"]["gross"] == pytest.approx(-0.20)
    assert rep["outlived_pct"] == pytest.approx(50.0)


def test_the_tick_rate_is_per_minute_held_and_skips_zero_length_holds():
    # 15 ticks over 30 minutes is 0.5/min. The 0-minute row must not divide.
    rep = hold_time_edge(rows=[_r(30.0, 15, -0.10), _r(0.0, 0, -0.10)])
    slow = [b for b in rep["buckets"] if b["label"] == "15-60 min"]
    assert slow and slow[0]["median_tick_rate"] == pytest.approx(0.5)


def test_the_longest_hold_is_reported_so_a_dark_feed_cannot_hide_in_an_average():
    rep = hold_time_edge(rows=[_r(5.0, 5, -0.10),
                               _r(1064.8, 13, -0.15, symbol="CRV-USDC")])
    assert rep["longest"]["symbol"] == "CRV-USDC"
    assert rep["longest"]["ticks"] == 13
