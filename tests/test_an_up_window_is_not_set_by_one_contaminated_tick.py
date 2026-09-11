"""A window is not UP because one endpoint tick was on the wrong denomination.

THE FAILURE THIS PREVENTS. ``scripts/market_stream_drift_scan.py`` exists to
find an UP window to re-run the horizon table on, because every horizon number
measured so far came from a FLAT-to-DOWN tape (-0.184% and -0.199% median
per-symbol drift) and a long-only cost finding tested in one direction is the
error this repo has already paid for twice.

The naive way to write it is ``last_price / first_price - 1``. That is exactly
the shape that manufactures a fake edge here: this feed has carried two price
regimes under one ticker (see ``feed-two-regime-census``, 47 of 132 symbols),
so ONE endpoint tick on the other denomination sets the whole symbol's drift,
and a handful of those across the symbol set sets the window's. The scan would
then hand somebody an ``--end-hours-ago`` that points at a contamination
artifact, and the horizon table would be re-run there and believed.

So the scan takes the MEDIAN of a block of ticks at each end, and drops any
symbol whose in-window max/min price ratio is implausible. These tests assert
BOTH, and the first one FAILS against an endpoint-pair implementation: the
series it builds is dead flat apart from a single 1000x tick at the end.
"""

from __future__ import annotations

import pytest

from scripts.market_stream_drift_scan import scan, symbol_drift


def _flat_series(t0: float, n: int, price: float, step: float = 60.0):
    return [(t0 + i * step, price) for i in range(n)]


def test_one_contaminated_endpoint_tick_does_not_set_the_drift():
    """A flat symbol with one 1000x final tick reads FLAT, not up 100000%."""
    series = _flat_series(0.0, 100, 10.0)
    series[-1] = (series[-1][0], 10_000.0)

    drift, reason = symbol_drift(
        series, lo=0.0, hi=6000.0, min_ticks=20, min_span_frac=0.6, max_price_ratio=50.0
    )

    # max/min is 1000x, well past the guard, so the symbol is dropped outright.
    assert drift is None
    assert reason == "price_ratio"


def test_a_contaminated_tick_inside_the_guard_still_cannot_set_the_drift():
    """Even a 10x tick that clears the ratio guard is outvoted by the median."""
    series = _flat_series(0.0, 100, 10.0)
    series[-1] = (series[-1][0], 100.0)  # 10x: inside max_price_ratio=50

    drift, reason = symbol_drift(
        series, lo=0.0, hi=6000.0, min_ticks=20, min_span_frac=0.6, max_price_ratio=50.0
    )

    assert reason == "ok"
    # Endpoint arithmetic would read +900%. The median of the last tenth is 10.0.
    assert drift == pytest.approx(0.0, abs=1e-9)


def test_a_genuine_move_is_still_reported():
    """The guard must not flatten a real move -- prove it reads a 10% rise."""
    series = _flat_series(0.0, 50, 10.0) + _flat_series(3000.0, 50, 11.0)

    drift, reason = symbol_drift(
        series, lo=0.0, hi=6000.0, min_ticks=20, min_span_frac=0.6, max_price_ratio=50.0
    )

    assert reason == "ok"
    assert drift == pytest.approx(0.10, abs=1e-9)


def test_a_symbol_whose_ticks_stop_early_is_not_scored():
    """A frozen/dark symbol's stale last price is not a forward drift."""
    series = _flat_series(0.0, 30, 10.0)  # 30 minutes of a 100-minute window

    drift, reason = symbol_drift(
        series, lo=0.0, hi=6000.0, min_ticks=20, min_span_frac=0.6, max_price_ratio=50.0
    )

    assert drift is None
    assert reason == "short_span"


def test_the_window_statistic_is_the_median_not_the_mean():
    """One symbol that doubled must not make a flat window read UP."""
    now = 100_000.0
    hi = now
    lo = hi - 3600.0
    by_symbol = {}
    for i in range(10):
        by_symbol[f"FLAT{i}-USDC"] = _flat_series(lo + 1.0, 40, 10.0, step=80.0)
    by_symbol["MOON-USDC"] = _flat_series(lo + 1.0, 20, 10.0, step=80.0) + _flat_series(
        lo + 1700.0, 20, 20.0, step=80.0
    )

    windows = scan(
        by_symbol,
        now=now,
        window_hours=1.0,
        step_hours=1.0,
        max_end_hours_ago=1.0,
        min_ticks=20,
        min_symbols=5,
        min_span_frac=0.6,
        max_price_ratio=50.0,
    )

    assert len(windows) == 1
    w = windows[0]
    assert w["symbols"] == 11
    assert w["median_drift"] == pytest.approx(0.0, abs=1e-9)
    # The mean is dragged up by the one doubler -- which is why it is not the statistic.
    assert w["mean_drift"] > 0.08


def test_the_horizon_table_can_take_its_rows_from_the_tape():
    """The horizon table reads no prediction field, so it must not need one.

    Measured pass 112: fed prediction snapshots, the table printed NO ROWS for
    the 24h window ending 168h ago -- n blank at every horizon -- because the
    head emitted no usable snapshot that far back. The window itself held
    15,497 priced ticks. Gating a pure-tape measurement behind the model's own
    history is what kept every horizon number confined to the recent window.
    """
    from scripts.head_vs_realised_census import tape_rows

    stream = {
        "AAA-USDC": ([10.0, 20.0, 30.0, 99.0], [1.0, 1.0, 1.0, 1.0]),
        "BBB-USDC": ([15.0, 25.0], [2.0, 2.0]),
    }

    rows = tape_rows(stream, lo=10.0, hi=40.0)

    assert [r["symbol"] for r in rows] == [
        "AAA-USDC", "BBB-USDC", "AAA-USDC", "BBB-USDC", "AAA-USDC",
    ]
    assert [r["ts"] for r in rows] == [10.0, 15.0, 20.0, 25.0, 30.0]
    # ts=99.0 is outside the window and must not be priced forward from.
    assert all(r["ts"] < 40.0 for r in rows)
    # No direction_prob: anything scoring the HEAD must keep using preds, and
    # a caller that confuses them gets a KeyError, not a silent wrong answer.
    assert all("direction_prob" not in r for r in rows)
