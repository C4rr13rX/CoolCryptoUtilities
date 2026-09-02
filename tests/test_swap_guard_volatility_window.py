"""The swap guard's volatility clause must score the trade it is refusing.

Link 9 (LIVE) had moved on from ``token_unresolved`` to a new refusal: every
one of the 12 live entries attempted in the 24h to 2026-09-02 04:47 was logged
as ``guard-blocked-live`` with ``reason=swap_guard:volatility``. Measured
against the actual rows the guard was reading:

  * ``fetch_market_samples_for(limit=360)`` returned **147 hours** of history
    for SPACEX-USDC and 144 for BSTONK-USDC, while the guard's own
    ``lookback_sec`` is 2 hours. Only the liquidity clause honoured it.
  * that query is ``ORDER BY ts DESC``, so ``np.diff`` walked the series
    backwards through time.
  * SPACEX-USDC's window held 233 ticks near 1.5e-9 and a single tick at
    524.37 -- eleven orders of magnitude apart, two different assets under one
    symbol. That one pair produced a return of 1.29e11 and the guard reported
    ``volatility = 6.6e10`` against a limit of 0.18.
  * the reading was scaled by ``sqrt(min(n, 60))``, i.e. expressed over a
    number of samples, so the same market read calmer on a sparser feed.

With the series corrected and the 0.18 threshold untouched, BSTONK-USDC reads
0.126 and SPACEX-USDC reads 0.0 -- the pairs were inside the operator's stated
risk appetite while being refused for six days.

These tests pin each defect separately, and pin that the clause still bites on
a pair that is genuinely violent.
"""
from __future__ import annotations

import time

from trading.swap_validator import SwapValidator

from tests.test_swap_guard_liquidity_basis import _StubDB, _validator


def _series(points, *, newest_first=True):
    """Rows shaped like ``fetch_market_samples_for`` output.

    ``points`` is ``[(age_sec, price), ...]``. The default ordering is the
    reverse-chronological one the database actually returns.
    """
    now = time.time()
    rows = [
        {"ts": now - age, "chain": "base", "symbol": "T-USDC",
         "price": price, "volume": 0.0}
        for age, price in points
    ]
    rows.sort(key=lambda r: r["ts"], reverse=newest_first)
    return rows


def _walk(prices, *, spacing=95.0, start_age=0.0):
    return [(start_age + i * spacing, p) for i, p in enumerate(prices)]


def test_volatility_is_measured_over_the_guards_own_lookback() -> None:
    """A week-old crash is not the volatility of a trade placed now."""
    calm = _walk([1.0000, 1.0005, 1.0002, 1.0006, 1.0003, 1.0007, 1.0004])
    # Six days back, the same pair doubled and halved repeatedly.
    wild = _walk([1.0, 2.0, 0.5, 3.0, 0.4, 2.5, 0.6], start_age=6 * 86400.0)
    validator = _validator(_StubDB([]))

    vol, measurable, diag = validator._estimate_volatility(_series(calm + wild))

    assert measurable is True
    # Only the in-window ticks were considered at all.
    assert diag["vol_window_samples"] == len(calm)
    assert vol < validator.max_volatility


def test_reverse_chronological_rows_give_the_same_answer_as_forward() -> None:
    """``ORDER BY ts DESC`` must not change what the market did."""
    points = _walk([1.00, 1.02, 1.01, 1.03, 1.02, 1.04, 1.03, 1.05])
    validator = _validator(_StubDB([]))

    newest_first, _, _ = validator._estimate_volatility(_series(points, newest_first=True))
    oldest_first, _, _ = validator._estimate_volatility(_series(points, newest_first=False))

    assert newest_first == oldest_first
    assert newest_first > 0.0


def test_a_denomination_artifact_is_dropped_not_scored_as_risk() -> None:
    """One tick eleven orders of magnitude out is a different asset.

    This is the SPACEX-USDC shape that produced ``volatility = 6.6e10``.
    """
    points = _walk([1.5e-9, 1.6e-9, 1.55e-9, 1.7e-9, 1.6e-9, 1.65e-9,
                    524.37, 1.6e-9, 1.58e-9, 1.62e-9])
    validator = _validator(_StubDB([]))

    vol, measurable, diag = validator._estimate_volatility(_series(points))

    assert measurable is True
    assert diag["vol_dropped_outliers"] == 1.0
    # Not merely "smaller" -- inside the limit, which 6.6e10 was not.
    assert vol < validator.max_volatility


def test_a_window_of_mixed_scales_is_refused_as_unmeasurable() -> None:
    """Too contaminated to read is its own verdict, not "too volatile"."""
    # Half the window in each denomination: this is not one price series.
    points = _walk([1.5e-9, 900.0, 1.6e-9, 850.0, 1.55e-9, 880.0,
                    1.7e-9, 870.0, 1.6e-9, 860.0])
    db = _StubDB(_series(points))
    validator = _validator(db)

    vol, measurable, _diag = validator._estimate_volatility(_series(points))
    assert measurable is False
    assert vol == 0.0

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=1.0, price=1.6e-9, volume=0.0,
    )
    assert allowed is False
    assert "volatility_unmeasurable" in reasons
    # The honest reason, not the one that blames the market.
    assert "volatility" not in reasons
    assert metrics["volatility_measurable"] == 0.0


def test_a_hole_in_the_feed_is_not_differenced_as_one_tick() -> None:
    """A 14-hour gap between prints was being read as a single tick's move."""
    dense = _walk([1.00, 1.001, 1.002, 1.001, 1.003, 1.002])
    # One print from long before, still inside the lookback, after a big hole.
    stale = [(6000.0, 0.55)]
    validator = _validator(_StubDB([]))
    validator.lookback_sec = 7200

    vol, measurable, diag = validator._estimate_volatility(_series(dense + stale))

    assert measurable is True
    assert diag["vol_dropped_gaps"] >= 1.0
    assert vol < validator.max_volatility


def test_the_reading_does_not_depend_on_how_much_history_was_stored() -> None:
    """``sqrt(min(n, 60))`` expressed risk over a row count, not a duration.

    The same market at the same cadence, with twice as many rows retained,
    was reported as sqrt(2) more volatile -- a property of the table, not of
    the pair. The horizon is now a span of time, so the two agree.

    (A market that genuinely moves twice as fast *should* read higher; that is
    the cadence changing, which this test deliberately holds fixed.)
    """
    validator = _validator(_StubDB([]))
    validator.lookback_sec = 86400

    zigzag = [1.0 + 0.01 * ((-1) ** i) for i in range(120)]
    short, _, _ = validator._estimate_volatility(_series(_walk(zigzag[:45], spacing=95.0)))
    long, _, _ = validator._estimate_volatility(_series(_walk(zigzag, spacing=95.0)))

    assert short > 0.0
    assert abs(short - long) < 0.02 * max(short, long)


def test_a_genuinely_violent_pair_is_still_refused() -> None:
    """The clause has to keep biting, or fixing it would just be removing it."""
    prices = [1.0]
    for i in range(30):
        prices.append(prices[-1] * (1.25 if i % 2 else 0.8))
    db = _StubDB(_series(_walk(prices)))
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=1.0, price=prices[-1], volume=0.0,
    )

    assert metrics["volatility"] > validator.max_volatility
    assert allowed is False
    assert "volatility" in reasons


def test_too_little_history_is_not_a_violation() -> None:
    """Unmeasured is not a violation -- the same rule the liquidity clause keeps."""
    validator = _validator(_StubDB([]))
    vol, measurable, _ = validator._estimate_volatility(_series(_walk([1.0, 1.01])))
    assert measurable is True
    assert vol == 0.0
