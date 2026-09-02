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


def test_too_little_history_reads_as_unmeasured_not_as_calm() -> None:
    """Unmeasured is not a violation -- but it is not a clean bill of health.

    This test previously asserted `measurable is True` with `vol == 0.0`, on the
    stated rule that "unmeasured is not a violation -- the same rule the
    liquidity clause keeps". The principle is right; the citation was not. The
    liquidity clause keeps `unmeasured` DISTINCT from `measured and fine`:

        if liquidity_ratio is not None:
            if liquidity_ratio > self.max_liquidity_ratio: ... "liquidity"
        elif trade_usd > self.unknown_liquidity_max_usd:  ... "liquidity_unmeasured"

    -- when it has no basis it says so and falls back to an absolute bound. The
    volatility path did the opposite: `(0.0, True)` reports the number as
    successfully measured AND at its lowest possible value, so
    `volatility > max_volatility` is trivially false and the gate raises no
    objection at all. Unmeasured was not merely "not a violation", it was the
    guard's best possible grade.

    Measured 2026-09-02 across the 40 most active symbols, 30 cleared the
    volatility gate on a number that was never computed -- VIRTUAL-USDC and
    MTGA-USDC on ZERO prints in the 2h window, BSTONK-USDC on one.

    So the reading is now `measurable=False`, which routes to the
    `volatility_unmeasurable` refusal the guard already has for exactly this.
    It still is not scored as a violation: `vol` stays 0.0 and the
    `volatility > threshold` branch is never reached.
    """
    validator = _validator(_StubDB([]))
    vol, measurable, diag = validator._estimate_volatility(_series(_walk([1.0, 1.01])))
    assert measurable is False
    assert vol == 0.0
    assert diag.get("vol_insufficient_samples") == 1.0


def test_unmeasurable_volatility_refuses_the_swap() -> None:
    """A pair we have seen twice must not authorise real money."""
    prices = [1.0, 1.01]
    db = _StubDB(_series(_walk(prices)))
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=1.0, price=prices[-1], volume=0.0,
    )

    assert allowed is False
    assert "volatility_unmeasurable" in reasons
    # Refused for being unreadable, not for being violent: the threshold
    # comparison never ran.
    assert metrics["volatility_measurable"] == 0.0
    assert "volatility" not in reasons


def test_a_measurable_series_still_passes() -> None:
    """The refusal above must not have closed the gate on real data.

    Six symbols still measured cleanly when this landed (BASECAT, CBXRP, CBBTC,
    AERO, COMP, UMIA) -- including the three atf_static trades most -- so the
    live lane stays open to the pairs the feed actually resolves.
    """
    prices = [1.0, 1.002, 1.001, 1.003, 1.0025, 1.004, 1.0035, 1.005]
    validator = _validator(_StubDB([]))

    vol, measurable, _ = validator._estimate_volatility(_series(_walk(prices)))
    assert measurable is True
    assert 0.0 < vol <= validator.max_volatility


def test_a_trade_price_at_the_wrong_scale_is_refused() -> None:
    """The entry price itself must belong to the same series as the history.

    Four ghost outcomes on 2026-08-26 crossed a denomination boundary between
    entry and exit -- AERO exiting at 1.14 when AERO is $0.478, COMP entering
    at 42.82 when COMP is $19 -- and booked +174% and +161% as wins. Those
    stayed harmless only because the swap guard refused every live entry for
    unrelated reasons. Fixing the volatility clause opened that gate, so the
    check has to be explicit now.

    The bound is 3.0x, chosen from the measured distribution: over 11,249
    ticks scored against their trailing 2h median, p99 is 1.61x and the
    distribution is empty between 3x and 5x.
    """
    history = _walk([1.5e-9, 1.6e-9, 1.55e-9, 1.7e-9, 1.6e-9, 1.65e-9, 1.58e-9])
    db = _StubDB(_series(history))
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="SPACEX-USDC", route=["SPACEX", "USDC"],
        trade_size=1.0, price=524.37, volume=0.0,   # the contaminated quote
    )

    assert allowed is False
    assert "price_off_scale" in reasons
    assert metrics["price_scale_offset"] > 1e10


def test_the_two_to_three_x_blind_spot_is_pinned_not_pretended_away() -> None:
    """COMP-USDC's artifacts sit at 2.2-2.9x and this clause does NOT catch them.

    COMP prints both 42.82 and ~19.13; the offset is 2.24x. A genuine
    two-hour move on a microcap lives in that same band, so no threshold on
    price alone separates them, and lowering the bound to catch COMP would
    refuse real moves without becoming a measurement.

    Pinned deliberately: if someone later tightens the bound, this test should
    make them state what new evidence justified it rather than discovering the
    tradeoff in production.
    """
    history = _walk([19.1, 19.2, 19.0, 19.3, 19.1, 19.4, 19.2])
    db = _StubDB(_series(history))
    validator = _validator(db)

    _allowed, metrics, reasons = validator.validate(
        symbol="COMP-USDC", route=["COMP", "USDC"],
        trade_size=0.02, price=42.82, volume=0.0,
    )

    assert 2.0 < metrics["price_scale_offset"] < 3.0
    assert "price_off_scale" not in reasons


def test_an_ordinary_price_moves_through() -> None:
    """A price that simply moved is not an artifact."""
    history = _walk([0.478, 0.479, 0.477, 0.480, 0.478, 0.481, 0.479, 0.478])
    db = _StubDB(_series(history))
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="AERO-USDC", route=["AERO", "USDC"],
        trade_size=0.7, price=0.502, volume=0.0,   # +5%, a real move
    )

    assert "price_off_scale" not in reasons
    assert allowed is True
    assert metrics["price_scale_offset"] < 1.1


def test_too_little_history_does_not_refuse_the_price() -> None:
    """Unmeasured is not a violation -- the rule the whole guard keeps."""
    validator = _validator(_StubDB([]))
    assert validator._price_scale_offset(_series(_walk([1.0, 1.01])), 1.0) is None


def test_a_frozen_feed_is_not_scored_as_the_calmest_market() -> None:
    """A stuck price must not earn the guard's best possible grade.

    This is the hole the four fixes above opened. Correcting the window turned
    SPACEX-USDC -- 35 ticks over 100 minutes, every one of them exactly
    1.52588956496421e-09 -- from ``volatility = 6.6e10`` into ``0.0``, i.e.
    from the most-refused pair on the book into the least-refused one, while
    the live path was actively proposing entries on it.
    """
    frozen = _walk([1.52588956496421e-09] * 35, spacing=172.0)
    db = _StubDB(_series(frozen))
    validator = _validator(db)

    vol, measurable, diag = validator._estimate_volatility(_series(frozen))

    assert vol == 0.0
    assert measurable is False, "a price that never moved is not low volatility"
    assert diag["vol_frozen"] == 1.0

    allowed, metrics, reasons = validator.validate(
        symbol="SPACEX-USDC", route=["SPACEX", "USDC"],
        trade_size=1.0, price=1.52588956496421e-09, volume=0.0,
    )
    assert allowed is False
    # Named for the fault that is actually there: the feed, not the market,
    # and not the mixed-scale case which is repaired somewhere else.
    assert "feed_frozen" in reasons
    assert "volatility" not in reasons
    assert "volatility_unmeasurable" not in reasons
    assert metrics["volatility_measurable"] == 0.0


def test_a_briefly_flat_price_is_not_called_frozen() -> None:
    """Unmeasured is not a violation -- the rule the whole guard keeps.

    A handful of identical prints over a couple of minutes is a quiet feed,
    not a stuck one. WOJAK-USDC sat at one price for 507s on 2026-09-02 and is
    deliberately below the bound: the verdict needs an observation behind it.
    """
    brief = _walk([8.26609e-07] * 6, spacing=45.0)
    validator = _validator(_StubDB([]))

    vol, measurable, diag = validator._estimate_volatility(_series(brief))

    assert vol == 0.0
    assert measurable is True
    assert "vol_frozen" not in diag


def test_a_feed_whose_only_move_was_dropped_is_still_frozen() -> None:
    """``max != min`` is not the same question as "did the scored series move".

    MTGA-USDC, measured 2026-09-02: 26 ticks over 5694s at a 128s median
    cadence, holding exactly two prices. Because the window held two prices the
    frozen clause passed it -- but the single transition sat across a 1378s
    hole, which the gap clause discards (correctly; a 1378s hole is not one
    tick's move). What remained were 23 returns of exactly 0.0, so the guard
    reported volatility 0.0 *and* measurable: its best possible grade, awarded
    to a feed that never reported a move.

    The two clauses have to agree, and the one that decides is the series the
    threshold is actually compared against.
    """
    # One price for the first stretch, a hole, then a different price.
    before = _walk([8.225e-06] * 20, spacing=128.0)
    after = _walk([8.55727072348181e-06] * 6, spacing=128.0,
                  start_age=20 * 128.0 + 1378.0)
    points = before + after
    db = _StubDB(_series(points))
    validator = _validator(db)

    vol, measurable, diag = validator._estimate_volatility(_series(points))

    assert vol == 0.0
    # The disguise: the window genuinely holds two distinct prices.
    assert len({p for _, p in points}) == 2
    assert diag["vol_dropped_gaps"] >= 1.0
    assert measurable is False, "every scored return was zero"
    assert diag["vol_frozen"] == 1.0

    allowed, metrics, reasons = validator.validate(
        symbol="MTGA-USDC", route=["MTGA", "USDC"],
        trade_size=1.0, price=8.55727072348181e-06, volume=0.0,
    )
    assert allowed is False
    assert "feed_frozen" in reasons
    assert metrics["volatility_measurable"] == 0.0


def test_a_sparse_flat_window_is_still_only_unmeasured() -> None:
    """The second frozen test must keep the same bounds as the first.

    Four of the five symbols that scored 0.0-and-measurable on 2026-09-02
    (CBHYPE, CBLTC, GARFI, WOJAK) had 2 to 6 returns over 163-800s. That is not
    enough observation to call a feed stuck, and widening the frozen verdict to
    cover them would turn "we barely looked" into a violation.
    """
    brief = _walk([48.98] * 3, spacing=82.0)   # CBLTC-USDC's actual shape
    validator = _validator(_StubDB([]))

    vol, measurable, diag = validator._estimate_volatility(_series(brief))

    assert vol == 0.0
    assert measurable is True
    assert "vol_frozen" not in diag


def test_a_pair_that_moves_at_all_is_still_measurable() -> None:
    """One real tick of movement is enough to make the window a measurement.

    The frozen clause keys on exact equality across the window, so it must not
    swallow a genuinely calm pair that nonetheless prints different numbers.
    """
    nearly_flat = _walk([1.0000, 1.0000, 1.0000, 1.0001, 1.0000,
                         1.0000, 1.0000, 1.0000], spacing=200.0)
    validator = _validator(_StubDB([]))

    vol, measurable, diag = validator._estimate_volatility(_series(nearly_flat))

    assert measurable is True
    assert "vol_frozen" not in diag
    assert vol < validator.max_volatility
