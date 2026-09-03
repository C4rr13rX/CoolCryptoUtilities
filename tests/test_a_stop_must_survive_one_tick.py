"""A stop-loss only bounds a loss if one tick cannot step over it.

Measured 2026-09-03 on the live book. Seven live trades, net -$0.140460, of
which ONE exit -- BSTONK-USDC, ``stop_loss:-0.1840`` -- realised -$0.142865.
That single loss is 101.7% of all live P/L and 12.7x the sum of the three
wins (+$0.011281). The standing order is that this must be impossible: "a
single losing exit must never exceed the sum of the wins; if it can, the
stop-loss mechanism is broken -- fix the mechanism, not the threshold."

The previous pass (7147339) found one reason the stop did not fire: every
sample carrying an entry directive was routed into the entry path, so a held
position was invisible to its own bracket for 72 of the 73 samples in its
life. That fix is real and it is not enough. Its own measurement says the
first feed sample after entry was ALREADY -13.08%, so a stop permitted to
look on every sample would still have exited 8.7x past a 1.5% stop.

The remaining defect is upstream of the exit entirely: the guard let the
position be opened on a feed against which a 1.5% stop is not enforceable.

  * The swap guard scored BSTONK's volatility at **0.0412 against its 0.18
    limit** and ALLOWED the entry (metrics row, ts 1788454873.5276577,
    strategy_id=atf_static). It was not a threshold that was too loose.
  * ``_estimate_volatility`` returns ``std(returns) * sqrt(steps)`` -- a
    symmetric dispersion scaled to an hour. It answers "how far does this
    pair wander over the horizon".
  * A stop is not enforced over an hour. It is enforced between two
    consecutive looks at the feed. The governing quantity is how far ONE tick
    can jump, and on a fat-tailed series the two numbers are unrelated:
    BSTONK's median tick move is 0.032% while its p99 is 6.975% -- a 218x
    spread that a standard deviation, dominated by the 99 quiet ticks,
    cannot express.

Scored through the guard's own pipeline (its 7200s lookback, its outlier
filter, its gap filter) over the trailing day, the statistic splits the book
exactly the way the live results did -- and note that BOTH refusals pass the
volatility clause, so nothing else in the guard was catching them:

    symbol           volatility  (limit .18)   p99 tick jump  (stop .015)
    CBBTC-USDC           0.0028   pass              0.1281%   allow   won
    AERO-USDC            0.0015   pass              0.0860%   allow   won
    CBETH-USDC           0.0121   pass              0.8250%   allow   won
    BASEPEPE-USDC        0.0648   pass              3.9059%   REFUSE
    BSTONK-USDC          0.0712   pass              4.5896%   REFUSE  -18.40%
    BASECAT-USDC         0.1029   pass              7.7876%   REFUSE  unsellable

Nothing was fitted to those outcomes. The statistic asks one question -- can
a single tick clear the stop -- and the outcomes agree with it.

These tests pin: the refusal fires on a fat tail the dispersion bound passes;
it does NOT fire on a tight feed; it is read off the same filtered series as
the volatility beside it, so contamination cannot fabricate a tail; and the
distance it judges is the same number trading/triggers.py will enforce.
"""
from __future__ import annotations

import time

import pytest

from trading.swap_validator import SwapValidator, _stop_loss_pct
from trading.triggers import evaluate_long_triggers

from tests.test_swap_guard_liquidity_basis import _StubDB, _validator


def _feed(points):
    """Rows shaped like ``fetch_market_samples_for``: newest first."""
    now = time.time()
    rows = [
        {"ts": now - age, "chain": "base", "symbol": "T-USDC",
         "price": price, "volume": 0.0}
        for age, price in points
    ]
    rows.sort(key=lambda r: r["ts"], reverse=True)
    return rows


def _fat_tailed(*, quiet_ticks=180, jumps=6, spacing=14.0):
    """BSTONK's shape: mostly flat, punctuated by single-tick gaps.

    Deliberately built so the horizon-scaled standard deviation stays modest
    while individual ticks step several percent -- which is the real series,
    not a caricature: 819 BSTONK ticks over the trailing day have a median
    move of 0.032% and a maximum of 27.478%.
    """
    prices = []
    price = 1.0
    for i in range(quiet_ticks):
        # A tiny alternating wobble: enough that the feed is not "frozen".
        price *= 1.0002 if i % 2 else 0.9998
        prices.append(price)
        if jumps and i and i % (quiet_ticks // (jumps + 1)) == 0:
            # One tick that steps ~5%, the way a thin book actually prints.
            price *= 1.05 if (i // 10) % 2 else 0.95
            prices.append(price)
    return _feed([(i * spacing, p) for i, p in enumerate(reversed(prices))])


def _tight(*, ticks=180, spacing=14.0):
    """CBBTC's shape: a real book, moving in basis points per tick."""
    prices = []
    price = 1.0
    for i in range(ticks):
        price *= 1.0004 if i % 3 else 0.9997
        prices.append(price)
    return _feed([(i * spacing, p) for i, p in enumerate(reversed(prices))])


def _validate(feed, **kw):
    """Score ``feed``, trading at a price that belongs to it.

    The price defaults to the newest tick rather than a constant. The guard
    also refuses a quote that sits far from the window's own median
    (``price_off_scale``, bound 3.0), so passing an unrelated number here
    would refuse every fixture for a reason these tests are not about.
    """
    validator = _validator(_StubDB(feed))
    newest = max(feed, key=lambda row: row["ts"])["price"]
    return validator, validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=kw.pop("trade_size", 0.0075), price=kw.pop("price", newest),
        volume=0.0, **kw,
    )


def test_a_fat_tail_is_refused_even_though_dispersion_passes() -> None:
    """The BSTONK case: inside the volatility limit, outside the stop."""
    validator, (allowed, metrics, reasons) = _validate(_fat_tailed())

    # The point of the test: the clause that already existed does NOT object.
    # If this assertion ever fails the fixture has drifted into being merely
    # "too volatile", and the test would pass for the wrong reason.
    assert metrics["volatility_measurable"] == 1.0
    assert metrics["volatility"] <= validator.max_volatility
    assert "volatility" not in reasons
    assert "feed_frozen" not in reasons

    # ...and the new one does.
    assert "stop_unenforceable" in reasons
    assert allowed is False
    assert metrics["vol_jump_p99"] >= metrics["stop_jump_budget"]


def test_a_tight_feed_still_trades() -> None:
    """CBBTC/CBETH/AERO produced the only live wins; they must stay open.

    A guard that refuses everything is not a guard, it is an outage -- this
    repo has already spent six days with every live entry refused.
    """
    _validator_, (allowed, metrics, reasons) = _validate(_tight())

    assert "stop_unenforceable" not in reasons
    assert allowed is True
    assert 0.0 < metrics["vol_jump_p99"] < metrics["stop_jump_budget"]


def test_the_budget_tracks_the_stop_the_exit_path_enforces(monkeypatch) -> None:
    """Guard and enforcer must read one number, not two that agree today.

    The verdict has to move when ``LIVE_STOP_LOSS_PCT`` moves: a guard that
    clears a trade for a 1.5% stop while the exit path applies 0.5% is
    asserting something it has not checked.
    """
    feed = _fat_tailed()

    monkeypatch.setenv("LIVE_STOP_LOSS_PCT", "0.015")
    _v1, (allowed_tight_stop, metrics_tight, reasons_tight) = _validate(feed)

    # Widen the stop past the measured tail: the same feed becomes tradeable,
    # because now the stop can actually absorb a tick.
    monkeypatch.setenv("LIVE_STOP_LOSS_PCT", "0.25")
    _v2, (allowed_wide_stop, metrics_wide, reasons_wide) = _validate(feed)

    assert allowed_tight_stop is False and "stop_unenforceable" in reasons_tight
    assert allowed_wide_stop is True and "stop_unenforceable" not in reasons_wide

    # The budget moved because the STOP moved; the feed never changed.
    assert metrics_tight["vol_jump_p99"] == pytest.approx(metrics_wide["vol_jump_p99"])
    assert metrics_wide["stop_loss_pct"] > metrics_tight["stop_loss_pct"]

    # And it is the same number the exit path applies. Drive the real trigger
    # at a loss just past the guard's stop and confirm it fires there.
    stop = _stop_loss_pct(live=True)
    assert metrics_wide["stop_loss_pct"] == pytest.approx(stop)
    now = time.time()
    decision = evaluate_long_triggers(
        {"entry_price": 1.0, "size": 1.0, "entry_ts": now},
        price=1.0 - stop - 1e-6,
        fee_rate=0.0065,
        now_ts=now,
        live=True,
    )
    assert decision.should_exit is True
    assert decision.reason.startswith("stop_loss:")


def test_the_tail_is_read_off_the_filtered_series() -> None:
    """A denomination artifact must not be reported as a tick jump.

    SPACEX-USDC carries two assets under one ticker, eleven orders of
    magnitude apart. If the jump statistic were computed before the outlier
    and gap filters, that pair alone would produce a 1.29e11 "tick move" and
    this clause would inherit the exact bug the volatility clause was fixed
    for. Same window, same filters, same returns -- so a contaminated feed is
    refused as *unmeasurable*, never as a measured tail.
    """
    real = [(i * 95.0, 1.5e-09 * (1.001 if i % 2 else 0.999)) for i in range(1, 40)]
    impostor = [(0.0, 524.37)]
    _v, (_allowed, metrics, reasons) = _validate(
        _feed(real + impostor), price=1.5e-09,
    )

    # The filter ran and took the impostor with it.
    assert metrics["vol_dropped_outliers"] == 1.0

    # The 3.5e11 return that pairing 1.5e-09 with 524.37 would produce never
    # reaches the statistic. The tail reported is the real series' own tick
    # size -- a fifth of a percent -- so the artifact is not laundered into a
    # measured "jump" and the pair is not refused on a number no market made.
    assert metrics["vol_jump_max"] < 0.01
    assert "stop_unenforceable" not in reasons
