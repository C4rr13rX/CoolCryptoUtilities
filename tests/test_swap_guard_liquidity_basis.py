"""The swap guard must not invent a liquidity violation out of missing data.

Observed 2026-08-28: zero live trades had ever been placed. Every tick in
market_stream carries volume=0 (the DexScreener REST consensus path publishes
price only), so the guard's observed mean volume was 0.0 for every pair, and
``trade_usd / max(0.0, 1e-6)`` scored a $0.35 trade on AERO -- $1.26M pooled,
$386k/h traded -- at a liquidity ratio of 348,862 against a limit of 0.35.
Every live entry on every pair was refused on that number, indefinitely.

These pin the distinction the guard has to keep: a measured ratio blocks, an
unmeasured one does not get to pretend it measured infinity.
"""
from __future__ import annotations

import time

from trading.swap_validator import SwapValidator


class _StubDB:
    """Minimal stand-in for TradingDatabase: samples in, nothing out."""

    def __init__(self, samples, hourly_reference=None):
        self._samples = samples
        self._hourly = hourly_reference

    def fetch_market_samples_for(self, symbol, limit=512):
        return list(self._samples)

    def fetch_trade_fills(self, limit=100):
        return []

    def reference_hourly_volume_usd(self, symbol):
        return self._hourly

    # MetricsCollector touches these; keep them inert.
    def record_metric(self, *a, **k):
        return None

    def log_feedback(self, *a, **k):
        return None

    def record_feedback(self, *a, **k):
        return None


def _samples(*, price, volume, n=40, spacing=300.0):
    """A price series carrying ``volume``, which is what these tests vary.

    The prices alternate by a tenth of a percent around ``price`` rather than
    repeating it exactly. A perfectly constant series is a *stuck feed*, which
    the volatility clause refuses under ``feed_frozen``; holding it constant
    here would make every one of these liquidity assertions pass or fail for a
    reason that has nothing to do with liquidity.
    """
    now = time.time()
    return [
        {"ts": now - i * spacing, "chain": "base", "symbol": "T-USDC",
         "price": price * (1.001 if i % 2 else 0.999), "volume": volume}
        for i in range(n)
    ]


def _validator(db):
    v = SwapValidator(db=db)
    # MetricsCollector writes through the stub; silence it entirely so the
    # assertions are about the verdict, not about persistence.
    v.metrics.record = lambda *a, **k: None
    v.metrics.feedback = lambda *a, **k: None
    return v


def test_volume_free_feed_does_not_fabricate_a_liquidity_block() -> None:
    """A price-only feed means unmeasured, not illiquid."""
    db = _StubDB(_samples(price=100.0, volume=0.0))
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=0.0075, price=100.0, volume=0.0,
    )

    assert "liquidity" not in reasons
    assert allowed is True
    # The ratio is reported as absent, not as a fabricated number.
    assert metrics["liquidity_measured"] == 0.0


def test_unmeasured_liquidity_still_caps_absolute_notional() -> None:
    """Without a denominator the guard falls back to a hard USD ceiling."""
    db = _StubDB(_samples(price=100.0, volume=0.0))
    validator = _validator(db)
    validator.unknown_liquidity_max_usd = 25.0

    allowed, _metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=5.0, price=100.0, volume=0.0,  # $500
    )

    assert allowed is False
    assert "liquidity_unmeasured" in reasons


def test_independent_hourly_volume_answers_a_silent_feed() -> None:
    """DexScreener pair volume gives the ratio a real denominator."""
    # $360k/h over 300s samples => $30k per sample; a $0.75 trade is 2.5e-5.
    db = _StubDB(_samples(price=100.0, volume=0.0), hourly_reference=360_000.0)
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=0.0075, price=100.0, volume=0.0,
    )

    assert allowed is True
    assert reasons == []
    assert metrics["liquidity_measured"] == 1.0
    assert metrics["avg_volume_usd"] == 30_000.0
    assert metrics["liquidity_ratio"] < 0.001


def test_reference_volume_still_blocks_a_genuinely_thin_pair() -> None:
    """The guard keeps biting where the book really is too small."""
    # $12/h over 300s samples => $1 per sample; a $0.75 trade is 75% of it.
    db = _StubDB(_samples(price=100.0, volume=0.0), hourly_reference=12.0)
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=0.0075, price=100.0, volume=0.0,
    )

    assert allowed is False
    assert "liquidity" in reasons
    assert metrics["liquidity_ratio"] > validator.max_liquidity_ratio


def test_observed_volume_still_takes_precedence_and_blocks() -> None:
    """When the feed does report volume, nothing changed."""
    # price 100 x volume 0.01 => $1 per sample.
    db = _StubDB(_samples(price=100.0, volume=0.01), hourly_reference=10_000_000.0)
    validator = _validator(db)

    allowed, metrics, reasons = validator.validate(
        symbol="T-USDC", route=["T", "USDC"],
        trade_size=0.0075, price=100.0, volume=0.01,
    )

    assert metrics["avg_volume_usd"] == 1.0  # observed wins over reference
    assert allowed is False
    assert "liquidity" in reasons


def test_average_volume_reports_absence_as_none_not_zero() -> None:
    validator = _validator(_StubDB([]))
    assert validator._average_volume_usd(_samples(price=100.0, volume=0.0)) is None
    assert validator._average_volume_usd(_samples(price=100.0, volume=2.0)) == 200.0


def test_sample_interval_is_derived_from_the_samples() -> None:
    validator = _validator(_StubDB([]))
    assert validator._sample_interval_sec(_samples(price=1.0, volume=0.0, spacing=300.0)) == 300.0
    # No usable stamps -> a stated default rather than a divide by zero.
    assert validator._sample_interval_sec([]) == 300.0
