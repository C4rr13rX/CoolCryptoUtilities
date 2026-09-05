"""The lattice gate on the live entry path.

The gate adds a reason to REFUSE. It must never become the reason nothing
trades, so every test here is either "does it catch the thing it exists for"
or "does it get out of the way when it cannot answer".
"""

from __future__ import annotations

import random
import time

import pytest

from trading.bot import TradingBot


def _persistent_series(n: int = 300, bar_sec: float = 300.0) -> list:
    """A series with autocorrelated returns and a measurable horizon."""
    rng = random.Random(20260905)
    now = time.time()
    out = []
    price, previous = 100.0, 0.0
    for i in range(n):
        step = 0.6 * previous + rng.gauss(0, 0.003)
        price *= 1 + step
        previous = step
        out.append({"ts": now - (n - i) * bar_sec, "price": price})
    return out


class _Directive:
    def __init__(self, horizon="15m", expected_return=0.03, action="enter"):
        self.action = action
        self.expected_return = expected_return
        self.horizon = horizon
        self.size = 1.0
        self.strategy_id = "test"


def _bot(buffer):
    bot = TradingBot.__new__(TradingBot)
    bot._buffer = buffer
    bot._live_clip_usd = lambda: 0.75
    bot._roundtrip_fee_rate = lambda notional_hint=None: 0.0065
    return bot


class TestBarLength:
    def test_the_tick_gap_is_measured_not_assumed(self):
        """A wrong bar length scales every horizon by exactly that factor.

        The chaos layer converts a divergence rate per BAR into seconds, so
        assuming 300s on a feed that ticks every 30 would overstate every
        usable horizon tenfold.
        """
        bot = _bot([])
        assert bot._median_tick_gap_sec(_persistent_series(bar_sec=300.0)) == pytest.approx(300.0, rel=0.02)
        assert bot._median_tick_gap_sec(_persistent_series(bar_sec=30.0)) == pytest.approx(30.0, rel=0.02)

    def test_burst_writes_do_not_collapse_the_bar(self):
        """The bug that made the gate refuse every entry on every symbol.

        The feed writes bursts of ticks ~1s apart, then waits minutes. The
        gap distribution is bimodal, so the MEDIAN lands inside a burst and
        describes how fast rows are written rather than how often the price
        is sampled. Measured on AERO-USDC over 300 samples spanning 12.1
        hours: median 2.9s, p75 66s, p90 365s, mean 146s.

        A 3s bar made every usable horizon ~7 seconds, and the gate refused
        everything.
        """
        now = time.time()
        bursty = []
        stamp = now - 12 * 3600
        for _ in range(60):
            for _ in range(5):            # a burst, one second apart
                bursty.append({"ts": stamp, "price": 1.0})
                stamp += 1.0
            stamp += 700.0                # then a long real gap

        bar = _bot([])._median_tick_gap_sec(bursty)
        assert bar > 100.0, (
            f"bar collapsed to {bar:.1f}s -- the median landed in a burst")

    def test_a_stalled_feed_does_not_produce_an_enormous_bar(self):
        now = time.time()
        stalled = [{"ts": now - 100000, "price": 1.0},
                   {"ts": now - 50000, "price": 1.0},
                   {"ts": now - 10, "price": 1.0},
                   {"ts": now, "price": 1.0}]
        assert _bot([])._median_tick_gap_sec(stalled) <= 3600.0


class TestRefusal:
    def test_a_forecast_past_the_horizon_is_refused(self):
        """The failure that made bus_schedule the worst strategy in the book."""
        bot = _bot(_persistent_series())
        sample = {"price": bot._buffer[-1]["price"], "ts": time.time()}
        refusal = bot._lattice_refusal("T", _Directive(horizon="1w"), sample)
        assert refusal is not None
        assert refusal.startswith("chaos")

    def test_a_forecast_inside_the_horizon_is_allowed(self):
        bot = _bot(_persistent_series())
        sample = {"price": bot._buffer[-1]["price"], "ts": time.time()}
        assert bot._lattice_refusal("T", _Directive(horizon="15m"), sample) is None

    def test_an_exit_is_not_judged_as_a_forecast(self):
        bot = _bot(_persistent_series())
        sample = {"price": bot._buffer[-1]["price"], "ts": time.time()}
        directive = _Directive(horizon="1w", action="exit")
        assert bot._lattice_refusal("T", directive, sample) is None


class TestFailsOpen:
    """A new check that can silently stop all trading is worse than the gap
    it closes. Every unanswerable case must let the trade through to the
    guards that were the whole defence before this existed."""

    def test_a_short_window_allows(self):
        bot = _bot(_persistent_series()[:10])
        sample = {"price": 1.0, "ts": time.time()}
        assert bot._lattice_refusal("T", _Directive(), sample) is None

    def test_a_raised_exception_allows(self):
        bot = _bot(_persistent_series())

        def _boom(notional_hint=None):
            raise RuntimeError("cost model unavailable")

        bot._roundtrip_fee_rate = _boom
        sample = {"price": 1.0, "ts": time.time()}
        assert bot._lattice_refusal("T", _Directive(), sample) is None

    def test_an_unknown_horizon_label_allows(self):
        bot = _bot(_persistent_series())
        sample = {"price": bot._buffer[-1]["price"], "ts": time.time()}
        assert bot._lattice_refusal("T", _Directive(horizon="42y"), sample) is None

    def test_a_disabled_gate_allows(self, monkeypatch):
        monkeypatch.setenv("LATTICE_GATE_ENABLED", "0")
        bot = _bot(_persistent_series())
        sample = {"price": bot._buffer[-1]["price"], "ts": time.time()}
        assert bot._lattice_refusal("T", _Directive(horizon="1w"), sample) is None
