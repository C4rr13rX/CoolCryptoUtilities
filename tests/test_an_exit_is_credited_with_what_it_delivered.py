"""An exit may not claim a benefit its own series has never delivered.

The exit-side twin of ``test_an_entry_is_credited_with_what_it_delivered``.
There the entry gate took its expected return from the strategy's own
``target_price`` (atf_static builds it as ``price * 1.05``), so it asked "is 5%
more than the cost?" and approved all 20 live entries ever taken. Here every
reversion exit took its expected return from the distance between the price and
some reference -- the rolling mean, the VWAP, the window median, the recent low
-- and handed that to the CDCL ``return_above_fees`` clause as the benefit of
exiting now.

Distance from a reference measures EXTENSION. Measured 2026-09-05 over 14 days
of ``market_stream``, 2585 firings of the RSI-overbought exit across 18
symbols: mean claim +5.40%, realised benefit -0.04% at 300s, +0.16% at 900s,
-0.25% at 1800s, against a 0.32% exit leg. Price fell after the signal 41-47%
of the time. On AERO-USDC, the only symbol the live strategy is permitted to
trade, the realised benefit over 240 firings is -0.09%.

It spent real money: atf_static's live exit at 2026-09-05 16:52 recorded
"RSI 74 overbought, harvesting 8.22%" and realised -1.35% for -0.023520, one of
the two trades that make its permitted book negative and hold it demoted off
live trading entirely.
"""
from __future__ import annotations

import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from trading.strategies.base import (
    StrategyContext,
    measured_exit_benefit,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_vwap,
)
from trading.strategies.rsi_reversal import RsiReversalStrategy


def _State(samples):
    """Minimal stand-in for RouteState -- the fields make_candidate reads."""
    return SimpleNamespace(
        symbol="TEST-USDC",
        base_token="TEST",
        quote_token="USDC",
        samples=deque(samples),
    )


def _ctx(last_price: float) -> StrategyContext:
    return StrategyContext(
        chain="base",
        last_price=last_price,
        last_volume=1.0,
        fee_rate=0.0032,
        available_quote=0.0,
        available_base=10.0,
    )


def _series(prices, step=60.0, volume=1.0):
    return [(float(i) * step, float(p), float(volume)) for i, p in enumerate(prices)]


class RollingHelpers(unittest.TestCase):
    def test_rolling_windows_are_trailing_and_nan_padded(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for fn, want_last in ((rolling_mean, 4.0), (rolling_median, 4.0), (rolling_min, 3.0)):
            out = fn(v, 3)
            self.assertEqual(out.size, v.size, fn.__name__)
            self.assertTrue(np.all(np.isnan(out[:2])), fn.__name__)
            self.assertAlmostEqual(float(out[-1]), want_last, places=9, msg=fn.__name__)
        vw = rolling_vwap(v, np.array([1.0, 1.0, 1.0, 1.0, 1.0]), 3)
        self.assertAlmostEqual(float(vw[-1]), 4.0, places=9)

    def test_rolling_vwap_is_nan_when_the_window_has_no_volume(self):
        out = rolling_vwap(np.array([1.0, 2.0, 3.0]), np.zeros(3), 3)
        self.assertTrue(np.isnan(float(out[-1])))


class MeasuredExitBenefit(unittest.TestCase):
    def test_a_series_that_keeps_rising_offers_no_benefit_from_exiting(self):
        """The defect, stated as arithmetic.

        A steadily rising series sits far above its own trailing mean at every
        bar -- a large extension -- while the forward move is always UP, so the
        benefit of exiting is negative and must floor at zero rather than be
        reported as the extension.
        """
        prices = np.array([100.0 * (1.01 ** i) for i in range(80)])
        ts = np.arange(80, dtype=np.float64) * 60.0
        ref = rolling_mean(prices, 20)
        extension = float((prices[-1] - ref[-1]) / prices[-1])
        self.assertGreater(extension, 0.05, "the old expected_return would be >5%")

        benefit = measured_exit_benefit(ts, prices, ref, horizon_sec=300.0)
        self.assertIsNotNone(benefit)
        self.assertEqual(benefit, 0.0, "a series that only rises pays nothing to exit")

    def test_a_series_that_actually_reverts_is_credited(self):
        """The guard must not simply refuse everything.

        A sawtooth that genuinely falls back after each spike has a real,
        positive benefit to exiting, and it is reported.
        """
        base = []
        for _ in range(12):
            base.extend([100.0, 103.0, 106.0, 103.0, 100.0, 98.0])
        base.extend([100.0, 103.0, 106.0])  # judged at a peak, where an exit fires
        prices = np.array(base, dtype=np.float64)
        ts = np.arange(prices.size, dtype=np.float64) * 60.0
        ref = rolling_mean(prices, 6)
        benefit = measured_exit_benefit(ts, prices, ref, horizon_sec=180.0, min_comparable=5)
        self.assertIsNotNone(benefit)
        self.assertGreater(benefit, 0.0, "a series that reverts does pay to exit")

    def test_an_unmeasurable_benefit_is_not_claimed(self):
        """None, not the optimistic default.

        This repo already treats an unmeasurable COST defaulted to zero as a
        defect; the optimistic default on a BENEFIT is the same error with the
        sign reversed.
        """
        prices = np.array([100.0 * (1.01 ** i) for i in range(30)])
        ts = np.arange(30, dtype=np.float64) * 60.0
        ref = rolling_mean(prices, 20)
        self.assertIsNone(
            measured_exit_benefit(ts, prices, ref, horizon_sec=300.0, min_comparable=50)
        )

    def test_no_benefit_when_price_sits_below_its_reference(self):
        prices = np.array([100.0 * (0.99 ** i) for i in range(60)])
        ts = np.arange(60, dtype=np.float64) * 60.0
        self.assertIsNone(
            measured_exit_benefit(ts, prices, rolling_mean(prices, 20), horizon_sec=300.0)
        )

    def test_the_bar_being_judged_is_not_evidence_about_itself(self):
        prices = np.array([100.0 + i for i in range(60)], dtype=np.float64)
        ts = np.arange(60, dtype=np.float64) * 60.0
        ref = rolling_mean(prices, 10)
        # With a horizon past the end of the series the last bar has no forward
        # return at all; the call must still work off the earlier bars.
        self.assertIsNotNone(
            measured_exit_benefit(ts, prices, ref, horizon_sec=120.0, min_comparable=3)
        )


class RsiExitUsesTheMeasurement(unittest.TestCase):
    def test_a_relentless_rally_no_longer_fires_an_overbought_harvest(self):
        """The live 16:52 shape: RSI pinned overbought, price above its mean,
        and nothing behind the claim that exiting captures anything."""
        prices = [100.0 * (1.006 ** i) for i in range(120)]
        state = _State(_series(prices))
        ctx = _ctx(prices[-1])

        candidate = RsiReversalStrategy().evaluate(state, ctx)
        self.assertIsNone(
            candidate,
            "an exit whose comparable history only rose must not claim a benefit",
        )

    def test_an_exit_that_its_history_supports_still_fires(self):
        """The other half: this must not become a gate that refuses everything.

        A series that has repeatedly given the extension straight back does
        offer a real benefit to exiting, and the exit fires -- reporting the
        measurement, with the old extension kept beside it the way the
        stop-jump fix kept ``vol_adjacent_jump_p99`` beside its replacement.
        """
        rally = [1.010 ** i for i in range(16)]
        fall = [rally[-1] * (0.975 ** j) for j in range(1, 9)]
        series = []
        for _ in range(8):
            series.extend(100.0 * x for x in rally + fall)
        series.extend(100.0 * x for x in rally)  # judged at the top of a rally

        candidate = RsiReversalStrategy().evaluate(_State(_series(series)), _ctx(series[-1]))
        self.assertIsNotNone(candidate, "a genuinely reverting series must still exit")
        directive = candidate["directive"]
        self.assertEqual(directive.action, "exit")

        meta = candidate["meta"]
        self.assertIn("extension", meta)
        self.assertIn("measured_benefit", meta)
        self.assertEqual(meta["measured_benefit"], directive.expected_return)
        self.assertNotIn("harvesting", directive.reason)
        # The two readings are genuinely different numbers, which is the whole
        # point: the old code shipped the first one as if it were the second.
        self.assertNotAlmostEqual(meta["extension"], meta["measured_benefit"], places=4)


if __name__ == "__main__":
    unittest.main()
