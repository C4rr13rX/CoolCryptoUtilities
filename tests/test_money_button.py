"""
The Money Button must clear its own round-trip cost before it fires.

This is the 5-10 minute lane. Its purpose is to shorten the ghost->live
evidence loop, and the way a fast lane fails is by trading constantly on moves
too small to pay for themselves -- which is exactly how the previous ledger
accumulated 969 trades at a 0% win rate.

Measured against the live feed on 2026-08-26, across the bot's own traded
universe in a single 5-minute window: the median absolute move was 0.000% and
only 1 symbol of 11 (BPAD, 8.13%) cleared the ~1.4% round trip implied by the
0.65% one-way fee. A lane that fires on the other ten is not finding an edge,
it is paying fees for the privilege of losing slowly.

So the tests below pin the *refusals* as hard as the entries.

Why the firing cases use ~25% moves: at the real 0.65% one-way fee the lane
needs roughly a 20%-per-40-minutes trend to clear its gate (at a 0.15% fee it
needs about 10%). That is not a tuning accident, it is what a 1.4% round trip
costs on an 8-minute hold. It also means this lane will decline most symbols
most of the time, and that is the intended behaviour rather than a fault.
"""

from __future__ import annotations

import time
import types
import unittest

import numpy as np

from trading.strategies.base import StrategyContext
from trading.strategies.money_button import MoneyButtonStrategy


def make_state(prices, volume=100.0, dt=60.0):
    """A RouteState-alike carrying `prices` one sample per `dt` seconds."""
    now = time.time()
    n = len(prices)
    state = types.SimpleNamespace()
    state.symbol = "TEST-USDC"
    state.base_token = "TEST"
    state.quote_token = "USDC"
    state.samples = [
        (now - (n - 1 - i) * dt, float(p), volume) for i, p in enumerate(prices)
    ]
    return state


def make_ctx(last_price, fee_rate=0.0065):
    """Default fee is the real one: bot.py uses 0.0015 + 0.005."""
    return StrategyContext(
        chain="base",
        last_price=float(last_price),
        last_volume=100.0,
        fee_rate=fee_rate,
        available_quote=1000.0,
        available_base=0.0,
    )


def evaluate(prices, fee_rate=0.0065, volume=100.0):
    strategy = MoneyButtonStrategy()
    state = make_state(prices, volume=volume)
    return strategy.evaluate(state, make_ctx(prices[-1], fee_rate))


class MoneyButtonRefusals(unittest.TestCase):
    def test_a_frozen_feed_never_trades(self):
        """
        The failure that poisoned the ledger.

        When a feed cannot price a Base-chain token it repeats a seed value.
        A strategy reading that sees a perfectly stable price -- which looks
        like certainty, not absence of data -- and every resulting trade exits
        at exactly its entry.
        """
        self.assertIsNone(evaluate([1.0] * 40))

    def test_a_nearly_frozen_feed_never_trades(self):
        """A handful of distinct prints is still a dead feed."""
        prices = [1.0] * 36 + [1.0001, 1.0, 1.0001, 1.0]
        self.assertIsNone(evaluate(prices))

    def test_pure_noise_does_not_trade(self):
        rng = np.random.default_rng(7)
        self.assertIsNone(evaluate(list(1.0 + rng.normal(0, 0.0002, 40))))

    def test_a_move_too_small_to_pay_its_fees_does_not_trade(self):
        """0.1% over 40 minutes cannot cover a 1.4% round trip."""
        self.assertIsNone(evaluate(list(np.linspace(1.0, 1.001, 40))))

    def test_a_downtrend_does_not_trade(self):
        """This lane is long-only; a falling price is not an entry."""
        self.assertIsNone(evaluate(list(np.linspace(1.06, 1.0, 40))))

    def test_a_good_move_is_refused_when_fees_eat_it(self):
        """
        Same price series, different fee: the decision must flip.

        This is the property that makes the gate a cost gate rather than a
        momentum gate wearing one as a hat.
        """
        prices = list(np.linspace(1.0, 1.12, 40))
        self.assertIsNotNone(evaluate(prices, fee_rate=0.0015))
        self.assertIsNone(evaluate(prices, fee_rate=0.05))

    def test_a_burst_of_prints_in_no_elapsed_time_does_not_trade(self):
        """
        min_samples can be satisfied inside one minute.

        Forty prints spanning 40 seconds say nothing about a 10-minute
        horizon, however clean the trend through them looks.
        """
        packed = make_state(list(np.linspace(1.0, 1.10, 40)), dt=1.0)
        self.assertIsNone(
            MoneyButtonStrategy().evaluate(packed, make_ctx(1.10))
        )


class MoneyButtonEntries(unittest.TestCase):
    def test_a_strong_confirmed_move_fires(self):
        """A move large enough to pay the round trip and keep change."""
        result = evaluate(list(np.linspace(1.0, 1.25, 40)))
        self.assertIsNotNone(result, "a 25% confirmed uptrend should fire")
        directive = result["directive"]
        self.assertEqual(directive.action, "enter")
        self.assertEqual(directive.strategy_id, "money_button")

    def test_the_expected_return_beats_the_round_trip_cost(self):
        """
        The CDCL solver only checks expected_return against a one-way fee.

        The strategy must therefore hand it the already-discounted figure, so
        a candidate can never look better to the solver than it did to the
        gate that produced it.
        """
        fee = 0.0065
        result = evaluate(list(np.linspace(1.0, 1.25, 40)), fee_rate=fee)
        self.assertIsNotNone(result)
        self.assertGreater(result["directive"].expected_return, 2.0 * fee)

    def test_it_stays_inside_its_short_horizon_lane(self):
        """
        The lane is the strategy's identity; drifting out of it is a bug.

        Widened from 5-10 to 5-15 minutes deliberately. At the live 0.65%
        one-way fee a round trip costs ~1.4%, and an 8-minute hold projected
        from a 25%-over-73-minute trend yields only 1.17% after decay -- a
        decline. The same trend clears comfortably at 15 minutes (2.27%).
        Holding the 10-minute ceiling would have produced a strategy that is
        honest and never trades. The fee floor, not the clock, is what makes
        this lane safe.
        """
        result = evaluate(list(np.linspace(1.0, 1.25, 40)))
        self.assertIsNotNone(result)
        horizon = result["directive"].horizon
        self.assertRegex(horizon, r"^\d+m$")
        self.assertTrue(
            5 <= int(horizon.rstrip("m")) <= 15,
            f"horizon {horizon} escaped the 5-15 minute lane",
        )

    def test_it_reports_its_cost_arithmetic(self):
        """
        The gate's inputs must be inspectable after the fact.

        Without these a losing streak is unattributable: there is no way to
        tell whether the edge model or the cost estimate was wrong.
        """
        result = evaluate(list(np.linspace(1.0, 1.25, 40)))
        self.assertIsNotNone(result)
        meta = result["meta"]
        for key in ("round_trip_cost", "required_edge", "conservative_edge",
                    "projected_edge", "edge_headroom"):
            self.assertIn(key, meta)
        self.assertGreaterEqual(meta["conservative_edge"], meta["required_edge"])


class MoneyButtonRegistration(unittest.TestCase):
    def test_it_is_registered_exactly_once(self):
        """
        Registered once, and deliberately NOT multi-horizon swept.

        The sweep rebuilds strategies at 5h..1w. This one projects an edge
        from a per-minute slope over an 8-minute hold; at a 1w horizon that
        projection is meaningless, so a swept variant would be a different
        strategy reporting into a ledger id implying it is this one.
        """
        from trading.strategies import build_default_registry

        ids = list(build_default_registry().ids())
        money = [i for i in ids if i.startswith("money_button")]
        self.assertEqual(money, ["money_button"], f"unexpected variants: {money}")


class MoneyButtonOnASparseFeed(unittest.TestCase):
    """
    The strategy must work on the feed that exists, not an ideal one.

    Production sustains **0.13-0.17 ticks per minute per symbol** (measured
    over 1h, 3h and 24h windows: ~24 streams sharing one event loop on a
    6-core box). Two settings assumed a roughly 1/sec feed and made the
    strategy permanently unevaluable at that rate:

      * `min_samples = 20` inside a 45-minute window. At 0.15/min that window
        settles at ~7 samples, so `evaluate_all` skipped this strategy forever
        while everything looked healthy.
      * fixed 5/10/30-minute return windows. With prints ~6.7 minutes apart
        the trailing 5-minute window often contains NO sample, so
        `_return_over` compared the last price to itself, returned exactly
        0.0, and `r5 <= 0.0` rejected every candidate -- reported as "no
        momentum" rather than "the window was empty".
    """

    @staticmethod
    def _sparse(prices, gap_sec=400.0):
        """12 samples ~6.7 minutes apart: the real production shape."""
        return make_state(list(prices), volume=5000.0, dt=gap_sec)

    def _evaluate(self, prices, gap_sec=400.0, fee_rate=0.0065):
        state = self._sparse(prices, gap_sec)
        return MoneyButtonStrategy().evaluate(
            state, make_ctx(float(prices[-1]), fee_rate)
        )

    def test_a_strong_trend_fires_on_widely_spaced_samples(self):
        """The case that was permanently impossible before."""
        result = self._evaluate(np.linspace(1.0, 1.25, 12))
        self.assertIsNotNone(
            result, "a 25% trend must fire even when prints are 6.7 min apart"
        )

    def test_min_samples_is_reachable_at_the_measured_tick_rate(self):
        """
        A requirement the feed cannot satisfy is a permanent silent refusal.

        At 0.15 ticks/min the lookback window must be wide enough to hold
        min_samples, or the strategy never runs at all.
        """
        strategy = MoneyButtonStrategy()
        window_minutes = strategy.LOOKBACK_SEC / 60.0
        holds = 0.15 * window_minutes
        self.assertGreaterEqual(
            holds, strategy.min_samples,
            f"window holds ~{holds:.1f} samples at the measured rate but "
            f"min_samples is {strategy.min_samples}: unreachable",
        )

    def test_refusals_still_hold_on_a_sparse_feed(self):
        """Adapting to sparse data must not weaken any refusal."""
        for label, prices in (
            ("frozen", np.full(12, 1.0)),
            ("downtrend", np.linspace(1.25, 1.0, 12)),
            ("weak drift", np.linspace(1.0, 1.01, 12)),
        ):
            with self.subTest(label):
                self.assertIsNone(self._evaluate(prices))

    def test_a_spike_that_stalls_is_still_refused(self):
        """Momentum must be live, not merely present earlier in the window."""
        prices = np.concatenate([np.linspace(1.0, 1.25, 9), np.full(3, 1.25)])
        self.assertIsNone(self._evaluate(prices))


if __name__ == "__main__":
    unittest.main()
