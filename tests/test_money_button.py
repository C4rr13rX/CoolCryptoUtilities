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

import json
import os
import pathlib
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

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


class MixedDenominationFeeds(unittest.TestCase):
    """A window holding two price scales is not one asset's price history.

    Measured 2026-09-02 over 168h of stored ticks: 15 of 163 symbols publish
    prices differing from their own median by more than 50x -- 220 ticks, 1.8%
    of the feed. ARB-USDC prints both 5.3e-7 and 0.6491; WETH-USDT prints 2450
    and 0.9996; SPACEX-USDC holds 233 ticks near 1.5e-9 and one at 524.37.

    Read as a move, that SPACEX pair is a return of +1.29e11. Slope,
    projection and volatility are all computed off the same array, so a single
    such tick decides all three -- and the entry it produces would be sized at
    the wrong scale with real money.

    Worth stating plainly what this gate did and did not do: replayed over the
    same 12,239 evaluations, it fired 3 times and changed no fire and no
    outcome, because the affected symbols were already stopped by the feed
    gates. It is a bound on a failure that has not yet reached an entry, not a
    fix for one that has.
    """

    def _mixed(self):
        # A clean trend, with one tick quoted in the other denomination --
        # the SPACEX shape, at a scale a momentum lane would otherwise love.
        prices = [1.0 + 0.01 * i for i in range(20)]
        prices[10] = prices[10] * 1e9
        return prices

    def test_a_second_price_scale_is_refused(self):
        strategy = MoneyButtonStrategy()
        prices = self._mixed()
        self.assertIsNone(strategy.evaluate(make_state(prices), make_ctx(prices[-1])))
        self.assertEqual(strategy.last_decline, "mixed_denomination")

    def test_it_refuses_rather_than_quietly_dropping_the_tick(self):
        """Filtering would let the lane keep trading a pair it cannot price.

        The refusal has to be visible in the census, or a broken feed looks
        identical to a market with no opportunity in it.
        """
        strategy = MoneyButtonStrategy()
        prices = self._mixed()
        strategy.evaluate(make_state(prices), make_ctx(prices[-1]))
        self.assertNotIn(strategy.last_decline, (None, "momentum_not_aligned"))

    def test_an_ordinary_large_move_still_trades(self):
        """A 25% trend is a move; it must not be mistaken for an artifact."""
        strategy = MoneyButtonStrategy()
        prices = [1.0 * (1.0125 ** i) for i in range(20)]
        strategy.evaluate(make_state(prices), make_ctx(prices[-1]))
        self.assertNotEqual(strategy.last_decline, "mixed_denomination")

    def test_the_scale_limit_is_read_from_the_environment(self):
        """Raising the bound past the outlier must actually stop the refusal.

        Pinned with a 100x outlier rather than the 1e9 one so the two settings
        give different verdicts -- a test where both branches refuse would pass
        just as happily if the environment were never consulted.
        """
        prices = [1.0 + 0.01 * i for i in range(20)]
        prices[10] = prices[10] * 100.0

        strategy = MoneyButtonStrategy()
        strategy.evaluate(make_state(prices), make_ctx(prices[-1]))
        self.assertEqual(strategy.last_decline, "mixed_denomination")

        with mock.patch.dict(
            os.environ, {"MONEY_BUTTON_MAX_PRICE_SCALE": "1000"}, clear=False
        ):
            relaxed = MoneyButtonStrategy()
            relaxed.evaluate(make_state(prices), make_ctx(prices[-1]))
            self.assertNotEqual(relaxed.last_decline, "mixed_denomination")


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

        The window must hold min_samples with MARGIN at the observed gap, or
        the strategy stops being evaluated the moment the feed dips.
        """
        strategy = MoneyButtonStrategy()
        window_minutes = strategy.LOOKBACK_SEC / 60.0
        # Observed median gap between prints on a live symbol, not the
        # smoothed average: 7.5 minutes. Sizing from the average hid the fact
        # that a 12/90 configuration computed to EXACTLY 12.0 samples in
        # steady state -- satisfiable only if the feed never slows at all.
        holds = window_minutes / 7.5
        headroom = holds - strategy.min_samples
        self.assertGreaterEqual(
            headroom, 3.0,
            f"window holds ~{holds:.1f} samples at the observed 7.5-min gap "
            f"but min_samples is {strategy.min_samples} (headroom "
            f"{headroom:+.1f}). A requirement the feed can only just meet is "
            f"one it will fail, and the failure is silent: evaluate_all simply "
            f"stops considering this strategy.",
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


class FrozenFeedGuardScalesWithTheWindow(unittest.TestCase):
    """
    The frozen-feed guard must scale, not sit at a fixed count.

    It was written as `max(6, n // 4)` for a 40-sample window, where n//4 = 10
    and the 6 was the *lenient* branch. At the 10-sample window this strategy
    actually runs on, that same 6 demands 60% distinct prices and became the
    binding constraint. Measured on live data every single symbol was blocked
    by it -- AERO 2 distinct of 7, CBBTC 5 of 6, DRIFT 1 of 4 -- so the
    strategy could not have fired regardless of what the market did.

    An on-chain quote legitimately repeats between trades, so the requirement
    is now proportional: about a third of the window must differ.
    """

    @staticmethod
    def _evaluate(prices, gap_sec=450.0):
        state = make_state(list(prices), volume=5000.0, dt=gap_sec)
        return MoneyButtonStrategy().evaluate(
            state, make_ctx(float(prices[-1]), 0.0065)
        )

    def test_a_frozen_feed_is_still_rejected(self):
        self.assertIsNone(self._evaluate([1.0] * 10))

    def test_two_plateaus_are_still_rejected(self):
        """
        A single step is not a trend.

        Real BASECAT data looked like this -- 0.02457 twice, then 0.02608
        twice -- and reported as "+6.15%". That jump cannot be projected
        forward over a 12-minute hold.
        """
        self.assertIsNone(self._evaluate([1.0] * 5 + [1.06] * 5))

    def test_a_genuine_trend_is_not_blocked_by_the_guard(self):
        """The regression: the guard must not veto real movement."""
        self.assertIsNotNone(self._evaluate(np.linspace(1.0, 1.25, 10)))

    def test_the_requirement_scales_with_the_sample_count(self):
        """
        Pinned as a property so a fixed floor cannot creep back in.

        A hard number is what made this unreachable at small windows.
        """
        for n in (10, 20, 40):
            with self.subTest(samples=n):
                required = max(3, int(np.ceil(n / 3.0)))
                self.assertLessEqual(
                    required, n,
                    "the guard must never demand more distinct prices than "
                    "there are samples",
                )
                self.assertGreaterEqual(required, 3)


class FeedMustResolveTheTradeItIsAsked(unittest.TestCase):
    """A window must fit its data, and a hold must contain a price.

    The momentum windows scale to the sample gap so they are never empty, but
    nothing tied that stretching back to the HOLD, which is fixed at ~12
    minutes. On a coarse enough feed the two decouple silently: the strategy
    confirms a multi-hour trend and then holds for twelve minutes.

    Both guards are structural. There is no threshold to pick: a 6x window
    either fits inside LOOKBACK_SEC or it is the whole window wearing a label,
    and a holding period either contains a price or the position cannot be
    exited when intended.

    Calibration note, because the first attempt got this wrong: requiring TWO
    ticks inside the hold rejected the production feed wholesale. The measured
    rate is 0.13-0.17 ticks/min, so gaps run 360-460s against a 720s hold --
    about 1.8 ticks per hold. One tick is the structural minimum and the only
    bound that can be argued from first principles.
    """

    @staticmethod
    def _evaluate(prices, gap_sec):
        state = make_state(list(prices), volume=5000.0, dt=gap_sec)
        return MoneyButtonStrategy().evaluate(
            state, make_ctx(float(prices[-1]), 0.0065)
        )

    def _decline_reason(self, prices, gap_sec):
        strategy = MoneyButtonStrategy()
        state = make_state(list(prices), volume=5000.0, dt=gap_sec)
        strategy.evaluate(state, make_ctx(float(prices[-1]), 0.0065))
        return strategy.last_decline

    def test_the_production_feed_rate_is_not_rejected(self):
        """The regression that matters: 0.13-0.17 ticks/min must still trade.

        A guard that blocks the only feed this runs on is the same class of
        mistake as the min_samples=20 that once made the lane unevaluable.
        """
        for gap in (360.0, 400.0, 460.0):
            with self.subTest(gap_sec=gap):
                self.assertIsNotNone(
                    self._evaluate(np.linspace(1.0, 1.25, 12), gap),
                    "a 25%% trend at a %.0fs gap is the production case" % gap,
                )

    def test_a_hold_that_contains_no_price_is_refused(self):
        """Gap wider than the 12-minute hold: nothing to exit against.

        740s is used rather than something far coarser because past ~900s the
        lookback stops holding `min_samples` and `too_few_samples` refuses
        first -- which is correct, but tests a different guard than this one.
        """
        self.assertEqual(
            self._decline_reason(np.linspace(1.0, 1.30, 16), 740.0),
            "feed_too_sparse_for_hold",
        )

    def test_a_confirmation_window_longer_than_the_lookback_is_refused(self):
        """unit*6 > LOOKBACK_SEC means r30 silently becomes the whole window.

        The binding range is a narrow one -- gaps of about 610-730s, where the
        6x window no longer fits but the hold still contains a tick -- so the
        fixture is checked against the arithmetic rather than assumed.
        """
        strategy = MoneyButtonStrategy()
        gap = 650.0
        unit = max(5.0 * 60.0, 2.0 * gap)
        self.assertGreater(
            unit * 6.0, strategy.LOOKBACK_SEC,
            "fixture must actually exceed the lookback for this to test anything",
        )
        self.assertLessEqual(
            gap, 12.0 * 60.0,
            "fixture must still resolve the hold, or the other guard fires first",
        )
        self.assertEqual(
            self._decline_reason(np.linspace(1.0, 1.30, 16), gap),
            "window_exceeds_lookback",
        )

    def test_a_gap_that_fits_both_bounds_still_trades(self):
        """600s fits the 6x window exactly and must not be refused by these."""
        reason = self._decline_reason(np.linspace(1.0, 1.30, 16), 600.0)
        self.assertNotIn(
            reason, {"feed_too_sparse_for_hold", "window_exceeds_lookback"},
            "the boundary case must fall through to the edge gates, got %r" % reason,
        )

    def test_the_guards_do_not_fire_on_a_dense_feed(self):
        """BASECAT-USDC's real shape: 26s median gap, 52 samples."""
        self.assertIsNotNone(self._evaluate(np.linspace(1.0, 1.25, 52), 26.0))


if __name__ == "__main__":
    unittest.main()


class MoneyButtonOutcomesReachTheLedger(unittest.TestCase):
    """The ledger is the file that gates graduation, and money_button was not in it.

    Measured 2026-09-02: data/strategy_registry.json held money_button with one
    ghost trade (+0.0053 on TOAD-USDC, corroborated by the matching row in
    trade_outcomes) while data/strategy_ledger.json had no money_button entry at
    all. So it could never graduate no matter how well it traded.

    That asymmetry names the cause exactly. StrategyLedger.record() writes the
    registry FIRST and the ledger second, so a lost ledger write leaves the
    registry holding an outcome the ledger never got -- which is what was on
    disk. The strategy_id and symbol were being passed correctly all along by
    trading/bot.py; the write was being dropped by the concurrent-writer race
    fixed in services/atomic_json.py.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = pathlib.Path(self.dir) / "strategy_ledger.json"

    def test_an_outcome_lands_under_its_own_strategy_id(self):
        from trading.strategies.ledger import StrategyLedger

        StrategyLedger(self.path).record(
            "money_button", profit=0.005277519134108606, mode="ghost", symbol="TOAD-USDC"
        )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("money_button", data)
        self.assertEqual(data["money_button"]["ghost"]["trades"], 1)

    def test_its_symbol_is_recorded(self):
        """Without symbols, thirteen correlated bets on ONE token look like
        thirteen independent trades.

        Only the PRODUCTION ledger feeds the lifetime registry -- a ledger on
        any other path is a test or a replay and its outcomes are not
        measurements of this account. So DEFAULT_PATH is redirected here rather
        than merely passing a temp path, which would skip the registry write
        entirely and prove nothing.
        """
        import services.strategy_registry as registry
        from trading.strategies.ledger import StrategyLedger

        with mock.patch.object(StrategyLedger, "DEFAULT_PATH", self.path), \
             mock.patch.object(registry, "REGISTRY_PATH", pathlib.Path(self.dir) / "reg.json"):
            StrategyLedger(self.path).record(
                "money_button", profit=0.0053, mode="ghost", symbol="TOAD-USDC"
            )
            entry = registry.get_strategy("money_button")
        self.assertIsNotNone(entry, "the production path did not reach the registry")
        self.assertEqual(entry["lifetime"]["ghost"]["symbols"], {"TOAD-USDC": 1})

    def test_a_non_production_ledger_never_writes_the_lifetime_registry(self):
        """How money_button got a fabricated 76-trade record: test fixtures
        isolated the ledger path but record_outcome takes no path, so every run
        wrote straight into the production registry."""
        import services.strategy_registry as registry
        from trading.strategies.ledger import StrategyLedger

        reg_path = pathlib.Path(self.dir) / "reg.json"
        with mock.patch.object(registry, "REGISTRY_PATH", reg_path):
            StrategyLedger(self.path).record(
                "money_button", profit=1.0, mode="ghost", symbol="FAKE-USDC"
            )
        self.assertFalse(reg_path.exists(), "a test ledger wrote the lifetime registry")

    def test_concurrent_money_button_exits_are_all_kept(self):
        """The lane fires often by design, so its writes contend the most."""
        from trading.strategies.ledger import StrategyLedger

        threads = [
            threading.Thread(
                target=lambda: StrategyLedger(self.path).record(
                    "money_button", profit=-0.004, mode="ghost", symbol="TOAD-USDC"
                )
            )
            for _ in range(24)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(data["money_button"]["ghost"]["trades"], 24)
        self.assertEqual(data["money_button"]["ghost"]["losses"], 24)

    def test_it_still_needs_the_full_sample_to_graduate(self):
        """Being in the ledger is not the same as being allowed to trade."""
        from trading.strategies.ledger import StrategyLedger

        with mock.patch.dict(
            os.environ,
            {"STRATEGY_GRADUATION_MIN_TRADES": "20", "STRATEGY_GRADUATION_MIN_WINRATE": "0.55"},
            clear=False,
        ):
            for _ in range(19):
                StrategyLedger(self.path).record(
                    "money_button", profit=0.01, mode="ghost", symbol="TOAD-USDC"
                )
            self.assertFalse(StrategyLedger(self.path).is_live_approved("money_button"))
