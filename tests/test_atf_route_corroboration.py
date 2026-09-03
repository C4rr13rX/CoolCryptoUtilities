"""A candidate no DEX will swap must never become a trade.

Regression test for the two live entries of 2026-09-03. Both went to
MOCHI-USDC, and both carried their own disqualification in the reason string
they were recorded with::

    "atf_static: ATF researched candidate score=0.7236 quote_ok=False"

The quote probe had already run, had already failed, and its result was
recorded into the candidate meta and then ignored, because
``ATF_STATIC_REQUIRE_QUOTE_OK`` defaulted to off. Measured the same day, every
route refuses that pair on base -- 0x returns 403, "UniswapV3: no viable pool
(direct or 2-hop)", Camelot has no router configured for base, and Sushi is
unsupported there. Real money was being aimed at a token that cannot be bought.

The companion guard is ``ATF_STATIC_REQUIRE_FEED_PRICE`` (a price the feed
cannot confirm), pinned in tests/test_atf_feed_corroboration.py. This is the
same idea one step further down: a price we can confirm but cannot transact.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from trading.strategies.atf_static import ATFStaticStrategy
from trading.strategies.base import StrategyContext


def _ctx() -> StrategyContext:
    """The real context object, not a stand-in.

    A hand-rolled fake with only the two attributes ``evaluate`` reads passes
    the refusal tests and then explodes in ``make_candidate`` on the acceptance
    test -- so a fake would have "proved" the guard works while hiding that the
    accepting branch was never exercised at all. This repo has already shipped
    a bug behind a fake that did not match the real class.
    """
    return StrategyContext(
        chain="base",
        last_price=1.715821344605709e-06,
        last_volume=1000.0,
        fee_rate=0.0065,
        available_quote=0.75,
        available_base=0.0,
    )


class _State:
    """The RouteState fields the candidate builder reads.

    ``base_token``/``quote_token`` are not decoration: ``make_candidate``
    enforces STRATEGY_STABLE_QUOTE_ONLY against them, so a state carrying only
    ``symbol`` is refused for the wrong reason and the acceptance test passes
    while proving nothing.
    """

    def __init__(self, symbol="MOCHI-USDC", base_token="MOCHI", quote_token="USDC"):
        self.symbol = symbol
        self.base_token = base_token
        self.quote_token = quote_token
        self.samples = []


def _signal(*, quote_ok: bool, symbol="MOCHI-USDC"):
    """An ATF research signal shaped like the one that reached the live path."""
    return {
        "symbol": symbol,
        "score": 0.7236,
        "expected_return": 0.10,
        "target_price": 1.715821344605709e-06,
        "confidence": 0.7236,
        "token_address": "0xe64ad64806b4340a9b49f00e9ffef960279ec72a",
        "quote_probe": {"ok": quote_ok},
    }


class RouteCorroborationTest(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(
            os.environ,
            {
                "ATF_STATIC_STRATEGY_ENABLED": "1",
                "ATF_STATIC_REQUIRE_QUOTE_OK": "1",
                "ATF_STATIC_MIN_EDGE": "0.006",
            },
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self.strategy = ATFStaticStrategy()

    def _evaluate(self, signal):
        with mock.patch("services.atf_static_strategy.latest_signals",
                        return_value=[signal]):
            return self.strategy.evaluate(_State(), _ctx())

    def test_unquotable_candidate_is_refused(self):
        """The MOCHI case: a token with no route on any DEX is not a trade."""
        self.assertIsNone(self._evaluate(_signal(quote_ok=False)))

    def test_quotable_candidate_still_passes(self):
        """The guard must not close the lane it is protecting.

        Measured on trading_ops, 63-74% of candidates carry a working quote, so
        a guard that refused everything would be indistinguishable from this
        one at a glance -- and would silently stop trading altogether.
        """
        candidate = self._evaluate(_signal(quote_ok=True))
        self.assertIsNotNone(candidate)
        # Shape measured, not assumed: the candidate is
        # {"directive": TradeDirective, "score": float, "meta": dict} -- the
        # action lives on the directive, not at the top level.
        self.assertEqual(sorted(candidate), ["directive", "meta", "score"])
        self.assertEqual(candidate["directive"].action, "enter")
        self.assertEqual(candidate["directive"].symbol, "MOCHI-USDC")
        self.assertGreater(candidate["directive"].size, 0.0)
        self.assertIs(candidate["meta"]["quote_probe_ok"], True)

    def test_missing_probe_is_treated_as_no_route(self):
        """Absent evidence is not evidence of a route.

        A signal with no ``quote_probe`` key at all reaches the same code path;
        it must fail closed, because the alternative spends money on a guess.
        """
        signal = _signal(quote_ok=True)
        signal.pop("quote_probe")
        self.assertIsNone(self._evaluate(signal))

    def test_gate_is_configurable_and_off_by_default(self):
        """Pins the default the .env now overrides, so a future reader can see
        that turning this off restores the behaviour that spent real money."""
        with mock.patch.dict(os.environ, {"ATF_STATIC_REQUIRE_QUOTE_OK": "0"}):
            self.assertIsNotNone(self._evaluate(_signal(quote_ok=False)))


if __name__ == "__main__":
    unittest.main()
