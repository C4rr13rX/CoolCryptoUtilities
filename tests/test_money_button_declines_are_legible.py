"""money_button must say why it declined.

The lane returned None from eleven different places and recorded nothing, so
"money_button never fires" and "money_button fires and loses" looked
identical from outside. That is how a fabricated 16W/60L record survived for
days: there was no measurement to contradict it.

scripts/money_button_gate_census.py replays the real strategy over stored
ticks and counts the reasons, which only works while every decline path sets
one. These tests pin that -- and pin the harness contract the census depends
on, because getting it wrong silently converted 44 genuine fires into
reasonless declines the first time this was run.

Nothing here asserts a threshold. The gates are the strategy's business; this
is about being able to see them.
"""

from __future__ import annotations

import unittest

import numpy as np

from trading.strategies.base import StrategyContext
from trading.strategies.money_button import MoneyButtonStrategy


class _State:
    def __init__(self, samples, symbol="TREND-USDC"):
        self.samples = samples
        self.symbol = symbol
        base, _, quote = symbol.partition("-")
        self.base_token = base
        self.quote_token = quote


def _ctx(price, fee=0.003):
    return StrategyContext(
        chain="base",
        last_price=price,
        last_volume=1000.0,
        fee_rate=fee,
        available_quote=10.0,
        available_base=0.0,
    )


def _ramp(n=40, minutes=8.0, step=0.004, start=1.0, t0=1_700_000_000.0):
    """A clean, steadily rising series on real volume."""
    out = []
    price = start
    for i in range(n):
        price *= 1.0 + step
        out.append((t0 + i * minutes * 60.0, price, 1000.0 + i))
    return out


class EveryDeclinePathRecordsAReason(unittest.TestCase):
    def test_too_few_samples(self):
        strat = MoneyButtonStrategy()
        self.assertIsNone(strat.evaluate(_State(_ramp(n=3)), _ctx(1.0)))
        self.assertEqual(strat.last_decline, "too_few_samples")

    def test_window_too_short(self):
        """Enough prints, but all inside a couple of minutes."""
        strat = MoneyButtonStrategy()
        samples = _ramp(n=30, minutes=0.05)
        self.assertIsNone(strat.evaluate(_State(samples), _ctx(samples[-1][1])))
        self.assertEqual(strat.last_decline, "window_too_short")

    def test_frozen_feed(self):
        """A repeated seed price is the failure this lane is most exposed to."""
        strat = MoneyButtonStrategy()
        samples = [(1_700_000_000.0 + i * 300.0, 1.0, 1000.0) for i in range(30)]
        self.assertIsNone(strat.evaluate(_State(samples), _ctx(1.0)))
        self.assertEqual(strat.last_decline, "feed_frozen")

    def test_downtrend_is_not_aligned(self):
        strat = MoneyButtonStrategy()
        samples = _ramp(n=30, step=-0.004)
        self.assertIsNone(strat.evaluate(_State(samples), _ctx(samples[-1][1])))
        self.assertEqual(strat.last_decline, "momentum_not_aligned")

    def test_no_price(self):
        strat = MoneyButtonStrategy()
        self.assertIsNone(strat.evaluate(_State(_ramp()), _ctx(0.0)))
        self.assertEqual(strat.last_decline, "no_price")

    def test_a_reason_is_always_set_when_declining(self):
        """The property the census actually relies on.

        Swept across trend strengths and fee rates so the assertion covers
        the cost gate and the confidence floor too, not just the cheap ones.
        """
        for step in (-0.01, -0.001, 0.0, 0.0005, 0.002, 0.01, 0.05):
            for fee in (0.0005, 0.003, 0.0065, 0.02):
                strat = MoneyButtonStrategy()
                samples = _ramp(n=30, step=step)
                result = strat.evaluate(_State(samples), _ctx(samples[-1][1], fee=fee))
                if result is None:
                    self.assertIsNotNone(
                        strat.last_decline,
                        "declined with no reason at step=%s fee=%s" % (step, fee),
                    )
                else:
                    self.assertIsNone(
                        strat.last_decline,
                        "fired but left a stale decline reason",
                    )

    def test_reason_is_cleared_between_evaluations(self):
        """A stale reason read back as current would misattribute a fire."""
        strat = MoneyButtonStrategy()
        strat.evaluate(_State(_ramp(n=3)), _ctx(1.0))
        self.assertEqual(strat.last_decline, "too_few_samples")
        samples = _ramp(n=30, step=0.05)
        if strat.evaluate(_State(samples), _ctx(samples[-1][1], fee=0.0005)) is not None:
            self.assertIsNone(strat.last_decline)


class TheLaneStillFires(unittest.TestCase):
    """Instrumentation must not have changed a single decision.

    A strong, clean, cheap-to-trade uptrend has to produce a candidate; if
    this stops passing, a `return None` was rewritten into something that
    declines rather than something that reports.
    """

    def test_a_strong_cheap_uptrend_produces_a_candidate(self):
        strat = MoneyButtonStrategy()
        samples = _ramp(n=30, step=0.02)
        result = strat.evaluate(_State(samples), _ctx(samples[-1][1], fee=0.0005))
        self.assertIsNotNone(result, "the lane declined a 2%/print uptrend at 5bp fees")
        self.assertEqual(result["meta"]["strategy"], "money_button")
        self.assertGreater(result["directive"].expected_return, 0.0)

    def test_the_census_harness_contract_holds(self):
        """make_candidate reads the token fields, not just samples.

        The first census run left them unset, so every genuine fire came back
        as a decline with no reason -- 44 of them, reported as zero.
        """
        strat = MoneyButtonStrategy()
        samples = _ramp(n=30, step=0.02)

        class _NoTokens:
            def __init__(self, s):
                self.samples = s

        self.assertIsNone(
            strat.evaluate(_NoTokens(samples), _ctx(samples[-1][1], fee=0.0005)),
            "a state without token fields must not silently produce a candidate",
        )
        self.assertIsNone(
            strat.last_decline,
            "make_candidate's refusal is not a gate decline and must not be "
            "reported as one",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
