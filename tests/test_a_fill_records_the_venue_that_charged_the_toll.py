"""A live fill must record the router path, not just the toll it produced.

THE FAILURE THIS PREVENTS. Every live fill in the book records HOW MUCH
slippage it suffered and nothing about WHERE. ``_leg_cost_fields`` was handed
``route`` from both call sites -- ``trading/bot.py`` live_entry and live_exit --
and consumed it only to price the native token for gas, then dropped it. The
result, measured at pass 710:

    live fills in trade_fills            38
    live fills carrying a router path     0

That is what blocked [3f942885]. The surviving explanation for an entry toll
that is one-sided, size-invariant AND independent of tick age -- latency,
depth impact and CEX-to-DEX basis were each eliminated by measurement at
p688 -- is that the decision prices off an AGGREGATE last-trade reference
across a token's pools while execution hits ONE pool at its marginal price.
The gap is spread plus fee-tier difference, one-sided on the taker side by
construction: a 0.05% reference against a 0.30% execution tier is 25bp by
arithmetic, and the measured entry toll is 13.9-14bp, the right order.

Testing that costs nothing and needs exactly one field: the tier the router
actually used. Without it the mechanism cannot be confirmed or killed, so the
~14bp -- 87% of the ~16bp floor on the symbols we have actually filled -- stays
unattributable and no cost model can be corrected with confidence.

WHAT THIS TEST DOES NOT CLAIM. It does not assert the toll tracks the fee
tier; that is the measurement the field makes possible, not a result. It
asserts only that the venue is recorded alongside the toll, and that a fill
with no route says so rather than inventing one.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Recorder:
    """The two fields of the bot that ``_leg_cost_fields`` actually touches."""

    def _native_price_with_source(self, chain, route, price, symbol):
        return 2498.77748000803, "test-fixture"


def _leg_fields(**kw):
    """Call the real ``_leg_cost_fields`` unbound, with a minimal receiver.

    Importing trading.bot pulls the whole bot in; binding the real function to
    a stub receiver exercises the production code path without constructing a
    live bot, which is what the sibling cost tests do.
    """
    from trading.bot import TradingBot

    defaults = dict(
        leg="buy",
        chain="base",
        symbol="WETH-USDC",
        price=2498.77748000803,
        notional_usd=6.0,
        gas_spent_native=1.871e-06,
        expected_price=2498.77748000803,
        executed_price=2502.2,
    )
    defaults.update(kw)
    return TradingBot._leg_cost_fields(_Recorder(), **defaults)


class AFillRecordsTheVenueThatChargedTheToll(unittest.TestCase):
    def test_the_router_path_survives_into_the_fill_row(self):
        fields = _leg_fields(route=["UniswapV3", "0.05%"])
        self.assertIn(
            "route",
            fields,
            "the router path was reached and dropped -- this is the exact gap "
            "that blocked [3f942885]",
        )
        self.assertEqual(["UniswapV3", "0.05%"], fields["route"])

    def test_the_toll_and_the_venue_are_recorded_together(self):
        """A venue with no toll beside it cannot test the fee-tier mechanism."""
        fields = _leg_fields(route=["UniswapV3", "0.30%"])
        self.assertIsNotNone(fields.get("route"))
        self.assertIsNotNone(
            fields.get("realised_slippage_bps"),
            "the mechanism test needs BOTH on the same row",
        )

    def test_both_legs_record_it_because_the_toll_is_one_sided(self):
        """The entry leg carries the whole toll, so the sell leg is the control."""
        for leg in ("buy", "sell"):
            with self.subTest(leg=leg):
                fields = _leg_fields(leg=leg, route=["Aerodrome"])
                self.assertEqual(["Aerodrome"], fields["route"])

    def test_an_absent_route_is_recorded_as_absent_not_invented(self):
        """An empty route must read as 'not known', never as a default venue.

        A fabricated venue is worse than a missing one: it would attribute a
        toll to a pool that did not charge it.
        """
        for missing in (None, []):
            with self.subTest(route=missing):
                fields = _leg_fields(route=missing)
                self.assertEqual([], fields["route"])

    def test_the_route_is_stored_as_plain_strings_so_the_row_stays_json(self):
        """trade_fills.details is JSON; a non-serialisable hop would be dropped."""
        import json

        class _Hop:
            def __str__(self):
                return "UniswapV3"

        fields = _leg_fields(route=[_Hop()])
        self.assertEqual(["UniswapV3"], fields["route"])
        json.dumps(fields)  # must not raise


if __name__ == "__main__":
    unittest.main()
