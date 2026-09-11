"""The BUY leg must carry a priced cost, or a round trip cannot be split.

THE FAILURE THIS PREVENTS, measured 2026-09-11 over all 1394 rows of
``trade_fills``. The two live legs recorded different things:

    field                  live_entry   live_exit
    gas_spent_native          20/20       18/18
    the native token's price   0/20       18/18
    any fee field             0/20       18/18

So the buy leg had a receipt and nothing to value it with, and the sell leg
carried a single scalar for the whole round trip. That is the mechanical reason
nobody could say which leg the cost sat in -- and the reason the published
constant, $0.004047 fixed plus 0.3187% of notional, could not be checked against
the legs it claims to describe.

The re-derivation once both legs are priced: gas is $0.002536 on the buy and
$0.002157 on the sell at a median ETH $2498.78, so the buy leg carries 54% of
the gas and the sell leg 46%. Realised slippage is +21.73 bps on the buy against
+6.04 bps on the sell -- the buy leg is where the cost actually is, by a factor
of 3.6, which no scalar booked against the exit could ever have shown.

WHAT THIS TEST CHECKS IS THE SEAM, NOT THE ARITHMETIC: that the live entry
writer puts a priced cost on the buy leg at all, and that the decomposition
refuses to report a leg as measured when it is not.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.fill_cost import leg_cost_usd

ETH_USD = 2498.77748000803


class TheLiveEntryWriterPricesTheBuyLeg(unittest.TestCase):
    def setUp(self):
        from trading.bot import TradingBot

        self.bot_source = inspect.getsource(TradingBot)

    def test_the_live_entry_fill_is_built_through_the_leg_cost_helper(self):
        # The live_entry extra used to be four keys, none of which priced
        # anything: mode, trade_id, quote_spent, gas_spent_native, slippage_bps.
        marker = '"mode": "live_entry",'
        self.assertIn(marker, self.bot_source)
        block = self.bot_source.split(marker, 1)[1][:1200]
        self.assertIn("_leg_cost_fields(", block)
        self.assertIn('leg="buy"', block)

    def test_the_live_exit_fill_is_built_through_the_same_helper(self):
        marker = '"mode": "live_exit",'
        self.assertIn(marker, self.bot_source)
        block = self.bot_source.split(marker, 1)[1][:1600]
        self.assertIn("_leg_cost_fields(", block)
        self.assertIn('leg="sell"', block)

    def test_the_helper_emits_every_field_a_leg_split_needs(self):
        from trading.bot import TradingBot

        source = inspect.getsource(TradingBot._leg_cost_fields)
        for field in (
            "leg",
            "notional_usd",
            "native_token_price_usd",
            "native_token_price_source",
            "gas_cost_usd",
            "realised_slippage_bps",
            "slippage_cost_usd",
            "leg_cost_usd",
        ):
            self.assertIn(f'"{field}"', source, f"{field} is not written to the fill")

    def test_the_helper_does_not_invent_a_dex_fee(self):
        from trading.bot import TradingBot

        source = inspect.getsource(TradingBot._leg_cost_fields)
        self.assertNotIn(
            '"dex_fee_usd"',
            source,
            "an AMM's fee is inside the output amount; writing it as a separate "
            "number means writing a configured constant and calling it measured",
        )

    def test_the_ghost_legs_are_labelled_so_they_pair(self):
        for marker in ('"mode": "ghost_entry",', '"mode": "ghost_exit",'):
            self.assertIn(marker, self.bot_source)
            block = self.bot_source.split(marker, 1)[1][:900]
            self.assertIn('"leg"', block, f"{marker} writes no leg label")
            self.assertIn('"notional_usd"', block, f"{marker} writes no notional")


class ALegIsOnlyMeasuredWhenBothHalvesAre(unittest.TestCase):
    BUY = {
        "mode": "live_entry",
        "leg": "buy",
        "chain": "base",
        "gas_spent_native": 9.8874e-07,
        "native_token_price_usd": ETH_USD,
        "native_token_price_source": "price_book",
        "expected_price": 0.54446397,
        "executed_price": 0.54385728,
        "notional_usd": 0.75,
    }

    def test_a_fully_instrumented_buy_leg_reports_measured(self):
        cost = leg_cost_usd(self.BUY)
        self.assertTrue(cost["measured"], cost["unmeasured_reason"])
        self.assertEqual("buy", cost["leg"])
        self.assertAlmostEqual(0.00247, cost["gas_cost_usd"], places=5)
        # This buy filled BELOW its quote, so its slippage is favourable and the
        # sign convention must show that as negative, not as a cost.
        self.assertLess(cost["realised_slippage_bps"], 0.0)
        self.assertLess(cost["slippage_cost_usd"], 0.0)

    def test_a_buy_leg_with_no_native_price_is_not_measured(self):
        # This is every live entry ever written before this pass: a receipt with
        # nothing to value it with. It must not report a cost of zero.
        row = dict(self.BUY)
        row.pop("native_token_price_usd")
        row.pop("native_token_price_source")
        cost = leg_cost_usd(row)
        self.assertIsNone(cost["gas_cost_usd"])
        self.assertFalse(cost["measured"])
        self.assertNotEqual(0.0, cost["total_cost_usd"])

    def test_the_two_legs_add_to_a_round_trip(self):
        sell = {
            "mode": "live_exit",
            "leg": "sell",
            "chain": "base",
            "gas_spent_native": 8.4025e-07,
            "native_token_price_usd": ETH_USD,
            "native_token_price_source": "price_book",
            "expected_price": 0.53752556,
            "executed_price": 0.53409904,
            "notional_usd": 1.36692,
        }
        buy_cost = leg_cost_usd(self.BUY)
        sell_cost = leg_cost_usd(sell)
        self.assertTrue(buy_cost["measured"] and sell_cost["measured"])
        total = buy_cost["total_cost_usd"] + sell_cost["total_cost_usd"]
        # A round trip whose cost is a single scalar cannot produce these two
        # numbers, which is the whole complaint.
        self.assertNotAlmostEqual(buy_cost["total_cost_usd"], sell_cost["total_cost_usd"])
        self.assertGreater(total, 0.0)

    def test_the_mode_decides_the_leg_when_no_label_was_written(self):
        # Every historical row has a mode and no leg, so the decomposition must
        # keep working on them or the 1394 rows already in the book are lost.
        row = dict(self.BUY)
        row.pop("leg")
        self.assertEqual("buy", leg_cost_usd(row)["leg"])
        row["mode"] = "ghost_exit"
        self.assertEqual("sell", leg_cost_usd(row)["leg"])


class TheCensusSplitsTheBookIntoLegs(unittest.TestCase):
    def test_the_census_reports_both_legs_and_quarantines_the_bad_rows(self):
        from scripts.roundtrip_cost_census import collect

        data = collect()
        live = data["lanes"]["live"]
        ghost = data["lanes"]["ghost"]

        # Criterion: a round trip can be split, over at least 100 round trips.
        self.assertGreaterEqual(
            live["round_trips"] + ghost["round_trips"],
            100,
            "fewer than 100 paired round trips exist to split",
        )
        for lane in (live, ghost):
            for key in (
                "buy_gas_usd_median",
                "sell_gas_usd_median",
                "buy_slippage_bps_median",
                "sell_slippage_bps_median",
            ):
                self.assertIn(key, lane)

        # The five mispriced rows must never be averaged into a constant.
        self.assertGreater(
            len(live["quarantined_legs"]),
            0,
            "the AERO rows priced at $0.49 are still in trade_fills and must be "
            "quarantined, not averaged",
        )
        for row in live["quarantined_legs"]:
            self.assertTrue(row["reason"])

        # And a round trip whose legs are in different lanes is not a cost sample.
        for row in data["cross_lane_round_trips"]:
            self.assertNotEqual(
                str(row["entry_mode"]).split("_")[0], str(row["exit_mode"]).split("_")[0]
            )

    def test_the_constant_is_re_derived_rather_than_restated(self):
        from scripts.roundtrip_cost_census import collect
        from services.roundtrip_cost import DEFAULT_FIXED_USD, DEFAULT_RATE

        r = collect()["rederived"]
        self.assertEqual(DEFAULT_FIXED_USD, r["fixed_usd_published"])
        self.assertEqual(DEFAULT_RATE, r["rate_published"])
        self.assertIsNotNone(r["fixed_usd"], "the fixed part could not be re-derived")
        self.assertIsNotNone(r["rate"], "the proportional part could not be re-derived")
        # Both re-derived numbers must come from the receipts, so neither may be
        # exactly the published value -- that would mean the census is echoing
        # the constant back rather than measuring it.
        self.assertNotAlmostEqual(r["fixed_usd"], r["fixed_usd_published"], places=9)
        self.assertNotAlmostEqual(r["rate"], r["rate_published"], places=9)


if __name__ == "__main__":
    unittest.main()
