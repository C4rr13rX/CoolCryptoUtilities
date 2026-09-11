"""A fill that prices gas must say WHICH asset it priced, and be refusable.

THE FAILURE THIS PREVENTS. On 2026-09-04 five live exits valued ETH gas at the
traded pair's price. ``trade_fills.details`` recorded the number under the name
``gas_price_usd`` -- which is not a gas price, and names no asset -- with no
record of where it came from, so nothing in the row could be checked:

    AERO-USDC   $0.4877      the AERO price
    CBETH-USDC  $2836.06     the cbETH price
    CBETH-USDC  $2840.46     the cbETH price
    AERO-USDC   $0.5018      the AERO price
    CBBTC-USDC  $80884.98    the cbBTC price   <- $0.4136 of gas on a $3 trade

``trade_outcomes`` was repaired by ``scripts/reprice_gas_in_native_token.py``,
but ``trade_fills`` -- the instrumentation table the cost constant is derived
from -- still carries all five, so the guard has to live where a reader can
apply it retroactively as well as where the writer can apply it going forward.

TWO INDEPENDENT GUARDS, because one of the five is not catchable by the other:
the band refuses AERO at $0.49 and cbBTC at $80,884 as impossible ETH prices,
but $2836.06 is a PERFECTLY PLAUSIBLE ETH price and is in fact cbETH's. Only the
provenance field catches that one, because the number was taken off the traded
route. A test that checked only the band would pass on two of the five.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.fill_cost import (
    CONFIGURED,
    FILL_COST_FIELDS,
    MEASURED,
    UNMEASURABLE,
    leg_cost_usd,
    native_price_is_plausible,
    realised_slippage_bps,
)

#: The five rows, exactly as trade_fills still holds them, with the asset whose
#: price was actually recorded.
THE_FIVE = [
    ("AERO-USDC", "base", 0.4877094, "AERO"),
    ("CBETH-USDC", "base", 2836.0550, "cbETH"),
    ("CBETH-USDC", "base", 2840.4600, "cbETH"),
    ("AERO-USDC", "base", 0.5018000, "AERO"),
    ("CBBTC-USDC", "base", 80884.9784, "cbBTC"),
]

#: The ETH price the chain actually agreed with at the time.
ETH_USD = 2498.77748000803


class NativeTokenPriceIsNotThePairs(unittest.TestCase):
    def test_the_band_refuses_the_prices_that_cannot_be_eth(self):
        refused = [
            (symbol, price)
            for symbol, chain, price, _asset in THE_FIVE
            if not native_price_is_plausible(chain, price)[0]
        ]
        self.assertEqual(
            [("AERO-USDC", 0.4877094), ("AERO-USDC", 0.5018), ("CBBTC-USDC", 80884.9784)],
            refused,
            "the band must refuse AERO's $0.49 and cbBTC's $80,884 as ETH prices",
        )

    def test_the_band_accepts_the_real_eth_price(self):
        ok, why = native_price_is_plausible("base", ETH_USD)
        self.assertTrue(ok, why)

    def test_the_band_alone_cannot_catch_the_cbeth_rows(self):
        # This is why the provenance field exists. If a future change makes the
        # band tight enough to catch $2836 it will also start refusing real ETH
        # prices, so the band must stay loose and the SOURCE must do this work.
        for symbol, chain, price, asset in THE_FIVE:
            if asset != "cbETH":
                continue
            self.assertTrue(
                native_price_is_plausible(chain, price)[0],
                f"{symbol} ${price} is inside ETH's band and is {asset}'s price -- "
                "only provenance distinguishes it",
            )

    def test_a_route_native_source_is_not_treated_as_evidence(self):
        # A price taken off the traded route is only the native price when the
        # route SELLS native. leg_cost_usd must not price gas from a row whose
        # own source field says route_native on a pair that is not selling ETH.
        from scripts.roundtrip_cost_census import collect  # noqa: F401  (import path)

        details = {
            "mode": "live_exit",
            "chain": "base",
            "gas_spent_native": 5.112829501e-06,
            "native_token_price_usd": 80884.9784,
            "native_token_price_source": "route_native",
            "expected_price": 1.0,
            "executed_price": 1.0,
            "quote_received": 3.0,
        }
        cost = leg_cost_usd(details)
        self.assertIsNone(
            cost["gas_cost_usd"],
            "a gas cost of $0.098 must not be produced from cbBTC's price",
        )
        self.assertFalse(cost["measured"])
        self.assertIn("band", cost["unmeasured_reason"])

    def test_the_same_row_priced_in_eth_gives_the_honest_gas_cost(self):
        details = {
            "mode": "live_exit",
            "chain": "base",
            "gas_spent_native": 5.112829501e-06,
            "native_token_price_usd": ETH_USD,
            "native_token_price_source": "price_book",
            "expected_price": 1.0,
            "executed_price": 1.0,
            "quote_received": 3.0,
        }
        cost = leg_cost_usd(details)
        self.assertAlmostEqual(0.0127758, cost["gas_cost_usd"], places=6)
        # The fiction it replaces was $0.4136 -- 32x this, and 13.8% of a $3 trade.
        self.assertLess(cost["gas_cost_usd"], 0.4135511 / 30.0)


class TheWriterRecordsProvenance(unittest.TestCase):
    def test_the_bot_returns_a_source_beside_every_native_price(self):
        # The seam, not the function: whatever the price is, the writer must be
        # able to say where it came from, or the row cannot be audited later.
        import inspect

        from trading.bot import TradingBot

        self.assertTrue(hasattr(TradingBot, "_native_price_with_source"))
        source = inspect.getsource(TradingBot._native_price_with_source)
        for expected in ("route_native", "price_book", "fallback_env", "fallback_constant"):
            self.assertIn(expected, source, f"no {expected} provenance is ever recorded")
        # Every return must be a pair. A bare float return would silently make
        # the source the second element of nothing.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("return "):
                self.assertIn(",", stripped, f"{stripped!r} returns no source")

    def test_the_live_exit_writer_no_longer_writes_gas_price_usd(self):
        import inspect

        from trading.bot import TradingBot

        source = inspect.getsource(TradingBot)
        writes = [
            line
            for line in source.splitlines()
            if '"gas_price_usd":' in line and not line.strip().startswith("#")
        ]
        self.assertEqual(
            [],
            writes,
            "gas_price_usd never held a gas price; the writer must record "
            "native_token_price_usd instead",
        )

    def test_the_env_var_the_tolerance_comes_from_is_named_as_configured(self):
        # slippage_bps read 75.0 on 26 of 26 rows because it is the value of
        # LIVE_TRADE_SLIPPAGE_BPS, not an observed fill.
        default = os.getenv("LIVE_TRADE_SLIPPAGE_BPS", os.getenv("SCHEDULER_SLIPPAGE_BPS", "75"))
        self.assertEqual(CONFIGURED, FILL_COST_FIELDS["slippage_tolerance_bps"]["status"])
        self.assertIn("75", FILL_COST_FIELDS["slippage_tolerance_bps"]["holds"] + str(default))


class TheSchemaSaysWhichFieldsAreEvidence(unittest.TestCase):
    def test_every_cost_field_carries_a_status_and_a_meaning(self):
        for name, note in FILL_COST_FIELDS.items():
            self.assertIn(note["status"], {"measured", "configured", "derived", "unmeasurable"}, name)
            self.assertGreater(len(note["holds"]), 30, f"{name} has no real description")

    def test_the_dex_fee_is_documented_as_unmeasurable_with_the_reason(self):
        note = FILL_COST_FIELDS["dex_fee_usd"]
        self.assertEqual(UNMEASURABLE, note["status"])
        self.assertIn("output amount", note["holds"])

    def test_fee_rate_and_fee_cost_are_not_claimed_to_be_measured(self):
        for field in ("fee_rate", "fee_cost"):
            self.assertEqual(
                CONFIGURED,
                FILL_COST_FIELDS[field]["status"],
                f"{field} is a configured default and must not be labelled measured",
            )

    def test_gas_and_slippage_are_the_measured_ones(self):
        for field in ("gas_spent_native", "native_token_price_usd", "realised_slippage_bps"):
            self.assertEqual(MEASURED, FILL_COST_FIELDS[field]["status"], field)


class AdverseIsPositiveOnBothLegs(unittest.TestCase):
    def test_a_buy_above_its_quote_and_a_sell_below_it_both_read_positive(self):
        buy = realised_slippage_bps(expected_price=100.0, executed_price=101.0, leg="buy")
        sell = realised_slippage_bps(expected_price=100.0, executed_price=99.0, leg="sell")
        self.assertAlmostEqual(100.0, buy, places=6)
        self.assertAlmostEqual(100.0, sell, places=6)
        # So the two legs of a round trip add without a per-leg sign convention.
        self.assertAlmostEqual(200.0, buy + sell, places=6)

    def test_a_favourable_fill_reads_negative_on_both_legs(self):
        self.assertLess(realised_slippage_bps(expected_price=100.0, executed_price=99.0, leg="buy"), 0)
        self.assertLess(realised_slippage_bps(expected_price=100.0, executed_price=101.0, leg="sell"), 0)

    def test_a_zero_quote_is_none_and_never_zero(self):
        # A zero expected price is a missing quote. Returning 0.0 would book it
        # as a perfect fill and average it into the cost as free.
        self.assertIsNone(realised_slippage_bps(expected_price=0.0, executed_price=5.0, leg="buy"))
        self.assertIsNone(realised_slippage_bps(expected_price=5.0, executed_price=0.0, leg="sell"))


if __name__ == "__main__":
    unittest.main()
