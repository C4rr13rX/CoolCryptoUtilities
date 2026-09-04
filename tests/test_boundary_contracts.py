"""Every expensive bug in this repo, as a test that now catches it.

Constraint 2 -- check type, shape, units, range and time at every boundary --
was an instruction, and instructions get skipped because the code always looks
right locally. These are the actual values that shipped, each asserted to be
refused now.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import boundary_contracts as bc  # noqa: E402
from services.boundary_contracts import BoundaryViolation  # noqa: E402


class HistoricalBugsAreCaughtTest(unittest.TestCase):
    """Each case is a value that really reached production and cost money."""

    def test_wrong_units_price(self):
        """0.75 USDC read as raw base units arrives as 750000."""
        with self.assertRaises(BoundaryViolation):
            bc.usd(750000, name="entry_price")

    def test_exit_sized_in_raw_base_units(self):
        """The exit-sizing bug: 18-decimal raw where a quantity was expected."""
        with self.assertRaises(BoundaryViolation):
            bc.token_quantity(2.6e14, name="exit_size", decimals=18)

    def test_percent_passed_as_a_fraction(self):
        with self.assertRaises(BoundaryViolation):
            bc.fraction(55.0, name="win_rate")

    def test_milliseconds_passed_as_epoch_seconds(self):
        """A ms timestamp read as seconds disables every staleness guard."""
        with self.assertRaises(BoundaryViolation):
            bc.epoch_seconds(1788498423603, name="entry_ts")

    def test_v4_pool_id_used_as_a_token_address(self):
        """66 chars is a tx hash or a pool id, never an address."""
        with self.assertRaises(BoundaryViolation):
            bc.address("0x" + "a" * 64, name="token")

    def test_symbol_passed_where_an_address_belongs(self):
        with self.assertRaises(BoundaryViolation):
            bc.address("CBETH", name="token")

    def test_nan_never_passes(self):
        """NaN compares False against every threshold, so every guard passes it."""
        with self.assertRaises(BoundaryViolation):
            bc.number(float("nan"), name="price")

    def test_infinity_never_passes(self):
        with self.assertRaises(BoundaryViolation):
            bc.number(float("inf"), name="price")

    def test_a_string_where_a_list_belongs(self):
        """Iterating 'USDC' yields characters, and every consumer is wrong."""
        with self.assertRaises(BoundaryViolation):
            bc.sequence("USDC", name="symbols")

    def test_bool_where_a_number_belongs(self):
        """bool is an int in Python, so True reaches arithmetic as 1.0."""
        with self.assertRaises(BoundaryViolation):
            bc.number(True, name="size")

    def test_missing_keys_are_named(self):
        with self.assertRaises(BoundaryViolation):
            bc.mapping({"symbol": "CBETH"}, name="fill",
                       required=("symbol", "tx_hash", "sold"))


class RealValuesPassSilentlyTest(unittest.TestCase):
    """A guard that refuses good values is worse than no guard."""

    def test_the_live_wallet_and_clip_sizes_pass(self):
        self.assertAlmostEqual(bc.usd(19.196394, name="wallet"), 19.196394)
        self.assertAlmostEqual(bc.usd(0.75, name="clip"), 0.75)
        self.assertAlmostEqual(bc.usd(-0.1405, name="pnl"), -0.1405)

    def test_a_real_cbeth_position_size_passes(self):
        self.assertAlmostEqual(
            bc.token_quantity(0.000262122199594547, name="size", decimals=18),
            0.000262122199594547)

    def test_a_real_win_rate_passes(self):
        self.assertAlmostEqual(bc.fraction(0.55, name="rate"), 0.55)

    def test_a_real_timestamp_passes(self):
        self.assertAlmostEqual(bc.epoch_seconds(1788498423, name="ts"),
                               1788498423)

    def test_real_addresses_pass(self):
        for addr in ("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                     "0x2ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22"):
            self.assertEqual(bc.address(addr, name="token"), addr)

    def test_native_sentinels_pass(self):
        self.assertEqual(bc.address("native", name="sell"), "native")

    def test_a_real_route_passes(self):
        self.assertEqual(bc.sequence(["CBETH", "USDC"], name="route"),
                         ["CBETH", "USDC"])


class FailureDirectionTest(unittest.TestCase):
    def test_production_logs_rather_than_halting_trading(self):
        """A contract bug must not stop live trading on an assertion."""
        import unittest.mock as mock

        with mock.patch.object(bc, "_strict", lambda: False):
            # Returns the offending value instead of raising, so the caller
            # keeps its own downstream guards.
            self.assertEqual(bc.usd(750000, name="entry_price"), 750000.0)


if __name__ == "__main__":
    unittest.main()
