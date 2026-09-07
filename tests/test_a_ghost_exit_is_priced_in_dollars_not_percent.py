"""The ghost book must not add fractions to dollars, and a timer must not sell a fee.

Two failures, one shape: a return compared against zero instead of against cost.

MEASURED 2026-09-06 on the live 5-day ghost book (242 paired round trips).

1. UNIT.  ``services/atf_static_strategy.py`` wrote ``(mark/entry) - 1`` -- a
   FRACTION -- into ``details["profit"]``, the same column ``trading/bot.py``
   writes USD into. 104 of the 242 trades were scout rows and 104/104 of them
   had ``profit`` exactly equal to their own return:

       strategy              n    profit==return_pct    sum(profit)
       atf_static_scout    104         104/104            +2.06867
       atf_static           45           7/45             +1.49895
       (29 others)          93           0/93             +0.01037

   The live gate, ``StrategyLedger`` and graduation each sum that column into
   one number, so 43% of the evidence was in the wrong unit AND had never been
   charged a fee. Denominating it at the live clip and charging the round trip:

       net           +3.578  ->  +11.512
       expectancy   +0.0148  ->  +0.0476  USD/trade
       payoff ratio   1.081  ->    2.698
       loss rate      0.331  ->    0.512  <- fees turn flat trades into losses
       loss streak        7  ->       13

   The risk numbers got WORSE. That is the finding: the fractional unit was
   reporting the book as safe by never charging it for what it lost.

2. TIMER.  ``max_hold`` fired on ``profit > 0.0``. 83 exits closed on that
   timer and 47 (56.6%) marked out inside one round trip -- CBXRP +0.045%,
   AERO +0.004%, CBBTC +0.333% -- each booked a "win", each a guaranteed loss
   once the 0.386% round trip on a $6 clip is charged. Across all 242 trades,
   80 exits (33.1%) closed inside the cost: +0.0597 of gross return between
   them, -$1.4951 net.

Each test below fails against the pre-fix code.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.roundtrip_cost import (  # noqa: E402
    ghost_clip_usd,
    net_profit_usd,
    roundtrip_cost_rate,
    roundtrip_cost_usd,
)
from trading.metrics import _profit_usd  # noqa: E402


class RoundTripCostTest(unittest.TestCase):
    """The cost model is size-dependent, and a flat trade loses a round trip."""

    def test_a_trade_that_went_nowhere_loses_exactly_one_round_trip(self):
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            self.assertAlmostEqual(net_profit_usd(0.0), -roundtrip_cost_usd(6.0), places=9)
            self.assertLess(net_profit_usd(0.0), 0.0)

    def test_the_fixed_part_does_not_shrink_with_size(self):
        """A $0.75 trade and a $6.00 trade do not pay the same percentage."""
        small = roundtrip_cost_rate(0.75)
        large = roundtrip_cost_rate(6.00)
        self.assertGreater(small, large)
        # ...and never below the pure proportional rate, however large.
        self.assertGreaterEqual(roundtrip_cost_rate(10_000.0), 0.003187 - 1e-12)

    def test_the_clip_is_never_zero(self):
        """A simulated trade priced at nothing books a bare fraction again."""
        with mock.patch.dict(
            os.environ, {"LIVE_MIN_CLIP_USD": "0", "GHOST_MIN_TRADE_USD": "0"}, clear=False
        ):
            self.assertGreater(ghost_clip_usd(), 0.0)


class LegacyRowNormalisationTest(unittest.TestCase):
    """The gate reads a 5-day window, so legacy rows must be converted on read."""

    def test_a_legacy_scout_row_is_converted_from_its_own_prices(self):
        # BEFORE: profit came back as 0.00333 -- a 0.333% return read as a third
        # of a cent of profit. AFTER: USD at the clip, net of the round trip,
        # which is NEGATIVE because 0.333% does not cover a 0.386% round trip.
        details = {"profit": 0.00333, "entry_price": 100.0, "exit_price": 100.333}
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            got = _profit_usd(details, "atf_static_scout", 0.00333)
            self.assertAlmostEqual(got, 0.00333 * 6.0 - roundtrip_cost_usd(6.0), places=9)
        self.assertLess(got, 0.0, "a return inside the round trip is not a win")
        self.assertNotAlmostEqual(got, 0.00333, places=6)

    def test_a_marked_row_is_trusted_as_written(self):
        """Once the writer stamps the unit, the reader must not convert again."""
        details = {"profit": -0.0032, "profit_unit": "usd",
                   "entry_price": 100.0, "exit_price": 100.333}
        self.assertAlmostEqual(
            _profit_usd(details, "atf_static_scout", 0.00333), -0.0032, places=9
        )

    def test_another_writers_dollars_are_left_alone(self):
        """trading/bot.py already records USD; converting it would double-charge."""
        details = {"profit": -0.009141, "entry_price": 10.0, "exit_price": 10.0007}
        self.assertAlmostEqual(
            _profit_usd(details, "obv_accumulation@1w", 0.00007), -0.009141, places=9
        )


class SourceContractTest(unittest.TestCase):
    """The scout's own source must show the two corrected comparisons.

    Read from the source because the exit loop needs a live db, a corroborated
    feed and an open position to reach; these assert on what the code DOES
    compare, and both strings are absent from the pre-fix file.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "services" / "atf_static_strategy.py").read_text(encoding="utf-8")

    def test_the_hold_timer_compares_against_cost_not_zero(self):
        self.assertNotIn(
            "hold_forces_exit or profit > 0.0", self.src,
            "max_hold compared a gross return against zero: 47 of 83 timer "
            "exits closed inside one round trip",
        )
        self.assertIn("hold_forces_exit or profit > cost_rate", self.src)

    def test_the_exit_row_records_usd_and_says_so(self):
        self.assertIn('"profit": profit_usd', self.src)
        self.assertIn('"profit_unit": "usd"', self.src)
        self.assertNotIn('"profit": profit,', self.src)

    def test_the_ledger_is_paid_the_same_unit_as_the_row(self):
        # StrategyLedger.record() gets economic_profit (USD) from trading/bot.py;
        # sending it a fraction from here made graduation score two units.
        self.assertIn(
            "_record_ghost_outcome(SCOUT_STRATEGY_ID, profit_usd", self.src
        )

    def test_the_target_floor_cannot_be_set_below_the_round_trip(self):
        self.assertIn("max(min_profit, target_return, cost_rate)", self.src)


class SingleFormulaTest(unittest.TestCase):
    """One cost formula. Two copies is how the scout came to charge nothing."""

    def test_the_bot_does_not_carry_its_own_copy_of_the_constants(self):
        src = (ROOT / "trading" / "bot.py").read_text(encoding="utf-8")
        head = src[: src.index("def _lattice_refusal")]
        self.assertIn("from services.roundtrip_cost import roundtrip_cost_rate", head)
        self.assertNotIn('float(os.getenv("ROUNDTRIP_FEE_FIXED_USD", "0.004047"))', head)

    def test_both_lanes_price_the_same_round_trip_identically(self):
        from trading.bot import TradingBot

        bot = TradingBot.__new__(TradingBot)
        with mock.patch.dict(os.environ, {"LIVE_MIN_CLIP_USD": "6.0"}, clear=False):
            self.assertAlmostEqual(
                bot._roundtrip_fee_rate(notional_hint=6.0),
                roundtrip_cost_rate(6.0),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
