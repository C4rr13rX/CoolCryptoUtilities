"""A live fill must record how old the price was that it decided on.

THE FAILURE THIS PREVENTS. [3f942885] criterion 5 names TWO things that are
not logged on a live leg: the router path, and the quote-to-submit gap. The
router path shipped at 276523c. The gap did not, and its owner said so in the
reopen note rather than letting it pass as done. So the fill row still cannot
distinguish a decision priced two seconds before the swap from one priced two
minutes before it.

That matters because of what has already been eliminated. At p688 the entry
toll was shown to be one-sided (+13.93bp median on 15 entries, 11 of 15
adverse; exits are zero-mean noise), and depth impact was killed outright --
a $0.75 notional moves no pool on AERO or CBBTC. What survives is that the
decision prices off a reference that is not the price we execute at. Fee-tier
divergence is one arm of that; the reference having simply MOVED between the
decision and the fill is the other, and it is the one no field could see.

WHAT THE FIELD IS, AND THE BOUND ON IT. ``decision_ts`` is the TAPE timestamp
of the sample the decision priced off (``sample_ts`` at both call sites).
``decision_to_fill_sec`` is the wall clock read when the fill row is written,
minus that. The clock is read AFTER the swap settles, so the interval contains
decision latency plus submit plus one or two Base blocks:

    decision_to_fill_sec is an UPPER BOUND on the quote-to-submit gap.

The asymmetry is what makes it worth logging anyway. A small value KILLS the
staleness mechanism -- the price cannot have gone stale in an interval that
also contained settlement. A large value does NOT convict it, because
settlement is inside the number. This test asserts the bound is recorded and
is sane; it asserts nothing about the toll, which is the measurement the field
makes possible, not a result.
"""

from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Recorder:
    """The one bot method ``_leg_cost_fields`` reaches for."""

    def _native_price_with_source(self, chain, route, price, symbol):
        return 2498.77748000803, "test-fixture"


def _leg_fields(**kw):
    """Call the real ``_leg_cost_fields`` unbound, with a minimal receiver."""
    from trading.bot import TradingBot

    defaults = dict(
        leg="buy",
        chain="base",
        symbol="AERO-USDC",
        route=["Aerodrome"],
        price=0.4877,
        notional_usd=0.75,
        gas_spent_native=1.871e-06,
        expected_price=0.4877,
        executed_price=0.4884,
    )
    defaults.update(kw)
    return TradingBot._leg_cost_fields(_Recorder(), **defaults)


class AFillRecordsHowStaleThePriceItDecidedOnWas(unittest.TestCase):
    def test_the_decision_timestamp_survives_into_the_fill_row(self):
        decided = time.time() - 4.0
        fields = _leg_fields(decision_ts=decided)
        self.assertIn(
            "decision_ts",
            fields,
            "the second half of [3f942885] criterion 5 -- the fill row cannot "
            "tell a 2-second decision from a 2-minute one without this",
        )
        self.assertAlmostEqual(decided, fields["decision_ts"], places=6)

    def test_the_gap_is_recorded_beside_the_toll_on_the_same_row(self):
        """A gap with no toll beside it cannot test the staleness mechanism."""
        fields = _leg_fields(decision_ts=time.time() - 9.0)
        self.assertIsNotNone(fields.get("decision_to_fill_sec"))
        self.assertIsNotNone(
            fields.get("realised_slippage_bps"),
            "the mechanism test needs BOTH on the same row",
        )

    def test_the_gap_measures_the_interval_and_is_not_a_constant(self):
        """26 of 26 rows reading 75.0 is what a recorded constant looks like.

        ``slippage_bps`` used to hold the TOLERANCE, and every row agreed with
        every other row because none of them had measured anything. Two
        decisions at different ages must produce two different gaps.
        """
        now = time.time()
        near = _leg_fields(decision_ts=now - 3.0)["decision_to_fill_sec"]
        far = _leg_fields(decision_ts=now - 120.0)["decision_to_fill_sec"]
        self.assertGreater(far, near)
        self.assertAlmostEqual(3.0, near, delta=5.0)
        self.assertAlmostEqual(120.0, far, delta=5.0)

    def test_both_legs_record_it_because_the_entry_is_the_suspect_leg(self):
        """The exit leg is the control: it carries no systematic toll."""
        for leg in ("buy", "sell"):
            with self.subTest(leg=leg):
                fields = _leg_fields(leg=leg, decision_ts=time.time() - 6.0)
                self.assertIn("decision_to_fill_sec", fields)

    def test_an_unknown_decision_time_is_absent_not_zero(self):
        """A zero gap would read as 'decided this instant' and kill the

        mechanism on rows that never measured it. Absent must stay absent.
        """
        for missing in (None, 0.0, "", "not-a-time"):
            with self.subTest(decision_ts=missing):
                fields = _leg_fields(decision_ts=missing)
                self.assertNotIn("decision_to_fill_sec", fields)
                self.assertNotIn("decision_ts", fields)

    def test_a_clock_that_runs_backwards_cannot_book_a_negative_age(self):
        """A tape timestamp ahead of the local clock is a skew, not a negative

        staleness. Clamped at zero so no reader can average a gap below it.
        """
        fields = _leg_fields(decision_ts=time.time() + 30.0)
        self.assertGreaterEqual(fields["decision_to_fill_sec"], 0.0)

    def test_the_row_stays_json_serialisable(self):
        """trade_fills.details is JSON; a non-serialisable field is dropped."""
        json.dumps(_leg_fields(decision_ts=time.time() - 2.0))


if __name__ == "__main__":
    unittest.main()
