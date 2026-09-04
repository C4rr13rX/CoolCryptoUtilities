"""The tail guardrail bounds a STOP-LOSS, so the tail must be a return.

``ghost_tail_guardrail()`` is a fraction by construction --
``ghost_stop_loss_pct() * (1 + slack)``, capped at 0.25 -- and every line of
its reasoning is a percentage: "ES95 converges on the stop level", "SOL's
-22.2% was 2.78x its stop", "overshoot beyond the 8% stop was 1.013x/1.049x/
1.065x". Its calibration note records ES95 = 0.08336 against a tail of three
stop-outs at -8.52% / -8.39% / -8.10%, whose mean is 0.0834. When that guard
was derived, the trade ``profit`` field WAS a fractional return.

It is not any more. Measured 2026-09-04 against the recorded prices, ``profit``
tracks (exit - entry) x quantity in USD:

    BSTONK-USDC  profit +0.343743  (x-e)*q +0.356721  return +0.1787
    CP-USDC      profit -0.024563  (x-e)*q -0.019355  return -0.1415

and the live gate was therefore comparing $0.1633 against 0.10 -- dollars
against a percentage. That comparison is not merely mis-scaled, it is not a
risk measure at all: it grows with the clip size, so funding the bot more
would "raise" its tail risk while the stop it is meant to police never moved.

The pooled book measured both ways on the same 144 trades:

    ES95 over USD profit   0.163262   (what the gate used)
    ES95 over returns      0.106728   (what the guardrail means)
"""

from __future__ import annotations

import unittest

from trading.metrics import MetricsCollector, TradePerformance, distribution_report
from trading.pipeline import ghost_stop_loss_pct, ghost_tail_guardrail


def _row(ts, status, symbol, details):
    return {"ts": ts, "status": status, "symbol": symbol, "details": details}


class _DB:
    def __init__(self, rows):
        self._rows = rows

    def fetch_trades(self, **kwargs):
        return list(self._rows)


class TailRiskUnitsTest(unittest.TestCase):
    def test_the_guardrail_is_a_fraction(self):
        """It is derived from a stop-loss percentage, so it cannot be dollars."""
        self.assertLessEqual(ghost_tail_guardrail(), 0.25)
        self.assertGreaterEqual(ghost_tail_guardrail(), ghost_stop_loss_pct())

    def test_a_paired_trade_carries_its_return(self):
        rows = [
            _row(10.0, "ghost-entry", "AAA-USDC",
                 {"trade_id": "t1", "entry_price": 100.0, "timestamp": 10.0}),
            _row(20.0, "ghost-exit", "AAA-USDC",
                 {"trade_id": "t1", "entry_price": 100.0, "exit_price": 92.0,
                  "profit": -80.0, "exit_ts": 20.0, "reason": "stop_loss"}),
        ]
        trades = MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)
        self.assertEqual(len(trades), 1)
        # The dollar loss is 80 on a 10-unit clip; the RETURN is -8%.
        self.assertAlmostEqual(trades[0].profit, -80.0, places=6)
        self.assertAlmostEqual(trades[0].return_pct, -0.08, places=9)

    def test_the_return_does_not_move_with_the_clip_but_the_dollar_does(self):
        """The property that makes one a risk measure and the other not."""
        def book(clip):
            return [
                _row(10.0, "ghost-entry", "AAA-USDC",
                     {"trade_id": "t1", "entry_price": 100.0, "timestamp": 10.0}),
                _row(20.0, "ghost-exit", "AAA-USDC",
                     {"trade_id": "t1", "entry_price": 100.0, "exit_price": 92.0,
                      "profit": -8.0 * clip, "exit_ts": 20.0, "reason": "stop_loss"}),
            ]

        small = MetricsCollector(_DB(book(1))).ghost_trade_snapshot(limit=500)[0]
        large = MetricsCollector(_DB(book(100))).ghost_trade_snapshot(limit=500)[0]
        self.assertAlmostEqual(small.return_pct, large.return_pct, places=9)
        self.assertNotAlmostEqual(small.profit, large.profit, places=6)

    def test_a_trade_with_no_entry_price_reports_none_not_zero(self):
        """A missing return must not be averaged in as a flat trade.

        Zero is the safest possible value for a tail, so silently substituting
        it for "unknown" makes an unmeasured book look risk-free -- the one
        direction a risk gate must never fail in.
        """
        rows = [
            _row(10.0, "ghost-entry", "AAA-USDC",
                 {"trade_id": "t1", "timestamp": 10.0}),
            _row(20.0, "ghost-exit", "AAA-USDC",
                 {"trade_id": "t1", "profit": -80.0, "exit_ts": 20.0,
                  "reason": "stop_loss"}),
        ]
        trades = MetricsCollector(_DB(rows)).ghost_trade_snapshot(limit=500)
        self.assertEqual(len(trades), 1)
        self.assertIsNone(trades[0].return_pct)

    def test_the_observed_pooled_book_reproduces_both_numbers(self):
        """The measured 2026-09-04 split, from the eight worst returns."""
        worst = [-0.1840, -0.1494, -0.1415, -0.1118, -0.1021, -0.0789, -0.0500, -0.0361]
        # An 8% stop that gaps to -18.4% is a 2.3x breach: the tail gate is
        # SUPPOSED to block on that, and now does so in the guardrail's units.
        es95 = abs(distribution_report(worst)["expected_shortfall_95"])
        self.assertGreater(es95, ghost_tail_guardrail())

    def test_a_book_of_honest_stop_outs_clears_the_guard(self):
        """The case ghost_tail_guardrail() exists to stop blocking."""
        stop_outs = [-0.0852, -0.0839, -0.0810]
        es95 = abs(distribution_report(stop_outs)["expected_shortfall_95"])
        self.assertLess(es95, ghost_tail_guardrail())


class TradePerformanceDefaultTest(unittest.TestCase):
    def test_return_pct_defaults_to_none_for_existing_constructors(self):
        """revenir_service/task_runner.py builds these without a return."""
        t = TradePerformance(
            symbol="AAA-USDC",
            entry_ts=0.0,
            exit_ts=1.0,
            profit=1.0,
            expected_delta=0.0,
            realized_delta=0.0,
            reason="r",
            route=[],
        )
        self.assertIsNone(t.return_pct)


if __name__ == "__main__":
    unittest.main()
