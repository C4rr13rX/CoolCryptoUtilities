"""The tail guardrail must measure the stop FAILING, not the stop WORKING.

ES95 is the mean of the worst 5% of ghost outcomes. When a strategy behaves,
those worst outcomes are exactly its stop-outs, so ES95 converges on the
stop-loss level. Comparing it against a constant 0.08 while
ATF_STATIC_GHOST_STOP_LOSS is also 0.08 demands a tail better than the
strategy's own stop -- satisfiable only if fewer than 5% of trades ever stop
out. Two unrelated defaults collided and deadlocked live trading.

Measured 2026-08-28 on the live ghost book: ES95 = 0.08336 against a 0.08
guardrail, block_reason=ghost_validation_block, reason=tail_risk. The entire
tail was three trades, ALL reason=stop_loss, at -8.52% / -8.39% / -8.10%, each
reproducing exactly from its own recorded prices. Real stop-outs, not
artifacts. The gate was blocking because the stop worked.

What must still block is the stop being BREACHED -- SOL-USDC's -22.2% on an 8%
stop, 2.78x, the case 67bc54e and adad8bc addressed from the feed side.
"""

from __future__ import annotations

import unittest
from unittest import mock

from trading.pipeline import ghost_stop_loss_pct, ghost_tail_guardrail


class GhostTailGuardrailTest(unittest.TestCase):
    def test_guardrail_clears_a_tail_made_of_honest_stop_outs(self):
        """The exact production numbers that deadlocked live trading."""
        with mock.patch.dict("os.environ", {
            "GHOST_TAIL_GUARDRAIL": "0.08",
            "ATF_STATIC_GHOST_STOP_LOSS": "0.08",
        }, clear=False):
            guard = ghost_tail_guardrail()
        observed_es95 = 0.083357
        self.assertGreater(
            guard, observed_es95,
            "a tail of pure stop-outs must not block; guard=%.5f es95=%.5f"
            % (guard, observed_es95),
        )
        self.assertAlmostEqual(guard, 0.10, places=6)

    def test_a_stop_that_gaps_through_still_blocks(self):
        """SOL-USDC lost 22.2% on an 8% stop. That must never pass."""
        with mock.patch.dict("os.environ", {
            "GHOST_TAIL_GUARDRAIL": "0.08",
            "ATF_STATIC_GHOST_STOP_LOSS": "0.08",
        }, clear=False):
            guard = ghost_tail_guardrail()
        self.assertLess(
            guard, 0.222,
            "a 2.78x breach of the stop must still trip the tail gate",
        )

    def test_slack_stays_above_the_worst_observed_overshoot(self):
        """Overshoot beyond the stop was 1.013x / 1.049x / 1.065x."""
        with mock.patch.dict("os.environ", {
            "GHOST_TAIL_GUARDRAIL": "0.08",
            "ATF_STATIC_GHOST_STOP_LOSS": "0.08",
        }, clear=False):
            guard = ghost_tail_guardrail()
        self.assertGreater(guard, 0.08 * 1.065)

    def test_explicit_guardrail_wins_when_set_higher(self):
        with mock.patch.dict("os.environ", {
            "GHOST_TAIL_GUARDRAIL": "0.15",
            "ATF_STATIC_GHOST_STOP_LOSS": "0.08",
        }, clear=False):
            self.assertAlmostEqual(ghost_tail_guardrail(), 0.15, places=6)

    def test_zero_still_means_gate_off(self):
        with mock.patch.dict("os.environ", {"GHOST_TAIL_GUARDRAIL": "0"}, clear=False):
            self.assertEqual(ghost_tail_guardrail(), 0.0)

    def test_widening_the_stop_cannot_retire_the_gate(self):
        """A 50% stop must not float the guardrail to 0.625."""
        with mock.patch.dict("os.environ", {
            "GHOST_TAIL_GUARDRAIL": "0.08",
            "ATF_STATIC_GHOST_STOP_LOSS": "0.50",
        }, clear=False):
            self.assertLessEqual(ghost_tail_guardrail(), 0.25)

    def test_stop_is_the_widest_in_the_pooled_book(self):
        """ES95 is pooled, so it is bounded by the loosest contributing stop."""
        with mock.patch.dict("os.environ", {
            "ATF_STATIC_GHOST_STOP_LOSS": "0.08",
            "GHOST_STOP_LOSS_PCT": "0.02",
        }, clear=False):
            self.assertAlmostEqual(ghost_stop_loss_pct(), 0.08, places=6)
        with mock.patch.dict("os.environ", {
            "ATF_STATIC_GHOST_STOP_LOSS": "0.02",
            "GHOST_STOP_LOSS_PCT": "0.12",
        }, clear=False):
            self.assertAlmostEqual(ghost_stop_loss_pct(), 0.12, places=6)


if __name__ == "__main__":
    unittest.main()
