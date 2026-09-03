"""The first live trade must not be refused for 1.11e-16 of a dollar.

Link 6 (RISK) on 2026-09-03: ``recommended_live_usd = $0.00`` with every risk
gate open. atf_static was graduated and live-approved, the ghost book was
+0.5952 over 27 samples, tail risk 0.0336 against a 0.1 guard, loss rate 0.192
against 0.6, profit factor 6.77, wallet $5.4775 deployable, capital deficit
$0.00, no bus actions pending, ``safe_to_live=True``. Nothing was refused on
its merits.

``_build_transition_plan`` sizes the opening trade TWICE. The first pass
rescues a sub-clip recommendation -- a percentage-of-capital ratio can never
reach a fixed clip on a small wallet -- by pinning the dollars to the clip and
back-solving a ratio::

    recommended_live_usd = affordable            = 0.75
    recommended_ratio    = affordable / deployable_stable

Roughly 240 lines later the second pass throws that dollar figure away and
re-derives it from the ratio::

    recommended_live_usd = recommended_ratio * deployable_stable
    if recommended_live_usd < min_clip_usd:   ->  ratio = 0, usd = 0, blocked

The usd -> ratio -> usd round trip is lossy in IEEE-754, and whether it loses
anything depends on the wallet balance. Measured against the two balances this
account has actually held::

    deployable $6.977258   (0.75/d)*d = 0.75000000000000000000   ok
    deployable $5.477500   (0.75/d)*d = 0.74999999999999988898   BLOCKED

So the rescue worked at 06:18 with $6.98 (the balance recorded verbatim in
tests/test_live_clip_matches_the_plan.py, where the plan reads
``recommended_live_usd: 0.75``) and stopped working once the wallet spent down
to $5.48. The strategy, the code and the risk did not change; the balance did.

A second defect made it unreadable. ``live_mode`` and ``live_blocked_reason``
are captured before the second sizing pass and were never refreshed, so
risk_flags reported ``live_mode="ready"`` and ``live_blocked_reason=""`` on a
plan that had ended blocked -- while ``guardrails.live_mode`` in the same dict
said ``"blocked"``. scripts/live_gate_map.py:252 and scripts/live_path_check.py
:163 both read the stale field, so the tools built to diagnose this link
printed "block_reason (none)" next to "$0.0000" for several passes.
"""

from __future__ import annotations

import inspect
import random
import unittest

from trading import pipeline as pipeline_module
from trading.pipeline import below_min_clip

#: The exact state measured on 2026-09-03, from the trace of the locals at
#: risk_flags construction.
DEPLOYABLE = 5.4775
CLIP = 0.75
#: The balance the plan was recorded at on 2026-09-03 06:18, which did work.
DEPLOYABLE_THAT_WORKED = 6.977258


def _second_pass_usd(deployable: float, clip: float) -> float:
    """The dollars pass two re-derives from the ratio pass one back-solved."""
    ratio = clip / max(deployable, 1e-9)
    return ratio * deployable


class RoundTripIsAbsorbedTest(unittest.TestCase):
    def test_the_measured_balance_really_does_lose_a_bit(self):
        """Guard the premise: without tolerance this balance IS under the clip.

        If a future refactor makes the round trip exact, this test should fail
        loudly rather than quietly passing for the wrong reason.
        """
        back = _second_pass_usd(DEPLOYABLE, CLIP)
        self.assertLess(
            back,
            CLIP,
            "premise broken: (0.75/5.4775)*5.4775 no longer lands under 0.75, "
            "so this test is no longer exercising the bug it was written for",
        )
        self.assertAlmostEqual(CLIP - back, 1.11e-16, delta=1e-17)

    def test_the_balance_that_blocked_live_trading_no_longer_blocks(self):
        self.assertFalse(
            below_min_clip(_second_pass_usd(DEPLOYABLE, CLIP), CLIP),
            "a $5.4775 wallet was refused the first live trade for being "
            "1.11e-16 dollars short of a $0.75 floor",
        )

    def test_the_balance_that_worked_still_works(self):
        self.assertFalse(
            below_min_clip(_second_pass_usd(DEPLOYABLE_THAT_WORKED, CLIP), CLIP)
        )

    def test_no_wallet_balance_can_lose_the_rescue(self):
        """The invariant, over the whole range this account can plausibly hold.

        6.2% of balances in this range put the round trip on the losing side,
        so a spot check at one number proves nothing.
        """
        rng = random.Random(1)
        for _ in range(200_000):
            deployable = round(rng.uniform(0.80, 50.0), 6)
            self.assertFalse(
                below_min_clip(_second_pass_usd(deployable, CLIP), CLIP),
                "rescue to the clip lost to float round-off at deployable "
                "$%.6f" % deployable,
            )


class GenuinelySmallTradesAreStillRefusedTest(unittest.TestCase):
    """The tolerance absorbs representation error, not money.

    This is the half that matters for safety: loosening a floor is how this
    repo has previously turned a guard into a rubber stamp.
    """

    def test_a_cent_under_the_clip_is_still_under(self):
        self.assertTrue(below_min_clip(0.74, CLIP))

    def test_a_hundredth_of_a_cent_under_the_clip_is_still_under(self):
        self.assertTrue(below_min_clip(0.7499, CLIP))

    def test_the_derated_size_that_should_block_still_blocks(self):
        """Pass two's bus-action de-rate, the case the floor exists for."""
        derated = (CLIP / DEPLOYABLE) * 0.35 * DEPLOYABLE  # $0.2625
        self.assertTrue(below_min_clip(derated, CLIP))

    def test_zero_is_under_the_clip(self):
        self.assertTrue(below_min_clip(0.0, CLIP))

    def test_tolerance_is_far_below_a_cent(self):
        """Whatever the tolerance is, it must not be able to pass real money."""
        for clip in (0.05, 0.75, 10.0, 150.0):
            largest_absorbed = clip - min(
                v for v in (clip * 10 ** -k for k in range(1, 12))
                if not below_min_clip(clip - v, clip)
            )
            self.assertLess(
                clip - largest_absorbed,
                0.0001,
                "tolerance at clip %.2f absorbs more than a hundredth of a "
                "cent" % clip,
            )

    def test_bad_input_does_not_read_as_a_block(self):
        self.assertFalse(below_min_clip(None, CLIP))
        self.assertFalse(below_min_clip(float("nan"), CLIP))
        self.assertFalse(below_min_clip(0.5, float("nan")))


class BothSizingPassesUseOneRuleTest(unittest.TestCase):
    """Two passes applying the same floor two different ways is the bug class."""

    def test_neither_pass_compares_against_the_clip_directly(self):
        source = inspect.getsource(
            pipeline_module.TrainingPipeline._build_transition_plan
        )
        code = "\n".join(
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn(
            "recommended_live_usd < min_clip_usd",
            code,
            "a bare `<` against the clip is what blocked live trading on a "
            "float round-off; use below_min_clip() in both sizing passes",
        )
        self.assertEqual(
            code.count("below_min_clip("),
            2,
            "both sizing passes must apply the clip through the same helper",
        )


class BlockedPlansMustNotReportReadyTest(unittest.TestCase):
    def test_live_mode_and_reason_are_refreshed_after_the_second_pass(self):
        """risk_flags said ready/'' on a plan guardrails called blocked."""
        source = inspect.getsource(
            pipeline_module.TrainingPipeline._build_transition_plan
        )
        _, _, after_update = source.partition('plan["risk_flags"].update(')
        self.assertTrue(after_update, "risk_flags update block not found")
        for key in ('"live_mode": live_mode', '"live_blocked_reason": block_reason'):
            self.assertIn(
                key,
                after_update,
                "%s is captured before the second sizing pass can block the "
                "plan; it must be refreshed in the update block or every "
                "diagnostic reads a stale 'ready'" % key,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
