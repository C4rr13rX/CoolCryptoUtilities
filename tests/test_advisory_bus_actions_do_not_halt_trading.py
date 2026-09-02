"""A pending bus action is advice; only a fault or an explicit freeze halts live.

Measured 2026-09-02 10:09:16, the single scheduler halt in production reported

    [feedback] [WARN] scheduler:halted -> {'reason': 'wallet_sparse',
                                           'ghost_multiplier': 0.35,
                                           'bus_actions_pending': True}

while every risk gate passed on a fresh evaluation (live_safe=True,
live_mode="ready", recommended_live_usd=0.75, $6.98 deployable). The 0.35 is the
clamp `_build_transition_plan` applies whenever `bus_actions` is non-empty, which
identifies pending bus actions as the trigger.

`_build_transition_plan` already encodes the intended rule: pending actions zero
the live ratio only when paired with a real fault (wallet_sparse /
capital_deficit / native_starved) or with an explicit "freeze_live"/"pause_live"
action. Most bus actions are suggestions -- "preload_gas_buffer",
"scan_micro_opportunities", "refresh_wallet_balances", "notify_add_funds" -- and
a bare `or bus_actions_pending` in the halt_live disjunction made every one of
them a full stop.

That mattered beyond live: trading/bot.py maps halt_live to `risk_budget = 0.0`,
and the risk budget gates the whole scheduler, so a gas-buffer suggestion also
halted ghost evaluation -- starving the evidence graduation waits on. It is also
self-defeating, because publishing ATF candidates is itself what puts actions on
the bus.

These tests pin the rule at the level that decides it, so the clause cannot come
back without a failure that says why it is wrong.
"""

from __future__ import annotations

import inspect
import re
import unittest

from trading import pipeline as pipeline_module


def _halt_live_source() -> str:
    """The text of the halt_live/halt_reason disjunction as it ships."""
    source = inspect.getsource(pipeline_module.TrainingPipeline._build_transition_plan)
    match = re.search(r'"halt_live":(.*?)"bus_actions_pending":', source, re.S)
    assert match, "halt_live disjunction not found in _build_transition_plan"
    return match.group(1)


class AdvisoryBusActionsDoNotHaltTrading(unittest.TestCase):
    def test_halt_live_does_not_trip_on_bus_actions_alone(self) -> None:
        """No bare `or bus_actions_pending` in the halt_live disjunction."""
        clause = _halt_live_source()
        # Strip comments: the prose explains the removed clause by name, and
        # only executable code decides whether trading halts.
        code = "\n".join(
            line for line in clause.splitlines() if not line.strip().startswith("#")
        )
        self.assertNotRegex(
            code,
            r"\bor\s+bus_actions_pending\b",
            "halt_live trips on any pending bus action again. Advisory actions "
            "(preload_gas_buffer, scan_micro_opportunities) would halt the whole "
            "scheduler, ghost included.",
        )

    def test_genuine_bus_blocks_still_halt_live(self) -> None:
        """Removing the clause must not lose a real block.

        When pending actions SHOULD stop live, the rule above sets
        recommended_ratio to 0, and that condition stays in the disjunction.
        """
        code = "\n".join(
            line for line in _halt_live_source().splitlines()
            if not line.strip().startswith("#")
        )
        self.assertRegex(
            code,
            r"recommended_ratio\s*<=\s*0",
            "recommended_ratio <= 0 is what carries a genuine bus-driven block "
            "into halt_live; without it the removal above loses real refusals.",
        )
        for fault in ("wallet_sparse", "native_starved", "loss_rate_block"):
            self.assertIn(
                fault,
                code,
                f"halt_live no longer trips on {fault}; that is a real fault.",
            )

    def test_freeze_and_pause_actions_are_still_enumerated(self) -> None:
        """The conjunction that names the blocking actions must survive."""
        source = inspect.getsource(pipeline_module.TrainingPipeline._build_transition_plan)
        self.assertIn('"freeze_live" in bus_freeze_actions', source)
        self.assertIn('"pause_live" in bus_freeze_actions', source)
        self.assertRegex(
            source,
            r'block_reason\s*=\s*"bus_actions_pending"',
            "the genuine bus block must still name itself in block_reason, "
            "which is what halt_reason now reports it through.",
        )

    def test_advisory_actions_are_not_freeze_actions(self) -> None:
        """The advisory actions this fix protects are not freeze/pause."""
        source = inspect.getsource(pipeline_module.TrainingPipeline._build_transition_plan)
        for advisory in (
            "swap_stable_to_native",
            "scan_micro_opportunities",
            "refresh_wallet_balances",
            "notify_add_funds",
        ):
            self.assertIn(
                advisory,
                source,
                f"{advisory} is one of the suggestions that must not halt trading",
            )
            # None of them are raised as freeze_live/pause_live, so under the
            # enumerated rule they leave the live ratio alone.
            self.assertNotRegex(
                source,
                rf'add_bus_action\(\s*"(?:freeze|pause)_live",\s*"{advisory}"',
                f"{advisory} must not be published as a freeze/pause action",
            )


if __name__ == "__main__":
    unittest.main()
