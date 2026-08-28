"""A bot must be BUILT able to trade live when the ghost record was earned.

Observed 2026-08-28: every one of the six live risk gates PASSED with
block_reason empty, recommended_live_usd $0.75 against a $0.75 min clip, and
still zero live rows had ever been written. The reason was upstream of every
gate -- ``GhostTradingSupervisor._readiness_permits_live`` returned False, so
each bot was CONSTRUCTED with ``live_trading_enabled = False`` and the
execution path could not run no matter what any strategy earned.

It returned False because it applied a block-list of ghost reasons that did not
match what ``_ghost_validation`` emits. There are exactly four reasons attached
to a ready=True verdict:

    ""                     strict path -- min_trades, win rate, every guard
    "fast_track"           Wilson lower-bound path
    "positive_expectancy"  asymmetric path, net of fees
    "cold_start_bootstrap" ZERO trades; the bypass that lets collection begin

The block-list named "cold_start", "bootstrap" and "no_metrics". None of those
three is ever emitted alongside ready=True, so the rule was inverted twice
over: it could never reject the one verdict that deserves rejecting
(cold_start_bootstrap, whose name it did not know), and it DID reject "" --
which is not a missing reason but the strongest pass the strict path can
return. Production readiness at the time was exactly ghost_ready=True with
ghost_reason="" over 60 samples at a 0.70 win rate.

These tests pin the allow-list against what _ghost_validation actually emits.
"""

from __future__ import annotations

import pathlib
import types
import unittest
from unittest import mock

from trading.bot import TradingBot
from trading.pipeline import (
    GHOST_EARNED_READY_REASONS,
    TrainingPipeline,
    ghost_reason_is_earned,
)
from trading.selector import GhostTradingSupervisor

PERMITS = GhostTradingSupervisor._readiness_permits_live


def readiness(**over):
    """The degenerate-model readiness this deployment actually reports."""
    base = {
        "ready": False,            # precision 0.0 / recall 0.0 over 18802 samples
        "reason": "insufficient_accuracy",
        "ghost_ready": True,
        "ghost_reason": "",
        "ghost_samples": 60,
        "ghost_win_rate": 0.7,
    }
    base.update(over)
    return base


class ReadinessPermitsLiveTest(unittest.TestCase):
    def test_production_readiness_permits_live(self):
        """The exact report that was building bots unable to trade."""
        self.assertTrue(
            PERMITS(readiness()),
            "ghost_ready=True with the strict-path reason '' is the strongest "
            "evidence the ghost book can produce; it must build a live-capable bot",
        )

    def test_every_earned_ghost_path_permits_live(self):
        for reason in ("", "fast_track", "positive_expectancy"):
            with self.subTest(ghost_reason=reason):
                self.assertTrue(PERMITS(readiness(ghost_reason=reason)))

    def test_cold_start_bypass_does_not_permit_live(self):
        """The one ready=True reason that is evidence of nothing."""
        self.assertFalse(
            PERMITS(readiness(ghost_reason="cold_start_bootstrap")),
            "a zero-trade bootstrap allowance must never enable live trading",
        )

    def test_cold_start_sentinel_matches_what_ghost_validation_emits(self):
        """Pins the allow-list to the source, not to a guessed string.

        The old block-list failed precisely here: it guarded against names
        _ghost_validation never produces. Drive the real function with an empty
        ghost book and reject whatever it actually calls that verdict.
        """
        stub = types.SimpleNamespace(
            metrics=types.SimpleNamespace(
                ghost_trade_snapshot=lambda **_: [],
                aggregate_trade_metrics=lambda _trades: {},
            )
        )
        with mock.patch.dict("os.environ", {"GHOST_COLD_START_BYPASS": "1"}, clear=False):
            verdict = TrainingPipeline._ghost_validation(stub)

        self.assertTrue(verdict["ready"], "cold-start bypass should report ready")
        self.assertEqual(verdict["samples"], 0, "this is the zero-evidence case")
        self.assertFalse(
            ghost_reason_is_earned(verdict["reason"]),
            "the cold-start reason _ghost_validation emits (%r) is not in the "
            "earned allow-list %r -- the two have drifted apart again"
            % (verdict["reason"], sorted(GHOST_EARNED_READY_REASONS)),
        )
        self.assertFalse(PERMITS(readiness(ghost_reason=verdict["reason"])))

    def test_unready_ghost_book_never_permits(self):
        for reason in ("insufficient_samples", "tail_risk", "low_win_rate", ""):
            with self.subTest(ghost_reason=reason):
                self.assertFalse(
                    PERMITS(readiness(ghost_ready=False, ghost_reason=reason))
                )

    def test_model_ready_permits_regardless_of_ghost_reason(self):
        self.assertTrue(
            PERMITS(readiness(ready=True, ghost_ready=False, ghost_reason="tail_risk"))
        )

    def test_strict_coupling_can_be_restored(self):
        with mock.patch.dict(
            "os.environ", {"LIVE_REQUIRE_MODEL_READY": "1"}, clear=False
        ):
            self.assertFalse(PERMITS(readiness()))
            self.assertTrue(PERMITS(readiness(ready=True)))


class LiveReadyFlagSharesTheRuleTest(unittest.TestCase):
    """pipeline._live_ready_flag carried a copy of the same block-list."""

    def _flag(self, ghost_check, readiness_dict=None):
        return TrainingPipeline._live_ready_flag(
            types.SimpleNamespace(),
            readiness_dict or {"ready": False},
            ghost_check,
        )

    def test_earned_ghost_record_is_live_ready(self):
        self.assertTrue(
            self._flag({"ready": True, "reason": "", "total_net_profit": 1.04})
        )

    def test_cold_start_is_not_live_ready(self):
        self.assertFalse(
            self._flag(
                {"ready": True, "reason": "cold_start_bootstrap", "total_net_profit": 0.0}
            )
        )

    def test_profit_requirement_still_applies(self):
        self.assertFalse(
            self._flag({"ready": True, "reason": "", "total_net_profit": 0.0})
        )


class BotGhostEarnedPathSharesTheRuleTest(unittest.TestCase):
    """bot._maybe_transition_to_live carried a THIRD copy of the block-list.

    Its copy was inverted both ways too. It rejected "" -- so even a
    live-capable bot fell through to reason="model_accuracy_gate" and never
    reached ``_refresh_auto_execute``, which is what clears the
    LIVE_TRADES_DRY_RUN="1" default. And it never named
    "cold_start_bootstrap", so a wallet with ZERO ghost trades could promote
    itself to live through this path on no evidence at all.
    """

    def _transition_reason(self, ghost_reason, *, ghost_ready=True):
        """Drive the real method; report the veto it records (None = passed)."""
        pipeline = types.SimpleNamespace(
            ghost_live_transition_plan=lambda: {},
            live_readiness_report=lambda: readiness(
                ghost_ready=ghost_ready, ghost_reason=ghost_reason
            ),
        )
        bot = types.SimpleNamespace(
            auto_promote_live=True,
            apply_transition_plan=lambda _plan: None,
            _transition_plan={},
            _live_transition_state={},
            live_trading_enabled=False,
            global_risk_budget=1.0,
            max_trade_share=0.12,
            decision_threshold=0.5,
            required_live_win_rate=0.55,
            required_live_trades=40,
            required_live_profit=0.0,
            pipeline=pipeline,
        )
        TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        return (bot._live_transition_state or {}).get("reason")

    def test_earned_ghost_record_clears_the_model_accuracy_gate(self):
        """The production verdict: ghost_ready=True, strict-path reason ''."""
        self.assertNotEqual(
            self._transition_reason(""),
            "model_accuracy_gate",
            "the strict-path ghost pass must not be vetoed by the degenerate "
            "model metric -- this veto is what left LIVE_TRADES_DRY_RUN at '1'",
        )

    def test_every_earned_ghost_path_clears_it(self):
        for reason in sorted(GHOST_EARNED_READY_REASONS):
            with self.subTest(ghost_reason=reason):
                self.assertNotEqual(
                    self._transition_reason(reason), "model_accuracy_gate"
                )

    def test_cold_start_bypass_is_vetoed_here(self):
        """The hole the block-list left open: promotion on zero ghost trades."""
        self.assertEqual(
            self._transition_reason("cold_start_bootstrap"),
            "model_accuracy_gate",
            "a zero-trade bootstrap allowance must never promote a bot to live",
        )

    def test_unready_ghost_book_is_vetoed_here(self):
        self.assertEqual(
            self._transition_reason("", ghost_ready=False), "model_accuracy_gate"
        )


class NoModuleRestatesTheRuleTest(unittest.TestCase):
    """The bug was one rule copied into three modules, then drifting apart.

    Two of the three copies had already been corrected once while the third
    kept the inverted set, so the executor stayed broken. Pin the shape: every
    module that gates live money on a ghost reason consults the shared
    allow-list, and none of them spells the reason strings out in code again.
    """

    MODULES = ("trading/pipeline.py", "trading/selector.py", "trading/bot.py")

    def _code_lines(self, rel):
        path = pathlib.Path(__file__).resolve().parents[1] / rel
        return [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        ]

    def test_each_module_consults_the_shared_allow_list(self):
        for rel in self.MODULES:
            with self.subTest(module=rel):
                self.assertTrue(
                    any("ghost_reason_is_earned" in l for l in self._code_lines(rel)),
                    "%s gates live money on a ghost reason but does not use the "
                    "shared allow-list" % rel,
                )

    # Every reason _ghost_validation can attach to a ready=True verdict. The
    # bug was a hand-written membership test over these; "bootstrap" alone is
    # excluded because pipeline uses it legitimately as a live_mode value.
    REASON_LITERALS = (
        '"cold_start"',
        '"cold_start_bootstrap"',
        '"no_metrics"',
        '"fast_track"',
        '"positive_expectancy"',
    )

    def test_no_module_restates_the_reason_strings(self):
        for rel in self.MODULES:
            for line in self._code_lines(rel):
                if " in {" not in line and " in (" not in line:
                    continue
                hit = next((s for s in self.REASON_LITERALS if s in line), None)
                if hit is not None:
                    self.fail(
                        "%s membership-tests a ghost reason in code (%s) -- "
                        "that hand-written set is exactly what drifted out of "
                        "sync with _ghost_validation; use ghost_reason_is_earned"
                        % (rel, line.strip())
                    )


if __name__ == "__main__":
    unittest.main()
