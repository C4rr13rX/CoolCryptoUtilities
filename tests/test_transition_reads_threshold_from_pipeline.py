"""The live transition must not die on an attribute the bot never had.

Observed 2026-09-03. ``atf_static`` graduated at 01:03:24 -- the first strategy
ever to clear the 20-trade ghost gate. One second later, at 01:03:25, the log
began repeating every ~3 seconds:

    live-transition: transition raised:
        AttributeError: 'TradingBot' object has no attribute 'decision_threshold'

``_maybe_transition_to_live`` is the ONLY code path that turns live trading on.
It is wrapped in a bare ``except Exception`` in the trading cycle (bot.py), so
the crash was logged and swallowed: the bot went on ghost-trading forever while
every gate in ``scripts/live_path_check.py`` reported PASS. Links 1-8 green,
link 9 "no live trades yet", and nothing in between to explain it.

The line was::

    threshold = float(readiness.get("threshold", self.decision_threshold))

Two things made this survive as long as it did.

1. ``decision_threshold`` lives on the PIPELINE, not the bot. Every other
   reference in bot.py already says ``self.pipeline.decision_threshold``. This
   one did not, and ``TradingBot`` has no such attribute -- measured, zero
   assignments anywhere in the class.

2. Python evaluates a ``.get()`` default EAGERLY. The fallback is not "used
   when the key is missing"; it is evaluated on every single call. Production
   readiness always carries ``threshold`` (measured: 0.5), so the value was
   discarded immediately -- the expression that killed the transition computed
   something nothing ever read.

It could not fire before graduation because every earlier return in that
function -- ``live_trading_enabled``, then ``not ready_flag`` -- sits ABOVE the
line. The statement was unreachable until a strategy graduated and flipped
``ready_flag`` True. The bug was planted for however long and detonated at the
exact moment the system first became able to trade.

And it was invisible to the suite for the same reason it was invisible in
review. ``tests/test_readiness_permits_live.py`` drives this method with a
``SimpleNamespace`` bot carrying ``decision_threshold=0.5`` -- an attribute the
real class does not define. The fake was more capable than the real object, so
the test exercised the broken line and passed.

So these tests pin BOTH halves:
  * the real ``TradingBot`` does not define ``decision_threshold`` (if it ever
    gains one, the stand-in below stops being a faithful stand-in);
  * the transition survives a bot shaped like the real one, with the attribute
    absent, whether or not readiness carries a threshold.
"""

from __future__ import annotations

import types
import unittest

from trading.bot import TradingBot


def _readiness(**over):
    """The ready=True verdict that first reached the crashing line."""
    base = {
        "ready": True,
        "reason": "",
        "ghost_ready": True,
        "ghost_reason": "",
        "ghost_samples": 60,
        "ghost_win_rate": 0.7,
        "threshold": 0.5,          # measured in production 2026-09-03
        "precision": 0.535,
        "recall": 0.680,
        "samples": 713,
    }
    base.update(over)
    return base


def _bot(readiness_payload, *, min_ghost_win_rate=0.55):
    """A stand-in shaped like the REAL TradingBot.

    Deliberately does NOT define ``decision_threshold``. That omission is the
    whole point: the pre-existing fake defined it, which is precisely why the
    suite could not see this crash.

    ``min_ghost_win_rate`` defaults to the value the production pipeline
    actually reports (measured 0.55 on 2026-09-03), because it is what sets the
    effective precision bar: ``max(min_ghost_win_rate, required_win_rate *
    fast_track_factor)``. A stand-in carrying the library default of 0.5 puts
    the bar UNDER production's measured precision of 0.5355, so the gate under
    test never fires and the test passes without exercising anything.
    """
    pipeline = types.SimpleNamespace(
        decision_threshold=0.89,           # measured in production 2026-09-03
        ghost_live_transition_plan=lambda: {},
        live_readiness_report=lambda: readiness_payload,
        min_ghost_win_rate=min_ghost_win_rate,
        focus_lookback_sec=3600,
        metrics=types.SimpleNamespace(
            ghost_trade_snapshot=lambda **_kw: [],
            aggregate_trade_metrics=lambda _t: {},
        ),
    )
    return types.SimpleNamespace(
        auto_promote_live=True,
        apply_transition_plan=lambda _plan: None,
        _transition_plan={},
        _live_transition_state={},
        _replay_gate_allows=lambda: (True, ""),
        live_trading_enabled=False,
        global_risk_budget=1.0,
        max_trade_share=0.12,
        required_live_win_rate=0.55,
        required_live_trades=40,
        required_live_profit=0.0,
        pipeline=pipeline,
    )


class TransitionSurvivesWithoutBotThresholdTest(unittest.TestCase):

    def test_real_bot_class_does_not_define_decision_threshold(self):
        """Pins the premise the stand-in above relies on.

        If TradingBot ever grows a real ``decision_threshold``, this fails and
        tells the next reader to revisit the fake -- rather than letting the
        fake quietly drift back into being more capable than the real thing.
        """
        self.assertFalse(
            hasattr(TradingBot, "decision_threshold"),
            "TradingBot defines decision_threshold now; the pipeline was its "
            "only home. Re-check trading/bot.py's transition path.",
        )

    def test_transition_does_not_raise_when_readiness_carries_threshold(self):
        """The production case: ready=True and ``threshold`` present.

        Before the fix this raised AttributeError even though the default was
        never needed -- ``.get()`` evaluates it eagerly.
        """
        bot = _bot(_readiness())
        try:
            TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        except AttributeError as exc:
            if "decision_threshold" in str(exc):
                self.fail(
                    "the live transition still dies on the bot attribute that "
                    "does not exist: %s" % exc
                )
            raise

    def test_transition_does_not_raise_when_threshold_is_absent(self):
        """And the case the default was actually written for."""
        payload = _readiness()
        payload.pop("threshold")
        bot = _bot(payload)
        try:
            TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        except AttributeError as exc:
            if "decision_threshold" in str(exc):
                self.fail(
                    "the fallback path reads the bot instead of the pipeline: "
                    "%s" % exc
                )
            raise

    def test_transition_records_a_reason_rather_than_crashing(self):
        """A veto is a decision; a crash is not.

        The distinction is what cost the time here: a swallowed AttributeError
        is indistinguishable from "the gates said no" unless the transition
        leaves a reason behind.
        """
        bot = _bot(_readiness())
        TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        self.assertIsInstance(
            bot._live_transition_state, dict,
            "the transition must leave its verdict behind, not blow up",
        )


class TransitionRecordsItsVetoTest(unittest.TestCase):
    """A refusal must say why. These two vetoes returned mute.

    Measured 2026-09-03, right after the AttributeError above was fixed: the
    transition stopped crashing and began refusing at the precision gate
    (0.5355 against 0.5500). It recorded nothing -- so `_live_transition_state`
    still held the raw readiness report assigned at bot.py:1965, whose own
    `reason` field reads "mini_ready".

    The telemetry therefore said "mini_ready" while the real verdict was a
    rejection. A veto that reports the reason the bot was ADMITTED is worse
    than one that reports nothing, because it looks like progress.
    """

    def test_precision_shortfall_is_recorded(self):
        payload = _readiness(precision=0.5355, recall=0.6803, samples=713,
                             reason="mini_ready")
        bot = _bot(payload)
        TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        state = bot._live_transition_state or {}
        self.assertEqual(
            state.get("reason"), "model_precision_gate",
            "the precision veto must name itself, not leave readiness's own "
            "'mini_ready' standing as the apparent verdict",
        )
        self.assertFalse(state.get("enabled"))
        self.assertLess(state.get("shortfall"), 0.0)

    def test_recorded_veto_does_not_claim_readiness(self):
        """The specific trap: reason must not survive as 'mini_ready'."""
        payload = _readiness(precision=0.1, recall=0.1, reason="mini_ready")
        bot = _bot(payload)
        TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        self.assertNotEqual(
            (bot._live_transition_state or {}).get("reason"), "mini_ready",
            "a rejected transition still reported the readiness verdict that "
            "let it in -- this is what made the block invisible",
        )

    def test_passing_precision_is_not_vetoed_here(self):
        """The gate must still only fire on a genuine shortfall."""
        payload = _readiness(precision=0.95, recall=0.95, samples=713)
        bot = _bot(payload)
        TradingBot._maybe_transition_to_live(bot, latest_decision=None)
        self.assertNotIn(
            (bot._live_transition_state or {}).get("reason"),
            ("model_precision_gate", "model_sample_gate"),
            "a model clearing both bars must not be stopped by these vetoes",
        )


if __name__ == "__main__":       # pragma: no cover
    unittest.main()
