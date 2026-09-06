"""The brain graduation path must not promote a brain that is worse than a constant.

MEASURED 2026-09-05, the first brain training run that ever finished (598
pairs, 200 held-out queries on WETH-USDC hourly):

    exact-bucket accuracy   0.515   (103/200)
    majority-class baseline 0.605   (always answer "flat")
    mean confidence         0.278
    conf when RIGHT         0.2918
    conf when WRONG         0.2634    <- separation 0.0284

The brain answered "flat" for 73.5% of queries and "loss" for 1.5%. Against
44 actual losses it caught ONE. It learned the marginal distribution, not the
conditional.

THE TRAP THIS TEST EXISTS TO CLOSE. `_maybe_promote_to_live` grants live
trading on the brain path when `_brain_conf_ema >= BRAIN_GRADUATION_MIN_CONF_EMA`
(0.20) alongside a trade count and win rate. The brain's mean confidence is
now **0.278, which CLEARS that bar** -- and it is worse than a constant. The
only thing standing between this brain and a live promotion is
`BRAIN_CONFIDENCE_FLOOR` (0.5), which rejects every one of those readings as
an abstention so the EMA never moves.

That is a single environment variable holding the line, and the obvious
"improvement" -- lowering the floor so the brain can start contributing --
would promote a brain that loses to always saying flat. These tests pin the
relationship so that change cannot be made silently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


#: The measured held-out result. If a later training run changes these, update
#: them from the report and re-read the conclusions -- do not delete the test.
MEASURED_EXACT_ACCURACY = 0.515
MEASURED_BASELINE = 0.605          # always answer "flat" (121 of 200)
MEASURED_MEAN_CONFIDENCE = 0.278
MEASURED_CONF_RIGHT = 0.2918
MEASURED_CONF_WRONG = 0.2634


def test_the_measured_brain_loses_to_the_majority_class():
    """Guards the reading of the result, not the code.

    0.515 looks like "better than a coin flip" and is not: the label
    distribution is 60.5% flat, so a constant answer scores higher. An
    accuracy figure means nothing without its baseline, and this repo has
    now nearly banked the same mistake twice (86% in an earlier probe was
    also majority-class).
    """
    assert MEASURED_EXACT_ACCURACY < MEASURED_BASELINE, (
        "if a training run ever beats the majority-class baseline, that is "
        "the moment the brain became worth trading -- update this test "
        "deliberately, with the new report in hand")


def test_confidence_does_not_separate_right_from_wrong():
    """A confidence gate needs the two distributions to be distinguishable.

    trading/bot.py still documents a 2026-06-20 probe claiming 0.935 when
    right against 0.124 when wrong -- a separation of 0.811. The measured
    separation is 0.0284, 3.5% of that. Any code reasoning about "the brain
    knows when it knows" is reasoning from a stale number.
    """
    separation = MEASURED_CONF_RIGHT - MEASURED_CONF_WRONG
    assert separation < 0.05, (
        f"separation {separation:.4f}: no threshold can sort this brain's "
        f"good calls from its bad ones")


def test_the_confidence_floor_is_the_only_thing_blocking_promotion():
    """THE LOAD-BEARING ASSERTION.

    Mean confidence (0.278) already clears the graduation bar (0.20). The
    floor (0.5) is what turns every reading into an abstention so the EMA
    never accumulates. Remove or lower the floor below the brain's mean and a
    worse-than-constant brain graduates to live trading.
    """
    floor = float(os.getenv("BRAIN_CONFIDENCE_FLOOR", "0.5"))
    grad_bar = float(os.getenv("BRAIN_GRADUATION_MIN_CONF_EMA", "0.20"))

    assert MEASURED_MEAN_CONFIDENCE > grad_bar, (
        "the brain's confidence already exceeds the graduation bar; the "
        "floor is what is holding it back")
    assert floor > MEASURED_MEAN_CONFIDENCE, (
        f"BRAIN_CONFIDENCE_FLOOR ({floor}) must stay above the brain's mean "
        f"confidence ({MEASURED_MEAN_CONFIDENCE}) until the brain beats the "
        f"majority-class baseline. Below it, readings stop being abstentions, "
        f"_brain_conf_ema climbs past {grad_bar}, and a brain that scores "
        f"{MEASURED_EXACT_ACCURACY} against a {MEASURED_BASELINE} baseline is "
        f"promoted to spend real money.")


def test_abstentions_do_not_move_the_graduation_ema():
    """Below-floor readings must not update _brain_conf_ema.

    This is the mechanism the test above depends on. If a refactor ever
    updates the EMA before the floor check, the floor stops protecting
    anything: the brain's 0.278 readings would accumulate straight through
    the 0.20 graduation bar.
    """
    import trading.bot as bot_module
    from trading.bot import TradingBot

    class Bridge:
        def predict_outcome(self, features: str):
            return "outcome flat", MEASURED_MEAN_CONFIDENCE

    bot = TradingBot.__new__(TradingBot)
    bot._brain_conf_ema = 0.0

    original_bridge = bot_module._brain_bridge
    original_feats = bot_module._brain_features_text
    bot_module._brain_bridge = lambda: Bridge()
    bot_module._brain_features_text = lambda **kwargs: "features"
    previous = os.environ.get("BRAIN_CONFIDENCE_FLOOR")
    os.environ["BRAIN_CONFIDENCE_FLOOR"] = "0.5"
    try:
        decision: dict = {}
        result = bot._brain_record_entry(
            decision, side="buy", symbol="WETH-USDC", chain_name="base",
            price=100.0, momentum=0.0, confidence=0.5,
        )
        assert result == 0.0, "a below-floor reading must contribute nothing"
        assert bot._brain_conf_ema == 0.0, (
            "an abstention moved the graduation EMA: the confidence floor no "
            "longer protects the brain graduation path")
        assert decision["brain"]["bridge_confidence_rejected"] is True
    finally:
        bot_module._brain_bridge = original_bridge
        bot_module._brain_features_text = original_feats
        if previous is None:
            os.environ.pop("BRAIN_CONFIDENCE_FLOOR", None)
        else:
            os.environ["BRAIN_CONFIDENCE_FLOOR"] = previous
