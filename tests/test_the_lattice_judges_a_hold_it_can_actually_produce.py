"""The chaos leg must judge a hold this executor can actually produce.

WHY THIS FILE EXISTS. `directive.horizon` is a LABEL, not a forecast. On the
ids that carry 100% of the lattice refusals it is a *lookback* span: the
multi-horizon wrapper overwrites the directive's horizon with its own tier
label (trading/strategies/horizons.py:212), and HORIZON_SPECS says in its own
comments that "5d" means "5.5d of 2h bars" -- history the rule LOOKS BACK
over. The chaos leg then read that as a forward prediction and refused
`donchian_breakout@1w` for "looking 10080.0 min ahead".

Nothing on the directive states the intended hold, so the leg needs an honest
stand-in, and MAX_HOLD_SECONDS is the one this bot already publishes.

⚠ CORRECTED BY ITS OWN AUTHOR, SAME PASS: MAX_HOLD_SECONDS is NOT an eviction.
It filters which positions get EVALUATED; the only forced exit is
MAX_HOLD_FORCE_SECONDS, which defaults to 0 = OFF. Measured over 259 paired
round trips in 30d, realised hold median 56.2 min, p90 220.1, MAX 6970.6 --
47.9% exceed 60 minutes. So the clamp is justified NOT by "the bot cannot hold
longer" but by "60 min is within 7% of the realised median, and 10080 min is
not within anything". The file name says "can actually produce" and that is the
claim these tests DO prove: the clamp never widens the leg.

THE CLAMP IS NOT A LOOSENING. It is an upper bound only. A short label passes
through untouched, so nothing that was refused on an honest horizon starts
passing. Measured pass 680 over 6335 refusals: a 60-minute hold still fails
the leg on 98.0% of them. This makes the printed number true; it does not
open the gate.
"""

from __future__ import annotations

import random
import time

import pytest

from trading.bot import TradingBot


def _persistent_series(n: int = 300, bar_sec: float = 300.0) -> list:
    """Autocorrelated returns, so the chaos leg has something to measure."""
    rng = random.Random(20260923)
    now = time.time()
    out = []
    price, previous = 100.0, 0.0
    for i in range(n):
        step = 0.6 * previous + rng.gauss(0, 0.003)
        price *= 1 + step
        previous = step
        out.append({"ts": now - (n - i) * bar_sec, "price": price})
    return out


class _Directive:
    def __init__(self, horizon: str) -> None:
        self.action = "enter"
        self.expected_return = 0.03
        self.horizon = horizon
        self.size = 1.0
        self.strategy_id = f"donchian_breakout@{horizon}"


def _bot(buffer):
    bot = TradingBot.__new__(TradingBot)
    bot._buffer = buffer
    bot._live_clip_usd = lambda: 0.75
    bot._roundtrip_fee_rate = lambda notional_hint=None: 0.0065
    return bot


def _judged_horizon_sec(monkeypatch, horizon_label: str, max_hold: str):
    """Run the leg and return the horizon it actually handed the lattice."""
    import sys
    from pathlib import Path

    web = str(Path(__file__).resolve().parents[1] / "web")
    if web not in sys.path:
        sys.path.insert(0, web)
    from tradingagent import lattice as lattice_mod

    seen = {}

    def _spy(**kwargs):
        seen["proposed_horizon_sec"] = kwargs.get("proposed_horizon_sec")
        return {"passed": True}

    monkeypatch.setattr(lattice_mod, "evaluate_signal", _spy)
    monkeypatch.setenv("LATTICE_GATE_ENABLED", "1")
    monkeypatch.setenv("MAX_HOLD_SECONDS", max_hold)

    bot = _bot(_persistent_series())
    bot._lattice_refusal("AERO-USDC", _Directive(horizon_label),
                         {"price": 1.0})
    return seen.get("proposed_horizon_sec"), bot


class TestTheHorizonIsClampedToTheHoldCeiling:
    def test_a_week_long_lookback_label_is_judged_at_the_hold_ceiling(
            self, monkeypatch):
        """The whole defect in one assertion.

        `donchian_breakout@1w` carried a 10080-minute horizon into a leg that
        asks "can anyone predict that far ahead". The bot evicts at 3600s, so
        the answer it needed was about 60 minutes, not a week.
        """
        judged, bot = _judged_horizon_sec(monkeypatch, "1w", "3600")
        assert judged == pytest.approx(3600.0), (
            f"the leg judged {judged}s; the executor cannot hold past 3600s")
        assert bot._lattice_last_exit.endswith("_clamped"), (
            "a clamped judgement must be countable -- otherwise the next "
            "census cannot tell a 60-minute refusal from a week-long one")

    @pytest.mark.parametrize("label,expected", [
        ("5d", 3600.0),     # 7200 min of lookback -> the 60-minute ceiling
        ("3d", 3600.0),
        ("12h", 3600.0),
    ])
    def test_every_sweep_tier_above_the_ceiling_is_pulled_down(
            self, monkeypatch, label, expected):
        judged, _ = _judged_horizon_sec(monkeypatch, label, "3600")
        assert judged == pytest.approx(expected)


class TestTheClampNeverWidensTheLeg:
    """The leg must not become easier to pass. This is the safety half."""

    @pytest.mark.parametrize("label,expected", [
        ("10m", 600.0),     # money_button's own default horizon
        ("45m", 2700.0),    # rsi_reversal's class default_horizon
        ("1h", 3600.0),     # ema_cross, exactly at the ceiling
    ])
    def test_a_horizon_inside_the_ceiling_passes_through_untouched(
            self, monkeypatch, label, expected):
        judged, bot = _judged_horizon_sec(monkeypatch, label, "3600")
        assert judged == pytest.approx(expected), (
            "the clamp is an upper bound; it must never lengthen a horizon")
        assert not bot._lattice_last_exit.endswith("_clamped")

    def test_tightening_the_hold_policy_tightens_the_judgement(
            self, monkeypatch):
        """A shorter hold ceiling must give the leg a SHORTER horizon.

        This is the route the census points at: the leg refuses because the
        hold policy is long, not because the parse is wrong. If an operator
        drops MAX_HOLD_SECONDS to 300 the leg must be asked about 5 minutes.
        """
        judged, _ = _judged_horizon_sec(monkeypatch, "1w", "300")
        assert judged == pytest.approx(300.0)

    def test_a_disabled_ceiling_leaves_the_label_alone(self, monkeypatch):
        """MAX_HOLD_SECONDS=0 means no eviction, so there is no honest bound.

        The leg then judges the label as before. This is the fail-open half:
        an operator who turns the hold ceiling off does not silently get a
        different gate.
        """
        judged, _ = _judged_horizon_sec(monkeypatch, "1w", "0")
        assert judged == pytest.approx(604800.0)
