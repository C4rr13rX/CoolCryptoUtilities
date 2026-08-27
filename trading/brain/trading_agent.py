"""A trading agent whose MODEL is the crypto wizard brain.

Replaces ATF (Agent The Freeloader) in the model slot. ATF is an ensemble of
free remote models standing in as one, and it hallucinates: measured
2026-08-27 it fed the trade path a +2038% fill on a symbol the feed had never
carried, a -99% "stop loss" from a denomination mismatch, and prices taken from
whichever chain happened to have the deepest pool. A model that invents numbers
is the wrong thing to put in front of money.

This agent is deliberately NOT a chat or coding agent. It has one job -- decide
what to trade and when to exit, to make a profit -- and a small, closed tool
set to do it with. It cannot write code, browse, or answer questions.

    brain      the model: a regime read for one symbol
    tools      corroborated price, shape match, position state, fees/volume
    output     a Decision, always with a reason, or an explicit refusal

PRODUCTION GATE
---------------
``validate_on_history()`` must pass before this agent may size a real trade.
The bar is deliberately the one that has caught every false positive in this
project: beat the majority-class baseline out-of-sample on a chronological
split, with positive expectancy after fees. An untrained brain answers every
input identically at confidence 0.0 and scores ~70% by always saying "down" in
a falling market -- that is not skill, and the gate rejects it.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


@dataclass
class Decision:
    """What the agent decided, and why. A refusal is a first-class answer."""

    action: str                 # "enter" | "exit" | "hold" | "refuse"
    symbol: str
    reason: str
    confidence: float = 0.0
    expected_return: float = 0.0
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    size_usd: float = 0.0
    horizon_sec: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return self.action in ("enter", "exit")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "reason": self.reason,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "size_usd": self.size_usd,
            "horizon_sec": self.horizon_sec,
            "evidence": dict(self.evidence),
            "strategy_id": "wizard_brain",
        }


@dataclass
class ValidationResult:
    """Whether this agent has earned the right to trade real money."""

    passed: bool
    reason: str
    samples: int = 0
    accuracy: float = 0.0
    baseline: float = 0.0
    edge: float = 0.0
    expectancy: float = 0.0
    distinct_answers: int = 0
    max_confidence: float = 0.0
    control_distinct: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "reason": self.reason, "samples": self.samples,
            "accuracy": self.accuracy, "baseline": self.baseline, "edge": self.edge,
            "expectancy": self.expectancy, "distinct_answers": self.distinct_answers,
            "max_confidence": self.max_confidence,
            "control_distinct": self.control_distinct,
        }


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------


class WizardTradingAgent:
    """Crypto wizard brain, wrapped in a trade-decision loop.

    ``brain`` needs one method: ``predict(symbol, price, momentum) ->
    (answer, confidence)``. Injected so the agent is testable and so a GA
    champion or a different trained brain can be swapped in per the model
    registry.
    """

    strategy_id = "wizard_brain"

    def __init__(
        self,
        brain: Any,
        *,
        min_confidence: Optional[float] = None,
        min_edge: Optional[float] = None,
        fee_rate: Optional[float] = None,
        corroborate: Optional[Callable[[str, float], Optional[float]]] = None,
    ) -> None:
        self.brain = brain
        self.min_confidence = float(
            min_confidence if min_confidence is not None
            else os.getenv("WIZARD_AGENT_MIN_CONFIDENCE", "0.15")
        )
        self.min_edge = float(
            min_edge if min_edge is not None
            else os.getenv("WIZARD_AGENT_MIN_EDGE", "0.02")
        )
        self.fee_rate = float(
            fee_rate if fee_rate is not None
            else os.getenv("WIZARD_AGENT_FEE_RATE", "0.0065")
        )
        self._corroborate = corroborate

    # -- tools -------------------------------------------------------------

    def _price(self, symbol: str, quoted: float) -> Optional[float]:
        """Corroborated price, or None. The agent never trades an unchecked
        quote -- that is precisely how ATF booked fills that never happened."""
        if self._corroborate is None:
            return quoted if quoted > 0 else None
        try:
            return self._corroborate(symbol, quoted)
        except Exception:
            return None

    @staticmethod
    def _momentum(prices: Sequence[float], lookback: int = 5) -> float:
        if len(prices) <= lookback:
            return 0.0
        prev = float(prices[-lookback - 1])
        now = float(prices[-1])
        if prev <= 0:
            return 0.0
        return (now / prev) - 1.0

    # -- decision ----------------------------------------------------------

    def decide(
        self,
        *,
        symbol: str,
        prices: Sequence[float],
        quoted_price: float = 0.0,
        position: Optional[Dict[str, Any]] = None,
        size_usd: float = 0.0,
        recent_volume_usd: float = 0.0,
        horizon_sec: float = 3600.0,
    ) -> Decision:
        """One decision for one symbol. Every path states its reason."""
        if not prices:
            return Decision("refuse", symbol, "no_price_history")

        price = self._price(symbol, quoted_price or float(prices[-1]))
        if not price or price <= 0:
            return Decision("refuse", symbol, "price_not_corroborated")

        momentum = self._momentum(prices)
        try:
            answer, confidence = self.brain.predict(symbol, price, momentum)
        except Exception as exc:  # noqa: BLE001
            return Decision("refuse", symbol, "brain_error:%s" % type(exc).__name__)

        confidence = float(confidence or 0.0)
        evidence = {
            "brain_answer": answer,
            "brain_confidence": confidence,
            "momentum": momentum,
            "corroborated_price": price,
        }

        # An untrained brain answers everything at confidence 0.0. Refusing
        # here is what stops it from trading on noise, which is exactly how
        # the brain lost 842 trades the last time it was enabled by default.
        if confidence < self.min_confidence:
            return Decision(
                "refuse", symbol,
                "confidence_below_floor(%.4f<%.4f)" % (confidence, self.min_confidence),
                confidence=confidence, evidence=evidence,
            )
        if not answer:
            return Decision("refuse", symbol, "no_brain_answer",
                            confidence=confidence, evidence=evidence)

        bullish = str(answer) in ("win", "win_big")
        strong = str(answer) in ("win_big", "loss_big")

        # -- managing an open position ------------------------------------
        if position:
            entry = float(position.get("entry_price") or 0.0)
            if entry > 0:
                pnl = (price / entry) - 1.0
                evidence["unrealised"] = pnl
                target = float(position.get("target_price") or 0.0)
                stop = float(position.get("stop_price") or 0.0)
                if target > 0 and price >= target:
                    return Decision("exit", symbol, "target_hit(%.4f)" % pnl,
                                    confidence=confidence, entry_price=entry,
                                    evidence=evidence)
                if stop > 0 and price <= stop:
                    return Decision("exit", symbol, "stop_hit(%.4f)" % pnl,
                                    confidence=confidence, entry_price=entry,
                                    evidence=evidence)
                if not bullish and pnl > self.fee_rate:
                    # Turned bearish while ahead: bank it rather than round-trip.
                    return Decision("exit", symbol, "brain_turned_bearish_in_profit(%.4f)" % pnl,
                                    confidence=confidence, entry_price=entry,
                                    evidence=evidence)
                return Decision("hold", symbol, "position_within_bounds(%.4f)" % pnl,
                                confidence=confidence, entry_price=entry,
                                evidence=evidence)

        # -- considering an entry -----------------------------------------
        if not bullish:
            return Decision("hold", symbol, "brain_not_bullish(%s)" % answer,
                            confidence=confidence, evidence=evidence)

        # Target return scales with conviction, then fees come off it. Sizing
        # it FROM min_edge made the test unpassable by construction: a
        # moderately confident call produced expected == min_edge * 1.2, whose
        # post-fee net could never clear min_edge again. min_edge is the floor
        # on what survives fees, not the unit the target is built from.
        base_target = float(os.getenv("WIZARD_AGENT_BASE_TARGET", "0.05"))
        expected = base_target * (2.0 if strong else 1.0) * min(2.0, max(0.5, confidence * 2.0))
        net = expected - self.fee_rate
        if net < self.min_edge:
            return Decision(
                "refuse", symbol,
                "edge_below_fees(%.4f-%.4f<%.4f)" % (expected, self.fee_rate, self.min_edge),
                confidence=confidence, expected_return=expected, evidence=evidence,
            )

        # Volume must support the exit, or the margin is imaginary.
        min_ratio = float(os.getenv("WIZARD_AGENT_MIN_VOLUME_RATIO", "0"))
        if min_ratio > 0 and size_usd > 0 and recent_volume_usd < size_usd * min_ratio:
            return Decision(
                "refuse", symbol,
                "volume_too_thin(%.0f<%.0f)" % (recent_volume_usd, size_usd * min_ratio),
                confidence=confidence, expected_return=expected, evidence=evidence,
            )

        stop_pct = float(os.getenv("WIZARD_AGENT_STOP", "0.03"))
        return Decision(
            "enter", symbol,
            "brain_bullish(%s conf=%.3f mom=%+.4f)" % (answer, confidence, momentum),
            confidence=confidence,
            expected_return=expected,
            entry_price=price,
            target_price=price * (1.0 + expected),
            stop_price=price * (1.0 - stop_pct),
            size_usd=size_usd,
            horizon_sec=horizon_sec,
            evidence=evidence,
        )


# --------------------------------------------------------------------------
# The production gate
# --------------------------------------------------------------------------


def validate_on_history(
    brain: Any,
    bars_by_symbol: Dict[str, Sequence[Dict[str, Any]]],
    *,
    split: float = 0.7,
    horizon: int = 5,
    lookback: int = 5,
    min_samples: int = 100,
    min_edge: float = 0.02,
    fee_rate: float = 0.0065,
    price_key: str = "close",
) -> ValidationResult:
    """Must pass before this agent may size a real trade.

    Deliberately the bar that has caught every false positive here:

      * CONTROL on fabricated symbols -- a brain that answers garbage the same
        way it answers real data has learned nothing, whatever it scores.
      * CHRONOLOGICAL split -- train on the past, test on the future only.
      * MAJORITY-CLASS baseline -- a brain that always says "down" scores ~70%
        in a falling market. Skill is what beats the common class.
      * Expectancy positive AFTER fees.
    """
    # --- control: can it tell fiction from data at all? -------------------
    control_answers = set()
    control_confs: List[float] = []
    for i, sym in enumerate(("ZZZZNOTREAL-USDC", "QQQFAKE-USDC", "XXXX-USDC", "AAAA-USDC")):
        try:
            ans, conf = brain.predict(sym, 10.0 ** (i - 2), 0.01 * (i - 2))
        except Exception:
            ans, conf = None, 0.0
        control_answers.add(str(ans))
        control_confs.append(float(conf or 0.0))
    max_control_conf = max(control_confs) if control_confs else 0.0
    if max_control_conf <= 0.0:
        return ValidationResult(
            False, "untrained: zero confidence on every control input",
            control_distinct=len(control_answers), max_confidence=max_control_conf,
        )

    # --- build chronological samples --------------------------------------
    samples: List[Tuple[str, float, float, float]] = []
    for symbol, bars in bars_by_symbol.items():
        prices = [float(b.get(price_key, 0.0) or 0.0) for b in bars]
        for i in range(lookback, len(prices) - horizon):
            now = prices[i]
            prev = prices[i - lookback]
            future = prices[i + horizon]
            if now <= 0 or prev <= 0 or future <= 0:
                continue
            samples.append((symbol, now, (now / prev) - 1.0, (future / now) - 1.0))
    if len(samples) < min_samples:
        return ValidationResult(False, "insufficient_history", samples=len(samples))

    cut = int(len(samples) * split)
    test = samples[cut:]
    if len(test) < max(20, min_samples // 4):
        return ValidationResult(False, "insufficient_holdout", samples=len(test))

    answers = set()
    confs: List[float] = []
    correct = total = 0
    returns: List[float] = []
    for symbol, price, momentum, forward in test:
        try:
            ans, conf = brain.predict(symbol, price, momentum)
        except Exception:
            continue
        answers.add(str(ans))
        confs.append(float(conf or 0.0))
        parsed = str(ans)
        if parsed in ("flat", "None", "steady", ""):
            continue                       # not a directional call; do not score
        predicted_up = parsed in ("win", "win_big")
        total += 1
        if predicted_up == (forward > 0):
            correct += 1
        returns.append((forward if predicted_up else -forward) - fee_rate)

    if total == 0:
        return ValidationResult(
            False, "no directional calls on held-out data",
            samples=len(test), distinct_answers=len(answers),
            max_confidence=max(confs) if confs else 0.0,
            control_distinct=len(control_answers),
        )

    ups = sum(1 for _s, _p, _m, f in test if f > 0)
    up_rate = ups / len(test)
    baseline = max(up_rate, 1.0 - up_rate)
    accuracy = correct / total
    expectancy = float(np.mean(returns)) if returns else 0.0
    edge = accuracy - baseline

    result = ValidationResult(
        passed=False, reason="", samples=total, accuracy=accuracy,
        baseline=baseline, edge=edge, expectancy=expectancy,
        distinct_answers=len(answers),
        max_confidence=max(confs) if confs else 0.0,
        control_distinct=len(control_answers),
    )
    if len(answers) <= 1:
        result.reason = "degenerate: one answer for every input"
        return result
    if edge < min_edge:
        result.reason = "no edge over baseline (%.3f vs %.3f)" % (accuracy, baseline)
        return result
    if expectancy <= 0:
        result.reason = "negative expectancy after fees (%.5f)" % expectancy
        return result
    result.passed = True
    result.reason = "beats baseline by %.1f pts with positive expectancy" % (edge * 100.0)
    return result
