from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ScenarioResult:
    label: str
    expected_return: float
    confidence: float


class ScenarioReactor:
    """
    Builds optimistic / pessimistic / neutral scenario adjustments for a trade.

    The scenarios help sanity-check decisions; if the spread between optimistic
    and pessimistic exceeds a tolerance the trade size can be reduced or deferred.
    """

    def __init__(self, tolerance: float = 0.015) -> None:
        self.tolerance = tolerance

    def analyse(self, base_expected: float, confidence: float, volatility: float) -> List[ScenarioResult]:
        volatility = max(1e-6, volatility)
        optimistic = base_expected + volatility * 1.5
        pessimistic = base_expected - volatility * 1.5
        neutral = base_expected
        return [
            ScenarioResult("optimistic", optimistic, min(1.0, confidence + 0.1)),
            ScenarioResult("neutral", neutral, confidence),
            ScenarioResult("pessimistic", pessimistic, max(0.0, confidence - 0.1)),
        ]

    def divergence(self, scenarios: List[ScenarioResult]) -> float:
        values = [s.expected_return for s in scenarios]
        if not values:
            return 0.0
        return max(values) - min(values)

    def should_defer(self, scenarios: List[ScenarioResult]) -> bool:
        return self.divergence(scenarios) > self.tolerance
