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
        """Build the three scenarios.

        UNITS: ``base_expected`` and ``volatility`` must be in the SAME units,
        and because ``tolerance`` is a fraction (0.015 = 1.5%) that unit has to
        be a RETURN FRACTION. ``optimistic`` adds them together, so a caller
        passing an absolute price deviation for ``volatility`` and a margin for
        ``base_expected`` produces a sum with no unit -- and, since
        ``divergence`` reduces to exactly ``3 * volatility`` (see below), a
        ``should_defer`` that ranks symbols by PRICE instead of by risk.
        ``trading/bot.py`` did precisely that until 2026-09-07 and deferred
        69% of CBBTC against 0% of every symbol priced under $1.
        """
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
        """Defer when the scenarios disagree by more than ``tolerance``.

        Note what this does NOT consider: the edge. For scenarios built by
        ``analyse`` the divergence is
        ``(b + 1.5v) - (b - 1.5v) == 3v`` exactly -- ``base_expected``
        cancels -- so a trade with a large positive expected margin is
        deferred on precisely the same terms as one with a large negative
        margin. The rule is a volatility ceiling of ``tolerance / 3``, and
        reading it as anything richer than that has misled two passes.
        """
        return self.divergence(scenarios) > self.tolerance
