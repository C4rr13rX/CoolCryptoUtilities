from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class StrategyScore:
    name: str
    score: float
    last_update: float


class SwarmStrategySelector:
    """
    Tiny scorer for a handful of micro-strategies; ranks by recent profit/penalty.
    Lightweight to fit low-power hosts.
    """

    def __init__(self, max_strategies: int = 5, decay: float = 0.92) -> None:
        self.max_strategies = max(1, max_strategies)
        self.decay = decay
        self.scores: Dict[str, StrategyScore] = {}

    def update(self, name: str, profit: float, ts: float) -> None:
        entry = self.scores.get(name, StrategyScore(name=name, score=0.0, last_update=ts))
        gap = max(0.0, ts - entry.last_update)
        # Simple exponential decay on idle time to favor recent performers.
        entry.score *= self.decay ** min(gap / 60.0, 10.0)
        entry.score += profit
        entry.last_update = ts
        self.scores[name] = entry
        if len(self.scores) > self.max_strategies * 3:
            self._prune()

    def best(self) -> Tuple[str, float]:
        if not self.scores:
            return "", 0.0
        best_entry = max(self.scores.values(), key=lambda s: s.score)
        return best_entry.name, best_entry.score

    def _prune(self) -> None:
        ranked = sorted(self.scores.values(), key=lambda s: s.score, reverse=True)
        keep = ranked[: self.max_strategies]
        self.scores = {s.name: s for s in keep}
