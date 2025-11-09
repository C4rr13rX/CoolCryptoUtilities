from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class _RpcStats:
    score: float = field(default=10.0)   # lower is better
    latency: float = field(default=0.3)  # seconds
    last_fail: float = field(default=0.0)


class RpcHealthTracker:
    """
    Tiny helper that keeps a rolling score for each RPC endpoint so the wallet
    backend can prefer fast, healthy providers without hardcoding a priority
    list. The score blends the most recent latency with exponential decay and
    penalizes failures for a short cooldown window.
    """

    def __init__(
        self,
        *,
        decay: float = 0.85,
        failure_penalty: float = 6.0,
        cooldown_sec: float = 25.0,
        floor_latency: float = 0.08,
    ) -> None:
        self.decay = max(0.1, min(0.99, float(decay)))
        self.failure_penalty = max(1.0, float(failure_penalty))
        self.cooldown_sec = max(1.0, float(cooldown_sec))
        self.floor_latency = max(0.01, float(floor_latency))
        self._lock = threading.Lock()
        self._stats: Dict[str, _RpcStats] = {}

    # ------------------------------------------------------------------ helpers
    def _ensure(self, url: str) -> _RpcStats:
        with self._lock:
            row = self._stats.get(url)
            if row is None:
                row = _RpcStats()
                self._stats[url] = row
            return row

    def record_success(self, url: str, latency: float) -> None:
        row = self._ensure(url)
        with self._lock:
            lat = max(self.floor_latency, float(latency))
            row.latency = lat
            row.score = (row.score * self.decay) + (lat * (1.0 - self.decay))
            row.last_fail = 0.0

    def record_failure(self, url: str) -> None:
        row = self._ensure(url)
        with self._lock:
            row.score += self.failure_penalty
            row.last_fail = time.time()

    def _penalized_score(self, url: str) -> float:
        row = self._ensure(url)
        score = float(row.score)
        if row.last_fail:
            ago = time.time() - row.last_fail
            if ago < self.cooldown_sec:
                # Apply a sliding penalty that shrinks as the cooldown expires.
                frac = 1.0 - (ago / self.cooldown_sec)
                score += self.failure_penalty * (1.0 + frac)
        return score

    def score(self, url: str) -> float:
        """Public hook so callers can use the weighted score in their own sorts."""
        return self._penalized_score(url)

    def rank(self, urls: Iterable[str]) -> List[str]:
        """
        Return URLs ordered from healthiest → riskiest.
        Unknown endpoints bubble toward the front so the wallet still tries
        new providers instead of sticking to a stale default forever.
        """
        uniq = []
        seen = set()
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            uniq.append(url)
        if not uniq:
            return []

        def _sort_key(endpoint: str) -> float:
            if endpoint not in self._stats:
                # Fresh endpoints get a slight boost so they are tried early.
                return -math.inf
            return self._penalized_score(endpoint)

        ranked = sorted(uniq, key=_sort_key)
        return ranked

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Expose current stats for debugging/telemetry."""
        with self._lock:
            return {
                url: {
                    "score": row.score,
                    "latency": row.latency,
                    "last_fail": row.last_fail,
                }
                for url, row in self._stats.items()
            }


__all__ = ["RpcHealthTracker"]
