from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, Optional


@dataclass
class ProfitSample:
    ts: float
    pnl: float


class EquilibriumTracker:
    """
    Tracks recent ghost PnL to approximate an \"equilibrium\" brightness.
    Brightness rises when positive PnL becomes denser in time; fades when
    losses/absence increase. Lightweight for low-power hosts.
    """

    def __init__(self, window_sec: int = 600) -> None:
        self.window_sec = max(60, window_sec)
        self.samples: Deque[ProfitSample] = deque()
        self._last_brightness: float = 0.0
        self._last_trend: float = 0.0

    def record(self, pnl: float, ts: Optional[float] = None) -> None:
        now = ts or time.time()
        self.samples.append(ProfitSample(ts=now, pnl=pnl))
        self._trim(now)
        self._update(now)

    def snapshot(self) -> Dict[str, float]:
        return {
            "brightness": self._last_brightness,
            "trend": self._last_trend,
            "count": float(len(self.samples)),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_sec
        while self.samples and self.samples[0].ts < cutoff:
            self.samples.popleft()

    def _update(self, now: float) -> None:
        if not self.samples:
            self._last_trend = -0.1
            self._last_brightness = max(0.0, self._last_brightness * 0.9)
            return
        positives = [s for s in self.samples if s.pnl > 0]
        negatives = [s for s in self.samples if s.pnl < 0]
        density = len(self.samples) / max(1.0, self.window_sec)
        pos_density = len(positives) / max(1.0, self.window_sec)
        avg_pos = sum(s.pnl for s in positives) / max(1, len(positives)) if positives else 0.0
        avg_neg = sum(abs(s.pnl) for s in negatives) / max(1, len(negatives)) if negatives else 0.0
        brightness = max(0.0, pos_density * (1.0 + avg_pos)) - avg_neg * 0.1
        brightness *= 1.0 + density
        trend = brightness - self._last_brightness
        self._last_trend = 0.7 * self._last_trend + 0.3 * trend
        self._last_brightness = 0.7 * self._last_brightness + 0.3 * brightness
