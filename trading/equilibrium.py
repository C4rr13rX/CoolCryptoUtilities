from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np


class EquilibriumTracker:
    """
    Tracks how closely the realized trade margins match the model's expectations.
    When the moving window of trades maintains low error and stable confidence,
    we treat the system as being in a Nash-like equilibrium.
    """

    def __init__(self, window: int = 128, tolerance: float = 0.15) -> None:
        self._records: Deque[Dict[str, float]] = deque(maxlen=max(16, window))
        self.tolerance = tolerance

    def observe(self, *, predicted_margin: float, realized_margin: float, confidence: float) -> None:
        error = abs(predicted_margin - realized_margin)
        self._records.append(
            {
                "error": error,
                "realized": realized_margin,
                "confidence": confidence,
            }
        )

    def score(self) -> float:
        if len(self._records) < 8:
            return 0.0
        errors = np.array([rec["error"] for rec in self._records], dtype=np.float32)
        confidences = np.array([rec["confidence"] for rec in self._records], dtype=np.float32)
        realized = np.array([rec["realized"] for rec in self._records], dtype=np.float32)
        mae = float(np.mean(errors))
        confidence_score = float(np.clip(np.mean(confidences), 0.0, 1.0))
        momentum = float(np.mean(np.tanh(np.abs(realized))))
        base = 1.0 / (1.0 + mae)
        return float(base * 0.6 + confidence_score * 0.3 + momentum * 0.1)

    def is_equilibrium(self) -> bool:
        return len(self._records) >= 16 and self.score() >= self.tolerance

    def summary(self) -> Dict[str, float]:
        return {
            "count": float(len(self._records)),
            "score": self.score(),
            "mae": float(np.mean([rec["error"] for rec in self._records])) if self._records else 0.0,
        }
