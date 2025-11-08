from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class SwarmVote:
    horizon: str
    expected_return: float
    confidence: float
    energy: float
    samples: float


class LinearCell:
    """
    Simple linear model trained via online gradient descent.
    """

    def __init__(self, input_dim: int, lr: float = 0.01, l2: float = 1e-4) -> None:
        self.lr = lr
        self.l2 = l2
        self.w = np.zeros((input_dim,), dtype=np.float64)
        self.bias = 0.0

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x) + self.bias)

    def update(self, x: np.ndarray, y: float) -> None:
        pred = self.predict(x)
        err = pred - y
        self.w -= self.lr * (err * x + self.l2 * self.w)
        self.bias -= self.lr * err


class MultiResolutionSwarm:
    """
    Maintains a collection of lightweight models operating at different horizons.

    Each horizon cell receives aggregated features (mean, slope, volatility, sentiment)
    derived from price windows. The swarm outputs aggregate votes and keeps a rolling
    accuracy score for self-weighting.
    """

    def __init__(self, horizons: Iterable[Tuple[str, int]]) -> None:
        self.horizon_defs = list(horizons)
        self.cells: Dict[str, LinearCell] = {}
        self.stats: Dict[str, Dict[str, float]] = {}
        self.energy_alpha = 0.1
        self._feature_dim = 8
        for label, _ in self.horizon_defs:
            self.cells[label] = LinearCell(input_dim=self._feature_dim)
            self.stats[label] = {"correct": 1.0, "total": 2.0, "energy": 1.0}

    def _features(self, price_slice: np.ndarray, sentiment_slice: np.ndarray) -> np.ndarray:
        price = price_slice.astype(np.float64)
        returns = np.diff(price)
        slope = 0.0
        if price.size > 1:
            x = np.arange(price.size)
            slope = float(np.polyfit(x, price, 1)[0])
        vol = float(np.std(returns)) if returns.size else 0.0
        mean_price = float(np.mean(price))
        last_ret = float(returns[-1]) if returns.size else 0.0
        sentiment_mean = float(np.mean(sentiment_slice)) if sentiment_slice.size else 0.0
        sentiment_last = float(sentiment_slice[-1]) if sentiment_slice.size else 0.0
        window = max(1, min(price.size, 5))
        ema_fast = price[-1] - float(np.mean(price[-window:]))
        ema_slow = price[-1] - float(np.mean(price))
        drift = float(np.mean(returns)) if returns.size else 0.0
        return np.array(
            [
                mean_price,
                slope,
                vol,
                last_ret,
                sentiment_mean,
                sentiment_last,
                ema_fast,
                ema_slow + drift,
            ],
            dtype=np.float64,
        )

    def vote(
        self,
        price_windows: Dict[str, np.ndarray],
        sentiment_windows: Dict[str, np.ndarray],
    ) -> List[SwarmVote]:
        votes: List[SwarmVote] = []
        for label, _ in self.horizon_defs:
            prices = price_windows.get(label)
            sentiments = sentiment_windows.get(label)
            if prices is None or sentiments is None:
                continue
            x = self._features(prices, sentiments)
            expected = math.tanh(self.cells[label].predict(x))
            stats = self.stats[label]
            accuracy = stats["correct"] / stats["total"]
            confidence = 0.5 + 0.5 * math.tanh(accuracy * 2.0)
            energy = stats.get("energy", 1.0)
            confidence = 0.4 + 0.6 * max(0.0, min(1.0, accuracy)) * max(0.2, min(1.2, energy))
            votes.append(
                SwarmVote(
                    horizon=label,
                    expected_return=expected,
                    confidence=confidence,
                    energy=energy,
                    samples=stats["total"],
                )
            )
        return votes

    def learn(
        self,
        price_windows: Dict[str, np.ndarray],
        sentiment_windows: Dict[str, np.ndarray],
        realized_returns: Dict[str, float],
    ) -> None:
        for label, _ in self.horizon_defs:
            prices = price_windows.get(label)
            sentiments = sentiment_windows.get(label)
            realized = realized_returns.get(label)
            if prices is None or sentiments is None or realized is None:
                continue
            x = self._features(prices, sentiments)
            self.cells[label].update(x, realized)
            pred = self.cells[label].predict(x)
            correct = 1.0 if np.sign(pred) == np.sign(realized) else 0.0
            stats = self.stats[label]
            stats["correct"] += correct
            stats["total"] += 1.0
            energy = math.exp(-abs(pred - realized))
            alpha = self.energy_alpha
            stats["energy"] = (1.0 - alpha) * stats.get("energy", 1.0) + alpha * energy

    def diagnostics(self) -> List[Dict[str, float]]:
        diag: List[Dict[str, float]] = []
        for label, stats in self.stats.items():
            total = max(1.0, stats.get("total", 1.0))
            accuracy = stats.get("correct", 0.0) / total
            diag.append(
                {
                    "horizon": label,
                    "accuracy": float(max(0.0, min(1.0, accuracy))),
                    "samples": float(total),
                    "energy": float(stats.get("energy", 1.0)),
                }
            )
        return diag

    def weights(self) -> Dict[str, float]:
        raw: List[Tuple[str, float]] = []
        for label, stats in self.stats.items():
            total = max(1.0, stats.get("total", 1.0))
            accuracy = max(0.0, min(1.0, stats.get("correct", 0.0) / total))
            energy = max(0.1, stats.get("energy", 1.0))
            weight = accuracy * energy
            raw.append((label, weight))
        denom = sum(weight for _, weight in raw)
        if denom <= 0:
            size = len(raw)
            return {label: 1.0 / size for label, _ in raw} if size else {}
        return {label: weight / denom for label, weight in raw}
