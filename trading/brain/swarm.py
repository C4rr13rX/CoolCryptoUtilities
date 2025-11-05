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
        for label, _ in self.horizon_defs:
            self.cells[label] = LinearCell(input_dim=6)
            self.stats[label] = {"correct": 1.0, "total": 2.0}

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
        return np.array([mean_price, slope, vol, last_ret, sentiment_mean, sentiment_last], dtype=np.float64)

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
            expected = self.cells[label].predict(x)
            accuracy = self.stats[label]["correct"] / self.stats[label]["total"]
            confidence = 0.5 + 0.5 * math.tanh(accuracy * 2.0)
            votes.append(SwarmVote(horizon=label, expected_return=expected, confidence=confidence))
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

