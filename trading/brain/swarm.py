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
        grad = err * x + self.l2 * self.w
        # Clip gradients to prevent overflow / NaN explosion
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1.0:
            grad = grad / grad_norm
        err = max(-1.0, min(1.0, err))
        self.w -= self.lr * grad
        self.bias -= self.lr * err
        # Clamp weights to sane range — if anything went NaN, reset
        if not np.all(np.isfinite(self.w)):
            self.w = np.zeros_like(self.w)
            self.bias = 0.0
        else:
            np.clip(self.w, -10.0, 10.0, out=self.w)
            self.bias = max(-10.0, min(10.0, self.bias))


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
        self.mae_alpha = 0.15
        self._feature_dim = 8
        for label, _ in self.horizon_defs:
            self.cells[label] = LinearCell(input_dim=self._feature_dim)
            self.stats[label] = {"correct": 1.0, "total": 2.0, "energy": 1.0, "mae": 1.0, "realized": 0.0}

    def _features(self, price_slice: np.ndarray, sentiment_slice: np.ndarray) -> np.ndarray:
        price = price_slice.astype(np.float64)
        mean_price = float(np.mean(price))
        # Normalise price-derived features as fractions of mean price to keep
        # all features in a comparable range and prevent gradient explosion.
        scale = max(abs(mean_price), 1e-9)
        returns = np.diff(price) / scale
        slope = 0.0
        if price.size > 1:
            x = np.arange(price.size)
            slope = float(np.polyfit(x, price / scale, 1)[0])
        vol = float(np.std(returns)) if returns.size else 0.0
        last_ret = float(returns[-1]) if returns.size else 0.0
        sentiment_mean = float(np.mean(sentiment_slice)) if sentiment_slice.size else 0.0
        sentiment_last = float(sentiment_slice[-1]) if sentiment_slice.size else 0.0
        window = max(1, min(price.size, 5))
        ema_fast = (price[-1] - float(np.mean(price[-window:]))) / scale
        ema_slow = (price[-1] - mean_price) / scale
        drift = float(np.mean(returns)) if returns.size else 0.0
        return np.array(
            [
                0.0,  # was mean_price (raw USD) — replaced with zero to keep dim stable
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
            accuracy = stats["correct"] / max(1.0, stats["total"])
            energy = stats.get("energy", 1.0)
            mae = max(1e-6, stats.get("mae", 1.0))
            stability = 1.0 / (1.0 + mae)
            confidence = 0.35 + 0.65 * max(0.0, min(1.0, accuracy)) * max(0.2, min(1.2, energy)) * stability
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
            # Use exponential moving average for accuracy so recent performance
            # is weighted more heavily than stale history.
            acc_alpha = 0.05  # ~20-sample effective window
            stats["correct"] = (1.0 - acc_alpha) * stats["correct"] + acc_alpha * correct
            stats["total"] = (1.0 - acc_alpha) * stats["total"] + acc_alpha * 1.0
            energy = math.exp(-abs(pred - realized))
            alpha = self.energy_alpha
            stats["energy"] = (1.0 - alpha) * stats.get("energy", 1.0) + alpha * energy
            error = abs(pred - realized)
            mae_alpha = self.mae_alpha
            stats["mae"] = (1.0 - mae_alpha) * stats.get("mae", error) + mae_alpha * error
            stats["realized"] = (1.0 - mae_alpha) * stats.get("realized", 0.0) + mae_alpha * realized

    def aggregate_votes(self, votes: List[SwarmVote]) -> Tuple[float, float]:
        """Return (weighted_expected_return, aggregate_confidence) from a list of votes.

        Uses the internal accuracy/energy/stability weights so callers don't
        need to replicate the weighting logic.  When horizons disagree on
        direction the aggregate confidence is penalised proportionally to the
        disagreement spread so callers can tighten thresholds automatically.
        """
        if not votes:
            return 0.0, 0.0
        w = self.weights()
        numerator = 0.0
        denom = 0.0
        conf_num = 0.0
        for vote in votes:
            weight = w.get(vote.horizon, vote.confidence)
            numerator += vote.expected_return * weight
            conf_num += vote.confidence * weight
            denom += weight
        if denom <= 0:
            return 0.0, 0.0
        agg_return = numerator / denom
        agg_conf = conf_num / denom
        # Penalise confidence when horizons disagree on direction.
        if len(votes) >= 2:
            returns = [v.expected_return for v in votes]
            spread = max(returns) - min(returns)
            # spread of 2.0 is maximum possible (tanh range -1..+1)
            disagreement = min(1.0, spread / 2.0)
            agg_conf *= 1.0 - 0.5 * disagreement
        return agg_return, agg_conf

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
                    "mae": float(stats.get("mae", 0.0)),
                    "realized": float(stats.get("realized", 0.0)),
                }
            )
        return diag

    def weights(self) -> Dict[str, float]:
        raw: List[Tuple[str, float]] = []
        for label, stats in self.stats.items():
            total = max(1.0, stats.get("total", 1.0))
            accuracy = max(0.0, min(1.0, stats.get("correct", 0.0) / total))
            energy = max(0.1, stats.get("energy", 1.0))
            mae = max(1e-6, stats.get("mae", 1.0))
            stability = max(0.2, 1.0 / (1.0 + mae))
            weight = accuracy * energy * stability
            raw.append((label, weight))
        denom = sum(weight for _, weight in raw)
        if denom <= 0:
            size = len(raw)
            return {label: 1.0 / size for label, _ in raw} if size else {}
        return {label: weight / denom for label, weight in raw}
