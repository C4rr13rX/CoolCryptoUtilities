from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class SwarmVote:
    horizon: str
    expected_return: float
    confidence: float
    energy: float
    samples: float
    direction_prob: float = 0.5   # P(return > 0) from this cell
    momentum: float = 0.0         # short-term momentum signal
    borda_score: float = 0.0      # filled by BordaConsensus


@dataclass
class ConsensusResult:
    """Aggregated output from BordaConsensus."""
    expected_return: float
    confidence: float
    direction_prob: float
    entropy: float               # Shannon entropy of direction votes (0=unanimous, 1=max disagreement)
    disagreement_penalty: float  # multiplicative confidence penalty in [0, 1]
    horizon_count: int
    dominant_horizon: str        # horizon with highest Borda score


class LinearCell:
    """
    Online linear model with gradient clipping.
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
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1.0:
            grad = grad / grad_norm
        err = max(-1.0, min(1.0, err))
        self.w -= self.lr * grad
        self.bias -= self.lr * err
        if not np.all(np.isfinite(self.w)):
            self.w = np.zeros_like(self.w)
            self.bias = 0.0
        else:
            np.clip(self.w, -10.0, 10.0, out=self.w)
            self.bias = max(-10.0, min(10.0, self.bias))


class BordaConsensus:
    """
    Borda count voting over horizon cells.

    Each horizon is ranked by magnitude of predicted return.
    Borda score = (N-1) for the strongest signal, 0 for weakest.
    Direction votes determine direction_prob and entropy penalty.

    Shannon entropy of the direction distribution penalizes confidence
    when horizons disagree: entropy = 0 (all agree) → no penalty,
    entropy = 1 (50/50 split) → maximum penalty.
    """

    def aggregate(self, votes: List[SwarmVote]) -> ConsensusResult:
        if not votes:
            return ConsensusResult(
                expected_return=0.0, confidence=0.0,
                direction_prob=0.5, entropy=1.0,
                disagreement_penalty=0.5, horizon_count=0,
                dominant_horizon="",
            )

        n = len(votes)

        # ── Borda ranking by |expected_return| ───────────────────────────
        sorted_by_magnitude = sorted(votes, key=lambda v: abs(v.expected_return))
        for rank, vote in enumerate(sorted_by_magnitude):
            vote.borda_score = float(rank)  # 0 = weakest, n-1 = strongest

        # ── Direction votes ───────────────────────────────────────────────
        bullish = sum(1 for v in votes if v.expected_return > 0)
        bearish = n - bullish
        p_bull = bullish / n
        p_bear = bearish / n

        # Shannon entropy of direction (bits normalised to [0,1])
        def _ent(p: float) -> float:
            return 0.0 if p <= 0 else -p * math.log2(p)
        entropy = _ent(p_bull) + _ent(p_bear)  # max = 1.0 when p_bull = p_bear = 0.5

        # Disagreement penalty: 0.5 at max entropy, 1.0 at full consensus
        disagreement_penalty = 1.0 - 0.5 * entropy

        # ── Weighted aggregation (Borda weight × confidence) ──────────────
        total_borda = sum(v.borda_score for v in votes) or 1.0
        numerator_ret = 0.0
        numerator_conf = 0.0
        numerator_mom = 0.0
        denom = 0.0

        for v in votes:
            w = (v.borda_score / total_borda) * v.confidence
            numerator_ret += v.expected_return * w
            numerator_conf += v.confidence * w
            numerator_mom += v.momentum * w
            denom += w

        if denom <= 0:
            agg_ret = float(np.mean([v.expected_return for v in votes]))
            agg_conf = float(np.mean([v.confidence for v in votes]))
        else:
            agg_ret = numerator_ret / denom
            agg_conf = numerator_conf / denom

        # Apply disagreement penalty to confidence
        agg_conf *= disagreement_penalty

        # direction_prob: Borda-weighted fraction in the dominant direction
        direction_prob = max(p_bull, p_bear)
        # Blend with entropy to soften overconfident direction calls
        direction_prob = 0.5 + (direction_prob - 0.5) * disagreement_penalty

        dominant = max(votes, key=lambda v: v.borda_score)

        return ConsensusResult(
            expected_return=agg_ret,
            confidence=max(0.0, min(1.0, agg_conf)),
            direction_prob=max(0.5, min(1.0, direction_prob)),
            entropy=entropy,
            disagreement_penalty=disagreement_penalty,
            horizon_count=n,
            dominant_horizon=dominant.horizon,
        )


class MultiResolutionSwarm:
    """
    Collection of lightweight linear cells across multiple time horizons.

    Improvements over v1:
      - 12-feature vector: adds volume imbalance, momentum window, spread proxy,
        RSI-like overbought/oversold signal
      - BordaConsensus replaces simple weighted average: proper rank-based voting
      - Entropy-based disagreement detection penalises confidence when horizons split
      - Per-horizon momentum tracking (EMA of recent returns) fed back as feature
      - Directional accuracy tracked separately from total accuracy
    """

    _FEATURE_DIM = 12

    def __init__(self, horizons: Iterable[Tuple[str, int]]) -> None:
        self.horizon_defs = list(horizons)
        self.cells: Dict[str, LinearCell] = {}
        self.stats: Dict[str, Dict[str, float]] = {}
        self.energy_alpha = 0.10
        self.mae_alpha = 0.15
        self._momentum_alpha = 0.20          # EMA for per-horizon momentum
        self._momentum: Dict[str, float] = {}
        self._recent_returns: Dict[str, Deque[float]] = {}
        self._borda = BordaConsensus()

        for label, _ in self.horizon_defs:
            self.cells[label] = LinearCell(input_dim=self._FEATURE_DIM)
            self.stats[label] = {
                "correct": 1.0, "total": 2.0,
                "energy": 1.0, "mae": 1.0, "realized": 0.0,
                "dir_correct": 1.0, "dir_total": 2.0,
            }
            self._momentum[label] = 0.0
            self._recent_returns[label] = deque(maxlen=32)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _features(
        self,
        price_slice: np.ndarray,
        sentiment_slice: np.ndarray,
        volume_slice: Optional[np.ndarray] = None,
        horizon_label: str = "",
    ) -> np.ndarray:
        price = price_slice.astype(np.float64)
        mean_price = float(np.mean(price)) or 1e-9
        scale = max(abs(mean_price), 1e-9)

        returns = np.diff(price) / scale
        slope = 0.0
        if price.size > 1:
            x = np.arange(price.size)
            try:
                slope = float(np.polyfit(x, price / scale, 1)[0])
            except Exception:
                slope = 0.0
        vol = float(np.std(returns)) if returns.size else 1e-6
        last_ret = float(returns[-1]) if returns.size else 0.0
        drift = float(np.mean(returns)) if returns.size else 0.0

        sentiment_mean = float(np.mean(sentiment_slice)) if sentiment_slice.size else 0.0
        sentiment_last = float(sentiment_slice[-1]) if sentiment_slice.size else 0.0

        window = max(1, min(price.size, 5))
        ema_fast = (price[-1] - float(np.mean(price[-window:]))) / scale
        ema_slow = (price[-1] - mean_price) / scale

        # RSI-like: fraction of up-moves in recent returns
        if returns.size >= 3:
            rsi_proxy = float(np.sum(returns > 0)) / returns.size
        else:
            rsi_proxy = 0.5
        rsi_signal = (rsi_proxy - 0.5) * 2.0  # normalised -1..+1

        # Volume imbalance: rising vs falling bars by volume
        vol_imbalance = 0.0
        if volume_slice is not None and volume_slice.size >= 2 and returns.size >= 1:
            volumes = volume_slice.astype(np.float64)
            n = min(len(volumes), len(returns))
            up_vol = float(np.sum(volumes[-n:][returns[-n:] > 0]))
            dn_vol = float(np.sum(volumes[-n:][returns[-n:] < 0]))
            total_vol = up_vol + dn_vol
            if total_vol > 0:
                vol_imbalance = (up_vol - dn_vol) / total_vol

        # Per-horizon momentum (EMA)
        momentum = self._momentum.get(horizon_label, 0.0)

        # Spread proxy: high-frequency price jitter relative to drift
        spread_proxy = float(vol / (abs(drift) + 1e-9)) if abs(drift) > 1e-12 else vol * 10.0
        spread_proxy = float(np.clip(spread_proxy, 0.0, 5.0)) / 5.0  # normalise

        return np.array(
            [
                slope,
                vol,
                last_ret,
                sentiment_mean,
                sentiment_last,
                ema_fast,
                ema_slow + drift,
                rsi_signal,
                vol_imbalance,
                momentum,
                spread_proxy,
                drift * 10.0,  # drift amplified for sensitivity
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def vote(
        self,
        price_windows: Dict[str, np.ndarray],
        sentiment_windows: Dict[str, np.ndarray],
        volume_windows: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[SwarmVote]:
        votes: List[SwarmVote] = []
        for label, _ in self.horizon_defs:
            prices = price_windows.get(label)
            sentiments = sentiment_windows.get(label)
            if prices is None or sentiments is None:
                continue
            volumes = (volume_windows or {}).get(label)
            x = self._features(prices, sentiments, volumes, label)
            raw_pred = self.cells[label].predict(x)
            expected = math.tanh(raw_pred)
            stats = self.stats[label]

            accuracy = stats["correct"] / max(1.0, stats["total"])
            dir_acc = stats["dir_correct"] / max(1.0, stats["dir_total"])
            energy = stats.get("energy", 1.0)
            mae = max(1e-6, stats.get("mae", 1.0))
            stability = 1.0 / (1.0 + mae)

            # Confidence incorporates directional accuracy separately
            confidence = (
                0.30
                + 0.40 * max(0.0, min(1.0, accuracy))
                + 0.20 * max(0.0, min(1.0, dir_acc))
                + 0.10 * max(0.2, min(1.2, energy)) * stability
            )
            confidence = max(0.0, min(1.0, confidence))

            # direction_prob for this horizon: blend accuracy and sigmoid(raw_pred)
            cell_dir_prob = 1.0 / (1.0 + math.exp(-raw_pred * 3.0))  # sigmoid, steeper
            direction_prob = 0.4 * dir_acc + 0.6 * cell_dir_prob

            votes.append(SwarmVote(
                horizon=label,
                expected_return=expected,
                confidence=confidence,
                energy=energy,
                samples=stats["total"],
                direction_prob=direction_prob,
                momentum=self._momentum.get(label, 0.0),
            ))
        return votes

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        price_windows: Dict[str, np.ndarray],
        sentiment_windows: Dict[str, np.ndarray],
        realized_returns: Dict[str, float],
        volume_windows: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        for label, _ in self.horizon_defs:
            prices = price_windows.get(label)
            sentiments = sentiment_windows.get(label)
            realized = realized_returns.get(label)
            if prices is None or sentiments is None or realized is None:
                continue
            volumes = (volume_windows or {}).get(label)
            x = self._features(prices, sentiments, volumes, label)
            self.cells[label].update(x, realized)

            pred = self.cells[label].predict(x)
            correct = 1.0 if np.sign(pred) == np.sign(realized) else 0.0
            stats = self.stats[label]

            acc_alpha = 0.05
            stats["correct"] = (1.0 - acc_alpha) * stats["correct"] + acc_alpha * correct
            stats["total"] = (1.0 - acc_alpha) * stats["total"] + acc_alpha * 1.0
            stats["dir_correct"] = (1.0 - acc_alpha) * stats["dir_correct"] + acc_alpha * correct
            stats["dir_total"] = (1.0 - acc_alpha) * stats["dir_total"] + acc_alpha * 1.0

            energy = math.exp(-abs(pred - realized))
            alpha = self.energy_alpha
            stats["energy"] = (1.0 - alpha) * stats.get("energy", 1.0) + alpha * energy

            error = abs(pred - realized)
            mae_alpha = self.mae_alpha
            stats["mae"] = (1.0 - mae_alpha) * stats.get("mae", error) + mae_alpha * error
            stats["realized"] = (1.0 - mae_alpha) * stats.get("realized", 0.0) + mae_alpha * realized

            # Update per-horizon momentum (EMA of realized returns)
            self._recent_returns[label].append(realized)
            mom_alpha = self._momentum_alpha
            self._momentum[label] = (
                (1.0 - mom_alpha) * self._momentum.get(label, 0.0)
                + mom_alpha * realized
            )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_votes(self, votes: List[SwarmVote]) -> Tuple[float, float]:
        """
        Return (weighted_expected_return, aggregate_confidence).

        Uses BordaConsensus + entropy penalty for disagreement detection.
        Backward-compatible with callers that used the old weighted-mean version.
        """
        result = self.consensus(votes)
        return result.expected_return, result.confidence

    def consensus(self, votes: List[SwarmVote]) -> ConsensusResult:
        """Full Borda + entropy consensus — richer output than aggregate_votes."""
        return self._borda.aggregate(votes)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> List[Dict[str, Any]]:
        diag: List[Dict[str, Any]] = []
        for label, stats in self.stats.items():
            total = max(1.0, stats.get("total", 1.0))
            accuracy = stats.get("correct", 0.0) / total
            dir_acc = stats.get("dir_correct", 0.0) / max(1.0, stats.get("dir_total", 1.0))
            diag.append({
                "horizon": label,
                "accuracy": float(max(0.0, min(1.0, accuracy))),
                "dir_accuracy": float(max(0.0, min(1.0, dir_acc))),
                "samples": float(total),
                "energy": float(stats.get("energy", 1.0)),
                "mae": float(stats.get("mae", 0.0)),
                "realized": float(stats.get("realized", 0.0)),
                "momentum": float(self._momentum.get(label, 0.0)),
            })
        return diag

    def weights(self) -> Dict[str, float]:
        raw: List[Tuple[str, float]] = []
        for label, stats in self.stats.items():
            total = max(1.0, stats.get("total", 1.0))
            accuracy = max(0.0, min(1.0, stats.get("correct", 0.0) / total))
            dir_acc = max(0.0, min(1.0, stats.get("dir_correct", 0.0) / max(1.0, stats.get("dir_total", 1.0))))
            energy = max(0.1, stats.get("energy", 1.0))
            mae = max(1e-6, stats.get("mae", 1.0))
            stability = max(0.2, 1.0 / (1.0 + mae))
            # Blend accuracy + directional accuracy
            blended_acc = 0.6 * accuracy + 0.4 * dir_acc
            weight = blended_acc * energy * stability
            raw.append((label, weight))
        denom = sum(w for _, w in raw)
        if denom <= 0:
            n = len(raw)
            return {label: 1.0 / n for label, _ in raw} if n else {}
        return {label: w / denom for label, w in raw}
