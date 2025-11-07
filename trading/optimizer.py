from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ParameterState:
    mean: float
    variance: float
    bounds: Tuple[float, float]


@dataclass
class SignalSpec:
    weight: float
    goal: str = "max"  # "max" or "min"
    clip: Optional[Tuple[float, float]] = None


DEFAULT_SIGNAL_SPECS: Dict[str, SignalSpec] = {
    "dir_accuracy": SignalSpec(weight=0.45, goal="max", clip=(0.0, 1.0)),
    "best_f1": SignalSpec(weight=0.25, goal="max", clip=(0.0, 1.0)),
    "profit_factor": SignalSpec(weight=0.15, goal="max", clip=(0.0, 5.0)),
    "ghost_win_rate": SignalSpec(weight=0.2, goal="max", clip=(0.0, 1.0)),
    "ghost_avg_profit": SignalSpec(weight=0.1, goal="max"),
    "ghost_kelly": SignalSpec(weight=0.1, goal="max", clip=(0.0, 1.0)),
    "kelly_fraction": SignalSpec(weight=0.05, goal="max", clip=(0.0, 1.0)),
    "margin_mae": SignalSpec(weight=0.15, goal="min"),
    "drift_stat": SignalSpec(weight=0.1, goal="min"),
    "positive_ratio": SignalSpec(weight=0.05, goal="max", clip=(0.0, 1.0)),
    "avg_duration_sec": SignalSpec(weight=0.05, goal="min"),
    "ghost_trades_best": SignalSpec(weight=0.05, goal="max"),
}


class BayesianBruteForceOptimizer:
    """
    Lightweight optimizer that blends coarse brute-force search with a Bayesian
    style update. Each parameter is modelled as a Gaussian whose mean/variance
    adapt to observed scores. Cheap enough to run continuously on CPU.
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        *,
        exploration: float = 0.05,
        signal_specs: Optional[Dict[str, SignalSpec]] = None,
    ) -> None:
        self.params: Dict[str, ParameterState] = {}
        for name, (lo, hi) in param_bounds.items():
            mid = (lo + hi) / 2.0
            span = (hi - lo) or 1.0
            self.params[name] = ParameterState(mean=mid, variance=(span / 4.0) ** 2, bounds=(lo, hi))
        self.best_score: float = -math.inf
        self.best_params: Dict[str, float] = {}
        self.exploration = exploration
        self.signal_specs: Dict[str, SignalSpec] = signal_specs or DEFAULT_SIGNAL_SPECS.copy()
        self.signal_history: list[Dict[str, Any]] = []

    def propose(self) -> Dict[str, float]:
        proposal: Dict[str, float] = {}
        for name, state in self.params.items():
            std = math.sqrt(max(state.variance, 1e-6))
            sample = random.gauss(state.mean, std)
            # light epsilon-greedy exploration
            if random.random() < self.exploration:
                sample = random.uniform(*state.bounds)
            lo, hi = state.bounds
            proposal[name] = max(min(sample, hi), lo)
        return proposal

    def _score_signals(self, signals: Dict[str, float]) -> float:
        composite = 0.0
        for name, value in signals.items():
            spec = self.signal_specs.get(name)
            if spec is None:
                continue
            val = float(value)
            if spec.clip:
                lo, hi = spec.clip
                val = max(min(val, hi), lo)
            if spec.goal == "min":
                val = -val
            composite += spec.weight * val
        return composite

    def update(self, params: Dict[str, float], score: float, signals: Optional[Dict[str, float]] = None) -> float:
        if not params:
            return float(score)
        composite_score = float(score)
        if signals:
            composite_score += self._score_signals(signals)
            self.signal_history.append({"signals": dict(signals), "score": composite_score})
            self.signal_history = self.signal_history[-256:]
        if composite_score > self.best_score:
            self.best_score = composite_score
            self.best_params = dict(params)
        for name, value in params.items():
            state = self.params[name]
            lr = 0.2
            delta = value - state.mean
            state.mean += lr * delta * composite_score
            variance_update = (delta ** 2) * max(composite_score, 1e-3)
            state.variance = 0.8 * state.variance + 0.2 * variance_update
            lo, hi = state.bounds
            state.mean = max(min(state.mean, hi), lo)
            state.variance = max(min(state.variance, (hi - lo) ** 2), 1e-6)
        return composite_score

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "params": {
                name: {
                    "mean": state.mean,
                    "variance": state.variance,
                    "bounds": list(state.bounds),
                }
                for name, state in self.params.items()
            },
            "best_score": self.best_score,
            "best_params": self.best_params,
            "exploration": self.exploration,
            "signal_specs": {
                name: {
                    "weight": spec.weight,
                    "goal": spec.goal,
                    "clip": list(spec.clip) if spec.clip else None,
                }
                for name, spec in self.signal_specs.items()
            },
        }

    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.exploration = float(state.get("exploration", self.exploration))
        params_state = state.get("params", {})
        for name, info in params_state.items():
            if name not in self.params:
                bounds = info.get("bounds", [0.0, 1.0])
                self.params[name] = ParameterState(
                    mean=float(info.get("mean", 0.0)),
                    variance=float(info.get("variance", 1.0)),
                    bounds=(float(bounds[0]), float(bounds[1])),
                )
                continue
            param = self.params[name]
            bounds = info.get("bounds", param.bounds)
            param.mean = float(info.get("mean", param.mean))
            param.variance = max(float(info.get("variance", param.variance)), 1e-6)
            param.bounds = (float(bounds[0]), float(bounds[1]))
        self.best_score = float(state.get("best_score", self.best_score))
        self.best_params = {k: float(v) for k, v in state.get("best_params", self.best_params).items()}
        signal_specs = state.get("signal_specs")
        if signal_specs:
            updated: Dict[str, SignalSpec] = {}
            for name, info in signal_specs.items():
                clip = info.get("clip")
                updated[name] = SignalSpec(
                    weight=float(info.get("weight", 0.0)),
                    goal=str(info.get("goal", "max")),
                    clip=(float(clip[0]), float(clip[1])) if clip else None,
                )
            self.signal_specs.update(updated)
