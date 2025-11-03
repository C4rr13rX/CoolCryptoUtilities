from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class ParameterState:
    mean: float
    variance: float
    bounds: Tuple[float, float]


class BayesianBruteForceOptimizer:
    """
    Lightweight optimizer that blends coarse brute-force search with a Bayesian
    style update. Each parameter is modelled as a Gaussian whose mean/variance
    adapt to observed scores. Cheap enough to run continuously on CPU.
    """

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], *, exploration: float = 0.05) -> None:
        self.params: Dict[str, ParameterState] = {}
        for name, (lo, hi) in param_bounds.items():
            mid = (lo + hi) / 2.0
            span = (hi - lo) or 1.0
            self.params[name] = ParameterState(mean=mid, variance=(span / 4.0) ** 2, bounds=(lo, hi))
        self.best_score: float = -math.inf
        self.best_params: Dict[str, float] = {}
        self.exploration = exploration

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

    def update(self, params: Dict[str, float], score: float) -> None:
        if not params:
            return
        if score > self.best_score:
            self.best_score = score
            self.best_params = dict(params)
        for name, value in params.items():
            state = self.params[name]
            lr = 0.2
            delta = value - state.mean
            state.mean += lr * delta * score
            variance_update = (delta ** 2) * max(score, 1e-3)
            state.variance = 0.8 * state.variance + 0.2 * variance_update
            lo, hi = state.bounds
            state.mean = max(min(state.mean, hi), lo)
            state.variance = max(min(state.variance, (hi - lo) ** 2), 1e-6)

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
