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
        population_size: int = 12,
        elite_frac: float = 0.35,
        mutation_scale: float = 0.25,
        seed: Optional[int] = 1337,
        deterministic: bool = True,
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
        self.population: list[Dict[str, Any]] = []
        self.population_size = max(2, int(population_size))
        self.elite_frac = float(max(0.1, min(0.75, elite_frac)))
        self.mutation_scale = float(max(0.0, min(1.0, mutation_scale)))
        self.deterministic = bool(deterministic)
        self.seed = int(seed or 0)
        self._rng = random.Random(self.seed)
        self._update_count = 0

    def _rng_state_json(self) -> list[Any]:
        state = self._rng.getstate()

        def _to_jsonable(value: Any) -> Any:
            if isinstance(value, tuple):
                return [_to_jsonable(item) for item in value]
            if isinstance(value, list):
                return [_to_jsonable(item) for item in value]
            return value

        return _to_jsonable(state)

    def _set_rng_state_json(self, state: Any) -> None:
        def _to_tuple(value: Any) -> Any:
            if isinstance(value, list):
                return tuple(_to_tuple(item) for item in value)
            return value

        try:
            restored = _to_tuple(state)
            self._rng.setstate(restored)
        except Exception:
            # If the state payload is malformed or incompatible, fall back to seed.
            self._rng = random.Random(self.seed)

    def _effective_exploration(self) -> float:
        # Anneal exploration as we collect more evidence.
        base = max(0.0, float(self.exploration))
        if base <= 0.0:
            return 0.0
        decay = float(0.985) ** max(0, int(self._update_count))
        return max(0.01, base * decay)

    def _elite_pool(self) -> list[Dict[str, Any]]:
        if not self.population:
            return []
        ordered = sorted(self.population, key=lambda row: float(row.get("score", -math.inf)), reverse=True)
        top_k = max(2, int(math.ceil(len(ordered) * self.elite_frac)))
        return ordered[: max(2, min(len(ordered), top_k))]

    def _pick_parent(self, pool: list[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not pool:
            return None
        # Weight elites higher but keep deterministic variety.
        weights = [1.0 / (idx + 1.0) for idx in range(len(pool))]
        total = sum(weights) or 1.0
        r = self._rng.random() * total
        acc = 0.0
        for row, w in zip(pool, weights):
            acc += w
            if r <= acc:
                return row
        return pool[0]

    def propose(self) -> Dict[str, float]:
        proposal: Dict[str, float] = {}
        exploration = self._effective_exploration()
        pool = self._elite_pool()
        parent_a = pool[0] if pool else None
        parent_b = self._pick_parent(pool[1:]) if len(pool) > 2 else (pool[1] if len(pool) == 2 else None)

        use_genetic = bool(parent_a and parent_b)
        for name, state in self.params.items():
            lo, hi = state.bounds
            std = math.sqrt(max(state.variance, 1e-6))
            std = max(1e-6, std * max(0.15, 1.0 - min(0.8, self._update_count / 100.0)))

            base_val = state.mean
            if self.best_params and name in self.best_params:
                base_val = float(self.best_params[name])

            if use_genetic:
                p1 = float((parent_a.get("params") or {}).get(name, base_val))
                p2 = float((parent_b.get("params") or {}).get(name, base_val))
                mix = self._rng.random()
                sample = mix * p1 + (1.0 - mix) * p2
                # Mutation nudges around the elite manifold.
                if self.mutation_scale > 0:
                    sample += self._rng.gauss(0.0, std * self.mutation_scale)
            else:
                sample = self._rng.gauss(base_val, std)

            # Light epsilon-greedy exploration, but annealed.
            if exploration > 0.0 and self._rng.random() < exploration:
                sample = self._rng.uniform(lo, hi)
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
        signature = tuple((k, round(float(v), 8)) for k, v in sorted(params.items()))
        entry: Dict[str, Any] = {"params": {k: float(v) for k, v in params.items()}, "score": composite_score}
        if signals:
            entry["signals"] = {k: float(v) for k, v in signals.items() if v is not None}
        self.population = [row for row in self.population if row.get("sig") != signature]
        entry["sig"] = signature
        self.population.append(entry)
        self.population.sort(key=lambda row: float(row.get("score", -math.inf)), reverse=True)
        self.population = self.population[: self.population_size]

        for name, value in params.items():
            state = self.params[name]
            lr = 0.18
            delta = float(value) - state.mean
            score_scale = math.tanh(float(composite_score))
            state.mean += lr * delta * score_scale
            variance_update = (delta**2) * max(abs(score_scale), 1e-3)
            state.variance = 0.85 * state.variance + 0.15 * variance_update
            lo, hi = state.bounds
            state.mean = max(min(state.mean, hi), lo)
            state.variance = max(min(state.variance, (hi - lo) ** 2), 1e-6)
        self._update_count += 1
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
            "population": [
                {"params": row.get("params", {}), "score": float(row.get("score", 0.0)), "signals": row.get("signals")}
                for row in self.population
            ],
            "population_size": int(self.population_size),
            "elite_frac": float(self.elite_frac),
            "mutation_scale": float(self.mutation_scale),
            "seed": int(self.seed),
            "deterministic": bool(self.deterministic),
            "update_count": int(self._update_count),
            "rng_state": self._rng_state_json(),
        }

    def set_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        self.exploration = float(state.get("exploration", self.exploration))
        self.population_size = max(2, int(state.get("population_size", self.population_size)))
        self.elite_frac = float(max(0.1, min(0.75, float(state.get("elite_frac", self.elite_frac)))))
        self.mutation_scale = float(max(0.0, min(1.0, float(state.get("mutation_scale", self.mutation_scale)))))
        self.seed = int(state.get("seed", self.seed))
        self.deterministic = bool(state.get("deterministic", self.deterministic))
        self._update_count = int(state.get("update_count", self._update_count))
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
        pop = state.get("population")
        if isinstance(pop, list):
            rebuilt: list[Dict[str, Any]] = []
            for row in pop:
                if not isinstance(row, dict):
                    continue
                params = row.get("params")
                if not isinstance(params, dict):
                    continue
                score = row.get("score", 0.0)
                try:
                    score_f = float(score)
                except Exception:
                    score_f = 0.0
                params_f = {k: float(v) for k, v in params.items()}
                sig = tuple((k, round(float(v), 8)) for k, v in sorted(params_f.items()))
                rebuilt.append(
                    {
                        "params": params_f,
                        "score": score_f,
                        "signals": row.get("signals"),
                        "sig": sig,
                    }
                )
            rebuilt.sort(key=lambda r: float(r.get("score", -math.inf)), reverse=True)
            deduped: list[Dict[str, Any]] = []
            seen: set[Any] = set()
            for row in rebuilt:
                sig = row.get("sig")
                if sig in seen:
                    continue
                seen.add(sig)
                deduped.append(row)
            self.population = deduped[: self.population_size]

        # Re-seed RNG then apply stored state if present.
        self._rng = random.Random(self.seed)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self._set_rng_state_json(rng_state)
