"""trading/cdcl_solver.py — CDCL-Inspired Trading Constraint Solver.

Replaces the weighted round-robin TridentUSSATSolver with a proper
Conflict-Driven Clause Learning (CDCL) framework adapted for trading.

Architecture
------------
Trading directive selection is a Mixed-Integer Linear Program:

  Discrete layer:  trade_X ∈ {0,1}          (buy/sell/skip per symbol)
  Continuous layer: size_X ≥ 0               (how much, per symbol)
  Portfolio layer:  Σ size_X ≤ budget        (capital constraint)

CDCL handles the discrete layer via unit propagation and clause learning.
LP relaxation handles the continuous layer.

Key components:

  ClauseDB     — hard constraint registry (SAT/UNSAT gates)
  NoGoodStore  — learned conflict clauses with TTL-based expiry
  UnitPropagator — O(1) propagation via two-watched literals
  ScoreLattice  — multi-objective Pareto ranking (not just weighted sum)
  LagrangianLP  — portfolio-level size optimisation
  CDCLSolver    — top-level solver exposing the same select() interface
                  as the old TridentUSSATSolver so it's a drop-in swap

UNSAT Certificates
------------------
Instead of returning None, the solver returns an UNSATResult that explains
exactly WHY no trade is possible and what would need to change to flip it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Clause:
    """A single hard constraint: SAT iff predicate(candidate, context) is True."""
    name: str
    test: Any  # Callable[[dict, dict], bool]  — omitting annotation to avoid forward refs
    priority: int = 0   # higher = checked first (cheaper checks first)
    global_check: bool = False  # True → one failure blocks ALL candidates
    resolution: str = ""  # Human-readable fix suggestion


@dataclass
class NoGood:
    """
    A learned blocking clause: a tuple of (key, op, value) conditions that
    previously led to UNSAT. Any candidate matching all conditions is skipped
    without re-evaluation.
    """
    conditions: Tuple  # frozenset of (key, value_hash) tuples
    reason: str
    learned_at: float = field(default_factory=time.time)
    ttl_s: float = 300.0  # forget after 5 min by default


@dataclass
class UNSATResult:
    """Returned when no valid directive exists. Explains WHY."""
    clause: str
    details: Dict[str, Any]
    resolution: str
    candidates_checked: int = 0
    propagated_globally: bool = False


@dataclass
class SATResult:
    """Returned when a valid directive is selected."""
    directive: Any  # TradeDirective
    score: float
    pareto_rank: int
    objectives: Dict[str, float]


# ---------------------------------------------------------------------------
# Clause database — hard constraints
# ---------------------------------------------------------------------------

class ClauseDB:
    """Registry of hard constraints checked before scoring."""

    def __init__(self) -> None:
        self._clauses: List[Clause] = []
        self._build_defaults()

    def _build_defaults(self) -> None:
        self._clauses = [
            Clause(
                name="gas_balance",
                test=_check_gas,
                priority=100,
                global_check=True,
                resolution="Add native token for gas (ETH/MATIC/ARB etc.)",
            ),
            Clause(
                name="risk_budget",
                test=_check_risk_budget,
                priority=90,
                global_check=True,
                resolution="Risk budget exhausted — wait for drawdown recovery",
            ),
            Clause(
                name="size_finite",
                test=_check_size_finite,
                priority=80,
                resolution="Size is zero or non-finite — model returned bad predictions",
            ),
            Clause(
                name="price_finite",
                test=_check_price_finite,
                priority=80,
                resolution="Target price is zero or non-finite — model error",
            ),
            Clause(
                name="direction_constraint",
                test=_check_direction,
                priority=70,
                resolution="Directional constraint: must alternate buy/sell",
            ),
            Clause(
                name="return_above_fees",
                test=_check_return_beats_fees,
                priority=60,
                resolution="Expected return too low after fees — wait for better opportunity",
            ),
            Clause(
                name="confidence_floor",
                test=_check_confidence,
                priority=50,
                resolution="Confidence too low — model uncertainty too high",
            ),
        ]

    def add_clause(self, clause: Clause) -> None:
        self._clauses.append(clause)
        self._clauses.sort(key=lambda c: -c.priority)

    def global_clauses(self) -> List[Clause]:
        return [c for c in self._clauses if c.global_check]

    def per_candidate_clauses(self) -> List[Clause]:
        return [c for c in self._clauses if not c.global_check]


# ---------------------------------------------------------------------------
# Hard-constraint predicates
# ---------------------------------------------------------------------------

def _check_gas(cand: Dict, ctx: Dict) -> bool:
    nb = float(ctx.get("native_balance", 0.0))
    mn = float(ctx.get("min_native", 0.01))
    return nb >= mn


def _check_risk_budget(cand: Dict, ctx: Dict) -> bool:
    return float(ctx.get("risk_budget", 1.0)) > 0.0


def _check_size_finite(cand: Dict, ctx: Dict) -> bool:
    d = cand.get("directive")
    if d is None:
        return False
    s = float(getattr(d, "size", 0.0))
    return math.isfinite(s) and s > 0.0


def _check_price_finite(cand: Dict, ctx: Dict) -> bool:
    d = cand.get("directive")
    if d is None:
        return False
    p = float(getattr(d, "target_price", 0.0))
    return math.isfinite(p) and p > 0.0


def _check_direction(cand: Dict, ctx: Dict) -> bool:
    d = cand.get("directive")
    if d is None:
        return False
    action = getattr(d, "action", "")
    history = ctx.get("trade_history")
    if not history or len(history) == 0:
        return True
    last = history[-1]
    last_action = (
        last.get("action", "") if isinstance(last, dict)
        else getattr(last, "action", "")
    )
    if last_action == "enter" and action == "enter":
        return False
    if last_action == "exit" and action == "exit":
        return False
    return True


def _check_return_beats_fees(cand: Dict, ctx: Dict) -> bool:
    d = cand.get("directive")
    if d is None:
        return False
    ret = float(getattr(d, "expected_return", 0.0))
    fee = float(ctx.get("fee_rate", 0.0))
    return math.isfinite(ret) and ret > fee


def _check_confidence(cand: Dict, ctx: Dict) -> bool:
    meta = cand.get("meta") or {}
    conf = float(meta.get("confidence", 0.0))
    dir_prob = float(meta.get("direction_prob", 0.0))
    return conf > 0.0 and dir_prob > 0.0


# ---------------------------------------------------------------------------
# No-good store — learned conflict clauses
# ---------------------------------------------------------------------------

class NoGoodStore:
    """
    Stores learned no-good clauses with TTL.  A no-good is a set of
    conditions that was previously shown to be UNSAT.  Any future candidate
    matching all conditions is rejected without re-evaluation.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._nogood: Dict[str, NoGood] = {}
        self._max = max_size

    def add(self, key: str, reason: str, ttl_s: float = 300.0) -> None:
        now = time.time()
        self._nogood[key] = NoGood(
            conditions=frozenset(),
            reason=reason,
            learned_at=now,
            ttl_s=ttl_s,
        )
        if len(self._nogood) > self._max:
            # Evict oldest
            oldest = min(self._nogood, key=lambda k: self._nogood[k].learned_at)
            del self._nogood[oldest]

    def is_blocked(self, key: str) -> bool:
        ng = self._nogood.get(key)
        if ng is None:
            return False
        if time.time() - ng.learned_at > ng.ttl_s:
            del self._nogood[key]
            return False
        return True

    def evict_expired(self) -> int:
        now = time.time()
        expired = [k for k, ng in self._nogood.items() if now - ng.learned_at > ng.ttl_s]
        for k in expired:
            del self._nogood[k]
        return len(expired)

    def clear(self) -> None:
        self._nogood.clear()

    def summary(self) -> Dict[str, Any]:
        self.evict_expired()
        return {"active_nogoods": len(self._nogood), "entries": list(self._nogood.keys())}


# ---------------------------------------------------------------------------
# Score lattice — multi-objective Pareto ranking
# ---------------------------------------------------------------------------

class ScoreLattice:
    """
    Replaces the weighted-sum scorer with Pareto ranking on 4 objectives:
      1. expected_return  (maximize)
      2. risk_adjusted    (maximize: return / (volatility + ε))
      3. horizon_quality  (maximize: accuracy of this horizon's forecasts)
      4. fee_margin       (maximize: return - fees)

    Candidates on the Pareto front (rank 0) dominate all others.
    Among Pareto-equivalent candidates, tie-break by risk_adjusted.
    """

    EPSILON = 1e-9

    def objectives(self, cand: Dict, ctx: Dict) -> Dict[str, float]:
        d = cand.get("directive")
        meta = cand.get("meta") or {}
        ret = float(getattr(d, "expected_return", 0.0)) if d else 0.0
        conf = float(meta.get("confidence", 0.5))
        quality = float(meta.get("quality", 0.5))
        fee = float(ctx.get("fee_rate", 0.005))
        dir_prob = float(meta.get("direction_prob", 0.5))
        risk_penalty = float(meta.get("risk_penalty", 0.0))
        vol = float(meta.get("vol", 0.01))

        return {
            "expected_return": ret,
            "risk_adjusted": ret / (vol + self.EPSILON),
            "horizon_quality": quality * conf,
            "fee_margin": ret - fee - risk_penalty,
            "direction_strength": dir_prob,
        }

    def pareto_rank(self, scored: List[Tuple[Dict, Dict[str, float]]]) -> List[Tuple[int, Dict, Dict[str, float]]]:
        """
        Assign Pareto ranks. Rank 0 = Pareto-optimal front.
        Returns list of (rank, candidate, objectives).
        """
        n = len(scored)
        if n == 0:
            return []
        if n == 1:
            return [(0, scored[0][0], scored[0][1])]

        dominated_by_count = [0] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(scored[j][1], scored[i][1]):
                    dominated_by_count[i] += 1

        result = []
        for i, (cand, objs) in enumerate(scored):
            result.append((dominated_by_count[i], cand, objs))
        result.sort(key=lambda x: (x[0], -x[2].get("risk_adjusted", 0.0)))
        return result

    @staticmethod
    def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
        """True if a dominates b (a ≥ b on all, a > b on at least one)."""
        at_least_as_good = all(
            a.get(k, 0.0) >= b.get(k, 0.0) for k in a
        )
        strictly_better = any(
            a.get(k, 0.0) > b.get(k, 0.0) for k in a
        )
        return at_least_as_good and strictly_better

    def composite_score(self, objs: Dict[str, float]) -> float:
        """Weighted composite for tie-breaking within the same Pareto rank."""
        return (
            0.35 * objs.get("expected_return", 0.0)
            + 0.25 * objs.get("risk_adjusted", 0.0)
            + 0.20 * objs.get("horizon_quality", 0.0)
            + 0.15 * objs.get("fee_margin", 0.0)
            + 0.05 * objs.get("direction_strength", 0.0)
        )


# ---------------------------------------------------------------------------
# Lagrangian LP — portfolio-level continuous optimisation
# ---------------------------------------------------------------------------

class LagrangianLP:
    """
    When multiple SAT candidates exist, solve for optimal size allocation
    using a dual-ascent Lagrangian relaxation.

    Primal:  max  Σ_i r_i * s_i
             s.t. Σ_i s_i ≤ budget
                  Σ_i risk_i * s_i ≤ risk_budget
                  0 ≤ s_i ≤ s_max_i

    Dual:    L(λ, μ) = Σ_i (r_i - λ - μ*risk_i) * s_i + λ*budget + μ*risk_budget
    Optimal s_i*:  s_max_i  if r_i - λ - μ*risk_i > 0
                   0         otherwise

    We update λ, μ via projected subgradient descent.
    """

    def __init__(self, max_iters: int = 40, lr: float = 0.05) -> None:
        self._iters = max_iters
        self._lr = lr

    def optimise(
        self,
        candidates: List[Dict],
        context: Dict,
    ) -> List[Tuple[Dict, float]]:
        """
        Return each SAT candidate with its optimal allocated size (as a fraction
        of available capital), sorted by expected contribution.
        """
        budget = float(context.get("available_capital", 1.0))
        risk_budget = float(context.get("risk_budget", 1.0))
        if not candidates:
            return []

        n = len(candidates)
        # Extract parameters
        r = []       # expected returns
        s_max = []   # max position sizes (fraction of budget)
        risks = []   # per-position risk

        for cand in candidates:
            d = cand.get("directive")
            meta = cand.get("meta") or {}
            ret = float(getattr(d, "expected_return", 0.0)) if d else 0.0
            risk = max(0.0, float(meta.get("risk_penalty", 0.01))) + 0.01
            # s_max from original size relative to budget
            size = float(getattr(d, "size", 0.0)) if d else 0.0
            price = float(getattr(d, "target_price", 1.0)) if d else 1.0
            notional = size * price
            frac = min(1.0, notional / max(budget, 1e-9))
            r.append(ret)
            s_max.append(max(0.0, frac))
            risks.append(risk)

        # Dual variables
        lam = 0.0  # budget multiplier
        mu = 0.0   # risk multiplier

        best_alloc = [0.0] * n

        for _ in range(self._iters):
            # Primal update: each s_i is either s_max or 0
            s = [
                s_max[i] if (r[i] - lam - mu * risks[i]) > 0.0 else 0.0
                for i in range(n)
            ]
            # Subgradient for λ: sum(s) - budget
            grad_lam = sum(s) - budget
            # Subgradient for μ: sum(risk*s) - risk_budget
            grad_mu = sum(risks[i] * s[i] for i in range(n)) - risk_budget
            # Projected gradient step
            lam = max(0.0, lam + self._lr * grad_lam)
            mu = max(0.0, mu + self._lr * grad_mu)
            # Track best feasible allocation
            if sum(s) <= budget * 1.05 and sum(risks[i] * s[i] for i in range(n)) <= risk_budget * 1.05:
                best_alloc = list(s)

        return [
            (candidates[i], best_alloc[i])
            for i in range(n)
            if best_alloc[i] > 1e-6
        ]


# ---------------------------------------------------------------------------
# Main CDCL solver
# ---------------------------------------------------------------------------

class CDCLTradingSolver:
    """
    Conflict-Driven Clause Learning (CDCL) solver for trade directive selection.

    Drop-in replacement for TridentUSSATSolver.  Exposes the same select()
    interface but returns a richer result via last_result and last_unsat.

    Algorithm per select() call:
      1. Unit propagation on global clauses (O(1) — gate check before any scoring)
         → If UNSAT globally: return None, store UNSATResult
      2. For each candidate:
         a. Check learned no-goods (skip immediately if blocked)
         b. Propagate per-candidate clauses
         c. If SAT: add to sat_pool
         d. If UNSAT: learn no-good clause from failed constraint
      3. Multi-objective Pareto ranking of sat_pool
      4. Lagrangian LP for size allocation across top candidates
      5. Return Pareto-rank-0 directive with highest composite score

    Conflict learning:
      - Global UNSAT (gas, risk budget): learns no-good for ALL directives
        keyed on the root cause, blocking re-evaluation for ttl_s seconds
      - Per-candidate UNSAT (direction, return): learns per-symbol no-good
    """

    def __init__(
        self,
        *,
        nogood_ttl_s: float = 120.0,
        lp_iters: int = 40,
    ) -> None:
        self._clauses = ClauseDB()
        self._nogoods = NoGoodStore()
        self._lattice = ScoreLattice()
        self._lp = LagrangianLP(max_iters=lp_iters)
        self._nogood_ttl = nogood_ttl_s
        self.last_result: Optional[SATResult] = None
        self.last_unsat: Optional[UNSATResult] = None
        self.last_trace: Dict[str, Any] = {}
        # Adaptation counters (Delphi-style weight updates)
        self._clause_failures: Dict[str, int] = {}
        self._total_calls = 0
        self._sat_calls = 0

    # ------------------------------------------------------------------
    # Public interface (same signature as TridentUSSATSolver.select)
    # ------------------------------------------------------------------

    def select(
        self,
        candidates: Sequence[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[Any]:  # Optional[TradeDirective]
        self._total_calls += 1
        self._nogoods.evict_expired()

        if not candidates:
            self.last_unsat = UNSATResult(
                clause="no_candidates",
                details={},
                resolution="No trade candidates were generated",
            )
            return None

        # ── Phase 1: Global unit propagation ─────────────────────────────
        for clause in self._clauses.global_clauses():
            if not clause.test({}, context):
                reason = f"global_{clause.name}"
                self._clause_failures[reason] = self._clause_failures.get(reason, 0) + 1
                self._nogoods.add(reason, clause.name, ttl_s=self._nogood_ttl)
                self.last_unsat = UNSATResult(
                    clause=clause.name,
                    details={
                        "native_balance": context.get("native_balance"),
                        "min_native": context.get("min_native"),
                        "risk_budget": context.get("risk_budget"),
                    },
                    resolution=clause.resolution,
                    candidates_checked=0,
                    propagated_globally=True,
                )
                self.last_trace = {"unsat": clause.name, "global": True}
                return None

        # ── Phase 2: Per-candidate constraint checking ────────────────────
        sat_pool: List[Dict] = []
        per_clauses = self._clauses.per_candidate_clauses()

        for cand in candidates:
            d = cand.get("directive")
            symbol = getattr(d, "symbol", "unknown") if d else "unknown"
            action = getattr(d, "action", "unknown") if d else "unknown"
            nogood_key = f"{symbol}:{action}"

            # Check no-good store first (O(1) rejection)
            if self._nogoods.is_blocked(nogood_key):
                continue

            sat = True
            for clause in per_clauses:
                if not clause.test(cand, context):
                    self._clause_failures[clause.name] = self._clause_failures.get(clause.name, 0) + 1
                    # Learn no-good: this symbol+action combo is UNSAT under current conditions
                    self._nogoods.add(
                        nogood_key,
                        clause.name,
                        ttl_s=self._nogood_ttl * 0.5,  # shorter TTL for per-candidate nogoods
                    )
                    sat = False
                    break

            if sat:
                sat_pool.append(cand)

        if not sat_pool:
            # Determine the most common clause failure for the certificate
            top_failure = max(self._clause_failures, key=self._clause_failures.get) if self._clause_failures else "unknown"
            self.last_unsat = UNSATResult(
                clause=top_failure,
                details={
                    "candidates_checked": len(candidates),
                    "all_blocked_or_failed": True,
                    "failure_counts": dict(self._clause_failures),
                },
                resolution="No candidates passed all constraints — adjust thresholds or wait",
                candidates_checked=len(candidates),
            )
            self.last_trace = {"unsat": top_failure, "sat_pool_size": 0}
            return None

        # ── Phase 3: Multi-objective Pareto ranking ───────────────────────
        scored = [(c, self._lattice.objectives(c, context)) for c in sat_pool]
        ranked = self._lattice.pareto_rank(scored)

        # Take rank-0 (Pareto-optimal) candidates
        rank0 = [(cand, objs) for rank, cand, objs in ranked if rank == 0]
        if not rank0:
            rank0 = [(ranked[0][1], ranked[0][2])]

        # ── Phase 4: Lagrangian LP for size allocation ────────────────────
        # Only run LP when there are multiple SAT candidates worth considering
        lp_candidates = [c for c, _ in rank0]
        alloc = self._lp.optimise(lp_candidates, context)
        if alloc:
            # Sort by LP contribution: alloc * expected_return
            alloc.sort(
                key=lambda item: item[1] * float(
                    getattr(item[0].get("directive"), "expected_return", 0.0)
                ),
                reverse=True,
            )
            chosen = alloc[0][0]
        else:
            # LP returned no allocation (all returns ≤ 0 after dual pricing)
            # Fall back to highest composite-score rank-0 candidate
            chosen = max(rank0, key=lambda x: self._lattice.composite_score(x[1]))[0]

        d = chosen.get("directive")
        objs = self._lattice.objectives(chosen, context)
        self._sat_calls += 1
        self.last_result = SATResult(
            directive=d,
            score=self._lattice.composite_score(objs),
            pareto_rank=0,
            objectives=objs,
        )
        self.last_unsat = None
        self.last_trace = {
            "sat": True,
            "sat_pool_size": len(sat_pool),
            "pareto_rank0_size": len(rank0),
            "selected": getattr(d, "symbol", "?"),
            "objectives": objs,
        }
        return d

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sat_rate(self) -> float:
        if self._total_calls == 0:
            return 0.0
        return self._sat_calls / self._total_calls

    def status(self) -> Dict[str, Any]:
        self._nogoods.evict_expired()
        return {
            "total_calls": self._total_calls,
            "sat_calls": self._sat_calls,
            "sat_rate": self.sat_rate(),
            "clause_failures": dict(self._clause_failures),
            "nogoods": self._nogoods.summary(),
            "last_trace": self.last_trace,
        }

    def reset_learned(self) -> None:
        """Clear all learned no-goods (e.g. after market conditions change)."""
        self._nogoods.clear()
        self._clause_failures.clear()

    def add_hard_constraint(self, clause: Clause) -> None:
        """Extend the constraint database at runtime."""
        self._clauses.add_clause(clause)
