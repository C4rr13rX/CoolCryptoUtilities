"""tools/wizard_learning/kuzu_workbench.py

Kùzu Mathematical Workbench for W1z4rD Brain & C0d3rV2.

Purpose:
  Binds unbounded engineering and scientific problems by formulating the environment
  as a mathematical graph problem until it is resolved.
  
  Prevents hallucinations:
    - Never declares a problem solved based on prose assertions.
    - Requires deterministic equation-matrix closure from known environmental variables
      to target quantities.
    - Validates dimensional consistency using SymPy.
    - If variables are unbound or equations are missing, generates targeted mathematical
      sub-questions rather than guessing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)


@dataclass
class EnvironmentalBindingResult:
    """Outcome of binding an environmental problem to the mathematical graph."""
    bound: bool
    closure: str  # "closed" | "unbound"
    problem: str
    knowns: List[str]
    targets: List[str]
    reachable_targets: List[str]
    missing_targets: List[str]
    equations: List[str]
    chain_paths: Dict[str, List[str]]
    dimensional_valid: bool
    anomalies: List[str]
    sub_questions: List[str]
    workbench_equations: List[Dict[str, Any]]
    summary: str


class WizardKuzuWorkbench:
    """Interactive graph-backed mathematical workbench."""

    def __init__(self) -> None:
        self._store = None
        self._ensure_store()

    def _ensure_store(self):
        if self._store is None:
            try:
                from services.graph_store import get_graph_store, sync_graph_from_django
                self._store = get_graph_store()
                if self._store:
                    # Quick check if empty, sync if needed
                    stats = self.get_stats()
                    if stats.get("equations_count", 0) == 0:
                        sync_graph_from_django()
            except Exception as e:
                _logger.warning("Could not initialize Kuzu store in WizardKuzuWorkbench: %s", e)
        return self._store

    def close(self) -> None:
        if self._store:
            try:
                from services.graph_store import close_graph_store
                close_graph_store()
            except Exception:
                pass
            self._store = None

    def is_available(self) -> bool:
        return self._ensure_store() is not None

    def get_stats(self) -> Dict[str, Any]:
        store = self._ensure_store()
        if not store:
            return {"available": False, "equations_count": 0, "disciplines_count": 0, "variables_count": 0}
        try:
            res_eq = store.conn.execute("MATCH (e:Equation) RETURN count(e)")
            eq_cnt = res_eq.get_next()[0] if res_eq.has_next() else 0

            res_disc = store.conn.execute("MATCH (d:Discipline) RETURN count(d)")
            disc_cnt = res_disc.get_next()[0] if res_disc.has_next() else 0

            res_var = store.conn.execute("MATCH (v:Variable) RETURN count(v)")
            var_cnt = res_var.get_next()[0] if res_var.has_next() else 0

            return {
                "available": True,
                "db_path": str(store.db_path),
                "equations_count": eq_cnt,
                "disciplines_count": disc_cnt,
                "variables_count": var_cnt,
            }
        except Exception as e:
            return {"available": True, "error": str(e), "equations_count": 0}

    def search_workbench(
        self,
        query: str,
        discipline: str = "",
        variable: str = "",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Search equations in Kùzu by query, discipline, or variable."""
        store = self._ensure_store()
        if not store:
            return []
        hits: List[Dict[str, Any]] = []
        try:
            if variable:
                var_hits = store.search_by_variable(variable, limit=limit)
                hits.extend(var_hits)
            if discipline:
                disc_hits = store.search_by_discipline(discipline, limit=limit)
                hits.extend(disc_hits)
            if query:
                q_hits = store.search_equations(query, limit=limit)
                hits.extend(q_hits)

            # Deduplicate by eq_id or equation text
            seen = set()
            unique = []
            for h in hits:
                key = h.get("eq_id") or h.get("equation")
                if key not in seen:
                    seen.add(key)
                    unique.append(h)
            return unique[:limit]
        except Exception as e:
            _logger.error("Error searching workbench: %s", e)
            return []

    def bind_environmental_problem(
        self,
        problem: str,
        knowns: List[str],
        targets: List[str],
        candidate_equations: Optional[List[str]] = None,
        dimensions: Optional[Dict[str, str]] = None,
    ) -> EnvironmentalBindingResult:
        """
        Formulate and bind an environmental problem as a mathematical graph problem.

        Steps:
        1. Query Kùzu graph store for all equations touching the knowns, targets, or keywords.
        2. Combine with candidate equations provided by the solver or brain.
        3. Check mathematical closure using SymPy and variable dependency chain.
        4. Validate dimensions.
        5. If unbound, generate deterministic sub-questions identifying the missing physics.
        """
        store = self._ensure_store()
        knowns_clean = [str(k).strip() for k in knowns if str(k).strip()]
        targets_clean = [str(t).strip() for t in targets if str(t).strip()]

        # 1. Gather equations from Kùzu graph workbench
        workbench_hits: List[Dict[str, Any]] = []
        if store:
            # Query by known variables
            for k in knowns_clean:
                workbench_hits.extend(store.search_by_variable(k, limit=10))
            # Query by target variables
            for t in targets_clean:
                workbench_hits.extend(store.search_by_variable(t, limit=10))
            # Query by problem terms
            tokens = [tok for tok in re.findall(r"[A-Za-z]{3,}", problem) if tok.lower() not in {"what", "how", "find", "calculate", "when", "with", "from"}]
            for tok in tokens[:4]:
                workbench_hits.extend(store.search_equations(tok, limit=5))

        # Deduplicate workbench equations
        seen_eqs: Set[str] = set()
        collected_eq_strings: List[str] = []
        unique_workbench_hits: List[Dict[str, Any]] = []

        for h in workbench_hits:
            eq_text = h.get("equation", "").strip()
            if eq_text and eq_text not in seen_eqs:
                seen_eqs.add(eq_text)
                collected_eq_strings.append(eq_text)
                unique_workbench_hits.append(h)

        if candidate_equations:
            for ce in candidate_equations:
                ce_clean = str(ce).strip()
                if ce_clean and ce_clean not in seen_eqs:
                    seen_eqs.add(ce_clean)
                    collected_eq_strings.append(ce_clean)

        # 2. Mathematical Binding Analysis
        from tools.c0d3rV2.matrix_helpers import assess_binding

        analysis = assess_binding(
            equations=collected_eq_strings,
            knowns=knowns_clean,
            targets=targets_clean,
            dimensions=dimensions,
        )

        anomalies: List[str] = []
        sub_questions: List[str] = []

        if analysis.get("rejected_equations"):
            for rej in analysis["rejected_equations"]:
                anomalies.append(f"Rejected equation '{rej['equation']}': {rej['reason']}")

        chain = analysis.get("chain", {})
        reachable = chain.get("reachable", [])
        unreachable = chain.get("unreachable", [])
        paths = chain.get("paths", {})
        dim_val = analysis.get("dimensional_validation", {})
        dim_ok = dim_val.get("valid", True)

        if not dim_ok:
            for inv in dim_val.get("invalid", []):
                anomalies.append(f"Dimensional mismatch in equation {inv.get('equation')}: left={inv.get('left')} != right={inv.get('right')}")

        # If any target is unreachable or dimensions failed, problem is UNBOUND
        is_bound = bool(targets_clean) and not unreachable and dim_ok and len(analysis.get("parseable_equations", [])) > 0

        if not is_bound:
            if not targets_clean:
                anomalies.append("No mathematical targets specified to bind the problem.")
                sub_questions.append("What exact physical quantity or variable is the target of this environment?")
            for missing_target in unreachable:
                sub_questions.append(
                    f"What validated physical relation connects knowns ({', '.join(knowns_clean)}) to target '{missing_target}'?"
                )

        closure_status = "closed" if is_bound else "unbound"

        summary_lines = [
            f"=== Environmental Mathematical Workbench ===",
            f"Problem: {problem[:100]}",
            f"Closure Status: {closure_status.upper()}",
            f"Known variables: {', '.join(knowns_clean) or 'None'}",
            f"Target variables: {', '.join(targets_clean) or 'None'}",
            f"Reachable targets: {', '.join(reachable)}",
            f"Missing / Unbound targets: {', '.join(unreachable) or 'None'}",
            f"Dimensional Consistency: {'PASS' if dim_ok else 'FAIL'}",
            f"Equations active on workbench: {len(collected_eq_strings)}",
        ]
        if is_bound:
            summary_lines.append("Proof Paths:")
            for tgt, path_steps in paths.items():
                summary_lines.append(f"  {tgt} <= " + " -> ".join(path_steps))

        summary = "\n".join(summary_lines)

        return EnvironmentalBindingResult(
            bound=is_bound,
            closure=closure_status,
            problem=problem,
            knowns=knowns_clean,
            targets=targets_clean,
            reachable_targets=reachable,
            missing_targets=unreachable,
            equations=collected_eq_strings,
            chain_paths=paths,
            dimensional_valid=dim_ok,
            anomalies=anomalies,
            sub_questions=sub_questions,
            workbench_equations=unique_workbench_hits,
            summary=summary,
        )


_DEFAULT_WORKBENCH: Optional[WizardKuzuWorkbench] = None


def get_kuzu_workbench() -> WizardKuzuWorkbench:
    global _DEFAULT_WORKBENCH
    if _DEFAULT_WORKBENCH is None:
        _DEFAULT_WORKBENCH = WizardKuzuWorkbench()
    return _DEFAULT_WORKBENCH
