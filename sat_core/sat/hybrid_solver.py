from __future__ import annotations
from typing import Dict, List, Optional, Tuple

def _eval_clause(clause: List[int], assignment: Dict[int, bool]) -> Optional[bool]:
    undecided = False
    for lit in clause:
        var = abs(lit)
        if var not in assignment:
            undecided = True
            continue
        val = assignment[var]
        if (lit > 0 and val) or (lit < 0 and not val):
            return True
    return None if undecided else False

def dpll(clauses: List[List[int]], assignment: Dict[int, bool]) -> Optional[Dict[int, bool]]:
    # Unit propagation
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            res = _eval_clause(clause, assignment)
            if res is False:
                # check for unit clause
                unassigned = [lit for lit in clause if abs(lit) not in assignment]
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    assignment[abs(lit)] = lit > 0
                    changed = True
                else:
                    return None
    # check if all clauses satisfied
    if all(_eval_clause(c, assignment) is True for c in clauses):
        return assignment
    # choose an unassigned variable
    vars_all = {abs(lit) for clause in clauses for lit in clause}
    for var in vars_all:
        if var not in assignment:
            for value in (True, False):
                new_assignment = dict(assignment)
                new_assignment[var] = value
                result = dpll(clauses, new_assignment)
                if result is not None:
                    return result
            return None
    return None

class HybridSATSolver:
    def __init__(self):
        self.clauses: List[List[int]] = []
    def add_clause(self, clause: List[int]) -> None:
        self.clauses.append(list(clause))
    def solve(self) -> Optional[Dict[int, bool]]:
        return dpll(self.clauses, {})
