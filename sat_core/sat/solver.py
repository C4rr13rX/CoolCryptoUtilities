import random
from typing import List, Dict

class HybridSATSolver:
    def __init__(self):
        self.clauses = []
        self.variables = set()
        
    def add_clause(self, clause: List[int]):
        self.clauses.append(clause)
        self.variables.update(abs(x) for x in clause)
        
    def solve(self) -> Dict[int, bool]:
        assignment = {var: random.choice([True, False]) 
                     for var in self.variables}
        max_flips = len(self.variables) * 4
        
        for _ in range(max_flips):
            unsat_clauses = self._get_unsat_clauses(assignment)
            if not unsat_clauses:
                return assignment
                
            clause = random.choice(unsat_clauses)
            var = abs(random.choice(clause))
            assignment[var] = not assignment[var]
            
        return None
    
    def _get_unsat_clauses(self, assignment: Dict[int, bool]) -> List[List[int]]:
        return [clause for clause in self.clauses 
                if not self._is_clause_satisfied(clause, assignment)]
    
    def _is_clause_satisfied(self, clause: List[int], 
                           assignment: Dict[int, bool]) -> bool:
        return any((lit > 0 and assignment[abs(lit)]) or
                   (lit < 0 and not assignment[abs(lit)])
                   for lit in clause)