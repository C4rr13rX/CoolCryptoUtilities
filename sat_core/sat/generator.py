import random
from typing import List

def generate_sat_problem(num_vars: int, num_clauses: int) -> List[List[int]]:
    clauses = []
    for _ in range(num_clauses):
        clause = []
        while len(clause) < 3:
            var = random.randint(1, num_vars)
            lit = var if random.random() > 0.5 else -var
            if lit not in clause and -lit not in clause:
                clause.append(lit)
        clauses.append(clause)
    return clauses
