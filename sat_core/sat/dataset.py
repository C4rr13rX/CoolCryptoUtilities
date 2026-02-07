import random
from typing import List, Set, Tuple

def generate_sat_instance(num_vars: int, num_clauses: int) -> List[Set[int]]:
    clauses = []
    for _ in range(num_clauses):
        clause = set()
        while len(clause) < 3:
            var = random.randint(1, num_vars)
            lit = var if random.random() > 0.5 else -var
            clause.add(lit)
        clauses.append(clause)
    return clauses