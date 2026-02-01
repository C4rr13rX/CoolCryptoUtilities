import random
import json
from typing import List, Tuple

def generate_sat_problem(num_vars: int, num_clauses: int) -> List[List[int]]:
    clauses = []
    for _ in range(num_clauses):
        clause = []
        vars_in_clause = random.randint(1, 4)
        for _ in range(vars_in_clause):
            var = random.randint(1, num_vars)
            lit = var if random.random() > 0.5 else -var
            if lit not in clause and -lit not in clause:
                clause.append(lit)
        if clause:
            clauses.append(clause)
    return clauses

def run_benchmark(solver_class, problems: List[Tuple[int, int]], trials: int = 5) -> dict:
    results = {
        "problems": [],
        "avg_time": [],
        "success_rate": []
    }
    
    import time
    for vars, clauses in problems:
        total_time = 0
        successes = 0
        for _ in range(trials):
            problem = generate_sat_problem(vars, clauses)
            solver = solver_class()
            for clause in problem:
                solver.add_clause(clause)
            
            start = time.time()
            result = solver.solve()
            end = time.time()
            
            if result is not None:
                successes += 1
            total_time += (end - start)
            
        results["problems"].append(f"{vars}vars_{clauses}clauses")
        results["avg_time"].append(total_time / trials)
        results["success_rate"].append(successes / trials)
        
    return results