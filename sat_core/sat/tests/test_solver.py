import random
import pytest
from sat_core.sat.solver import NovelSATSolver

def test_simple_sat():
    solver = NovelSATSolver()
    solver.add_clause([1, -2])
    solver.add_clause([-1, 2])
    is_sat, assignment = solver.solve()
    assert is_sat
    assert len(assignment) == 2

def test_simple_unsat():
    solver = NovelSATSolver()
    solver.add_clause([1])
    solver.add_clause([-1])
    is_sat, assignment = solver.solve()
    assert not is_sat

def test_empty():
    solver = NovelSATSolver()
    is_sat, assignment = solver.solve()
    assert is_sat
    assert len(assignment) == 0
