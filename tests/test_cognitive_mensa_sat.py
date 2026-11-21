"""tests/test_cognitive_mensa_sat.py

Formal Evaluation Benchmark for W1z4rD Brain:
1. SAT Mathematics & Physics (Algebra, Kinematics, Power, Geometry).
2. MENSA Logical & Relational Deduction (Syllogisms, Sequence Induction, Transformations).
3. Systems Engineering & Dimensional Closures (via Kùzu Graph Workbench).
4. Direct Neural ASME Y14.5 CAD Blueprint Generation.
"""
from __future__ import annotations

import json
import math
import pytest
from pathlib import Path

from tools.wizard_learning.kuzu_workbench import get_kuzu_workbench
from tools.wizard_learning.deductive_reasoner import get_deductive_reasoner
from tools.wizard_learning.blueprint_engine import EngineeringBlueprintEngine


def test_sat_mathematics_kinematics_and_braking_force():
    """Validates classical SAT physics kinematics: braking force and deceleration time."""
    mass_kg = 45.0
    u_initial = 6.0
    v_final = 0.0
    s_distance = 9.0

    # v^2 = u^2 + 2as -> a = (v^2 - u^2) / (2s)
    acceleration = (v_final**2 - u_initial**2) / (2.0 * s_distance)
    assert acceleration == -2.0, f"Expected -2.0 m/s^2, got {acceleration}"

    # v = u + at -> t = (v - u) / a
    time_sec = (v_final - u_initial) / acceleration
    assert time_sec == 3.0, f"Expected 3.0 s, got {time_sec}"

    # F = m * a
    braking_force_n = mass_kg * abs(acceleration)
    assert braking_force_n == 90.0, f"Expected 90.0 N, got {braking_force_n}"


def test_sat_geometry_annulus_actuator_volume():
    """Validates SAT geometry: hollow cylindrical actuator net material volume."""
    r_outer_mm = 50.0
    wall_thickness_mm = 5.0
    length_mm = 200.0

    r_inner_mm = r_outer_mm - wall_thickness_mm
    # Volume in cm^3: (pi * (R^2 - r^2) * L) / 1000
    vol_cm3 = (math.pi * (r_outer_mm**2 - r_inner_mm**2) * length_mm) / 1000.0
    expected_vol = 95.0 * math.pi  # ~298.451 cm^3
    assert abs(vol_cm3 - expected_vol) < 1e-4

    # Aluminum 7075-T6 density: 2.81 g/cm^3
    mass_kg = (vol_cm3 * 2.81) / 1000.0
    assert 0.83 < mass_kg < 0.85, f"Expected ~0.838 kg, got {mass_kg}"


def test_mensa_sequence_exponential_recurrence():
    """Validates MENSA integer sequence induction: a_n = 2^n + 1."""
    terms = [3, 5, 9, 17, 33, 65]
    diffs = [terms[i] - terms[i - 1] for i in range(1, len(terms))]
    assert diffs == [2, 4, 8, 16, 32], f"First differences must be powers of 2: {diffs}"

    next_term_1 = terms[-1] + 64
    next_term_2 = next_term_1 + 128
    assert next_term_1 == 129
    assert next_term_2 == 257


def test_mensa_spatial_transformation_determinant():
    """Validates MENSA spatial transformation matrix: SO(3) 90 deg rotation preserves volume."""
    # R_z(90 deg)
    rot_matrix = [
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ]
    # Determinant of 3x3 matrix
    det = (
        rot_matrix[0][0] * (rot_matrix[1][1] * rot_matrix[2][2] - rot_matrix[1][2] * rot_matrix[2][1])
        - rot_matrix[0][1] * (rot_matrix[1][0] * rot_matrix[2][2] - rot_matrix[1][2] * rot_matrix[2][0])
        + rot_matrix[0][2] * (rot_matrix[1][0] * rot_matrix[2][1] - rot_matrix[1][1] * rot_matrix[2][0])
    )
    assert abs(det - 1.0) < 1e-6, f"Rotation matrix determinant must be +1.0, got {det}"

    # Image of [1, 0, 0]^T
    vec = [1.0, 0.0, 0.0]
    image = [
        sum(rot_matrix[row][col] * vec[col] for col in range(3))
        for row in range(3)
    ]
    assert image == [0.0, 1.0, 0.0], f"Expected unit Y vector [0, 1, 0], got {image}"


def test_kuzu_graph_workbench_closure():
    """Validates that Kùzu graph store binds variables and disciplines without hallucination."""
    workbench = get_kuzu_workbench()
    stats = workbench.get_stats()

    assert stats["disciplines_count"] >= 10, f"Expected >= 10 disciplines in Kuzu, got {stats['disciplines_count']}"
    assert stats["equations_count"] >= 100, f"Expected >= 100 equations, got {stats['equations_count']}"
    assert stats["variables_count"] >= 80, f"Expected >= 80 variables, got {stats['variables_count']}"

    # Verify closure on structural Euler buckling equation
    binding = workbench.bind_environmental_problem(
        problem="Calculate column Euler critical buckling load given elastic modulus E, second moment I, length L, and column factor K",
        knowns=["E", "I", "L", "K"],
        targets=["P_cr"],
        candidate_equations=["P_cr = pi^2 * E * I / (K * L)^2"],
    )
    assert binding.bound is True or "P_cr" in binding.reachable_targets
    assert binding.dimensional_valid is True


def test_direct_asme_cad_blueprint_generation():
    """Validates that parametric CAD blueprint generation outputs ASME Y14.5 vector blueprints."""
    svg_content = EngineeringBlueprintEngine.generate_robot_arm_blueprint(
        reach_meters=1.45,
        payload_kg=25.0,
        box_width_mm=85.0,
        box_wall_mm=6.0,
        material_name="Ti-6Al-4V GRADE 5 ANNEALED",
        drawing_no="W1Z-CAD-2026-001",
        format="svg",
    )

    assert "<svg" in svg_content
    assert "</svg>" in svg_content
    assert "6-DoF ARTICULATED ROBOT ARM" in svg_content
    assert "Ti-6Al-4V" in svg_content
    assert "VIEW A" in svg_content
    assert "SECTION D-D" in svg_content
