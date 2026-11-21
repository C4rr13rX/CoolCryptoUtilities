"""tests/test_hard_obstacle_courses.py

Formal Pytest Verification for W1z4rD Hard Cognitive & Software Debugging Obstacle Courses.
Ensures 100% / 100% compliance across all polyglot debugging and engineering tracks.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools.wizard_learning.advanced_obstacle_courses import load_hard_obstacle_tasks

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_FILE = _PROJECT_ROOT / "runtime" / "c0d3r" / "obstacle_course_evaluation_report.json"


def test_obstacle_courses_all_tasks_defined():
    """Validates that all 10 advanced obstacle course tasks are loaded and structured."""
    tasks = load_hard_obstacle_tasks()
    assert len(tasks) == 10, f"Expected 10 hard obstacle tasks, found {len(tasks)}"

    categories = {t.category for t in tasks}
    assert "compiler_debugging" in categories
    assert "scientific_runtime" in categories
    assert "robotics_kinematics" in categories
    assert "gdt_manufacturing" in categories
    assert "algorithmic_deduction" in categories


def test_obstacle_courses_evaluation_report_100_percent():
    """Validates that the latest tutor obstacle course run achieved exactly 100.0% passing score."""
    assert REPORT_FILE.exists(), f"Evaluation report file missing at {REPORT_FILE}"

    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("all_passed") is True, f"Not all obstacle tasks passed: {data.get('passed_tasks')}/{data.get('total_tasks')}"
    assert data.get("overall_percentage") == 100.0, f"Expected 100.0% score, got {data.get('overall_percentage')}%"
    assert data.get("total_tasks") == 10, f"Expected 10 tasks evaluated, got {data.get('total_tasks')}"
    assert data.get("passed_tasks") == 10, f"Expected 10 tasks passed, got {data.get('passed_tasks')}"

    # Verify each individual task has 100.0 score
    for task_res in data.get("results", []):
        assert task_res.get("passed") is True, f"Task {task_res.get('task_key')} failed"
        assert task_res.get("score") == 100.0, f"Task {task_res.get('task_key')} score is {task_res.get('score')}, expected 100.0"
