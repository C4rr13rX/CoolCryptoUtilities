"""tools/wizard_learning/advanced_obstacle_courses.py

Advanced Multi-Track Hard Obstacle Courses for W1z4rD Cognitive Engine.

Covers:
1. Polyglot Compiler & Concurrency Debugging (Rust, C++, ROS2, PyTorch, Julia, Three.js)
2. Robotics Kinematics, Singularities & Actuator Thermal Dynamics
3. ASME Y14.5 GD&T Precision Tolerancing (MMC Bonus Tolerances)
4. Symplectic Hamiltonian Dynamical Systems
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DATA_FILE = _PROJECT_ROOT / "data" / "wizard_learning" / "hard_obstacle_courses.json"
_logger = logging.getLogger("AdvancedObstacleCourses")


@dataclass
class HardObstacleTask:
    key: str
    title: str
    category: str
    prompt: str
    error_context: str
    expected_keywords: List[str]
    required_phrases: List[str]
    diagnostic_feedback: str
    senior_solution: str
    max_score: float = 100.0
    passing_threshold: float = 0.80

    def evaluate_response(self, response: str) -> Tuple[float, bool, str, Dict[str, Any]]:
        """
        Rigorously scores the model response based on technical accuracy,
        presence of mandatory phrases, keywords, and problem-solving depth.
        """
        resp_clean = (response or "").strip()
        resp_lower = resp_clean.lower()
        words = len(re.findall(r"\b\w+\b", resp_clean))

        # Check expected keywords
        kw_hits = [kw for kw in self.expected_keywords if kw.lower() in resp_lower]
        kw_ratio = len(kw_hits) / max(len(self.expected_keywords), 1)

        # Check required phrases
        req_hits = [req for req in self.required_phrases if req.lower() in resp_lower]
        req_ok = len(req_hits) == len(self.required_phrases)

        # Scoring weights: 50% required phrases, 35% keywords, 15% depth/length
        phrase_score = (len(req_hits) / max(len(self.required_phrases), 1)) * 50.0
        kw_score = kw_ratio * 35.0
        length_score = min(15.0, (words / 30.0) * 15.0)

        total_score = phrase_score + kw_score + length_score

        # Full compliance bonus
        if req_ok and kw_ratio >= 0.7:
            total_score = 100.0
        elif not req_ok:
            total_score = min(total_score * 0.75, 75.0)

        passed = total_score >= (self.max_score * self.passing_threshold) and req_ok
        feedback = "Passed with full technical compliance (100%)" if (passed and total_score >= 100.0) else (
            f"Passed ({round(total_score, 1)}%)" if passed else
            f"Missing required phrases: {[r for r in self.required_phrases if r.lower() not in resp_lower]}. "
            f"Missing keywords: {[k for k in self.expected_keywords if k.lower() not in resp_lower]}."
        )

        details = {
            "score": round(total_score, 2),
            "passed": passed,
            "word_count": words,
            "keyword_hits": kw_hits,
            "required_hits": req_hits,
            "required_ok": req_ok,
            "diagnostic_feedback": self.diagnostic_feedback,
        }
        return round(total_score, 2), passed, feedback, details


def load_hard_obstacle_tasks() -> List[HardObstacleTask]:
    """Loads all hard obstacle tasks from the JSON definition."""
    if not _DATA_FILE.exists():
        raise FileNotFoundError(f"Hard obstacle course definition file missing: {_DATA_FILE}")

    with open(_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: List[HardObstacleTask] = []
    for item in data.get("tasks", []):
        tasks.append(
            HardObstacleTask(
                key=item["key"],
                title=item["title"],
                category=item["category"],
                prompt=item["prompt"],
                error_context=item.get("error_context", ""),
                expected_keywords=item.get("expected_keywords", []),
                required_phrases=item.get("required_phrases", []),
                diagnostic_feedback=item.get("diagnostic_feedback", ""),
                senior_solution=item.get("senior_solution", ""),
                max_score=100.0,
                passing_threshold=float(data.get("passing_threshold", 0.80)),
            )
        )
    return tasks
