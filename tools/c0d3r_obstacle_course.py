#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import json
import re
import time


@dataclass
class ObstacleTask:
    key: str
    title: str
    description: str
    prompt: str
    checker: Callable[[str], Tuple[float, str, Dict[str, Any]]]
    max_score: float = 100.0
    timeout_s: int = 600
    complexity_level: int = 5
    steps: List["ObstacleStep"] = field(default_factory=list)
    completion_threshold: float = 0.7
    min_word_count: int = 0


@dataclass
class ObstacleResult:
    key: str
    score: float
    max_score: float
    passed: bool
    feedback: str
    details: Dict[str, Any]


@dataclass
class ObstacleStep:
    step_id: str
    title: str
    description: str
    expected_keywords: List[str]
    required_phrases: List[str] = field(default_factory=list)
    weight: float = 1.0
    checker: Callable[[str], Tuple[float, str, Dict[str, Any]]] | None = None


@dataclass
class StepResult:
    step_id: str
    score: float
    max_score: float
    passed: bool
    feedback: str
    details: Dict[str, Any]


def _normalize_tokens(values: List[str]) -> List[str]:
    cleaned: List[str] = []
    for token in values:
        if not token:
            continue
        text = str(token).strip().lower()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _keyword_checker(expected: List[str], required: List[str] | None = None) -> Callable[[str], Tuple[float, str, Dict[str, Any]]]:
    expected_lower = _normalize_tokens(expected)
    required_lower = _normalize_tokens(required or [])

    def _check(response: str) -> Tuple[float, str, Dict[str, Any]]:
        response_lower = (response or "").lower()
        hits = [word for word in expected_lower if word in response_lower]
        required_hits = [word for word in required_lower if word in response_lower]
        coverage = len(hits) / max(len(expected_lower), 1)
        required_ok = len(required_hits) == len(required_lower)
        score = coverage * 100.0
        if required_lower and not required_ok:
            score *= 0.6
        feedback = "keywords matched" if hits else "missing expected keywords"
        if required_lower and not required_ok:
            feedback = "missing required phrases"
        return score, feedback, {
            "matched": hits,
            "expected": expected_lower,
            "required": required_lower,
            "required_hits": required_hits,
        }

    return _check


def _step_mentions(text: str) -> int:
    if not text:
        return 0
    patterns = [
        r"\bstep\s*\d+",
        r"\bphase\s*\d+",
        r"^\s*\d+\.",
        r"\btask\s*\d+",
    ]
    count = 0
    for pat in patterns:
        count += len(re.findall(pat, text, re.IGNORECASE | re.MULTILINE))
    return count


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def build_step(entry: Dict[str, Any], *, index: int = 0) -> ObstacleStep:
    expected = entry.get("expected_keywords") or entry.get("expected") or []
    required = entry.get("required_phrases") or entry.get("required") or []
    if isinstance(expected, str):
        expected = [expected]
    if isinstance(required, str):
        required = [required]
    checker = entry.get("checker")
    if not callable(checker):
        checker = _keyword_checker([str(item) for item in expected], [str(item) for item in required])
    return ObstacleStep(
        step_id=str(entry.get("id") or entry.get("key") or entry.get("title") or f"step_{index + 1}"),
        title=str(entry.get("title") or f"Step {index + 1}"),
        description=str(entry.get("description") or ""),
        expected_keywords=[str(item) for item in expected],
        required_phrases=[str(item) for item in required],
        weight=float(entry.get("weight", 1.0)),
        checker=checker,
    )


def build_task(entry: Dict[str, Any]) -> ObstacleTask:
    expected = entry.get("expected_keywords") or entry.get("expected") or []
    if isinstance(expected, str):
        expected = [expected]
    required = entry.get("required_phrases") or entry.get("required") or []
    if isinstance(required, str):
        required = [required]
    checker = entry.get("checker")
    if not callable(checker):
        checker = _keyword_checker([str(item) for item in expected], [str(item) for item in required])
    steps_raw = entry.get("steps") or []
    steps: List[ObstacleStep] = []
    if isinstance(steps_raw, list):
        for idx, step in enumerate(steps_raw):
            if isinstance(step, dict):
                steps.append(build_step(step, index=idx))
    if not steps and expected:
        steps = [build_step({"expected_keywords": expected, "required_phrases": required}, index=0)]
    return ObstacleTask(
        key=str(entry.get("key") or entry.get("id") or entry.get("title") or "task"),
        title=str(entry.get("title") or "Obstacle task"),
        description=str(entry.get("description") or ""),
        prompt=str(entry.get("prompt") or ""),
        checker=checker,
        max_score=float(entry.get("max_score", 100.0)),
        timeout_s=int(entry.get("timeout_s", 600)),
        complexity_level=int(entry.get("complexity_level", 5)),
        steps=steps,
        completion_threshold=float(entry.get("completion_threshold", 0.7)),
        min_word_count=int(entry.get("min_word_count", 0)),
    )


def load_tasks(path: str | Path) -> List[ObstacleTask]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks") if isinstance(payload, dict) else payload
    if not isinstance(raw_tasks, list):
        return []
    return [build_task(entry) for entry in raw_tasks if isinstance(entry, dict)]


def _evaluate_steps(task: ObstacleTask, response: str) -> Tuple[List[StepResult], float, float]:
    if not task.steps:
        score, feedback, details = task.checker(response)
        return (
            [
                StepResult(
                    step_id=task.key,
                    score=score,
                    max_score=task.max_score,
                    passed=score >= task.max_score * task.completion_threshold,
                    feedback=feedback,
                    details=details,
                )
            ],
            score,
            1.0 if score >= task.max_score * task.completion_threshold else 0.0,
        )

    results: List[StepResult] = []
    total_weight = sum(step.weight for step in task.steps) or 1.0
    weighted_score = 0.0
    covered = 0
    for step in task.steps:
        checker = step.checker or _keyword_checker(step.expected_keywords, step.required_phrases)
        score, feedback, details = checker(response)
        max_score = step.weight * 100.0
        weighted_score += (score / 100.0) * step.weight
        passed = score >= 70.0
        if passed:
            covered += 1
        results.append(
            StepResult(
                step_id=step.step_id,
                score=score,
                max_score=max_score,
                passed=passed,
                feedback=feedback,
                details=details,
            )
        )
    coverage = covered / max(len(task.steps), 1)
    normalized = (weighted_score / total_weight) * 100.0
    return results, normalized, coverage


def _evaluate_task(task: ObstacleTask, response: str) -> ObstacleResult:
    step_results, detail_score, coverage = _evaluate_steps(task, response)
    step_mentions = _step_mentions(response)
    min_words = task.min_word_count or max(20, len(task.steps) * 12)
    words = _word_count(response)
    length_score = min(1.0, words / max(min_words, 1))
    penalty = 1.0
    expected_mentions = max(2, len(task.steps) // 2) if len(task.steps) > 1 else 0
    if len(task.steps) > 1 and coverage < 0.5:
        penalty *= 0.7
    if len(task.steps) > 1 and step_mentions <= expected_mentions:
        penalty *= 0.7
    if len(task.steps) > 1 and coverage < task.completion_threshold:
        penalty *= 0.8
    blended = (detail_score * 0.6 + coverage * 100.0 * 0.3 + length_score * 100.0 * 0.1) * penalty
    final_score = min(task.max_score, blended)
    passed = final_score >= task.max_score * task.completion_threshold and coverage >= task.completion_threshold
    feedback = "passed" if passed else "needs more complete multi-step coverage"
    details = {
        "coverage": round(coverage, 3),
        "detail_score": round(detail_score, 2),
        "length_score": round(length_score, 3),
        "word_count": words,
        "step_mentions": step_mentions,
        "expected_step_mentions": expected_mentions,
        "steps": [result.__dict__ for result in step_results],
    }
    return ObstacleResult(
        key=task.key,
        score=round(final_score, 2),
        max_score=task.max_score,
        passed=passed,
        feedback=feedback,
        details=details,
    )


def run_obstacle_course(
    tasks: List[ObstacleTask],
    responder: Callable[[str, Dict[str, Any]], str],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = context or {}
    results: List[ObstacleResult] = []
    for task in tasks:
        started = time.time()
        response = responder(task.prompt, context)
        result = _evaluate_task(task, response)
        result.details["elapsed_s"] = round(time.time() - started, 3)
        results.append(result)
    total_score = sum(result.score for result in results)
    total_max = sum(result.max_score for result in results) or 1.0
    return {
        "score": round(total_score, 2),
        "max_score": round(total_max, 2),
        "passed": total_score >= total_max * 0.7,
        "results": [result.__dict__ for result in results],
    }


__all__ = [
    "ObstacleTask",
    "ObstacleResult",
    "ObstacleStep",
    "StepResult",
    "load_tasks",
    "run_obstacle_course",
]
