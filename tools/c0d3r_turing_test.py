#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import re

from c0d3r_obstacle_course import _keyword_checker, build_step, _step_mentions, _word_count, ObstacleStep


@dataclass
class TuringTestResult:
    prompt: str
    response: str
    score: float
    notes: List[str]
    memory_hits: List[str] = field(default_factory=list)
    missing_memory: List[str] = field(default_factory=list)
    step_coverage: float = 0.0


@dataclass
class TuringTestStep:
    prompt: str
    expected_keywords: List[str] = field(default_factory=list)
    required_memory: List[str] = field(default_factory=list)
    memory_update: Dict[str, str] = field(default_factory=dict)
    steps: List[ObstacleStep] = field(default_factory=list)
    weight: float = 1.0


def _score_response(prompt: str, response: str, step: TuringTestStep | None = None) -> TuringTestResult:
    notes: List[str] = []
    text = response.strip()
    score = 0.0

    if len(text) > 40:
        score += 0.3
    else:
        notes.append("short_response")

    if re.search(r"[.!?]", text):
        score += 0.2
    else:
        notes.append("no_sentence_punctuation")

    if re.search(r"\bI\b|\bwe\b", text):
        score += 0.2
    else:
        notes.append("no_first_person")

    if re.search(r"\bbecause\b|\btherefore\b|\bso\b", text.lower()):
        score += 0.2
    else:
        notes.append("low_explanatory_connectives")

    if len(set(text.split())) / max(len(text.split()), 1) > 0.5:
        score += 0.1
    else:
        notes.append("low_word_diversity")

    memory_hits: List[str] = []
    missing_memory: List[str] = []
    step_coverage = 0.0
    if step:
        required = [kw.lower() for kw in step.required_memory if kw]
        if required:
            for token in required:
                if token in text.lower():
                    memory_hits.append(token)
                else:
                    missing_memory.append(token)
            if missing_memory:
                notes.append("missing_memory")
                score -= 0.1
            else:
                score += 0.05

        if step.expected_keywords:
            checker = _keyword_checker(step.expected_keywords)
            keyword_score, _, _ = checker(response)
            score += min(0.2, keyword_score / 500.0)

        if step.steps:
            covered = 0
            for obs_step in step.steps:
                checker = obs_step.checker or _keyword_checker(obs_step.expected_keywords, obs_step.required_phrases)
                step_score, _, _ = checker(response)
                if step_score >= 70.0:
                    covered += 1
            step_coverage = covered / max(len(step.steps), 1)
            if step_coverage < 0.5 and len(step.steps) > 1:
                notes.append("low_step_coverage")
                score -= 0.1
            if len(step.steps) > 2 and step_coverage < 0.7:
                notes.append("low_multi_step_coverage")
                score -= 0.15

        expected_mentions = max(2, len(step.steps) // 2) if len(step.steps) > 1 else 0
        if len(step.steps) > 1 and _step_mentions(response) <= expected_mentions:
            notes.append("single_step_response")
            score -= 0.1

        min_words = max(20, len(step.steps) * 10)
        if _word_count(response) < min_words:
            notes.append("too_short_for_steps")
            score -= 0.05

    return TuringTestResult(
        prompt=prompt,
        response=response,
        score=min(max(score, 0.0), 1.0),
        notes=notes,
        memory_hits=memory_hits,
        missing_memory=missing_memory,
        step_coverage=round(step_coverage, 3),
    )


def run_turing_test(
    prompts: List[str | Dict[str, Any]],
    responder: Callable[[str, Dict[str, str]], str],
    context: Dict[str, str] | None = None,
) -> Dict[str, object]:
    context = context or {}
    results: List[TuringTestResult] = []
    memory_state: Dict[str, str] = {}

    steps: List[TuringTestStep] = []
    for entry in prompts:
        if isinstance(entry, str):
            steps.append(TuringTestStep(prompt=entry))
        elif isinstance(entry, dict):
            expected = entry.get("expected_keywords") or entry.get("expected") or []
            required_memory = entry.get("required_memory") or entry.get("memory") or []
            if isinstance(expected, str):
                expected = [expected]
            if isinstance(required_memory, str):
                required_memory = [required_memory]
            raw_steps = entry.get("steps") or []
            parsed_steps: List[ObstacleStep] = []
            if isinstance(raw_steps, list):
                for idx, step_entry in enumerate(raw_steps):
                    if isinstance(step_entry, dict):
                        parsed_steps.append(build_step(step_entry, index=idx))
            steps.append(
                TuringTestStep(
                    prompt=str(entry.get("prompt") or ""),
                    expected_keywords=[str(item) for item in expected],
                    required_memory=[str(item) for item in required_memory],
                    memory_update={str(k): str(v) for k, v in (entry.get("memory_update") or {}).items()},
                    steps=parsed_steps,
                    weight=float(entry.get("weight", 1.0)),
                )
            )

    for step in steps:
        prompt = step.prompt
        if not prompt:
            continue
        response = responder(prompt, context)
        required_memory = step.required_memory or list(memory_state.keys())
        step.required_memory = required_memory
        result = _score_response(prompt, response, step)
        results.append(result)
        if step.memory_update:
            memory_state.update(step.memory_update)
        elif required_memory:
            for token in required_memory:
                if token and token.lower() in response.lower():
                    memory_state[token] = "referenced"

    avg_score = sum(result.score for result in results) / max(len(results), 1)
    return {
        "average_score": round(avg_score, 3),
        "results": [result.__dict__ for result in results],
        "memory_state": memory_state,
    }


__all__ = ["TuringTestResult", "TuringTestStep", "run_turing_test"]
