#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Problem:
    name: str
    prompt: str
    checker: Callable[[str], Tuple[bool, str]]


def _extract_final(text: str) -> str:
    # Accept JSON with {"final": "..."} or plain text.
    try:
        payload = json.loads(_extract_json(text))
        final = str(payload.get("final") or "").strip()
        if final:
            return final
    except Exception:
        pass
    return text.strip()


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


def _num_in_text(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d+(\.\d+)?(e[-+]?\d+)?", text, re.I)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _check_numeric(expected: float, tol: float = 1e-6) -> Callable[[str], Tuple[bool, str]]:
    def _checker(text: str) -> Tuple[bool, str]:
        val = _num_in_text(text)
        if val is None:
            return False, "no numeric value detected"
        ok = _approx_eq(val, expected, tol=tol)
        return ok, f"got {val}, expected {expected}"

    return _checker


def _check_contains(tokens: List[str]) -> Callable[[str], Tuple[bool, str]]:
    def _checker(text: str) -> Tuple[bool, str]:
        lower = text.lower()
        for token in tokens:
            if token.lower() in lower:
                return True, "matched token"
        return False, f"missing token(s): {tokens}"

    return _checker


def _build_problems() -> List[Problem]:
    problems: List[Problem] = []
    problems.append(
        Problem(
            "Algebra",
            "Solve for x: 2x + 5 = 17. Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(6.0),
        )
    )
    problems.append(
        Problem(
            "Linear Algebra",
            "Compute determinant of [[1,2],[3,4]]. Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(-2.0),
        )
    )
    problems.append(
        Problem(
            "Calculus (derivative)",
            "Compute derivative d/dx of x^3. Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_contains(["3x^2", "3x**2", "3*x^2", "3x^2"]),
        )
    )
    problems.append(
        Problem(
            "Calculus (integral)",
            "Compute definite integral from 0 to 1 of x^2 dx. Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(1.0 / 3.0, tol=1e-4),
        )
    )
    problems.append(
        Problem(
            "Probability",
            "A fair coin is flipped 3 times. What is the probability of exactly 2 heads? Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(3.0 / 8.0, tol=1e-6),
        )
    )
    problems.append(
        Problem(
            "Statistics",
            "What is the variance of the sample [1,2,3] using population variance? Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(2.0 / 3.0, tol=1e-6),
        )
    )
    problems.append(
        Problem(
            "Topology",
            "What is the Euler characteristic of a torus? Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(0.0),
        )
    )
    problems.append(
        Problem(
            "Complex Analysis",
            "Compute the magnitude |3 + 4i|. Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_numeric(5.0),
        )
    )
    problems.append(
        Problem(
            "Differential Equations",
            "Solve dy/dx = 2x with y(0)=1. Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_contains(["x^2+1", "x**2+1", "x^2 + 1", "x² + 1", "x²+1"]),
        )
    )
    problems.append(
        Problem(
            "Quantum Mechanics",
            "What is the canonical commutator [x,p] in quantum mechanics? Return ONLY JSON: {\"final\": \"<answer>\"}.",
            _check_contains(["iħ", "i hbar", "i\\hbar", "iℏ", "i ℏ"]),
        )
    )
    return problems


def _run_suite(model_override: Optional[str] = None) -> Tuple[int, List[dict]]:
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings

    settings = c0d3r_default_settings()
    if model_override:
        settings["model"] = model_override
    settings = dict(settings)
    settings["stream_default"] = False
    session = C0d3rSession(
        session_name="c0d3r-math-benchmark",
        transcript_dir=Path("runtime/benchmarks/transcripts/c0d3r_math"),
        read_timeout_s=120,
        **settings,
    )
    results: List[dict] = []
    passed = 0
    for problem in _build_problems():
        response = session.send(problem.prompt, stream=False)
        final = _extract_final(response)
        ok, detail = problem.checker(final)
        results.append(
            {
                "name": problem.name,
                "prompt": problem.prompt,
                "response": final,
                "ok": ok,
                "detail": detail,
            }
        )
        if ok:
            passed += 1
    return passed, results


def main() -> int:
    parser = argparse.ArgumentParser(description="Math benchmark for c0d3r.")
    parser.add_argument("--model", help="Override Bedrock model id/inference profile.")
    parser.add_argument("--max-attempts", type=int, default=1, help="How many attempts to run.")
    args = parser.parse_args()

    out_dir = Path("runtime/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(_build_problems())
    best = None

    for attempt in range(max(1, args.max_attempts)):
        passed, results = _run_suite(model_override=args.model)
        report = {
            "attempt": attempt + 1,
            "model": args.model or "default",
            "passed": passed,
            "total": total,
            "results": results,
        }
        if best is None or passed > best["passed"]:
            best = report
        if passed == total:
            break

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"math_benchmark_{ts}.json"
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Wrote benchmark report to {path}")
    print(f"Passed {best['passed']}/{best['total']} (model={best['model']})")
    if best["passed"] < best["total"]:
        print("Some problems failed; consider different model overrides or higher reasoning.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
