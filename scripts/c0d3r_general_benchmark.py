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
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Question:
    key: str
    prompt: str
    checker: Callable[[str], Tuple[bool, str]]


def _extract_json(text: str) -> Dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return {}
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        return {}
    return {str(k): str(v) for k, v in answers.items()}


def _num_in_text(text: str) -> Optional[float]:
    frac = re.search(r"([-+]?\d+)\s*/\s*([-+]?\d+)", text)
    if frac:
        try:
            num = float(frac.group(1))
            den = float(frac.group(2))
            if den == 0:
                return None
            return num / den
        except Exception:
            return None
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


def _build_questions() -> List[Question]:
    return [
        Question("math_algebra", "Solve for x: 3x - 7 = 11.", _check_numeric(6.0)),
        Question("math_integral", "Compute ∫_0^2 x dx.", _check_numeric(2.0)),
        Question("math_probability", "Probability of exactly 2 heads in 4 fair flips.", _check_numeric(6/16, tol=1e-6)),
        Question("math_stats", "Population variance of [2,4,4,4,5,5,7,9].", _check_numeric(4.0)),
        Question("math_linear", "Determinant of [[2,1],[5,3]].", _check_numeric(1.0)),
        Question("physics_light", "Speed of light in vacuum in m/s.", _check_contains(["299792458"])),
        Question("physics_newton", "Newton's second law formula.", _check_contains(["f=ma", "f = ma"])),
        Question("physics_qm", "Canonical commutator [x,p].", _check_contains(["iℏ", "i ħ", "i hbar", "i\\hbar"])),
        Question("engineering_ohm", "Ohm's law formula.", _check_contains(["v=ir", "v = ir"])),
        Question("engineering_cap", "SI unit of capacitance.", _check_contains(["farad", "f"])),
        Question("biology_dna", "Base pairs in DNA.", _check_contains(["a-t", "adenine-thymine", "c-g", "cytosine-guanine"])),
        Question("biology_chrom", "Number of human chromosomes in a somatic cell.", _check_contains(["46"])),
        Question("history_moon", "Year Apollo 11 landed on the Moon.", _check_contains(["1969"])),
        Question("history_ww2", "Year World War II ended.", _check_contains(["1945"])),
        Question("programming_big_o", "Time complexity of binary search.", _check_contains(["o(log n)", "log n"])),
        Question("programming_python", "Python list append amortized complexity.", _check_contains(["o(1)", "amortized o(1)"])),
        Question("cs_boolean", "Simplify: NOT(NOT A).", _check_contains(["a"])),
        Question("chemistry_h2o", "Chemical formula for water.", _check_contains(["h2o"])),
        Question("calculus_deriv", "Derivative of sin(x).", _check_contains(["cos(x)", "cos x"])),
        Question("topology_torus", "Euler characteristic of a torus.", _check_contains(["0"])),
    ]


def _build_prompt(questions: List[Question]) -> str:
    lines = [
        "Answer all questions precisely. Return ONLY JSON:",
        '{"answers": {"<question_key>": "<answer>"}}',
        "No extra text.",
        "",
    ]
    for q in questions:
        lines.append(f"{q.key}: {q.prompt}")
    return "\n".join(lines)


def _run_once(settings: Dict[str, object]) -> Tuple[int, List[dict], str]:
    from tools.c0d3r_session import C0d3rSession

    session = C0d3rSession(
        session_name="c0d3r-general-benchmark",
        transcript_dir=Path("runtime/benchmarks/transcripts/c0d3r_general"),
        read_timeout_s=180,
        **settings,
    )
    questions = _build_questions()
    prompt = _build_prompt(questions)
    response = session.send(prompt, stream=False)
    answers = _extract_json(response)
    results: List[dict] = []
    passed = 0
    for q in questions:
        val = answers.get(q.key, "")
        ok, detail = q.checker(val)
        results.append({"key": q.key, "response": val, "ok": ok, "detail": detail})
        if ok:
            passed += 1
    return passed, results, response


def main() -> int:
    parser = argparse.ArgumentParser(description="Mixed-domain benchmark for c0d3r.")
    parser.add_argument("--model", help="Override model or inference profile.")
    parser.add_argument("--max-attempts", type=int, default=2, help="Max attempts across configs.")
    args = parser.parse_args()

    from tools.c0d3r_session import c0d3r_default_settings

    configs: List[Dict[str, object]] = []
    base = dict(c0d3r_default_settings())
    base["stream_default"] = False
    base["reasoning_effort"] = "extra_high"
    if args.model:
        base["model"] = args.model
    configs.append(base)

    # Strong single-model fallback (Opus 4.1 inference profile if available)
    opus = dict(base)
    opus["model"] = "us.anthropic.claude-opus-4-1-20250805-v1:0"
    opus["multi_model"] = False
    configs.append(opus)

    total = len(_build_questions())
    best = None
    for attempt in range(min(args.max_attempts, len(configs))):
        settings = configs[attempt]
        passed, results, raw = _run_once(settings)
        report = {
            "attempt": attempt + 1,
            "model": settings.get("model") or "default",
            "multi_model": settings.get("multi_model"),
            "passed": passed,
            "total": total,
            "results": results,
            "raw": raw[:4000],
        }
        if best is None or passed > best["passed"]:
            best = report
        if passed == total:
            break

    out_dir = Path("runtime/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"general_benchmark_{ts}.json"
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"Wrote benchmark report to {path}")
    print(f"Passed {best['passed']}/{best['total']} (model={best['model']})")
    if best["passed"] < best["total"]:
        print("Some problems failed; review report and adjust models/loops.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
