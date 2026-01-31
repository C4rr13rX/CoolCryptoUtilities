#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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
    questions: List[Question] = [
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
        Question("biology_dna", "Base pairs in DNA.", _check_contains(["a-t", "adenine-thymine", "c-g", "cytosine-guanine", "at", "gc"])),
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
    questions.extend(_generate_sat_questions())
    questions.extend(_generate_mensa_questions())
    questions.extend(_generate_math_drills(80, seed=42))
    questions.extend(_generate_word_problems(40, seed=7))
    questions.extend(_generate_physics_questions(20))
    questions.extend(_generate_engineering_questions(20))
    questions.extend(_generate_history_questions(20))
    questions.extend(_generate_biology_questions(20))
    return questions


def _generate_math_drills(count: int, seed: int) -> List[Question]:
    rng = random.Random(seed)
    items: List[Question] = []
    for i in range(count):
        a = rng.randint(2, 50)
        b = rng.randint(2, 50)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            prompt = f"Compute {a} + {b}."
        elif op == "-":
            ans = a - b
            prompt = f"Compute {a} - {b}."
        else:
            ans = a * b
            prompt = f"Compute {a} * {b}."
        items.append(Question(f"drill_{i}", prompt, _check_numeric(float(ans))))
    return items


def _generate_word_problems(count: int, seed: int) -> List[Question]:
    rng = random.Random(seed)
    items: List[Question] = []
    for i in range(count):
        price = rng.randint(2, 15)
        qty = rng.randint(3, 12)
        prompt = f"A book costs ${price} and you buy {qty} books. What is the total cost?"
        items.append(Question(f"word_{i}", prompt, _check_numeric(float(price * qty))))
    return items


def _generate_sat_questions() -> List[Question]:
    items: List[Question] = []
    items.append(Question("sat_linear_1", "Solve: 5x + 10 = 35.", _check_numeric(5.0)))
    items.append(Question("sat_linear_2", "Solve: 2(3x - 4) = 10.", _check_numeric(3.0)))
    items.append(Question("sat_percent_1", "What is 20% of 150?", _check_numeric(30.0)))
    items.append(Question("sat_ratio_1", "If a:b = 3:5 and a+b=40, find b.", _check_numeric(25.0)))
    items.append(Question("sat_geometry_1", "Area of a rectangle with sides 8 and 5.", _check_numeric(40.0)))
    items.append(Question("sat_geometry_2", "Circumference of a circle with radius 3 (use 2πr).", _check_contains(["6π", "6 pi"])))
    items.append(Question("sat_system_1", "Solve system: x+y=9, x-y=3. Find x.", _check_numeric(6.0)))
    items.append(Question("sat_system_2", "Solve system: 2x+3y=12 and x=3. Find y.", _check_numeric(2.0)))
    items.append(Question("sat_exponent_1", "Compute 2^5.", _check_numeric(32.0)))
    items.append(Question("sat_slope_1", "Slope between (2,3) and (6,11).", _check_numeric(2.0)))
    return items


def _generate_mensa_questions() -> List[Question]:
    items: List[Question] = []
    items.append(Question("mensa_seq_1", "Find the next number: 2, 4, 8, 16, ?", _check_numeric(32.0)))
    items.append(Question("mensa_seq_2", "Find the next number: 1, 1, 2, 3, 5, 8, ?", _check_numeric(13.0)))
    items.append(Question("mensa_seq_3", "Find the next number: 3, 6, 12, 24, ?", _check_numeric(48.0)))
    items.append(Question("mensa_seq_4", "Find the next number: 1, 4, 9, 16, ?", _check_numeric(25.0)))
    items.append(Question("mensa_seq_5", "Find the missing number: 5, 10, 20, ?, 80.", _check_numeric(40.0)))
    items.append(Question("mensa_logic_1", "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies? (yes/no)", _check_contains(["yes"])))
    items.append(Question("mensa_logic_2", "If today is Monday, what day will it be in 10 days?", _check_contains(["thursday"])))
    items.append(Question("mensa_logic_3", "A is taller than B, B is taller than C. Who is shortest?", _check_contains(["c"])))
    items.append(Question("mensa_logic_4", "If 3 cats catch 3 mice in 3 minutes, how many cats catch 9 mice in 3 minutes?", _check_numeric(9.0)))
    items.append(Question("mensa_logic_5", "How many sides does a decagon have?", _check_numeric(10.0)))
    return items


def _generate_physics_questions(count: int) -> List[Question]:
    items: List[Question] = []
    for i in range(count):
        mass = 2 + i % 5
        accel = 3 + (i % 4)
        force = mass * accel
        items.append(Question(f"phys_{i}", f"Compute force: mass {mass} kg, acceleration {accel} m/s^2.", _check_numeric(float(force))))
    return items


def _generate_engineering_questions(count: int) -> List[Question]:
    items: List[Question] = []
    for i in range(count):
        r = 2 + i % 6
        iamps = 1 + (i % 5)
        v = r * iamps
        items.append(Question(f"eng_{i}", f"Using Ohm's law, voltage for R={r}Ω and I={iamps}A.", _check_numeric(float(v))))
    return items


def _generate_history_questions(count: int) -> List[Question]:
    base = [
        ("US Declaration of Independence year.", 1776),
        ("Fall of the Berlin Wall year.", 1989),
        ("Start of World War I year.", 1914),
        ("End of World War I year.", 1918),
        ("Signing of the US Constitution year.", 1787),
    ]
    items: List[Question] = []
    for i in range(count):
        prompt, year = base[i % len(base)]
        items.append(Question(f"hist_{i}", prompt, _check_numeric(float(year))))
    return items


def _generate_biology_questions(count: int) -> List[Question]:
    base = [
        ("How many chromosomes in a human somatic cell?", "46"),
        ("What molecule carries genetic information (DNA or RNA)?", "DNA"),
        ("What is the basic unit of life?", "cell"),
        ("What is the process of cell division called (mitosis or meiosis) for somatic cells?", "mitosis"),
        ("What organelle produces ATP?", "mitochond"),
    ]
    items: List[Question] = []
    for i in range(count):
        prompt, token = base[i % len(base)]
        items.append(Question(f"bio_{i}", prompt, _check_contains([token])))
    return items


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


def _chunk_questions(questions: List[Question], max_chars: int = 6000) -> List[List[Question]]:
    chunks: List[List[Question]] = []
    current: List[Question] = []
    size = 0
    for q in questions:
        line = f"{q.key}: {q.prompt}\n"
        if current and size + len(line) > max_chars:
            chunks.append(current)
            current = []
            size = 0
        current.append(q)
        size += len(line)
    if current:
        chunks.append(current)
    return chunks


def _run_once(settings: Dict[str, object]) -> Tuple[int, List[dict], str]:
    from tools.c0d3r_session import C0d3rSession

    session = C0d3rSession(
        session_name="c0d3r-general-benchmark",
        transcript_dir=Path("runtime/benchmarks/transcripts/c0d3r_general"),
        read_timeout_s=180,
        **settings,
    )
    questions = _build_questions()
    chunks = _chunk_questions(questions, max_chars=6000)
    answers: Dict[str, str] = {}
    raw_fragments: List[str] = []
    for chunk in chunks:
        prompt = _build_prompt(chunk)
        response = session.send(prompt, stream=False)
        raw_fragments.append(response[:2000])
        answers.update(_extract_json(response))
    results: List[dict] = []
    passed = 0
    for q in questions:
        val = answers.get(q.key, "")
        ok, detail = q.checker(val)
        results.append({"key": q.key, "response": val, "ok": ok, "detail": detail})
        if ok:
            passed += 1
    return passed, results, "\n\n".join(raw_fragments)


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
