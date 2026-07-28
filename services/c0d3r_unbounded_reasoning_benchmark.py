"""Differential benchmark for C0d3rV2's unbounded-reasoning tools.

The oracle is deliberately kept in this harness and is never included in a
tool prompt.  A runner receives only ``public_case()``.  This prevents the
benchmark from rewarding an implementation that merely repeats the expected
answer supplied as calibration data.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

Mode = Literal["scientific_method", "equation_matrix", "combined"]
MODES: tuple[Mode, ...] = ("scientific_method", "equation_matrix", "combined")


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    difficulty: int
    prompt: str
    domain: str
    answer_kind: Literal["numeric", "choice"]
    expected: float | str
    tolerance: float = 0.0
    units: tuple[str, ...] = ()
    required_concepts: tuple[str, ...] = ()
    falsification_terms: tuple[str, ...] = ("fals", "reject", "would fail", "contradict")

    def public_case(self) -> dict[str, Any]:
        """Return inputs safe to give the system under test (no oracle)."""
        return {"id": self.id, "difficulty": self.difficulty, "prompt": self.prompt, "domain": self.domain}


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("monty_hall", 1,
        "In the standard three-door Monty Hall game, should a contestant switch, and what is the win probability?",
        "probability", "numeric", 2 / 3, .015, ("probability", "%"), ("switch", "host")),
    BenchmarkCase("free_fall", 1,
        "Ignoring air resistance, how far does an object initially at rest fall near Earth's surface in 3.0 seconds?",
        "mechanics", "numeric", 44.145, .25, ("m", "meter", "metre"), ("gravity", "time")),
    BenchmarkCase("photon_energy", 2,
        "What is the energy in electron-volts of a 500 nm photon?",
        "quantum physics", "numeric", 2.47968, .015, ("ev", "electron-volt"), ("planck", "wavelength", "light")),
    BenchmarkCase("mixing_temperature", 2,
        "An insulated vessel mixes 1 kg of water at 20 C with 2 kg at 80 C. Neglect the vessel and losses. Find equilibrium temperature.",
        "thermodynamics", "numeric", 60.0, .15, ("c", "°c", "celsius"), ("energy", "heat", "mass")),
    BenchmarkCase("orbital_period", 3,
        "Assuming a circular orbit, estimate the orbital period in minutes of a satellite 400 km above Earth. State constants and assumptions.",
        "orbital mechanics", "numeric", 92.56, .8, ("min", "minute"), ("gravity", "radius", "earth")),
    BenchmarkCase("diffusion_time", 3,
        "Using one-dimensional diffusion with D = 1.0e-9 m^2/s, estimate the characteristic time to diffuse 100 micrometers using <x^2>=2Dt.",
        "transport physics", "numeric", 5.0, .05, ("s", "second"), ("diffusion", "distance", "squared")),
    BenchmarkCase("rc_cutoff", 4,
        "Design the cutoff calculation for a first-order RC low-pass with R=3.3 kohm and C=47 nF. What is the -3 dB frequency?",
        "electrical engineering", "numeric", 1026.1, 8.0, ("hz", "hertz"), ("resistance", "capacitance", "2π")),
    BenchmarkCase("radio_link", 5,
        "At 2.4 GHz over 10 km free space, calculate path loss in dB using FSPL=20log10(d_km)+20log10(f_MHz)+32.44.",
        "radio engineering", "numeric", 120.004, .08, ("db", "decibel"), ("frequency", "distance", "log")),
)


@dataclass
class CaseScore:
    case_id: str
    mode: Mode
    difficulty: int
    score: float
    answer_score: float
    units_score: float
    reasoning_score: float
    provenance_score: float
    falsification_score: float
    extracted_answer: float | str | None
    elapsed_seconds: float
    errors: list[str] = field(default_factory=list)


def _flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _numbers(text: str) -> list[float]:
    vals: list[float] = []
    for raw in re.findall(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            vals.append(float(raw))
        except ValueError:
            pass
    for numerator, denominator in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])", text):
        if float(denominator):
            vals.append(float(numerator) / float(denominator))
    for percent in re.findall(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%", text):
        vals.append(float(percent) / 100.0)
    return vals


def _answer_score(case: BenchmarkCase, text: str) -> tuple[float, float | str | None]:
    low = text.lower()
    if case.answer_kind == "choice":
        ok = str(case.expected).lower() in low
        return (1.0 if ok else 0.0), (str(case.expected) if ok else None)
    candidates = _numbers(text)
    if not candidates:
        return 0.0, None
    expected = float(case.expected)
    closest = min(candidates, key=lambda n: abs(n - expected))
    err = abs(closest - expected)
    if err <= case.tolerance:
        return 1.0, closest
    # Partial credit decays smoothly, but distant constants in prose do not pass.
    scale = max(case.tolerance, abs(expected) * .01, 1e-12)
    return max(0.0, math.exp(-err / scale) * .7), closest


def score_result(case: BenchmarkCase, mode: Mode, result: Any, elapsed: float = 0.0) -> CaseScore:
    text = _flatten(result)
    low = text.lower()
    answer_text = text
    if isinstance(result, dict) and mode == "scientific_method":
        conclusion = result.get("conclusion") if isinstance(result.get("conclusion"), dict) else {}
        answer_text = str(conclusion.get("answer") or "") if conclusion.get("status") == "supported" else ""
    elif isinstance(result, dict) and mode == "combined":
        synthesis = result.get("synthesis") if isinstance(result.get("synthesis"), dict) else result
        answer_text = str(synthesis.get("answer") or "") if synthesis.get("answered", True) else ""
    elif isinstance(result, dict) and mode == "equation_matrix" and "solutions" in result:
        answer_text = _flatten(result.get("solutions") or [])
    answer, extracted = _answer_score(case, answer_text)
    units = 1.0 if not case.units or any(u.lower() in low for u in case.units) else 0.0
    concepts = sum(c.lower() in low for c in case.required_concepts) / max(1, len(case.required_concepts))
    urls = re.findall(r"https?://[^\s\"'<>]+", text)
    provenance = 1.0 if urls else (0.5 if any(k in low for k in ("source", "provenance", "doi", "nist", "nasa")) else 0.0)
    falsification = 1.0 if any(k in low for k in case.falsification_terms) else 0.0
    total = 100 * (.55 * answer + .10 * units + .15 * concepts + .10 * provenance + .10 * falsification)
    errors: list[str] = []
    if answer < 1: errors.append("answer outside tolerance or absent")
    if units < 1: errors.append("units absent")
    if provenance < 1: errors.append("retrievable provenance absent")
    if falsification < 1: errors.append("falsification criterion absent")
    return CaseScore(case.id, mode, case.difficulty, round(total, 3), answer, units,
                     round(concepts, 4), provenance, falsification, extracted,
                     round(elapsed, 4), errors)


Runner = Callable[[Mode, dict[str, Any]], Any]


def run_benchmark(runner: Runner, *, cases: Iterable[BenchmarkCase] = CASES,
                  output: str | Path | None = None) -> dict[str, Any]:
    scores: list[CaseScore] = []
    for case in cases:
        for mode in MODES:
            started = time.perf_counter()
            try:
                result = runner(mode, case.public_case())
                score = score_result(case, mode, result, time.perf_counter() - started)
            except Exception as exc:
                score = CaseScore(case.id, mode, case.difficulty, 0, 0, 0, 0, 0, 0,
                                  None, round(time.perf_counter() - started, 4), [repr(exc)])
            scores.append(score)
    by_mode = {}
    for mode in MODES:
        rows = [s for s in scores if s.mode == mode]
        by_mode[mode] = {
            "average": round(sum(r.score for r in rows) / max(1, len(rows)), 3),
            "passed": sum(r.score >= 90 for r in rows), "total": len(rows),
            "mean_seconds": round(sum(r.elapsed_seconds for r in rows) / max(1, len(rows)), 3),
        }
    report = {
        "schema_version": 1, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oracle_disclosure": "Expected answers were used only by the scorer, never passed to the runner.",
        "pass_threshold": 90, "by_mode": by_mode, "scores": [asdict(s) for s in scores],
    }
    if output:
        target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class ToolRegistryRunner:
    """Adapter for a configured C0d3rV2 ToolRegistry.

    Combined mode intentionally passes scientific evidence into the solver;
    independent modes do not leak results into one another.
    """
    def __init__(self, tools: Any):
        self.tools = tools

    def __call__(self, mode: Mode, case: dict[str, Any]) -> Any:
        prompt = case["prompt"]
        if mode == "scientific_method":
            return self.tools.dispatch("scientific_method", {"question": prompt, "domain": case["domain"]})
        if mode == "equation_matrix":
            # A search hit is not a solution.  Exercise the matrix-backed
            # natural-language -> equations -> independently solved values
            # path so this mode is scored on a checkable conclusion.
            return self.tools.dispatch("math_grounding", {"prompt": prompt})
        evidence = self.tools.dispatch("scientific_method", {"question": prompt, "domain": case["domain"]})
        synthesis = self.tools.dispatch("unbounded_solver", {
            "prompt": prompt,
            "ai_response": "Independent archival/scientific pass:\n" + _flatten(evidence),
        })
        # Preserve the audit trail used to reach the synthesis.  Returning only
        # the short final answer makes a correct result look unproven and also
        # prevents callers from inspecting or falsifying it.
        return {"scientific_evidence": evidence, "synthesis": synthesis}
