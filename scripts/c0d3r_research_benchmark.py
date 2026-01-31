#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ResearchQuestion:
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


def _contains_all(tokens: List[str]) -> Callable[[str], Tuple[bool, str]]:
    def _checker(text: str) -> Tuple[bool, str]:
        lower = text.lower()
        missing = [t for t in tokens if t.lower() not in lower]
        if missing:
            return False, f"missing tokens: {missing}"
        return True, "ok"

    return _checker


def _check_numeric(expected: float, tol: float = 1e-3) -> Callable[[str], Tuple[bool, str]]:
    def _checker(text: str) -> Tuple[bool, str]:
        import re

        match = re.search(r"[-+]?\d+(\.\d+)?", text)
        if not match:
            return False, "no numeric value detected"
        val = float(match.group(0))
        ok = abs(val - expected) <= tol
        return ok, f"got {val}, expected {expected}"

    return _checker


def _build_questions() -> List[ResearchQuestion]:
    return [
        ResearchQuestion(
            "bio_crispr",
            "Give one recent PubMed paper title about CRISPR gene editing and mention PubMed or NCBI as source.",
            _contains_all(["pubmed", "crispr"]),
        ),
        ResearchQuestion(
            "chem_pubchem",
            "What is caffeine's molecular formula? Cite PubChem in the answer.",
            _contains_all(["c8h10n4o2", "pubchem"]),
        ),
        ResearchQuestion(
            "physics_pdg",
            "What is the approximate mass of the electron in MeV/c^2? Cite PDG or NIST in the answer.",
            _check_numeric(0.511, tol=0.002),
        ),
        ResearchQuestion(
            "earth_noaa",
            "Which US agency provides a no-key weather API? Mention NOAA or api.weather.gov.",
            _contains_all(["noaa"]),
        ),
        ResearchQuestion(
            "math_dlmf",
            "Where can you find authoritative references for special functions? Mention NIST DLMF.",
            _contains_all(["dlmf", "nist"]),
        ),
        ResearchQuestion(
            "engineering_rfc",
            "Where are the authoritative internet protocol specifications published? Mention RFC Editor or IETF.",
            _contains_all(["rfc"]),
        ),
    ]


def _build_prompt(questions: List[ResearchQuestion]) -> str:
    lines = [
        "Use research sources. Answer all questions precisely and cite the source name in each answer.",
        "Return ONLY JSON:",
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
        session_name="c0d3r-research-benchmark",
        transcript_dir=Path("runtime/benchmarks/transcripts/c0d3r_research"),
        read_timeout_s=180,
        **settings,
    )
    questions = _build_questions()
    prompt = _build_prompt(questions)
    system = (
        "Use the provided research context. Do not claim sources are unavailable if research is present. "
        "Cite the source name in each answer."
    )
    response = session.send(prompt, system=system, stream=False)
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
    parser = argparse.ArgumentParser(description="Research benchmark for c0d3r.")
    parser.add_argument("--model", help="Override model or inference profile.")
    args = parser.parse_args()

    from tools.c0d3r_session import c0d3r_default_settings

    settings = dict(c0d3r_default_settings())
    settings["stream_default"] = False
    settings["reasoning_effort"] = "extra_high"
    settings["research"] = True
    settings["multi_model"] = False
    settings["model"] = "us.anthropic.claude-opus-4-1-20250805-v1:0"
    if args.model:
        settings["model"] = args.model

    total = len(_build_questions())
    passed, results, raw = _run_once(settings)
    report = {
        "model": settings.get("model") or "default",
        "passed": passed,
        "total": total,
        "results": results,
        "raw": raw[:4000],
    }

    out_dir = Path("runtime/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"research_benchmark_{ts}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote benchmark report to {path}")
    print(f"Passed {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
