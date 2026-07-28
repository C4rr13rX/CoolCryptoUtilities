from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.c0d3rV2.delivery_runner import run_delivery_turn_detailed


RUNTIME_ROOT = Path("runtime/atf_class_refinement")
WORKSPACE_ROOT = RUNTIME_ROOT / "workspace"
RESULTS_PATH = RUNTIME_ROOT / "results.jsonl"
GUIDE_PATH = RUNTIME_ROOT / "class_generation_prompt_guide.md"


def _now() -> float:
    return time.time()


@dataclass(frozen=True)
class ClassTask:
    task_id: str
    title: str
    module_name: str
    class_name: str
    description: str
    contract: Dict[str, Any]
    tests: str


def _task_templates() -> List[ClassTask]:
    return [
        ClassTask(
            task_id="dog_ethology_001",
            title="Dog ethology state model",
            module_name="solution",
            class_name="Dog",
            description=(
                "Represent a domestic dog as a small behavioral state machine. "
                "Use clear OOP encapsulation. Track name, breed, age_years, energy, hunger, and mood. "
                "Energy and hunger are normalized floats in [0, 1]. Methods must clamp values. "
                "Feeding lowers hunger and can improve mood. Playing consumes energy and increases hunger. "
                "Resting restores energy. The class should expose a serializable state snapshot."
            ),
            contract={
                "constructor": "Dog(name: str, breed: str, age_years: float, energy: float = 0.6, hunger: float = 0.4)",
                "methods": {
                    "feed(amount: float) -> dict": "decrease hunger by amount, clamp, update mood",
                    "play(minutes: float) -> dict": "decrease energy and increase hunger based on minutes",
                    "rest(hours: float) -> dict": "increase energy based on hours",
                    "state() -> dict": "return name, breed, age_years, energy, hunger, mood",
                },
                "invariants": ["0 <= energy <= 1", "0 <= hunger <= 1", "mood is one of tired, hungry, calm, happy"],
            },
            tests=r'''
from solution import Dog
d = Dog("Ada", "collie", 3, energy=0.5, hunger=0.6)
assert d.state()["name"] == "Ada"
d.feed(0.25)
assert 0 <= d.state()["hunger"] <= 0.35
d.play(30)
assert 0 <= d.state()["energy"] < 0.5
assert 0 <= d.state()["hunger"] <= 1
d.rest(2)
assert d.state()["energy"] > 0.2
assert d.state()["mood"] in {"tired", "hungry", "calm", "happy"}
''',
        ),
        ClassTask(
            task_id="bicycle_drivetrain_001",
            title="Bicycle drivetrain physics-lite model",
            module_name="solution",
            class_name="Bicycle",
            description=(
                "Model a bicycle drivetrain with wheel radius, front chainring teeth, rear sprocket teeth, "
                "cadence rpm, and rider mass. Compute gear ratio, wheel angular velocity, linear speed, "
                "and approximate kinetic energy. Use SI units internally and validate positive dimensions."
            ),
            contract={
                "constructor": "Bicycle(wheel_radius_m: float, chainring_teeth: int, sprocket_teeth: int, rider_mass_kg: float)",
                "methods": {
                    "gear_ratio() -> float": "chainring_teeth / sprocket_teeth",
                    "speed_mps(cadence_rpm: float) -> float": "cadence revolutions through gear ratio into wheel circumference per second",
                    "kinetic_energy_j(cadence_rpm: float) -> float": "0.5 * rider_mass_kg * speed^2",
                    "shift(chainring_teeth: int | None = None, sprocket_teeth: int | None = None) -> None": "update gears with validation",
                },
                "invariants": ["all dimensions/teeth/mass positive", "speed at zero cadence is zero"],
            },
            tests=r'''
from solution import Bicycle
b = Bicycle(0.34, 50, 25, 80)
assert abs(b.gear_ratio() - 2.0) < 1e-9
assert b.speed_mps(0) == 0
s = b.speed_mps(90)
assert 6.0 < s < 7.0
assert b.kinetic_energy_j(90) > 1400
b.shift(sprocket_teeth=20)
assert abs(b.gear_ratio() - 2.5) < 1e-9
try:
    Bicycle(0, 50, 25, 80)
    raise AssertionError("expected validation")
except ValueError:
    pass
''',
        ),
        ClassTask(
            task_id="avian_development_001",
            title="Bird development lifecycle model",
            module_name="solution",
            class_name="BirdDevelopment",
            description=(
                "Represent the life-stage development of an altricial bird from egg to adult. "
                "Track species, age_days, mass_g, and stage. Stage thresholds: egg <14, hatchling <28, "
                "nestling <45, fledgling <90, juvenile <180, adult otherwise. Growth should be monotonic "
                "for positive food intake and stage should update as age advances."
            ),
            contract={
                "constructor": "BirdDevelopment(species: str, age_days: int = 0, mass_g: float = 1.0)",
                "methods": {
                    "advance(days: int, food_factor: float = 1.0) -> dict": "increase age and mass, update stage",
                    "stage_for_age(age_days: int) -> str": "pure stage classifier",
                    "state() -> dict": "return species, age_days, mass_g, stage",
                },
                "invariants": ["age_days non-negative", "mass_g positive", "stage matches thresholds"],
            },
            tests=r'''
from solution import BirdDevelopment
b = BirdDevelopment("sparrow")
assert b.stage_for_age(0) == "egg"
assert b.stage_for_age(14) == "hatchling"
b.advance(30, food_factor=1.2)
st = b.state()
assert st["age_days"] == 30
assert st["stage"] == "nestling"
assert st["mass_g"] > 1.0
b.advance(200)
assert b.state()["stage"] == "adult"
''',
        ),
        ClassTask(
            task_id="thermal_reservoir_001",
            title="Thermodynamic reservoir class",
            module_name="solution",
            class_name="ThermalReservoir",
            description=(
                "Model a lumped thermal reservoir with mass, specific heat capacity, and temperature. "
                "It must absorb/release heat in joules, report internal energy relative to 0 K, and mix "
                "with another reservoir by energy conservation. Use Kelvin internally and reject negative temperatures."
            ),
            contract={
                "constructor": "ThermalReservoir(mass_kg: float, specific_heat_j_per_kg_k: float, temperature_k: float)",
                "methods": {
                    "heat_capacity() -> float": "mass * specific heat",
                    "add_heat(joules: float) -> float": "update and return temperature K",
                    "energy_j() -> float": "heat capacity * temperature_k",
                    "mix(other: ThermalReservoir) -> ThermalReservoir": "return new reservoir with conserved energy",
                },
                "invariants": ["positive mass and heat capacity", "temperature_k >= 0", "mix conserves energy"],
            },
            tests=r'''
from solution import ThermalReservoir
a = ThermalReservoir(2, 1000, 300)
assert a.heat_capacity() == 2000
assert abs(a.add_heat(2000) - 301) < 1e-9
b = ThermalReservoir(1, 1000, 290)
c = a.mix(b)
assert abs(c.energy_j() - (a.energy_j() + b.energy_j())) < 1e-6
assert 295 < c.temperature_k < 301
try:
    ThermalReservoir(1, 1, -1)
    raise AssertionError("expected validation")
except ValueError:
    pass
''',
        ),
    ]


def generate_tasks(count: int) -> List[ClassTask]:
    base = _task_templates()
    tasks: List[ClassTask] = []
    for i in range(max(1, int(count))):
        template = base[i % len(base)]
        if i < len(base):
            tasks.append(template)
        else:
            suffix = f"v{i + 1:03d}"
            tasks.append(
                ClassTask(
                    task_id=f"{template.task_id}_{suffix}",
                    title=f"{template.title} ({suffix})",
                    module_name=template.module_name,
                    class_name=template.class_name,
                    description=template.description + f" Variant {suffix}: preserve the same public contract exactly.",
                    contract=template.contract,
                    tests=template.tests,
                )
            )
    return tasks


def _write_task_workspace(task: ClassTask, *, clean: bool = True) -> Path:
    root = WORKSPACE_ROOT / task.task_id
    root.mkdir(parents=True, exist_ok=True)
    if clean:
        (root / "solution.py").unlink(missing_ok=True)
        shutil.rmtree(root / ".c0d3r", ignore_errors=True)
    (root / "contract.json").write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "title": task.title,
                "class_name": task.class_name,
                "description": task.description,
                "contract": task.contract,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "test_solution.py").write_text(task.tests.strip() + "\nprint('PASS')\n", encoding="utf-8")
    return root


def _reference_guidance() -> str:
    return """# ATF/C0D3R class generation guide

When the requested artifact is a class, use this path:

1. Convert the prompt into an explicit public contract: constructor, fields, methods, return shapes, invariants, validation errors, and units.
2. Generate one class at a time unless the contract explicitly requires collaborators.
3. Keep state private or consistently owned by the instance.
4. Clamp/validate numeric domains before calculations.
5. Return dictionaries only where the contract asks for serializable state.
6. Run the provided tests before claiming success.
7. If tests fail, repair the minimal behavior that failed instead of rewriting randomly.
"""


def update_prompt_guide(result: Dict[str, Any]) -> None:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    current = GUIDE_PATH.read_text(encoding="utf-8") if GUIDE_PATH.exists() else _reference_guidance()
    failures = result.get("failures") or []
    if failures:
        additions = ["\n## Recent failure-derived refinements\n"]
        for failure in failures[-10:]:
            additions.append(
                f"- `{failure.get('task_id')}`: {failure.get('error', '')[:240]} "
                "=> make constructor/method signatures exact and run behavioral tests.\n"
            )
        current = current.rstrip() + "\n" + "".join(additions)
    GUIDE_PATH.write_text(current, encoding="utf-8")


def run_task(task: ClassTask, *, attempts: int = 2) -> Dict[str, Any]:
    started_at = time.perf_counter()
    root = _write_task_workspace(task)
    guide = GUIDE_PATH.read_text(encoding="utf-8") if GUIDE_PATH.exists() else _reference_guidance()
    last_detail: Dict[str, Any] = {}
    last_error = ""
    synthesis_path = "c0d3rv2_delivery"
    for attempt in range(1, max(1, attempts) + 1):
        prompt = f"""Create the Python class described by contract.json.

Requirements:
- Write the implementation to solution.py.
- The public class must be named {task.class_name}.
- Do not change test_solution.py.
- Run: {sys.executable} test_solution.py
- If the test fails, fix solution.py and rerun once before final response.

Class description:
{task.description}

Contract:
{json.dumps(task.contract, indent=2)}
"""
        try:
            detail = run_delivery_turn_detailed(
                prompt,
                session_key=f"atf-class-refinement:{task.task_id}:{attempt}",
                workdir=root,
                backend="freeloader",
                system_context=guide,
                reset=True,
            )
            last_detail = detail
        except Exception as exc:
            last_error = str(exc)
            last_detail = {"error": last_error}

        solution_path = root / "solution.py"
        if not solution_path.exists() or not solution_path.read_text(encoding="utf-8", errors="ignore").strip():
            direct = _direct_atf_class_synthesis(task, root, guide)
            if direct.get("wrote_file"):
                synthesis_path = "c0d3rv2_atf_strict_class_synthesis"
                last_detail = {**last_detail, "strict_synthesis": direct}

        test = subprocess.run(
            [sys.executable, "test_solution.py"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if test.returncode == 0:
            return {
                "task_id": task.task_id,
                "title": task.title,
                "status": "passed",
                "attempt": attempt,
                "synthesis_path": synthesis_path,
                "stdout": test.stdout[-2000:],
                "stderr": test.stderr[-2000:],
                "detail": _compact_detail(last_detail),
                "duration_sec": round(time.perf_counter() - started_at, 3),
            }
        last_error = (test.stderr or test.stdout or "test failed")[-3000:]
    return {
        "task_id": task.task_id,
        "title": task.title,
        "status": "failed",
        "attempt": attempts,
        "synthesis_path": synthesis_path,
        "error": last_error,
        "detail": _compact_detail(last_detail),
        "duration_sec": round(time.perf_counter() - started_at, 3),
    }


def _direct_atf_class_synthesis(task: ClassTask, root: Path, guide: str) -> Dict[str, Any]:
    """
    Strict single-file synthesis fallback used only when the broader C0D3R
    delivery orchestration fails to create the class artifact at all.

    This still uses AgentTheFreeloader as the model source, but it narrows the
    problem to the smallest deterministic class-generation unit so the result
    can be tested and fed back into C0D3R's prompt guide.
    """
    try:
        from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession
    except Exception as exc:
        return {"wrote_file": False, "error": f"import_failed:{exc}"}
    session = AgentTheFreeloaderSession(
        session_name=f"atf-strict-class-{task.task_id}",
        transcript_dir=RUNTIME_ROOT / "transcripts",
        workdir=root,
        timeout_s=float(os.getenv("ATF_CLASS_DIRECT_TIMEOUT_S", "90")),
        max_attempts=max(1, int(os.getenv("ATF_CLASS_DIRECT_ATTEMPTS", "3"))),
        max_tokens=int(os.getenv("ATF_CLASS_DIRECT_MAX_TOKENS", "2048")),
    )
    prompt = f"""Return ONLY Python source code for solution.py. No markdown. No explanation.

The file must define class {task.class_name}.

Description:
{task.description}

Contract:
{json.dumps(task.contract, indent=2)}

Tests that must pass:
{task.tests}
"""
    try:
        reply = session.send(prompt, system=guide, temperature=0.1, max_tokens=2048)
    except Exception as exc:
        return {
            "wrote_file": False,
            "error": str(exc),
            "route_history": getattr(session, "route_history", []),
        }
    code = _extract_python_code(reply)
    if not code.strip() or f"class {task.class_name}" not in code:
        return {
            "wrote_file": False,
            "error": "model_returned_no_matching_class",
            "reply_tail": reply[-1000:],
            "route_history": getattr(session, "route_history", []),
        }
    (root / "solution.py").write_text(code.rstrip() + "\n", encoding="utf-8")
    return {
        "wrote_file": True,
        "route_history": getattr(session, "route_history", []),
        "model": session.get_model_id(),
    }


def _extract_python_code(text: str) -> str:
    raw = text or ""
    if "```" not in raw:
        return raw.strip()
    parts = raw.split("```")
    for part in parts:
        cleaned = part.strip()
        if cleaned.lower().startswith("python"):
            return cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
    for part in parts:
        if "class " in part:
            return part.strip()
    return raw.strip()


def _compact_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "models": detail.get("models", []),
        "turn_model_calls": detail.get("turn_model_calls"),
        "session_error": detail.get("session_error", ""),
        "error": detail.get("error", ""),
        "tool_events": detail.get("tool_events", [])[-20:] if isinstance(detail.get("tool_events"), list) else [],
    }


def run_class_refinement_benchmark(*, count: int = 4, attempts: int = 2) -> Dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if not GUIDE_PATH.exists():
        GUIDE_PATH.write_text(_reference_guidance(), encoding="utf-8")
    tasks = generate_tasks(count)
    results = []
    for task in tasks:
        result = run_task(task, attempts=attempts)
        results.append(result)
        with RESULTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": _now(), **result}, ensure_ascii=True) + "\n")
    failures = [r for r in results if r.get("status") != "passed"]
    summary = {
        "count": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "pass_rate": (len(results) - len(failures)) / max(1, len(results)),
        "failures": failures,
        "results_path": str(RESULTS_PATH),
        "guide_path": str(GUIDE_PATH),
    }
    update_prompt_guide(summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ATF/C0D3R class-generation refinement benchmark.")
    parser.add_argument("--count", type=int, default=int(os.getenv("ATF_CLASS_BENCH_COUNT", "4")))
    parser.add_argument("--attempts", type=int, default=int(os.getenv("ATF_CLASS_BENCH_ATTEMPTS", "2")))
    args = parser.parse_args()
    print(json.dumps(run_class_refinement_benchmark(count=args.count, attempts=args.attempts), indent=2))
