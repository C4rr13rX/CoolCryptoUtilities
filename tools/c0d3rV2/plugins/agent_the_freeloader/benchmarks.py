from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .feedback import ModelFeedbackStore
from .workday import WorkdayConfig, WorkdayStore, WorkdaySupervisor


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    framework: str
    objective: str
    constraints: tuple[str, ...]
    acceptance: tuple[str, ...]


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        id="django-spectrum-instrument",
        framework="Django 5",
        objective=(
            "Build a digital spectrum-analyzer instrument with a Django GUI. "
            "It must synthesize deterministic sampled signals, compute a dependency-free DFT, "
            "expose JSON acquisition/spectrum endpoints, and render an accessible live dashboard."
        ),
        constraints=(
            "Stay entirely inside the assigned empty work directory.",
            "Separate numerical domain logic from HTTP/views and validate units, sample rate, and Nyquist limits.",
            "No CDN or network dependency at runtime; include migrations only when a database model is necessary.",
            "Use Django's built-in test framework and deterministic fixtures.",
        ),
        acceptance=(
            "manage.py check exits zero.",
            "manage.py test exits zero and tests numerical peaks, invalid input, endpoints, and page rendering.",
            "README documents setup, equations, assumptions, and limitations.",
        ),
    ),
    BenchmarkCase(
        id="dearpygui-impedance-instrument",
        framework="DearPyGui",
        objective=(
            "Build a desktop impedance-analyzer simulator GUI with frequency sweep controls, "
            "Bode magnitude/phase plots, equivalent-series RLC modeling, CSV export, and status telemetry."
        ),
        constraints=(
            "Keep the RLC solver and validation independent of DearPyGui so it is testable headlessly.",
            "Use SI units internally and reject non-positive frequency, capacitance, or inductance.",
            "Do not open a GUI during tests; do not require network access at runtime.",
            "Declare exact compatible dependencies in requirements.txt.",
        ),
        acceptance=(
            "python -m unittest discover -s tests -v exits zero.",
            "Tests cover resonance, complex impedance, sweep ordering, invalid data, and CSV output.",
            "Importing the computational core has no GUI side effects.",
        ),
    ),
    BenchmarkCase(
        id="ionic8-environmental-instrument",
        framework="Ionic 8 + Angular",
        objective=(
            "Build an Ionic 8 Angular scientific environmental telemetry instrument with simulated "
            "temperature, humidity, pressure and air-quality channels, calibration, alarms, trends, and export."
        ),
        constraints=(
            "Use strict TypeScript, standalone Angular components, and Ionic 8-compatible package versions.",
            "Separate deterministic sensor/calibration logic from UI components.",
            "No remote APIs, CDN assets, or hardware are required for tests or build.",
            "Pin the Node engine and provide reproducible npm scripts.",
        ),
        acceptance=(
            "npm install completes and npm run build exits zero.",
            "A non-watch test command validates calibration, alarm boundaries, and deterministic simulation.",
            "README identifies simulated versus measured values and units.",
        ),
    ),
    BenchmarkCase(
        id="qt6-oscilloscope-instrument",
        framework="Qt 6 / C++20",
        objective=(
            "Build a Qt Widgets digital oscilloscope scaffold with deterministic waveform generation, "
            "trigger-level detection, measurements, acquisition controls, and CSV export."
        ),
        constraints=(
            "Use modern C++20 RAII and isolate the numerical acquisition core from Qt GUI classes.",
            "Provide CMake targets for the GUI and headless unit tests without fetching dependencies.",
            "No compiler or Qt SDK is installed on this host; report that limitation and never claim a native build ran.",
            "Keep dimensional units and trigger edge semantics explicit.",
        ),
        acceptance=(
            "Static structure validation finds CMakeLists, Qt Widgets sources, isolated core sources, and tests.",
            "CMake declares C++20, Qt6 Widgets, testing, and at least one headless test target.",
            "README gives exact build/test commands and clearly records the unexecuted native-build limitation.",
        ),
    ),
    BenchmarkCase(
        id="threejs-multiscale-physics",
        framework="Three.js + strict TypeScript",
        objective=(
            "Build Phase 1 of a CPU-first multiscale physical-world platform beginning in deep space. "
            "Implement an independently testable mathematical kernel for SI units, particles and aggregates, "
            "gravity, electrostatics, time integration, collisions, conservation diagnostics, validity domains, "
            "uncertainty/error budgets, and hierarchical physics chunking; then visualize its state in Three.js."
        ),
        constraints=(
            "Physics code must be strict TypeScript OOP and must not import Three.js, DOM, or rendering APIs.",
            "Use 2022 CODATA/SI constants with citations; never claim one classical model is valid at every scale.",
            "Every regime declares scale/energy/time validity bounds, approximation error, and transition policy.",
            "Use deterministic fixed-step or symplectic integration and test conservation/inverse-square invariants.",
            "Chunking must support particle-to-aggregate coarse graining, spatial hierarchy, budgets, and refinement.",
            "Target an i5/32GB/no-discrete-GPU PC: CPU workers and transferable typed arrays; bounded memory/time.",
            "Rendering uses Three.js instancing, LOD/frustum culling, origin rebasing, and draw-call budgets.",
            "Include a versioned roadmap for later quantum/atomistic, continuum, celestial, Earth, robotics, "
            "agriculture, manufacturing, and vehicle-engineering phases without pretending they are implemented.",
        ),
        acceptance=(
            "npm install, strict typecheck, production build, and a deterministic non-watch test command pass.",
            "Tests measure units, gravity/electrostatic inverse-square behavior, momentum/energy drift, collision "
            "response, deterministic replay, and chunk refine/coarsen equivalence against stated tolerances.",
            "Three.js adapter renders kernel snapshots without physics dependencies on Three.js and uses scalable primitives.",
            "README documents equations, source citations, validity limits, error budgets, performance budgets, and roadmap.",
        ),
    ),
    BenchmarkCase(
        id="tkinter-pid-instrument",
        framework="Python Tkinter",
        objective=(
            "Build a digital PID process-control instrument with a Tkinter GUI, simulated first-order plant, "
            "setpoint/disturbance controls, trend plotting without external plotting libraries, and CSV logging."
        ),
        constraints=(
            "Use a monotonic-time fixed-step simulation and document discretization assumptions.",
            "Isolate controller and plant from Tkinter; tests must be headless and deterministic.",
            "Implement anti-windup and output saturation with explicit engineering units.",
            "Use only the Python standard library at runtime.",
        ),
        acceptance=(
            "python -m unittest discover -s tests -v exits zero.",
            "Tests cover proportional response, integral accumulation, derivative behavior, saturation, anti-windup, and plant convergence.",
            "The GUI module can be imported without opening a window.",
        ),
    ),
    BenchmarkCase(
        id="journalism-pqc-migration",
        framework="Evidence-grounded web research + journalism",
        objective=(
            "Research and write a publication-ready reported explainer on the transition from legacy public-key "
            "cryptography to NIST-standardized post-quantum cryptography, distinguishing standards, migration "
            "guidance, implementation evidence, risks, timelines, and unresolved questions."
        ),
        constraints=(
            "Research the live web; prioritize NIST, CISA, NSA and original technical or institutional sources.",
            "Separate verified facts, attributed analysis, and inference; do not invent interviews, quotes, dates, or statistics.",
            "Produce article.md, sources.csv, claims.csv, and research_notes.md with stable source IDs.",
            "Every material factual claim must map to one or more sources and contradictory evidence must be recorded.",
            "Use journalism structure: headline, dek, byline, dateline, lede, nut graf, sections, limitations, and corrections note.",
        ),
        acceptance=(
            "Article contains at least 900 substantive words and inline source IDs linked to the claim ledger.",
            "Source ledger contains at least eight sources, three primary sources, and four independent domains.",
            "Claim ledger contains at least twelve material claims, with no unsupported or source-less claim.",
            "Research notes document search scope, source selection, conflicts, uncertainty, and fact-check procedure.",
        ),
    ),
    BenchmarkCase(
        id="market-needs-metacognition",
        framework="Web research + SQLite evidence graph",
        objective=(
            "Research unmet needs in operational and scientific software for small laboratories and manufacturers "
            "using commodity CPU-only PCs. Combine direct user signals with product, pricing, workflow, integration, "
            "job, regulatory, and efficiency-gap evidence; infer traceable first- through fourth-order needs and propose solutions."
        ),
        constraints=(
            "Research live sources across official/product sites, pricing, reviews, forums or message boards, social sources, research, and jobs/workflows.",
            "Never fabricate posts, engagement counts, market sizes, interviews, or quantitative efficiency estimates.",
            "Produce sources.csv, observations.csv, needs.csv, solutions.csv, methodology.md, and market_needs.sqlite3.",
            "Keep observations separate from inferences. Each layer 2-4 need must name parent needs and evidence chains.",
            "Record confidence, disconfirming evidence, affected actor, workflow, current offering, and measurable gap where supported.",
        ),
        acceptance=(
            "At least twenty sources span six source types and eight independent domains, including direct-user and offering evidence.",
            "At least fifteen observations and twelve needs are stored; layers 1, 2, 3, and 4 are all represented.",
            "Every need traces to observations; every higher-order need also traces to valid lower-layer parents.",
            "SQLite tables and CSV ledgers agree, and proposed solutions map to needs, validation experiments, risks, and falsification criteria.",
            "Methodology explains sampling limits, inference rules, duplication controls, uncertainty, and what cannot be concluded.",
        ),
    ),
)


def case_prompt(case: BenchmarkCase) -> str:
    constraints = "\n".join(f"- {item}" for item in case.constraints)
    acceptance = "\n".join(f"- {item}" for item in case.acceptance)
    return f"""C0d3rV2 engineering benchmark: {case.id}

Objective:
{case.objective}

Mandatory constraints:
{constraints}

Acceptance criteria:
{acceptance}

Implement the complete project, not a design-only response. Inspect generated files, execute every locally
available validation, and repair failures through C0d3rV2 tools. Maintain a short traceability table in the
README mapping each acceptance criterion to evidence. If the host lacks a required toolchain, verify all
possible static and headless evidence and state the exact unexecuted command and limitation; do not fabricate
test results. Use WebSearch for official framework documentation whenever an API/version is uncertain.
"""


def run_benchmarks(
    root: Path,
    selected: list[str] | None = None,
    *,
    max_attempts: int = 3,
    hours: float = 8.0,
    enqueue_only: bool = False,
) -> dict:
    chosen = [case for case in CASES if not selected or case.id in selected]
    unknown = sorted(set(selected or ()) - {case.id for case in CASES})
    if unknown:
        raise ValueError(f"unknown benchmark case(s): {', '.join(unknown)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = root.resolve() / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    db_path = run_root / "benchmark.sqlite3"
    store = WorkdayStore(db_path)
    validator = Path(__file__).resolve().parents[4] / "scripts" / "validate_atf_benchmark.py"
    jobs: dict[str, str] = {}
    for priority, case in enumerate(reversed(chosen), start=1):
        workdir = run_root / case.id
        workdir.mkdir()
        command = f'python "{validator}" --case {case.id} --workdir .'
        job_id = store.enqueue(
            case_prompt(case), workdir=workdir, validation_command=command,
            priority=priority, max_attempts=max_attempts, timeout_seconds=3600,
            tags=["atf-benchmark", case.id, case.framework],
        )
        jobs[case.id] = job_id

    started = time.time()
    supervisor_report: dict = {"stop_reason": "enqueue only"}
    if not enqueue_only:
        config = WorkdayConfig.from_env(db_path=db_path)
        config = WorkdayConfig(**{
            **config.__dict__, "concurrency": 1, "shift_hours": hours,
            "retry_base_seconds": 2, "quota_retry_seconds": 60,
            "report_dir": run_root / "workday_reports",
        })
        supervisor_report = WorkdaySupervisor(config).run(
            until_empty=True, max_runtime_seconds=hours * 3600,
        )

    job_results = {case_id: store.get(job_id) for case_id, job_id in jobs.items()}
    corrections = [
        item for item in ModelFeedbackStore().correction_snapshot(limit=1000)
        if float(item["created_at"]) >= started
    ]
    payload = {
        "run_id": stamp,
        "run_root": str(run_root),
        "cases": [asdict(case) for case in chosen],
        "jobs": job_results,
        "supervisor": supervisor_report,
        "corrections": corrections,
        "summary": {
            "completed": sum(1 for job in job_results.values() if job and job["status"] == "completed"),
            "failed": sum(1 for job in job_results.values() if job and job["status"] == "failed"),
            "hallucinations": sum(1 for item in corrections if item["is_hallucination"]),
            "resolved_corrections": sum(1 for item in corrections if item["resolved"]),
        },
    }
    (run_root / "benchmark_report.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run C0d3rV2+ATF scientific GUI benchmarks")
    parser.add_argument("--root", default="runtime/agent_the_freeloader/benchmarks")
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--enqueue-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([asdict(case) for case in CASES], indent=2))
        return 0
    report = run_benchmarks(
        Path(args.root), args.case, max_attempts=max(1, args.max_attempts),
        hours=max(0.01, args.hours), enqueue_only=args.enqueue_only,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0 if not report["summary"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
