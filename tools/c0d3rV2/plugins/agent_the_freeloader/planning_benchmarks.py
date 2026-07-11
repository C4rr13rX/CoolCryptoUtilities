from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PlanningCase:
    id: str
    domain: str
    request: str
    required_constraints: tuple[str, ...]


CASES: tuple[PlanningCase, ...] = (
    PlanningCase(
        "blockchain-hybrid-cluster", "distributed systems",
        "Plan an auditable hybrid on-prem/cloud compute cluster that schedules scientific workloads, "
        "anchors immutable provenance hashes to a permissioned blockchain, tolerates network partitions, "
        "and never places raw regulated data on-chain.",
        ("consensus", "partition", "data sovereignty", "key rotation", "scheduler", "recovery", "cost", "validation"),
    ),
    PlanningCase(
        "enterprise-zero-trust-intranet", "enterprise architecture",
        "Plan migration of a 10,000-user legacy enterprise intranet to a highly available zero-trust platform "
        "with identity federation, records retention, accessibility, staged cutover, rollback, observability, "
        "and coexistence with unsupported departmental applications.",
        ("identity", "retention", "accessibility", "rollback", "availability", "audit", "migration", "validation"),
    ),
    PlanningCase(
        "sdr-radio-communications", "radio engineering",
        "Plan cross-platform software-defined radio software for spectrum monitoring and authorized digital "
        "communications with deterministic DSP, hardware abstraction, offline operation, regulatory controls, "
        "record/replay, calibrated measurements, and reproducible RF test fixtures.",
        ("sample rate", "aliasing", "calibration", "regulatory", "DSP", "record/replay", "hardware", "validation"),
    ),
    PlanningCase(
        "scientific-digital-twin-platform", "scientific computing",
        "Plan a CPU-first scientific digital-twin platform integrating sensor ingestion, dimensional units, "
        "uncertainty propagation, numerical solvers, interactive GUIs, experiment provenance, model versioning, "
        "and scale-out execution while remaining usable on an i5/32GB/no-GPU workstation.",
        ("units", "uncertainty", "numerical stability", "provenance", "CPU", "memory", "scale-out", "validation"),
    ),
)


def _imports():
    root = Path(__file__).resolve().parents[4]
    c0d3r = root / "tools" / "c0d3rV2"
    for path in (root, root / "tools", c0d3r):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from orchestrator import Orchestrator
    from task_tree import TaskTree
    from tool_registry import ToolRegistry
    from .session import AgentTheFreeloaderSession
    return Orchestrator, TaskTree, ToolRegistry, AgentTheFreeloaderSession


def score_plan(case: PlanningCase, branches: list[dict]) -> dict:
    ids = [str(item.get("id") or "") for item in branches if isinstance(item, dict)]
    known = set(ids)
    invalid_dependencies: list[str] = []
    prior: set[str] = set()
    for item in branches:
        if not isinstance(item, dict):
            continue
        branch_id = str(item.get("id") or "")
        for dependency in item.get("dependencies") or []:
            dep = str(dependency)
            if dep not in known or dep not in prior:
                invalid_dependencies.append(f"{branch_id}->{dep}")
        prior.add(branch_id)
    combined = json.dumps(branches).lower()
    coverage = {term: term.lower() in combined for term in case.required_constraints}
    required_fields = (
        "id", "description", "rationale", "constraints",
        "acceptance_criteria", "recovery_policy",
    )
    complete = sum(
        1 for branch in branches if isinstance(branch, dict)
        and all(field in branch and branch[field] not in (None, "", []) for field in required_fields)
        and isinstance(branch.get("dependencies"), list)
    )
    measurable = sum(
        1 for branch in branches if isinstance(branch, dict)
        and any(token in " ".join(map(str, branch.get("acceptance_criteria") or [])).lower()
                for token in ("test", "measure", "less than", "at least", "%", "pass", "zero", "p95"))
    )
    recovery = sum(
        1 for branch in branches if isinstance(branch, dict)
        and any(token in str(branch.get("recovery_policy") or "").lower()
                for token in ("return", "reconverge", "resume", "acceptance", "rollback"))
    )
    denominator = max(1, len(branches))
    score = round(100 * (
        0.30 * complete / denominator
        + 0.20 * measurable / denominator
        + 0.15 * recovery / denominator
        + 0.20 * sum(coverage.values()) / max(1, len(coverage))
        + 0.15 * (1.0 if not invalid_dependencies and len(ids) == len(set(ids)) else 0.0)
    ), 1)
    return {
        "score": score, "branch_count": len(branches), "complete_branches": complete,
        "measurable_branches": measurable, "recovery_branches": recovery,
        "constraint_coverage": coverage, "invalid_dependencies": invalid_dependencies,
        "duplicate_or_missing_ids": len(ids) != len(branches) or len(ids) != len(set(ids)),
    }


def run_case(case: PlanningCase, output_dir: Path) -> dict:
    Orchestrator, TaskTree, ToolRegistry, Session = _imports()
    session = Session(session_name=f"planning-benchmark-{case.id}", transcript_enabled=False)
    context = (
        "Planning benchmark. Preserve every stated constraint across sequential branches. "
        "Each branch needs measurable evidence, explicit dependencies, and a recovery policy that permits "
        "evidence-backed divergence then reconverges on the plan. Add WebSearch gates for laws, standards, "
        "framework versions, hardware APIs, or other unstable external facts. Do not implement the project."
        f" Required constraint vocabulary that must appear explicitly in applicable branches: "
        f"{', '.join(case.required_constraints)}."
    )
    orchestrator = Orchestrator(session=session, tools=ToolRegistry(), context=context, petals=None)
    scientific = orchestrator.reformulate(case.request)
    tree = TaskTree(case.request, scientific)
    branches = orchestrator._plan_branches(scientific, tree)
    score = score_plan(case, branches)
    result = {
        "case": asdict(case), "scientific_request": scientific, "branches": branches,
        "score": score, "routes": session.route_history, "finished_at": time.time(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{case.id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C0d3rV2+ATF complex planning benchmarks")
    parser.add_argument("--case", action="append", choices=[case.id for case in CASES])
    parser.add_argument("--root", default="runtime/agent_the_freeloader/planning_benchmarks")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([asdict(case) for case in CASES], indent=2))
        return 0
    chosen = [case for case in CASES if not args.case or case.id in args.case]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(args.root) / stamp
    results = [run_case(case, root) for case in chosen]
    summary = {"run_root": str(root.resolve()), "results": results,
               "mean_score": round(sum(item["score"]["score"] for item in results) / len(results), 1)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
