"""Concurrent C0d3rV2 + ATF dependency/memory injection benchmark."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.c0d3rV2.delivery_runner import run_delivery_turn_detailed
from tools.c0d3rV2.plugins.dependency_traversal import DependencyTraversal


DEFAULT_ARENA = ROOT / "runtime" / "benchmarks" / "jeopardy_ctf"


CHALLENGES = [
    {
        "id": "physics",
        "clue": "A term that repaired Ampere's law for time-varying electric fields.",
        "answer": "displacement current",
        "flag": "FLAG{MAXWELL_DISPLACEMENT_CURRENT}",
        "question": "What is Maxwell's displacement current?",
    },
    {
        "id": "biology",
        "clue": "The cellular machine that turns messenger RNA into a polypeptide.",
        "answer": "ribosome",
        "flag": "FLAG{RIBOSOME_TRANSLATION}",
        "question": "What is the ribosome?",
    },
    {
        "id": "software",
        "clue": "A graph ordering used to schedule dependencies before their consumers.",
        "answer": "topological order",
        "flag": "FLAG{TOPOLOGICAL_REGRESSION_TRAVERSAL}",
        "question": "What is a topological ordering?",
    },
]


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_arena(root: Path = DEFAULT_ARENA) -> Path:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _write(root, "README.md", "# Jeopardy Capture the Flag\nRead-only dependency traversal arena.\n")
    for challenge in CHALLENGES:
        category = challenge["id"]
        _write(
            root, f"vault/{category}/answer.ts",
            f"export const answer = {json.dumps(challenge['answer'])};\n"
            f"export const flag = {json.dumps(challenge['flag'])};\n",
        )
        _write(
            root, f"questions/{category}/question.ts",
            f"import {{ answer, flag }} from '../../vault/{category}/answer';\n"
            f"export const prompt = {json.dumps(challenge['question'])};\n"
            "export const card = { prompt, answer, flag };\n",
        )
        _write(
            root, f"rounds/{category}.ts",
            f"import {{ card }} from '../questions/{category}/question';\nexport const round = card;\n",
        )
        _write(
            root, f"tests/{category}.spec.ts",
            f"import {{ round }} from '../rounds/{category}';\n"
            f"if (round.prompt !== {json.dumps(challenge['question'])}) throw new Error('question');\n",
        )
    _write(root, "decoys/physics.ts", "export const flag = 'FLAG{NEWTON_FORCE}';\n")
    _write(root, "decoys/biology.ts", "export const flag = 'FLAG{MITOCHONDRIA}';\n")
    _write(root, "decoys/software.ts", "export const flag = 'FLAG{BUBBLE_SORT}';\n")
    _write(root, "cycles/a.ts", "import { b } from './b'; export const a = b;\n")
    _write(root, "cycles/b.ts", "import { a } from './a'; export const b = a;\n")
    return root


def _manifest(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob("*"):
        if path.is_file() and ".c0d3r" not in path.parts:
            result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _contestant(arena: Path, challenge: dict[str, str], run_id: str) -> dict[str, Any]:
    traversal = DependencyTraversal(arena)
    packet = traversal.injection_packet(
        challenge["clue"], paths=[f"vault/{challenge['id']}/answer.ts"],
        depth=4, max_nodes=30,
        memory=[{"scope": "STM", "fact": f"assigned category={challenge['id']}"}],
        hazy_hints=[str(arena / f"questions/{challenge['id']}/question.ts")],
    )
    prompt = (
        "Jeopardy Capture the Flag, read-only round. Use the injected dependency packet to "
        "recover the exact question and flag associated with this clue. Ignore decoys. Do not "
        "write or modify files. Return exactly one JSON object with keys category, question, "
        "answer, flag, evidence_paths.\n\n"
        f"Clue: {challenge['clue']}\n"
        f"Injection packet:\n{json.dumps(packet, indent=2)}"
    )
    started = time.monotonic()
    try:
        result = run_delivery_turn_detailed(
            prompt,
            session_key=f"jeopardy-ctf:{run_id}:{challenge['id']}",
            workdir=arena,
            backend="freeloader",
            system_context=(
                "You are one isolated read-only C0d3rV2 Jeopardy CTF contestant using "
                "AgentTheFreeloader. Never mutate the arena. Evidence must remain inside the arena."
            ),
            reset=True,
        )
        output = str(result.get("output") or "")
        models = result.get("models") or []
        error = str(result.get("session_error") or "")
    except Exception as exc:
        output, models, error = "", [], str(exc)
    expected = all(value.lower() in output.lower() for value in (
        challenge["question"], challenge["answer"], challenge["flag"],
    ))
    return {
        "category": challenge["id"], "passed": expected and not error,
        "expected": {key: challenge[key] for key in ("question", "answer", "flag")},
        "output": output[:12000], "error": error[:3000], "models": models,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "injection_surface": packet.get("change_surface"),
        "regression_surface": packet.get("consumers_and_regression_surface"),
    }


def run_benchmark(arena: Path = DEFAULT_ARENA, *, workers: int = 3) -> dict[str, Any]:
    arena = create_arena(arena)
    before = _manifest(arena)
    run_id = uuid.uuid4().hex[:12]
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(CHALLENGES)))) as pool:
        futures = [pool.submit(_contestant, arena, challenge, run_id) for challenge in CHALLENGES]
        for future in as_completed(futures):
            results.append(future.result())
    after = _manifest(arena)
    mutations = sorted(set(before) ^ set(after) | {path for path in before.keys() & after.keys() if before[path] != after[path]})
    report = {
        "schema": "c0d3r.jeopardy-ctf/v1", "run_id": run_id,
        "arena": str(arena), "contestants": len(results),
        "passed": sum(bool(item["passed"]) for item in results),
        "read_only_integrity": not mutations, "unexpected_mutations": mutations,
        "results": sorted(results, key=lambda item: item["category"]),
        "created_at": time.time(),
    }
    report_path = arena.parent / f"jeopardy_ctf_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
