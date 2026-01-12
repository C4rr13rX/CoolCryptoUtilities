#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_django(settings_module: str) -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    from services.env_loader import EnvLoader

    EnvLoader.load()
    import django

    django.setup()


def _safe_tail(path: str | Path, lines: int = 40) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as handle:
            data = handle.readlines()
        return [line.rstrip("\n") for line in data[-lines:]]
    except Exception:
        return []


def _run_snapshot(run_id: str) -> Dict[str, Any]:
    from branddozer.models import DeliveryRun, GateRun, DeliverySession

    run = DeliveryRun.objects.filter(id=run_id).first()
    if not run:
        return {"error": "run not found"}
    gates = list(GateRun.objects.filter(run=run).order_by("created_at"))
    gate_counts = {}
    for g in gates:
        gate_counts[g.stage] = gate_counts.get(g.stage, 0) + 1
    gate_statuses = {g.name: g.status for g in gates}
    sessions = list(DeliverySession.objects.filter(run=run).order_by("-created_at"))
    orchestrator_log = None
    for s in sessions:
        if s.role == "orchestrator" and s.log_path:
            orchestrator_log = s.log_path
            break
    log_tail = _safe_tail(orchestrator_log, 60) if orchestrator_log else []
    return {
        "run": {
            "id": str(run.id),
            "status": run.status,
            "phase": run.phase,
            "mode": run.mode,
            "iteration": run.iteration,
            "sprints": run.sprint_count,
            "started_at": run.started_at.isoformat() if run.started_at else "",
            "completed_at": run.completed_at.isoformat() if run.completed_at else "",
            "note": (run.context or {}).get("status_note", ""),
        },
        "gate_counts": gate_counts,
        "gate_statuses": gate_statuses,
        "log_tail": log_tail,
    }


def _compose_prompt(snapshot: Dict[str, Any]) -> str:
    if snapshot.get("error"):
        return f"Run snapshot error: {snapshot['error']}"
    run = snapshot.get("run", {})
    tail = "\n".join(snapshot.get("log_tail") or [])
    gate_statuses = snapshot.get("gate_statuses", {})
    gate_text = "\n".join(f"- {name}: {status}" for name, status in gate_statuses.items()) or "none yet"
    return (
        "You are the BrandDozer monitor. Provide a brief status for the delivery run and concrete next actions.\n"
        "Keep it concise: status, blockers, next steps. If gates are stalled, advise on scoping/parallelizing.\n\n"
        f"Run: {run.get('id')}\n"
        f"Status: {run.get('status')} | Phase: {run.get('phase')} | Mode: {run.get('mode')}\n"
        f"Iteration: {run.get('iteration')} | Sprints: {run.get('sprints')}\n"
        f"Started: {run.get('started_at')} | Completed: {run.get('completed_at')}\n"
        f"Note: {run.get('note')}\n"
        f"Gate statuses:\n{gate_text}\n\n"
        "Log tail:\n"
        f"{tail or '[no log tail]'}\n"
        "\nReturn in 4-6 bullets."
    )


def _write_output(output_dir: Path, snapshot: Dict[str, Any], text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"feedback_{ts}.log"
    lines = [
        f"timestamp: {ts}",
        f"run: {snapshot.get('run', {}).get('id')}",
        "",
        text,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _send_codex(prompt: str, workdir: str | Path) -> str:
    from tools.codex_session import CodexSession, codex_default_settings

    settings = codex_default_settings()
    session = CodexSession(
        session_name="branddozer-feedback",
        transcript_dir=Path("runtime/branddozer/feedback_transcripts"),
        read_timeout_s=None,
        workdir=str(workdir),
        **settings,
    )
    return session.send(prompt, stream=False)


def run_once(run_id: str, workdir: str | Path, output_dir: Path) -> Path:
    snapshot = _run_snapshot(run_id)
    prompt = _compose_prompt(snapshot)
    text = _send_codex(prompt, workdir)
    return _write_output(output_dir, snapshot, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="BrandDozer feedback loop using CodexSession.")
    parser.add_argument("run_id", help="Delivery run id (UUID).")
    parser.add_argument("--settings", default="coolcrypto_dashboard.settings", help="Django settings module.")
    parser.add_argument("--interval", type=int, default=1200, help="Seconds between feedback runs (default 1200s).")
    parser.add_argument("--output-dir", default="runtime/branddozer/feedback", help="Where to write feedback logs.")
    parser.add_argument("--once", action="store_true", help="Run once and exit.")
    args = parser.parse_args()

    _load_django(args.settings)
    workdir = PROJECT_ROOT
    output_dir = Path(args.output_dir)

    if args.once:
        path = run_once(args.run_id, workdir, output_dir)
        print(f"Wrote feedback to {path}")
        return

    print(f"Starting feedback loop for run {args.run_id} every {args.interval} seconds...")
    while True:
        path = run_once(args.run_id, workdir, output_dir)
        print(f"[{datetime.utcnow().isoformat()}] Wrote feedback to {path}")
        time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    main()
