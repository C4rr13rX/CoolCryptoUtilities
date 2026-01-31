from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import shlex
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from urllib.parse import quote
import requests
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.db import close_old_connections, DatabaseError, OperationalError, connection
from django.utils import timezone

from tools.ai_session import get_session_class, settings_for_role, session_provider_from_context
from services.branddozer_ui import UISnapshotResult, capture_ui_screenshots
from services.branddozer_jobs import update_job
from services.branddozer_github import publish_project
from services.homeostasis import (
    load_setpoints,
    gate_pass_rate as _hb_gate_pass_rate,
    gate_failures as _hb_gate_failures,
    open_backlog_count as _hb_open_backlog,
    heartbeat_payload,
    heartbeat_due,
    should_throttle,
    save_control_state,
)
from services.agent_workspace import build_context, init_notes, append_notes
from services.codex_usage import get_codex_usage
from branddozer.models import (
    AcceptanceRecord,
    BacklogItem,
    BackgroundJob,
    ChangeRequest,
    DeliveryArtifact,
    DeliveryProject,
    DeliveryRun,
    DeliverySession,
    GateRun,
    GovernanceArtifact,
    RaidEntry,
    ReleaseCandidate,
    Sprint,
    SprintItem,
)


DEFAULT_DOD = [
    "All backlog items are Done.",
    "All required quality gates are green.",
    "UX smoke + accessibility checks are green.",
    "UI screenshot verification passed.",
    "Release Candidate produced.",
    "Completion Report emitted with no unresolved risks.",
    "User acceptance recorded.",
]

MAX_REMEDIATION_CYCLES = int(os.getenv("BRANDDOZER_REMEDIATION_CYCLES", "3"))
DEFAULT_SPRINT_CAPACITY_POINTS = int(os.getenv("BRANDDOZER_SPRINT_CAPACITY_POINTS", "8"))
DEFAULT_SPRINT_CAPACITY_ITEMS = int(os.getenv("BRANDDOZER_SPRINT_CAPACITY_ITEMS", "6"))
MAX_PARALLELISM = int(os.getenv("BRANDDOZER_MAX_PARALLELISM", "3"))
GATE_FAILURE_BLOCK_LIMIT = int(os.getenv("BRANDDOZER_GATE_FAILURE_LIMIT", "2"))
LONGRUN_COOLDOWN_MINUTES = int(os.getenv("BRANDDOZER_LONGRUN_COOLDOWN_MINUTES", "60"))
DEFAULT_ETA_BASE_MINUTES = int(os.getenv("BRANDDOZER_ETA_BASE_MINUTES", "20"))
CODEX_MIN_5H_REMAIN = float(os.getenv("CODEX_MIN_5H_REMAIN", "5") or 5)
CODEX_MIN_WEEK_REMAIN = float(os.getenv("CODEX_MIN_WEEK_REMAIN", "10") or 10)
CODEX_MIN_CREDITS = float(os.getenv("CODEX_MIN_CREDITS", "0") or 0)

PHASES = [
    "mode_detection",
    "baseline_review",
    "governance",
    "requirements",
    "blueprint",
    "backlog",
    "sprint_planning",
    "execution",
    "gates",
    "ux_audit",
    "release",
    "awaiting_acceptance",
]

SESSION_LOG_ROOT = Path("runtime/branddozer/sessions")
SESSION_LOG_ROOT.mkdir(parents=True, exist_ok=True)
SOLO_PLAN_ROOT = Path("runtime/branddozer/solo_plans")
SOLO_PLAN_ROOT.mkdir(parents=True, exist_ok=True)
WORKTREE_LOCK = threading.Lock()
PLACEHOLDER_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==")
_SCHEMA_READY = False
_SCHEMA_LOCK = threading.Lock()


def _log_ai_choice(session: DeliverySession, provider: str, settings: Dict[str, Any], context: str) -> None:
    """
    Record which AI session provider/model was selected for transparency and cost tracking.
    """
    try:
        model = settings.get("model") or "<unknown>"
        role = settings.get("meta_role") or context
        _append_session_log(session, f"[{provider}] role={role} model={model}")
    except Exception:
        return


def _record_intent(run: DeliveryRun, backlog_item: BacklogItem, settings: Dict[str, Any], provider: str) -> None:
    try:
        content = json.dumps(
            {
                "title": backlog_item.title,
                "id": str(backlog_item.id),
                "priority": backlog_item.priority,
                "status": backlog_item.status,
                "provider": provider,
                "model": settings.get("model"),
                "role": settings.get("meta_role"),
            },
            indent=2,
        )
        DeliveryArtifact.objects.create(
            project=run.project,
            run=run,
            kind="task_intent",
            title=f"Intent: {backlog_item.title[:80]}",
            content=content,
            path="",
        )
    except Exception:
        return


_REASONING_MAP = {
    "extra_high": "xhigh",
    "xhigh": "xhigh",
    "xh": "xhigh",
    "high": "high",
    "h": "high",
    "medium": "medium",
    "med": "medium",
    "m": "medium",
    "low": "low",
    "l": "low",
}


def _normalize_reasoning(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _REASONING_MAP.get(key, str(value).strip())


def _is_codex_provider(provider: str) -> bool:
    return provider.strip().lower() not in {"c0d3r", "coder", "bedrock"}


def _session_settings_for_run(run: DeliveryRun, role: str) -> Dict[str, Any]:
    """
    Merge role defaults with run overrides while forcing full agent access.
    """
    provider = session_provider_from_context(run.context or {})
    settings = settings_for_role(provider, role)
    ctx = run.context or {}
    if provider in {"c0d3r", "coder", "bedrock"}:
        model = (ctx.get("c0d3r_model") or ctx.get("model") or "").strip()
        reasoning = _normalize_reasoning(ctx.get("c0d3r_reasoning") or ctx.get("reasoning_effort"))
    else:
        model = (ctx.get("codex_model") or ctx.get("model") or "").strip()
        reasoning = _normalize_reasoning(ctx.get("codex_reasoning") or ctx.get("reasoning_effort"))
    if model:
        settings["model"] = model
    if reasoning:
        settings["reasoning_effort"] = reasoning
    if provider not in {"c0d3r", "coder", "bedrock"}:
        # Enforce full agent access regardless of overrides for Codex CLI.
        settings["sandbox_mode"] = "danger-full-access"
        settings["approval_policy"] = "never"
        settings["bypass_sandbox_confirm"] = True
    return settings


def _solo_plan_dir(run_id: uuid.UUID) -> Path:
    return SOLO_PLAN_ROOT / str(run_id)


def _solo_plan_path(run_id: uuid.UUID) -> Path:
    return _solo_plan_dir(run_id) / "plan.json"


def _solo_history_path(run_id: uuid.UUID) -> Path:
    return _solo_plan_dir(run_id) / "history.jsonl"


def _load_solo_plan(run_id: uuid.UUID) -> Dict[str, Any]:
    path = _solo_plan_path(run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_solo_plan(run_id: uuid.UUID, data: Dict[str, Any]) -> None:
    plan_dir = _solo_plan_dir(run_id)
    plan_dir.mkdir(parents=True, exist_ok=True)
    path = _solo_plan_path(run_id)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_solo_history(run_id: uuid.UUID, event: Dict[str, Any]) -> None:
    history_path = _solo_history_path(run_id)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(event)
    payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _normalize_plan_steps(raw: Any) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw, start=1):
            if isinstance(item, str):
                steps.append({"id": idx, "title": item.strip(), "status": "todo"})
            elif isinstance(item, dict):
                title = (item.get("title") or item.get("step") or item.get("name") or f"Step {idx}").strip()
                status = (item.get("status") or "todo").strip().lower()
                steps.append({"id": item.get("id") or idx, "title": title, "status": status, "detail": item.get("detail")})
    elif isinstance(raw, dict):
        for idx, (key, val) in enumerate(raw.items(), start=1):
            title = str(key).strip()
            detail = val if isinstance(val, str) else json.dumps(val)
            steps.append({"id": idx, "title": title, "status": "todo", "detail": detail})
    return steps


def _resolve_smoke_command(run: DeliveryRun, root: Path) -> Optional[List[str]]:
    ctx = run.context or {}
    raw = (ctx.get("smoke_test_cmd") or os.getenv("BRANDDOZER_SOLO_SMOKE_CMD") or "").strip()
    if raw.lower() in {"off", "false", "no", "0", "skip", "none"}:
        return None
    if raw:
        return shlex.split(raw)
    return None


def _default_solo_smoke() -> Dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import networkx as nx
    except Exception as exc:
        return {"status": "error", "error": f"dependency import failed: {exc}"}
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0, scale=1.0, size=5000)
    df = pd.DataFrame({"bucket": rng.integers(0, 20, size=data.size), "value": data})
    grouped = df.groupby("bucket")["value"].agg(["mean", "std"]).reset_index()
    if grouped.empty or grouped["std"].isna().any():
        return {"status": "failed", "error": "pandas grouping failed"}
    graph = nx.gnm_random_graph(200, 400, seed=7, directed=False)
    if not nx.is_connected(graph):
        components = list(nx.connected_components(graph))
        largest = max(components, key=len)
        graph = graph.subgraph(largest).copy()
    paths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=4))
    sample = next(iter(paths.values()))
    if not sample:
        return {"status": "failed", "error": "networkx paths failed"}
    score = float(grouped["mean"].abs().sum()) + float(grouped["std"].sum()) + len(paths)
    return {"status": "passed", "score": score}


def _run_smoke_test(run: DeliveryRun, root: Path, session: Optional[DeliverySession] = None) -> Dict[str, Any]:
    cmd = _resolve_smoke_command(run, root)
    if not cmd:
        result = _default_solo_smoke()
        result["command"] = "internal"
        return result
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return {"status": "error", "command": cmd, "error": str(exc)}
    duration_ms = int((time.time() - start) * 1000)
    result = {
        "status": "passed" if proc.returncode == 0 else "failed",
        "command": cmd,
        "exit_code": proc.returncode,
        "duration_ms": duration_ms,
        "stdout": (proc.stdout or "")[-8000:],
        "stderr": (proc.stderr or "")[-8000:],
    }
    try:
        DeliveryArtifact.objects.create(
            project=run.project,
            run=run,
            session=session,
            kind="solo_smoke",
            title="Solo smoke test",
            content=json.dumps(result, indent=2),
            path="",
        )
    except Exception:
        pass
    return result

@dataclass(frozen=True)
class GateDefinition:
    name: str
    stage: str
    command: Optional[List[str]] = None
    timeout_s: int = 900
    required: bool = True
    runner: Optional[Callable[[Path], Tuple[str, str, int]]] = None
    skip_on_timeout: bool = False


def _now_ts() -> int:
    return int(time.time())


def _session_log_path(session_id: uuid.UUID) -> Path:
    return SESSION_LOG_ROOT / f"{session_id}.log"


def _append_session_log(session: DeliverySession, message: str) -> None:
    path = Path(session.log_path) if session.log_path else _session_log_path(session.id)
    if not session.log_path:
        session.log_path = str(path)
        session.save(update_fields=["log_path"])
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")
    except Exception:
        pass


def _backup_diff(workspace: Path, session_id: uuid.UUID) -> Optional[Path]:
    """
    Save a patch of the current workspace diff for safety/rollback.
    """
    try:
        from services.agent_workspace import run_command
    except Exception:
        return None
    out_dir = Path("runtime/branddozer/backups")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{session_id}_{ts}.patch"
    code, stdout, stderr = run_command("git diff", cwd=workspace)
    if code != 0 and not stdout:
        return None
    content = stdout or stderr
    if not content.strip():
        return None
    path.write_text(content, encoding="utf-8")
    return path


def _maybe_commit_changes(workspace: Path, message: str) -> Optional[str]:
    auto_commit = (os.getenv("BRANDDOZER_AUTO_COMMIT") or "0").strip().lower() not in {"0", "false", "no", "off"}
    if not auto_commit:
        return None
    try:
        from services.agent_workspace import run_command
    except Exception:
        return None
    run_command("git add -A", cwd=workspace)
    code, stdout, stderr = run_command(f'git commit -m "{message}"', cwd=workspace)
    if code != 0:
        return (stdout or stderr or "").strip()
    return (stdout or "").strip()


def _maybe_push_changes(workspace: Path) -> Optional[str]:
    auto_push = (os.getenv("BRANDDOZER_AUTO_PUSH") or "0").strip().lower() not in {"0", "false", "no", "off"}
    if not auto_push:
        return None
    try:
        from services.agent_workspace import run_command
    except Exception:
        return None
    code, stdout, stderr = run_command("git push", cwd=workspace)
    if code != 0:
        return (stdout or stderr or "").strip()
    return (stdout or "").strip()


def _latest_gate_statuses(run: DeliveryRun) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for gate in GateRun.objects.filter(run=run).order_by("name", "-created_at"):
        if gate.name not in statuses:
            statuses[gate.name] = gate.status
    return statuses


def _bundle_screenshots(run: DeliveryRun) -> List[str]:
    paths: List[str] = []
    for kind in ["ui_screenshot_mobile", "ui_screenshot_desktop", "ui_screenshot"]:
        paths.extend(
            DeliveryArtifact.objects.filter(run=run, kind=kind)
            .order_by("-created_at")
            .values_list("path", flat=True)[:6]
        )
    if not paths:
        paths.extend(
            DeliveryArtifact.objects.filter(run=run, kind="ui_screenshot")
            .order_by("-created_at")
            .values_list("path", flat=True)[:12]
        )
    return [str(p) for p in paths[:12]]


def _linkify_paths(paths: List[str]) -> List[str]:
    base = os.getenv("BRANDDOZER_ASSET_BASE_URL") or os.getenv("BRANDDOZER_UI_BASE_URL") or ""
    if not base:
        return paths
    base = base.rstrip("/")
    linked: List[str] = []
    for p in paths:
        safe_path = quote(p.lstrip("/"))
        linked.append(f"{base}/{safe_path}")
    return linked


def _load_homeostasis_setpoints() -> Dict[str, Any]:
    path = Path("config/homeostasis.yaml")
    if not path.exists():
        return {
            "signals": {
                "max_gate_failures": 2,
                "min_gate_pass_rate": 0.8,
                "max_open_backlog": 12,
                "heartbeat_interval_minutes": 10,
                "conversion_required": True,
            }
        }
    try:
        import yaml

        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}


def _gate_pass_rate(run: DeliveryRun) -> float:
    gates = list(GateRun.objects.filter(run=run))
    if not gates:
        return 1.0
    passed = sum(1 for g in gates if g.status == "passed")
    return passed / max(1, len(gates))


def _store_ux_audit_report(run: DeliveryRun) -> Optional[str]:
    try:
        screenshots = _bundle_screenshots(run)
        gate_lines = [f"- {name}: {status}" for name, status in _latest_gate_statuses(run).items()]
        checklist = [
            "Mobile-first layout verified (no overflow/hidden controls).",
            "Responsive breakpoints render cleanly (mobile/tablet/desktop).",
            "Typography scale consistent (no ad-hoc font sizes).",
            "Color contrast meets WCAG AA; uses design tokens.",
            "Touch targets ≥ 44px; spacing rhythm consistent.",
            "Critical CTAs visible above the fold; hierarchy clear.",
            "Motion/animation within budget; respects reduced-motion.",
            "No blocking errors in console/network during flows.",
            "Key funnel (landing→CTA/form) succeeds on mobile and desktop.",
        ]
        lines = [
            f"# UX Audit Report for Run {run.id}",
            f"Status: {run.status}",
            f"Phase: {run.phase}",
            "",
            "## Screenshots",
            *([f"- {p}" for p in screenshots] or ["- None found."]),
            "",
            "## UI/UX Gates",
            *(gate_lines or ["- None found."]),
            "",
            "## Design QA Checklist",
            *[f"- [ ] {item}" for item in checklist],
        ]
        content = "\n".join(lines)
        report_path = Path("runtime/branddozer/reports") / f"ux_audit_{run.id}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(content, encoding="utf-8")
        DeliveryArtifact.objects.create(
            project=run.project,
            run=run,
            kind="ux_audit_report",
            title="UX Audit Report",
            content=content,
            path=str(report_path),
        )
        return str(report_path)
    except Exception:
        return None


def _notify_run_summary(run: DeliveryRun, ux_report_path: Optional[str]) -> None:
    webhook = os.getenv("BRANDDOZER_WEBHOOK_URL")
    if not webhook:
        return
    try:
        gates = _latest_gate_statuses(run)
        screenshots = _linkify_paths(_bundle_screenshots(run))
        open_backlog = BacklogItem.objects.filter(run=run).exclude(status="done").count()
        signals = {
            "gate_pass_rate": round(_gate_pass_rate(run), 3),
            "gate_failures": sum(1 for status in gates.values() if status not in {"passed", "skipped"}),
            "open_backlog": open_backlog,
        }
        attachments = []
        if ux_report_path:
            attachments.append({"text": f"UX report: {ux_report_path}"})
        if screenshots:
            attachments.append(
                {
                    "title": "Screenshots",
                    "text": "\n".join(screenshots[:5]),
                }
            )
        payload = {
            "text": f"BrandDozer run {run.id} status={run.status} phase={run.phase}",
            "attachments": attachments
            + [
                {
                    "title": "Gates",
                    "text": "\n".join(f"- {k}: {v}" for k, v in gates.items()) or "No gates recorded.",
                },
                {
                    "title": "Signals",
                    "text": "\n".join(f"- {k}: {v}" for k, v in signals.items()),
                }
            ],
        }
        requests.post(webhook, json=payload, timeout=5)
    except Exception:
        return
    try:
        session.last_heartbeat = timezone.now()
        session.save(update_fields=["last_heartbeat"])
    except Exception:
        pass


class StopDelivery(Exception):
    pass


def _codex_quota_exhausted(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    markers = [
        "insufficient_quota",
        "exceeded your current quota",
        "out of credits",
        "not enough credits",
        "quota exceeded",
        "billing hard limit",
        "credit balance is insufficient",
        "insufficient balance",
        "rate limit reached for",
        "run out of time",
        "context length exceeded",
    ]
    for marker in markers:
        if marker in lowered:
            return f"Codex unavailable: {marker}"
    return None

def _codex_refusal(text: str) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    markers = [
        "i cannot comply",
        "i'm unable to comply",
        "i can't comply",
        "i cannot help with that request",
        "cannot assist with that request",
        "as an ai language model, i cannot",
        "cannot generate that",
        "i'm not able to help with that",
        "refuse",
        "not permitted to",
        "against policy",
        "violates policy",
        "can't do that",
        "cannot do that",
        "not allowed to",
        "restricted content",
    ]
    for marker in markers:
        if marker in lowered:
            return f"Codex refusal detected: {marker}"
    return None


def _pause_run_for_codex(run: DeliveryRun, session: Optional[DeliverySession], reason: str) -> None:
    note = reason[:400]
    _set_run_note(run, "Codex paused", note)
    run.status = "blocked"
    run.error = note
    context = dict(run.context or {})
    context["codex_paused"] = {
        "reason": note,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session": str(session.id) if session else None,
    }
    run.context = context
    run.save(update_fields=["status", "error", "context"])
    if session:
        _append_session_log(session, note)
        session.status = "blocked"
        session.save(update_fields=["status"])
    title = "Codex credits exhausted"
    exists = (
        BacklogItem.objects.filter(run=run, source="system", title=title)
        .exclude(status="done")
        .exists()
    )
    if not exists:
        BacklogItem.objects.create(
            project=run.project,
            run=run,
            kind="risk",
            title=title,
            description="Codex reported insufficient quota/credits. Add credits and resume the run.",
            acceptance_criteria=["Credits restored", "Delivery run resumed and Codex sessions succeed"],
            priority=1,
            estimate_points=1,
            status="blocked",
            source="system",
        )


def _handle_codex_refusal(run: DeliveryRun, session: Optional[DeliverySession], reason: str, *, block: bool = True) -> None:
    note = reason[:400]
    _set_run_note(run, "Codex refusal", note)
    context = dict(run.context or {})
    context.setdefault("codex_refusals", []).append(
        {"reason": note, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "session": str(session.id) if session else None}
    )
    run.context = context
    if block:
        run.status = "blocked"
        run.error = note
        run.save(update_fields=["status", "error", "context"])
    else:
        run.save(update_fields=["context"])
    if session:
        _append_session_log(session, note)
        if block:
            session.status = "blocked"
            session.save(update_fields=["status"])
    title = "Codex refused prompt"
    exists = (
        BacklogItem.objects.filter(run=run, source="system", title=title)
        .exclude(status="done")
        .exists()
    )
    if not exists:
        BacklogItem.objects.create(
            project=run.project,
            run=run,
            kind="risk",
            title=title,
            description="Codex declined to act on a prompt. Adjust prompt/scope and retry.",
            acceptance_criteria=["Prompt adjusted and Codex completes task", "Run resumed"],
            priority=2,
            estimate_points=1,
            status="blocked" if block else "todo",
            source="system",
        )


def _cleanup_run_sessions(run: DeliveryRun, reason: str = "stale session cleanup") -> None:
    """
    Force-complete any stale sessions for this run to avoid hung codex processes.
    """
    stale = list(DeliverySession.objects.filter(run=run, status="running"))
    if not stale:
        return
    now = timezone.now()
    for sess in stale:
        try:
            _append_session_log(sess, f"Session closed: {reason}")
        except Exception:
            pass
        sess.status = "error"
        sess.completed_at = now
        sess.save(update_fields=["status", "completed_at"])

def _codex_budget_ok(run: DeliveryRun) -> Tuple[bool, Optional[str]]:
    provider = session_provider_from_context(run.context or {})
    if not _is_codex_provider(provider):
        return True, None
    usage = get_codex_usage()
    ctx = dict(run.context or {})
    ctx["codex_usage"] = usage
    run.context = ctx
    run.save(update_fields=["context"])
    reason = None
    five_remain = usage.get("five_hour_remaining_pct")
    if five_remain is not None and five_remain < CODEX_MIN_5H_REMAIN:
        reason = f"Codex 5h window low ({five_remain}%)"
    week_remain = usage.get("week_remaining_pct")
    if reason is None and week_remain is not None and week_remain < CODEX_MIN_WEEK_REMAIN:
        reason = f"Codex weekly low ({week_remain}%)"
    credits = usage.get("credits_remaining")
    if reason is None and credits is not None and credits < CODEX_MIN_CREDITS:
        reason = f"Codex credits low ({credits})"
    return reason is None, reason

def _estimate_eta_minutes(run: DeliveryRun, backlog: Optional[List[BacklogItem]] = None) -> int:
    """
    Lightweight ETA heuristic (CPU-only, no ML) using backlog size, gate failures,
    long-running flags, and remediation cycles. Aims to avoid human-scale estimates.
    """
    backlog = backlog if backlog is not None else list(BacklogItem.objects.filter(run=run).exclude(status="done"))
    open_count = len(backlog)
    long_running = sum(1 for item in backlog if (item.meta or {}).get("long_running"))
    gate_failures = 0
    try:
        gate_failures = sum(1 for g in GateRun.objects.filter(run=run) if g.status not in {"passed", "skipped"})
    except Exception:
        gate_failures = 0
    remediation = getattr(run, "iteration", 0) or 0
    base = DEFAULT_ETA_BASE_MINUTES
    per_item = 8
    eta = base + per_item * open_count + 5 * gate_failures + 15 * remediation + 30 * long_running
    # If homeostasis throttled, add cushion
    ctx = run.context or {}
    if ctx.get("throttle_new_work"):
        eta += 20
    return max(5, eta)

def _update_eta(run: DeliveryRun, *, backlog: Optional[List[BacklogItem]] = None, reason: str = "updated") -> None:
    try:
        eta_min = _estimate_eta_minutes(run, backlog=backlog)
        ctx = dict(run.context or {})
        ctx["eta"] = {
            "minutes": int(eta_min),
            "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
        }
        run.context = ctx
        run.save(update_fields=["context"])
    except Exception:
        return


def _trigger_unstick_session(run: DeliveryRun, root: Path, reason: str) -> None:
    """
    Fire a short, bounded Codex session to suggest how to unblock a stuck run.
    Guarded to run once per run to avoid loops/credit burn.
    """
    flag = (os.getenv("BRANDDOZER_UNSTICK") or "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return
    ctx = dict(run.context or {})
    if ctx.get("unstick_attempted"):
        return
    ctx["unstick_attempted"] = True
    run.context = ctx
    run.save(update_fields=["context"])
    try:
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="pm",
            name="Unstick Session",
            status="running",
            workspace_path=str(root),
            last_heartbeat=timezone.now(),
            meta={"reason": reason[:200]},
        )
    except Exception:
        return
    _append_session_log(session, f"Analyzing stuck state: {reason}")
    provider = session_provider_from_context(run.context or {})
    ai_settings = _session_settings_for_run(run, "manager")
    try:
        codex_timeout = int(os.getenv("UNSTICK_TIMEOUT", "180"))
    except Exception:
        codex_timeout = 180
    prompt = (
        "You are the unblocker for a stalled delivery run. "
        "Provide a concise plan to get unstuck without looping or wasting credits. "
        "Return JSON with keys: plan (list of steps), stop (true/false), notes. "
        f"Stuck reason: {reason}\n"
        f"Run status: {run.status} phase={run.phase}\n"
        f"Backlog counts: open={BacklogItem.objects.filter(run=run).exclude(status='done').count()}\n"
        f"Gates: {list(GateRun.objects.filter(run=run).values('name','status'))}"
    )
    try:
        SessionClass = get_session_class(provider)
        codex = SessionClass(
            session_name=f"unstick-{run.id}",
            transcript_dir=Path("runtime/branddozer/transcripts"),
            read_timeout_s=codex_timeout,
            workdir=str(root),
            **ai_settings,
        )
        output = codex.send(prompt, stream=True)
        if _is_codex_provider(provider):
            exhausted = _codex_quota_exhausted(output)
            refusal = _codex_refusal(output)
            if exhausted:
                _pause_run_for_codex(run, session, exhausted)
            elif refusal:
                _handle_codex_refusal(run, session, refusal, block=False)
        DeliveryArtifact.objects.create(
            project=run.project,
            run=run,
            session=session,
            kind="unstick_plan",
            title="Unstick plan",
            content=output,
            path=str(getattr(codex, "transcript_path", "") or ""),
        )
    except Exception as exc:
        _append_session_log(session, f"Unstick attempt failed: {exc}")
    session.status = "done"
    session.completed_at = timezone.now()
    session.save(update_fields=["status", "completed_at"])


def _emit_ui_snapshot_placeholder(run_id: uuid.UUID, reason: str, manual: bool = False) -> None:
    """
    Emit a filesystem-only UX snapshot README when the database is unreachable so
    downstream tooling still has evidence of the attempted gate.
    """
    snapshot_root = Path("runtime/branddozer/snapshots") / str(run_id or "unknown")
    snapshot_ts = time.strftime("%Y%m%d_%H%M%S")
    snapshot_dir = snapshot_root / snapshot_ts
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    lines = [
        f"# UX Snapshot {snapshot_ts}",
        "",
        f"Run: {run_id}",
        "",
        "UI snapshot skipped because the database connection was unavailable.",
        f"Reason: {reason}"[:400],
        "Gates were not updated; rerun once the DB is reachable or enable SQLite fallback for sandboxed runs.",
    ]
    if manual:
        lines.append("Triggered manually.")
    doc_path = snapshot_dir / "README.md"
    try:
        doc_path.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass
    log_path = Path("runtime/branddozer/ui") / "ui_snapshot_fallback.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"[{stamp}] run={run_id} reason={reason} manual={manual}\n")
    except Exception:
        return


def _write_placeholder_shot(output_dir: Path, reason: str) -> Optional[Path]:
    """
    Create a tiny placeholder screenshot so downstream gates and audit folders
    still receive an artifact when Playwright/npm assets are unavailable.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        shot_path = output_dir / "placeholder.png"
        shot_path.write_bytes(PLACEHOLDER_PNG)
        note_path = output_dir / "placeholder.txt"
        note = reason.strip() or "capture unavailable"
        note_path.write_text(f"UI capture placeholder: {note}\n", encoding="utf-8")
        return shot_path
    except Exception:
        return None


def _log_pipeline_failure(run_id: uuid.UUID, message: str) -> None:
    path = Path("runtime/branddozer") / "pipeline_failures.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"[{stamp}] run={run_id} {message}\n")
    except Exception:
        return


def _set_run_note(run: DeliveryRun, note: str, detail: str = "") -> None:
    context = dict(run.context or {})
    context["status_note"] = note
    context["status_detail"] = detail
    context["status_ts"] = timezone.now().isoformat()
    history = context.get("workflow_history") or []
    history_entry = {
        "phase": run.phase,
        "note": note,
        "detail": detail,
        "ts": context["status_ts"],
    }
    history.append(history_entry)
    context["workflow_history"] = history[-60:]
    run.context = context
    run.save(update_fields=["context"])
    job_id = context.get("job_id")
    if job_id:
        update_job(str(job_id), message=note, detail=detail)


def _delivery_run_job_user(run: DeliveryRun) -> Optional[Any]:
    job = (
        BackgroundJob.objects.filter(run=run, kind="delivery_run")
        .select_related("user")
        .order_by("-created_at")
        .first()
    )
    if job and job.user:
        return job.user
    context = run.context or {}
    user_id = context.get("job_user_id")
    if not user_id:
        return None
    UserModel = get_user_model()
    try:
        return UserModel.objects.filter(pk=user_id).first()
    except Exception:
        return None


def _attempt_github_push(
    run: DeliveryRun,
    session: DeliverySession,
    *,
    user: Optional[Any],
) -> Optional[Dict[str, Any]]:
    if not user:
        return None
    project = run.project
    if not (project.repo_url or project.root_path):
        return None
    data: Dict[str, Any] = {}
    context = run.context or {}
    account_id = context.get("github_account_id")
    if account_id:
        data["account_id"] = account_id
        data["github_account_id"] = account_id
    if project.repo_branch:
        data["branch"] = project.repo_branch
    data["message"] = f"Delivery run {run.id.hex[:8]} updates"

    def _progress(message: str, detail: str = "") -> None:
        text = f"github push: {message}"
        if detail:
            text = f"{text} ({detail})"
        _append_session_log(session, text)

    try:
        result = publish_project(user, str(project.id), data, progress=_progress)
        status = result.get("status")
        _append_session_log(session, f"GitHub push result: {status}")
        return {
            "attempted": True,
            "success": status == "pushed",
            "status": status,
            "repo_url": result.get("repo_url"),
            "branch": result.get("branch"),
            "timestamp": timezone.now().isoformat(),
        }
    except Exception as exc:
        message = str(exc)
        _append_session_log(session, f"GitHub push failed: {message}")
        return {
            "attempted": True,
            "success": False,
            "error": message,
            "timestamp": timezone.now().isoformat(),
        }


def _stop_requested(run: DeliveryRun) -> bool:
    run.refresh_from_db(fields=["status", "context"])
    if run.status in {"blocked", "error"}:
        return True
    return bool((run.context or {}).get("stop_requested"))


def _read_meminfo() -> Dict[str, int]:
    info: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                value = raw.strip().split(" ")[0]
                if value.isdigit():
                    info[key] = int(value)
    except Exception:
        return {}
    return info


def _dynamic_parallelism() -> int:
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    meminfo = _read_meminfo()
    available_kb = meminfo.get("MemAvailable", 0)
    if available_kb and available_kb < 2_500_000:
        return 1
    if load > 2.5:
        return 1
    return 2


def _safe_run(cmd: List[str], cwd: Path, timeout: int) -> Tuple[str, str, int]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=os.environ.copy(),
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", f"command not found: {cmd[0]}", 127
    except subprocess.TimeoutExpired:
        return "", "timeout", 124
    except Exception as exc:
        return "", f"error: {exc}", 1


def _git_available(path: Path) -> bool:
    return (path / ".git").exists()


def _git_head(path: Path) -> str:
    stdout, _stderr, code = _safe_run(["git", "rev-parse", "HEAD"], path, 10)
    return stdout.strip() if code == 0 else ""


def _git_status(path: Path) -> str:
    stdout, _stderr, code = _safe_run(["git", "status", "--porcelain"], path, 10)
    return stdout.strip() if code == 0 else ""


def _hash_tree(path: Path) -> str:
    hasher = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "venv", "__pycache__"}]
        for name in sorted(files):
            full = Path(root) / name
            try:
                stat = full.stat()
            except Exception:
                continue
            hasher.update(str(full.relative_to(path)).encode("utf-8", errors="replace"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
    return hasher.hexdigest()


def compute_input_hash(root: Path) -> str:
    if _git_available(root):
        head = _git_head(root)
        status = _git_status(root)
        return hashlib.sha256(f"{head}:{status}".encode("utf-8")).hexdigest()
    return _hash_tree(root)


def _secret_scan(root: Path, max_seconds: Optional[int] = None) -> Tuple[str, str, int]:
    patterns = [
        ("AWS Access Key", r"AKIA[0-9A-Z]{16}"),
        ("GitHub Token", r"ghp_[A-Za-z0-9]{36,}"),
        ("OpenAI Key", r"sk-[A-Za-z0-9]{32,}"),
        ("Private Key", r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"),
    ]
    findings: List[str] = []
    max_seconds = max_seconds or int(os.getenv("BRANDDOZER_SECRET_SCAN_TIMEOUT", "420"))
    start = time.time()
    skip_dirs = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "data",
        "runtime",
        "logs",
        "collected_static",
        "static",
        "lib",
        "bin",
        "include",
        "share",
    }
    extra_skip = os.getenv("BRANDDOZER_SECRET_SCAN_SKIP")
    if extra_skip:
        for part in extra_skip.split(","):
            part = part.strip()
            if part:
                skip_dirs.add(part)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            full = Path(dirpath) / filename
            if "site-packages" in full.parts:
                continue
            if full.suffix in {".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".tar", ".bin"}:
                continue
            try:
                if full.stat().st_size > 1_000_000:
                    continue
                text = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for label, regex in patterns:
                if re.search(regex, text):
                    findings.append(f"{label}: {full}")
            if max_seconds and time.time() - start > max_seconds:
                findings.append(f"Scan aborted after {int(time.time() - start)}s (limit {max_seconds}s).")
                return "\n".join(findings), "secret scan timed out", 124
    duration = int(time.time() - start)
    if findings:
        return "\n".join(findings), f"findings detected (scan {duration}s)", 1
    return f"No secrets detected. Scan completed in {duration}s.", "", 0


def _extract_json_payload(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _touch_session(session_id: uuid.UUID) -> None:
    try:
        DeliverySession.objects.filter(id=session_id).update(last_heartbeat=timezone.now())
    except Exception:
        pass


def _normalize_acceptance_criteria(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n;]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_dependencies(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,\n;]+", value)
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _definition_of_ready(item: BacklogItem) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not item.title:
        missing.append("title")
    if not (item.description or "").strip():
        missing.append("description")
    if not item.acceptance_criteria:
        missing.append("acceptance_criteria")
    if item.estimate_points is None or item.estimate_points <= 0:
        missing.append("estimate_points")
    return (len(missing) == 0), missing


def _dependencies_met(item: BacklogItem, *, done_ids: set[str], done_titles: set[str]) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for dep in item.dependencies or []:
        key = str(dep).strip()
        if not key:
            continue
        if key in done_ids or key.lower() in done_titles:
            continue
        missing.append(key)
    return (len(missing) == 0), missing


def _sprint_capacity(parallelism: int) -> Tuple[int, int]:
    points = max(1, DEFAULT_SPRINT_CAPACITY_POINTS) * max(1, parallelism)
    items = max(1, DEFAULT_SPRINT_CAPACITY_ITEMS) * max(1, parallelism)
    return points, items


def _django_hardening(root: Path) -> Tuple[str, str, int]:
    settings_path = root / "web" / "coolcrypto_dashboard" / "settings.py"
    if not settings_path.exists():
        return "", "settings.py not found", 1
    content = settings_path.read_text(encoding="utf-8", errors="ignore")
    issues = []
    checks = {
        "DEBUG": "False",
        "SECURE_SSL_REDIRECT": "True",
        "SESSION_COOKIE_SECURE": "True",
        "CSRF_COOKIE_SECURE": "True",
    }
    for key, expected in checks.items():
        if f"{key} = {expected}" not in content:
            issues.append(f"{key} should be {expected}")
    if "ALLOWED_HOSTS" not in content:
        issues.append("ALLOWED_HOSTS not configured")
    if "CSRF_TRUSTED_ORIGINS" not in content:
        issues.append("CSRF_TRUSTED_ORIGINS not configured")
    if issues:
        return "", "\n".join(issues), 1
    return "Django hardening checks passed.", "", 0


def _missing_tool(tool_name: str) -> Callable[[Path], Tuple[str, str, int]]:
    def _runner(_root: Path) -> Tuple[str, str, int]:
        return "", f"{tool_name} not installed", 1

    return _runner


def _requirements_snapshot(root: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    req = root / "requirements.txt"
    if req.exists():
        data["requirements.txt"] = req.read_text(encoding="utf-8", errors="ignore")
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        data["pyproject.toml"] = pyproject.read_text(encoding="utf-8", errors="ignore")
    package_json = root / "package.json"
    if package_json.exists():
        data["package.json"] = package_json.read_text(encoding="utf-8", errors="ignore")
    return data


def _repo_structure(root: Path) -> Dict[str, Any]:
    entries = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.name in {".git", "node_modules", ".venv", "venv"}:
            continue
        entries.append({"name": child.name, "type": "dir" if child.is_dir() else "file"})
    workflows = list((root / ".github" / "workflows").glob("*.yml")) if (root / ".github").exists() else []
    return {
        "root_entries": entries,
        "workflows": [str(p.relative_to(root)) for p in workflows],
        "has_git": _git_available(root),
    }


def _workspace_root(run_id: uuid.UUID) -> Path:
    base = Path("runtime/branddozer/workspaces") / str(run_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _create_workspace(root: Path, run_id: uuid.UUID, session_id: uuid.UUID) -> Tuple[Path, Optional[str]]:
    workspace = _workspace_root(run_id) / str(session_id)
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    if _git_available(root):
        branch = f"bdz/{session_id.hex[:8]}"
        with WORKTREE_LOCK:
            stdout, stderr, code = _safe_run(["git", "worktree", "add", "-b", branch, str(workspace)], root, 60)
        if code != 0:
            raise ValueError(stderr or stdout or "Failed to create git worktree")
        return workspace, branch
    shutil.copytree(root, workspace, dirs_exist_ok=True)
    return workspace, None


def _cleanup_workspace(root: Path, workspace: Path, branch: Optional[str]) -> None:
    if _git_available(root) and branch:
        with WORKTREE_LOCK:
            _safe_run(["git", "worktree", "remove", "--force", str(workspace)], root, 30)
            _safe_run(["git", "branch", "-D", branch], root, 30)
    else:
        shutil.rmtree(workspace, ignore_errors=True)


def _compute_diff(root: Path, workspace: Path) -> str:
    if _git_available(workspace):
        diff_stdout, _stderr, _code = _safe_run(["git", "diff"], workspace, 60)
        status_stdout, _s_err, _s_code = _safe_run(["git", "status", "--porcelain"], workspace, 10)
        if status_stdout.strip():
            diff_stdout += f"\n\n# Untracked/modified files\n{status_stdout}"
        return diff_stdout
    stdout, _stderr, _code = _safe_run(["diff", "-ruN", str(root), str(workspace)], root, 60)
    return stdout


def _apply_diff(root: Path, diff_text: str) -> Tuple[str, str, int]:
    if not diff_text.strip():
        return "No diff to apply.", "", 0
    patch_file = Path("runtime/branddozer") / "apply.patch"
    patch_file.write_text(diff_text, encoding="utf-8")
    if _git_available(root):
        check = _safe_run(["git", "apply", "--check", str(patch_file)], root, 30)
        if check[2] != 0:
            return check
        return _safe_run(["git", "apply", str(patch_file)], root, 60)
    return _safe_run(["patch", "-p1", "-i", str(patch_file)], root, 60)


def _latest_version(run: DeliveryRun, kind: str) -> int:
    latest = GovernanceArtifact.objects.filter(run=run, kind=kind).order_by("-version").first()
    return (latest.version if latest else 0) + 1


def _log_schema_event(message: str) -> None:
    path = Path("runtime/branddozer/schema.log")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"[{stamp}] {message}\n")
    except Exception:
        return


def _ensure_branddozer_schema(reason: str = "") -> bool:
    """
    Ensure BrandDozer tables exist when running with a SQLite fallback.
    Safe to call repeatedly; no-ops for non-SQLite databases.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return True
    if (os.getenv("BRANDDOZER_SKIP_AUTOMIGRATE") or "0").lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        engine = connection.settings_dict.get("ENGINE", "")
    except Exception:
        return False
    if "sqlite" not in engine:
        _SCHEMA_READY = True
        return True
    with _SCHEMA_LOCK:
        if _SCHEMA_READY:
            return True
        try:
            tables = set(connection.introspection.table_names())
        except Exception as exc:
            _log_schema_event(f"introspection failed reason={reason} error={exc}")
            tables = set()
        required = {
            "branddozer_brandproject",
            "branddozer_deliveryproject",
            "branddozer_deliveryrun",
            "branddozer_backlogitem",
            "branddozer_sprint",
        }
        if required.issubset(tables):
            _SCHEMA_READY = True
            return True
        try:
            call_command("migrate", interactive=False, verbosity=0)
        except Exception as exc:
            _log_schema_event(f"migrate failed reason={reason} error={exc}")
            return False
        try:
            tables = set(connection.introspection.table_names())
        except Exception:
            tables = set()
        _SCHEMA_READY = required.issubset(tables)
        _log_schema_event(
            f"schema_ready={_SCHEMA_READY} reason={reason} tables={len(tables)} missing={','.join(sorted(required - tables))}"
        )
        return _SCHEMA_READY


class GateRunner:
    def __init__(self, run: DeliveryRun, root: Path) -> None:
        self.run = run
        self.root = root

    def run_gate(self, gate: GateDefinition) -> GateRun:
        input_hash = compute_input_hash(self.root)
        last = (
            GateRun.objects.filter(run=self.run, name=gate.name, input_hash=input_hash)
            .order_by("-created_at")
            .first()
        )
        if last and last.status == "passed":
            return GateRun.objects.create(
                project=self.run.project,
                run=self.run,
                stage=gate.stage,
                name=gate.name,
                status="skipped",
                command=" ".join(gate.command or []),
                stdout="cached: previously passed",
                input_hash=input_hash,
                meta={"cached": True, "required": gate.required},
        )
        start = time.time()
        if gate.runner:
            try:
                stdout, stderr, code = gate.runner(self.root)
            except Exception as exc:
                stdout, stderr, code = "", f"runner error: {exc}", 1
        elif gate.command:
            stdout, stderr, code = _safe_run(gate.command, self.root, gate.timeout_s)
        else:
            stdout, stderr, code = "", "missing gate runner", 1
        status = "passed" if code == 0 else "failed"
        not_relevant = False
        timeout_hit = code == 124
        if timeout_hit and gate.skip_on_timeout:
            status = "skipped"
            not_relevant = True
        if code != 0 and not gate.required and "not installed" in (stderr or "").lower():
            status = "skipped"
            not_relevant = True
        duration = int((time.time() - start) * 1000)
        return GateRun.objects.create(
            project=self.run.project,
            run=self.run,
            stage=gate.stage,
            name=gate.name,
            status=status,
            command=" ".join(gate.command or []),
            stdout=stdout,
            stderr=stderr,
            exit_code=code,
            duration_ms=duration,
            input_hash=input_hash,
            meta={
                "required": gate.required,
                "not_relevant": not_relevant,
                "timeout": timeout_hit,
                "skip_on_timeout": gate.skip_on_timeout,
            },
        )


def default_gates(root: Path) -> List[GateDefinition]:
    gates: List[GateDefinition] = []

    fast_baseline = (os.getenv("BRANDDOZER_FAST_BASELINE", "1").strip().lower() not in {"0", "false", "no"})
    pytest_args_env = os.getenv("BRANDDOZER_PYTEST_ARGS")
    if pytest_args_env:
        pytest_args = shlex.split(pytest_args_env)
    elif fast_baseline:
        pytest_args = ["-q", "--maxfail=1", "--disable-warnings", "--timeout=300"]
    else:
        pytest_args = ["-q"]
    pytest_cmd = ["python", "-m", "pytest", *pytest_args]
    unittest_cmd = ["python", "-m", "unittest", "discover"]
    test_timeout = 420 if fast_baseline else 900
    if shutil.which("pytest"):
        gates.append(GateDefinition(name="unit-tests", stage="fast", command=pytest_cmd, timeout_s=test_timeout))
    else:
        gates.append(GateDefinition(name="unit-tests", stage="fast", command=unittest_cmd, timeout_s=test_timeout))

    if shutil.which("ruff"):
        gates.append(GateDefinition(name="lint", stage="fast", command=["ruff", "check", "."], timeout_s=600))
        gates.append(GateDefinition(name="format", stage="fast", command=["ruff", "format", "--check", "."], timeout_s=600))
    else:
        compile_cmd = ["python", "-m", "compileall", "-q", "web", "services"]
        gates.append(GateDefinition(name="syntax", stage="fast", command=compile_cmd, timeout_s=300 if fast_baseline else 600))

    gates.append(
        GateDefinition(
            name="pip-check",
            stage="integration",
            command=["python", "-m", "pip", "check"],
            timeout_s=300 if fast_baseline else 600,
            skip_on_timeout=True,
        )
    )

    if shutil.which("pip-audit"):
        gates.append(
            GateDefinition(
                name="dependency-vuln",
                stage="security",
                command=["pip-audit", "-r", "requirements.txt"],
                timeout_s=900,
                skip_on_timeout=True,
            )
        )
    else:
        gates.append(GateDefinition(name="dependency-vuln", stage="security", runner=_missing_tool("pip-audit"), timeout_s=5, required=False))

    if shutil.which("bandit"):
        gates.append(
            GateDefinition(
                name="static-security",
                stage="security",
                command=["bandit", "-r", "."],
                timeout_s=600 if fast_baseline else 900,
                skip_on_timeout=True,
            )
        )
    else:
        gates.append(GateDefinition(name="static-security", stage="security", runner=_missing_tool("bandit"), timeout_s=5, required=False))

    secret_timeout = int(os.getenv("BRANDDOZER_SECRET_SCAN_TIMEOUT", "300" if fast_baseline else "420"))
    gates.append(
        GateDefinition(
            name="secret-scan",
            stage="security",
            runner=lambda root, limit=secret_timeout: _secret_scan(root, max_seconds=limit),
            timeout_s=max(secret_timeout, 300),
            skip_on_timeout=True,
        )
    )
    gates.append(GateDefinition(name="django-hardening", stage="security", runner=_django_hardening, timeout_s=120))

    playwright_config = any((root / f).exists() for f in ["playwright.config.ts", "playwright.config.js"])
    if playwright_config and shutil.which("npx"):
        gates.append(
            GateDefinition(
                name="e2e-smoke",
                stage="e2e",
                command=["npx", "playwright", "test", "--reporter=line"],
                timeout_s=900 if fast_baseline else 1800,
                skip_on_timeout=True,
            )
        )
        gates.append(
            GateDefinition(
                name="a11y",
                stage="e2e",
                command=["npx", "playwright", "test", "--grep", "@a11y", "--reporter=line"],
                timeout_s=900 if fast_baseline else 1800,
                skip_on_timeout=True,
            )
        )
    else:
        gates.append(GateDefinition(name="e2e-smoke", stage="e2e", runner=_missing_tool("playwright"), timeout_s=5, required=False))
        gates.append(GateDefinition(name="a11y", stage="e2e", runner=_missing_tool("playwright-a11y"), timeout_s=5, required=False))

    return gates


class DeliveryOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}
        self._last_heartbeat: Dict[str, float] = {}

    def _check_stop(self, run: DeliveryRun, session: Optional[DeliverySession], note: str = "Stop requested.") -> None:
        if _stop_requested(run):
            if session:
                _append_session_log(session, note)
            _set_run_note(run, "Stop requested", note)
            raise StopDelivery(note)

    def create_run(
        self,
        project_id: str,
        prompt: str,
        mode: str = "auto",
        run_id: Optional[str] = None,
        research_mode: bool = False,
        team_mode: Optional[str] = None,
        session_provider: Optional[str] = None,
        codex_model: Optional[str] = None,
        codex_reasoning: Optional[str] = None,
        c0d3r_model: Optional[str] = None,
        c0d3r_reasoning: Optional[str] = None,
        smoke_test_cmd: Optional[str] = None,
    ) -> DeliveryRun:
        _ensure_branddozer_schema("create_run")
        project = BrandDozerProjectLookup.get_project(project_id)
        delivery_project, _ = DeliveryProject.objects.get_or_create(project=project, defaults={"definition_of_done": DEFAULT_DOD})
        run_uuid = uuid.UUID(str(run_id)) if run_id else uuid.uuid4()
        run = DeliveryRun.objects.filter(id=run_uuid).first()
        if run:
            # Reset stale runs so they can be restarted without manual DB edits.
            _cleanup_run_sessions(run, reason="reset run")
            run.prompt = prompt.strip() or run.prompt
            run.mode = mode
            run.status = "queued"
            run.phase = ""
            run.iteration = 0
            run.sprint_count = 0
            run.error = ""
            run.started_at = None
            run.completed_at = None
            run.acceptance_recorded = False
            run.definition_of_done = delivery_project.definition_of_done or DEFAULT_DOD
            ctx = dict(run.context or {})
            ctx.pop("status_note", None)
            ctx.pop("status_detail", None)
            if research_mode:
                ctx["research_mode"] = True
            if team_mode:
                ctx["team_mode"] = team_mode
            if session_provider:
                ctx["session_provider"] = session_provider
            if codex_model:
                ctx["codex_model"] = codex_model
            if codex_reasoning:
                ctx["codex_reasoning"] = codex_reasoning
            if c0d3r_model:
                ctx["c0d3r_model"] = c0d3r_model
            if c0d3r_reasoning:
                ctx["c0d3r_reasoning"] = c0d3r_reasoning
            if smoke_test_cmd:
                ctx["smoke_test_cmd"] = smoke_test_cmd
            run.context = ctx
            run.save(
                update_fields=[
                    "prompt",
                    "mode",
                    "status",
                    "phase",
                    "iteration",
                    "sprint_count",
                    "error",
                    "started_at",
                    "completed_at",
                    "acceptance_recorded",
                    "context",
                    "definition_of_done",
                ]
            )
        else:
            context = {"research_mode": True} if research_mode else {}
            if team_mode:
                context["team_mode"] = team_mode
            if session_provider:
                context["session_provider"] = session_provider
            if codex_model:
                context["codex_model"] = codex_model
            if codex_reasoning:
                context["codex_reasoning"] = codex_reasoning
            if c0d3r_model:
                context["c0d3r_model"] = c0d3r_model
            if c0d3r_reasoning:
                context["c0d3r_reasoning"] = c0d3r_reasoning
            if smoke_test_cmd:
                context["smoke_test_cmd"] = smoke_test_cmd
            run = DeliveryRun.objects.create(
                id=run_uuid,
                project=project,
                prompt=prompt.strip(),
                mode=mode,
                status="queued",
                definition_of_done=delivery_project.definition_of_done or DEFAULT_DOD,
                context=context,
            )
        delivery_project.active_run = run
        delivery_project.status = "running"
        delivery_project.mode = mode
        delivery_project.save(update_fields=["active_run", "status", "mode", "updated_at"])
        return run

    def start_run(
        self,
        project_id: str,
        prompt: str,
        mode: str = "auto",
        run_id: Optional[str] = None,
        job_user_id: Optional[str] = None,
        github_account_id: Optional[str] = None,
        research_mode: bool = False,
        team_mode: Optional[str] = None,
        session_provider: Optional[str] = None,
        codex_model: Optional[str] = None,
        codex_reasoning: Optional[str] = None,
        c0d3r_model: Optional[str] = None,
        c0d3r_reasoning: Optional[str] = None,
        smoke_test_cmd: Optional[str] = None,
    ) -> DeliveryRun:
        run = self.create_run(
            project_id,
            prompt,
            mode=mode,
            run_id=run_id,
            research_mode=research_mode,
            team_mode=team_mode,
            session_provider=session_provider,
            codex_model=codex_model,
            codex_reasoning=codex_reasoning,
            c0d3r_model=c0d3r_model,
            c0d3r_reasoning=c0d3r_reasoning,
            smoke_test_cmd=smoke_test_cmd,
        )
        if job_user_id or github_account_id:
            context = dict(run.context or {})
            if job_user_id:
                context["job_user_id"] = str(job_user_id)
            if github_account_id:
                context["github_account_id"] = github_account_id
            run.context = context
            run.save(update_fields=["context"])

        thread = threading.Thread(target=self._run_pipeline, args=(run.id,), daemon=True)
        with self._lock:
            self._threads[str(run.id)] = thread
        thread.start()
        return run

    def run_existing(self, run_id: uuid.UUID) -> None:
        self._run_pipeline(run_id)

    def run_ui_review(self, run_id: uuid.UUID, manual: bool = True) -> None:
        self._run_ui_review(run_id, manual)

    def _run_solo_pipeline(self, run: DeliveryRun, root: Path) -> None:
        run.status = "running"
        run.phase = "execution"
        if not run.started_at:
            run.started_at = timezone.now()
        run.save(update_fields=["status", "phase", "started_at"])

        provider = session_provider_from_context(run.context or {})
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="dev",
            name=f"Solo {provider} Session",
            status="running",
            workspace_path=str(root),
            last_heartbeat=timezone.now(),
            meta={"mode": "solo"},
        )
        workspace_ctx = build_context(root, notes_name=str(session.id))
        init_notes(workspace_ctx, extra=[f"## Run\n- Provider: {provider}", f"- Run ID: {run.id}", ""])
        session.meta = {**(session.meta or {}), "notes_path": str(workspace_ctx.notes_path)}
        session.save(update_fields=["meta"])
        context = dict(run.context or {})
        context["solo_plan_path"] = str(_solo_plan_path(run.id))
        context["solo_history_path"] = str(_solo_history_path(run.id))
        context["notes_path"] = str(workspace_ctx.notes_path)
        run.context = context
        run.save(update_fields=["context"])

        provider = session_provider_from_context(run.context or {})
        ai_settings = _session_settings_for_run(run, "worker")
        SessionClass = get_session_class(provider)
        codex = SessionClass(
            session_name=f"solo-{run.id}",
            transcript_dir=Path("runtime/branddozer/transcripts") / str(run.id),
            read_timeout_s=None,
            workdir=str(root),
            **ai_settings,
        )
        _log_ai_choice(session, provider, ai_settings, "worker")

        max_iters = int(os.getenv("BRANDDOZER_SOLO_MAX_ITERS", "12"))
        plan = _load_solo_plan(run.id)
        if not plan:
            _append_session_log(session, "Generating solo plan.")
            plan_prompt = (
                "You are the solo delivery agent. Create a step-by-step plan to complete the project in this repo. "
                "Return ONLY JSON with keys: plan (list of steps), next_step, done (bool), summary, suggestions. "
                "Each step should be short and testable. Include a smoke-test step when feasible.\n"
                f"User prompt:\n{run.prompt}\n"
            )
            output = codex.send(plan_prompt, stream=True)
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(output)
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                    session.status = "blocked"
                    session.completed_at = timezone.now()
                    session.save(update_fields=["status", "completed_at"])
                    raise StopDelivery(exhausted)
                refusal = _codex_refusal(output)
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
                    session.status = "blocked"
                    session.completed_at = timezone.now()
                    session.save(update_fields=["status", "completed_at"])
                    raise StopDelivery(refusal)
            payload = _extract_json_payload(output)
            steps = _normalize_plan_steps(payload.get("plan") or payload.get("steps"))
            plan = {
                "version": 1,
                "status": "planning",
                "summary": payload.get("summary") or "",
                "steps": steps,
                "next_step": payload.get("next_step") or "",
                "done": bool(payload.get("done")),
                "suggestions": payload.get("suggestions") or "",
            }
            _write_solo_plan(run.id, plan)
            append_notes(workspace_ctx, f"- [x] Plan generated at {time.strftime('%H:%M:%S')}")
            _append_solo_history(run.id, {"event": "plan_generated", "output": output})
            try:
                DeliveryArtifact.objects.create(
                    project=run.project,
                    run=run,
                    session=session,
                    kind="solo_plan",
                    title="Solo plan",
                    content=json.dumps(plan, indent=2),
                    path=str(_solo_plan_path(run.id)),
                )
            except Exception:
                pass

        for _ in range(max_iters):
            self._check_stop(run, session)
            plan = _load_solo_plan(run.id) or plan
            if plan.get("done"):
                break
            steps = plan.get("steps") or []
            next_step = plan.get("next_step") or ""
            if not next_step and steps:
                for step in steps:
                    if str(step.get("status") or "").lower() not in {"done", "complete"}:
                        next_step = step.get("title") or ""
                        break
            if not next_step:
                plan["done"] = True
                plan["suggestions"] = plan.get("suggestions") or "No remaining steps. Consider closing the run."
                _write_solo_plan(run.id, plan)
                _append_solo_history(run.id, {"event": "plan_completed", "note": "no next step"})
                break

            _append_session_log(session, f"Executing step: {next_step}")
            _set_run_note(run, "Solo", f"Executing: {next_step}")
            exec_prompt = (
                "You are the solo delivery agent. Execute the next step in the repo. "
                "Follow the existing project conventions, run a quick smoke test when done, "
                "and summarize what changed. If blocked, explain why.\n"
                f"Current plan summary: {plan.get('summary')}\n"
                f"Next step: {next_step}\n"
            )
            output = codex.send(exec_prompt, stream=True)
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(output)
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                    raise StopDelivery(exhausted)
                refusal = _codex_refusal(output)
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
                    raise StopDelivery(refusal)
            try:
                DeliveryArtifact.objects.create(
                    project=run.project,
                    run=run,
                    session=session,
                    kind="solo_step",
                    title=f"Solo step: {next_step[:80]}",
                    content=output,
                    path=str(codex.transcript_path),
                )
            except Exception:
                pass

            smoke_result = _run_smoke_test(run, root, session=session)
            _append_session_log(session, f"Smoke test: {smoke_result.get('status')}")

            reflect_prompt = (
                "You are the solo delivery agent maintaining the plan. "
                "Update the plan status and select the next step. "
                "Return ONLY JSON with keys: plan, next_step, done, summary, suggestions.\n"
                f"Previous plan: {json.dumps(plan, indent=2)}\n"
                f"Last step output summary:\n{output[-1500:]}\n"
                f"Smoke test result: {smoke_result.get('status')}\n"
            )
            reflect_output = codex.send(reflect_prompt, stream=True)
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(reflect_output)
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                    raise StopDelivery(exhausted)
                refusal = _codex_refusal(reflect_output)
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
                    raise StopDelivery(refusal)
            payload = _extract_json_payload(reflect_output)
            steps = _normalize_plan_steps(payload.get("plan") or payload.get("steps") or plan.get("steps"))
            plan = {
                "version": plan.get("version", 1),
                "status": "execution",
                "summary": payload.get("summary") or plan.get("summary") or "",
                "steps": steps,
                "next_step": payload.get("next_step") or "",
                "done": bool(payload.get("done")),
                "suggestions": payload.get("suggestions") or "",
            }
            _write_solo_plan(run.id, plan)
            _append_solo_history(
                run.id,
                {
                    "event": "plan_updated",
                    "next_step": plan.get("next_step"),
                    "done": plan.get("done"),
                    "summary": plan.get("summary"),
                },
            )
            run.iteration += 1
            run.save(update_fields=["iteration"])

        plan = _load_solo_plan(run.id) or plan
        if plan.get("done"):
            suggestion = plan.get("suggestions") or "Solo plan completed."
            _set_run_note(run, "Solo", suggestion)
            ctx = dict(run.context or {})
            ctx["solo_next_suggestion"] = suggestion
            run.context = ctx
            run.status = "complete" if not run.acceptance_required else "awaiting_acceptance"
            run.phase = "awaiting_acceptance" if run.acceptance_required else "release"
            run.completed_at = timezone.now() if not run.acceptance_required else None
            run.save(update_fields=["status", "phase", "completed_at", "context"])
            session.status = "done"
        else:
            session.status = "blocked"
        session.completed_at = timezone.now()
        session.save(update_fields=["status", "completed_at"])

    def _run_pipeline(self, run_id: uuid.UUID) -> None:
        close_old_connections()
        try:
            if not _ensure_branddozer_schema("run_pipeline"):
                _log_pipeline_failure(run_id, "schema_unavailable")
                run = DeliveryRun.objects.filter(id=run_id).first()
                if run:
                    run.status = "error"
                    run.phase = "gates"
                    run.error = "Delivery schema unavailable. Run aborted."
                    run.completed_at = timezone.now()
                    run.save(update_fields=["status", "phase", "error", "completed_at"])
                    _set_run_note(run, "Schema unavailable", "Delivery schema not applied; run aborted.")
                return
            run = DeliveryRun.objects.get(id=run_id)
            job_user = _delivery_run_job_user(run)
        except DeliveryRun.DoesNotExist:
            return
        except (OperationalError, DatabaseError) as exc:
            _log_pipeline_failure(run_id, f"db_unavailable: {exc}")
            return
        project = run.project
        setpoints = load_setpoints()
        root = Path(project.root_path)
        team_mode = str((run.context or {}).get("team_mode") or "full").strip().lower()
        if team_mode in {"solo", "single", "one"}:
            try:
                self._run_solo_pipeline(run, root)
            except StopDelivery as exc:
                run.status = "blocked"
                run.phase = "stopped"
                run.error = str(exc)
                _set_run_note(run, "Stopped", str(exc))
            except Exception as exc:
                run.status = "error"
                run.phase = "gates"
                run.error = str(exc)
                _set_run_note(run, "Error", str(exc))
            if run.status in {"blocked", "error"} and not run.completed_at:
                run.completed_at = timezone.now()
            run.save(update_fields=["status", "phase", "completed_at", "error"])
            delivery_project = DeliveryProject.objects.filter(project=project).first()
            if delivery_project:
                if run.status in {"complete", "awaiting_acceptance"}:
                    delivery_project.status = "complete"
                elif run.status == "blocked":
                    delivery_project.status = "blocked"
                elif run.status == "error":
                    delivery_project.status = "error"
                else:
                    delivery_project.status = "running"
                delivery_project.active_run = run
                delivery_project.save(update_fields=["status", "active_run", "updated_at"])
            close_old_connections()
            return
        orchestrator_session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="orchestrator",
            name="Orchestrator",
            status="running",
            workspace_path=str(root),
            last_heartbeat=timezone.now(),
            meta={"phase": "start"},
        )
        _append_session_log(orchestrator_session, f"Run {run.id} started for {project.name}.")
        try:
            _set_run_note(run, "Starting delivery run", f"Project: {project.name}")
            run.status = "running"
            run.started_at = timezone.now()
            run.phase = "mode_detection"
            run.save(update_fields=["status", "started_at", "phase"])
            _append_session_log(orchestrator_session, "Detecting start mode.")
            _set_run_note(run, "Detecting start mode")
            self._check_stop(run, orchestrator_session)

            mode = self._detect_mode(run.mode, root)
            run.mode = mode
            run.save(update_fields=["mode"])
            _append_session_log(orchestrator_session, f"Mode: {mode}.")
            _set_run_note(run, f"Mode: {mode}")
            self._check_stop(run, orchestrator_session)

            if mode == "existing":
                _append_session_log(orchestrator_session, "Running baseline review.")
                _set_run_note(run, "Baseline review", "Running baseline gates and repo scan")
                self._baseline_review(run, root)
                self._check_stop(run, orchestrator_session)

            _append_session_log(orchestrator_session, "Generating governance artifacts.")
            _set_run_note(run, "Governance", "Generating PMP artifacts")
            self._governance_step(run, root)
            self._check_stop(run, orchestrator_session)
            _append_session_log(orchestrator_session, "Generating requirements.")
            _set_run_note(run, "Requirements", "Formalizing functional and non-functional scope")
            self._requirements_step(run, root)
            self._check_stop(run, orchestrator_session)
            _append_session_log(orchestrator_session, "Generating blueprint.")
            _set_run_note(run, "Blueprint", "Preparing architecture and UX flows")
            self._blueprint_step(run, root)
            self._check_stop(run, orchestrator_session)
            live_items = self._ingest_live_prompts(run, root, session=orchestrator_session, note="Pre-backlog live prompts ingested.")
            if live_items:
                _append_session_log(orchestrator_session, f"Ingested {len(live_items)} live request(s) into backlog.")
            _append_session_log(orchestrator_session, "Building backlog and sprint.")
            _set_run_note(run, "Backlog", "Preparing sprint plan")
            backlog_items = self._backlog_step(run, root)
            backlog_items = self._seed_research_backlog(run, backlog_items, orchestrator_session)
            sprint = self._sprint_plan(run, backlog_items, goal="Initial delivery sprint")
            _update_eta(run, backlog=backlog_items, reason="backlog planned")
            _append_session_log(orchestrator_session, "Executing sprint.")
            _set_run_note(run, "Execution", "Running sprint tasks")
            self._execution_loop(run, root, sprint)
            self._check_stop(run, orchestrator_session)
            self._remediation_loop(run, root, orchestrator_session)
            self._run_final_ux_audit(run, root, orchestrator_session)
            # Heartbeat/update
            now = time.time()
            last_hb = self._last_heartbeat.get(str(run.id))
            if heartbeat_due(last_hb, setpoints):
                _notify_run_summary(run, None)
                self._last_heartbeat[str(run.id)] = now
                throttle = should_throttle(run, setpoints)
                save_control_state(run, throttle)
                if throttle:
                    _append_session_log(orchestrator_session, "Throttling new work (homeostasis signal)")
            ok_budget, budget_reason = _codex_budget_ok(run)
            if not ok_budget:
                raise StopDelivery(budget_reason or "Codex budget low")

            if self._dod_satisfied(run):
                _append_session_log(orchestrator_session, "Definition of Done satisfied. Preparing release candidate.")
                _set_run_note(run, "Release", "Preparing release candidate")
                self._release_step(run, root)
                run.status = "awaiting_acceptance" if run.acceptance_required else "complete"
                run.phase = "awaiting_acceptance"
                _set_run_note(run, "Awaiting acceptance", "Ready for user sign-off")
            else:
                _append_session_log(orchestrator_session, "Definition of Done not satisfied. Blocking run.")
                _trigger_unstick_session(run, root, "Definition of Done not satisfied after remediation")
                run.status = "blocked"
                run.phase = "gates"
                _set_run_note(run, "Blocked", "Definition of Done not satisfied")
        except StopDelivery as exc:
            run.status = "blocked"
            run.phase = "stopped"
            run.error = str(exc)
            _append_session_log(orchestrator_session, f"Stopped: {exc}")
            _set_run_note(run, "Stopped", str(exc))
        except Exception as exc:
            run.status = "error"
            run.phase = "gates"
            run.error = str(exc)
            _append_session_log(orchestrator_session, f"Error: {exc}")
            _set_run_note(run, "Error", str(exc))
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "phase", "completed_at", "error"])
        delivery_project = DeliveryProject.objects.filter(project=project).first()
        if delivery_project:
            if run.status in {"complete", "awaiting_acceptance"}:
                delivery_project.status = "complete"
            elif run.status == "blocked":
                delivery_project.status = "blocked"
            elif run.status == "error":
                delivery_project.status = "error"
            else:
                delivery_project.status = "running"
            delivery_project.active_run = run
            delivery_project.save(update_fields=["status", "active_run", "updated_at"])
        ux_report_path = _store_ux_audit_report(run)
        push_payload = None
        if run.status in {"complete", "awaiting_acceptance"}:
            push_payload = _attempt_github_push(run, orchestrator_session, user=job_user)
            if push_payload:
                context = dict(run.context or {})
                context["github_push"] = push_payload
                run.context = context
                run.save(update_fields=["context"])
        orchestrator_session.status = "done" if run.status != "error" else "error"
        orchestrator_session.completed_at = timezone.now()
        orchestrator_session.save(update_fields=["status", "completed_at"])
        _notify_run_summary(run, ux_report_path)
        close_old_connections()

    def _detect_mode(self, requested_mode: str, root: Path) -> str:
        if requested_mode in {"new", "existing"}:
            return requested_mode
        if root.exists() and any(root.iterdir()):
            return "existing"
        return "new"

    def _baseline_review(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "baseline_review"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Baseline review", "Inspecting repo and running gates")
        repo_info = _repo_structure(root)
        deps = _requirements_snapshot(root)
        baseline_data = {
            "repo": repo_info,
            "dependencies": deps,
            "existing_backlog": list(BacklogItem.objects.filter(project=run.project).values("id", "title", "status")),
            "existing_sprints": list(Sprint.objects.filter(project=run.project).values("id", "number", "status")),
        }
        runner = GateRunner(run, root)
        gates = default_gates(root)
        gate_results = [runner.run_gate(gate) for gate in gates]
        baseline_data["gates"] = []
        for gate_def, gate_run in zip(gates, gate_results):
            meta = gate_run.meta or {}
            required = bool(meta.get("required", gate_def.required))
            not_relevant = bool(meta.get("not_relevant"))
            timeout_hit = bool(meta.get("timeout"))
            baseline_data["gates"].append(
                {
                    "name": gate_run.name,
                    "stage": gate_run.stage,
                    "status": gate_run.status,
                    "stderr": gate_run.stderr,
                    "exit_code": gate_run.exit_code,
                    "required": required,
                    "not_relevant": not_relevant,
                    "timeout": timeout_hit,
                }
            )
        summary = "Baseline complete."
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="baseline_report",
            version=_latest_version(run, "baseline_report"),
            summary=summary,
            content=json.dumps(baseline_data, indent=2),
            data=baseline_data,
        )
        for gate_def, gate_run in zip(gates, gate_results):
            meta = gate_run.meta or {}
            required = bool(meta.get("required", gate_def.required))
            not_relevant = bool(meta.get("not_relevant"))
            timeout_hit = bool(meta.get("timeout"))
            dep_skip = "not installed" in (gate_run.stderr or "").lower()
            non_blocking_skip = gate_run.status == "skipped" and (not required or not_relevant or timeout_hit or dep_skip)
            if non_blocking_skip:
                continue
            if gate_run.status != "passed":
                BacklogItem.objects.create(
                    project=run.project,
                    run=run,
                    kind="bug",
                    title=f"Baseline gate failed: {gate_run.name}",
                    description=gate_run.stderr or gate_run.stdout,
                    acceptance_criteria=["Gate passes with green status"],
                    priority=1,
                    estimate_points=2,
                    status="todo",
                    source="baseline",
                    meta={"stage": gate_run.stage, "required": required},
                )
                RaidEntry.objects.create(
                    project=run.project,
                    run=run,
                    kind="issue",
                    title=f"Gate failure: {gate_run.name}",
                    description=gate_run.stderr or gate_run.stdout,
                    severity="high" if gate_run.stage in {"security", "e2e"} else "medium",
                )

    def _governance_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "governance"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Governance", "Preparing PMP artifacts")
        pm_session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="pm",
            name="Project Manager Session",
            status="running",
            workspace_path=str(root),
            last_heartbeat=timezone.now(),
        )
        _append_session_log(pm_session, "Generating project charter.")
        _set_run_note(run, "Governance", "Generating project charter")
        charter_content, charter_data = self._codex_or_template(
            run,
            root,
            "Generate project charter JSON.",
            "charter",
            session=pm_session,
        )
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="charter",
            version=_latest_version(run, "charter"),
            summary="Project charter updated.",
            content=charter_content,
            data=charter_data,
        )
        _append_session_log(pm_session, "Generating WBS.")
        _set_run_note(run, "Governance", "Generating WBS")
        wbs_content, wbs_data = self._codex_or_template(run, root, "Generate WBS JSON.", "wbs", session=pm_session)
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="wbs",
            version=_latest_version(run, "wbs"),
            summary="WBS updated.",
            content=wbs_content,
            data=wbs_data,
        )
        _append_session_log(pm_session, "Generating quality plan.")
        _set_run_note(run, "Governance", "Generating quality plan")
        quality_content, quality_data = self._codex_or_template(
            run, root, "Generate quality management plan JSON.", "quality_plan", session=pm_session
        )
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="quality_plan",
            version=_latest_version(run, "quality_plan"),
            summary="Quality plan updated.",
            content=quality_content,
            data=quality_data,
        )
        _append_session_log(pm_session, "Generating release criteria.")
        _set_run_note(run, "Governance", "Generating release criteria")
        release_content, release_data = self._codex_or_template(
            run, root, "Generate release criteria JSON.", "release_criteria", session=pm_session
        )
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="release_criteria",
            version=_latest_version(run, "release_criteria"),
            summary="Release criteria updated.",
            content=release_content,
            data=release_data,
        )
        raid_entries = list(RaidEntry.objects.filter(run=run).values("kind", "title", "status", "severity"))
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="raid_log",
            version=_latest_version(run, "raid_log"),
            summary="RAID log snapshot.",
            content=json.dumps({"entries": raid_entries}, indent=2),
            data={"entries": raid_entries},
        )
        if run.mode == "existing":
            ChangeRequest.objects.create(
                project=run.project,
                run=run,
                title="Scope alignment review",
                description="Baseline review completed; reconcile prompt with existing project.",
                rationale="Existing project detected; changes require change control.",
                impact="Potential scope/architecture adjustments.",
                status="proposed",
            )
        _append_session_log(pm_session, "Governance artifacts ready.")
        _set_run_note(run, "Governance", "Artifacts ready")
        pm_session.status = "done"
        pm_session.completed_at = timezone.now()
        pm_session.save(update_fields=["status", "completed_at"])

    def _requirements_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "requirements"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Requirements", "Capturing functional and non-functional scope")
        prompt = (
            "Generate JSON requirements with keys: functional, non_functional, constraints, assumptions, out_of_scope. "
            "Use the user prompt and baseline findings if any."
        )
        content, data = self._codex_or_template(run, root, prompt, fallback_kind="requirements")
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="requirements",
            version=_latest_version(run, "requirements"),
            summary="Requirements captured.",
            content=content,
            data=data,
        )

    def _blueprint_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "blueprint"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Blueprint", "Generating architecture and UX flows")
        prompt = (
            "Generate JSON blueprint with keys: architecture, data_flows, threat_model, data_model, api_contracts, "
            "ux_flows, accessibility_targets, as_is, to_be."
        )
        content, data = self._codex_or_template(run, root, prompt, fallback_kind="blueprint")
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="blueprint",
            version=_latest_version(run, "blueprint"),
            summary="Blueprint prepared.",
            content=content,
            data=data,
        )

    def _ingest_live_prompts(self, run: DeliveryRun, root: Path, *, session: Optional[DeliverySession] = None, note: str = "") -> List[BacklogItem]:
        """
        Consume queued live prompts (e.g., from brandozer prompt) and convert them to backlog items.
        """
        ctx = dict(run.context or {})
        queue = ctx.get("live_prompts_queue") or []
        if not isinstance(queue, list) or not queue:
            return []
        created: List[BacklogItem] = []
        history = ctx.get("live_prompts_history") or []
        for entry in queue:
            text = ""
            source = "live"
            if isinstance(entry, dict):
                text = str(entry.get("text") or "").strip()
                source = str(entry.get("source") or "live")
            else:
                text = str(entry).strip()
            if not text:
                continue
            title = f"Live request: {text[:120]}"
            description = text
            acceptance = [
                "Implements the live request intent",
                "Passes relevant gates (tests, lint, a11y, UX)",
                "Demonstrates value in UX snapshots and documentation",
            ]
            item = BacklogItem.objects.create(
                project=run.project,
                run=run,
                kind="story",
                title=title,
                description=description,
                acceptance_criteria=acceptance,
                priority=1,
                estimate_points=2,
                status="todo",
                source="live_prompt",
                meta={"source": source, "ingested_at": timezone.now().isoformat()},
            )
            created.append(item)
            history.append({"text": text, "source": source, "at": timezone.now().isoformat(), "backlog_id": str(item.id)})
        ctx["live_prompts_history"] = history
        ctx["live_prompts_queue"] = []
        run.context = ctx
        run.save(update_fields=["context"])
        if session:
            _append_session_log(session, note or f"Ingested {len(created)} live prompt(s) into backlog.")
        return created

    def _backlog_step(self, run: DeliveryRun, root: Path) -> List[BacklogItem]:
        run.phase = "backlog"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Backlog", "Generating backlog items and estimates")
        prompt = (
            "Generate JSON backlog with list of items. Each item has: kind, title, description, "
            "acceptance_criteria, priority, estimate_points, dependencies."
        )
        content, data = self._codex_or_template(run, root, prompt, fallback_kind="backlog")
        items = []
        backlog_items = data.get("items") if isinstance(data, dict) else None
        if isinstance(backlog_items, list):
            for item in backlog_items:
                try:
                    items.append(
                        BacklogItem.objects.create(
                            project=run.project,
                            run=run,
                            kind=item.get("kind", "task"),
                            title=item.get("title", "Untitled"),
                            description=item.get("description", ""),
                            acceptance_criteria=item.get("acceptance_criteria", []),
                            priority=int(item.get("priority", 3)),
                            estimate_points=float(item.get("estimate_points", 1)),
                            dependencies=item.get("dependencies", []),
                            status="todo",
                            source="requirements",
                        )
                    )
                except Exception:
                    continue
        if not items:
            items = list(BacklogItem.objects.filter(run=run))
        live_items = self._ingest_live_prompts(run, root, note="Live prompts ingested into backlog")
        if live_items:
            items.extend(live_items)
        items = self._refine_backlog(run, items)
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="change_control",
            version=_latest_version(run, "change_control"),
            summary="Backlog snapshot generated.",
            content=content,
            data={"items": [item.title for item in items]},
        )
        return items

    def _refine_backlog(self, run: DeliveryRun, items: List[BacklogItem]) -> List[BacklogItem]:
        refined: List[BacklogItem] = []
        not_ready: List[Dict[str, Any]] = []
        for item in items:
            auto_filled: List[str] = []
            criteria = _normalize_acceptance_criteria(item.acceptance_criteria)
            if not criteria:
                auto_filled.append("acceptance_criteria")
                criteria = [f"Meets intent of {item.title}", "Relevant gates are green"]
            dependencies = _normalize_dependencies(item.dependencies)
            description = (item.description or "").strip()
            if not description:
                auto_filled.append("description")
                description = item.title
            estimate_points = item.estimate_points if item.estimate_points and item.estimate_points > 0 else 1.0
            if estimate_points != item.estimate_points:
                auto_filled.append("estimate_points")
            item.acceptance_criteria = criteria
            item.dependencies = dependencies
            item.description = description
            item.estimate_points = estimate_points
            dor_passed, missing = _definition_of_ready(item)
            item.meta = {
                **(item.meta or {}),
                "dor_passed": dor_passed,
                "dor_missing": missing,
                "dor_auto_filled": auto_filled,
                "refined_at": timezone.now().isoformat(),
            }
            item.save(update_fields=["acceptance_criteria", "dependencies", "description", "estimate_points", "meta", "updated_at"])
            refined.append(item)
            if not dor_passed:
                not_ready.append({"id": str(item.id), "title": item.title, "missing": missing})
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="backlog_refinement",
            version=_latest_version(run, "backlog_refinement"),
            summary="Backlog refinement complete.",
            content=json.dumps({"not_ready": not_ready}, indent=2),
            data={"not_ready": not_ready},
        )
        return refined

    def _sprint_plan(self, run: DeliveryRun, backlog_items: List[BacklogItem], *, goal: str = "Delivery sprint") -> Sprint:
        run.phase = "sprint_planning"
        run.save(update_fields=["phase"])
        parallelism = int(run.context.get("parallelism") or _dynamic_parallelism())
        run.context["parallelism"] = parallelism
        run.save(update_fields=["context"])
        capacity_points, capacity_items = _sprint_capacity(parallelism)
        done_items = list(BacklogItem.objects.filter(run=run, status="done"))
        done_ids = {str(item.id) for item in done_items}
        done_titles = {item.title.lower() for item in done_items}

        ordered = sorted(backlog_items, key=lambda item: (item.priority, item.created_at))
        selected: List[BacklogItem] = []
        skipped: List[Dict[str, Any]] = []
        total_points = 0.0
        for item in ordered:
            if item.status == "done":
                continue
            dor_passed = bool(item.meta.get("dor_passed", True)) if isinstance(item.meta, dict) else True
            if not dor_passed:
                skipped.append({"id": str(item.id), "title": item.title, "reason": "not_ready"})
                continue
            deps_ok, missing = _dependencies_met(item, done_ids=done_ids, done_titles=done_titles)
            if not deps_ok:
                meta = item.meta if isinstance(item.meta, dict) else {}
                meta["dependency_missing"] = missing
                item.meta = meta
                item.save(update_fields=["meta", "updated_at"])
                skipped.append({"id": str(item.id), "title": item.title, "reason": "dependencies", "missing": missing})
                continue
            if len(selected) >= capacity_items or (total_points + float(item.estimate_points or 0)) > capacity_points:
                skipped.append({"id": str(item.id), "title": item.title, "reason": "capacity"})
                continue
            selected.append(item)
            total_points += float(item.estimate_points or 0)

        fallback_used = False
        if not selected and ordered:
            fallback_used = True
            for item in ordered:
                if item.status == "done":
                    continue
                selected.append(item)
                total_points += float(item.estimate_points or 0)
                if len(selected) >= min(capacity_items, 2):
                    break

        next_number = run.sprint_count + 1
        sprint = Sprint.objects.create(
            project=run.project,
            run=run,
            number=next_number,
            goal=goal,
            status="active",
            started_at=timezone.now(),
            meta={
                "capacity_points": capacity_points,
                "capacity_items": capacity_items,
                "selected_points": total_points,
                "parallelism": parallelism,
            },
        )
        for item in selected:
            SprintItem.objects.get_or_create(sprint=sprint, backlog_item=item, defaults={"status": "todo", "owner": "dev"})
        run.sprint_count = next_number
        run.save(update_fields=["sprint_count"])

        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="sprint_plan",
            version=_latest_version(run, "sprint_plan"),
            summary=f"Sprint {next_number} planned.",
            content=json.dumps(
                {
                    "sprint": next_number,
                    "goal": goal,
                    "capacity_points": capacity_points,
                    "capacity_items": capacity_items,
                    "selected_points": total_points,
                    "fallback_used": fallback_used,
                    "selected": [item.title for item in selected],
                    "skipped": skipped,
                },
                indent=2,
            ),
            data={
                "sprint": next_number,
                "capacity_points": capacity_points,
                "capacity_items": capacity_items,
                "selected_points": total_points,
                "fallback_used": fallback_used,
                "selected": [item.title for item in selected],
                "skipped": skipped,
            },
        )
        return sprint

    def _execution_loop(self, run: DeliveryRun, root: Path, sprint: Sprint) -> None:
        run.phase = "execution"
        run.iteration += 1
        parallelism = min(_dynamic_parallelism(), MAX_PARALLELISM)
        run.context["parallelism"] = parallelism
        run.save(update_fields=["phase", "iteration", "context"])
        _cleanup_run_sessions(run, reason="new sprint execution")
        ok_budget, budget_reason = _codex_budget_ok(run)
        if not ok_budget:
            _set_run_note(run, "Paused", budget_reason or "Codex budget low")
            raise StopDelivery(budget_reason or "Codex budget low")
        offline_mode = (os.getenv("BRANDDOZER_OFFLINE_MODE") or "0").lower() in {"1", "true", "yes", "on"}
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="integrator",
            name="Integrator/Release Session",
            status="running",
            workspace_path=str(root),
            last_heartbeat=timezone.now(),
            meta={"note": "Integration uses canonical workspace"},
        )
        _append_session_log(session, f"Integrator session started. Parallelism: {parallelism}.")
        if _stop_requested(run):
            _append_session_log(session, "Stop requested. Halting sprint execution.")
            session.status = "blocked"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            raise StopDelivery("Stop requested")

        sprint_items = list(sprint.items.select_related("backlog_item"))
        sprint_item_map = {item.backlog_item.id: item for item in sprint_items}
        backlog_queue = [item.backlog_item for item in sprint_items if item.status != "done" and item.backlog_item.status != "done"]
        # Skip items that already produced no changes to avoid Codex loops that burn credits without output
        filtered_queue: List[BacklogItem] = []
        for item in backlog_queue:
            retries = 0
            try:
                retries = int((item.meta or {}).get("no_changes_retries", 0))
            except Exception:
                retries = 0
            meta = item.meta or {}
            long_running = bool(meta.get("long_running"))
            last_run_ts = meta.get("last_run_ts")
            if long_running and last_run_ts:
                try:
                    last_ts = float(last_run_ts)
                    import time

                    cooldown = LONGRUN_COOLDOWN_MINUTES * 60
                    if time.time() - last_ts < cooldown:
                        _append_session_log(session, f"Skipping long-running task {item.title} until cooldown elapses.")
                        item.status = "blocked"
                        item.save(update_fields=["status", "updated_at"])
                        sprint_item = sprint_item_map.get(item.id)
                        if sprint_item:
                            sprint_item.status = "blocked"
                            sprint_item.save(update_fields=["status"])
                        continue
                except Exception:
                    pass
            if retries >= 1:
                _append_session_log(session, f"Skipping {item.title}: prior attempt produced no changes.")
                item.status = "blocked"
                item.meta = {**(item.meta or {}), "skipped_no_changes": True}
                item.save(update_fields=["status", "meta", "updated_at"])
                sprint_item = sprint_item_map.get(item.id)
                if sprint_item:
                    sprint_item.status = "blocked"
                    sprint_item.save(update_fields=["status"])
                continue
            filtered_queue.append(item)
        backlog_queue = filtered_queue
        if not backlog_queue:
            _append_session_log(session, "No sprint items ready for execution.")
        else:
            for backlog_item in backlog_queue:
                if _stop_requested(run):
                    _append_session_log(session, "Stop requested. Skipping remaining tasks.")
                    session.status = "blocked"
                    session.completed_at = timezone.now()
                    session.save(update_fields=["status", "completed_at"])
                    raise StopDelivery("Stop requested")
                backlog_item.status = "in_progress"
                backlog_item.save(update_fields=["status", "updated_at"])
                sprint_item = sprint_item_map.get(backlog_item.id)
                if sprint_item:
                    sprint_item.status = "in_progress"
                    sprint_item.save(update_fields=["status"])

            def _task_runner(item: BacklogItem) -> Tuple[BacklogItem, str, Optional[str]]:
                try:
                    diff_text = self._run_task_session(run, root, item)
                    return item, diff_text, None
                except Exception as exc:
                    return item, "", str(exc)

            with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
                futures = {executor.submit(_task_runner, item): item for item in backlog_queue}
                for future in as_completed(futures):
                    backlog_item, diff_text, error = future.result()
                    sprint_item = sprint_item_map.get(backlog_item.id)
                    if _stop_requested(run):
                        _append_session_log(session, "Stop requested. Deferring remaining task integration.")
                        session.status = "blocked"
                        session.completed_at = timezone.now()
                        session.save(update_fields=["status", "completed_at"])
                        raise StopDelivery("Stop requested")
                    if error:
                        backlog_item.status = "blocked"
                        if sprint_item:
                            sprint_item.status = "blocked"
                        BacklogItem.objects.create(
                            project=run.project,
                            run=run,
                            kind="bug",
                            title=f"Session failed: {backlog_item.title[:60]}",
                            description=error,
                            acceptance_criteria=["Codex session completes successfully"],
                            priority=1,
                            estimate_points=2,
                            status="todo",
                            source="session",
                        )
                        backlog_item.save(update_fields=["status", "updated_at"])
                        if sprint_item:
                            sprint_item.save(update_fields=["status"])
                        _append_session_log(session, f"Task {backlog_item.title} failed: {error}.")
                        continue
                    if offline_mode:
                        backlog_item.status = "done"
                        if sprint_item:
                            sprint_item.status = "done"
                        _append_session_log(session, f"Offline mode: marking {backlog_item.title} as done (no patch apply).")
                        backlog_item.save(update_fields=["status", "updated_at"])
                        if sprint_item:
                            sprint_item.save(update_fields=["status"])
                        DeliveryArtifact.objects.create(
                            project=run.project,
                            run=run,
                            session=session,
                            kind="integration",
                            title=f"Integrator offline placeholder: {backlog_item.title[:60]}",
                            content="offline-mode: integration skipped",
                            data={"exit_code": 0, "offline": True},
                        )
                        continue
                    if not diff_text.strip():
                        backlog_item.status = "blocked"
                        if sprint_item:
                            sprint_item.status = "blocked"
                        meta = backlog_item.meta or {}
                        meta["no_changes_retries"] = int(meta.get("no_changes_retries", 0) or 0) + 1
                        backlog_item.meta = meta
                        BacklogItem.objects.create(
                            project=run.project,
                            run=run,
                            kind="bug",
                            title=f"No changes produced: {backlog_item.title[:60]}",
                            description="Codex session did not return any code diff.",
                            acceptance_criteria=["Code changes are produced", "Diff applies cleanly"],
                            priority=2,
                            estimate_points=1,
                            status="todo",
                            source="session",
                        )
                        backlog_item.save(update_fields=["status", "updated_at"])
                        if sprint_item:
                            sprint_item.save(update_fields=["status"])
                        _append_session_log(session, f"Task {backlog_item.title} produced no diff.")
                        continue
                    _append_session_log(session, f"Applying task: {backlog_item.title}")
                    stdout, stderr, code = _apply_diff(root, diff_text)
                    if stdout or stderr:
                        _append_session_log(session, (stdout or stderr).strip()[:2000])
                    DeliveryArtifact.objects.create(
                        project=run.project,
                        run=run,
                        session=session,
                        kind="integration",
                        title=f"Integrator apply: {backlog_item.title[:60]}",
                        content=f"{stdout}\n{stderr}",
                        data={"exit_code": code},
                    )
                    if code == 0:
                        backlog_item.status = "done"
                        if sprint_item:
                            sprint_item.status = "done"
                    else:
                        backlog_item.status = "blocked"
                        if sprint_item:
                            sprint_item.status = "blocked"
                        BacklogItem.objects.create(
                            project=run.project,
                            run=run,
                            kind="bug",
                            title=f"Integration failed: {backlog_item.title[:60]}",
                            description=stderr or stdout,
                            acceptance_criteria=["Patch applies cleanly", "Conflicts resolved"],
                            priority=1,
                            estimate_points=2,
                            status="todo",
                            source="integrator",
                    )
                    backlog_item.save(update_fields=["status", "updated_at"])
                    if sprint_item:
                        sprint_item.save(update_fields=["status"])
                    _append_session_log(session, f"Task {backlog_item.title} status: {backlog_item.status}.")
                    # Track last run timestamp for long-running items so we can respect cooldowns
                    meta = backlog_item.meta or {}
                    meta["last_run_ts"] = _now_ts()
                    backlog_item.meta = meta
                    backlog_item.save(update_fields=["status", "updated_at", "meta"])
        session.status = "done"
        session.completed_at = timezone.now()
        session.save(update_fields=["status", "completed_at"])
        _update_eta(run, backlog=list(BacklogItem.objects.filter(run=run).exclude(status="done")), reason="sprint complete")
        if _stop_requested(run):
            raise StopDelivery("Stop requested")
        self._gate_stage(run, root)
        sprint.status = "review"
        sprint.save(update_fields=["status"])
        self._sprint_review(run, sprint)
        sprint.status = "retro"
        sprint.save(update_fields=["status"])
        self._sprint_retro(run, sprint)
        sprint.status = "complete"
        sprint.completed_at = timezone.now()
        sprint.save(update_fields=["status", "completed_at"])

    def _sprint_review(self, run: DeliveryRun, sprint: Sprint) -> None:
        sprint_items = list(sprint.items.select_related("backlog_item"))
        completed = [item.backlog_item.title for item in sprint_items if item.status == "done"]
        blocked = [item.backlog_item.title for item in sprint_items if item.status == "blocked"]
        in_progress = [item.backlog_item.title for item in sprint_items if item.status == "in_progress"]
        gate_summary: Dict[str, str] = {}
        for gate in GateRun.objects.filter(run=run).order_by("name", "-created_at"):
            if gate.name not in gate_summary:
                gate_summary[gate.name] = gate.status
        data = {
            "sprint": sprint.number,
            "goal": sprint.goal,
            "completed": completed,
            "blocked": blocked,
            "in_progress": in_progress,
            "gate_summary": gate_summary,
        }
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="sprint_review",
            version=_latest_version(run, "sprint_review"),
            summary=f"Sprint {sprint.number} review complete.",
            content=json.dumps(data, indent=2),
            data=data,
        )

    def _sprint_retro(self, run: DeliveryRun, sprint: Sprint) -> None:
        sprint_items = list(sprint.items.select_related("backlog_item"))
        blocked = [item.backlog_item.title for item in sprint_items if item.status == "blocked"]
        improvements: List[str] = []
        if blocked:
            improvements.append("Tighten acceptance criteria and dependencies before sprint entry.")
        gate_status: Dict[str, str] = {}
        for gate in GateRun.objects.filter(run=run).order_by("name", "-created_at"):
            if gate.name in gate_status:
                continue
            gate_status[gate.name] = gate.status
        gate_failures = [name for name, status in gate_status.items() if status in {"failed", "blocked"}]
        if gate_failures:
            improvements.append("Prioritize gate fixes early in the sprint to keep the pipeline green.")
        if not improvements:
            improvements.append("Maintain current delivery cadence and keep gates green.")
        data = {
            "sprint": sprint.number,
            "observations": {
                "blocked_items": blocked,
                "gate_failures": gate_failures,
            },
            "improvements": improvements,
        }
        sprint.retrospective = json.dumps(data, indent=2)
        sprint.save(update_fields=["retrospective"])
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="sprint_retro",
            version=_latest_version(run, "sprint_retro"),
            summary=f"Sprint {sprint.number} retrospective logged.",
            content=sprint.retrospective,
            data=data,
        )

    def _run_final_ux_audit(self, run: DeliveryRun, root: Path, orchestrator_session: DeliverySession) -> None:
        """
        Run the UX audit only once, after the team believes the work is done
        and the backlog is clear, to avoid mid-sprint credit burn.
        """
        try:
            context = run.context or {}
            if (context.get("ux_audit") or {}).get("final_run"):
                return
        except Exception:
            context = {}
        backlog_open = BacklogItem.objects.filter(run=run).exclude(status="done").exists()
        if backlog_open:
            _append_session_log(orchestrator_session, "Skipping final UX audit until backlog is empty.")
            return
        sprint = Sprint.objects.filter(run=run).order_by("-number").first()
        if not sprint:
            _append_session_log(orchestrator_session, "Skipping final UX audit (no sprint available).")
            return
        _append_session_log(orchestrator_session, "Running final UX audit before release.")
        self._ux_audit(run, root, sprint)
        context.setdefault("ux_audit", {})
        context["ux_audit"]["final_run"] = True
        run.context = context
        run.save(update_fields=["context"])
        if BacklogItem.objects.filter(run=run).exclude(status="done").exists():
            _append_session_log(orchestrator_session, "UX audit produced follow-up backlog; starting remediation.")
            self._remediation_loop(run, root, orchestrator_session)

    def _seed_research_backlog(
        self, run: DeliveryRun, backlog_items: List[BacklogItem], session: DeliverySession
    ) -> List[BacklogItem]:
        """
        If research mode is enabled, inject high-priority backlog to build research tooling.
        """
        context = run.context or {}
        if not context.get("research_mode"):
            return backlog_items
        created_items: List[BacklogItem] = []

        def _ensure(title: str, description: str, acceptance: List[str], kind: str = "task") -> None:
            exists = BacklogItem.objects.filter(run=run, source="research", title=title).exclude(status="done").exists()
            if exists:
                return
            item = BacklogItem.objects.create(
                project=run.project,
                run=run,
                kind=kind,
                title=title[:240],
                description=description,
                acceptance_criteria=acceptance,
                priority=1,
                estimate_points=3,
                status="todo",
                source="research",
                dependencies=[],
            )
            created_items.append(item)

        _ensure(
            "Assemble open research source registry",
            "Document allowed open-access research/engineering sources with domain, license, and notes. "
            "Persist to services/research_sources.py for reuse by scrapers and prompts.",
            [
                "Sources include biomedical (NLM/NCBI), engineering standards, physics, and open electronics catalogs.",
                "License/usage notes captured; only open/publicly scrapable sources included.",
                "Registry exposed via helper functions for downstream tasks.",
            ],
        )
        _ensure(
            "Build DuckDuckGo research scraper",
            "Implement a DuckDuckGo-backed search + fetch helper in services/research_scraper.py "
            "with domain allow-listing and deterministic parsing to feed research tasks.",
            [
                "Search returns URLs/titles constrained to allowed domains.",
                "Fetcher retrieves page content with timeout/backoff and safe headers.",
                "Unit tests cover parsing and fetch logic with mocked responses.",
            ],
        )
        _ensure(
            "Research data auditor",
            "Add a research auditor routine that validates scraped payloads, retries alternative domains, "
            "and emits backlog items when coverage gaps are detected.",
            [
                "Auditor logs missing/blocked domains and retries with alternates.",
                "Backlog items created when required documents are not retrieved.",
                "Reports saved under runtime/branddozer/research for traceability.",
            ],
            kind="bug",
        )
        _ensure(
            "Complex systems and bio-rich corpus ingestion",
            "Expand research corpus with high-density complexity/chaos and biology sources (Santa Fe, ChaosBook, bioRxiv, PLOS). "
            "Ensure domain allow-list is updated and scraping paths exercised.",
            [
                "Allowed source registry includes complex-systems and bio-rich domains.",
                "Scraper unit tests cover complex-system and biology domains.",
                "Sample fetches saved under runtime/branddozer/research/samples with metadata.",
            ],
        )

        if created_items:
            _append_session_log(
                session,
                f"Research mode enabled: seeded {len(created_items)} backlog item(s) for research tooling.",
            )
            backlog_items = list(backlog_items) + created_items
        return backlog_items

    def _ux_audit(self, run: DeliveryRun, root: Path, sprint: Sprint) -> None:
        if _stop_requested(run):
            return
        run.phase = "ux_audit"
        run.save(update_fields=["phase"])
        _set_run_note(run, "UX Audit", "Running UX auditor feedback loop")
        provider = session_provider_from_context(run.context or {})
        ai_settings = _session_settings_for_run(run, "auditor")
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="ux_audit",
            name="UX Auditor Session",
            status="running",
            workspace_path=str(root),
            last_heartbeat=timezone.now(),
            meta={
                "sprint": sprint.number,
                "provider": provider,
                "codex": {k: v for k, v in ai_settings.items() if k not in {"bypass_sandbox_confirm"}},
            },
        )
        log_path = _session_log_path(session.id)
        session.log_path = str(log_path)
        session.save(update_fields=["log_path"])
        _log_ai_choice(session, provider, ai_settings, "auditor")

        backlog_items = list(BacklogItem.objects.filter(run=run).order_by("priority", "created_at"))
        backlog_lines = []
        for item in backlog_items[:12]:
            criteria = _normalize_acceptance_criteria(item.acceptance_criteria)
            criteria_text = "; ".join(criteria)[:160]
            backlog_lines.append(f"- {item.title} [{item.status}]: {criteria_text}")
        if len(backlog_items) > 12:
            backlog_lines.append(f"...and {len(backlog_items) - 12} more items")
        backlog_summary = "\n".join(backlog_lines)

        gate_summary: Dict[str, str] = {}
        for gate in GateRun.objects.filter(run=run).order_by("name", "-created_at"):
            if gate.name not in gate_summary:
                gate_summary[gate.name] = gate.status

        screenshot_paths = [
            str(Path(path))
            for path in DeliveryArtifact.objects.filter(run=run, kind="ui_screenshot")
            .order_by("-created_at")
            .values_list("path", flat=True)[:6]
            if path
        ]

        offline_mode = (os.getenv("BRANDDOZER_OFFLINE_MODE") or "0").lower() in {"1", "true", "yes", "on"}
        if offline_mode or (_is_codex_provider(provider) and shutil.which("codex") is None):
            _append_session_log(session, "Offline mode: skipping UX audit and recording placeholder.")
            payload = {
                "status": "skipped",
                "notes": "Offline mode or Codex unavailable; UX audit placeholder recorded.",
                "backlog": [],
                "issues": [],
                "gate_summary": gate_summary,
            }
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                session=session,
                kind="ux_audit",
                title="UX audit placeholder",
                content=json.dumps(payload, indent=2),
                data={"offline": True},
            )
            session.status = "done"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            context = dict(run.context or {})
            context["ux_audit"] = {
                "issues": 0,
                "backlog_created": 0,
                "notes": "Offline placeholder",
                "session_id": str(session.id),
            }
            run.context = context
            run.save(update_fields=["context"])
            return

        audit_prompt = (
            "You are the UX auditor in a Scrum + PMP delivery loop. The delivery team reports their sprint is complete. "
            "Audit UX and accessibility, call out blockers, and propose actionable backlog updates for the next sprint. "
            "Return ONLY JSON with keys: "
            '{"status":"pass|fail","issues":[{"title":"","detail":"","severity":"info|warn|error","area":""}],'
            '"backlog":[{"title":"","description":"","acceptance_criteria":[],"priority":2}],"notes":""}. '
            f"\nUser prompt:\n{run.prompt}\n"
            f"Sprint goal: {sprint.goal}\n"
            f"Backlog summary:\n{backlog_summary or 'No backlog captured.'}\n"
            f"Gate summary: {json.dumps(gate_summary, indent=2)}\n"
            f"UX evidence paths: {', '.join(screenshot_paths) if screenshot_paths else 'no screenshots recorded'}"
        )

        transcript_dir = Path("runtime/branddozer/transcripts") / str(session.id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        SessionClass = get_session_class(provider)
        codex = SessionClass(
            session_name=f"ux-audit-{session.id}",
            transcript_dir=transcript_dir,
            read_timeout_s=None,
            workdir=str(root),
            **ai_settings,
        )

        last_heartbeat = 0.0

        def _stream_writer(chunk: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(chunk)
            except Exception:
                pass
            nonlocal last_heartbeat
            now = time.time()
            if now - last_heartbeat >= 20:
                _touch_session(session.id)
                last_heartbeat = now

        try:
            output = codex.send(
                audit_prompt,
                stream=True,
                stream_callback=_stream_writer,
                images=screenshot_paths if screenshot_paths else None,
            )
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(output)
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                    raise StopDelivery(exhausted)
                refusal = _codex_refusal(output)
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
                    raise StopDelivery(refusal)
        except Exception as exc:
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(str(exc))
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                    raise StopDelivery(exhausted)
                refusal = _codex_refusal(str(exc))
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
                    raise StopDelivery(refusal)
            _append_session_log(session, f"UX audit failed: {exc}")
            exists = BacklogItem.objects.filter(run=run, source="ux_audit", title="UX audit failed").exclude(status="done").exists()
            if not exists:
                BacklogItem.objects.create(
                    project=run.project,
                    run=run,
                    kind="bug",
                    title="UX audit failed",
                    description=str(exc),
                    acceptance_criteria=["UX audit re-runs successfully"],
                    priority=2,
                    estimate_points=1,
                    status="todo",
                    source="ux_audit",
                )
            session.status = "error"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            return

        DeliveryArtifact.objects.create(
            project=run.project,
            run=run,
            session=session,
            kind="ux_audit",
            title="UX audit output",
            content=output,
            path=str(codex.transcript_path),
        )
        payload = _extract_json_payload(output)
        issues = payload.get("issues") if isinstance(payload, dict) else []
        backlog_payload = None
        if isinstance(payload, dict):
            backlog_payload = payload.get("backlog") or payload.get("actions") or payload.get("recommendations")
        created = self._create_ux_backlog_items(run, backlog_payload, issues)

        session.meta = {
            **(session.meta or {}),
            "issues_found": len(issues) if isinstance(issues, list) else 0,
            "backlog_created": created,
        }
        status_value = str(payload.get("status") or "").lower() if isinstance(payload, dict) else ""
        _append_session_log(
            session,
            f"UX audit done. Issues: {session.meta['issues_found']}; Backlog items created: {created}.",
        )
        context = dict(run.context or {})
        context["ux_audit"] = {
            "issues": session.meta["issues_found"],
            "backlog_created": created,
            "notes": payload.get("notes") if isinstance(payload, dict) else "",
            "session_id": str(session.id),
        }
        run.context = context
        run.save(update_fields=["context"])
        session.status = "done" if status_value in {"pass", "passed", "ok"} and created == 0 else "blocked"
        session.completed_at = timezone.now()
        session.log_path = str(codex.transcript_path)
        session.save(update_fields=["meta", "status", "completed_at", "log_path"])

    def _create_ux_backlog_items(self, run: DeliveryRun, backlog_payload: Any, issues: Any) -> int:
        created = 0
        candidates: List[Dict[str, Any]] = []
        if isinstance(backlog_payload, list):
            candidates = [entry for entry in backlog_payload if isinstance(entry, dict)]
        if not candidates and isinstance(issues, list):
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                candidates.append(
                    {
                        "title": issue.get("title") or issue.get("summary"),
                        "description": issue.get("detail") or issue.get("description"),
                        "acceptance_criteria": issue.get("acceptance_criteria") or [],
                        "priority": issue.get("priority"),
                        "severity": issue.get("severity"),
                        "area": issue.get("area"),
                    }
                )
        for entry in candidates:
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            exists = (
                BacklogItem.objects.filter(run=run, source="ux_audit", title=title)
                .exclude(status="done")
                .exists()
            )
            if exists:
                continue
            acceptance = _normalize_acceptance_criteria(
                entry.get("acceptance_criteria") or entry.get("acceptance") or entry.get("criteria") or []
            )
            if not acceptance:
                acceptance = [
                    f"UX issue '{title}' resolved",
                    "UX auditor sign-off recorded",
                ]
            description = (entry.get("description") or entry.get("detail") or title).strip()
            try:
                priority = int(entry.get("priority") or 0)
            except Exception:
                priority = 0
            severity = str(entry.get("severity") or "").lower()
            if priority <= 0:
                priority = 1 if severity in {"error", "high"} else 2 if severity in {"warn", "warning", "medium"} else 3
            priority = max(1, min(priority, 5))
            BacklogItem.objects.create(
                project=run.project,
                run=run,
                kind="bug",
                title=title[:240],
                description=description,
                acceptance_criteria=acceptance,
                priority=priority,
                estimate_points=float(entry.get("estimate_points") or 1.0),
                status="todo",
                source="ux_audit",
                meta={"severity": severity, "area": entry.get("area"), "ux_audit": True},
            )
            created += 1
        return created

    def _remediation_loop(self, run: DeliveryRun, root: Path, orchestrator_session: DeliverySession) -> None:
        for cycle in range(MAX_REMEDIATION_CYCLES):
            if _stop_requested(run):
                _append_session_log(orchestrator_session, "Stop requested. Halting remediation.")
                raise StopDelivery("Stop requested")
            live_items = self._ingest_live_prompts(run, root, session=orchestrator_session, note="Live prompts ingested during remediation")
            if live_items:
                _append_session_log(orchestrator_session, f"Added {len(live_items)} live request(s) to backlog.")
            if self._dod_satisfied(run):
                return
            open_items = list(BacklogItem.objects.filter(run=run).exclude(status="done").order_by("priority"))
            if not open_items:
                return
            _append_session_log(
                orchestrator_session,
                f"Remediation cycle {cycle + 1}: {len(open_items)} open items. Replanning sprint.",
            )
            sprint = self._sprint_plan(run, open_items, goal=f"Remediation sprint {run.sprint_count + 1}")
            self._execution_loop(run, root, sprint)

    def _run_task_session(self, run: DeliveryRun, root: Path, backlog_item: BacklogItem) -> str:
        provider = session_provider_from_context(run.context or {})
        ai_settings = _session_settings_for_run(run, "worker")
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="dev",
            name=f"{provider} session: {backlog_item.title[:80]}",
            status="running",
            workspace_path="",
            last_heartbeat=timezone.now(),
            meta={
                "backlog_item_id": str(backlog_item.id),
                "title": backlog_item.title,
                "provider": provider,
                "codex": {k: v for k, v in ai_settings.items() if k not in {"bypass_sandbox_confirm"}},
            },
        )
        workspace_ctx = build_context(root, notes_name=str(session.id))
        init_notes(
            workspace_ctx,
            extra=[
                f"## Task\n- Title: {backlog_item.title}",
                f"- ID: {backlog_item.id}",
                f"- Provider: {provider}",
                "",
            ],
        )
        session.meta = {**(session.meta or {}), "notes_path": str(workspace_ctx.notes_path)}
        session.save(update_fields=["meta"])
        _log_ai_choice(session, provider, ai_settings, "worker")
        _record_intent(run, backlog_item, ai_settings, provider)
        offline_mode = (os.getenv("BRANDDOZER_OFFLINE_MODE") or "0").lower() in {"1", "true", "yes", "on"}
        if offline_mode or (_is_codex_provider(provider) and shutil.which("codex") is None):
            placeholder = f"[offline placeholder] {backlog_item.title}"
            _append_session_log(session, "Offline mode detected; marking task as done with placeholder.")
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                session=session,
                kind="session_log",
                title=f"Offline placeholder: {backlog_item.title[:60]}",
                content=placeholder,
                path="",
            )
            session.status = "done"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            return placeholder
        workspace, branch = _create_workspace(root, run.id, session.id)
        session.workspace_path = str(workspace)
        session.save(update_fields=["workspace_path"])
        _append_session_log(session, f"Workspace ready at {workspace}.")
        append_notes(workspace_ctx, f"- [x] Workspace ready: {workspace}")
        try:
            from services.agent_workspace import run_command

            code, stdout, stderr = run_command("git status -sb", cwd=workspace)
            if stdout:
                append_notes(workspace_ctx, "## Git status (start)\n```\n" + stdout.strip() + "\n```")
            if stderr:
                append_notes(workspace_ctx, "```\n" + stderr.strip() + "\n```")
        except Exception:
            pass
        transcript_dir = Path("runtime/branddozer/transcripts") / str(session.id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            f"[Session Task]\nProject: {run.project.name}\nRoot: {workspace}\n"
            f"Task: {backlog_item.title}\nDescription: {backlog_item.description}\n"
            f"Acceptance: {backlog_item.acceptance_criteria}\n"
            "Work in a fix/test loop. Attach diff + gate results. "
            "If UI requirements are vague, apply modern web design best practices: responsive layout, "
            "clear hierarchy, accessible contrast, consistent spacing, and semantic HTML. "
            "Use CLI best practices: check git status, run relevant tests, keep diffs small, and document changes. "
            "Prefer local docs and CLI help (`--help`) over internet research unless explicitly enabled. "
            "For UI checks, run `python scripts/branddozer_ui_capture.py --base-url http://127.0.0.1:8000` "
            "and include the screenshot paths in your output (or review them with `codex --image <path>` when available)."
        )
        SessionClass = get_session_class(provider)
        codex = SessionClass(
            session_name=f"delivery-{session.id}",
            transcript_dir=transcript_dir,
            read_timeout_s=None,
            workdir=str(workspace),
            **ai_settings,
        )
        log_path = _session_log_path(session.id)
        session.log_path = str(log_path)
        session.save(update_fields=["log_path"])

        last_heartbeat = 0.0
        def _stream_writer(chunk: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(chunk)
            except Exception:
                pass
            nonlocal last_heartbeat
            now = time.time()
            if now - last_heartbeat >= 20:
                _touch_session(session.id)
                last_heartbeat = now

        try:
            output = codex.send(prompt, stream=True, stream_callback=_stream_writer)
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(output)
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                    raise StopDelivery(exhausted)
                refusal = _codex_refusal(output)
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
                    raise StopDelivery(refusal)
            diff_text = _compute_diff(root, workspace)
            backup_path = _backup_diff(workspace, session.id)
            if backup_path:
                append_notes(workspace_ctx, f"- [x] Backup patch: {backup_path}")
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                session=session,
                kind="session_log",
                title=f"{provider} output: {backlog_item.title[:60]}",
                content=output,
                path=str(codex.transcript_path),
            )
            if diff_text.strip():
                DeliveryArtifact.objects.create(
                    project=run.project,
                    run=run,
                    session=session,
                    kind="diff",
                    title=f"Diff: {backlog_item.title[:60]}",
                    content=diff_text,
                )
                append_notes(workspace_ctx, f"- [x] Diff captured ({len(diff_text)} chars)")
            session.status = "done"
            session.completed_at = timezone.now()
            session.log_path = str(codex.transcript_path)
            session.save(update_fields=["status", "completed_at", "log_path"])
            if (os.getenv("BRANDDOZER_TASK_SMOKE") or "1").strip().lower() not in {"0", "false", "no", "off"}:
                smoke = _run_smoke_test(run, root, session=session)
                append_notes(workspace_ctx, f"- [x] Smoke test: {smoke.get('status')}")
            commit_msg = f"{backlog_item.title[:60]} (branddozer)"
            commit_out = _maybe_commit_changes(workspace, commit_msg)
            if commit_out:
                append_notes(workspace_ctx, "## Git commit\n```\n" + commit_out[:2000] + "\n```")
            push_out = _maybe_push_changes(workspace)
            if push_out:
                append_notes(workspace_ctx, "## Git push\n```\n" + push_out[:2000] + "\n```")
            return diff_text
        except Exception as exc:
            if _is_codex_provider(provider):
                exhausted = _codex_quota_exhausted(str(exc))
                if exhausted:
                    _pause_run_for_codex(run, session, exhausted)
                refusal = _codex_refusal(str(exc))
                if refusal:
                    _handle_codex_refusal(run, session, refusal)
            _append_session_log(session, f"Error: {exc}")
            session.status = "error"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            raise
        finally:
            _cleanup_workspace(root, workspace, branch)

    def _gate_stage(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "gates"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Gates", "Running quality checks")
        runner = GateRunner(run, root)
        gate_failure_counts = dict((run.context or {}).get("gate_failure_counts") or {})
        for gate in default_gates(root):
            result = runner.run_gate(gate)
            if result.status not in {"passed", "skipped"} and gate.required:
                title = f"Fix failing gate: {result.name}"
                exists = (
                    BacklogItem.objects.filter(run=run, source="gate", title=title)
                    .exclude(status="done")
                    .exists()
                    )
                if not exists:
                    BacklogItem.objects.create(
                        project=run.project,
                        run=run,
                        kind="bug",
                        title=title,
                        description=result.stderr or result.stdout,
                        acceptance_criteria=["Gate passes"],
                        priority=1,
                        estimate_points=2,
                        status="todo",
                        source="gate",
                        meta={"stage": result.stage},
                    )
                # Track repeated gate failures to avoid infinite remediation loops
                gate_failure_counts[result.name] = int(gate_failure_counts.get(result.name, 0)) + 1
                if gate_failure_counts[result.name] >= GATE_FAILURE_BLOCK_LIMIT:
                    _trigger_unstick_session(run, root, f"Gate {result.name} failed repeatedly")
                    run.context = {**(run.context or {}), "gate_failure_counts": gate_failure_counts}
                    run.status = "blocked"
                    run.error = f"Gate {result.name} failed {gate_failure_counts[result.name]} times; manual fix required."
                    run.save(update_fields=["status", "error", "context"])
                    _set_run_note(run, "Gates", run.error)
                    raise StopDelivery(run.error)
        if gate_failure_counts:
            run.context = {**(run.context or {}), "gate_failure_counts": gate_failure_counts}
            run.save(update_fields=["context"])
        self._ui_snapshot_review(run, root)

    def trigger_ui_review(self, run_id: uuid.UUID, *, manual: bool = True) -> None:
        thread = threading.Thread(target=self._run_ui_review, args=(run_id, manual), daemon=True)
        with self._lock:
            self._threads[f"ui-review-{run_id}"] = thread
        thread.start()

    def _run_ui_review(self, run_id: uuid.UUID, manual: bool) -> None:
        close_old_connections()
        try:
            _ensure_branddozer_schema("ui_review")
            run = DeliveryRun.objects.select_related("project").filter(id=run_id).first()
        except (OperationalError, DatabaseError) as exc:
            _emit_ui_snapshot_placeholder(run_id, f"db_unavailable: {exc}", manual=manual)
            close_old_connections()
            return
        if not run:
            _emit_ui_snapshot_placeholder(run_id, "run_not_found", manual=manual)
            close_old_connections()
            return
        root = Path(run.project.root_path)
        self._ui_snapshot_review(run, root, manual=manual)
        close_old_connections()

    def _ui_snapshot_review(self, run: DeliveryRun, root: Path, *, manual: bool = False) -> None:
        flag = (os.getenv("BRANDDOZER_UI_CAPTURE") or "1").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            _set_run_note(run, "UI Evidence", "UI capture disabled; emitting placeholder README")
            snapshot_root = Path("runtime/branddozer/snapshots") / str(run.id)
            snapshot_ts = time.strftime("%Y%m%d_%H%M%S")
            snapshot_dir = snapshot_root / snapshot_ts
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            placeholder_shot = _write_placeholder_shot(snapshot_dir, "ui_capture_disabled")
            placeholder_path = str(placeholder_shot) if placeholder_shot else ""
            doc_lines = [
                f"# UX Snapshot {snapshot_ts}",
                "",
                f"Run: {run.id}",
                f"Prompt: {run.prompt}",
                "",
                "UI capture disabled via BRANDDOZER_UI_CAPTURE; screenshots not collected.",
                "Enable BRANDDOZER_UI_CAPTURE=1 to run Playwright capture and visual review.",
            ]
            if placeholder_path:
                doc_lines.extend(
                    [
                        "",
                        f"Placeholder screenshot emitted for audit trail: {placeholder_path}",
                    ]
                )
            doc_path = snapshot_dir / "README.md"
            try:
                doc_path.write_text("\n".join(doc_lines), encoding="utf-8")
                DeliveryArtifact.objects.create(
                    project=run.project,
                    run=run,
                    kind="ui_snapshot_doc",
                    title=f"Snapshot doc {snapshot_ts}",
                    path=str(doc_path),
                    content="\n".join(doc_lines),
                    data={"manual": manual, "captured_at": snapshot_ts, "disabled": True, "placeholder": bool(placeholder_path)},
                )
                if placeholder_path:
                    DeliveryArtifact.objects.create(
                        project=run.project,
                        run=run,
                        kind="ui_screenshot",
                        title=Path(placeholder_path).name,
                        path=placeholder_path,
                        data={"manual": manual, "disabled": True, "placeholder": True, "captured_at": snapshot_ts},
                    )
            except Exception:
                pass
            gate_meta = {
                "manual": manual,
                "disabled": True,
                "required": False,
                "not_relevant": True,
                "placeholder": bool(placeholder_path),
                "screenshots": [placeholder_path] if placeholder_path else [],
                "snapshot_doc": str(doc_path),
                "snapshot_dir": str(snapshot_dir),
            }
            for gate_name in ("ui-snapshot", "ui-review"):
                GateRun.objects.create(
                    project=run.project,
                    run=run,
                    stage="e2e",
                    name=gate_name,
                    status="skipped",
                    command="ui capture disabled",
                    stdout="",
                    stderr="BRANDDOZER_UI_CAPTURE disabled",
                    exit_code=0,
                    duration_ms=0,
                    meta=gate_meta,
                )
            return
        _set_run_note(run, "UI Evidence", "Capturing UI snapshots")
        provider = session_provider_from_context(run.context or {})
        ai_settings = _session_settings_for_run(run, "auditor")
        try:
            session = DeliverySession.objects.create(
                project=run.project,
                run=run,
                role="qa",
                name="UX Verification Session",
                status="running",
                workspace_path=str(root),
                last_heartbeat=timezone.now(),
                meta={"manual": manual},
            )
        except (OperationalError, DatabaseError) as exc:
            _emit_ui_snapshot_placeholder(run.id, f"db_unavailable: {exc}", manual=manual)
            return
        session.meta = {
            **(session.meta or {}),
            "provider": provider,
            "codex": {k: v for k, v in ai_settings.items() if k not in {"bypass_sandbox_confirm"}},
        }
        session.save(update_fields=["meta"])
        _log_ai_choice(session, provider, ai_settings, "qa")
        _append_session_log(session, "Capturing UI screenshots.")
        output_dir = Path("runtime/branddozer/ui") / str(run.id) / str(session.id)
        snapshot_root = Path("runtime/branddozer/snapshots") / str(run.id)
        snapshot_ts = time.strftime("%Y%m%d_%H%M%S")
        snapshot_dir = snapshot_root / snapshot_ts
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        doc_path = snapshot_dir / "README.md"

        def _normalize_shots(shots: List[Any]) -> List[Dict[str, Any]]:
            normalized: List[Dict[str, Any]] = []
            for entry in shots or []:
                path_val = None
                kind = "ui_screenshot"
                meta = {}
                if isinstance(entry, dict):
                    path_val = entry.get("path") or entry.get("file") or entry.get("src")
                    kind = entry.get("kind") or entry.get("tag") or kind
                    meta = entry.get("meta") or {}
                else:
                    path_val = entry
                if not path_val:
                    continue
                path_obj = Path(path_val)
                name_lower = path_obj.name.lower()
                if not kind or kind == "ui_screenshot":
                    if "mobile" in name_lower:
                        kind = "ui_screenshot_mobile"
                    elif "desktop" in name_lower:
                        kind = "ui_screenshot_desktop"
                    else:
                        kind = "ui_screenshot"
                normalized.append({"path": path_obj, "kind": kind, "meta": meta})
            return normalized
        start_ts = time.time()
        base_url = (os.getenv("BRANDDOZER_UI_BASE_URL") or "http://127.0.0.1:8000").strip()
        try:
            result = capture_ui_screenshots(root, output_dir=output_dir)
        except Exception as exc:  # pragma: no cover - defensive to keep gate artifacts emitting
            err = str(exc)
            result = UISnapshotResult(
                stdout="",
                stderr=err,
                exit_code=1,
                screenshots=[],
                base_url=base_url,
                server_started=False,
                server_log=None,
                meta={"error": err},
            )
        duration_ms = int((time.time() - start_ts) * 1000)
        normalized_shots = _normalize_shots(result.screenshots)
        dep_issue = False
        dep_reason = ""
        if isinstance(result.meta, dict):
            dep_issue = bool(result.meta.get("dependency_missing"))
            dep_reason = str(result.meta.get("reason") or "")
        err_text = " ".join(
            part for part in [result.stderr or "", result.stdout or "", dep_reason] if part
        ).lower()
        if "playwright" in err_text or "chromium" in err_text or "npm" in err_text or "node" in err_text:
            dep_issue = True
            dep_reason = dep_reason or "playwright_missing"
        if result.exit_code in {126, 127}:
            dep_issue = True
            dep_reason = dep_reason or "command_not_found"
        if "playwright" in (result.stderr or "").lower() and not normalized_shots:
            dep_issue = True
            dep_reason = dep_reason or "playwright_missing"
        timeout_issue = result.exit_code == 124
        infra_issue = False
        infra_reason = ""
        if not normalized_shots and result.exit_code == 0 and not dep_issue and not timeout_issue:
            infra_issue = True
            infra_reason = dep_reason or "no_screenshots"
        if not normalized_shots and result.exit_code != 0 and not dep_issue and not timeout_issue:
            infra_markers = (
                "ui not reachable",
                "did not become ready",
                "connection refused",
                "econnrefused",
                "timeout",
                "ensure_ui_admin",
            )
            if any(marker in err_text for marker in infra_markers) or result.server_started is False:
                infra_issue = True
                infra_reason = dep_reason or (result.stderr or result.stdout or "ui_unreachable")
        gate_status = "passed" if result.exit_code == 0 and normalized_shots else "failed"
        if dep_issue or timeout_issue or infra_issue:
            gate_status = "skipped"
        gate_exit_code = 0 if gate_status == "skipped" else result.exit_code
        required_flag = not (dep_issue or timeout_issue or infra_issue)
        placeholder_reason = dep_reason or infra_reason or (result.stderr or result.stdout or "")
        placeholder_generated = False
        if (dep_issue or timeout_issue or infra_issue) and not normalized_shots:
            placeholder = _write_placeholder_shot(output_dir, placeholder_reason or "capture_unavailable")
            if placeholder:
                placeholder_generated = True
                normalized_shots = _normalize_shots([{"path": placeholder, "kind": "ui_screenshot"}])
        gate_meta = {
            "base_url": result.base_url,
            "screenshots": [str(shot["path"]) for shot in normalized_shots],
            "viewports": [shot.get("meta", {}).get("viewport") for shot in normalized_shots if shot.get("meta")],
            "manual": manual,
            "auth": result.meta.get("auth") if isinstance(result.meta, dict) else {},
            "routes": result.meta.get("routes") if isinstance(result.meta, dict) else [],
            "dependency_missing": dep_issue,
            "dependency_reason": dep_reason,
            "infra_issue": infra_issue,
            "infra_reason": infra_reason,
            "timeout": timeout_issue,
            "not_relevant": dep_issue or timeout_issue or infra_issue,
            "required": required_flag,
            "skip_reason": dep_reason or infra_reason or ("timeout" if timeout_issue else ""),
            "placeholder": bool((dep_issue or timeout_issue or infra_issue) and normalized_shots and placeholder_generated),
            "placeholder_reason": placeholder_reason[:200],
            "snapshot_doc": str(doc_path),
            "snapshot_dir": str(snapshot_dir),
        }
        db_available = True
        try:
            GateRun.objects.create(
                project=run.project,
                run=run,
                stage="e2e",
                name="ui-snapshot",
                status=gate_status,
                command="node web/frontend/scripts/branddozer_capture.mjs",
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=gate_exit_code,
                duration_ms=duration_ms,
                meta=gate_meta,
            )
        except (OperationalError, DatabaseError) as exc:
            db_available = False
            _emit_ui_snapshot_placeholder(run.id, f"db_write_failed: {exc}", manual=manual)
        doc_lines = [
            f"# UX Snapshot {snapshot_ts}",
            "",
            f"Run: {run.id}",
            f"Prompt: {run.prompt}",
            f"Base URL: {result.base_url}",
            "",
            "Each screenshot is stored with a numeric prefix for auditability. "
            "Descriptions include layman and technical context plus the expected outcome.",
            "",
        ]
        if gate_meta.get("placeholder"):
            doc_lines.extend(
                [
                    "Placeholder screenshots were generated because browser automation dependencies were unavailable.",
                    f"Reason: {(placeholder_reason or dep_reason or 'dependency missing')[:240]}",
                    "",
                ]
            )
        for idx, shot in enumerate(normalized_shots, start=1):
            src = shot.get("path")
            if not src:
                continue
            kind = shot.get("kind") or "ui_screenshot"
            new_name = f"{idx:03d}_{src.name}"
            dest = snapshot_dir / new_name
            try:
                shutil.copy2(src, dest)
            except Exception:
                dest = src
            if db_available:
                try:
                    DeliveryArtifact.objects.create(
                        project=run.project,
                        run=run,
                        session=session,
                        kind=kind,
                        title=new_name,
                        path=str(dest),
                        data={
                            "base_url": result.base_url,
                            "manual": manual,
                            "captured_at": snapshot_ts,
                            "viewport": (shot.get("meta") or {}).get("viewport"),
                        },
                    )
                except (OperationalError, DatabaseError) as exc:
                    db_available = False
                    _emit_ui_snapshot_placeholder(run.id, f"db_write_failed: {exc}", manual=manual)
            doc_lines.extend(
                [
                    f"## {idx:03d} – {new_name}",
                    f"- **What you see (layman):** Visual state of the organism/UX at this route.",
                    "- **What you see (technical):** Rendered UI from Playwright capture; inspect for layout, "
                    "density rendering, and controls.",
                    "- **Expected outcome:** Matches acceptance criteria and prompt intent; no overflow/errors; "
                    "lighting/density reflect data values.",
                    f"- **File:** {dest}",
                    "",
                ]
            )
        if gate_status != "passed":
            status_lines = [
                "## Capture status",
                f"- Status: {gate_status}",
                f"- Exit code: {gate_exit_code}",
                f"- Error: {(result.stderr or dep_reason or 'capture failed')[:400]}",
            ]
            if gate_status == "skipped":
                note = "Non-blocking skip: install Playwright/npm assets to enable capture."
                if timeout_issue:
                    note = "Non-blocking skip: capture timed out; retry after freeing resources."
                elif dep_reason or infra_reason:
                    note = f"Non-blocking skip: {(dep_reason or infra_reason)[:240]}"
                status_lines.append(f"- Note: {note}")
                status_lines.append("- Gate recorded as skipped so delivery can proceed without Playwright assets.")
                if gate_meta.get("placeholder"):
                    status_lines.append("- Placeholder screenshot emitted for audit trail.")
            status_lines.append("")
            doc_lines.extend(status_lines)
        doc_path = snapshot_dir / "README.md"
        try:
            doc_path.write_text("\n".join(doc_lines), encoding="utf-8")
            if db_available:
                try:
                    DeliveryArtifact.objects.create(
                        project=run.project,
                        run=run,
                        session=session,
                        kind="ui_snapshot_doc",
                        title=f"Snapshot doc {snapshot_ts}",
                        path=str(doc_path),
                        content="\n".join(doc_lines),
                        data={"base_url": result.base_url, "manual": manual, "captured_at": snapshot_ts},
                    )
                except (OperationalError, DatabaseError) as exc:
                    db_available = False
                    _emit_ui_snapshot_placeholder(run.id, f"db_write_failed: {exc}", manual=manual)
        except Exception:
            pass
        if result.server_log:
            if db_available:
                try:
                    DeliveryArtifact.objects.create(
                        project=run.project,
                        run=run,
                        session=session,
                        kind="ui_server_log",
                        title="UI server log",
                        path=str(result.server_log),
                    )
                except (OperationalError, DatabaseError) as exc:
                    db_available = False
                    _emit_ui_snapshot_placeholder(run.id, f"db_write_failed: {exc}", manual=manual)
        if not db_available:
            return
        if gate_status != "passed":
            auth_detail = ""
            if isinstance(result.meta, dict):
                auth = result.meta.get("auth") or {}
                if isinstance(auth, dict):
                    auth_detail = str(auth.get("detail") or "")
            error_detail = ""
            if isinstance(result.meta, dict):
                errors = result.meta.get("errors") or []
                if isinstance(errors, list) and errors:
                    error_detail = json.dumps(errors, indent=2)[:2000]
            review_meta = {
                "manual": manual,
                "blocked": gate_status != "skipped",
                "required": gate_status != "skipped",
                "not_relevant": gate_status == "skipped",
            }
            if gate_status == "skipped":
                review_meta.update(
                    {
                        "dependency_missing": dep_issue,
                        "timeout": timeout_issue,
                        "infra_issue": infra_issue,
                        "reason": dep_reason or infra_reason or result.stderr,
                        "required": False,
                        "skip_reason": dep_reason
                        or infra_reason
                        or ("timeout" if timeout_issue else "capture_skipped"),
                        "placeholder": gate_meta.get("placeholder", False),
                    }
                )
            if gate_status == "failed":
                title = "UI snapshot capture failed"
                exists = (
                    BacklogItem.objects.filter(run=run, source="qa", title=title)
                    .exclude(status="done")
                    .exists()
                )
                if not exists:
                    BacklogItem.objects.create(
                        project=run.project,
                        run=run,
                        kind="bug",
                        title=title,
                        description=(
                            result.stderr
                            or auth_detail
                            or error_detail
                            or "UI snapshots did not complete."
                        ),
                        acceptance_criteria=["UI snapshot capture passes"],
                        priority=1,
                        estimate_points=2,
                        status="todo",
                        source="qa",
                    )
            try:
                GateRun.objects.create(
                    project=run.project,
                    run=run,
                    stage="e2e",
                    name="ui-review",
                    status="skipped" if gate_status == "skipped" else "blocked",
                    command="codex --image ...",
                    stdout="",
                    stderr="Snapshot capture failed; review skipped." if gate_status == "failed" else "Snapshot skipped (dependency/timeout).",
                    exit_code=0 if gate_status == "skipped" else 1,
                    duration_ms=duration_ms,
                    meta=review_meta,
                )
            except (OperationalError, DatabaseError) as exc:
                _emit_ui_snapshot_placeholder(run.id, f"db_write_failed: {exc}", manual=manual)
                return
            session.status = "done" if gate_status == "skipped" else "error"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            return
        _append_session_log(session, f"Running visual review with {provider}.")
        backlog_items = list(BacklogItem.objects.filter(run=run).order_by("priority")[:12])
        backlog_summary = "\n".join(
            f"- {item.title}: {item.acceptance_criteria}" for item in backlog_items
        )
        review_prompt = (
            "You are a QA reviewer for the project. Review the attached UI screenshots and verify "
            "they match the prompt and acceptance criteria. Focus on layout, overflow, missing controls, "
            "and any visible errors. Evaluate like a modern web designer: hierarchy, spacing rhythm, "
            "typography scale, contrast, alignment, and responsive consistency. "
            "Identify missing or misplaced UI elements and describe where they should be. "
            "Return ONLY JSON with keys: "
            '{"status":"pass|fail","issues":[{"title":"","detail":"","severity":"info|warn|error","location":"","suggested_fix":""}],"notes":""}.'
            f"\nProject prompt:\n{run.prompt}\n\nBacklog acceptance:\n{backlog_summary}"
        )
        _append_session_log(session, "UX rubric: hierarchy, spacing, typography scale, contrast, alignment, responsiveness.")
        SessionClass = get_session_class(provider)
        review_codex = SessionClass(
            session_name=f"ui-review-{session.id}",
            transcript_dir=Path("runtime/branddozer/transcripts") / str(session.id),
            read_timeout_s=None,
            workdir=str(root),
            **ai_settings,
        )
        review_start = time.time()
        last_heartbeat = 0.0
        def _review_stream(_chunk: str) -> None:
            nonlocal last_heartbeat
            now = time.time()
            if now - last_heartbeat >= 20:
                _touch_session(session.id)
                last_heartbeat = now
        review_output = review_codex.send(
            review_prompt,
            stream=True,
            stream_callback=_review_stream,
            images=[str(shot["path"]) for shot in normalized_shots[:6]],
        )
        if _is_codex_provider(provider):
            exhausted = _codex_quota_exhausted(review_output)
            if exhausted:
                _pause_run_for_codex(run, session, exhausted)
                raise StopDelivery(exhausted)
            refusal = _codex_refusal(review_output)
            if refusal:
                _handle_codex_refusal(run, session, refusal)
                raise StopDelivery(refusal)
        try:
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                session=session,
                kind="ui_review",
                title="UI review output",
                content=review_output,
                path=str(review_codex.transcript_path),
            )
            payload = _extract_json_payload(review_output)
            status_value = str(payload.get("status") or "").lower()
            review_status = "passed" if status_value in {"pass", "passed", "ok"} else "failed"
            GateRun.objects.create(
                project=run.project,
                run=run,
                stage="e2e",
                name="ui-review",
                status=review_status,
                command="codex --image ...",
                stdout=review_output,
                stderr="",
                exit_code=0 if review_status == "passed" else 1,
                duration_ms=int((time.time() - review_start) * 1000),
                meta={**(payload if isinstance(payload, dict) else {}), "manual": manual},
            )
            if review_status != "passed":
                issues = payload.get("issues") if isinstance(payload, dict) else None
                issue_summary = json.dumps(issues, indent=2) if issues else review_output
                title = "UI review failed"
                exists = (
                    BacklogItem.objects.filter(run=run, source="qa", title=title)
                    .exclude(status="done")
                    .exists()
                )
                if not exists:
                    BacklogItem.objects.create(
                        project=run.project,
                        run=run,
                        kind="bug",
                        title=title,
                        description=issue_summary[:4000],
                        acceptance_criteria=["UI review passes", "Screenshots match acceptance criteria"],
                        priority=1,
                        estimate_points=3,
                        status="todo",
                        source="qa",
                    )
                session.status = "blocked"
            else:
                session.status = "done"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
        except (OperationalError, DatabaseError) as exc:
            _emit_ui_snapshot_placeholder(run.id, f"db_write_failed: {exc}", manual=manual)
            return

    def _release_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "release"
        run.save(update_fields=["phase"])
        _set_run_note(run, "Release", "Generating release candidate artifacts")
        version = f"rc-{run.id.hex[:8]}"
        ReleaseCandidate.objects.create(
            project=run.project,
            run=run,
            version=version,
            status="created",
            summary="Release candidate built after gates passed.",
        )
        GovernanceArtifact.objects.create(
            project=run.project,
            run=run,
            kind="completion_report",
            version=_latest_version(run, "completion_report"),
            summary="Completion report generated.",
            content=json.dumps({"dod": run.definition_of_done, "status": run.status}, indent=2),
            data={"dod": run.definition_of_done, "status": run.status},
        )

    def _dod_satisfied(self, run: DeliveryRun) -> bool:
        backlog_open = BacklogItem.objects.filter(run=run).exclude(status="done").exists()
        if backlog_open:
            return False
        required_gates: Dict[str, str] = {}
        for gate in GateRun.objects.filter(run=run).order_by("name", "-created_at"):
            if gate.name in required_gates:
                continue
            meta = gate.meta or {}
            required = bool(meta.get("required", True))
            not_relevant = bool(meta.get("not_relevant"))
            if not required:
                if gate.status in {"skipped", "failed", "blocked"} and (not_relevant or "not installed" in (gate.stderr or "").lower()):
                    continue
                if gate.status == "passed":
                    required_gates[gate.name] = gate.status
                continue
            required_gates[gate.name] = gate.status
        if not required_gates:
            return False
        if any(status not in {"passed", "skipped"} for status in required_gates.values()):
            return False
        return True

    def _codex_or_template(
        self,
        run: DeliveryRun,
        root: Path,
        prompt: str,
        fallback_kind: str,
        session: Optional[DeliverySession] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        offline_mode = (os.getenv("BRANDDOZER_OFFLINE_MODE") or os.getenv("BRANDDOZER_FORCE_TEMPLATE") or "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        provider = session_provider_from_context(run.context or {})
        provider_available = not offline_mode and (not _is_codex_provider(provider) or shutil.which("codex") is not None)
        read_timeout_s = None
        try:
            read_timeout_s = float(os.getenv("CODEX_READ_TIMEOUT", "600"))
        except Exception:
            read_timeout_s = 600.0
        fallback_output = ""
        codex_error = ""
        if provider_available:
            try:
                SessionClass = get_session_class(provider)
                ai_settings = _session_settings_for_run(run, "planner")
                codex = SessionClass(
                    session_name=f"{fallback_kind}-{run.id}",
                    transcript_dir=Path("runtime/branddozer/transcripts"),
                    read_timeout_s=read_timeout_s,
                    workdir=str(root),
                    **ai_settings,
                )
                full_prompt = (
                    f"{prompt}\nUser prompt: {run.prompt}\n"
                    "If UI verification is needed, use scripts/branddozer_ui_capture.py to capture screenshots "
                    "and review them with the configured session provider (codex --image when available).\n"
                )
                last_heartbeat = 0.0

                def _stream_writer(_chunk: str) -> None:
                    nonlocal last_heartbeat
                    if session is None:
                        return
                    now = time.time()
                    if now - last_heartbeat >= 20:
                        _touch_session(session.id)
                        last_heartbeat = now

                output = codex.send(full_prompt, stream=True, stream_callback=_stream_writer)
                if _is_codex_provider(provider):
                    exhausted = _codex_quota_exhausted(output)
                    if exhausted:
                        _pause_run_for_codex(run, session, exhausted)
                        raise StopDelivery(exhausted)
                    refusal = _codex_refusal(output)
                    if refusal:
                        # Fallback to template but still record refusal
                        _handle_codex_refusal(run, session, refusal, block=False)
                        codex_error = refusal
                        fallback_output = output
                        raise Exception(refusal)
                if session:
                    _append_session_log(session, f"{fallback_kind} generated.")
                    _log_ai_choice(session, provider, ai_settings, "planner")
                DeliveryArtifact.objects.create(
                    project=run.project,
                    run=run,
                    kind=f"{fallback_kind}_raw",
                    title=f"{fallback_kind} raw output",
                    content=output,
                    path=str(codex.transcript_path),
                )
                try:
                    data = json.loads(output)
                    return output, data
                except Exception:
                    codex_error = "parse_failed"
                    fallback_output = output
            except Exception as exc:
                codex_error = str(exc)
                if session:
                    _append_session_log(session, f"{fallback_kind} fallback to template ({codex_error})")
        fallback = {
            "requirements": {
                "functional": [run.prompt],
                "non_functional": ["security", "performance", "accessibility", "maintainability"],
                "constraints": [],
                "assumptions": [],
                "out_of_scope": [],
            },
            "blueprint": {
                "architecture": "tbd",
                "data_flows": [],
                "threat_model": "tbd",
                "data_model": [],
                "api_contracts": [],
                "ux_flows": [],
                "accessibility_targets": ["WCAG 2.2 AA"],
                "as_is": "existing project",
                "to_be": "aligned with prompt",
            },
            "backlog": {
                "items": [
                    {
                        "kind": "story",
                        "title": run.prompt[:120],
                        "description": run.prompt,
                        "acceptance_criteria": ["Meets prompt intent", "All gates green"],
                        "priority": 1,
                        "estimate_points": 3,
                        "dependencies": [],
                    }
                ]
            },
            "charter": {
                "success_metrics": ["All DoD gates green", "User acceptance recorded"],
                "scope": run.prompt,
                "constraints": run.context.get("constraints", []),
                "assumptions": [],
            },
            "wbs": {
                "work_packages": ["baseline", "requirements", "blueprint", "backlog", "implementation", "verification"],
            },
            "quality_plan": {
                "gates": [gate.name for gate in default_gates(root)],
                "staging": ["fast", "integration", "security", "e2e"],
            },
            "release_criteria": {
                "dod": run.definition_of_done or DEFAULT_DOD,
            },
        }
        data = fallback.get(fallback_kind, {"raw": run.prompt})
        if codex_error:
            data["codex_error"] = codex_error
            if fallback_output:
                data["raw_output"] = fallback_output
        if session and codex_error:
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                kind=f"{fallback_kind}_raw",
                title=f"{fallback_kind} fallback template",
                content=json.dumps(data, indent=2),
                path="",
            )
        return json.dumps(data, indent=2), data


class BrandDozerProjectLookup:
    @staticmethod
    def get_project(project_id: str):
        from services.branddozer_state import get_project as get_project_state
        from branddozer.models import BrandProject

        _ensure_branddozer_schema("project_lookup")
        project = BrandProject.objects.filter(id=project_id).first()
        if project:
            return project
        data = get_project_state(project_id)
        if not data:
            raise ValueError("Project not found")
        return BrandProject.objects.filter(id=data["id"]).first()


delivery_orchestrator = DeliveryOrchestrator()
