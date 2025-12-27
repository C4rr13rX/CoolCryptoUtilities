from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from django.db import close_old_connections
from django.utils import timezone

from tools.codex_session import CodexSession
from services.branddozer_ui import capture_ui_screenshots
from branddozer.models import (
    AcceptanceRecord,
    BacklogItem,
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
    "release",
    "awaiting_acceptance",
]

SESSION_LOG_ROOT = Path("runtime/branddozer/sessions")
SESSION_LOG_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GateDefinition:
    name: str
    stage: str
    command: Optional[List[str]] = None
    timeout_s: int = 900
    required: bool = True
    runner: Optional[Callable[[Path], Tuple[str, str, int]]] = None


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


def _secret_scan(root: Path) -> Tuple[str, str, int]:
    patterns = [
        ("AWS Access Key", r"AKIA[0-9A-Z]{16}"),
        ("GitHub Token", r"ghp_[A-Za-z0-9]{36,}"),
        ("OpenAI Key", r"sk-[A-Za-z0-9]{32,}"),
        ("Private Key", r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----"),
    ]
    findings: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", ".venv", "venv", "__pycache__"}]
        for filename in filenames:
            full = Path(dirpath) / filename
            if full.suffix in {".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".tar", ".bin"}:
                continue
            try:
                if full.stat().st_size > 2_000_000:
                    continue
                text = full.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for label, regex in patterns:
                if re.search(regex, text):
                    findings.append(f"{label}: {full}")
    if findings:
        return "", "\n".join(findings), 1
    return "No secrets detected.", "", 0


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
        stdout, stderr, code = _safe_run(["git", "worktree", "add", "-b", branch, str(workspace)], root, 60)
        if code != 0:
            raise ValueError(stderr or stdout or "Failed to create git worktree")
        return workspace, branch
    shutil.copytree(root, workspace, dirs_exist_ok=True)
    return workspace, None


def _cleanup_workspace(root: Path, workspace: Path, branch: Optional[str]) -> None:
    if _git_available(root) and branch:
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
        return _safe_run(["git", "apply", "--check", str(patch_file)], root, 30)
    return _safe_run(["patch", "-p1", "-i", str(patch_file)], root, 60)


def _latest_version(run: DeliveryRun, kind: str) -> int:
    latest = GovernanceArtifact.objects.filter(run=run, kind=kind).order_by("-version").first()
    return (latest.version if latest else 0) + 1


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
                meta={"cached": True},
            )
        start = time.time()
        if gate.runner:
            stdout, stderr, code = gate.runner(self.root)
        elif gate.command:
            stdout, stderr, code = _safe_run(gate.command, self.root, gate.timeout_s)
        else:
            stdout, stderr, code = "", "missing gate runner", 1
        status = "passed" if code == 0 else "failed"
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
        )


def default_gates(root: Path) -> List[GateDefinition]:
    gates: List[GateDefinition] = []

    pytest_cmd = ["python", "-m", "pytest", "-q"]
    unittest_cmd = ["python", "-m", "unittest", "discover"]
    if shutil.which("pytest"):
        gates.append(GateDefinition(name="unit-tests", stage="fast", command=pytest_cmd, timeout_s=900))
    else:
        gates.append(GateDefinition(name="unit-tests", stage="fast", command=unittest_cmd, timeout_s=900))

    if shutil.which("ruff"):
        gates.append(GateDefinition(name="lint", stage="fast", command=["ruff", "check", "."], timeout_s=600))
        gates.append(GateDefinition(name="format", stage="fast", command=["ruff", "format", "--check", "."], timeout_s=600))
    else:
        gates.append(GateDefinition(name="syntax", stage="fast", command=["python", "-m", "compileall", "-q", "."], timeout_s=600))

    gates.append(GateDefinition(name="pip-check", stage="integration", command=["python", "-m", "pip", "check"], timeout_s=600))

    if shutil.which("pip-audit"):
        gates.append(GateDefinition(name="dependency-vuln", stage="security", command=["pip-audit", "-r", "requirements.txt"], timeout_s=900))
    else:
        gates.append(GateDefinition(name="dependency-vuln", stage="security", runner=_missing_tool("pip-audit"), timeout_s=5, required=True))

    if shutil.which("bandit"):
        gates.append(GateDefinition(name="static-security", stage="security", command=["bandit", "-r", "."], timeout_s=900))
    else:
        gates.append(GateDefinition(name="static-security", stage="security", runner=_missing_tool("bandit"), timeout_s=5, required=True))

    gates.append(GateDefinition(name="secret-scan", stage="security", runner=_secret_scan, timeout_s=900))
    gates.append(GateDefinition(name="django-hardening", stage="security", runner=_django_hardening, timeout_s=120))

    playwright_config = any((root / f).exists() for f in ["playwright.config.ts", "playwright.config.js"])
    if playwright_config and shutil.which("npx"):
        gates.append(GateDefinition(name="e2e-smoke", stage="e2e", command=["npx", "playwright", "test", "--reporter=line"], timeout_s=1800))
        gates.append(GateDefinition(name="a11y", stage="e2e", command=["npx", "playwright", "test", "--grep", "@a11y", "--reporter=line"], timeout_s=1800))
    else:
        gates.append(GateDefinition(name="e2e-smoke", stage="e2e", runner=_missing_tool("playwright"), timeout_s=5, required=True))
        gates.append(GateDefinition(name="a11y", stage="e2e", runner=_missing_tool("playwright-a11y"), timeout_s=5, required=True))

    return gates


class DeliveryOrchestrator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._threads: Dict[str, threading.Thread] = {}

    def start_run(self, project_id: str, prompt: str, mode: str = "auto") -> DeliveryRun:
        project = BrandDozerProjectLookup.get_project(project_id)
        delivery_project, _ = DeliveryProject.objects.get_or_create(project=project, defaults={"definition_of_done": DEFAULT_DOD})
        run = DeliveryRun.objects.create(
            project=project,
            prompt=prompt.strip(),
            mode=mode,
            status="queued",
            definition_of_done=delivery_project.definition_of_done or DEFAULT_DOD,
        )
        delivery_project.active_run = run
        delivery_project.status = "running"
        delivery_project.mode = mode
        delivery_project.save(update_fields=["active_run", "status", "mode", "updated_at"])

        thread = threading.Thread(target=self._run_pipeline, args=(run.id,), daemon=True)
        with self._lock:
            self._threads[str(run.id)] = thread
        thread.start()
        return run

    def _run_pipeline(self, run_id: uuid.UUID) -> None:
        close_old_connections()
        try:
            run = DeliveryRun.objects.get(id=run_id)
        except DeliveryRun.DoesNotExist:
            return
        project = run.project
        root = Path(project.root_path)
        orchestrator_session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="orchestrator",
            name="Orchestrator",
            status="running",
            workspace_path=str(root),
            meta={"phase": "start"},
        )
        _append_session_log(orchestrator_session, f"Run {run.id} started for {project.name}.")
        try:
            run.status = "running"
            run.started_at = timezone.now()
            run.phase = "mode_detection"
            run.save(update_fields=["status", "started_at", "phase"])
            _append_session_log(orchestrator_session, "Detecting start mode.")

            mode = self._detect_mode(run.mode, root)
            run.mode = mode
            run.save(update_fields=["mode"])
            _append_session_log(orchestrator_session, f"Mode: {mode}.")

            if mode == "existing":
                _append_session_log(orchestrator_session, "Running baseline review.")
                self._baseline_review(run, root)

            _append_session_log(orchestrator_session, "Generating governance artifacts.")
            self._governance_step(run, root)
            _append_session_log(orchestrator_session, "Generating requirements.")
            self._requirements_step(run, root)
            _append_session_log(orchestrator_session, "Generating blueprint.")
            self._blueprint_step(run, root)
            _append_session_log(orchestrator_session, "Building backlog and sprint.")
            backlog_items = self._backlog_step(run, root)
            sprint = self._sprint_plan(run, backlog_items)
            _append_session_log(orchestrator_session, "Executing sprint.")
            self._execution_loop(run, root, sprint)

            if self._dod_satisfied(run):
                _append_session_log(orchestrator_session, "Definition of Done satisfied. Preparing release candidate.")
                self._release_step(run, root)
                run.status = "awaiting_acceptance" if run.acceptance_required else "complete"
                run.phase = "awaiting_acceptance"
            else:
                _append_session_log(orchestrator_session, "Definition of Done not satisfied. Blocking run.")
                run.status = "blocked"
                run.phase = "gates"
        except Exception as exc:
            run.status = "error"
            run.phase = "gates"
            run.error = str(exc)
            _append_session_log(orchestrator_session, f"Error: {exc}")
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
        orchestrator_session.status = "done" if run.status != "error" else "error"
        orchestrator_session.completed_at = timezone.now()
        orchestrator_session.save(update_fields=["status", "completed_at"])
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
        baseline_data["gates"] = [
            {"name": g.name, "stage": g.stage, "status": g.status, "stderr": g.stderr} for g in gate_results
        ]
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
        for gate in gate_results:
            if gate.status != "passed":
                BacklogItem.objects.create(
                    project=run.project,
                    run=run,
                    kind="bug",
                    title=f"Baseline gate failed: {gate.name}",
                    description=gate.stderr or gate.stdout,
                    acceptance_criteria=["Gate passes with green status"],
                    priority=1,
                    estimate_points=2,
                    status="todo",
                    source="baseline",
                    meta={"stage": gate.stage},
                )
                RaidEntry.objects.create(
                    project=run.project,
                    run=run,
                    kind="issue",
                    title=f"Gate failure: {gate.name}",
                    description=gate.stderr or gate.stdout,
                    severity="high" if gate.stage in {"security", "e2e"} else "medium",
                )

    def _governance_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "governance"
        run.save(update_fields=["phase"])
        pm_session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="pm",
            name="Project Manager Session",
            status="running",
            workspace_path=str(root),
        )
        _append_session_log(pm_session, "Generating project charter.")
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
        pm_session.status = "done"
        pm_session.completed_at = timezone.now()
        pm_session.save(update_fields=["status", "completed_at"])

    def _requirements_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "requirements"
        run.save(update_fields=["phase"])
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

    def _backlog_step(self, run: DeliveryRun, root: Path) -> List[BacklogItem]:
        run.phase = "backlog"
        run.save(update_fields=["phase"])
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

    def _sprint_plan(self, run: DeliveryRun, backlog_items: List[BacklogItem]) -> Sprint:
        run.phase = "sprint_planning"
        run.save(update_fields=["phase"])
        sprint = Sprint.objects.create(project=run.project, run=run, number=1, goal="Initial delivery sprint", status="active")
        for item in backlog_items:
            SprintItem.objects.get_or_create(sprint=sprint, backlog_item=item, defaults={"status": "todo"})
        run.sprint_count = 1
        run.save(update_fields=["sprint_count"])
        return sprint

    def _execution_loop(self, run: DeliveryRun, root: Path, sprint: Sprint) -> None:
        run.phase = "execution"
        run.iteration += 1
        run.context["parallelism"] = _dynamic_parallelism()
        run.save(update_fields=["phase", "iteration", "context"])
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="integrator",
            name="Integrator/Release Session",
            status="running",
            workspace_path=str(root),
            meta={"note": "Integration uses canonical workspace"},
        )
        _append_session_log(session, "Integrator session started.")
        for sprint_item in sprint.items.select_related("backlog_item"):
            if sprint_item.status == "done":
                continue
            backlog_item = sprint_item.backlog_item
            _append_session_log(session, f"Applying task: {backlog_item.title}")
            backlog_item.status = "in_progress"
            backlog_item.save(update_fields=["status", "updated_at"])
            sprint_item.status = "in_progress"
            sprint_item.save(update_fields=["status"])
            diff_text = self._run_task_session(run, root, backlog_item)
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
                sprint_item.status = "done"
            else:
                backlog_item.status = "blocked"
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
            sprint_item.save(update_fields=["status"])
            _append_session_log(session, f"Task {backlog_item.title} status: {backlog_item.status}.")
        session.status = "done"
        session.completed_at = timezone.now()
        session.save(update_fields=["status", "completed_at"])
        sprint.status = "complete"
        sprint.completed_at = timezone.now()
        sprint.save(update_fields=["status", "completed_at"])
        self._gate_stage(run, root)

    def _run_task_session(self, run: DeliveryRun, root: Path, backlog_item: BacklogItem) -> str:
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="dev",
            name=f"CodexSession: {backlog_item.title[:80]}",
            status="running",
            workspace_path="",
        )
        workspace, branch = _create_workspace(root, run.id, session.id)
        session.workspace_path = str(workspace)
        session.save(update_fields=["workspace_path"])
        _append_session_log(session, f"Workspace ready at {workspace}.")
        transcript_dir = Path("runtime/branddozer/transcripts") / str(session.id)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        prompt = (
            f"[CodexSession Task]\nProject: {run.project.name}\nRoot: {workspace}\n"
            f"Task: {backlog_item.title}\nDescription: {backlog_item.description}\n"
            f"Acceptance: {backlog_item.acceptance_criteria}\n"
            "Work in a fix/test loop. Attach diff + gate results. "
            "For UI checks, run `python scripts/branddozer_ui_capture.py --base-url http://127.0.0.1:8000` "
            "and include the screenshot paths in your output (or review them with `codex --image <path>`)."
        )
        codex = CodexSession(
            session_name=f"delivery-{session.id}",
            transcript_dir=transcript_dir,
            sandbox_mode="danger-full-access",
            approval_policy="never",
            model="gpt-5.1-codex-max",
            reasoning_effort="xhigh",
            read_timeout_s=None,
            workdir=str(workspace),
        )
        log_path = _session_log_path(session.id)
        session.log_path = str(log_path)
        session.save(update_fields=["log_path"])

        def _stream_writer(chunk: str) -> None:
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(chunk)
            except Exception:
                pass

        output = codex.send(prompt, stream=True, stream_callback=_stream_writer)
        diff_text = _compute_diff(root, workspace)
        DeliveryArtifact.objects.create(
            project=run.project,
            run=run,
            session=session,
            kind="session_log",
            title=f"Codex output: {backlog_item.title[:60]}",
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
        session.status = "done"
        session.completed_at = timezone.now()
        session.log_path = str(codex.transcript_path)
        session.save(update_fields=["status", "completed_at", "log_path"])
        _cleanup_workspace(root, workspace, branch)
        return diff_text

    def _gate_stage(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "gates"
        run.save(update_fields=["phase"])
        runner = GateRunner(run, root)
        for gate in default_gates(root):
            result = runner.run_gate(gate)
            if result.status != "passed" and gate.required:
                BacklogItem.objects.create(
                    project=run.project,
                    run=run,
                    kind="bug",
                    title=f"Fix failing gate: {result.name}",
                    description=result.stderr or result.stdout,
                    acceptance_criteria=["Gate passes"],
                    priority=1,
                    estimate_points=2,
                    status="todo",
                    source="gate",
                    meta={"stage": result.stage},
                )
        self._ui_snapshot_review(run, root)

    def trigger_ui_review(self, run_id: uuid.UUID, *, manual: bool = True) -> None:
        thread = threading.Thread(target=self._run_ui_review, args=(run_id, manual), daemon=True)
        with self._lock:
            self._threads[f"ui-review-{run_id}"] = thread
        thread.start()

    def _run_ui_review(self, run_id: uuid.UUID, manual: bool) -> None:
        close_old_connections()
        run = DeliveryRun.objects.select_related("project").filter(id=run_id).first()
        if not run:
            return
        root = Path(run.project.root_path)
        self._ui_snapshot_review(run, root, manual=manual)
        close_old_connections()

    def _ui_snapshot_review(self, run: DeliveryRun, root: Path, *, manual: bool = False) -> None:
        flag = (os.getenv("BRANDDOZER_UI_CAPTURE") or "1").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            return
        session = DeliverySession.objects.create(
            project=run.project,
            run=run,
            role="qa",
            name="UX Verification Session",
            status="running",
            workspace_path=str(root),
            meta={"manual": manual},
        )
        _append_session_log(session, "Capturing UI screenshots.")
        output_dir = Path("runtime/branddozer/ui") / str(run.id) / str(session.id)
        start_ts = time.time()
        result = capture_ui_screenshots(root, output_dir=output_dir)
        duration_ms = int((time.time() - start_ts) * 1000)
        gate_status = "passed" if result.exit_code == 0 and result.screenshots else "failed"
        GateRun.objects.create(
            project=run.project,
            run=run,
            stage="e2e",
            name="ui-snapshot",
            status=gate_status,
            command="node web/frontend/scripts/branddozer_capture.mjs",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            duration_ms=duration_ms,
            meta={
                "base_url": result.base_url,
                "screenshots": [str(p) for p in result.screenshots],
                "manual": manual,
                "auth": result.meta.get("auth") if isinstance(result.meta, dict) else {},
                "routes": result.meta.get("routes") if isinstance(result.meta, dict) else [],
            },
        )
        for shot in result.screenshots:
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                session=session,
                kind="ui_screenshot",
                title=shot.name,
                path=str(shot),
            )
        if result.server_log:
            DeliveryArtifact.objects.create(
                project=run.project,
                run=run,
                session=session,
                kind="ui_server_log",
                title="UI server log",
                path=str(result.server_log),
            )
        if gate_status != "passed":
            auth_detail = ""
            if isinstance(result.meta, dict):
                auth = result.meta.get("auth") or {}
                if isinstance(auth, dict):
                    auth_detail = str(auth.get("detail") or "")
            BacklogItem.objects.create(
                project=run.project,
                run=run,
                kind="bug",
                title="UI snapshot capture failed",
                description=(
                    result.stderr
                    or auth_detail
                    or "UI snapshots did not complete."
                ),
                acceptance_criteria=["UI snapshot capture passes"],
                priority=1,
                estimate_points=2,
                status="todo",
                source="qa",
            )
            session.status = "error"
            session.completed_at = timezone.now()
            session.save(update_fields=["status", "completed_at"])
            return
        _append_session_log(session, "Running visual review with Codex.")
        backlog_items = list(BacklogItem.objects.filter(run=run).order_by("priority")[:12])
        backlog_summary = "\n".join(
            f"- {item.title}: {item.acceptance_criteria}" for item in backlog_items
        )
        review_prompt = (
            "You are a QA reviewer for the project. Review the attached UI screenshots and verify "
            "they match the prompt and acceptance criteria. Focus on layout, overflow, missing controls, "
            "and any visible errors. Return ONLY JSON with keys: "
            '{"status":"pass|fail","issues":[{"title":"","detail":"","severity":"info|warn|error"}],"notes":""}.'
            f"\nProject prompt:\n{run.prompt}\n\nBacklog acceptance:\n{backlog_summary}"
        )
        review_codex = CodexSession(
            session_name=f"ui-review-{session.id}",
            transcript_dir=Path("runtime/branddozer/transcripts") / str(session.id),
            sandbox_mode="danger-full-access",
            approval_policy="never",
            model="gpt-5.1-codex-max",
            reasoning_effort="xhigh",
            read_timeout_s=None,
            workdir=str(root),
        )
        review_start = time.time()
        review_output = review_codex.send(
            review_prompt,
            stream=True,
            images=[str(path) for path in result.screenshots[:6]],
        )
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
            BacklogItem.objects.create(
                project=run.project,
                run=run,
                kind="bug",
                title="UI review failed",
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

    def _release_step(self, run: DeliveryRun, root: Path) -> None:
        run.phase = "release"
        run.save(update_fields=["phase"])
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
        gate_failures = GateRun.objects.filter(run=run).exclude(status="passed").exists()
        if gate_failures:
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
        codex_available = shutil.which("codex") is not None
        if codex_available:
            codex = CodexSession(
                session_name=f"{fallback_kind}-{run.id}",
                transcript_dir=Path("runtime/branddozer/transcripts"),
                sandbox_mode="danger-full-access",
                approval_policy="never",
                model="gpt-5.1-codex-max",
                reasoning_effort="xhigh",
                read_timeout_s=None,
                workdir=str(root),
            )
            full_prompt = (
                f"{prompt}\nUser prompt: {run.prompt}\n"
                "If UI verification is needed, use scripts/branddozer_ui_capture.py to capture screenshots "
                "and review them with codex --image.\n"
            )
            output = codex.send(full_prompt, stream=False)
            if session:
                _append_session_log(session, f"{fallback_kind} generated.")
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
                return output, {"raw": output}
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
        return json.dumps(data, indent=2), data


class BrandDozerProjectLookup:
    @staticmethod
    def get_project(project_id: str):
        from services.branddozer_state import get_project as get_project_state
        from branddozer.models import BrandProject

        project = BrandProject.objects.filter(id=project_id).first()
        if project:
            return project
        data = get_project_state(project_id)
        if not data:
            raise ValueError("Project not found")
        return BrandProject.objects.filter(id=data["id"]).first()


delivery_orchestrator = DeliveryOrchestrator()
