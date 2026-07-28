from __future__ import annotations

import argparse
import ast
import json
import os
import re
import threading
import time
import traceback
from difflib import unified_diff
from pathlib import Path

from .feedback import ModelFeedbackStore
from .workday import WorkdayStore


def run_job(
    *,
    db_path: str | Path,
    job_id: str,
    owner: str,
    lease_seconds: int,
    heartbeat_seconds: int,
    retry_seconds: int,
    quota_retry_seconds: int,
) -> int:
    store = WorkdayStore(db_path)
    job = store.get(job_id)
    if not job or job["status"] != "running" or job["lease_owner"] != owner:
        return 2
    previous_backend = os.environ.get("C0D3R_BACKEND")
    os.environ["C0D3R_BACKEND"] = "freeloader"
    store.set_worker_pid(job_id, owner, os.getpid())

    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            if not store.heartbeat(job_id, owner, lease_seconds):
                return

    thread = threading.Thread(target=heartbeat, name=f"atf-heartbeat-{job_id[:8]}", daemon=True)
    thread.start()
    try:
        result = _execute(job, job_id)
        if result.get("capacity_wait"):
            store.defer_for_capacity(
                job_id, owner, result.get("error") or "no eligible model capacity",
                quota_retry_seconds, result,
            )
            return 4
        success = bool(result["success"])
        verified_progress = bool(result.get("verified_progress"))
        correction_event_id = _record_feedback(
            result.get("artifact_models") or result.get("models") or [], success,
            result.get("error") or "validation passed",
            previous_checkpoint=job.get("checkpoint") or {},
            verified_progress=verified_progress,
            failed_output=str(result.get("rollback_diff") or ""),
        )
        if correction_event_id:
            result["correction_event_id"] = correction_event_id
        store.checkpoint(job_id, owner, result)
        current = store.get(job_id)
        if not current or current["cancel_requested"]:
            return 3
        if success:
            store.complete(job_id, owner, result)
            return 0
        delay = _retry_delay(
            result.get("error") or "validation failed",
            int(job["attempts"]), retry_seconds, quota_retry_seconds,
        )
        store.retry(job_id, owner, result.get("error") or "validation failed", delay, result)
        return 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        checkpoint = {"success": False, "error": error, "traceback": traceback.format_exc()[-8000:]}
        delay = _retry_delay(error, int(job["attempts"]), retry_seconds, quota_retry_seconds)
        store.retry(job_id, owner, error, delay, checkpoint)
        return 1
    finally:
        stop.set()
        thread.join(timeout=2.0)
        if previous_backend is None:
            os.environ.pop("C0D3R_BACKEND", None)
        else:
            os.environ["C0D3R_BACKEND"] = previous_backend


def _execute(job: dict, job_id: str) -> dict:
    from tools.c0d3rV2.delivery_runner import run_delivery_turn_detailed
    from tools.c0d3rV2.executor import Executor

    workdir = Path(job["workdir"])
    before_files = _workspace_snapshot(workdir)
    before_contents = _workspace_contents(workdir)
    checkpoint = job.get("checkpoint") or {}
    checkpoint_text = ""
    if checkpoint:
        prior_output = str(checkpoint.get("output") or "")[-2000:]
        prior_error = str(checkpoint.get("error") or "")[-4000:]
        prior_rejected_diff = str(checkpoint.get("rollback_diff") or "")[-6000:]
        contract_summary = _python_contract_summary(workdir)
        prior_validation = checkpoint.get("validation") or {}
        raw_validation_evidence = str(
            prior_validation.get("stderr") or prior_validation.get("stdout") or ""
        )
        validation_evidence = (
            raw_validation_evidence
            if len(raw_validation_evidence) <= 8000
            else raw_validation_evidence[:4000] + "\n...[middle omitted]...\n" + raw_validation_evidence[-4000:]
        )
        checkpoint_text = (
            "\n\nCORRECTIVE RETRY — inspect the existing project and fix the specific "
            "acceptance failures below. Do not restart or merely restate the design."
            f"\nPrior validation error:\n{prior_error}"
            f"\nPrior validation evidence:\n{validation_evidence}"
            f"\nRejected patch from the prior attempt (already rolled back; do not repeat it):\n"
            f"{prior_rejected_diff or '(none)'}"
            f"\nLocal Python API contract analysis:\n{contract_summary or '(no mismatches detected)'}"
            f"\nPrior model output (truncated):\n{prior_output[-500:]}"
        )
    validation = str(job.get("validation_command") or "").strip()
    baseline_validation: dict = {}
    if validation and checkpoint:
        baseline_timeout = max(30, min(600, int(job.get("timeout_seconds") or 1800) // 3))
        base_code, base_stdout, base_stderr = Executor(workdir, timeout_s=baseline_timeout).run(validation)
        baseline_validation = {
            "return_code": base_code,
            "stdout": base_stdout[-12000:],
            "stderr": base_stderr[-12000:],
            "severity": _validation_severity(base_code, base_stdout, base_stderr),
            "progress_metric": _validation_progress_metric(base_stdout, base_stderr),
        }
        if base_code == 0:
            return {
                "success": True,
                "output": "Existing corrective checkpoint passes the acceptance validator; no model mutation needed.",
                "models": [],
                "artifact_models": [],
                "route_history": [],
                "tool_trace": [],
                "turn_model_calls": 0,
                "validation": {
                    "command": validation,
                    "return_code": base_code,
                    "stdout": base_stdout[-12000:],
                    "stderr": base_stderr[-12000:],
                    "severity": 0,
                    "progress_metric": _validation_progress_metric(base_stdout, base_stderr),
                },
                "baseline_validation": baseline_validation,
                "changed_paths": [],
                "verified_progress": True,
                "finished_at": time.time(),
            }
    validation_note = (
        f"\nAfter implementation, run or satisfy this acceptance command: {validation}"
        if validation else
        "\nFinish this atomic task completely and use C0d3rV2 tools to verify the result."
    )
    original_prompt = str(job["prompt"])
    contract_envelope = _typescript_contract_envelope(workdir, original_prompt)
    prompt = (
        (checkpoint_text + "\n\nORIGINAL ACCEPTANCE OBJECTIVE:\n" + original_prompt)
        if checkpoint else original_prompt
    )
    if contract_envelope:
        prompt += "\n\n" + contract_envelope
    prompt += (
        validation_note
        + "\nThis is an unattended atomic workday job. Make concrete progress; do not only describe a plan."
    )
    detail = run_delivery_turn_detailed(
        prompt,
        session_key=f"atf-workday:{job_id}",
        workdir=workdir,
        backend="freeloader",
        system_context=(
            "You are C0d3rV2 running exclusively through AgentTheFreeloader. "
            f"The exact writable workdir is {workdir.resolve()}. "
            "Use workdir-relative paths for tools and never target its parent or another repository. "
            "A framework generator may create one conventional child project directory, but do not "
            "repeat the workdir name or create multiple nested project roots. "
            "Use tools for all filesystem changes. Keep changes inside the supplied workdir. "
            "Return an evidence-backed completion summary."
        ),
        reset=True,
    )
    protected_paths_restored: list[str] = []
    prior_validation_stdout = str(
        ((checkpoint.get("validation") or {}).get("stdout") or "")
    )
    if (
        checkpoint
        and "validate_atf_benchmark.py" in validation
        and not _hidden_checks_passed(prior_validation_stdout)
    ):
        protected_paths_restored = _restore_benchmark_tests(workdir, before_contents)
    session_error = str(detail.get("session_error") or "")
    route_history = detail.get("route_history") or []
    last_trace = route_history[-1] if route_history else []
    last_trace_selected = any(
        isinstance(item, dict) and item.get("outcome") == "selected"
        for item in last_trace
    )
    if (
        str(detail.get("output") or "").startswith("[c0d3rv2-delivery]")
        and (not detail.get("models") or not last_trace_selected)
    ) or "no eligible model" in session_error.lower():
        return {
            "success": False,
            "capacity_wait": True,
            "output": str(detail.get("output") or ""),
            "models": [],
            "validation": {},
            "error": (
                "no eligible model capacity produced a C0d3rV2 turn; waiting for "
                "a provider cooldown or newly configured credentials"
                + (f": {session_error}" if session_error else "")
            ),
            "finished_at": time.time(),
        }
    validation_result: dict = {}
    success = True
    error = ""
    if validation:
        timeout = max(30, min(600, int(job.get("timeout_seconds") or 1800) // 3))
        code, stdout, stderr = Executor(workdir, timeout_s=timeout).run(validation)
        validation_result = {
            "command": validation,
            "return_code": code,
            "stdout": stdout[-12000:],
            "stderr": stderr[-12000:],
        }
        success = code == 0
        if not success:
            error = f"validation command failed with exit code {code}: {(stderr or stdout)[-2000:]}"
        validation_result["severity"] = _validation_severity(code, stdout, stderr)
        validation_result["progress_metric"] = _validation_progress_metric(stdout, stderr)
    after_files = _workspace_snapshot(workdir)
    restored_set = set(protected_paths_restored)
    changed_paths = sorted(
        path for path in set(before_files) | set(after_files)
        if before_files.get(path) != after_files.get(path)
        and Path(path).as_posix() not in restored_set
    )
    if not success and not changed_paths:
        error = f"no artifact progress during corrective turn; {error}"
    rollback_performed = False
    rollback_diff = ""
    post_severity = int(validation_result.get("severity") or 0)
    base_severity = int(baseline_validation.get("severity") or 0)
    post_progress = int(validation_result.get("progress_metric") or 0)
    base_progress = int(baseline_validation.get("progress_metric") or 0)
    neutral_preparation = _safe_import_only_change(workdir, before_contents, changed_paths)
    regressed = bool(
        post_severity > base_severity
        or (post_severity == base_severity and post_progress <= base_progress)
    )
    retain_neutral = bool(
        post_severity == base_severity and post_progress == base_progress and neutral_preparation
    )
    if not success and changed_paths and baseline_validation and regressed and not retain_neutral:
        rollback_diff = _workspace_diff(workdir, before_contents, changed_paths)
        _restore_workspace(workdir, before_contents)
        rollback_performed = True
        error = (
            "artifact regression rolled back; post-repair validation severity "
            f"{validation_result.get('severity')} did not improve baseline {baseline_validation.get('severity')}; "
            + error
        )
    verified_progress = bool(
        not success and not rollback_performed and changed_paths and baseline_validation
        and (
            post_severity < base_severity
            or (post_severity == base_severity and post_progress > base_progress)
        )
    )
    return {
        "success": success,
        "output": str(detail.get("output") or "")[-20000:],
        "models": detail.get("models") or [],
        "artifact_models": detail.get("artifact_models") or [],
        "tool_events": detail.get("tool_events") or [],
        "turn_model_calls": detail.get("turn_model_calls") or 0,
        "validation": validation_result,
        "baseline_validation": baseline_validation,
        "rollback_performed": rollback_performed,
        "rollback_diff": rollback_diff,
        "protected_paths_restored": protected_paths_restored,
        "verified_progress": verified_progress,
        "neutral_preparation": retain_neutral,
        "error": error,
        "changed_paths": changed_paths[:500],
        "finished_at": time.time(),
    }


def _record_feedback(
    models: list[dict],
    success: bool,
    reason: str,
    *,
    previous_checkpoint: dict | None = None,
    verified_progress: bool = False,
    failed_output: str = "",
) -> int | None:
    feedback = ModelFeedbackStore()
    previous_event_id = int((previous_checkpoint or {}).get("correction_event_id") or 0)
    if success and previous_event_id:
        feedback.resolve_correction(
            previous_event_id,
            f"A subsequent C0d3rV2+ATF attempt passed validation: {reason}",
        )
    if not models:
        return None
    attributable = [
        item for item in models
        if item.get("phase") in {"agent", "fix", "artifact_write"}
    ]
    if not attributable and any(item.get("phase") for item in models):
        return None
    final = (attributable or models)[-1]
    provider = str(final.get("provider") or "")
    model = str(final.get("model") or "")
    if provider and model:
        feedback.record(
            provider, model, success=success or verified_progress,
            reason=(f"verified partial progress: {reason}" if verified_progress else reason),
        )
        if verified_progress:
            return None
        if not success:
            classification = (
                "generated_artifact_regression"
                if reason.lower().startswith("artifact regression rolled back")
                else "generated_artifact_no_progress"
                if reason.lower().startswith("no artifact progress")
                else "generated_artifact_validation_failure"
            )
            return feedback.record_correction(
                provider,
                model,
                classification=classification,
                is_hallucination=True,
                trigger=reason,
                failed_output=failed_output,
                correction="queued for corrective retry",
                resolved=False,
                metadata={"source": "atf_workday_validation"},
            )
    return None


def _workspace_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    ignored = {
        ".git", "node_modules", "__pycache__", ".pytest_cache", ".venv",
        ".angular", "dist", "coverage",
    }
    snapshot: dict[str, tuple[int, int]] = {}
    try:
        for path in root.rglob("*"):
            if not path.is_file() or any(part in ignored for part in path.relative_to(root).parts):
                continue
            stat = path.stat()
            snapshot[str(path.relative_to(root))] = (int(stat.st_size), int(stat.st_mtime_ns))
    except OSError:
        pass
    return snapshot


def _typescript_contract_envelope(root: Path, prompt: str, *, max_chars: int = 7000) -> str:
    """Inject only task-relevant TypeScript contracts into small-model workday turns."""
    candidates: dict[str, Path] = {}
    try:
        for path in root.rglob("*.ts"):
            relative = path.relative_to(root)
            if any(part in {"node_modules", "dist", "coverage"} for part in relative.parts):
                continue
            candidates[relative.as_posix().casefold()] = path
    except OSError:
        return ""

    lowered = prompt.casefold()
    selected: dict[str, Path] = {}
    explicit = {
        value.replace("\\", "/").casefold()
        for value in re.findall(r"[A-Za-z0-9_./\\-]+\.tsx?", prompt, flags=re.IGNORECASE)
    }
    for relative, path in candidates.items():
        stem = path.stem.casefold()
        if (
            relative in explicit
            or any(relative.endswith("/" + value) or relative == value for value in explicit)
            or re.search(rf"(?<![\w$]){re.escape(stem)}(?![\w$])", lowered)
        ):
            selected[relative] = path

    # One import hop captures the actual input/output types used by a class under repair.
    for relative, path in list(selected.items()):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for specifier in re.findall(r"from\s+['\"]([^'\"]+)['\"]", source):
            if not specifier.startswith("."):
                continue
            target = (path.parent / specifier).resolve()
            for suffix in (".ts", ".tsx", "/index.ts"):
                resolved = Path(str(target) + suffix) if not suffix.startswith("/") else target / suffix[1:]
                try:
                    key = resolved.relative_to(root.resolve()).as_posix().casefold()
                except ValueError:
                    continue
                if key in candidates:
                    selected[key] = candidates[key]
                    break

    if not selected:
        return ""
    header = (
        "EXISTING TYPESCRIPT CONTRACT ENVELOPE (ground truth):\n"
        "Use these exact import paths, constructor arguments, members, and return shapes. "
        "Do not invent aliases, static helpers, wrappers, or test matchers. If a new class contract "
        "is not specified, define its input/output shape first and encode it in tests before implementation."
    )
    chunks = [header]
    used = len(header)
    for relative, path in sorted(selected.items()):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        allowance = max_chars - used - len(relative) - 30
        if allowance <= 200:
            break
        if len(source) > allowance:
            source = source[:allowance] + "\n/* contract truncated */"
        chunk = f"\n--- {path.relative_to(root).as_posix()} ---\n{source}"
        chunks.append(chunk)
        used += len(chunk)
    return "".join(chunks)


def _workspace_contents(root: Path) -> dict[str, bytes]:
    ignored = {
        ".git", "node_modules", "__pycache__", ".pytest_cache", ".venv",
        ".angular", "dist", "coverage",
    }
    contents: dict[str, bytes] = {}
    total = 0
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part in ignored for part in relative.parts):
                continue
            size = path.stat().st_size
            if size > 2_000_000 or total + size > 20_000_000:
                continue
            contents[str(relative)] = path.read_bytes()
            total += size
    except OSError:
        pass
    return contents


def _restore_workspace(root: Path, contents: dict[str, bytes]) -> None:
    current = _workspace_contents(root)
    for relative in set(current) - set(contents):
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
            path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
    for relative, data in contents.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except (OSError, ValueError):
            pass


def _restore_benchmark_tests(root: Path, contents: dict[str, bytes]) -> list[str]:
    """Keep initial benchmark tests immutable across validator-driven repairs."""
    root = root.resolve()
    protected_before = {
        relative: data for relative, data in contents.items()
        if "tests" in Path(relative).parts
    }
    current = _workspace_contents(root)
    protected_now = {
        relative for relative in current
        if "tests" in Path(relative).parts
    }
    changed: set[str] = set()
    for relative in protected_now - set(protected_before):
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            path.unlink(missing_ok=True)
            changed.add(Path(relative).as_posix())
        except (OSError, ValueError):
            pass
    for relative, data in protected_before.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
            if not path.exists() or path.read_bytes() != data:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                changed.add(Path(relative).as_posix())
        except (OSError, ValueError):
            pass
    return sorted(changed)


def _workspace_diff(root: Path, before: dict[str, bytes], changed_paths: list[str]) -> str:
    """Capture a compact text diff before transactional rollback."""
    chunks: list[str] = []
    for relative in changed_paths[:30]:
        old = before.get(relative, b"")
        path = root / relative
        try:
            new = path.read_bytes() if path.exists() else b""
            old_text = old.decode("utf-8")
            new_text = new.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        chunks.extend(unified_diff(
            old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"before/{Path(relative).as_posix()}",
            tofile=f"after/{Path(relative).as_posix()}",
        ))
        if sum(len(item) for item in chunks) >= 12000:
            break
    return "".join(chunks)[:12000]


def _python_contract_summary(root: Path) -> str:
    """Compare test call arities with production function signatures."""
    definitions: dict[str, tuple[int, int | None, str]] = {}
    calls: dict[str, set[int]] = {}
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        is_test = "tests" in path.relative_to(root).parts or path.name.startswith("test_")
        for node in ast.walk(tree):
            if is_test and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.setdefault(node.func.id, set()).add(len(node.args))
            elif not is_test and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = list(node.args.posonlyargs) + list(node.args.args)
                if positional and positional[0].arg in {"self", "cls"}:
                    positional = positional[1:]
                required = max(0, len(positional) - len(node.args.defaults))
                maximum = None if node.args.vararg else len(positional)
                definitions[node.name] = (required, maximum, path.relative_to(root).as_posix())
    lines: list[str] = []
    for name in sorted(set(calls) & set(definitions)):
        required, maximum, relative = definitions[name]
        invalid = sorted(
            count for count in calls[name]
            if count < required or (maximum is not None and count > maximum)
        )
        if invalid:
            accepted = f"{required}+" if maximum is None else str(required) if required == maximum else f"{required}..{maximum}"
            lines.append(
                f"- {name}: tests call positional arities {sorted(calls[name])}; "
                f"implementation at {relative} accepts {accepted}. Repair implementation, not tests."
            )
    return "\n".join(lines[:30])


def _safe_import_only_change(root: Path, before: dict[str, bytes], changed_paths: list[str]) -> bool:
    """Recognize compile-safe Python changes that only add/remove imports."""
    if not changed_paths:
        return False

    def without_imports(source: str) -> str:
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        return ast.dump(tree, include_attributes=False)

    for relative in changed_paths:
        if Path(relative).suffix.lower() != ".py" or relative not in before:
            return False
        try:
            old_source = before[relative].decode("utf-8")
            new_source = (root / relative).read_text(encoding="utf-8")
            if without_imports(old_source) != without_imports(new_source):
                return False
            compile(new_source, str(root / relative), "exec")
        except (OSError, UnicodeDecodeError, SyntaxError):
            return False
    return True


def _validation_severity(code: int, stdout: str, stderr: str) -> int:
    if code == 0:
        return 0
    text = f"{stdout}\n{stderr}"
    try:
        payload = json.loads(stdout or "{}")
    except Exception:
        payload = {}
    errors = payload.get("errors") if isinstance(payload, dict) else []
    evidence = payload.get("evidence") if isinstance(payload, dict) else []
    score = 10 * len(errors or [])
    score += 5 * sum(1 for item in (evidence or []) if isinstance(item, dict) and not item.get("ok", False))
    serialized = json.dumps(payload, default=str)
    score += 3 * sum(int(value) for value in re.findall(r"(?:errors|failures)=(\d+)", serialized))
    if any(marker in serialized for marker in (
        "ModuleNotFoundError", "SyntaxError", "Failed to import test module",
        "Ran 0 tests", "NO TESTS RAN",
    )):
        score += 100
    score += len(re.findall(r"(?m)^(?:FAIL|ERROR):", text))
    return max(1, score)


def _hidden_checks_passed(stdout: str) -> bool:
    """Return true only when validator JSON explicitly reports all hidden checks passing."""
    try:
        payload = json.loads(stdout or "{}")
    except Exception:
        return False
    evidence = payload.get("evidence") if isinstance(payload, dict) else []
    hidden = []
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        command = item.get("command") or ""
        command_text = " ".join(map(str, command)) if isinstance(command, list) else str(command)
        if "<hidden" in command_text.lower():
            hidden.append(item)
    return bool(hidden) and all(item.get("ok") is True for item in hidden)


def _validation_progress_metric(stdout: str, stderr: str) -> int:
    """Measure advancement through sequential test/hidden assertions on severity ties."""
    text = f"{stdout}\n{stderr}"
    try:
        payload = json.loads(stdout or "{}")
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        text += "\n" + "\n".join(str(item) for item in (payload.get("errors") or []))
        text += "\n" + "\n".join(
            str(item.get("output") or "")
            for item in (payload.get("evidence") or []) if isinstance(item, dict)
        )
    passed = len(re.findall(r"(?m)^test_.+\.\.\. ok$", text))
    hidden_lines = [int(value) for value in re.findall(r'File "<string>", line (\d+)', text)]
    test_lines = [int(value) for value in re.findall(r'File "[^"]*test[^"\\/]*\.py", line (\d+)', text)]
    build_stage = 0
    for marker, score in (
        ("npm install", 100),
        ("ng.cmd run", 200),
        ("Generating browser application bundles", 300),
        ("Browser application bundle generation complete", 400),
        ("Initial chunk files", 500),
        ("Build at:", 600),
    ):
        if marker.lower() in text.lower():
            build_stage = max(build_stage, score)
    ts_errors = len(re.findall(r"\berror TS\d+:", text))
    # A compiler error is negative evidence. The former 10,000-point bonus for
    # having one TypeScript error made an early missing-config failure outrank a
    # build that had reached bundle generation.
    typescript_penalty = min(ts_errors, 9_999)
    return (
        passed * 10_000 + max(hidden_lines or [0]) * 100
        + max(test_lines or [0]) + build_stage - typescript_penalty
    )


def _retry_delay(error: str, attempts: int, base: int, quota_delay: int) -> int:
    lowered = error.lower()
    if any(marker in lowered for marker in ("quota", "rate limit", "429", "no eligible model")):
        return quota_delay
    return min(3600, base * (2 ** max(0, attempts - 1)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one leased ATF workday job")
    parser.add_argument("--db", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--lease-seconds", type=int, required=True)
    parser.add_argument("--heartbeat-seconds", type=int, required=True)
    parser.add_argument("--retry-seconds", type=int, required=True)
    parser.add_argument("--quota-retry-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    return run_job(
        db_path=args.db, job_id=args.job, owner=args.owner,
        lease_seconds=args.lease_seconds, heartbeat_seconds=args.heartbeat_seconds,
        retry_seconds=args.retry_seconds, quota_retry_seconds=args.quota_retry_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
