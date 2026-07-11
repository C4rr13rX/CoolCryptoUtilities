from __future__ import annotations

import json
import time
from dataclasses import replace

from tools.c0d3rV2.plugins.agent_the_freeloader.adapters import ProviderResponse
from tools.c0d3rV2.plugins.agent_the_freeloader.feedback import ModelFeedbackStore
from tools.c0d3rV2.plugins.agent_the_freeloader.models import ModelSpec, PoolLimit
from tools.c0d3rV2.plugins.agent_the_freeloader.quota import QuotaLedger
from tools.c0d3rV2.plugins.agent_the_freeloader.router import FreeloaderRouter, classify_request
from tools.c0d3rV2.plugins.agent_the_freeloader.workday import (
    WorkdayConfig,
    WorkdayStore,
    WorkdaySupervisor,
)
from tools.c0d3rV2.plugins.agent_the_freeloader import workday_worker
from tools.c0d3rV2.plugins.agent_the_freeloader.notifications import WorkdayNotifier


def _spec(model: str, key: str = "ATF_WORKDAY_TEST_KEY") -> ModelSpec:
    return ModelSpec(
        provider="Test Provider",
        model_id=model,
        modalities=frozenset({"text"}),
        best_at="tests",
        base_url="https://example.invalid/v1",
        endpoint="chat/completions",
        api_style="OpenAI-compatible",
        api_key_env=key,
        pool_ids=(model,),
        limits=PoolLimit(requests_per_day=100),
        capabilities={
            "general": 0.7, "coding": 0.8, "tools": 0.8,
            "reasoning": 0.7, "structured": 0.7, "speed": 0.6,
            "multimodal": 0.1, "multilingual": 0.5,
        },
    )


def test_feedback_persists_and_changes_factor(tmp_path) -> None:
    path = tmp_path / "feedback.sqlite3"
    first = ModelFeedbackStore(path)
    assert first.factor("P:M") == 1.0
    first.record("P", "M", success=False, reason="tests failed")
    second = ModelFeedbackStore(path)
    assert second.factor("P:M") < 1.0
    second.record("P", "M", success=True, reason="tests passed")
    assert second.factor("P:M") == 1.0


def test_correction_events_persist_model_attribution(tmp_path) -> None:
    path = tmp_path / "feedback.sqlite3"
    store = ModelFeedbackStore(path)
    event_id = store.record_correction(
        "Test Provider", "bad-model", session_name="benchmark-1",
        classification="premature_completion", is_hallucination=True,
        trigger="claimed completion without executor evidence",
        failed_output="done", correction="run tests", resolved=True,
        metadata={"branch": "validate"},
    )
    event = ModelFeedbackStore(path).correction_snapshot()[0]
    assert event["id"] == event_id
    assert event["provider"] == "Test Provider"
    assert event["model"] == "bad-model"
    assert event["is_hallucination"] is True
    assert event["resolved"] is True
    assert event["metadata"]["branch"] == "validate"
    assert store.resolve_correction(event_id, "fixed by a validated retry")
    resolved = store.correction_snapshot()[0]
    assert resolved["resolved"] is True
    assert resolved["correction"] == "fixed by a validated retry"


def test_router_uses_semantic_feedback_for_equivalent_models(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ATF_WORKDAY_TEST_KEY", "configured")
    one = _spec("one")
    two = replace(_spec("two"), capabilities=dict(one.capabilities))
    feedback = ModelFeedbackStore(tmp_path / "feedback.sqlite3")
    for _ in range(4):
        feedback.record("Test Provider", "one", success=False, reason="validation")
    router = FreeloaderRouter(
        [one, two],
        QuotaLedger({"one": one.limits, "two": two.limits}),
        feedback=feedback,
        invoker=lambda spec, **kwargs: ProviderResponse(spec.model_id, {}, 1, 1),
    )
    ranked = router.rank(classify_request("implement code", "file_write", max_tokens=500))
    assert ranked[0].spec.model_id == "two"


def test_quota_usage_deduplicates_shared_reservation(tmp_path) -> None:
    ledger = QuotaLedger(
        {"provider": PoolLimit(), "model": PoolLimit()},
        state_path=tmp_path / "quota.json",
    )
    reservation = ledger.reserve(("provider", "model"), 50)
    ledger.reconcile(reservation, 75)
    assert ledger.usage_since() == {"requests": 1, "tokens": 75}


def test_store_claim_heartbeat_complete_and_report(tmp_path) -> None:
    store = WorkdayStore(tmp_path / "workday.sqlite3")
    job_id = store.enqueue("Make a file", workdir=tmp_path, validation_command="echo ok")
    job = store.claim("worker-1", 30)
    assert job and job["id"] == job_id and job["attempts"] == 1
    assert store.heartbeat(job_id, "worker-1", 30)
    store.checkpoint(job_id, "worker-1", {"output": "working"})
    store.complete(job_id, "worker-1", {"success": True})
    completed = store.get(job_id)
    assert completed and completed["status"] == "completed"
    assert completed["result"]["success"] is True


def test_store_reclaims_expired_lease_and_honors_cancellation(tmp_path) -> None:
    store = WorkdayStore(tmp_path / "workday.sqlite3")
    queued = store.enqueue("queued", workdir=tmp_path)
    assert store.cancel(queued)
    assert store.get(queued)["status"] == "cancelled"

    running = store.enqueue("running", workdir=tmp_path, max_attempts=2)
    claimed = store.claim("dead-worker", 30)
    assert claimed and claimed["id"] == running
    with store._connect() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at=? WHERE id=?", (time.time() - 1, running)
        )
    reclaimed = store.reclaim_expired()
    assert [job["id"] for job in reclaimed] == [running]
    assert store.get(running)["status"] == "retry"


def test_store_can_requeue_failed_job_with_checkpoint(tmp_path) -> None:
    store = WorkdayStore(tmp_path / "workday.sqlite3")
    job_id = store.enqueue("repair", workdir=tmp_path, max_attempts=1)
    job = store.claim("owner", 30)
    assert job and job["id"] == job_id
    store.checkpoint(job_id, "owner", {"correction_event_id": 42, "error": "tests failed"})
    store.retry(job_id, "owner", "tests failed", 0, store.get(job_id)["checkpoint"])
    assert store.get(job_id)["status"] == "failed"
    assert store.requeue(job_id, extra_attempts=2)
    resumed = store.get(job_id)
    assert resumed["status"] == "retry"
    assert resumed["max_attempts"] == 3
    assert resumed["checkpoint"]["correction_event_id"] == 42


def test_capacity_deferral_does_not_consume_attempt(tmp_path) -> None:
    store = WorkdayStore(tmp_path / "workday.sqlite3")
    job_id = store.enqueue("wait", workdir=tmp_path, max_attempts=1)
    claimed = store.claim("owner", 30)
    assert claimed and claimed["attempts"] == 1
    store.defer_for_capacity(job_id, "owner", "no eligible model", 30, {"capacity_wait": True})
    deferred = store.get(job_id)
    assert deferred["status"] == "retry"
    assert deferred["attempts"] == 0
    assert deferred["checkpoint"]["capacity_wait"] is True


def test_worker_validation_failure_retries_and_records_checkpoint(monkeypatch, tmp_path) -> None:
    store = WorkdayStore(tmp_path / "workday.sqlite3")
    job_id = store.enqueue("test", workdir=tmp_path, max_attempts=2)
    store.claim("owner", 30)
    monkeypatch.setattr(
        workday_worker,
        "_execute",
        lambda job, job_id: {
            "success": False,
            "output": "bad",
            "models": [],
            "validation": {"return_code": 1},
            "error": "validation failed",
        },
    )
    code = workday_worker.run_job(
        db_path=store.path, job_id=job_id, owner="owner",
        lease_seconds=30, heartbeat_seconds=5,
        retry_seconds=1, quota_retry_seconds=10,
    )
    job = store.get(job_id)
    assert code == 1
    assert job["status"] == "retry"
    assert job["checkpoint"]["error"] == "validation failed"


def test_worker_retry_prompt_includes_validation_error(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(
        "tools.c0d3rV2.delivery_runner.run_delivery_turn_detailed",
        lambda prompt, **kwargs: captured.update(prompt=prompt) or {"output": "fixed", "models": []},
    )
    job = {
        "prompt": "build it", "workdir": str(tmp_path), "validation_command": "",
        "checkpoint": {
            "output": "old model prose", "error": "missing README and zero tests",
            "validation": {"stdout": "Found 0 test(s)."},
        },
    }
    result = workday_worker._execute(job, "job-1")
    assert result["success"] is True
    assert "missing README and zero tests" in captured["prompt"]
    assert "Found 0 test(s)." in captured["prompt"]
    assert "Do not restart" in captured["prompt"]


def test_typescript_contract_envelope_includes_named_dependency_and_import(tmp_path) -> None:
    physics = tmp_path / "src" / "physics"
    physics.mkdir(parents=True)
    (physics / "vec3.ts").write_text(
        "export class Vec3 { constructor(public readonly x: number) {} mul(n: number): Vec3 { return this; } }",
        encoding="utf-8",
    )
    (physics / "Particle.ts").write_text(
        "import { Vec3 } from './vec3';\nexport class Particle { constructor(public position: Vec3) {} }",
        encoding="utf-8",
    )
    envelope = workday_worker._typescript_contract_envelope(
        tmp_path, "Implement Particle.ts using the existing Vec3 API",
    )
    assert "src/physics/Particle.ts" in envelope
    assert "src/physics/vec3.ts" in envelope
    assert "mul(n: number): Vec3" in envelope
    assert "Do not invent aliases" in envelope


def test_typescript_contract_envelope_excludes_unrelated_files(tmp_path) -> None:
    (tmp_path / "Used.ts").write_text("export class Used {}", encoding="utf-8")
    (tmp_path / "Unrelated.ts").write_text("export class Unrelated {}", encoding="utf-8")
    envelope = workday_worker._typescript_contract_envelope(tmp_path, "Repair Used.ts")
    assert "Used.ts" in envelope
    assert "Unrelated.ts" not in envelope


def test_workday_notifier_always_writes_landmark_log(tmp_path) -> None:
    log = tmp_path / "notifications.jsonl"
    notifier = WorkdayNotifier(log, enabled=False)
    assert notifier.send("ATF milestone passed", "journalism", job_id="job-1") is False
    event = json.loads(log.read_text(encoding="utf-8").strip())
    assert event["title"] == "ATF milestone passed"
    assert event["job_id"] == "job-1"


def test_feedback_attributes_only_agent_or_fix_phase(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_FREELOADER_FEEDBACK_PATH", str(tmp_path / "feedback.sqlite3"))
    assert workday_worker._record_feedback(
        [{"provider": "P", "model": "planner", "phase": "planning"}],
        False, "artifact failed",
    ) is None
    assert ModelFeedbackStore(tmp_path / "feedback.sqlite3").snapshot() == []


def test_feedback_rewards_verified_partial_progress_without_hallucination(monkeypatch, tmp_path) -> None:
    path = tmp_path / "feedback.sqlite3"
    monkeypatch.setenv("AGENT_FREELOADER_FEEDBACK_PATH", str(path))
    assert workday_worker._record_feedback(
        [{"provider": "P", "model": "repairer", "phase": "artifact_write"}],
        False, "validation still has one failure", verified_progress=True,
    ) is None
    snapshot = ModelFeedbackStore(path).snapshot()
    assert snapshot[0]["successes"] == 1
    assert snapshot[0]["failures"] == 0
    assert ModelFeedbackStore(path).correction_snapshot() == []


def test_workspace_restore_rolls_back_changed_and_new_files(tmp_path) -> None:
    original = tmp_path / "original.py"
    original.write_text("before", encoding="utf-8")
    backup = workday_worker._workspace_contents(tmp_path)
    original.write_text("after", encoding="utf-8")
    (tmp_path / "new.py").write_text("new", encoding="utf-8")
    workday_worker._restore_workspace(tmp_path, backup)
    assert original.read_text(encoding="utf-8") == "before"
    assert not (tmp_path / "new.py").exists()


def test_workspace_snapshot_ignores_generated_frontend_outputs(tmp_path) -> None:
    (tmp_path / "source.ts").write_text("export const value = 1;", encoding="utf-8")
    for directory in (".angular", "dist", "coverage", "node_modules"):
        target = tmp_path / directory / "generated.js"
        target.parent.mkdir(parents=True)
        target.write_text("generated", encoding="utf-8")
    snapshot = workday_worker._workspace_snapshot(tmp_path)
    assert set(snapshot) == {"source.ts"}


def test_benchmark_test_restore_preserves_sources_and_reverts_tests(tmp_path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    test_file = tests / "test_core.py"
    source_file = tmp_path / "core.py"
    test_file.write_text("original test", encoding="utf-8")
    source_file.write_text("original source", encoding="utf-8")
    backup = workday_worker._workspace_contents(tmp_path)
    test_file.write_text("mutated test", encoding="utf-8")
    (tests / "test_cheat.py").write_text("new test", encoding="utf-8")
    source_file.write_text("repaired source", encoding="utf-8")
    restored = workday_worker._restore_benchmark_tests(tmp_path, backup)
    assert test_file.read_text(encoding="utf-8") == "original test"
    assert not (tests / "test_cheat.py").exists()
    assert source_file.read_text(encoding="utf-8") == "repaired source"
    assert set(restored) == {"tests/test_cheat.py", "tests/test_core.py"}


def test_hidden_checks_passed_requires_explicit_success_for_every_hidden_check() -> None:
    passing = json.dumps({"evidence": [
        {"command": ["python", "<hidden impedance physics>"], "ok": True},
        {"command": ["python", "ordinary tests"], "ok": False},
    ]})
    failing = json.dumps({"evidence": [
        {"command": ["python", "<hidden impedance physics>"], "ok": False},
    ]})
    assert workday_worker._hidden_checks_passed(passing)
    assert not workday_worker._hidden_checks_passed(failing)
    assert not workday_worker._hidden_checks_passed("not json")


def test_validation_progress_tracks_angular_build_stages() -> None:
    missing_config = "ng.cmd run app:build\nerror TS5012: Cannot read file tsconfig.json"
    compiling = (
        "ng.cmd run app:build\nGenerating browser application bundles (phase: setup)...\n"
        "Browser application bundle generation complete.\nError: Can't resolve src/main.ts"
    )
    assert workday_worker._validation_progress_metric(compiling, "") > (
        workday_worker._validation_progress_metric(missing_config, "")
    )


def test_validation_progress_rewards_fewer_typescript_diagnostics() -> None:
    many = "\n".join(f"file.ts({i},1): error TS2307: missing" for i in range(8))
    few = "\n".join(f"file.ts({i},1): error TS2307: missing" for i in range(3))
    assert workday_worker._validation_progress_metric(few, "") > (
        workday_worker._validation_progress_metric(many, "")
    )


def test_workspace_diff_captures_bad_edit_before_rollback(tmp_path) -> None:
    target = tmp_path / "core.py"
    target.write_text("value = 1\n", encoding="utf-8")
    before = workday_worker._workspace_contents(tmp_path)
    target.write_text("value = 'hallucinated'\n", encoding="utf-8")
    diff = workday_worker._workspace_diff(tmp_path, before, ["core.py"])
    assert "-value = 1" in diff
    assert "+value = 'hallucinated'" in diff


def test_python_contract_summary_reports_test_call_arity_mismatch(tmp_path) -> None:
    (tmp_path / "core.py").write_text(
        "def export_to_csv(frequencies, impedances, filename):\n    pass\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text(
        "def test_export():\n    export_to_csv('out.csv', [])\n",
        encoding="utf-8",
    )
    summary = workday_worker._python_contract_summary(tmp_path)
    assert "export_to_csv" in summary
    assert "arities [2]" in summary
    assert "accepts 3" in summary


def test_safe_import_only_change_distinguishes_behavior_edit(tmp_path) -> None:
    target = tmp_path / "core.py"
    target.write_text("import math\n\ndef value():\n    return math.pi\n", encoding="utf-8")
    before = workday_worker._workspace_contents(tmp_path)
    target.write_text("import math\nimport csv\n\ndef value():\n    return math.pi\n", encoding="utf-8")
    assert workday_worker._safe_import_only_change(tmp_path, before, ["core.py"])
    target.write_text("import math\nimport csv\n\ndef value():\n    return 3\n", encoding="utf-8")
    assert not workday_worker._safe_import_only_change(tmp_path, before, ["core.py"])


def test_validation_severity_prefers_passing_and_fewer_errors() -> None:
    assert workday_worker._validation_severity(0, "ok", "") == 0
    one = '{"errors":["one"],"evidence":[{"ok":false}]}'
    two = '{"errors":["one","two"],"evidence":[{"ok":false},{"ok":false}]}'
    assert workday_worker._validation_severity(1, one, "") < workday_worker._validation_severity(1, two, "")


def test_validation_severity_treats_import_blocker_as_worse_than_assertion_failures() -> None:
    import_blocker = '{"errors":["ModuleNotFoundError: No module named instrument"]}'
    behavior_failures = (
        '{"errors":["three behavioral assertions failed"],'
        '"evidence":[{"ok":false},{"ok":false},{"ok":false}]}'
    )
    assert workday_worker._validation_severity(
        1, import_blocker, ""
    ) > workday_worker._validation_severity(1, behavior_failures, "")


def test_validation_progress_metric_detects_later_sequential_invariant() -> None:
    before = '''test_a ... ok
Traceback (most recent call last):
  File "C:\\project\\tests\\test_core.py", line 43, in test_invalid
  File "<string>", line 29, in <module>
'''
    after = '''test_a ... ok
Traceback (most recent call last):
  File "C:\\project\\tests\\test_core.py", line 49, in test_invalid
  File "<string>", line 32, in <module>
'''
    assert workday_worker._validation_progress_metric(
        after, ""
    ) > workday_worker._validation_progress_metric(before, "")
    before_json = json.dumps({"errors": [before], "evidence": [{"output": before}]})
    after_json = json.dumps({"errors": [after], "evidence": [{"output": after}]})
    assert workday_worker._validation_progress_metric(
        after_json, ""
    ) > workday_worker._validation_progress_metric(before_json, "")


def test_supervisor_drains_queue_and_writes_shift_report(monkeypatch, tmp_path) -> None:
    config = WorkdayConfig(
        db_path=tmp_path / "workday.sqlite3",
        concurrency=1,
        lease_seconds=30,
        heartbeat_seconds=5,
        poll_seconds=0.01,
        job_timeout_seconds=60,
        retry_base_seconds=1,
        quota_retry_seconds=1,
        max_requests_per_day=0,
        max_tokens_per_day=0,
        shift_hours=1,
        report_dir=tmp_path / "reports",
    )
    supervisor = WorkdaySupervisor(config)
    job_id = supervisor.store.enqueue("atomic", workdir=tmp_path)

    def complete_immediately(job):
        supervisor.store.complete(job["id"], supervisor.owner, {"success": True})

    monkeypatch.setattr(supervisor, "_launch", complete_immediately)
    report = supervisor.run(until_empty=True, max_runtime_seconds=2)
    assert supervisor.store.get(job_id)["status"] == "completed"
    assert report["stop_reason"] == "queue drained"
    assert list((tmp_path / "reports").glob("workday_*.md"))
    assert report["stats"]["by_status"] == {"completed": 1}
