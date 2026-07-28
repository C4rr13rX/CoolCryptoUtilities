from pathlib import Path

from tools.c0d3rV2.repair_packet import advance_repair_state, failure_signals


def _failure(message: str, quality=None):
    return {
        "status": "failed",
        "command": "npm test && npm run build",
        "commands": [{"exit_code": 1, "stderr": message, "stdout": ""}],
        "quality_gaps": quality or [],
    }


def test_repeated_validator_outcome_descends_to_method_scope(tmp_path: Path):
    source = tmp_path / "src" / "engine.ts"
    source.parent.mkdir()
    source.write_text(
        "export interface StepInput { dt: number }\n"
        "export class Engine { step(input: StepInput): number { return input.dt; } }",
        encoding="utf-8",
    )
    state = {}
    for _ in range(3):
        state, packet = advance_repair_state(
            state,
            _failure("Error: expected 2 but received 1 at src/engine.ts:2"),
            step="repair engine",
            paths=["src/engine.ts"],
            root=tmp_path,
        )
    assert packet.scope_level == "method"
    assert any("StepInput" in item for item in packet.contracts)
    assert "one method" in packet.to_dict()["required_transition"]


def test_repair_state_remembers_resolved_failures_and_detects_recurrence(tmp_path: Path):
    dependency = _failure("MISSING DEPENDENCY Cannot find dependency 'jsdom'")
    state, _ = advance_repair_state(
        {}, dependency, step="foundation", paths=["package.json"], root=tmp_path,
    )
    state, packet = advance_repair_state(
        state,
        _failure("Error: WebGL context unavailable"),
        step="foundation", paths=["src/scene.ts"], root=tmp_path,
    )
    assert any("jsdom" in item["message"].lower() for item in packet.resolved_failures)
    state, packet = advance_repair_state(
        state, dependency, step="foundation", paths=["package.json"], root=tmp_path,
    )
    assert packet.recurrence_count == 1


def test_failure_signals_normalize_volatile_duration():
    first = failure_signals(_failure("Error after 1200ms: expected 2 but got 1"))
    second = failure_signals(_failure("Error after 4800ms: expected 2 but got 1"))
    assert first[0]["id"] == second[0]["id"]


def test_multi_failure_packet_focuses_dependency_before_source_quality(tmp_path: Path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "src" / "robot.ts"
    source.parent.mkdir()
    source.write_text("// placeholder", encoding="utf-8")
    _, packet = advance_repair_state(
        {},
        _failure("MISSING DEPENDENCY Cannot find dependency 'jsdom'", ["source has placeholder"]),
        step="foundation",
        paths=["package.json", "src/robot.ts"],
        root=tmp_path,
    )
    assert packet.focus_failure["kind"] == "dependency"
    assert packet.focus_paths == ["package.json"]
    assert packet.to_dict()["deferred_failures"][0]["kind"] == "quality"


def test_failure_signal_prefers_specific_source_location_over_test_summary(tmp_path: Path):
    scene = tmp_path / "src" / "core" / "scene.ts"
    scene.parent.mkdir(parents=True)
    scene.write_text(
        "export interface RendererPort { render(): void }\n"
        "export class SceneHost { create(port: RendererPort): void { port.render(); } }",
        encoding="utf-8",
    )
    result = _failure(
        "Test Files 2 failed (2)\n"
        "TypeError: appendChild parameter is not a Node\n"
        "src/core/scene.ts:50:19"
    )
    _, packet = advance_repair_state(
        {}, result, step="foundation", paths=["package.json", "src/core/scene.ts"], root=tmp_path,
    )
    assert "src/core/scene.ts" in packet.focus_failure["message"]
    assert packet.focus_paths == ["src/core/scene.ts"]
    assert all("BoxGeometry" not in item for item in packet.contracts)


def test_failure_signal_preserves_source_and_failing_consumer_locations(tmp_path: Path):
    for relative in ("src/core/scene.ts", "tests/smoke.test.ts"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("export const value = 1;", encoding="utf-8")
    result = _failure(
        "Error: WebGL unavailable\n"
        "src/core/scene.ts:16:22\n"
        "src/core/scene.ts:58:20\n"
        "tests/smoke.test.ts:21:23\n"
    )

    _, packet = advance_repair_state(
        {}, result, step="foundation",
        paths=["src/core/scene.ts", "tests/smoke.test.ts"], root=tmp_path,
    )

    assert "src/core/scene.ts" in packet.focus_failure["message"]
    assert "tests/smoke.test.ts" in packet.focus_failure["message"]
    assert packet.focus_paths == ["src/core/scene.ts", "tests/smoke.test.ts"]


def test_failure_signal_preserves_composition_root_in_causal_chain(tmp_path: Path):
    paths = ["src/core/scene.ts", "src/main.ts", "tests/main.test.ts"]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("export const value = 1;", encoding="utf-8")
    result = _failure(
        "Error: WebGL unavailable\n"
        "src/core/scene.ts:17:22\n"
        "src/main.ts:10:15\n"
        "tests/main.test.ts:3:31\n"
    )

    _, packet = advance_repair_state(
        {}, result, step="foundation", paths=paths, root=tmp_path,
    )

    assert packet.focus_paths == paths
    assert all(path in packet.focus_failure["message"] for path in paths)


def test_platform_stack_through_main_recommends_pure_import_boundary(tmp_path: Path):
    paths = ["src/core/scene.ts", "src/main.ts", "tests/main.test.ts", "tests/smoke.test.ts"]
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("export const value = 1;", encoding="utf-8")
    _, packet = advance_repair_state(
        {}, _failure(
            "Error: WebGL unavailable\n"
            "src/core/scene.ts:17:22\n"
            "src/main.ts:43:15\n"
            "tests/main.test.ts:3:31\n"
            "tests/smoke.test.ts:7:41\n"
        ), step="foundation", paths=paths, root=tmp_path,
    )
    assert packet.focus_paths == paths
    assert "importing a module must not construct" in packet.recommended_pattern
    assert "bootstrap/composition-root" in packet.recommended_pattern
