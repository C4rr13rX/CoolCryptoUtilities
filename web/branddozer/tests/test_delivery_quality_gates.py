import json
from types import SimpleNamespace

from services.branddozer_delivery import (
    _concise_process_diagnostics,
    _compact_validation_evidence,
    _foundation_quality_gaps,
    _is_transient_atf_capacity_failure,
    _run_smoke_test,
    _validator_repair_paths,
)
from tools.c0d3rV2.tool_registry import EnvironmentBootstrapTool


def _write_valid_foundation(root):
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "node_modules").mkdir()
    (root / "package.json").write_text(
        json.dumps(
            {
                "scripts": {
                    "build": "node -e \"process.exit(0)\"",
                    "test": "node -e \"process.exit(0)\"",
                    "typecheck": "node -e \"process.exit(0)\"",
                },
                "dependencies": {"three": "test"},
            }
        ),
        encoding="utf-8",
    )
    (root / "tsconfig.json").write_text(
        json.dumps({"compilerOptions": {"strict": True}}), encoding="utf-8"
    )
    (root / "src" / "main.ts").write_text(
        "import * as THREE from 'three'; new THREE.Scene(); new THREE.WebGLRenderer();",
        encoding="utf-8",
    )
    (root / "tests" / "foundation.test.ts").write_text("export {};", encoding="utf-8")
    (root / "package-lock.json").write_text("{}", encoding="utf-8")


def test_foundation_quality_gate_rejects_placeholder_scaffold(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"build": "vite"}, "dependencies": {}}), encoding="utf-8"
    )
    (tmp_path / "src" / "main.jsx").write_text("<h1>ready</h1>", encoding="utf-8")

    gaps = _foundation_quality_gaps(tmp_path)

    assert "package.json does not declare the three dependency" in gaps
    assert "src contains no TypeScript .ts/.tsx files" in gaps
    assert "no TypeScript unit/smoke test file exists" in gaps


def test_foundation_quality_gate_rejects_placeholder_and_counterfeit_renderer(tmp_path):
    _write_valid_foundation(tmp_path)
    (tmp_path / "src" / "main.ts").write_text(
        "import * as THREE from 'three'; new THREE.Scene(); "
        "// placeholder\n({}) as unknown as THREE.WebGLRenderer;",
        encoding="utf-8",
    )
    gaps = _foundation_quality_gaps(tmp_path)
    assert any("placeholder" in gap for gap in gaps)
    assert any("unsafe double assertion" in gap for gap in gaps)


def test_foundation_quality_gate_rejects_test_environment_shim_in_production(tmp_path):
    _write_valid_foundation(tmp_path)
    (tmp_path / "src" / "main.ts").write_text(
        "import * as THREE from 'three'; new THREE.Scene(); "
        "HTMLCanvasElement.prototype.getContext = () => ({}) as WebGLRenderingContext; "
        "new THREE.WebGLRenderer();",
        encoding="utf-8",
    )
    gaps = _foundation_quality_gaps(tmp_path)
    assert any("test-environment shim" in gap for gap in gaps)


def test_foundation_quality_gate_accepts_verified_contract(tmp_path):
    _write_valid_foundation(tmp_path)
    assert _foundation_quality_gaps(tmp_path) == []


def test_atf_provider_pool_timeout_is_capacity_not_product_failure():
    assert _is_transient_atf_capacity_failure(
        "AgentTheFreeloader exhausted eligible fallbacks: provider exceeded hard wall-clock deadline"
    )
    assert not _is_transient_atf_capacity_failure("TypeScript compilation failed")


def test_validator_dependency_failure_targets_ecosystem_manifest(tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    paths = _validator_repair_paths(
        "npm error ETARGET No matching version found for vite@^5.5.0", tmp_path,
    )
    assert paths == ["package.json"]


def test_validator_source_failures_and_manifest_mapping_compose(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]", encoding="utf-8")
    paths = _validator_repair_paths(
        "cargo test failed at src/core/engine.rs:12", tmp_path,
    )
    assert paths == ["Cargo.toml", "src/core/engine.rs"]


def test_validator_quality_gaps_map_to_exact_offending_sources(tmp_path):
    source = tmp_path / "src" / "core"
    source.mkdir(parents=True)
    (source / "robot.ts").write_text("// Placeholder lifecycle", encoding="utf-8")
    (source / "scene.ts").write_text(
        "// jsdom fallback with dummy renderer", encoding="utf-8",
    )
    (source / "clean.ts").write_text("export const clean = true;", encoding="utf-8")
    paths = _validator_repair_paths(
        "quality_gaps: source still identifies its implementation as a placeholder; "
        "production source embeds a test-environment shim or platform mock",
        tmp_path,
    )
    assert paths == ["src/core/robot.ts", "src/core/scene.ts"]


def test_validator_counterfeit_renderer_gap_maps_to_source(tmp_path):
    source = tmp_path / "src" / "core"
    source.mkdir(parents=True)
    (source / "scene.ts").write_text(
        "const renderer = {} as unknown as THREE.WebGLRenderer;", encoding="utf-8",
    )
    assert _validator_repair_paths(
        "source counterfeits a WebGLRenderer with an unsafe double assertion", tmp_path,
    ) == ["src/core/scene.ts"]


def test_process_diagnostics_strip_ansi_and_keep_distinct_file_failures():
    stderr = (
        "\x1b[31mFAIL\x1b[0m tests/main.test.ts\n"
        "tests/main.test.ts:15:5: ERROR: Expected ';' but found 'container'\n"
        "\x1b[31mFAIL\x1b[0m tests/smoke.test.ts\n"
        "Error: Error creating WebGL context.\n"
        "src/core/scene.ts:11:20\n"
    )
    result = _concise_process_diagnostics("", stderr)
    assert "\x1b" not in result
    assert "tests/main.test.ts:15:5" in result
    assert "src/core/scene.ts:11:20" in result


def test_compact_validation_evidence_keeps_errors_and_quality_gates():
    raw = (
        "noise\n" * 100
        + 'Error: WebGL context failed at src/core/scene.ts:29:20\n'
        + '"quality_gaps": ["placeholder", "production test shim"]'
    )
    result = _compact_validation_evidence(raw, limit=600)
    assert "src/core/scene.ts:29:20" in result
    assert "placeholder" in result
    assert len(result) <= 600


def test_compound_smoke_command_is_split_without_a_shell(tmp_path):
    _write_valid_foundation(tmp_path)
    run = SimpleNamespace(
        context={"smoke_test_cmd": "npm test && npm run build"},
        project=None,
    )

    result = _run_smoke_test(
        run,
        tmp_path,
        step_title="Set up TypeScript/Three.js project foundation",
    )

    assert result["status"] == "passed"
    assert len(result["commands"]) == 2


def test_compound_smoke_collects_all_failures_in_one_pass(tmp_path, monkeypatch):
    _write_valid_foundation(tmp_path)
    calls = []

    def failed_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr=f"failure: {' '.join(command)}")

    monkeypatch.setattr("services.branddozer_delivery.subprocess.run", failed_run)
    run = SimpleNamespace(
        context={"smoke_test_cmd": "npm run typecheck && npm test && npm run build"},
        project=None,
    )

    result = _run_smoke_test(
        run, tmp_path, step_title="Set up TypeScript/Three.js project foundation",
    )

    assert result["status"] == "failed"
    assert len(result["commands"]) == 3
    assert len(calls) == 3


def test_environment_bootstrap_refuses_to_overwrite_existing_project(tmp_path):
    manifest = tmp_path / "package.json"
    manifest.write_text('{"name":"keep-me"}', encoding="utf-8")

    result = EnvironmentBootstrapTool(tmp_path).execute({"preset": "react_vite"})

    assert "Non-destructive bootstrap refused" in result["error"]
    assert manifest.read_text(encoding="utf-8") == '{"name":"keep-me"}'
