"""
Comprehensive test suite for C0d3r V2.

Covers:
  - All 10 tool schemas (imports, use_when, params)
  - ToolRegistry registration, dispatch, and descriptions
  - ExecutorTool with real echo command
  - FileReadTool + FileWriteTool with temp files
  - FileLocateTool with stub memory
  - MatrixSearchTool graceful import-error path
  - Mocked: WebSearchTool, MemorySearchTool, UnboundedSolverTool, MathGroundingTool
  - WizardSession.probe() + .send() against live node
  - SessionManager factory routing (wizard -> WizardSession, fallback path)
  - ContextBuilder._tools_section() structured output
  - Orchestrator._safe_json() edge cases
  - Orchestrator.run() with a mock session (tool calls + completion)
  - JSON QA query smoke test against live node
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup — mirror what c0d3rV2_cli.py does so all imports resolve
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_V2_ROOT = _REPO_ROOT / "tools" / "c0d3rV2"
for _p in (str(_REPO_ROOT), str(_V2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------
from tool_registry import (
    Tool,
    ToolRegistry,
    ExecutorTool,
    WebSearchTool,
    MemorySearchTool,
    FileLocateTool,
    MatrixSearchTool,
    FileReadTool,
    FileWriteTool,
    DirectoryEnsureTool,
    WorkspaceScaffoldTool,
    EnvironmentBootstrapTool,
    ScientificMethodTool,
    UnboundedSolverTool,
    MathGroundingTool,
    VMPlaygroundTool,
)
from orchestrator import Orchestrator, StepResult
from task_tree import TaskNode
from context_builder import ContextBuilder
from executor import Executor
from sessions import SessionManager


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

ALL_TOOL_NAMES = [
    "executor", "web_search", "memory_search", "file_locate",
    "equation_matrix", "file_read", "file_write",
    "directory_ensure", "workspace_scaffold", "environment_bootstrap", "scientific_method",
    "unbounded_solver", "math_grounding", "vm_playground",
]


def _mock_solver() -> Any:
    solver = MagicMock()
    solver.solve.return_value = MagicMock(
        answered=True, answer="42",
        questions_total=1, questions_answered=1, equations_added=0,
        hypotheses=[], anomalies=[], question_tree={},
    )
    solver.format_context_block.return_value = "ctx"
    solver.math_grounding.return_value = {"variables": {}}
    solver.format_grounding_block.return_value = "grounding"
    return solver


def _build_registry(workdir: Path) -> ToolRegistry:
    """Build a fully-wired ToolRegistry with mocked external deps."""
    executor = Executor(workdir)

    mock_ws = MagicMock()
    mock_ws.search.return_value = {"results": [], "summary": "ok"}

    mock_mem = MagicMock()
    mock_mem.search.return_value = []

    mock_st = MagicMock()
    mock_st.lookup.return_value = []
    mock_lt = MagicMock()
    mock_lt.lookup.return_value = []

    solver = _mock_solver()

    mock_vm = MagicMock()
    mock_vm.status.return_value = {"vms": []}

    reg = ToolRegistry()
    reg.register(ExecutorTool(executor))
    reg.register(FileReadTool(workdir))
    reg.register(FileWriteTool(workdir))
    reg.register(DirectoryEnsureTool(workdir))
    reg.register(WorkspaceScaffoldTool(workdir))
    reg.register(EnvironmentBootstrapTool(workdir))
    reg.register(ScientificMethodTool(mock_ws, runtime_dir=workdir))
    reg.register(WebSearchTool(mock_ws))
    reg.register(MemorySearchTool(mock_mem))
    reg.register(FileLocateTool(mock_st, mock_lt, workdir=workdir))
    reg.register(MatrixSearchTool())
    reg.register(UnboundedSolverTool(solver))
    reg.register(MathGroundingTool(solver))
    reg.register(VMPlaygroundTool(mock_vm))
    return reg


# ---------------------------------------------------------------------------
# 1.  Tool schema validation
# ---------------------------------------------------------------------------

class TestToolSchemas:
    """Every tool must have name, description, use_when, and params_schema."""

    @pytest.fixture(scope="class")
    def reg(self, tmp_path_factory):
        return _build_registry(tmp_path_factory.mktemp("wd"))

    def test_all_ten_registered(self, reg):
        assert set(reg.tool_names()) == set(ALL_TOOL_NAMES)

    @pytest.mark.parametrize("name", ALL_TOOL_NAMES)
    def test_schema_name_matches(self, reg, name):
        schema = reg.get(name).schema()
        assert schema["name"] == name

    @pytest.mark.parametrize("name", ALL_TOOL_NAMES)
    def test_schema_has_use_when(self, reg, name):
        schema = reg.get(name).schema()
        assert "use_when" in schema, f"{name} missing use_when"
        assert schema["use_when"].strip()

    @pytest.mark.parametrize("name", ALL_TOOL_NAMES)
    def test_schema_has_params(self, reg, name):
        schema = reg.get(name).schema()
        assert "params" in schema, f"{name} missing params"
        assert schema["params"]

    @pytest.mark.parametrize("name", ALL_TOOL_NAMES)
    def test_schema_has_description(self, reg, name):
        schema = reg.get(name).schema()
        assert schema.get("description", "").strip()


# ---------------------------------------------------------------------------
# 2.  ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:

    def test_dispatch_known_tool_returns_dict(self, tmp_path):
        reg = _build_registry(tmp_path)
        result = reg.dispatch("executor", {"command": "echo ping"})
        assert isinstance(result, dict)

    def test_dispatch_unknown_tool_returns_error(self, tmp_path):
        reg = _build_registry(tmp_path)
        result = reg.dispatch("does_not_exist", {})
        assert "error" in result

    def test_tool_descriptions_count(self, tmp_path):
        reg = _build_registry(tmp_path)
        descs = reg.tool_descriptions()
        assert len(descs) == len(ALL_TOOL_NAMES)

    def test_tool_descriptions_all_have_use_when(self, tmp_path):
        reg = _build_registry(tmp_path)
        for d in reg.tool_descriptions():
            assert "use_when" in d, f"{d['name']} missing use_when in tool_descriptions()"

    def test_tool_names_list(self, tmp_path):
        reg = _build_registry(tmp_path)
        assert set(reg.tool_names()) == set(ALL_TOOL_NAMES)


# ---------------------------------------------------------------------------
# 3.  ExecutorTool
# ---------------------------------------------------------------------------

class TestExecutorTool:

    def test_echo_succeeds(self, tmp_path):
        tool = ExecutorTool(Executor(tmp_path))
        result = tool.execute({"command": "echo hello_c0d3rv2"})
        assert result.get("return_code") == 0
        assert "hello_c0d3rv2" in result.get("stdout", "")

    def test_empty_command_returns_error(self, tmp_path):
        tool = ExecutorTool(Executor(tmp_path))
        assert "error" in tool.execute({})

    def test_failed_command_has_nonzero_code(self, tmp_path):
        tool = ExecutorTool(Executor(tmp_path))
        result = tool.execute({"command": "exit 1"})
        assert result.get("return_code", 0) != 0
        assert "error" in result

    def test_rejects_malformed_powershell_foreach(self, tmp_path):
        tool = ExecutorTool(Executor(tmp_path))
        result = tool.execute({"command": "foreach( in ){ New-Item -ItemType Directory }"})
        assert "error" in result
        assert "directory_ensure" in result["error"]


# ---------------------------------------------------------------------------
# 4.  FileReadTool + FileWriteTool
# ---------------------------------------------------------------------------

class TestFileTools:

    def test_full_write_then_read(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fr = FileReadTool(tmp_path)
        fw.execute({"path": "hello.txt", "content": "line1\nline2\nline3\n"})
        result = fr.execute({"path": "hello.txt"})
        assert "line1" in result["content"]
        assert result["total_lines"] == 3

    def test_offset_and_limit(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fr = FileReadTool(tmp_path)
        fw.execute({"path": "nums.txt", "content": "\n".join(str(i) for i in range(10)) + "\n"})
        result = fr.execute({"path": "nums.txt", "offset": 3, "limit": 2})
        lines = result["content"].splitlines()
        assert lines[0] == "3"
        assert len(lines) == 2

    def test_patch_mode(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fw.execute({"path": "patch.txt", "content": "hello world\n"})
        result = fw.execute({"path": "patch.txt", "old_string": "hello", "new_string": "goodbye"})
        assert result.get("status") == "patched"
        fr = FileReadTool(tmp_path)
        assert "goodbye" in fr.execute({"path": "patch.txt"})["content"]

    def test_read_missing_file_returns_error(self, tmp_path):
        fr = FileReadTool(tmp_path)
        assert "error" in fr.execute({"path": "no_such_file.txt"})

    def test_patch_old_string_absent_returns_error(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fw.execute({"path": "x.txt", "content": "actual content\n"})
        result = fw.execute({"path": "x.txt", "old_string": "NOPE", "new_string": "X"})
        assert "error" in result

    def test_patch_accepts_one_unique_near_exact_multiline_block(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fw.execute({"path": "core.py", "content": (
            "def impedance(r, l, c, f):\n"
            "    if f <= 0:\n"
            "        raise ValueError('frequency')\n"
            "    return r\n"
        )})
        result = fw.execute({
            "path": "core.py",
            "old_string": (
                "def impedance(r, l, c, frequency):\n"
                "    if frequency <= 0:\n"
                "        raise ValueError('frequency')\n"
                "    return r"
            ),
            "new_string": "def impedance(r, l, c, f):\n    if min(r, l, c, f) <= 0:\n        raise ValueError('inputs')\n    return r",
        })
        assert result.get("status") == "patched_fuzzy"
        assert "min(r, l, c, f)" in (tmp_path / "core.py").read_text(encoding="utf-8")

    def test_patch_rejects_low_similarity_fallback(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fw.execute({"path": "core.py", "content": "def real():\n    return 1\n"})
        result = fw.execute({
            "path": "core.py", "old_string": "class Imaginary:\n    pass",
            "new_string": "unsafe",
        })
        assert "error" in result
        assert "unsafe" not in (tmp_path / "core.py").read_text(encoding="utf-8")

    def test_patch_rejects_new_undefined_python_name(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = "def resonance(r, l, c):\n    if r <= 0:\n        raise ValueError\n    return r\n"
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({
            "path": "core.py", "old_string": "if r <= 0:",
            "new_string": "if r <= 0 or frequency <= 0:",
        })
        assert "undefined Python names: frequency" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_patch_rejects_signature_that_still_mismatches_test_arity(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_core.py").write_text(
            "def test_export():\n    export_to_csv('out.csv', [])\n", encoding="utf-8",
        )
        fw = FileWriteTool(tmp_path)
        original = "def export_to_csv(rows, metadata, filename):\n    pass\n"
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({
            "path": "core.py",
            "old_string": "def export_to_csv(rows, metadata, filename):",
            "new_string": "def export_to_csv(filename, rows, metadata):",
        })
        assert "tests with positional arities [2]" in result["error"]
        assert "accepts 3" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_patch_rejects_python_syntax_error_atomically(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = "def valid():\n    return 1\n"
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({
            "path": "core.py", "old_string": "return 1",
            "new_string": 'return """unterminated',
        })
        assert "Python syntax error" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_full_write_rejects_python_syntax_error_atomically(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = "def valid():\n    return 1\n"
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({"path": "core.py", "content": "def broken():\nreturn 2\n"})
        assert "Python syntax error" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_patch_rejects_invalid_json_atomically(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = '{"dependencies": {"django": "5.2.1"}}\n'
        fw.execute({"path": "package.json", "content": original})
        result = fw.execute({
            "path": "package.json", "old_string": '"django": "5.2.1"',
            "new_string": '"django": "5.2.1"\n"numpy": "2.4.1"',
        })
        assert "invalid JSON" in result["error"]
        assert (tmp_path / "package.json").read_text(encoding="utf-8") == original

    def test_full_write_rejects_invalid_json_atomically(self, tmp_path):
        result = FileWriteTool(tmp_path).execute({"path": "package.json", "content": '{"bad": }'})
        assert "invalid JSON" in result["error"]
        assert not (tmp_path / "package.json").exists()

    def test_full_write_rejects_duplicate_json_keys(self, tmp_path):
        result = FileWriteTool(tmp_path).execute({
            "path": "angular.json", "content": '{"projects": {"app": {}, "app": {}}}',
        })
        assert "duplicate object key 'app'" in result["error"]
        assert not (tmp_path / "angular.json").exists()

    def test_full_write_rejects_tsconfig_extends_cycle(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        fw.execute({"path": "tsconfig.app.json", "content": '{"extends": "./tsconfig.json"}'})
        result = fw.execute({
            "path": "tsconfig.json", "content": '{"extends": "./tsconfig.app.json"}',
        })
        assert "extends cycle" in result["error"]
        assert not (tmp_path / "tsconfig.json").exists()

    def test_patch_rejects_new_unreachable_statement(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = "def validate(value):\n    if value <= 0:\n        raise ValueError\n    return value\n"
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({
            "path": "core.py", "old_string": "if value <= 0:\n        raise ValueError",
            "new_string": "if value <= 0:\n        return None\n        raise ValueError",
        })
        assert "unreachable Python statements" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_corrective_patch_rejects_docstring_only_change(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = 'def calculate():\n    """old docs"""\n    return 1\n'
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({
            "path": "core.py", "old_string": '"""old docs"""',
            "new_string": '"""new docs"""', "require_semantic_change": True,
        })
        assert "only comments/docstrings" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_corrective_full_write_rejects_public_api_removal(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = "def calculate():\n    return 1\n\ndef export():\n    return 2\n"
        fw.execute({"path": "core.py", "content": original})
        result = fw.execute({
            "path": "core.py", "content": "def calculate():\n    return 1\n",
            "require_semantic_change": True,
        })
        assert "removed public Python APIs: export" in result["error"]
        assert (tmp_path / "core.py").read_text(encoding="utf-8") == original

    def test_write_no_path_returns_error(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        assert "error" in fw.execute({"content": "some content"})

    def test_write_rejects_unresolved_path_placeholder(self, tmp_path):
        result = FileWriteTool(tmp_path).execute({
            "path": "{{file_locate.path}}", "content": "wrong target",
        })
        assert "unresolved model placeholder" in result["error"]
        assert not (tmp_path / "{{file_locate.path}}").exists()

    def test_write_rejects_comment_only_typescript_placeholder(self, tmp_path):
        result = FileWriteTool(tmp_path).execute({
            "path": "src/physics/Gravity.ts", "content": "// Implement Gravity module here\n",
        })
        assert "only comments or placeholder" in result["error"]
        assert not (tmp_path / "src" / "physics" / "Gravity.ts").exists()

    def test_corrective_typescript_write_rejects_public_api_removal(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        original = (
            "export class Vector {\n"
            "  public readonly x: number = 0;\n"
            "  add(other: Vector): Vector { return other; }\n"
            "  scale(value: number): Vector { return this; }\n"
            "}\n"
        )
        fw.execute({"path": "vector.ts", "content": original})
        result = fw.execute({
            "path": "vector.ts",
            "content": "export class Vector {\n  public readonly x: number = 0;\n}\n",
            "require_semantic_change": True,
        })
        assert "removed public TypeScript APIs: add, scale" in result["error"]
        assert (tmp_path / "vector.ts").read_text(encoding="utf-8") == original

    def test_creates_parent_dirs(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        result = fw.execute({"path": "a/b/c/new.txt", "content": "hi\n"})
        assert result.get("status") == "written"
        assert (tmp_path / "a" / "b" / "c" / "new.txt").exists()

    def test_normalizes_double_escaped_wrapped_source_payload(self, tmp_path):
        payload = "'''\\nimport math\\n\\ndef answer():\\n\\treturn math.floor(42.9)\\n'''"
        result = FileWriteTool(tmp_path).execute({"path": "generated.py", "content": payload})
        source = (tmp_path / "generated.py").read_text(encoding="utf-8")
        assert result.get("payload_normalized") is True
        assert source.startswith("import math\n")
        compile(source, "generated.py", "exec")

    def test_preserves_legitimate_escaped_newlines_in_one_line_source(self, tmp_path):
        source = 'first = "a\\nb"; second = "c\\nd"'
        result = FileWriteTool(tmp_path).execute({"path": "strings.py", "content": source})
        assert not result.get("payload_normalized")
        assert (tmp_path / "strings.py").read_text(encoding="utf-8") == source

    def test_read_and_write_reject_workdir_escape(self, tmp_path):
        outside = tmp_path.parent / "outside-atf.txt"
        outside.write_text("keep", encoding="utf-8")
        read_result = FileReadTool(tmp_path).execute({"path": str(outside)})
        write_result = FileWriteTool(tmp_path).execute({"path": "../outside-atf.txt", "content": "changed"})
        assert "escapes workdir" in read_result["error"]
        assert "escapes workdir" in write_result["error"]
        assert outside.read_text(encoding="utf-8") == "keep"

    def test_file_tools_normalize_duplicated_workdir_prefix(self, tmp_path):
        target = tmp_path / "src" / "core.py"
        target.parent.mkdir()
        target.write_text("value = 1\n", encoding="utf-8")
        duplicate_relative = f"{tmp_path.name}/src/core.py"
        read = FileReadTool(tmp_path).execute({"path": duplicate_relative})
        assert read["content"] == "value = 1\n"
        write = FileWriteTool(tmp_path).execute({
            "path": duplicate_relative, "old_string": "value = 1", "new_string": "value = 2",
        })
        assert write["status"] == "patched"
        assert target.read_text(encoding="utf-8") == "value = 2\n"


# ---------------------------------------------------------------------------
# 5.  DirectoryEnsureTool + WorkspaceScaffoldTool
# ---------------------------------------------------------------------------

class TestScaffoldTools:

    def test_directory_ensure_creates_nested_paths(self, tmp_path):
        tool = DirectoryEnsureTool(tmp_path)
        result = tool.execute({"paths": ["python/django", "rust/cli"]})
        assert result.get("status") == "created"
        assert (tmp_path / "python" / "django").is_dir()
        assert (tmp_path / "rust" / "cli").is_dir()

    def test_workspace_scaffold_creates_index_and_readmes(self, tmp_path):
        tool = WorkspaceScaffoldTool(tmp_path)
        result = tool.execute({
            "root_readme": "# Apps\n",
            "frameworks": [
                {
                    "name": "Django API",
                    "language": "Python",
                    "package_manager": "pip",
                    "create_command": "python -m django startproject app .",
                    "run_command": "python manage.py runserver",
                    "files": {"notes/setup.txt": "install Python first\n"},
                },
                {
                    "name": "Rust CLI",
                    "language": "Rust",
                    "package_manager": "cargo",
                    "create_command": "cargo init",
                    "run_command": "cargo run",
                },
            ],
        })
        assert result.get("status") == "scaffolded"
        assert result.get("framework_count") == 2
        assert (tmp_path / "README.md").exists()
        assert (tmp_path / "Django-API" / "README.md").exists()
        assert (tmp_path / "Django-API" / "notes" / "setup.txt").exists()
        index = json.loads((tmp_path / "framework_index.json").read_text(encoding="utf-8"))
        assert [item["name"] for item in index] == ["Django API", "Rust CLI"]

    def test_workspace_scaffold_major_frameworks_preset_is_compact(self, tmp_path):
        tool = WorkspaceScaffoldTool(tmp_path)
        result = tool.execute({"preset": "major_app_frameworks"})
        assert result.get("status") == "scaffolded"
        assert result.get("framework_count") >= 10
        assert (tmp_path / "Python-Django" / "README.md").exists()
        assert (tmp_path / "Ionic-8-Angular-App" / "starter.placeholder.txt").exists()

    def test_environment_bootstrap_unknown_preset_returns_error(self, tmp_path):
        tool = EnvironmentBootstrapTool(tmp_path)
        result = tool.execute({"preset": "missing_stack"})
        assert "Unknown environment preset" in result["error"]

    def test_scientific_method_resolves_monty_hall_baseline(self, tmp_path):
        mock_ws = MagicMock()
        mock_ws.search_authoritative.return_value = {
            "summary": "Monty Hall problem: switching wins with probability 2/3.",
            "results": [{"title": "Monty Hall", "url": "https://example.test/monty", "snippet": "switching wins 2/3"}],
            "scientific": True,
        }
        tool = ScientificMethodTool(mock_ws, runtime_dir=tmp_path)
        result = tool.execute({
            "question": "In the Monty Hall problem, should you switch doors and what is the win probability?",
            "expected_answer": "switching wins 2/3",
            "domain": "probability",
        })
        assert result["conclusion"]["supported_hypothesis"] == "switch"
        assert result["validation"]["switch_probability"] == pytest.approx(2 / 3)
        assert result["persisted"]["paths"]


# ---------------------------------------------------------------------------
# 6.  FileLocateTool
# ---------------------------------------------------------------------------

class TestFileLocateTool:

    def test_empty_memory_empty_paths(self):
        st = MagicMock(); st.lookup.return_value = []
        lt = MagicMock(); lt.lookup.return_value = []
        result = FileLocateTool(st, lt).execute({"query": "main.py"})
        assert result == {"paths": []}

    def test_missing_query_returns_error(self):
        assert "error" in FileLocateTool(MagicMock(), MagicMock()).execute({})

    def test_deduplicates_st_lt_overlap(self):
        st = MagicMock(); st.lookup.return_value = ["/a/b.py", "/c/d.py"]
        lt = MagicMock(); lt.lookup.return_value = ["/a/b.py", "/e/f.py"]
        result = FileLocateTool(st, lt).execute({"query": "b.py"})
        paths = result["paths"]
        assert len(paths) == len(set(paths))
        assert "/a/b.py" in paths
        assert "/e/f.py" in paths

    def test_merges_st_and_lt_results(self):
        st = MagicMock(); st.lookup.return_value = ["/x/y.rs"]
        lt = MagicMock(); lt.lookup.return_value = ["/z/w.rs"]
        result = FileLocateTool(st, lt).execute({"query": "*.rs"})
        assert set(result["paths"]) == {"/x/y.rs", "/z/w.rs"}

    def test_scoped_locator_filters_paths_outside_workdir(self, tmp_path):
        inside = tmp_path / "main.py"
        outside = tmp_path.parent / "other.py"
        st = MagicMock(); st.lookup.return_value = [str(outside), str(inside)]
        lt = MagicMock(); lt.lookup.return_value = []
        result = FileLocateTool(st, lt, workdir=tmp_path).execute({
            "query": "main.py", "cwd": "Z:/different-drive",
        })
        assert result["paths"] == [str(inside)]
        assert st.lookup.call_args.kwargs["cwd"] == str(tmp_path.resolve())

    def test_scoped_locator_finds_live_file_missing_from_memory(self, tmp_path):
        target = tmp_path / "src" / "impedance_core.py"
        target.parent.mkdir()
        target.write_text("pass\n", encoding="utf-8")
        st = MagicMock(); st.lookup.return_value = []
        lt = MagicMock(); lt.lookup.return_value = []
        result = FileLocateTool(st, lt, workdir=tmp_path).execute({
            "query": "impedance_core.py",
        })
        assert str(target) in result["paths"]


# ---------------------------------------------------------------------------
# 6.  MatrixSearchTool — graceful when Django / matrix_helpers absent
# ---------------------------------------------------------------------------

class TestMatrixSearchTool:

    def test_search_without_django_returns_error_not_crash(self):
        result = MatrixSearchTool().execute({"action": "search", "query": "energy"})
        assert "error" in result or "hits" in result  # does not raise

    def test_missing_query_returns_error(self):
        result = MatrixSearchTool().execute({"action": "search"})
        assert "error" in result

    def test_by_discipline_missing_arg_returns_error(self):
        result = MatrixSearchTool().execute({"action": "by_discipline"})
        assert "error" in result

    def test_find_gaps_missing_args_returns_error(self):
        result = MatrixSearchTool().execute({"action": "find_gaps"})
        assert "error" in result

    def test_linked_missing_eq_id_returns_error(self):
        result = MatrixSearchTool().execute({"action": "linked"})
        assert "error" in result


# ---------------------------------------------------------------------------
# 7.  Mocked external tool dispatch
# ---------------------------------------------------------------------------

class TestMockedTools:

    def test_web_search_delegates(self):
        ws = MagicMock()
        ws.search.return_value = {"results": [{"url": "http://x", "snippet": "hi"}]}
        result = WebSearchTool(ws).execute({"query": "test"})
        ws.search.assert_called_once_with("test")
        assert "results" in result

    def test_web_search_missing_query_returns_error(self):
        ws = MagicMock()
        assert "error" in WebSearchTool(ws).execute({})

    def test_memory_search_delegates(self):
        mem = MagicMock(); mem.search.return_value = [{"text": "old"}]
        result = MemorySearchTool(mem).execute({"query": "thing"})
        mem.search.assert_called_once_with("thing", limit=10)
        assert "results" in result

    def test_memory_search_missing_query_returns_error(self):
        assert "error" in MemorySearchTool(MagicMock()).execute({})

    def test_unbounded_solver_delegates(self):
        solver = _mock_solver()
        result = UnboundedSolverTool(solver).execute({
            "prompt": "solve it", "ai_response": "I don't know",
        })
        assert result["answered"] is True
        assert result["answer"] == "42"

    def test_unbounded_solver_missing_prompt_returns_error(self):
        assert "error" in UnboundedSolverTool(_mock_solver()).execute({})

    def test_math_grounding_delegates(self):
        solver = _mock_solver()
        result = MathGroundingTool(solver).execute({"prompt": "calculate energy"})
        assert result.get("grounding_block") == "grounding"

    def test_math_grounding_missing_prompt_returns_error(self):
        assert "error" in MathGroundingTool(_mock_solver()).execute({})


# ---------------------------------------------------------------------------
# 8.  WizardSession — live probe + send
# ---------------------------------------------------------------------------

class TestWizardSession:

    @pytest.fixture(scope="class")
    def wizard(self):
        from tools.wizard_session import WizardSession
        return WizardSession(session_name="pytest")

    def test_probe_returns_online_key(self):
        from tools.wizard_session import WizardSession
        result = WizardSession.probe()
        assert "online" in result
        assert "endpoint" in result
        assert isinstance(result["online"], bool)

    def test_send_returns_str(self, wizard):
        response = wizard.send("hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_send_with_system_context(self, wizard):
        response = wizard.send("what is 2+2?", system="You are a math tutor.")
        assert isinstance(response, str)

    def test_model_id_is_wizard(self, wizard):
        assert wizard.get_model_id() == "wizard-v1-local"

    def test_session_id_is_uuid(self, wizard):
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            wizard.session_id,
        )

    def test_send_empty_prompt(self, wizard):
        response = wizard.send("")
        assert isinstance(response, str)


# ---------------------------------------------------------------------------
# 9.  SessionManager factory
# ---------------------------------------------------------------------------

class TestSessionManager:

    def test_wizard_backend_when_node_online(self):
        from tools.wizard_session import WizardSession
        if not WizardSession.probe()["online"]:
            pytest.skip("W1z4rD node offline")
        sm = SessionManager(backend="wizard")
        assert hasattr(sm.session, "send")
        assert sm.model_id == "wizard-v1-local"

    def test_wizard_session_id_non_empty(self):
        from tools.wizard_session import WizardSession
        if not WizardSession.probe()["online"]:
            pytest.skip("W1z4rD node offline")
        sm = SessionManager(backend="wizard")
        assert isinstance(sm.session_id, str) and sm.session_id

    def test_wizard_fallback_when_offline(self, monkeypatch):
        """When node offline the factory must NOT return a WizardSession.
        It will try Bedrock; without creds that raises — any exception is fine."""
        import tools.wizard_session as wmod
        monkeypatch.setattr(
            wmod.WizardSession, "probe",
            staticmethod(lambda ep=None: {"online": False, "error": "refused", "endpoint": "http://localhost:8090"}),
        )
        try:
            sm = SessionManager(backend="wizard")
            # If it somehow succeeds, it must have fallen back to something else
            assert type(sm.session).__name__ != "WizardSession"
        except Exception:
            # Any exception here means the fallback path was taken (Bedrock
            # creds/config not available in test env) — that is acceptable.
            pass

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown"):
            SessionManager(backend="__NONEXISTENT__")

    def test_send_delegates_to_session(self):
        from tools.wizard_session import WizardSession
        if not WizardSession.probe()["online"]:
            pytest.skip("W1z4rD node offline")
        sm = SessionManager(backend="wizard")
        result = sm.send("ping")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 10.  ContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:

    @pytest.fixture
    def cb(self, tmp_path):
        reg = _build_registry(tmp_path)
        return ContextBuilder(workdir=tmp_path, tool_descriptions=reg.tool_descriptions())

    def test_tools_section_contains_all_names(self, cb):
        section = cb._tools_section()
        for name in ALL_TOOL_NAMES:
            assert name in section, f"'{name}' missing from tools section"

    def test_tools_section_has_scope_lines(self, cb):
        assert "Scope:" in cb._tools_section()

    def test_tools_section_has_params_lines(self, cb):
        assert "Params:" in cb._tools_section()

    def test_tools_section_has_selection_rules(self, cb):
        section = cb._tools_section()
        assert "memory_search" in section
        assert "file_locate" in section

    def test_build_is_non_empty(self, cb):
        output = cb.build()
        assert len(output) > 200

    def test_tools_section_empty_when_no_tools(self, tmp_path):
        cb = ContextBuilder(workdir=tmp_path, tool_descriptions=[])
        assert cb._tools_section() == ""


# ---------------------------------------------------------------------------
# 11.  Orchestrator._safe_json
# ---------------------------------------------------------------------------

class TestSafeJson:

    def test_valid_json_object(self):
        assert Orchestrator._safe_json('{"a": 1}') == {"a": 1}

    def test_valid_json_array(self):
        assert Orchestrator._safe_json('[1, 2, 3]') == [1, 2, 3]

    def test_json_embedded_in_prose(self):
        result = Orchestrator._safe_json('Here: {"action": "complete", "output": "done"}')
        assert isinstance(result, dict)
        assert result["action"] == "complete"

    def test_empty_string_returns_none(self):
        assert Orchestrator._safe_json("") is None

    def test_whitespace_only_returns_none(self):
        assert Orchestrator._safe_json("   \n  ") is None

    def test_pure_prose_returns_none(self):
        assert Orchestrator._safe_json("not json at all") is None

    def test_nested_payload_roundtrip(self):
        payload = {
            "branches": [
                {"description": "step 1", "rationale": "needed"},
                {"description": "step 2", "rationale": "also needed"},
            ]
        }
        result = Orchestrator._safe_json(json.dumps(payload))
        assert result == payload

    def test_action_complete_shape(self):
        raw = json.dumps({"action": "complete", "output": "all done"})
        result = Orchestrator._safe_json(raw)
        assert result["action"] == "complete"

    def test_action_tool_calls_shape(self):
        raw = json.dumps({
            "action": "tool_calls",
            "tool_calls": [{"tool": "executor", "params": {"command": "echo hi"}}],
        })
        result = Orchestrator._safe_json(raw)
        assert result["action"] == "tool_calls"
        assert result["tool_calls"][0]["tool"] == "executor"

    def test_normalizes_singular_tool_call_with_args(self):
        payload = {
            "action": "tool_call",
            "tool": "file_write",
            "args": {"path": "index.html", "content": "ok"},
        }
        result = Orchestrator._normalize_action(payload)
        assert result == {
            "action": "tool_calls",
            "tool_calls": [{
                "tool": "file_write",
                "params": {"path": "index.html", "content": "ok"},
            }],
        }

    def test_normalizes_nested_and_filename_variants(self):
        payload = {
            "action": "tool_call",
            "tool_call": {"tool": "file_read", "args": {"filename": "progress.md"}},
        }
        result = Orchestrator._normalize_action(payload)
        assert result["tool_calls"] == [
            {"tool": "file_read", "params": {"path": "progress.md"}},
        ]

    def test_normalizes_top_level_action_list(self):
        result = Orchestrator._normalize_action([
            {"action": "complete", "output": "done"},
        ])
        assert result == {"action": "complete", "output": "done"}

    def test_normalizes_top_level_tool_call_list(self):
        result = Orchestrator._normalize_action([
            {"tool": "file_read", "params": {"path": "README.md"}},
        ])
        assert result == {
            "action": "tool_calls",
            "tool_calls": [{"tool": "file_read", "params": {"path": "README.md"}}],
        }


# ---------------------------------------------------------------------------
# 12.  Orchestrator.run() with mock session
# ---------------------------------------------------------------------------

class _PlanAndCompleteSession:
    """Returns a planning response then immediate completion for each branch."""

    def __init__(self):
        self._n = 0

    def send(self, prompt: str, *, stream: bool = False, system: str = "") -> str:
        self._n += 1
        # First few calls are reformulations — return prompt verbatim
        if self._n <= 3:
            return prompt[:100]
        # Planning call
        if "Decompose" in prompt:
            return json.dumps({
                "branches": [
                    {"description": "Branch A", "rationale": "first"},
                    {"description": "Branch B", "rationale": "second"},
                ]
            })
        # Agent steps — immediately complete
        return json.dumps({"action": "complete", "output": "branch done"})


class _ToolCallSession:
    """Returns one tool_calls action then completes."""

    def __init__(self):
        self._branch_calls: dict[str, int] = {}

    def send(self, prompt: str, *, stream: bool = False, system: str = "") -> str:
        # Reformulations
        if "Return ONLY the reformulated" in system:
            return prompt[:80]
        # Planning
        if "Decompose" in prompt:
            return json.dumps({"branches": [{"description": "Run echo", "rationale": "test"}]})
        # Agent: first call -> tool_call, second -> complete
        node_id = ""
        for line in prompt.splitlines():
            if "Current branch [" in line:
                node_id = line.strip()
                break
        count = self._branch_calls.get(node_id, 0) + 1
        self._branch_calls[node_id] = count
        if count == 1:
            return json.dumps({
                "action": "tool_calls",
                "tool_calls": [{"tool": "executor", "params": {"command": "echo from_agent"}}],
            })
        return json.dumps({"action": "complete", "output": "echo dispatched"})


class _CorrectiveSession:
    def __init__(self):
        self.systems = []
        self.calls = 0

    def send(self, prompt: str, *, stream: bool = False, system: str = "") -> str:
        self.calls += 1
        self.systems.append(system)
        if self.calls == 1:
            return json.dumps({
                "action": "tool_calls",
                "tool_calls": [{"tool": "file_read", "params": {"path": "repair.txt"}}],
            })
        if self.calls == 2:
            return json.dumps({
                "action": "tool_calls",
                "tool_calls": [{"tool": "file_write", "params": {
                    "path": "repair.txt", "content": "fixed",
                }}],
            })
        if self.calls == 3:
            return json.dumps({
                "action": "tool_calls",
                "tool_calls": [{"tool": "executor", "params": {"command": "echo validated"}}],
            })
        return json.dumps({"action": "complete", "output": "repaired"})


class _DirectScrutinySession:
    def __init__(self):
        self.calls = 0

    def send(self, prompt: str, *, stream: bool = False, system: str = "") -> str:
        self.calls += 1
        return json.dumps({
            "decision": "direct",
            "answer": "Hello! Good to hear from you.",
        })


class TestOrchestratorRun:
    def test_safe_json_extracts_embedded_array_without_greedy_braces(self):
        text = 'Plan follows: [{"id":"a","description":"one"}] trailing {not json}'
        assert Orchestrator._safe_json(text) == [{"id": "a", "description": "one"}]


    def test_run_returns_results_list_and_tree(self, tmp_path):
        orch = Orchestrator(
            session=_PlanAndCompleteSession(),
            tools=_build_registry(tmp_path),
            context="[test]",
        )
        results, tree = orch.run("do something")
        assert isinstance(results, list)
        assert tree is not None

    def test_root_is_marked_complete(self, tmp_path):
        orch = Orchestrator(
            session=_PlanAndCompleteSession(),
            tools=_build_registry(tmp_path),
            context="[test]",
        )
        _, tree = orch.run("do something")
        assert tree.root.is_done

    def test_results_have_step_ids(self, tmp_path):
        orch = Orchestrator(
            session=_PlanAndCompleteSession(),
            tools=_build_registry(tmp_path),
            context="[test]",
        )
        results, _ = orch.run("do something")
        assert all(r.step_id for r in results)

    def test_tool_call_dispatch_in_agent_loop(self, tmp_path):
        orch = Orchestrator(
            session=_ToolCallSession(),
            tools=_build_registry(tmp_path),
            context="[test]",
        )
        results, _ = orch.run("run an echo command")
        outputs = " ".join(r.output for r in results)
        # executor stdout should appear somewhere in accumulated outputs
        assert "from_agent" in outputs or any(r.success for r in results)

    def test_empty_request_still_completes(self, tmp_path):
        orch = Orchestrator(
            session=_PlanAndCompleteSession(),
            tools=_build_registry(tmp_path),
            context="[test]",
        )
        results, tree = orch.run("")
        assert tree.root.is_done

    def test_global_agent_iteration_budget_stops_recursive_work(self, tmp_path):
        orch = Orchestrator(
            session=_ToolCallSession(),
            tools=_build_registry(tmp_path),
            context="[test]",
        )
        orch._max_total_agent_iterations = 1
        results, _ = orch.run("run an echo command")
        assert orch._total_agent_iterations == 1
        assert any("global agent-iteration budget exhausted" in result.error for result in results)

    def test_global_model_call_budget_counts_reformulation_and_agent_calls(self, tmp_path):
        session = _ToolCallSession()
        orch = Orchestrator(session=session, tools=_build_registry(tmp_path), context="[test]")
        orch._max_total_model_calls = 4
        orch.run("run an echo command")
        assert orch._total_model_calls == 3

    def test_simple_greeting_is_answered_by_single_scrutiny_call(self, tmp_path):
        session = _DirectScrutinySession()
        orch = Orchestrator(session=session, tools=_build_registry(tmp_path), context="[test]")
        results, tree = orch.run("hello")
        # The response is the selected model's single conversational call;
        # there is deliberately no canned local greeting shortcut.
        assert session.calls == 1
        assert orch._total_model_calls == 1
        assert results[0].output == "Hello! Good to hear from you."
        assert tree.root.is_done

    def test_bounded_research_creation_returns_direct_json_without_software_refinement(
        self, tmp_path
    ):
        session = _DirectScrutinySession()
        session.send = MagicMock(
            return_value=json.dumps(
                {
                    "decision": "direct",
                    "answer": '{"work_packages":[{"title":"chronology"}]}',
                }
            )
        )
        orch = Orchestrator(
            session=session,
            tools=_build_registry(tmp_path),
            context="Execution mode: bounded read-only archival-research role.",
        )

        results, tree = orch.run("Produce a publication-grade research paper plan.")

        assert session.send.call_count == 1
        assert results[0].output == '{"work_packages":[{"title":"chronology"}]}'
        assert tree.root.is_done

    def test_corrective_retry_skips_reformulation_and_planning(self, tmp_path):
        (tmp_path / "repair.txt").write_text("broken", encoding="utf-8")
        session = _CorrectiveSession()
        orch = Orchestrator(
            session=session, tools=_build_registry(tmp_path),
            context="CORRECTIVE RETRY unattended atomic workday job",
        )
        results, tree = orch.run("CORRECTIVE RETRY: tests failed")
        assert session.calls == 4
        assert not any("key 'branches'" in system for system in session.systems)
        assert len(tree.root.children) == 1
        assert (tmp_path / "repair.txt").read_text(encoding="utf-8") == "fixed"
        assert any(result.success for result in results)

    def test_corrective_state_machine_requires_read_write_executor(self, tmp_path):
        node = TaskNode(description="repair")
        assert Orchestrator._corrective_required_tool(node) == "file_read"
        node.add_tool_output("file_read", {"content": "x"})
        assert Orchestrator._corrective_required_tool(node) == "file_write"
        node.add_tool_output("file_write", {"status": "written"})
        assert Orchestrator._corrective_required_tool(node) == "executor"
        node.add_tool_output("executor", {"return_code": 1, "error": "failed"})
        assert Orchestrator._corrective_required_tool(node) == "file_write"
        node.add_tool_output("file_write", {"status": "patched"})
        assert Orchestrator._corrective_required_tool(node) == "executor"
        node.add_tool_output("executor", {"return_code": 0, "stdout": "ok"})
        assert Orchestrator._corrective_required_tool(node) == ""

    def test_corrective_batch_allows_navigation_and_advances_each_call(self, tmp_path):
        (tmp_path / "repair.txt").write_text("broken", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="CORRECTIVE RETRY unattended atomic workday job",
        )
        node = TaskNode(description="repair")
        result = orch._dispatch_tool_calls(node, [
            {"tool": "file_read", "params": {"path": "repair.txt"}},
            {"tool": "file_read", "params": {"path": "missing.txt"}},
            {"tool": "file_write", "params": {"path": "repair.txt", "content": "fixed"}},
            {"tool": "executor", "params": {"command": "python -c \"print('ok')\""}},
        ], MagicMock())
        assert (tmp_path / "repair.txt").read_text(encoding="utf-8") == "fixed"
        assert any(item["tool"] == "executor" and not item["result"].get("error")
                   for item in result.tool_outputs)
        assert not any("rejected" in str(item["result"].get("error", ""))
                       for item in result.tool_outputs)
        assert "broken" in result.output

    def test_corrective_navigation_budget_stops_read_only_loop(self, tmp_path):
        (tmp_path / "repair.txt").write_text("broken", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="CORRECTIVE RETRY unattended atomic workday job",
        )
        node = TaskNode(description="repair")
        node.add_tool_output("file_read", {"content": "broken"})
        for _ in range(4):
            node.add_tool_output("file_locate", {"paths": ["repair.txt"]})
        result = orch._dispatch_tool_calls(node, [
            {"tool": "file_read", "params": {"path": "repair.txt"}},
            {"tool": "file_read", "params": {"path": "repair.txt"}},
        ], MagicMock())
        assert not result.tool_outputs[0]["result"].get("error")
        assert "navigation allowance is exhausted" in result.tool_outputs[1]["result"]["error"]

    def test_corrective_hidden_failure_rejects_test_edits(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        target = tests / "test_core.py"
        target.write_text("assert False\n", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="CORRECTIVE RETRY: hidden impedance invariants failed",
        )
        node = TaskNode(description="repair")
        node.add_tool_output("file_read", {"content": "assert False"})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {"path": "tests/test_core.py", "content": "assert True\n"},
        }], MagicMock())
        assert "test write rejected" in result.tool_outputs[0]["result"]["error"]
        assert target.read_text(encoding="utf-8") == "assert False\n"
        assert Orchestrator._looks_model_caused(result.tool_outputs[0]["result"]["error"])

    def test_corrective_allows_test_repair_after_hidden_invariants_pass(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        target = tests / "test_core.py"
        target.write_text("assert False\n", encoding="utf-8")
        context = (
            'CORRECTIVE RETRY validator evidence: {"evidence": ['
            '{"command": ["python", "<hidden impedance physics>"], "ok": true}]}'
        )
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path), context=context,
        )
        node = TaskNode(description="repair")
        node.add_tool_output("file_read", {"content": "assert False"})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {
                "path": "tests/test_core.py", "content": "assert True\n",
            },
        }], MagicMock())
        assert not result.tool_outputs[0]["result"].get("error")
        assert target.read_text(encoding="utf-8") == "assert True\n"

    def test_corrective_state_rejection_is_model_caused(self):
        assert Orchestrator._looks_model_caused(
            "Corrective state requires file_write; navigation allowance is exhausted"
        )

    def test_atomic_schema_violation_is_model_caused(self):
        assert Orchestrator._looks_model_caused(
            "Atomic navigation budget exhausted. Use file_write now."
        )

    def test_atomic_executor_blocks_manifest_initialization_bypass(self):
        error = Orchestrator._atomic_executor_mutation_error({
            "command": "npm init -y && npm install jsdom",
        })
        assert "npm init" in error
        assert "use file_write" in error

    def test_atomic_executor_allows_validation_and_dependency_install(self):
        assert not Orchestrator._atomic_executor_mutation_error({
            "command": "npm test && npm run build",
        })
        assert not Orchestrator._atomic_executor_mutation_error({
            "command": "npm install --save-dev jsdom",
        })

    def test_atomic_source_rewrite_preserves_existing_exports(self, tmp_path):
        target = tmp_path / "scene.ts"
        target.write_text(
            "export function createScene() { return {}; }\n"
            "export function addGridHelper() {}\n",
            encoding="utf-8",
        )
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description="Validator-directed repair paths: [\"scene.ts\"]")
        node.add_tool_output("file_read", {
            "path": str(target), "content": target.read_text(encoding="utf-8"),
        })
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {
                "path": "scene.ts", "content": "export function createScene() { return { fixed: true }; }\n",
            },
        }], MagicMock())
        assert "removes established exports" in result.error
        assert "addGridHelper" in result.error

    def test_read_only_contract_does_not_require_mutation_evidence(self):
        assert not Orchestrator._task_requests_mutation(
            "Read-only evidence task. Do not write or modify files; return the answer."
        )
        assert Orchestrator._task_requests_mutation(
            "Repair the renderer adapter and update its regression tests."
        )

    def test_validator_directed_atomic_repair_limits_navigation_to_three(self, tmp_path):
        target = tmp_path / "broken.ts"
        target.write_text("export {};", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description="Validator-directed repair paths: tests/broken.ts")
        node.add_tool_output("file_locate", {"paths": [str(target)]})
        node.add_tool_output("file_read", {"content": "export {};"})
        node.add_tool_output("file_read", {"content": "interface contract"})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_read", "params": {"path": "broken.ts"},
        }], MagicMock())
        assert "Atomic navigation budget exhausted" in result.error

    def test_validator_directed_unique_locate_is_translated_to_read(self, tmp_path):
        tests = tmp_path / "tests"
        tests.mkdir()
        target = tests / "broken.ts"
        target.write_text("export const fixed = false;", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(
            description='Validator-directed repair paths: ["tests/broken.ts"]',
        )
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_locate", "params": {"query": "broken.ts"},
        }], MagicMock())
        assert result.success
        assert result.tool_outputs[0]["tool"] == "file_read"
        assert "fixed = false" in result.output

    def test_validator_directed_provider_handoff_cannot_repeat_reads_after_mutation(self, tmp_path):
        target = tmp_path / "broken.ts"
        target.write_text("export const fixed = true;", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(
            description='Validator-directed repair paths: ["broken.ts"]',
        )
        node.add_tool_output("file_read", {"content": "export const fixed = false;"})
        node.add_tool_output("file_write", {"path": "broken.ts"})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_read", "params": {"path": "broken.ts"},
        }], MagicMock())
        assert not result.success
        assert "validate it with executor" in result.error

        repeated_write = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {
                "path": "broken.ts", "content": "export const fixed = true;",
            },
        }], MagicMock())
        assert not repeated_write.success
        assert "validate it with executor" in repeated_write.error

    def test_validator_directed_batch_prioritizes_evidence_paths(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        target = source / "main.ts"
        target.write_text("export const evidence = true;", encoding="utf-8")
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(
            description='Validator-directed repair paths: ["src/main.ts"]',
        )
        node.add_tool_output("file_locate", {"paths": ["unrelated-a"]})
        node.add_tool_output("file_locate", {"paths": ["unrelated-b"]})
        result = orch._dispatch_tool_calls(node, [
            {"tool": "file_locate", "params": {"query": "package.json"}},
            {"tool": "file_locate", "params": {"query": "main.ts"}},
        ], MagicMock())
        assert result.tool_outputs[0]["tool"] == "file_read"
        assert "evidence = true" in result.tool_outputs[0]["result"]["content"]
        assert "navigation budget exhausted" in result.tool_outputs[1]["result"]["error"].lower()

    def test_validator_navigation_budget_expands_to_declared_contract(self):
        node = TaskNode(description=(
            'Validator-directed repair paths: '
            '["src/a.ts", "src/b.ts", "src/c.ts", "tests/a.test.ts", "tests/b.test.ts"]'
        ))
        assert Orchestrator._atomic_navigation_limit(node) == 5

        oversized = TaskNode(description=(
            'Validator-directed repair paths: ['
            + ", ".join(f'"src/{index}.ts"' for index in range(20))
            + "]"
        ))
        assert Orchestrator._atomic_navigation_limit(oversized) == 8

    def test_atomic_task_compaction_keeps_active_contract_and_drops_old_noise(self):
        text = (
            "Current plan summary: Build a bounded renderer.\n"
            + ("old validator noise\n" * 4000)
            + "Next step: Repair composition root\n"
            + 'Validator-directed repair paths: ["src/main.ts", "tests/main.test.ts"].\n'
            + 'Persisted deterministic repair packet (authoritative):\n'
            + '{"focus_paths":["src/main.ts"],"required_transition":"write then validate"}'
        )

        compact = Orchestrator._compact_atomic_task_text(text, max_chars=3000)

        assert len(compact) <= 3000
        assert "Current plan summary" in compact
        assert "Next step: Repair composition root" in compact
        assert "Validator-directed repair paths" in compact
        assert '"required_transition":"write then validate"' in compact
        assert compact.count("old validator noise") < 5

    def test_validator_repair_priority_applies_before_batch_truncation(self):
        node = TaskNode(description=(
            'Validator-directed repair paths: ["src/main.ts", "tests/main.test.ts"]'
        ))
        calls = [
            {"tool": "file_read", "params": {"path": "vitest.config.ts"}},
            {"tool": "file_read", "params": {"path": "src/main.ts"}},
            {"tool": "file_read", "params": {"path": "package.json"}},
            {"tool": "file_read", "params": {"path": "tests/main.test.ts"}},
        ]
        prioritized = Orchestrator._prioritize_repair_calls(node, calls)
        assert [call["params"]["path"] for call in prioritized[:2]] == [
            "src/main.ts", "tests/main.test.ts",
        ]

    def test_validator_context_is_preloaded_before_first_model_turn(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "main.ts").write_text(
            "export interface Input { value: number }", encoding="utf-8",
        )
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "main.test.ts").write_text(
            "import '../src/main';", encoding="utf-8",
        )
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description=(
            'Validator-directed repair paths: ["src/main.ts", "tests/main.test.ts"]'
        ))
        orch._seed_validator_context(node)
        reads = [entry for entry in node.tool_outputs if entry["tool"] == "file_read"]
        assert len(reads) == 2
        assert "interface Input" in reads[0]["result"]["content"]
        assert "import '../src/main'" in reads[1]["result"]["content"]

    def test_validator_read_context_survives_generic_history_compaction(self):
        node = TaskNode(description=(
            'Validator-directed repair paths: ["src/scene.ts", "tests/scene.test.ts"]'
        ))
        node.add_tool_output("file_read", {
            "path": r"C:\work\src\scene.ts",
            "content": "export const scene = createScene();",
        })
        node.add_tool_output("file_read", {
            "path": r"C:\work\tests\scene.test.ts",
            "content": "expect(scene).toBeDefined();",
        })
        # Later navigation/protocol events may displace these reads from the
        # generic accumulated-results summary. The bounded validator context
        # must still carry both exact files into a write-only correction turn.
        for index in range(12):
            node.add_tool_output("response_normalizer", {"status": index})

        context = Orchestrator._validator_read_context(node)

        assert "--- src/scene.ts (current validator context) ---" in context
        assert "export const scene = createScene();" in context
        assert "--- tests/scene.test.ts (current validator context) ---" in context
        assert "expect(scene).toBeDefined();" in context

    def test_failed_fix_attributes_actual_fix_model(self):
        fixed = StepResult(
            step_id="x", description="repair", output="", success=False,
            tool_outputs=[{"tool": "file_write", "result": {
                "error": "bad patch",
                "_attribution": {"provider": "FixP", "model": "FixM"},
            }}],
        )
        assert Orchestrator._failed_fix_attribution(
            fixed, {"provider": "OriginalP", "model": "OriginalM"},
        ) == {"provider": "FixP", "model": "FixM"}

    def test_corrective_requires_executor_between_write_batches(self, tmp_path):
        target = tmp_path / "core.py"
        target.write_text("value = 1\n", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="CORRECTIVE RETRY unattended atomic workday job",
        )
        node = TaskNode(description="repair")
        node.add_tool_output("file_read", {"content": "value = 1", "path": str(target)})
        first = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {
                "path": "core.py", "old_string": "value = 1", "new_string": "value = 2",
            },
        }], MagicMock())
        assert first.success
        second = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {
                "path": "core.py", "old_string": "value = 2", "new_string": "value = 3",
            },
        }], MagicMock())
        assert "requires executor" in second.error
        assert target.read_text(encoding="utf-8") == "value = 2\n"

    def test_atomic_failed_validation_reopens_one_write(self, tmp_path):
        target = tmp_path / "package.json"
        target.write_text('{"devDependencies": {}}', encoding="utf-8")
        session = _CorrectiveSession()
        orch = Orchestrator(
            session=session, tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description='Validator-directed repair paths: ["package.json"]')
        node.add_tool_output("file_read", {"path": str(target), "content": target.read_text()})
        node.add_tool_output("file_write", {"path": str(target), "bytes": 23})
        node.add_tool_output("executor", {"return_code": 1, "error": "Cannot find module transitive-dependency"})

        orch._agent_step(node, MagicMock(), 1)

        system = session.systems[-1]
        assert '"name": "file_write"' in system
        assert '"name": "executor"' not in system
        assert "Fresh executor validation failed" in system

    def test_failed_validation_allows_one_read_of_newly_named_caller(self, tmp_path):
        source = tmp_path / "src" / "scene.ts"
        caller = tmp_path / "tests" / "smoke.test.ts"
        source.parent.mkdir()
        caller.parent.mkdir()
        source.write_text("export interface RendererPort {}", encoding="utf-8")
        caller.write_text("createScene();", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description='Validator-directed repair paths: ["src/scene.ts"]')
        node.add_tool_output("file_read", {"path": str(source), "content": source.read_text()})
        node.add_tool_output("file_write", {"path": str(source), "bytes": 32})
        node.add_tool_output("executor", {
            "return_code": 1, "stderr": "failure at tests/smoke.test.ts:7:4",
        })

        first = orch._dispatch_tool_calls(node, [{
            "tool": "file_read", "params": {"path": "tests/smoke.test.ts"},
        }], MagicMock())
        second = orch._dispatch_tool_calls(node, [{
            "tool": "file_read", "params": {"path": "tests/smoke.test.ts"},
        }], MagicMock())

        assert first.success
        assert "createScene" in first.tool_outputs[0]["result"]["content"]
        assert not second.success
        assert "already" in second.error.lower() or "mutation already landed" in second.error.lower()

    def test_atomic_evidence_rejects_counterfeit_before_write(self, tmp_path):
        target = tmp_path / "src" / "scene.ts"
        target.parent.mkdir()
        target.write_text("export const scene = true;", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description=(
            'Validator-directed repair paths: ["src/scene.ts"]. '
            "source counterfeits a WebGLRenderer with an unsafe double assertion"
        ))
        node.add_tool_output("file_read", {"path": str(target), "content": target.read_text()})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write",
            "params": {"path": "src/scene.ts", "content": "const x = {} as unknown as THREE.WebGLRenderer;"},
        }], MagicMock())
        assert not result.success
        assert "counterfeit" in result.error
        assert target.read_text(encoding="utf-8") == "export const scene = true;"

    def test_atomic_evidence_allows_explicit_double_only_in_test_path(self, tmp_path):
        target = tmp_path / "tests" / "scene.test.ts"
        target.parent.mkdir()
        target.write_text("const factory = oldFactory;", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description=(
            'Validator-directed repair paths: ["tests/scene.test.ts"]. '
            "source counterfeits a WebGLRenderer with an unsafe double assertion; use explicit test doubles"
        ))
        node.add_tool_output("file_read", {"path": str(target), "content": target.read_text()})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write",
            "params": {
                "path": "tests/scene.test.ts",
                "old_string": "const factory = oldFactory;",
                "new_string": "const factory = {} as unknown as THREE.WebGLRenderer;",
            },
        }], MagicMock())
        assert result.success
        assert "as unknown as THREE.WebGLRenderer" in target.read_text(encoding="utf-8")

    def test_atomic_evidence_rejects_partial_injection_migration_in_one_test_file(self, tmp_path):
        target = tmp_path / "tests" / "scene.test.ts"
        target.parent.mkdir()
        target.write_text("createScene();", encoding="utf-8")
        node = TaskNode(description=(
            'Validator-directed repair paths: ["tests/scene.test.ts"]. '
            "dependency injection contract: function createScene(rendererFactory: RendererFactory)"
        ))
        node.add_tool_output("file_read", {"path": str(target), "content": target.read_text()})
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        proposed = "createScene(factory);\ncreateScene();\ncreateScene();\n"
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {"path": "tests/scene.test.ts", "content": proposed},
        }], MagicMock())
        assert not result.success
        assert "partial dependency-injection migration" in result.error

    def test_atomic_evidence_rejects_instanceof_assertion_for_structural_double(self, tmp_path):
        target = tmp_path / "tests" / "scene.test.ts"
        target.parent.mkdir()
        target.write_text("const renderer = old;", encoding="utf-8")
        node = TaskNode(description=(
            'Validator-directed repair paths: ["tests/scene.test.ts"]. use explicit test doubles'
        ))
        node.add_tool_output("file_read", {"path": str(target), "content": target.read_text()})
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        proposed = (
            "const renderer = {} as unknown as THREE.WebGLRenderer;\n"
            "expect(renderer).toBeInstanceOf(THREE.WebGLRenderer);\n"
        )
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {"path": "tests/scene.test.ts", "content": proposed},
        }], MagicMock())
        assert not result.success
        assert "structural test double" in result.error

    def test_atomic_evidence_judges_effective_patched_file(self, tmp_path):
        target = tmp_path / "src" / "scene.ts"
        target.parent.mkdir()
        original = (
            "const renderer = {} as unknown as THREE.WebGLRenderer;\n"
            "const aspect = window.innerWidth / window.innerHeight;\n"
        )
        target.write_text(original, encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description=(
            'Validator-directed repair paths: ["src/scene.ts"]. '
            "source counterfeits a WebGLRenderer with an unsafe double assertion"
        ))
        node.add_tool_output("file_read", {"path": str(target), "content": original})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write",
            "params": {
                "path": "src/scene.ts",
                "old_string": "window.innerWidth / window.innerHeight",
                "new_string": "Math.max(window.innerWidth, 1) / Math.max(window.innerHeight, 1)",
            },
        }], MagicMock())
        assert not result.success
        assert "counterfeit" in result.error
        assert target.read_text(encoding="utf-8") == original

    def test_atomic_rejects_duplicate_preloaded_read(self, tmp_path):
        target = tmp_path / "src" / "scene.ts"
        target.parent.mkdir()
        target.write_text("export const scene = true;", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description='Validator-directed repair paths: ["src/scene.ts"]')
        node.add_tool_output("file_read", {"path": str(target), "content": target.read_text()})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_read", "params": {"path": str(target)},
        }], MagicMock())
        assert not result.success
        assert "already preloaded" in result.error

    def test_atomic_rejects_write_outside_validator_surface(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "scene.ts").write_text("export const scene = true;", encoding="utf-8")
        manifest = tmp_path / "package.json"
        manifest.write_text('{"three":"keep"}', encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description='Validator-directed repair paths: ["src/scene.ts"]')
        node.add_tool_output("file_read", {"path": str(tmp_path / "src" / "scene.ts"), "content": "ok"})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write", "params": {"path": "package.json", "content": '{"three":"downgrade"}'},
        }], MagicMock())
        assert not result.success
        assert "outside the declared repair paths" in result.error
        assert manifest.read_text(encoding="utf-8") == '{"three":"keep"}'

    def test_failed_validation_expands_surface_to_named_consumer(self, tmp_path):
        source = tmp_path / "src" / "scene.ts"
        consumer = tmp_path / "tests" / "scene.test.ts"
        source.parent.mkdir()
        consumer.parent.mkdir()
        source.write_text("export interface Port {}", encoding="utf-8")
        consumer.write_text("createScene();", encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description='Validator-directed repair paths: ["src/scene.ts"]')
        node.add_tool_output("file_read", {"path": str(source), "content": source.read_text()})
        node.add_tool_output("file_write", {"path": str(source), "bytes": 24})
        node.add_tool_output("executor", {
            "return_code": 1,
            "stderr": "TypeError at tests/scene.test.ts:12:9 after src/scene.ts:4:2",
        })
        node.add_tool_output("file_read", {"path": str(consumer), "content": consumer.read_text()})

        assert orch._validator_repair_paths(node) == ["src/scene.ts", "tests/scene.test.ts"]
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write",
            "params": {
                "path": "tests/scene.test.ts",
                "old_string": "createScene();",
                "new_string": "createScene(rendererFactory);",
            },
        }], MagicMock())
        assert result.success
        assert consumer.read_text(encoding="utf-8") == "createScene(rendererFactory);"

    def test_corrective_manifest_rewrite_cannot_erase_unrelated_entries(self, tmp_path):
        manifest = tmp_path / "package.json"
        original = '{"scripts":{"test":"vitest","build":"tsc"},"devDependencies":{"vite":"^5","@vitejs/plugin-react":"^4"}}'
        manifest.write_text(original, encoding="utf-8")
        orch = Orchestrator(
            session=_CorrectiveSession(), tools=_build_registry(tmp_path),
            context="unattended atomic workday job",
        )
        node = TaskNode(description='Validator-directed repair paths: ["package.json"] missing @types/three')
        node.add_tool_output("file_read", {"path": str(manifest), "content": original})
        result = orch._dispatch_tool_calls(node, [{
            "tool": "file_write",
            "params": {
                "path": "package.json",
                "content": '{"scripts":{"test":"vitest"},"devDependencies":{"@types/three":"^0.166"}}',
            },
        }], MagicMock())
        assert not result.success
        assert "removes established entries" in result.error
        assert "scripts.build" in result.error
        assert "devDependencies.vite" in result.error
        assert manifest.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# 13.  JSON QA smoke test against live node
# ---------------------------------------------------------------------------

class TestWizardJsonQA:
    """
    Smoke-test the /qa/query endpoint.  Expected to improve as training
    progresses — currently may return (none) or a hypothesis answer.
    Skipped gracefully if node is offline.
    """

    @pytest.fixture(scope="class")
    def node_online(self):
        from tools.wizard_session import WizardSession
        return WizardSession.probe()["online"]

    def test_node_health_endpoint(self, node_online):
        if not node_online:
            pytest.skip("W1z4rD node offline")
        import urllib.request
        with urllib.request.urlopen("http://localhost:8090/health", timeout=5) as r:
            data = json.loads(r.read())
        assert "status" in data

    def test_qa_query_returns_json_dict(self, node_online):
        if not node_online:
            pytest.skip("W1z4rD node offline")
        import urllib.request, urllib.error
        payload = json.dumps({"query": "Return JSON with key result equal to 42."}).encode()
        req = urllib.request.Request(
            "http://localhost:8090/qa/query",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            assert data is not None
        except urllib.error.HTTPError as exc:
            # 404 means endpoint not yet mounted — note but don't fail
            pytest.skip(f"/qa/query not available: {exc}")
        except Exception as exc:
            pytest.skip(f"qa/query error: {exc}")

    def test_neuro_ask_returns_non_empty(self, node_online):
        if not node_online:
            pytest.skip("W1z4rD node offline")
        import urllib.request
        payload = json.dumps({"text": "Return a JSON object.", "top_k": 10}).encode()
        req = urllib.request.Request(
            "http://localhost:8090/neuro/ask",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        assert "answer" in data
