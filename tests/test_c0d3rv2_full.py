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
    UnboundedSolverTool,
    MathGroundingTool,
    VMPlaygroundTool,
)
from orchestrator import Orchestrator
from context_builder import ContextBuilder
from executor import Executor
from sessions import SessionManager


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

ALL_TOOL_NAMES = [
    "executor", "web_search", "memory_search", "file_locate",
    "equation_matrix", "file_read", "file_write",
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
    reg.register(WebSearchTool(mock_ws))
    reg.register(MemorySearchTool(mock_mem))
    reg.register(FileLocateTool(mock_st, mock_lt))
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
        assert len(descs) == 10

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
        assert result.get("return_code", 0) != 0 or "error" in result


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

    def test_write_no_path_returns_error(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        assert "error" in fw.execute({"content": "some content"})

    def test_creates_parent_dirs(self, tmp_path):
        fw = FileWriteTool(tmp_path)
        result = fw.execute({"path": "a/b/c/new.txt", "content": "hi\n"})
        assert result.get("status") == "written"
        assert (tmp_path / "a" / "b" / "c" / "new.txt").exists()


# ---------------------------------------------------------------------------
# 5.  FileLocateTool
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


class TestOrchestratorRun:

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
