from __future__ import annotations

from dataclasses import replace

import pytest

from tools.c0d3rV2.plugins.agent_the_freeloader.adapters import ProviderError, ProviderResponse
from tools.c0d3rV2.plugins.agent_the_freeloader.models import ModelSpec, PoolLimit, load_catalog
from tools.c0d3rV2.plugins.agent_the_freeloader.quota import QuotaLedger
from tools.c0d3rV2.plugins.agent_the_freeloader.router import FreeloaderRouter, classify_request
from tools.c0d3rV2.plugins.agent_the_freeloader.session import AgentTheFreeloaderSession
from tools.c0d3rV2.sessions import SessionManager
from tools.ai_backend_mode import deactivate_freeloader_mode_for_tests
from tools.wizard_session import WizardSession
from tools.c0d3rV2.task_tree import TaskTree


@pytest.fixture(autouse=True)
def _reset_freeloader_exclusive_mode_after_test():
    yield
    deactivate_freeloader_mode_for_tests()


def _spec(
    model_id: str,
    *,
    pool: str,
    coding: float = 0.5,
    tools: float = 0.5,
    reasoning: float = 0.5,
    key: str = "TEST_FREE_KEY",
) -> ModelSpec:
    return ModelSpec(
        provider=f"Provider-{model_id}",
        model_id=model_id,
        modalities=frozenset({"text"}),
        best_at="tests",
        base_url="https://example.invalid/v1",
        endpoint="/chat/completions",
        api_style="OpenAI-compatible",
        api_key_env=key,
        pool_ids=(pool,),
        limits=PoolLimit(requests_per_day=100, tokens_per_day=1_000_000),
        capabilities={
            "general": 0.7,
            "coding": coding,
            "tools": tools,
            "reasoning": reasoning,
            "structured": 0.6,
            "speed": 0.6,
            "multimodal": 0.1,
            "multilingual": 0.5,
        },
    )


def test_catalog_loads_callable_chat_models_only():
    specs = load_catalog()
    assert specs
    assert all("text" in spec.modalities for spec in specs)
    assert all("embedding" not in spec.model_id.lower() for spec in specs)
    assert not any(spec.model_id == "whisper-large-v3" for spec in specs)
    openrouter = [spec for spec in specs if spec.provider == "OpenRouter"]
    assert len(openrouter) >= 2
    assert {spec.pool_ids for spec in openrouter} == {("openrouter:free",)}
    github = [spec for spec in specs if spec.provider == "GitHub Models"]
    assert len(github) == 7
    assert len({spec.pool_ids for spec in github}) == len(github)
    pollinations = [spec for spec in specs if spec.provider == "Pollinations.AI"]
    assert [spec.model_id for spec in pollinations] == ["openai-fast"]
    assert pollinations[0].api_key_env == ""
    assert pollinations[0].limits.requests_per_minute == 4
    routeway = [spec for spec in specs if spec.provider == "Routeway"]
    assert len(routeway) >= 10
    assert {spec.pool_ids for spec in routeway} == {("routeway:free-account",)}
    assert {spec.limits.requests_per_day for spec in routeway} == {200}
    llmrack = [spec for spec in specs if spec.provider == "LLMRack"]
    assert len(llmrack) == 4
    assert {spec.pool_ids for spec in llmrack} == {("llmrack:free-account",)}
    assert {spec.limits.tokens_per_day for spec in llmrack} == {10_000}
    assert {spec.provider for spec in specs} >= {"Logfare", "LLM.kiwi"}
    kilo = [spec for spec in specs if spec.provider == "Kilo Gateway"]
    assert kilo
    assert {spec.pool_ids for spec in kilo} == {("kilo:free-ip-pool",)}
    assert all(spec.api_key_env == "" for spec in kilo)
    vercel = [spec for spec in specs if spec.provider == "Vercel AI Gateway"]
    assert len(vercel) == 4
    assert {spec.pool_ids for spec in vercel} == {("vercel:monthly-credit",)}
    speka = [spec for spec in specs if spec.provider == "Speka"]
    assert len(speka) == 4
    assert {spec.pool_ids for spec in speka} == {("speka:monthly-credit",)}
    scaleway = [spec for spec in specs if spec.provider == "Scaleway Generative APIs"]
    assert len(scaleway) == 7
    assert {spec.pool_ids for spec in scaleway} == {("scaleway:new-customer-credit",)}
    zhipu = [spec for spec in specs if spec.provider == "Zhipu BigModel"]
    assert {spec.model_id for spec in zhipu} == {
        "glm-4.7-flash", "glm-4.6v-flash", "glm-4.1v-thinking-flash",
    }
    io_models = [spec for spec in specs if spec.provider == "IO Intelligence"]
    assert len(io_models) == 11
    assert {spec.pool_ids for spec in io_models} == {("io-intelligence:daily-credit",)}
    alibaba = [spec for spec in specs if spec.provider == "Alibaba Cloud Model Studio"]
    assert len(alibaba) == 7
    assert len({spec.pool_ids for spec in alibaba}) == len(alibaba)
    siliconflow = [spec for spec in specs if spec.provider == "SiliconFlow"]
    assert len(siliconflow) == 7
    assert len({spec.pool_ids for spec in siliconflow}) == len(siliconflow)
    hyperbolic = [spec for spec in specs if spec.provider == "Hyperbolic"]
    assert len(hyperbolic) == 4
    assert {spec.pool_ids for spec in hyperbolic} == {("hyperbolic:promotional-credit",)}


def test_payment_required_blocks_credit_pool_for_a_month_by_default():
    error = ProviderError(402, "promotional credit exhausted")
    assert error.is_quota
    assert error.retry_after == 30 * 24 * 60 * 60


def test_task_tree_feedback_accepts_symbol_like_mapping_keys():
    tree = TaskTree(root_description="test")
    non_json_key = object()
    tree.root.add_tool_output("math_grounding", {"variables": {non_json_key: "value"}})
    summary = tree.accumulated_results_summary()
    assert "math_grounding" in summary
    assert "value" in summary


def test_task_tree_feedback_retains_source_and_recent_error_after_many_events():
    tree = TaskTree(root_description="repair")
    source = "header\n" + ("padding\n" * 120) + "def export_to_csv(filename, rows):\n    pass\n"
    tree.root.add_tool_output("file_read", {
        "path": "core.py", "content": source, "offset": 0,
    })
    for index in range(20):
        tree.root.add_tool_output("executor", {"error": f"failure-{index}"})
    summary = tree.accumulated_results_summary()
    assert "def export_to_csv" in summary
    assert "failure-19" in summary


def test_c0d3rv2_session_manager_exposes_peer_backend(tmp_path):
    try:
        manager = SessionManager(backend="freeloader", workdir=str(tmp_path))
        assert isinstance(manager.session, AgentTheFreeloaderSession)
        assert manager.model_id == "agent-the-freeloader"
        probe = WizardSession.probe()
        assert probe["online"] is False
        assert "disabled by AgentTheFreeloader" in probe["error"]
        with pytest.raises(RuntimeError, match="disabled while AgentTheFreeloader"):
            WizardSession().send("must not reach the node")
    finally:
        deactivate_freeloader_mode_for_tests()


def test_global_freeloader_mode_overrides_wizard_factory(monkeypatch, tmp_path):
    monkeypatch.setenv("C0D3R_BACKEND", "freeloader")
    try:
        manager = SessionManager(backend="wizard", workdir=str(tmp_path))
        assert isinstance(manager.session, AgentTheFreeloaderSession)
    finally:
        deactivate_freeloader_mode_for_tests()


def test_request_ranking_prioritizes_iteration_capabilities(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    coder = _spec("coder", pool="coder", coding=1.0, tools=1.0, reasoning=0.8)
    chatter = _spec("chatter", pool="chatter", coding=0.2, tools=0.2, reasoning=0.5)
    ledger = QuotaLedger({"coder": coder.limits, "chatter": chatter.limits})
    router = FreeloaderRouter([chatter, coder], ledger)
    profile = classify_request(
        "Implement and test this repository change",
        'Available tools: file_write, executor. Return only JSON tool_calls.',
        max_tokens=1024,
    )
    assert router.rank(profile)[0].spec.model_id == "coder"


def test_runtime_model_filter_constrains_atf_candidates(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    first = _spec("first", pool="first")
    second = _spec("second", pool="second")
    ledger = QuotaLedger({"first": first.limits, "second": second.limits})
    router = FreeloaderRouter([first, second], ledger, allowed_models={"second"})
    ranked = router.rank(classify_request("implement code", "", max_tokens=512))
    assert [candidate.spec.model_id for candidate in ranked] == ["second"]


def test_equivalent_models_rotate_instead_of_sticking(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    first = _spec("first", pool="first")
    second = replace(_spec("second", pool="second"), capabilities=dict(first.capabilities))
    ledger = QuotaLedger({"first": first.limits, "second": second.limits})

    def fake(spec, **_kwargs):
        return ProviderResponse(text=spec.model_id, headers={}, input_tokens=10, output_tokens=10)

    router = FreeloaderRouter([first, second], ledger, invoker=fake)
    one = router.send("hello")
    two = router.send("hello")
    assert {one, two} == {"first", "second"}


def test_router_honors_healthy_phase_preference(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    first = _spec("first", pool="first")
    second = replace(_spec("second", pool="second"), capabilities=dict(first.capabilities))
    ledger = QuotaLedger({"first": first.limits, "second": second.limits})
    router = FreeloaderRouter(
        [first, second], ledger,
        invoker=lambda spec, **_kwargs: ProviderResponse(
            text=spec.model_id, headers={}, input_tokens=1, output_tokens=1,
        ),
    )
    assert router.send("hello", preferred_identity=second.identity) == "second"


def test_router_skips_turn_excluded_model_when_alternative_exists(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    first = _spec("first", pool="first")
    second = replace(_spec("second", pool="second"), capabilities=dict(first.capabilities))
    ledger = QuotaLedger({"first": first.limits, "second": second.limits})
    router = FreeloaderRouter(
        [first, second], ledger,
        invoker=lambda spec, **_kwargs: ProviderResponse(
            text=spec.model_id, headers={}, input_tokens=1, output_tokens=1,
        ),
    )
    assert router.send("hello", excluded_identities={first.identity}) == "second"


def test_shared_pool_exhaustion_applies_to_entire_provider(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    one = replace(_spec("one", pool="shared"), limits=PoolLimit(requests_per_day=1))
    two = replace(_spec("two", pool="shared"), limits=PoolLimit(requests_per_day=1))
    ledger = QuotaLedger({"shared": PoolLimit(requests_per_day=1)})

    def fake(spec, **_kwargs):
        return ProviderResponse(text=spec.model_id, headers={}, input_tokens=1, output_tokens=1)

    router = FreeloaderRouter([one, two], ledger, invoker=fake)
    assert router.send("first request") in {"one", "two"}
    with pytest.raises(RuntimeError, match="no eligible model"):
        router.send("second request")


def test_persisted_ledger_is_shared_between_worker_instances(tmp_path):
    state = tmp_path / "quota.json"
    limits = {"shared": PoolLimit(requests_per_day=1)}
    first = QuotaLedger(limits, state_path=state)
    second = QuotaLedger(limits, state_path=state)
    first.reserve(("shared",), 10)
    assert second.available(("shared",), 10) is False


def test_quota_failure_blocks_shared_pool_and_falls_back(monkeypatch):
    monkeypatch.setenv("TEST_FREE_KEY", "configured")
    best = _spec("best", pool="provider-shared", coding=1.0, tools=1.0, reasoning=1.0)
    sibling = _spec("sibling", pool="provider-shared", coding=0.9, tools=0.9, reasoning=0.9)
    fallback = _spec("fallback", pool="other", coding=0.6, tools=0.6, reasoning=0.6)
    ledger = QuotaLedger({
        "provider-shared": PoolLimit(requests_per_day=100),
        "other": PoolLimit(requests_per_day=100),
    })
    attempted: list[str] = []

    def fake(spec, **_kwargs):
        attempted.append(spec.model_id)
        if spec.model_id == "best":
            raise ProviderError(429, "quota exceeded", {"retry-after": "120"})
        return ProviderResponse(text=spec.model_id, headers={}, input_tokens=1, output_tokens=1)

    router = FreeloaderRouter([best, sibling, fallback], ledger, invoker=fake)
    assert router.send("implement code with tools") == "fallback"
    assert attempted == ["best", "fallback"]
    assert ledger.headroom(("provider-shared",), 1) == 0.0
def test_session_records_failed_route_trace() -> None:
    class FailingRouter:
        last_trace = [{"provider": "P", "model": "M", "outcome": "failed", "status": 429}]

        def send(self, *args, **kwargs):
            raise RuntimeError("quota")

    session = object.__new__(AgentTheFreeloaderSession)
    session.router = FailingRouter()
    session.route_history = []
    session.max_tokens = 100
    session.temperature = 0.2
    try:
        session.send("hello")
    except RuntimeError:
        pass
    assert session.route_history == [[{**FailingRouter.last_trace[0], "phase": "other"}]]


def test_session_labels_route_phase() -> None:
    class Router:
        last_trace = [{"provider": "P", "model": "M", "outcome": "selected"}]

        def send(self, *args, **kwargs):
            return "ok"

    session = object.__new__(AgentTheFreeloaderSession)
    session.router = Router()
    session.route_history = []
    session.max_tokens = 100
    session.temperature = 0.2
    session.last_error = ""
    session.transcript_enabled = False
    session.send("plan", system="Return ONLY a JSON object with key 'branches'")
    assert session.route_history[0][0]["phase"] == "planning"


def test_session_turn_budget_covers_all_send_callers() -> None:
    class Router:
        last_trace = [{"provider": "P", "model": "M", "outcome": "selected"}]

        def send(self, *args, **kwargs):
            return "ok"

    session = object.__new__(AgentTheFreeloaderSession)
    session.router = Router()
    session.route_history = []
    session.max_tokens = 100
    session.temperature = 0.2
    session.transcript_enabled = False
    session.begin_turn(2)
    assert session.send("one") == "ok"
    assert session.send("two") == "ok"
    try:
        session.send("three")
    except RuntimeError as exc:
        assert "budget exhausted" in str(exc)
    else:
        raise AssertionError("third call should exceed the turn budget")
    assert session._turn_calls == 2


def test_session_keeps_model_continuity_within_phase() -> None:
    class Router:
        def __init__(self):
            self.last_trace = []
            self.preferences = []

        def send(self, *args, preferred_identity="", **kwargs):
            self.preferences.append(preferred_identity)
            self.last_trace = [{"provider": "P", "model": "M", "outcome": "selected"}]
            return "ok"

    session = object.__new__(AgentTheFreeloaderSession)
    session.router = Router()
    session.route_history = []
    session.max_tokens = 100
    session.temperature = 0.2
    session.transcript_enabled = False
    session.begin_turn(3)
    system = "You are executing one branch with available tools"
    session.send("read", system=system)
    session.send("write", system=system)
    assert session.router.preferences == ["", "P:M"]


def test_hallucination_correction_unpins_phase_model() -> None:
    class Feedback:
        def record_correction(self, *args, **kwargs):
            return 7

    class Router:
        last_trace = [{"provider": "P", "model": "M", "outcome": "selected"}]
        feedback = Feedback()

        def report_outcome(self, *args, **kwargs):
            pass

    session = object.__new__(AgentTheFreeloaderSession)
    session.router = Router()
    session.session_name = "test"
    session._phase_sticky = {"agent": "P:M", "fix": "P:other"}
    event = session.report_correction(
        classification="premature_completion", trigger="no file write",
        is_hallucination=True,
    )
    assert event == 7
    assert session._phase_sticky == {"fix": "P:other"}
    assert session._turn_banned == {"P:M"}


def test_session_rotates_phase_affinity_after_four_calls(monkeypatch) -> None:
    monkeypatch.setenv("ATF_PHASE_MODEL_CALLS", "4")

    class Router:
        def __init__(self):
            self.last_trace = []
            self.exclusions = []

        def send(self, *args, excluded_identities=None, **kwargs):
            self.exclusions.append(set(excluded_identities or ()))
            self.last_trace = [{"provider": "P", "model": "M", "outcome": "selected"}]
            return "ok"

    session = object.__new__(AgentTheFreeloaderSession)
    session.router = Router()
    session.route_history = []
    session.max_tokens = 100
    session.temperature = 0.2
    session.transcript_enabled = False
    session.begin_turn(6)
    system = "You are executing one branch with available tools"
    for _ in range(5):
        session.send("step", system=system)
    assert session.router.exclusions[:4] == [set(), set(), set(), set()]
    assert session.router.exclusions[4] == {"P:M"}
