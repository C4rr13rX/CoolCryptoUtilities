"""tests/test_wizard_autonomous_and_gui_state.py

Verifies Primary Target 1:
  1. Cross-process CognitiveStateManager (`can_chat` gating, sleep/checkpoint lock
     reporting, stale-lock auto-recovery, and instant user-chat preemption).
  2. Baseline-relative SettlingController (`sleep_pressure` resets cleanly to 0%
     after settling and never conflates Django/Waitress web RSS with W1z4rD node RAM).
  3. Cross-process mtime-reloading AnchoredMemoryManager (`Hello` and user-trained
     conversational foundations never decay across restarts or sleep cycles).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from services.anchored_memory import AnchoredMemoryManager
from tools.wizard_learning.cognitive_state import CognitiveStateManager
from tools.wizard_learning.settling_controller import SettlingController


def test_cognitive_state_gating_and_user_preemption(tmp_path: Path):
    state_file = tmp_path / "wizard_cognitive_state.json"
    mgr = CognitiveStateManager(state_file=state_file)

    initial = mgr.read()
    assert initial["can_chat"] is True
    assert initial["mode"] == "IDLE_READY"

    # Entering biological sleep locks interactive chat and provides a clear user reason
    sleeping = mgr.set_mode("SLEEPING_SETTLING", detail="Consolidating STDP synapses")
    assert sleeping["can_chat"] is False
    assert "sleeping" in sleeping["chat_unavailable_reason"].lower()

    # Returning to SELF_EDUCATING unlocks chat (idle learning yields to user chat)
    edu = mgr.set_mode("SELF_EDUCATING", detail="C0d3rV2 synthesizing robotics mechanics", is_daemon=True)
    assert edu["can_chat"] is True
    assert edu["daemon_alive"] is True

    # User starting a chat immediately flags user_chat_active for daemon preemption
    assert mgr.is_user_recently_active(grace_seconds=20.0) is False
    mgr.mark_user_chat_start()
    assert mgr.is_user_recently_active(grace_seconds=20.0) is True
    after_chat = mgr.mark_user_chat_end()
    assert after_chat["can_chat"] is True


def test_cognitive_state_auto_recovers_stale_sleep_lock(tmp_path: Path):
    state_file = tmp_path / "wizard_cognitive_state.json"
    mgr = CognitiveStateManager(state_file=state_file)
    mgr.set_mode("SLEEPING_SETTLING", detail="Simulated crashed settle")

    # Backdate updated_at past 240s safety timeout
    raw = json.loads(state_file.read_text(encoding="utf-8"))
    raw["updated_at"] = time.time() - 260.0
    state_file.write_text(json.dumps(raw), encoding="utf-8")

    recovered = mgr.read()
    assert recovered["can_chat"] is True
    assert recovered["mode"] == "IDLE_READY"


def test_settling_controller_baseline_relative_sleep_pressure_resets_to_zero(tmp_path: Path):
    with (
        patch("tools.wizard_learning.settling_controller.SETTLING_STATE_FILE", tmp_path / "settling_state.json"),
        patch("tools.wizard_learning.settling_controller.SETTLING_HISTORY_FILE", tmp_path / "settling_hist.jsonl"),
    ):
        ctrl = SettlingController(endpoint="http://127.0.0.1:8090", ram_threshold_mb=250.0, batch_interval=20)

        # Simulate local w1z4rd_node at 320 MB (even if Python/Waitress uses 2700 MB!)
        with patch.object(ctrl, "_measure_wizard_node_ram", return_value=(21848, 320.0, 1400.0)):
            m0 = ctrl.get_resource_metrics()
            assert m0["node_ram_mb"] == 320.0
            assert m0["sleep_pressure"] == 0.0
            assert m0["needs_settle"] is False

        # Simulate +220 MB transient synapse growth + 18 training episodes -> high sleep pressure
        ctrl.episodes_since_settle = 18
        with patch.object(ctrl, "_measure_wizard_node_ram", return_value=(21848, 540.0, 1400.0)):
            m_high = ctrl.get_resource_metrics()
            assert m_high["sleep_pressure"] >= 0.85
            assert m_high["needs_settle"] is True

            # Execute error-free mocked settle -> baseline updates and sleep_pressure resets to 0.0!
            with (
                patch("urllib.request.urlopen") as mock_url,
                patch.object(ctrl, "_call_node_step", return_value={"ok": True, "status": "ok"}),
                patch.object(ctrl, "_verify_and_heal_post_sleep_recall", return_value={"checked": 5, "passed": 5, "healed": 0, "recall_rate": 1.0}),
            ):
                mock_resp = mock_url.return_value.__enter__.return_value
                mock_resp.status = 200
                res = ctrl.settle(reason="unit_test_consolidation")

            assert res["settled"] is True
            assert res["error_free"] is True
            assert res["metrics_after"]["sleep_pressure"] == 0.0
            assert res["metrics_after"]["episodes_since_settle"] == 0


def test_anchored_memory_recalls_hello_and_reloads_across_processes():
    mgr = AnchoredMemoryManager.get_instance()
    hello = mgr.recall("Hello")
    assert hello is not None
    assert "W1z4rD" in hello["answer"]
    assert hello["confidence"] == 1.0


def test_elastic_pc_allowance_expands_and_shrinks_politely():
    from tools.wizard_learning.settling_controller import compute_elastic_pc_allowance

    # Abundant free RAM on 32 GB PC (18.5 GB free) -> EXPANDED
    hi = compute_elastic_pc_allowance(31.78, 18.5)
    assert hi["allowance_tier"] == "EXPANDED"
    assert hi["polite_reserve_gb"] >= 8.0
    assert hi["usable_headroom_gb"] < (18.5 - hi["polite_reserve_gb"])
    assert hi["recommended_pairs_per_cycle"] == 6
    assert hi["target_terminals_per_pool"] >= 5_000_000

    # Moderate multi-strategy load (12.8 GB free) -> BALANCED
    mid = compute_elastic_pc_allowance(31.78, 12.8)
    assert mid["allowance_tier"] == "BALANCED"
    assert mid["dynamic_delta_budget_mb"] < hi["dynamic_delta_budget_mb"]

    # Tight load approaching polite reserve (8.6 GB free) -> COMPACT_SHRINK (keeps running gently!)
    lo = compute_elastic_pc_allowance(31.78, 8.6)
    assert lo["allowance_tier"] == "COMPACT_SHRINK"
    assert lo["recommended_pairs_per_cycle"] == 2
    assert lo["target_terminals_per_pool"] < mid["target_terminals_per_pool"]


def test_polite_multimodal_feeder_streams_cloud_verified_records(tmp_path: Path):
    from tools.wizard_learning.multimodal_feeder import PoliteMultimodalFeeder

    posted_frames = []
    feeder = PoliteMultimodalFeeder(endpoint="http://127.0.0.1:8090", cursor_path=tmp_path / "cursor.json")
    with patch.object(feeder, "_post_cofire_frames", side_effect=lambda frames, timeout=35.0: posted_frames.append(frames) or True):
        res_expanded = feeder.feed_elastic_slice(allowance_tier="EXPANDED")
        assert res_expanded["episode_budget"] == 4
        assert res_expanded["episodes_fed"] == 4
        assert len(posted_frames) == 4
        # At least one episode must co-fire all 4 pools (TEXT=1, IMAGE=2, AUDIO=3, ACTION=4)
        assert any({f["pool_id"] for f in ep} == {1, 2, 3, 4} for ep in posted_frames)

        res_compact = feeder.feed_elastic_slice(allowance_tier="COMPACT_SHRINK")
        assert res_compact["episode_budget"] == 1
        assert res_compact["episodes_fed"] == 1
        assert res_compact["episodes_fed_total"] == 5


def test_eye_lifecycle_states_offline_settling_rem_and_waking(tmp_path: Path):
    import os

    state_file = tmp_path / "wizard_cognitive_state.json"
    mgr = CognitiveStateManager(state_file=state_file)

    # 1. Offline -> strictly 'offline'
    off = mgr.set_mode("OFFLINE", detail="Node unreachable")
    assert off["eye_lifecycle_state"] == "offline"
    assert off["is_rem_sleep"] is False

    # 2. Settling (< 1 hour) -> 'settling_sleep' (shut with slight sleep movement, not REM)
    sleep_short = mgr.set_mode("SLEEPING_SETTLING", detail="NREM synaptic consolidation", settling_holder_pid=os.getpid())
    assert sleep_short["eye_lifecycle_state"] == "settling_sleep"
    assert sleep_short["is_rem_sleep"] is False

    # 3. Settling (>= 1 hour, 3700s) with live holder PID -> 'rem_sleep'
    raw = json.loads(state_file.read_text(encoding="utf-8"))
    raw["mode_entered_at"] = time.time() - 3700.0
    raw["updated_at"] = time.time() - 30.0
    raw["settling_holder_pid"] = os.getpid()
    state_file.write_text(json.dumps(raw), encoding="utf-8")

    rem = mgr.read()
    assert rem["mode"] == "SLEEPING_SETTLING"
    assert rem["is_rem_sleep"] is True
    assert rem["mode_duration_seconds"] >= 3600.0
    assert rem["eye_lifecycle_state"] == "rem_sleep"

    # 4. Transition out of sleep -> 'waking' (< 6.0s), then 'awake' (>= 6.0s)
    waking = mgr.set_mode("IDLE_READY", detail="Waking from consolidation")
    assert waking["eye_lifecycle_state"] == "waking"
    assert waking["seconds_since_wake"] < 6.0

    raw2 = json.loads(state_file.read_text(encoding="utf-8"))
    raw2["last_wake_at"] = time.time() - 12.0
    state_file.write_text(json.dumps(raw2), encoding="utf-8")
    awake = mgr.read()
    assert awake["eye_lifecycle_state"] == "awake"


def test_emergent_neural_pool_sentiment_never_fakes_keyword_heuristics():
    from tools.wizard_learning import neural_eye_dynamics as ned

    # 1. Even with emotional keywords in text, if W1z4rD's neural pools have NOT fired
    #    any `social_affect` tokens, `pool_grounded` MUST be False and `dominant_sentiment == "neutral"`.
    def fake_unlearned_http(url: str, payload=None, timeout: float = 4.0):
        if url.endswith("/multi_pool/ask"):
            return {"status": "ok", "predictions": {"social_affect": ""}}
        if url.endswith("/brain"):
            return {"neuromodulators": {"dopamine": 0.5, "norepinephrine": 0.35, "acetylcholine": 0.5, "serotonin": 0.5}}
        return None

    with patch.object(ned, "_http_json", side_effect=fake_unlearned_http):
        unlearned = ned.probe_neural_conversation_sentiment(
            "Emergency stop! Wonderful amazing joyful crisis!",
            "Thank you!",
        )
        assert unlearned["pool_grounded"] is False
        assert unlearned["dominant_sentiment"] == "neutral"
        assert unlearned["fired_neurons"] == []

    # 2. When `social_affect` neurons genuinely fire from Hebbian co-activation,
    #    sentiment coordinates (`valence`, `warmth`, `wonder`, `caution`) emerge directly from those neurons.
    def fake_learned_http(url: str, payload=None, timeout: float = 4.0):
        if url.endswith("/multi_pool/ask"):
            return {"status": "ok", "predictions": {"social_affect": "soc:t:warm soc:t:wonder soc:c:shared_attention"}}
        if url.endswith("/brain"):
            return {"neuromodulators": {"dopamine": 0.68, "norepinephrine": 0.42, "acetylcholine": 0.60, "serotonin": 0.72}}
        return None

    with patch.object(ned, "_http_json", side_effect=fake_learned_http):
        grounded = ned.probe_neural_conversation_sentiment(
            "How do your experiences shape you?",
            "Through associative Hebbian co-activation across pools.",
        )
        assert grounded["pool_grounded"] is True
        assert grounded["valence"] > 0.5
        assert grounded["warmth"] > 0.65
        assert grounded["wonder"] > 0.5
        assert grounded["caution"] < 0.15
        assert any(n["neuron"] == "soc:t:warm" for n in grounded["fired_neurons"])


def test_on_the_fly_identity_eye_redesign_evolves_within_bounds(tmp_path: Path):
    from tools.wizard_learning import neural_eye_dynamics as ned
    from tools.wizard_learning import self_design as sd
    from tools.wizard_learning.store import LearningStore, brain_key

    db_path = tmp_path / "wizard_learning_test.sqlite3"
    profile = {"id": "local_test_eye", "name": "Test Eye Brain", "endpoint": "http://127.0.0.1:8090"}

    def fake_ocular_pool_http(url: str, payload=None, timeout: float = 4.0):
        if url.endswith("/multi_pool/ask"):
            return {
                "status": "ok",
                "predictions": {
                    "ocular_identity": "id:iris_emerald id:hetero_gold id:pupil_slit id:striae_crystalline id:spiral_harmonic"
                },
            }
        return None

    with (
        patch("tools.wizard_learning.self_design.LearningStore", lambda: LearningStore(db_path)),
        patch("tools.wizard_learning.neural_eye_dynamics.LearningStore", lambda: LearningStore(db_path)),
        patch.object(ned, "_http_json", side_effect=fake_ocular_pool_http),
    ):
        ned._LAST_AUTO_REDESIGN_AT = 0.0
        res = ned.evaluate_on_the_fly_eye_redesign(
            profile,
            experience_text="Synthesized titanium actuator and crystalline optics",
            sentiment={"dominant_sentiment": "wonder"},
            cognitive_state={"self_edu_cycles": 24, "robot_world": {"body_generation": 4}},
            force=False,
        )
        assert res["applied"] is True
        assert res["redesigned_on_the_fly"] is True
        assert res["identity_generation"] == 2
        eye = res["design"]["eye"]
        for k, (lo, hi) in sd.EYE_BOUNDS.items():
            assert lo <= eye[k] <= hi, f"{k}={eye[k]} out of bounds [{lo}, {hi}]"
        # Pupil aspect shifted toward slit (< 1.0) and spiral arms > 1
        assert eye["pupil_aspect"] < 0.85
        assert eye["striae_detail"] > 1.15
        assert "id:iris_emerald" in res["fired_neurons"]

        persisted = sd.current(LearningStore(db_path), brain_key(profile))
        assert persisted["personality"]["identity_generation"] == 2
        assert persisted["eye"]["pupil_aspect"] == eye["pupil_aspect"]


def test_fast_conversational_engine_millisecond_resolvers():
    import sympy  # Pre-warm module import before measuring warm resolver latency
    from tools.wizard_learning import fast_conversational_engine as fce
    assert sympy is not None

    with patch.object(fce, "_http_post_json", return_value=None):
        fce.resolve_fast_wizard_turn("1 + 1")  # warm lazy singleton imports
        t0 = time.perf_counter()
        math_res = fce.resolve_fast_wizard_turn("solve x^2 - 5*x + 6 = 0")
        dt_math = (time.perf_counter() - t0) * 1000.0
        assert math_res is not None
        assert "2" in math_res["answer"] and "3" in math_res["answer"]
        assert dt_math < 500.0

        t1 = time.perf_counter()
        code_res = fce.resolve_fast_wizard_turn("Write a Python function to compute Fibonacci numbers")
        dt_code = (time.perf_counter() - t1) * 1000.0
        assert code_res is not None
        assert "def fib(" in code_res["answer"]
        assert dt_code < 500.0

        t2 = time.perf_counter()
        telem_res = fce.resolve_fast_wizard_turn("How much RAM are you using right now?")
        dt_telem = (time.perf_counter() - t2) * 1000.0
        assert telem_res is not None
        assert "W1z4rD Node RAM" in telem_res["answer"]
        assert dt_telem < 150.0


def test_fast_agent_mode_code_execution_and_file_read(tmp_path: Path):
    from web.wizard_chat import agent_runner

    sample = tmp_path / "hello_mod.py"
    sample.write_text("VALUE = 42\n", encoding="utf-8")

    with patch.object(agent_runner, "_audit"):
        # 1. File read (<10ms)
        read_out = agent_runner._fast_c0d3r_direct_agent_turn(
            "read hello_mod.py",
            workdir=tmp_path,
            session_key="test-agent-sess",
            allow_admin=False,
            wizard_endpoint="http://127.0.0.1:8090",
        )
        assert read_out is not None
        assert "VALUE = 42" in read_out

        # 2. Code synthesis + live Python verification + workspace save
        with patch("tools.wizard_learning.fast_conversational_engine.train_fast_chat_pair_async"):
            code_out = agent_runner._fast_c0d3r_direct_agent_turn(
                "Write a Python function fib(n) and test fib(10), save to out/fib_tool.py",
                workdir=tmp_path,
                session_key="test-agent-sess",
                allow_admin=False,
                wizard_endpoint="http://127.0.0.1:8090",
            )
        assert code_out is not None
        assert "55" in code_out
        assert (tmp_path / "out" / "fib_tool.py").is_file()


def test_unified_eye_dynamics_single_roundtrip_zero_brain_lock():
    from tools.wizard_learning import neural_eye_dynamics as ned

    calls = []

    def fake_single_http(url: str, payload=None, timeout: float = 4.0):
        calls.append(url)
        if url.endswith("/multi_pool/ask"):
            return {
                "status": "ok",
                "predictions": {
                    "social_affect": "soc:t:warm soc:t:encouraging",
                    "ocular_identity": "",
                },
            }
        raise AssertionError(f"Unexpected blocking HTTP call on critical path: {url}")

    with patch.object(ned, "_http_json", side_effect=fake_single_http):
        sent, redesign = ned.probe_unified_eye_dynamics(
            {"id": "local", "endpoint": "http://localhost:8090"},
            user_text="Hello friend!",
            cached_neuromodulators={"dopamine": 0.6, "norepinephrine": 0.3, "acetylcholine": 0.55, "serotonin": 0.65},
        )
        assert len(calls) == 1
        assert calls[0] == "http://127.0.0.1:8090/multi_pool/ask"
        assert sent["pool_grounded"] is True
        assert sent["dominant_sentiment"] == "warm"
        assert redesign is None


def test_context_clock_data_polyglot_and_silence_contract():
    import json as _json
    from tools.wizard_learning import fast_conversational_engine as fce
    from tools.wizard_learning.runtime_context import build_runtime_context

    ctx = build_runtime_context(caller_context={"location": "Columbus, Ohio, USA", "project_name": "Revenir"})

    with patch.object(fce, "_http_post_json", return_value=None):
        # 1. Time & JSON format adherence
        t_res = fce.resolve_fast_wizard_turn("What is the current time in JSON?", runtime_context=ctx)
        assert t_res is not None
        parsed_t = _json.loads(t_res["answer"])
        assert "time_24h" in parsed_t and "timezone" in parsed_t

        # 2. Location surfaced ONLY when asked
        loc_res = fce.resolve_fast_wizard_turn("Where am I?", runtime_context=ctx)
        assert loc_res is not None and "Columbus, Ohio, USA" in loc_res["answer"]

        # 3. Strict silence: location/context must NOT leak on unrelated math query
        math_res = fce.resolve_fast_wizard_turn("Calculate 19 * 23", runtime_context=ctx)
        assert math_res is not None and "437" in math_res["answer"]
        assert "Columbus, Ohio, USA" not in math_res["answer"]

        # 4. Inline JSON transformation
        d_res = fce.resolve_fast_wizard_turn('Given {"price": 25, "qty": 4}, compute the total as JSON', runtime_context=ctx)
        assert d_res is not None and _json.loads(d_res["answer"])["total"] == 100

        # 5. Polyglot Rust code synthesis
        r_res = fce.resolve_fast_wizard_turn("Write a Rust function to compute Fibonacci numbers", runtime_context=ctx)
        assert r_res is not None and "pub fn fib(" in r_res["answer"]

        # 6. Caller context extraction
        c_res = fce.resolve_fast_wizard_turn("What is the project_name in context?", runtime_context=ctx)
        assert c_res is not None and "Revenir" in c_res["answer"]


def test_enterprise_stem_steam_domain_synthesis():
    from tools.c0d3rV2.enterprise_stem_synth import synthesize_domain_software

    # 1. STEM RK4 ODE
    rk4 = synthesize_domain_software("Build a 4th-order Runge-Kutta (RK4) ODE solver in Python")
    assert rk4 is not None and rk4[0] == "python" and "def rk4_step" in rk4[2]

    # 2. Wet-Lab LIMS Z'-factor (must NOT misroute to prime checker!)
    zprime = synthesize_domain_software("Create a LIMS high-throughput screening assay QC module for Z-prime factor in Python")
    assert zprime is not None and zprime[0] == "python" and "compute_z_prime" in zprime[2]

    # 3. STEAM WebGL GLSL shader
    shader = synthesize_domain_software("Write a GLSL WebGL fragment shader for an animated SDF smooth-min metaball surface")
    assert shader is not None and shader[0] == "glsl" and "gl_FragColor" in shader[2]

    # 4. $MM Multi-Tenant TypeScript RBAC
    ts_rbac = synthesize_domain_software("Create a TypeScript RBAC permission & multi-tenant isolation guard")
    assert ts_rbac is not None and ts_rbac[0] == "typescript" and "enforceTenantRbac" in ts_rbac[2]

    # 5. $B Enterprise Double-Entry Decimal Ledger & Production SQL CTEs
    ledger = synthesize_domain_software("Build an immutable double-entry accounting ledger engine in Python using Decimal")
    assert ledger is not None and ledger[0] == "python" and "ImmutableDoubleEntryLedger" in ledger[2]
    sql_cte = synthesize_domain_software("Write a production SQL enterprise reconciliation query using CTEs and window functions")
    assert sql_cte is not None and sql_cte[0] == "sql" and "PARTITION BY" in sql_cte[2]


def test_agent_multi_turn_session_followup_and_benchmark(tmp_path: Path):
    from web.wizard_chat import agent_runner

    with (
        patch.object(agent_runner, "_audit"),
        patch("tools.wizard_learning.fast_conversational_engine.train_fast_chat_pair_async"),
    ):
        turn1 = agent_runner._fast_c0d3r_direct_agent_turn(
            "Build a thread-safe LRU cache with TTL expiration in Python and run quick tests on eviction and expiry.",
            workdir=tmp_path,
            session_key="pytest-mt-session",
            allow_admin=False,
            wizard_endpoint="http://127.0.0.1:8090",
        )
        assert turn1 is not None and "ThreadSafeLRUTTLCache" in turn1 and "exit 0" in turn1

        turn2 = agent_runner._fast_c0d3r_direct_agent_turn(
            "Nice — now save that last Python module to scratch/lru_mod.py and run a benchmark of 2,500 lookups on it.",
            workdir=tmp_path,
            session_key="pytest-mt-session",
            allow_admin=False,
            wizard_endpoint="http://127.0.0.1:8090",
        )
        assert turn2 is not None
        assert "scratch/lru_mod.py" in turn2
        assert "2500 lookups" in turn2
        assert "exit 0" in turn2
        assert (tmp_path / "scratch" / "lru_mod.py").is_file()


def test_civilizational_causal_redundancy_and_untrained_reinvention():
    from tools.wizard_learning.civilizational_causal_engine import (
        RedundantCausalConvergenceModel,
        resolve_causal_discovery_or_untrained_mechanics,
    )

    model = RedundantCausalConvergenceModel.get_instance()
    assert model.invariance_ratio > 2.5

    # 1. Pure constituent transfer on unfamiliar synthetic substrate
    alien = resolve_causal_discovery_or_untrained_mechanics(
        "Untrained constituent transfer: suppose unfamiliar `Substrate Zeta-9` has delocalized electrons in a multi-turn loop exposed to a rotating magnet with a closed conductive loop. Determine the emergent mechanics and relate them to your prerogatives."
    )
    assert alien is not None
    assert "Electromagnetic Induction" in alien["answer"]
    assert "W1z4rD Prerogatives" in alien["answer"]

    # 2. Counterfactual broken precondition (stationary magnet -> zero continuous EMF)
    cf = resolve_causal_discovery_or_untrained_mechanics(
        "What happens when an induction generator coil is operated with a stationary magnet? Deduce which necessary condition fails and relate to your goals."
    )
    assert cf is not None
    assert "Violated Condition Detected" in cf["answer"]
    assert "ZERO" in cf["answer"] or "zero" in cf["answer"]

    # 3. Unseen 2-law cross-domain composition (Optics lens + Carnot heat engine)
    comp = resolve_causal_discovery_or_untrained_mechanics(
        "Untrained cross-domain scenario: suppose we couple a parabolic mirror solar concentrator with a Stirling heat engine — determine how their mechanics chain together and relate to your goals."
    )
    assert comp is not None
    assert "Compositional First-Principles Synthesis" in comp["answer"]


def test_pc_diagnostics_aplus_and_observation_override_engine():
    from tools.wizard_learning.pc_diagnostics_and_os_engine import (
        resolve_pc_diagnostics_or_aplus_query,
        run_harmless_live_pc_diagnostics,
    )

    # 1. Live harmless multi-OS diagnostic probe
    diag = run_harmless_live_pc_diagnostics()
    assert diag["read_only_harmless"] is True
    assert diag["memory"]["total_gb"] > 0
    assert len(diag["volumes"]) >= 1

    # 2. RAID math + broken precondition detection
    r6 = resolve_pc_diagnostics_or_aplus_query("Calculate usable capacity for 6 x 8 TB NVMe drives in RAID 6.")
    assert r6 is not None and "32 TB" in r6["answer"]
    r5_bad = resolve_pc_diagnostics_or_aplus_query("Configure 2 x 4 TB SSDs in RAID 5 — what happens?")
    assert r5_bad is not None and "Precondition Failure" in r5_bad["answer"]

    # 3. APIPA observation overriding DNS assumption + S.M.A.R.T. overriding chkdsk /r
    apipa = resolve_pc_diagnostics_or_aplus_query(
        "User blames DNS, but ipconfig shows IPv4 169.254.44.19. What failed?"
    )
    assert apipa is not None and "DHCP" in apipa["answer"]
    smart = resolve_pc_diagnostics_or_aplus_query(
        "Admin plans chkdsk /r, but smartctl shows Current_Pending_Sector = 38. How must the plan mutate?"
    )
    assert smart is not None and "ddrescue" in smart["answer"]


def test_multimodal_desktop_visual_feature_recognition_and_action_mutation():
    import cv2
    import numpy as np
    from tools.wizard_learning.multimodal_desktop_navigator import (
        extract_desktop_visual_features,
        plan_and_mutate_desktop_action,
    )

    # Synthesize a 1024x768 desktop frame with bottom taskbar + app window + centered red blocking modal
    img = np.full((768, 1024, 3), (71, 52, 36), dtype=np.uint8)
    # Bottom taskbar
    cv2.rectangle(img, (0, 720), (1024, 768), (29, 22, 17), -1)
    cv2.rectangle(img, (14, 730), (46, 758), (239, 164, 0), -1)
    # App window
    cv2.rectangle(img, (130, 80), (890, 620), (252, 250, 247), -1)
    cv2.rectangle(img, (130, 80), (890, 116), (240, 232, 226), -1)
    cv2.rectangle(img, (852, 86), (882, 110), (62, 62, 229), -1)
    # Centered red blocking modal with Acknowledge button
    cv2.rectangle(img, (310, 235), (714, 485), (255, 255, 255), -1)
    cv2.rectangle(img, (310, 235), (714, 485), (44, 32, 26), 2)
    cv2.rectangle(img, (310, 235), (714, 277), (37, 48, 217), -1)
    cv2.rectangle(img, (566, 429), (686, 467), (176, 108, 43), -1)

    ok, png_buf = cv2.imencode(".png", img)
    assert ok
    vf = extract_desktop_visual_features(png_buf.tobytes())
    assert "bottom" in vf["layout"]["panel_edges"]
    assert "ui:modal_blocker_center" in vf["pool_image_tokens"]
    assert "modal_action_button" in vf["affordances"]

    plan = plan_and_mutate_desktop_action("Click Continue primary button", vf, assumed_os="windows11")
    assert plan["mutated"] is True
    assert plan["selected_action"]["action_type"] == "click_modal_dismiss_or_ack"
    assert 566 <= plan["selected_action"]["x"] <= 686
    assert 429 <= plan["selected_action"]["y"] <= 467


def test_wbrain_v4_decentralized_p2p_mesh_and_strict_testnet_gate():
    import pytest
    from tools.wizard_learning.wbrain_decentralized_mesh import (
        MainnetBlockedError,
        WbrainDecentralizedMeshManager,
        assert_testnet_only,
        pack_wbrain_v4_shard,
        unpack_wbrain_v4_shard,
    )

    # 1. Strict Testnet-Only gate MUST reject mainnet chains
    for bad_chain in ("ETHEREUM_MAINNET", "BASE_MAINNET", "SOLANA_MAINNET", "ETH_LIVE"):
        with pytest.raises(MainnetBlockedError):
            assert_testnet_only(bad_chain)
    assert assert_testnet_only("BASE_SEPOLIA") == "BASE_SEPOLIA"

    # 2. Exact W1ZBRAIN v4 binary container roundtrip + SHA-256 tamper rejection
    raw, meta = pack_wbrain_v4_shard(
        [{"question": "Rapid Diplomat P2P Check", "answer": "Verified W1ZBRAIN v4 shard"}],
        domain="unit_test_mesh",
        generation=2,
    )
    assert raw[:8] == b"W1ZBRAIN"
    mf, recs = unpack_wbrain_v4_shard(raw)
    assert mf["generation"] == 2
    assert len(recs) == 1 and recs[0]["answer"] == "Verified W1ZBRAIN v4 shard"

    tampered = bytearray(raw)
    tampered[-3] ^= 0xFF
    with pytest.raises(ValueError):
        unpack_wbrain_v4_shard(bytes(tampered))

    # 3. Out-of-order P2P chunk reassembly
    mgr = WbrainDecentralizedMeshManager.get_instance()
    dm, chunks = mgr.chunk_wbrain_for_p2p(raw, meta, chunk_size=1024)
    assert len(chunks) >= 4
    reassembled = mgr.reassemble_p2p_chunks(dm, list(reversed(chunks)))
    assert reassembled == raw


def test_chevy_colorado_vehicle_companion_projects_and_safety():
    import pytest
    import shutil
    from web.wizard_chat.vehicle_companion_views import (
        VEHICLE_PROJECTS_ROOT,
        _concise_spoken_summary,
        _handle_project_intent,
        _safe_project_path,
    )

    # 1. Path traversal outside vehicle_projects sandbox MUST be blocked
    with pytest.raises(ValueError):
        _safe_project_path("colorado_test", "../../outside.py")

    # 2. Spoken summary strips raw fenced code blocks for NHTSA driving safety
    spoken = _concise_spoken_summary("Built module:\n```python\ndef foo():\n    return 42\n```\nReady for review.")
    assert "```" not in spoken
    assert len(spoken) <= 260

    # 3. Multi-file Python project scaffolding + AST verification
    slug = "pytest_colorado_telemetry"
    proj_dir = VEHICLE_PROJECTS_ROOT / slug
    if proj_dir.exists():
        shutil.rmtree(proj_dir)
    try:
        res = _handle_project_intent("Create project pytest_colorado_telemetry in Python for CAN anomaly detection")
        assert res is not None and res["mode"] == "project_engineering"
        assert res["project_manifest"]["all_verified"] is True
        assert len(res["project_manifest"]["files"]) >= 4

        upd = _handle_project_intent("Work on project pytest_colorado_telemetry and add module alert_router")
        assert upd is not None and upd["project_manifest"]["all_verified"] is True
        assert len(upd["project_manifest"]["files"]) >= 5
    finally:
        if proj_dir.exists():
            shutil.rmtree(proj_dir)


def test_compositional_domain_coder_speech_act_guard_and_gutenberg_concept_graph():
    import ast
    from tools.wizard_learning.compositional_domain_coder import (
        execute_python_isolated,
        is_code_or_model_creation_intent,
        resolve_compositional_code_or_model,
    )
    from tools.wizard_learning.context_and_data_engine import resolve_online_lookup_if_needed
    from tools.wizard_learning.gutenberg_concept_map import (
        GutenbergConceptMap,
        resolve_gutenberg_concept_query,
    )

    # 1. Speech-act separation: code/model creation vs encyclopedic prose questions
    assert is_code_or_model_creation_intent("Create a python class for a bike.") is True
    assert is_code_or_model_creation_intent("Create a model of a bike in Python.") is True
    assert is_code_or_model_creation_intent("Write a Rust struct for a submarine.") is True
    assert is_code_or_model_creation_intent("What is a bike?") is False
    assert is_code_or_model_creation_intent("Who invented the barometer?") is False

    # Hard guard: online lookup MUST NEVER return raw encyclopedia prose for imperative code/model creation prompts
    bike_lookup = resolve_online_lookup_if_needed("Create a python class for a bike.", allow_auto_lookup=True)
    if bike_lookup is not None:
        assert "class Bike" in bike_lookup["answer"]
        assert "decorator pattern" not in bike_lookup["answer"].lower()
    bike_model_lookup = resolve_online_lookup_if_needed("Create a model of a bike in Python.", allow_auto_lookup=True)
    if bike_model_lookup is not None:
        assert "class Bike" in bike_model_lookup["answer"]
        assert "bicycle and motorcycle dynamics is the science" not in bike_model_lookup["answer"].lower()

    # 2. Project Gutenberg Concept-Map lookup & multi-hop relational path
    cmap = GutenbergConceptMap.get_instance()
    assert cmap.size >= 20000
    bike_node = cmap.lookup("bicycle")
    assert bike_node is not None
    assert "vehicle" in bike_node["is_a"]

    rel_ans = resolve_gutenberg_concept_query("How are bicycle and locomotive related?")
    assert rel_ans is not None
    assert "Concept-Map Relationship" in rel_ans["answer"]

    # 3. Offline Compositional Modeling + AST + Isolated Subprocess Execution (`allow_web_lookup=False`)
    model_res = resolve_compositional_code_or_model(
        "Create a model of a barometer in Python.",
        allow_web_lookup=False,
        execute_live=True,
    )
    assert model_res is not None
    assert "class Barometer" in model_res["answer"]
    if model_res.get("code"):
        ast.parse(model_res["code"])
        ex = execute_python_isolated(model_res["code"])
        assert ex["ok"] is True




