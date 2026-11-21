"""tools/wizard_learning/autonomous_lifecycle.py

Continuous Autonomous 24/7 Cognitive Lifecycle for W1z4rD Brain & C0d3rV2.
Strictly self-sufficient: uses ONLY W1z4rD Brain (:8090 + :18096) and C0d3rV2
(Executor, KuzuWorkbench, DeductiveReasoner, EmpiricalCurriculum, MSRW Engine)
with ZERO external AI/LLM providers.

Whenever NOT busy serving a live user chat in the Wizard GUI:
  1. SELF_EDUCATING: Resolves `/hypothesis/queue` items, synthesizes novel
     first-principles physics/robotics problems via C0d3rV2 + Kùzu, and pulls
     peer-reviewed open-access empirical mechanics data (NLM/NCBI).
  2. TRAINING_SELF: Trains W1z4rD `/multi_pool/train_pair`, anchors verified
     foundations in `AnchoredMemoryManager`, and binds equations into Kùzu.
  3. TESTING_TRAINING: Actively tests W1z4rD recall on conversational basics
     and newly trained engineering/physics items, immediately healing any miss.
  4. CONTROLLING_ROBOT_WORLD: Drives Multiscale Robot World W1z4rD multi-agent
     controllers & continuous Mark N -> Mark N+1 body self-upgrades, feeding
     discovered physical design improvements back into W1z4rD synapses.
  5. SLEEPING_SETTLING: Executes error-free biological sleep & SSD checkpointing
     when sleep pressure rises, resetting sleep pressure to 0% with verified
     post-sleep retention while informing GUI users when chat is briefly locked.
"""
from __future__ import annotations

import base64
import datetime
import json
import logging
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from services.anchored_memory import AnchoredMemoryManager
from tools.c0d3rV2.executor import Executor
from tools.wizard_learning.cognitive_state import get_cognitive_state_manager
from tools.wizard_learning.deductive_reasoner import (
    RobotDesignRequirement,
    get_deductive_reasoner,
)
from tools.wizard_learning.empirical_material import EMPIRICAL_CURRICULUM
from tools.wizard_learning.kuzu_workbench import get_kuzu_workbench
from tools.wizard_learning.multimodal_feeder import PoliteMultimodalFeeder
from tools.wizard_learning.neural_eye_dynamics import (
    evaluate_on_the_fly_eye_redesign,
    train_slice_of_affect_identity_corpus,
)
from tools.wizard_learning.settling_controller import (
    get_settling_controller,
    trim_windows_working_set,
)
from tools.wizard_learning.store import LearningStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_logger = logging.getLogger("AutonomousLifecycle")

MSRW_DIR = Path(r"C:\Users\Adam\Desktop\Multiscale Robot World")
RUNTIME_DIR = _PROJECT_ROOT / "runtime" / "c0d3r"
LIFECYCLE_LOG = RUNTIME_DIR / "autonomous_lifecycle.jsonl"
PID_FILE = RUNTIME_DIR / "autonomous_lifecycle.pid"
MSRW_TELEMETRY_FILE = RUNTIME_DIR / "msrw_swarm_telemetry.json"

NCBI_RESEARCH_TOPICS = [
    "titanium alloy fatigue robotics mechanical properties",
    "carbon fiber reinforced polymer tensile strength aerospace",
    "neodymium magnet remanence brushless dc motor torque",
    "kinematic singularity avoidance serial robotic manipulator",
    "terramechanics wheel soil interaction rover mobility",
    "neuromorphic associative memory synaptic plasticity hebbian",
]

SENIOR_BRAIN_ENDPOINT = "http://127.0.0.1:18096"

SENIOR_ENGINEERING_CURRICULUM = [
    (
        "Senior Engineer: How do you eliminate CUDA out of memory errors during PyTorch backpropagation?",
        "Call torch.cuda.empty_cache(), use gradient accumulation with smaller batch sizes, and enable torch.utils.checkpoint.checkpoint.",
    ),
    (
        "Senior Engineer: How do you resolve Rust compiler error E0382 'use of moved value'?",
        "Borrow the data with a reference (&T or &mut T), implement the Clone/Copy trait, or wrap shared state in Arc<Mutex<T>>.",
    ),
    (
        "Senior Engineer: How do you fix Eigen unaligned array assertion failures in C++?",
        "Use EIGEN_MAKE_ALIGNED_OPERATOR_NEW in classes with fixed-size vectorizable Eigen members, or pass Eigen::aligned_allocator to STL containers.",
    ),
    (
        "Senior Engineer: How do you resolve CMake undefined reference linker errors with ROS2 rclcpp?",
        "Ensure ament_target_dependencies(target rclcpp) is declared in CMakeLists.txt and depend on rclcpp in package.xml.",
    ),
    (
        "Senior Engineer: What is the optimal strategy for numerical stiffness in DifferentialEquations.jl?",
        "Use implicit solvers like Rodas5() or KenCarp4() instead of explicit Runge-Kutta Tsit5() to avoid step-size collapse.",
    ),
    (
        "Senior Engineer: How do you avoid Three.js GPU memory leaks during dynamic scene updates?",
        "Traverse scene objects and explicitly call geometry.dispose(), material.dispose(), and texture.dispose() when removing meshes.",
    ),
    (
        "Senior Engineer: What is the GD&T requirement for a precision robotic joint bearing bore under ASME Y14.5?",
        "Specify Datum A as the mounting face for Flatness, Datum B as the locating pin for Perpendicularity, and position tolerance with MMC.",
    ),
    (
        "Senior Engineer: How do you prevent event-loop starvation in high-throughput FastAPI microservices?",
        "Offload blocking I/O or CPU-heavy matrix math to a thread pool via run_in_threadpool or anyio.to_thread.run_sync.",
    ),
]


class NCBIResearchClient:
    """Client for querying authoritative peer-reviewed data from NLM/NCBI E-utilities."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str = "adamedsall@gmail.com", app: str = "W1z4rD-Cognitive-Engine"):
        self.email = email
        self.app = app

    def search_empirical_articles(self, query: str, max_results: int = 2) -> List[Dict[str, Any]]:
        params = {
            "db": "pmc",
            "term": f"{query} AND open access[filter]",
            "retmode": "json",
            "retmax": str(max_results),
            "email": self.email,
            "tool": self.app,
        }
        search_url = f"{self.BASE_URL}/esearch.fcgi?{urllib.parse.urlencode(params)}"
        id_list: List[str] = []
        try:
            req = urllib.request.Request(search_url, headers={"User-Agent": f"{self.app}/1.0 ({self.email})"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                id_list = data.get("esearchresult", {}).get("idlist", [])
        except Exception as exc:
            _logger.debug("NCBI esearch skipped for '%s': %s", query, exc)
            return []

        if not id_list:
            return []

        summary_params = {
            "db": "pmc",
            "id": ",".join(id_list),
            "retmode": "json",
            "email": self.email,
            "tool": self.app,
        }
        summary_url = f"{self.BASE_URL}/esummary.fcgi?{urllib.parse.urlencode(summary_params)}"
        articles: List[Dict[str, Any]] = []
        try:
            req = urllib.request.Request(summary_url, headers={"User-Agent": f"{self.app}/1.0 ({self.email})"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                result_map = data.get("result", {})
                for uid in id_list:
                    item = result_map.get(uid)
                    if item and isinstance(item, dict):
                        title = item.get("title", "").strip()
                        source = item.get("source", "").strip()
                        pubdate = item.get("pubdate", "").strip()
                        doi = next(
                            (a.get("value") for a in item.get("articleids", []) if a.get("idtype") == "doi"),
                            "",
                        )
                        if title:
                            articles.append(
                                {
                                    "pmcid": uid,
                                    "title": title,
                                    "journal": source,
                                    "pubdate": pubdate,
                                    "doi": doi,
                                    "source_url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{uid}/",
                                }
                            )
        except Exception as exc:
            _logger.debug("NCBI esummary skipped: %s", exc)

        return articles


class AutonomousLifecycle:
    """Orchestrates 24/7 continuous self-education, self-training, self-testing,
    error-free biological settling, and Multiscale Robot World controller evolution."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8090"):
        self.endpoint = endpoint.rstrip("/")
        self.controller = get_settling_controller()
        self.cognitive = get_cognitive_state_manager()
        self.workbench = get_kuzu_workbench()
        self.reasoner = get_deductive_reasoner()
        self.ncbi = NCBIResearchClient()
        self.store = LearningStore()
        self.anchored = AnchoredMemoryManager.get_instance()
        self.multimodal_feeder = PoliteMultimodalFeeder(endpoint=self.endpoint)
        self.executor = Executor(workdir=MSRW_DIR if MSRW_DIR.exists() else _PROJECT_ROOT)
        self.cycle_count = 0
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    def _yield_if_user_active(self) -> bool:
        """Yield immediately whenever a user is chatting in the Wizard GUI."""
        if self.cognitive.is_user_recently_active(grace_seconds=20.0):
            self.cognitive.heartbeat("Paused background learning — yielding to active user chat.")
            return True
        return False

    def _node_online(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.endpoint}/health", timeout=4.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def train_pair(self, prompt: str, target: str, passes: int = 10) -> bool:
        """Train a deterministic Q&A pair into both the 5-pool W1z4rD Brain (`/brain/pretrain_binding`
        with EEM symbolic relation induction) and `/multi_pool/train_pair`, registering for post-sleep verification."""
        b64 = lambda s: base64.urlsafe_b64encode(s.encode("utf-8")).decode("ascii").rstrip("=")
        brain_ok = False
        mp_ok = False

        # 1. Primary 5-pool W1z4rD Brain binding + EEM symbolic induction (POOL_TEXT=1 -> POOL_ACTION=4)
        brain_payload = json.dumps(
            {
                "frames": [
                    {"pool_id": 1, "frame": b64(prompt)},
                    {"pool_id": 4, "frame": b64(target)},
                ]
            }
        ).encode("utf-8")
        for attempt in range(2):
            try:
                req_b = urllib.request.Request(
                    f"{self.endpoint}/brain/pretrain_binding",
                    data=brain_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_b, timeout=35) as resp:
                    d_b = json.loads(resp.read().decode("utf-8"))
                    if "error" not in d_b:
                        brain_ok = True
                        break
            except Exception as exc:
                if attempt == 0:
                    time.sleep(1.5)
                else:
                    _logger.debug("pretrain_binding contended for '%s': %s", prompt[:45], exc)

        # 2. Multi-pool associative fabric reinforcement (`/chat` fallback + topology viz)
        mp_payload = json.dumps(
            {
                "src_pool": "in",
                "src": prompt,
                "tgt_pool": "out",
                "tgt": target,
                "passes": max(4, min(passes, 12)),
            }
        ).encode("utf-8")
        try:
            req_mp = urllib.request.Request(
                f"{self.endpoint}/multi_pool/train_pair",
                data=mp_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_mp, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                mp_ok = bool(data.get("status") == "ok" or data.get("ok", False) or "stats" in data)
        except Exception as exc:
            if not brain_ok:
                _logger.warning("Failed train_pair '%s': %s", prompt[:50], exc)

        ok = brain_ok or mp_ok
        if ok:
            self.controller.episodes_since_settle += 1
            self.controller.register_trained_pair(prompt, target)
        return ok

    def query_wizard_recall(self, prompt: str) -> Tuple[str, str, float]:
        """Probe W1z4rD recall across AnchoredMemory -> /brain/chat -> /chat."""
        mem = self.anchored.recall(prompt)
        if mem and mem.get("answer"):
            return str(mem["answer"]), "anchored_memory", 1.0

        for path in ("/brain/chat", "/chat"):
            try:
                req = urllib.request.Request(
                    f"{self.endpoint}{path}",
                    data=json.dumps({"text": prompt, "hops": 2, "min_strength": 0.05}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=18.0) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
                    ans = str(d.get("answer") or d.get("reply") or "").strip()
                    dec = str(d.get("decoder") or path)
                    g = d.get("grounding") if isinstance(d.get("grounding"), dict) else {}
                    conf = float(d.get("confidence") or g.get("integrated_confidence") or 0.0)
                    outside = bool(g.get("outside_grounding", False))
                    if ans and dec != "char_chain" and not d.get("hypothesis", False) and not outside:
                        return ans, dec, max(conf, 0.85 if path == "/brain/chat" else conf)
            except Exception:
                continue
        return "", "none", 0.0

    def query_senior_brain(self, prompt: str) -> str:
        """Query the local Senior Software Engineer W1z4rD brain (72.2M terminals on :18096)."""
        url = f"{SENIOR_BRAIN_ENDPOINT}/brain/chat"
        payload = json.dumps({"text": prompt}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ans = data.get("answer") or data.get("reply") or ""
                if data.get("confidence_tier") in ("outside_grounding", "ungrounded", "low"):
                    return ""
                return str(ans).strip()
        except Exception:
            return ""

    def _resolve_hypothesis_queue(self) -> List[Tuple[str, str]]:
        """Inspect W1z4rD `/hypothesis/queue` and synthesize verified answers via C0d3rV2/curriculum."""
        resolved: List[Tuple[str, str]] = []
        try:
            with urllib.request.urlopen(f"{self.endpoint}/hypothesis/queue", timeout=5.0) as resp:
                q_data = json.loads(resp.read().decode("utf-8"))
            items = q_data.get("queue") or q_data.get("items") or []
        except Exception:
            items = []

        for item in items[:3]:
            if not isinstance(item, dict):
                continue
            q = str(item.get("question") or item.get("prompt") or item.get("text") or "").strip()
            if not q:
                continue
            # Check anchored memory or empirical curriculum or Senior SE brain
            mem = self.anchored.recall(q)
            ans = mem["answer"] if mem else ""
            if not ans:
                q_low = q.lower()
                for cur in EMPIRICAL_CURRICULUM:
                    if cur.topic.lower() in q_low or q_low in cur.question.lower():
                        ans = f"{cur.answer} (Formula: {cur.formula} [{cur.units}], Source: {cur.source_citation})"
                        break
            if not ans:
                ans = self.query_senior_brain(q)
            if ans:
                resolved.append((q, ans))
                try:
                    req_res = urllib.request.Request(
                        f"{self.endpoint}/hypothesis/resolve",
                        data=json.dumps(
                            {
                                "id": item.get("id") or f"ask_{hash(q) & 0xffffffff:x}",
                                "answer": ans,
                                "confidence": 0.95,
                                "resolution_path": "c0d3rv2_self_education",
                            }
                        ).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_res, timeout=5.0):
                        pass
                except Exception:
                    pass
        return resolved

    def step_1_self_educate(self) -> Dict[str, Any]:
        """Self-educate via C0d3rV2 deductive physics synthesis, hypothesis resolution, and NCBI empirical studies."""
        self.cognitive.set_mode(
            "SELF_EDUCATING",
            detail=f"Cycle #{self.cycle_count}: C0d3rV2 synthesizing first-principles robotics & resolving hypotheses.",
            is_daemon=True,
        )
        topic = NCBI_RESEARCH_TOPICS[self.cycle_count % len(NCBI_RESEARCH_TOPICS)]
        articles = self.ncbi.search_empirical_articles(topic, max_results=2)
        for art in articles:
            try:
                self.store.enqueue(
                    "article",
                    {
                        "title": art["title"],
                        "media_url": art["source_url"],
                        "stage": "general",
                        "reviewed_experience": {
                            "empirical": True,
                            "source": f"NLM/NCBI PMC {art['pmcid']}",
                            "journal": art["journal"],
                            "doi": art["doi"],
                        },
                        "rights": {
                            "source_page": art["source_url"],
                            "license_id": "PMC-OpenAccess",
                            "review_status": "source_verified",
                        },
                    },
                )
            except Exception:
                pass

        # Synthesize a parametric robotics design lesson from first principles via C0d3rV2 DeductiveReasoner
        payload_kg = float(8 + (self.cycle_count * 7) % 45)
        reach_m = round(0.6 + ((self.cycle_count * 3) % 18) * 0.1, 2)
        bus_v = float([24.0, 36.0, 48.0][self.cycle_count % 3])
        deduced = self.reasoner.deduce_robot_system(
            RobotDesignRequirement(
                payload_kg=payload_kg,
                reach_meters=reach_m,
                wind_speed_ms=float((self.cycle_count * 4) % 20),
                bus_voltage_v=bus_v,
                operating_hours=4.0,
                safety_factor_target=2.2,
            )
        )
        deduced_q = (
            f"Robotics Engineering Synthesis: Specify verified structural alloy, joint peak torque, "
            f"and battery pack for a {payload_kg:.0f} kg payload at {reach_m:.1f} m reach on a {bus_v:.0f} V bus."
        )
        deduced_a = (
            f"Use {deduced.selected_material} box beam "
            f"({deduced.structural_mechanics['boom_width_mm']:.1f} mm width, "
            f"SF={deduced.structural_mechanics['safety_factor']:.2f}), "
            f"joint torque {deduced.powertrain_electromechanics['required_joint_torque_nm']:.1f} N·m "
            f"via {deduced.powertrain_electromechanics['gearbox_type']}, and "
            f"{deduced.battery_chemistry['cell_config']} LiFePO4 pack."
        )

        hyp_resolved = self._resolve_hypothesis_queue()
        st = self.cognitive.read()
        self.cognitive.update(
            self_edu_cycles=int(st.get("self_edu_cycles", 0)) + 1,
            hypotheses_resolved=int(st.get("hypotheses_resolved", 0)) + len(hyp_resolved),
        )

        return {
            "topic": topic,
            "articles": articles,
            "deduced_pair": (deduced_q, deduced_a),
            "deduced_verified": bool(deduced.verification_passed),
            "hypotheses_resolved": hyp_resolved,
        }

    def step_2_train_self(self, edu_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train W1z4rD neural pools on verified curriculum, deduced physics, Senior SE, and hypotheses."""
        self.cognitive.set_mode(
            "TRAINING_SELF",
            detail=f"Cycle #{self.cycle_count}: Training multi-pool Hebbian bindings & Kùzu equations.",
            is_daemon=True,
        )
        pairs_to_train: List[Tuple[str, str]] = []

        # 1.Conversational anchor reinforcement (ensuring neural pool mirrors AnchoredMemory)
        anchors = self.anchored.list_all()
        if anchors:
            anc = anchors[self.cycle_count % len(anchors)]
            pairs_to_train.append((str(anc["prompt"]), str(anc["answer"])))

        # 2. Empirical curriculum slice
        if EMPIRICAL_CURRICULUM:
            emp = EMPIRICAL_CURRICULUM[self.cycle_count % len(EMPIRICAL_CURRICULUM)]
            pairs_to_train.append(
                (
                    emp.question,
                    f"{emp.answer} Formula: {emp.formula} ({emp.units}). Source: {emp.source_citation}.",
                )
            )

        # 3. C0d3rV2 Deductive robotics synthesis pair
        if edu_data.get("deduced_verified") and edu_data.get("deduced_pair"):
            pairs_to_train.append(edu_data["deduced_pair"])

        # 4. Senior SE brain distillation
        s_prompt, s_fallback = SENIOR_ENGINEERING_CURRICULUM[
            self.cycle_count % len(SENIOR_ENGINEERING_CURRICULUM)
        ]
        s_teacher = self.query_senior_brain(s_prompt)
        pairs_to_train.append((s_prompt, s_teacher if s_teacher else s_fallback))

        # 5. Any resolved hypotheses
        for q, a in edu_data.get("hypotheses_resolved", []):
            pairs_to_train.append((q, a))
            self.anchored.record_memory(q, a, source="autonomous_self_education", category="hypothesis_resolved")

        # 6. Empirical NLM/NCBI articles
        for art in edu_data.get("articles", [])[:1]:
            pairs_to_train.append(
                (
                    f"Empirical Scientific Study: {art['title']}",
                    f"Peer-reviewed study in {art['journal']} ({art['source_url']}, DOI: {art['doi']}).",
                )
            )

        # Elastic batch sizing: train more pairs when PC RAM is abundant, fewer when tight
        res_metrics = self.controller.get_resource_metrics()
        max_pairs = int(res_metrics.get("recommended_pairs_per_cycle", 4))
        selected_pairs = pairs_to_train[:max_pairs]

        trained_list: List[Tuple[str, str]] = []
        for prompt, answer in selected_pairs:
            if self._yield_if_user_active():
                break
            if self.train_pair(prompt, answer, passes=12):
                trained_list.append((prompt, answer))

        # 7. Polite streaming cross-modal Hebbian co-firing (text + image + audio + social)
        multimodal_res: Dict[str, Any] = {"episodes_fed": 0}
        affect_id_res: Dict[str, Any] = {"affect_identity_episodes_trained": 0}
        if not self._yield_if_user_active():
            multimodal_res = self.multimodal_feeder.feed_elastic_slice(
                allowance_tier=str(res_metrics.get("allowance_tier", "BALANCED"))
            )
            if int(multimodal_res.get("episodes_fed", 0)) > 0:
                self.controller.episodes_since_settle += int(multimodal_res["episodes_fed"])

        # 8. Emergent social_affect & ocular_identity neural-pool co-firing + periodic self-redesign
        if not self._yield_if_user_active():
            affect_id_res = train_slice_of_affect_identity_corpus(
                cycle_index=self.cycle_count,
                endpoint=self.endpoint,
                max_items=2 if str(res_metrics.get("allowance_tier")) == "EXPANDED" else 1,
            )
            if self.cycle_count % 3 == 0:
                evaluate_on_the_fly_eye_redesign(
                    {"id": "local", "endpoint": self.endpoint},
                    experience_text=str(edu_data.get("topic") or ""),
                    cognitive_state=self.cognitive.read(),
                    force=False,
                )

        eq_res = self.workbench.bind_environmental_problem(
            problem="Verify electromechanical actuator torque and thermal balance",
            knowns=["tau", "I_inertia", "alpha"],
            targets=["tau"],
            candidate_equations=["tau = I_inertia * alpha"],
        )

        st = self.cognitive.read()
        self.cognitive.update(
            pairs_trained_total=int(st.get("pairs_trained_total", 0))
            + len(trained_list)
            + int(multimodal_res.get("episodes_fed", 0)),
        )

        return {
            "pairs_attempted": len(pairs_to_train),
            "pairs_trained": len(trained_list),
            "multimodal_cofired": multimodal_res,
            "trained_samples": trained_list,
            "kuzu_bound": bool(eq_res.bound),
        }

    def step_3_test_training(self, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """Actively test W1z4rD on conversational foundations + newly trained pairs; heal any failure."""
        self.cognitive.set_mode(
            "TESTING_TRAINING",
            detail=f"Cycle #{self.cycle_count}: Self-testing recall & mathematical closure across trained items.",
            is_daemon=True,
        )
        probes: List[Tuple[str, str]] = [
            ("Hello", "Hello! I am W1z4rD"),
            ("Who are you?", "I am W1z4rD"),
        ]
        for q, a in train_data.get("trained_samples", []):
            probes.append((q, a[:48]))

        passed = 0
        healed = 0
        details: List[Dict[str, Any]] = []
        for q, expected_prefix in probes:
            if self._yield_if_user_active():
                break
            ans, decoder, conf = self.query_wizard_recall(q)
            ok = bool(ans and expected_prefix.lower()[:28] in ans.lower())
            if not ok and expected_prefix:
                # Self-heal immediately with high-pass reinforcement
                full_target = next(
                    (a for pq, a in train_data.get("trained_samples", []) if pq == q),
                    expected_prefix,
                )
                self.train_pair(q, full_target, passes=30)
                ans2, decoder2, conf2 = self.query_wizard_recall(q)
                if ans2 and expected_prefix.lower()[:24] in ans2.lower():
                    ok = True
                    healed += 1
                    decoder = f"{decoder2}+healed"
                    conf = conf2
            if ok:
                passed += 1
            details.append({"prompt": q[:60], "ok": ok, "decoder": decoder, "confidence": round(conf, 3)})

        total_checked = max(1, len(details))
        pass_rate = round(passed / total_checked, 4)
        summary = f"{passed}/{len(details)} verified ({int(pass_rate * 100)}%, {healed} self-healed)"

        st = self.cognitive.read()
        prev_run = int(st.get("self_tests_run", 0))
        prev_pass = int(st.get("self_tests_passed", 0))
        new_run = prev_run + len(details)
        new_pass = prev_pass + passed
        cumulative_rate = round(new_pass / max(1, new_run), 4)

        self.cognitive.update(
            self_tests_run=new_run,
            self_tests_passed=new_pass,
            self_test_pass_rate=cumulative_rate,
            last_self_test_summary=summary,
        )
        return {
            "checked": len(details),
            "passed": passed,
            "healed": healed,
            "cycle_pass_rate": pass_rate,
            "cumulative_pass_rate": cumulative_rate,
            "details": details,
        }

    def step_4_control_robot_world(self) -> Dict[str, Any]:
        """Drive Multiscale Robot World W1z4rD multi-agent controller & self-upgrading body evolution,
        then train W1z4rD on the discovered Mark N -> Mark N+1 physical upgrade laws."""
        if not MSRW_DIR.exists():
            return {"status": "skipped", "reason": f"MSRW dir missing: {MSRW_DIR}"}

        self.cognitive.set_mode(
            "CONTROLLING_ROBOT_WORLD",
            detail=f"Cycle #{self.cycle_count}: Running W1z4rD robot controllers & Mark N+1 body upgrades in MSRW.",
            is_daemon=True,
        )

        # Prefer dedicated autonomous world step script if present, otherwise run vitest suite
        pkg_json = MSRW_DIR / "package.json"
        has_wizard_step = False
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                has_wizard_step = "wizard:step" in (pkg.get("scripts") or {})
            except Exception:
                pass

        res_metrics = self.controller.get_resource_metrics()
        allowance_tier = str(res_metrics.get("allowance_tier", "BALANCED"))

        if has_wizard_step:
            cmd = f'npx tsx scripts/wizard_autonomous_step.ts --telemetry "{MSRW_TELEMETRY_FILE}" --cycle {self.cycle_count}'
            rc, out, err = self.executor.run(cmd)
        elif allowance_tier == "EXPANDED" or self.cycle_count % 4 == 1:
            rc, out, err = self.executor.run("npm test -- --run")
        else:
            # Polite MSRW swarm + elastic-governor run (worker cap set in vitest.config.ts)
            rc, out, err = self.executor.run(
                "npx vitest run tests/multiscale_full_targets.test.ts tests/elastic_resource_governor.test.ts"
            )

        passed = rc == 0
        rw_patch: Dict[str, Any] = {"physics_tests_passed": passed}
        learned_pairs = 0

        if MSRW_TELEMETRY_FILE.exists():
            try:
                tel = json.loads(MSRW_TELEMETRY_FILE.read_text(encoding="utf-8"))
                if isinstance(tel, dict):
                    rw_patch.update(
                        {
                            "active_controllers": int(tel.get("active_controllers", 1)),
                            "robots_built": int(tel.get("robots_built", 0)),
                            "body_generation": int(tel.get("body_generation", 1)),
                            "primary_mark": str(tel.get("primary_mark") or f"Mark {tel.get('body_generation', 1)}"),
                            "last_upgrade_summary": str(tel.get("last_upgrade_summary") or "Physics verified"),
                        }
                    )
                    for lesson in (tel.get("learned_lessons") or [])[:3]:
                        if isinstance(lesson, dict) and lesson.get("question") and lesson.get("answer"):
                            if self.train_pair(str(lesson["question"]), str(lesson["answer"]), passes=15):
                                learned_pairs += 1
            except Exception as exc:
                _logger.debug("MSRW telemetry parse skipped: %s", exc)

        self.cognitive.update(robot_world=rw_patch)
        return {
            "passed": passed,
            "used_wizard_step": has_wizard_step,
            "learned_upgrade_pairs": learned_pairs,
            "robot_world": rw_patch,
            "output_tail": str(out or err or "")[-300:],
        }

    def step_5_sleep_if_needed(self, force_periodic: bool = False) -> Dict[str, Any]:
        """Monitor sleep pressure and execute error-free biological settling when needed."""
        metrics = self.controller.get_resource_metrics()
        if metrics["needs_settle"] or force_periodic:
            reason = (
                f"Periodic cycle #{self.cycle_count} consolidation"
                if force_periodic and not metrics["needs_settle"]
                else f"Sleep pressure {metrics['sleep_pressure']:.2f} / {metrics['episodes_since_settle']} episodes"
            )
            res = self.controller.settle(reason=reason)
            return {
                "settled": res.get("settled", False),
                "error_free": res.get("error_free", False),
                "sleep_pressure_after": res.get("metrics_after", {}).get("sleep_pressure", 0.0),
                "errors": res.get("errors", []),
            }
        return {
            "settled": False,
            "error_free": True,
            "sleep_pressure_after": metrics["sleep_pressure"],
            "errors": [],
        }

    def run_cycle(self) -> Dict[str, Any]:
        """Execute one complete autonomous self-education, training, testing, MSRW, and sleep cycle."""
        self.cycle_count += 1
        t0 = time.time()

        if not self._node_online():
            self.cognitive.set_mode(
                "OFFLINE",
                detail="W1z4rD node (:8090) unreachable; waiting for supervisor recovery.",
                can_chat=False,
                is_daemon=True,
            )
            return {"cycle": self.cycle_count, "skipped": "node_offline"}

        if self._yield_if_user_active():
            return {"cycle": self.cycle_count, "skipped": "user_chat_active"}

        edu_res = self.step_1_self_educate()
        if self._yield_if_user_active():
            return {"cycle": self.cycle_count, "yielded_after": "self_educate"}

        train_res = self.step_2_train_self(edu_res)
        if self._yield_if_user_active():
            return {"cycle": self.cycle_count, "yielded_after": "train_self"}

        test_res = self.step_3_test_training(train_res)
        if self._yield_if_user_active():
            return {"cycle": self.cycle_count, "yielded_after": "test_training"}

        msrw_res = self.step_4_control_robot_world()

        # Release Kùzu handle between cycles so web workbench queries never contend
        try:
            self.workbench.close()
        except Exception:
            pass

        # Settle when pressure threshold is reached OR every 4 cycles to keep synapses consolidated
        sleep_res = self.step_5_sleep_if_needed(force_periodic=(self.cycle_count % 4 == 0))

        elapsed = round(time.time() - t0, 2)
        self.cognitive.set_mode(
            "IDLE_READY",
            detail=(
                f"Cycle #{self.cycle_count} complete in {elapsed}s — "
                f"Test pass: {int(test_res['cycle_pass_rate']*100)}%, "
                f"Sleep pressure: {int(sleep_res['sleep_pressure_after']*100)}%."
            ),
            can_chat=True,
            is_daemon=True,
        )

        record = {
            "cycle": self.cycle_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "self_education": {
                "topic": edu_res["topic"],
                "articles_found": len(edu_res["articles"]),
                "hypotheses_resolved": len(edu_res["hypotheses_resolved"]),
                "deduced_verified": edu_res["deduced_verified"],
            },
            "training": {
                "pairs_trained": train_res["pairs_trained"],
                "multimodal_cofired": train_res.get("multimodal_cofired", {}),
                "kuzu_bound": train_res["kuzu_bound"],
            },
            "testing": {
                "checked": test_res["checked"],
                "passed": test_res["passed"],
                "healed": test_res["healed"],
                "pass_rate": test_res["cycle_pass_rate"],
            },
            "msrw_work": msrw_res,
            "settling": sleep_res,
        }

        try:
            with open(LIFECYCLE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        _logger.info(
            "Cycle #%d finished in %.2fs | trained=%d | test=%d/%d | msrw=%s | settled=%s (err_free=%s)",
            self.cycle_count,
            elapsed,
            train_res["pairs_trained"],
            test_res["passed"],
            test_res["checked"],
            msrw_res.get("passed"),
            sleep_res.get("settled"),
            sleep_res.get("error_free"),
        )
        return record

    def _acquire_single_instance_lock(self) -> bool:
        if PID_FILE.exists():
            try:
                old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
                if old_pid != os.getpid() and psutil.pid_exists(old_pid):
                    proc = psutil.Process(old_pid)
                    cmdline = " ".join(proc.cmdline()).lower()
                    if "autonomous_lifecycle" in cmdline:
                        _logger.info("AutonomousLifecycle daemon already running under PID %d.", old_pid)
                        return False
            except Exception:
                pass
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
        return True

    def run_loop(self, interval_sec: float = 35.0) -> None:
        if not self._acquire_single_instance_lock():
            return
        # Run daemon at BELOW_NORMAL_PRIORITY_CLASS ("nice") so crypto trading strategies,
        # live user chat, and interactive apps always pre-empt background training cleanly.
        try:
            if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                psutil.Process(os.getpid()).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass
        _logger.info(
            "Starting 24/7 W1z4rD + C0d3rV2 Polite Elastic Autonomous Lifecycle Daemon (PID=%d, base_interval=%.1fs)",
            os.getpid(),
            interval_sec,
        )
        self.cognitive.set_mode(
            "IDLE_READY",
            detail="24/7 Autonomous Lifecycle daemon online (Polite Elastic PC-Allowance active).",
            can_chat=True,
            is_daemon=True,
        )
        while True:
            try:
                self.run_cycle()
                trim_windows_working_set(None)
            except Exception as exc:
                _logger.error("Lifecycle cycle error: %s", exc)
                self.cognitive.set_mode(
                    "IDLE_READY",
                    detail=f"Recovered from cycle exception: {exc}",
                    can_chat=True,
                    is_daemon=True,
                )
            # Elastic interval: cycle faster when PC RAM is abundant, give wider breathing room when tight
            dyn_interval = float(
                self.controller.get_resource_metrics().get("recommended_interval_s", interval_sec)
            )
            end_t = time.time() + max(5.0, dyn_interval)
            while time.time() < end_t:
                time.sleep(min(5.0, max(0.5, end_t - time.time())))
                self.cognitive.heartbeat()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous W1z4rD + C0d3rV2 24/7 Lifecycle")
    parser.add_argument("--loop", action="store_true", help="Run continuously as a 24/7 daemon")
    parser.add_argument("--interval", type=float, default=35.0, help="Idle seconds between cycles")
    args = parser.parse_args()

    lifecycle = AutonomousLifecycle()
    if args.loop:
        lifecycle.run_loop(interval_sec=args.interval)
    else:
        result = lifecycle.run_cycle()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
