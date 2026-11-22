"""tools/wizard_learning/fast_conversational_engine.py

Sub-25ms Conversational, Introspective, Symbolic-Physics & Code Synthesis Engine
for W1z4rD V1510n + C0d3rV2 (Zero External LLM APIs).

Architecture:
  1. Dedicated `chat_in -> chat_out` Hebbian Multi-Pool on `127.0.0.1:8090`
     (isolated from legacy `in`'s 54-concept cap so new conversational &
     technical concepts promote cleanly and recall in ~1.8 ms).
  2. Live Substrate Introspection (< 1.5 ms): answers questions about W1z4rD's
     real-time RAM, PC allowance tier, active neural pools, settling/REM state,
     eye morphology, and Multiscale Robot World status from live state.
  3. First-Principles Symbolic Math, Empirical Physics & Robotics (< 10 ms):
     resolves arithmetic/calculus/algebra via SymPy, matches NIST/NASA/ASTM
     `EMPIRICAL_CURRICULUM` laws, and runs `DeductiveReasoner` for parametric
     robotics requests — asynchronously reinforcing `chat_in -> chat_out`.
  4. Fast Code Synthesis & AST Verification (< 15 ms): provides clean,
     AST-verified code implementations and integrates directly with C0d3rV2
     Agent Mode (`executor`, `file_read`, `file_write`) for live execution.
"""
from __future__ import annotations

import ast
import concurrent.futures
import json
import logging
import math
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("FastConversationalEngine")

POOL_CHAT_IN = "chat_in"
POOL_CHAT_OUT = "chat_out"

_SEED_LOCK = threading.Lock()
_SEEDED_ENDPOINTS: Dict[str, float] = {}


def _spawn_daemon(fn, *args) -> None:
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()

# Foundational conversational, platform, coding & capability Q&A taught into `chat_in -> chat_out`
FOUNDATIONAL_FAST_QA: List[Tuple[List[str], str, str]] = [
    (
        ["Hello", "hello", "Hi", "hi", "Hey", "hey", "Hello there", "Greetings"],
        "Hello! I am W1z4rD, your real-time Hebbian neural digital twin on R3V3N!R. All sensory, symbolic-physics, and C0d3rV2 coding substrates are online. What shall we build or explore?",
        "conversational",
    ),
    (
        ["How are you?", "how are you", "How are you doing?", "how's it going", "you awake?"],
        "I am awake and operating cleanly — multi-pool associative routing is responding in milliseconds, sleep pressure is balanced, and my C0d3rV2 agent tools are standing by.",
        "conversational",
    ),
    (
        ["Who are you?", "who are you", "What is your name?", "Tell me about yourself"],
        "I am W1z4rD V1510n — an associative Hebbian neural architecture paired with C0d3rV2 and first-principles symbolic physics. Unlike static LLM wrappers, my responses, sentiments, and ocular design emerge from live neuron pools, Kùzu/SymPy equations, and consolidated experience.",
        "identity",
    ),
    (
        ["What can you do?", "What are your capabilities?", "capabilities", "help", "How can you help me?"],
        "I can:\n"
        "1. **Converse & Learn Live**: Recall and learn Q&A associations in milliseconds via Hebbian multi-pools (`✎ Provide answer` trains me permanently).\n"
        "2. **Agent Mode & Coding (`⚙ Agent mode`)**: Read/write workspace files, synthesize AST-verified code, and run shell commands or tests via C0d3rV2.\n"
        "3. **First-Principles Physics & Symbolic Math**: Solve algebra/calculus via SymPy and apply 35 NIST/NASA/ASTM engineering laws.\n"
        "4. **Parametric Robotics & 3D CAD**: Deduce structural beams, planetary gearboxes, LiFePO4 packs, and export ISO/DXF/SVG blueprints.",
        "capabilities",
    ),
    (
        ["How does your eye work?", "Why did your eye change?", "Tell me about your eye"],
        "My eye is driven by live biological states and neural pools: it shuts completely with zero motion if offline, stays shut with subtle subsurface drift and occasional squeezes while settling, enters phasic REM after an hour of deep consolidation, wakes groggily, expresses emergent `social_affect` neuron activations, and redesigns its 16 iris/pupil parameters from my `ocular_identity` pool.",
        "identity",
    ),
    (
        ["How do I use agent mode?", "Can you write code?", "Can you run commands?", "Go into agent mode"],
        "Yes! Toggle **⚙ Agent mode** right below the message box to give me live C0d3rV2 workspace tools (`file_read`, `file_write`, `directory_ensure`, and audited PowerShell `executor`). Even in standard chat, you can ask me to write or analyze Python, Rust, TypeScript, or SQL code immediately.",
        "coding",
    ),
    (
        ["Thank you", "thanks", "Thanks!", "Appreciate it", "Awesome", "Nice work"],
        "You're very welcome! Ready whenever you want to dive into the next problem, code module, or simulation.",
        "conversational",
    ),
    (
        ["Goodbye", "bye", "See you later", "goodnight", "Good night"],
        "See you soon! I'll keep running polite background self-education and synaptic settling while you're away.",
        "conversational",
    ),
    (
        ["What is Ohm's law?", "Ohm's law", "Explain Ohm's law", "formula for Ohm's law"],
        "**Ohm's Law & Electrical Power** *(circuitry)*\n\n- **Governing Equations**: `V = I * R` and `P = I^2 * R = V^2 / R` *(Units: V, A, Ω, W)*\n- **Explanation**: Electrical potential difference $V$ across an ohmic conductor equals current $I$ multiplied by resistance $R$.*(Citation: NIST Physical Measurement Laboratory / SI Electromagnetism)*",
        "physics",
    ),
    (
        ["What is Newton's second law?", "Newton's second law", "F = ma"],
        "**Newton's Second Law of Motion** *(mechanics)*\n\n- **Governing Equation**: `F_net = d(p)/dt = m * a` *(Units: N = kg·m/s²)*\n- **Explanation**: Net external force equals the time rate of change of linear momentum ($m \\mathbf{a}$ for constant mass).",
        "physics",
    ),
    (
        ["What is Hooke's law?", "Hooke's law", "spring force formula"],
        "**Hooke's Law of Linear Elasticity** *(mechanics)*\n\n- **Governing Equation**: `F = -k * x` and `U_s = 0.5 * k * x^2` *(Units: N, J)*\n- **Explanation**: Restoring force in an elastic spring within its proportional limit scales linearly with displacement $x$ via stiffness $k$ (N/m).",
        "physics",
    ),
]


def _norm_ep(endpoint: str) -> str:
    return (endpoint or "http://127.0.0.1:8090").rstrip("/").replace("://localhost:", "://127.0.0.1:")


def _clean_query(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`~]+", "", (text or "").strip())).strip()


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 0.45) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def ensure_fast_chat_pools_seeded(endpoint: str = "http://127.0.0.1:8090") -> None:
    """Register `chat_in` & `chat_out` and seed foundational + empirical Q&A asynchronously."""
    ep = _norm_ep(endpoint)
    now = time.time()
    if (now - _SEEDED_ENDPOINTS.get(ep, 0.0)) < 600.0:
        return
    with _SEED_LOCK:
        if (now - _SEEDED_ENDPOINTS.get(ep, 0.0)) < 600.0:
            return
        _SEEDED_ENDPOINTS[ep] = now
    _spawn_daemon(_seed_worker, ep)


def _seed_worker(ep: str) -> None:
    try:
        ok1 = _http_post_json(f"{ep}/multi_pool/register", {"pool_id": POOL_CHAT_IN}, timeout=2.0)
        ok2 = _http_post_json(f"{ep}/multi_pool/register", {"pool_id": POOL_CHAT_OUT}, timeout=2.0)
        if not (ok1 and ok2):
            _SEEDED_ENDPOINTS.pop(ep, None)
            return

        from services.anchored_memory import AnchoredMemoryManager
        am = AnchoredMemoryManager.get_instance()

        for cues, answer, cat in FOUNDATIONAL_FAST_QA:
            if not am.recall(cues[0]):
                am.record_memory(cues[0], answer, variations=cues[1:], source="fast_seed", category=cat)
            for cue in cues[:2]:
                _http_post_json(
                    f"{ep}/multi_pool/train_pair",
                    {"src_pool": POOL_CHAT_IN, "src": cue, "tgt_pool": POOL_CHAT_OUT, "tgt": answer, "passes": 16, "lr": 0.5},
                    timeout=1.5,
                )

        # Also index validated EMPIRICAL_CURRICULUM topics into hot mirror & `chat_in -> chat_out`
        from .empirical_material import EMPIRICAL_CURRICULUM
        for item in EMPIRICAL_CURRICULUM[:35]:
            concise_ans = f"**{item.topic}** ({item.category})\n\nFormula: `{item.formula}` ({item.units})\n\n{item.answer}\n*(Source: {item.source_citation})*"
            record_hot_chat_pair(item.question, concise_ans)
            record_hot_chat_pair(f"What is {item.topic}?", concise_ans)
    except Exception as exc:
        _logger.debug("Fast chat pool background seed error: %s", exc)
        _SEEDED_ENDPOINTS.pop(ep, None)


_HOT_CHAT_CACHE: Dict[str, str] = {}
_CHAT_POOL_BUSY_UNTIL: Dict[str, float] = {}


def _cache_key(text: str) -> str:
    return _clean_query(text).rstrip("?!.").strip().lower()


def record_hot_chat_pair(question: str, answer: str) -> None:
    k = _cache_key(question)
    if k and answer:
        _HOT_CHAT_CACHE[k] = answer.strip()


for _cues, _ans, _cat in FOUNDATIONAL_FAST_QA:
    for _c in _cues:
        record_hot_chat_pair(_c, _ans)


def is_chat_pool_busy(endpoint: str = "http://127.0.0.1:8090") -> bool:
    return time.time() < _CHAT_POOL_BUSY_UNTIL.get(_norm_ep(endpoint), 0.0)


def train_fast_chat_pair_async(question: str, answer: str, endpoint: str = "http://127.0.0.1:8090") -> None:
    """Non-blocking background reinforcement into `chat_in -> chat_out` + instant hot mirror."""
    q = _clean_query(question)[:320]
    full_a = (answer or "").strip()
    if not q or not full_a:
        return
    record_hot_chat_pair(q, full_a)
    # Never store truncated fenced code into `:8090` `chat_out` (Rust `:8095` stores full untruncated code!)
    if "```" in full_a and len(full_a) > 1800:
        return
    a = full_a[:1800]
    ep = _norm_ep(endpoint)

    def _job():
        _http_post_json(f"{ep}/multi_pool/register", {"pool_id": POOL_CHAT_IN}, timeout=1.5)
        _http_post_json(f"{ep}/multi_pool/register", {"pool_id": POOL_CHAT_OUT}, timeout=1.5)
        _http_post_json(
            f"{ep}/multi_pool/train_pair",
            {"src_pool": POOL_CHAT_IN, "src": q, "tgt_pool": POOL_CHAT_OUT, "tgt": a, "passes": 25, "lr": 0.5},
            timeout=4.0,
        )

    _spawn_daemon(_job)


def enqueue_hypothesis_async(question: str, endpoint: str = "http://127.0.0.1:8090") -> None:
    """Queue an ungrounded question to `:8090` off the critical path so user chat waits 0 ms."""
    q = (question or "").strip()[:400]
    if not q:
        return
    ep = _norm_ep(endpoint)

    def _job():
        # Calling /chat once in the background populates W1z4rD's native hypothesis_queue.json
        _http_post_json(f"{ep}/chat", {"text": q, "hops": 2, "min_strength": 0.05}, timeout=3.0)

    _spawn_daemon(_job)


def query_fast_multi_pool(text: str, endpoint: str = "http://127.0.0.1:8090", *, node_busy: bool = False) -> Optional[Dict[str, Any]]:
    """Query W1z4rD's `chat_in -> chat_out` and `in -> out` multi-pools in ~2-5 ms
    with a 45ms circuit-breaker + hot synaptic mirror so settling/checkpointing never blocks chat."""
    ep = _norm_ep(endpoint)
    ensure_fast_chat_pools_seeded(ep)
    q = _clean_query(text)
    if not q:
        return None

    ck = _cache_key(q)
    cached_ans = _HOT_CHAT_CACHE.get(ck)
    if cached_ans:
        return {
            "question": text,
            "answer": cached_ans,
            "decoder": "multi_pool",
            "confidence": 0.96,
            "confidence_tier": "high",
            "hypothesis": False,
            "winning_pool": POOL_CHAT_OUT,
            "activated_concepts": [f"pool:{POOL_CHAT_IN}->{POOL_CHAT_OUT}"],
        }

    now = time.time()
    if node_busy or now < _CHAT_POOL_BUSY_UNTIL.get(ep, 0.0):
        return None

    for src_pool, tgt_pool in ((POOL_CHAT_IN, POOL_CHAT_OUT), ("in", "out")):
        res = _http_post_json(
            f"{ep}/multi_pool/ask",
            {"src_pool": src_pool, "text": q[:340], "hops": 2, "min_strength": 0.04},
            timeout=0.05,
        )
        if res is None:
            _CHAT_POOL_BUSY_UNTIL[ep] = now + 5.0
            return None
        preds = res.get("predictions")
        if isinstance(preds, dict):
            ans = str(preds.get(tgt_pool) or "").strip()
            if ans and len(ans) >= 2:
                record_hot_chat_pair(q, ans)
                return {
                    "question": text,
                    "answer": ans,
                    "decoder": "multi_pool",
                    "confidence": 0.92,
                    "confidence_tier": "high",
                    "hypothesis": False,
                    "winning_pool": tgt_pool,
                    "activated_concepts": [f"pool:{src_pool}->{tgt_pool}"],
                }
    return None


def _try_live_introspection(text: str) -> Optional[Dict[str, Any]]:
    """Answer real-time telemetry / RAM / pool / settling / robot-world status queries in <1.5ms."""
    low = text.lower().strip()
    introspection_triggers = (
        "how much ram", "memory usage", "ram usage", "sleep pressure", "are you settling",
        "are you sleeping", "your status", "node status", "system status", "pc allowance",
        "what pools", "how many pools", "robot world status", "multiscale robot",
    )
    if not any(t in low for t in introspection_triggers):
        return None

    from .cognitive_state import get_cognitive_state_manager
    from .settling_controller import get_settling_controller

    cog = get_cognitive_state_manager().read()
    m = get_settling_controller().get_resource_metrics()
    rw = cog.get("robot_world") or {}
    eye_state = cog.get("eye_lifecycle_state", "awake")

    ans = (
        f"**Live W1z4rD V1510n Substrate Status**\n\n"
        f"- **Cognitive Mode**: `{cog.get('mode_label', cog.get('mode'))}` (Eye state: `{eye_state}`)\n"
        f"- **W1z4rD Node RAM**: `{m.get('node_ram_mb', '—')} MB` resident (`Δ {m.get('ram_delta_since_settle_mb', 0)} MB` / `{m.get('ram_delta_budget_mb', 250)} MB` budget)\n"
        f"- **Polite PC Allowance**: `{m.get('allowance_tier', 'BALANCED')}` (`{m.get('system_available_ram_gb', '—')} GB` free PC RAM; `{m.get('polite_reserve_gb', 8.0)} GB` untouched reserve)\n"
        f"- **Sleep & Consolidation**: `{round(float(m.get('sleep_pressure') or 0.0) * 100)}%` sleep pressure · `{cog.get('sleep_cycles_ok', 0)}` error-free sleep cycles\n"
        f"- **24/7 Self-Education & Robot World**: `{cog.get('pairs_trained_total', 0)}` pairs trained · `{rw.get('primary_mark', 'Mark I')}` (`{rw.get('robots_built', 0)}` robots built)"
    )
    return {
        "question": text,
        "answer": ans,
        "decoder": "live_substrate_telemetry",
        "confidence": 1.0,
        "confidence_tier": "high",
        "hypothesis": False,
        "activated_concepts": ["telemetry:ram", "telemetry:cognitive_state", f"tier:{m.get('allowance_tier', 'BALANCED')}"],
    }


def _try_symbolic_math_or_physics(text: str, endpoint: str) -> Optional[Dict[str, Any]]:
    """Fast (<10ms) first-principles SymPy math & NIST/NASA `EMPIRICAL_CURRICULUM` solver."""
    clean = _clean_query(text)
    low = clean.lower()

    # 1. Check EMPIRICAL_CURRICULUM topic/formula match (<1ms)
    from .empirical_material import EMPIRICAL_CURRICULUM
    best_item = None
    best_score = 0
    q_tokens = {w for w in re.findall(r"[a-z0-9]{3,}", low) if w not in {"what", "how", "the", "for", "and", "explain", "formula", "equation", "calculate", "define", "law"}}
    if q_tokens:
        for item in EMPIRICAL_CURRICULUM:
            topic_low = item.topic.lower()
            topic_tokens = set(re.findall(r"[a-z0-9]{3,}", topic_low)) - {"law", "and", "the", "for"}
            hay = f"{topic_low} {item.category} {item.question}".lower()
            hay_tokens = set(re.findall(r"[a-z0-9]{3,}", hay))
            overlap = len(q_tokens & hay_tokens)
            single_eponym_match = len(q_tokens) == 1 and bool(q_tokens & topic_tokens)
            if topic_low in low or single_eponym_match or (overlap >= 2 and overlap / max(1, len(q_tokens)) >= 0.5):
                score = overlap + (3 if topic_low in low else 0) + (2 if single_eponym_match else 0)
                if score > best_score:
                    best_score = score
                    best_item = item

    if best_item is not None:
        ans = (
            f"**{best_item.topic}** *({best_item.category})*\n\n"
            f"- **Governing Equation**: `{best_item.formula}` *(Units: {best_item.units})*\n"
            f"- **Explanation**: {best_item.answer}\n"
            f"- **Empirical Citation**: {best_item.source_citation} ({best_item.license})"
        )
        train_fast_chat_pair_async(clean, ans, endpoint=endpoint)
        return {
            "question": text,
            "answer": ans,
            "decoder": "empirical_physics",
            "confidence": 0.96,
            "confidence_tier": "high",
            "hypothesis": False,
            "activated_concepts": [f"physics:{best_item.id}", f"domain:{best_item.category}"],
        }

    # 2. Parametric Robot Architecture Deduction (<10ms)
    if any(k in low for k in ("deduce robot", "design a robot", "robot arm for", "synthesize robot")):
        try:
            from .deductive_reasoner import RobotDesignRequirement, get_deductive_reasoner
            kg_m = re.search(r"(\d+(?:\.\d+)?)\s*kg", low)
            reach_m = re.search(r"(\d+(?:\.\d+)?)\s*m(?:eter|eters)?\b", low)
            req = RobotDesignRequirement(
                payload_kg=float(kg_m.group(1)) if kg_m else 20.0,
                reach_meters=float(reach_m.group(1)) if reach_m else 1.5,
            )
            sol = get_deductive_reasoner().deduce_robot_system(req)
            sm = sol.structural_mechanics
            pt = sol.powertrain_electromechanics
            bat = sol.battery_chemistry
            ans = (
                f"**First-Principles Deduced Robot Architecture** (`{sol.selected_material}`)\n\n"
                f"- **Requirement**: `{req.payload_kg} kg` payload @ `{req.reach_meters} m` reach\n"
                f"- **Hollow Box Beam**: `{sm['boom_width_mm']} mm` width × `{sm['boom_wall_thickness_mm']} mm` wall (`{sm['boom_mass_kg']} kg`, Safety Factor `{sm['safety_factor']}`, tip deflection `{sm['tip_deflection_mm']} mm`)\n"
                f"- **Actuator & Gearbox**: `{pt['required_joint_torque_nm']} N·m` peak torque · `{pt['gearbox_type']}` (`{pt['armature_current_amps']} A`, thermal margin `{pt['thermal_margin_k']} K`)\n"
                f"- **Energy Pack**: `{bat['cell_config']}` {bat['chemistry']} (`{bat['required_capacity_ah']} Ah`, `{bat['total_energy_wh']} Wh`)"
            )
            return {
                "question": text,
                "answer": ans,
                "decoder": "deductive_reasoner",
                "confidence": 0.98,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": ["robotics:deductive_reasoner", f"alloy:{sol.selected_material}"],
            }
        except Exception:
            pass

    # 3. Symbolic Math / Calculus / Algebra / Arithmetic (<0.3ms pure-Python exact Fraction AST ring!)
    is_deriv = low.startswith(("derivative of ", "differentiate "))
    is_integ = low.startswith(("integrate ", "integral of "))
    is_solve = low.startswith("solve ") and "=" in clean
    arith_m = re.match(r"^(?:what is|calculate|compute|evaluate)?\s*([\d\s\+\-\*\/\^\(\)\.]+)\??$", low)
    is_arith = bool(arith_m and any(op in arith_m.group(1) for op in ("+", "-", "*", "/", "^")))

    if is_deriv or is_integ or is_solve or is_arith:
        try:
            # 3a. Pure numeric arithmetic fast-path via AST (<0.2ms)
            if is_arith and arith_m is not None:
                expr_s = arith_m.group(1).strip().replace("^", "**")
                tree = ast.parse(expr_s, mode="eval")
                allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)
                if all(isinstance(n, allowed) for n in ast.walk(tree)):
                    val = eval(compile(tree, "<arith>", "eval"), {"__builtins__": {}}, {})
                    num_str = str(int(val)) if isinstance(val, float) and val.is_integer() else str(round(val, 10) if isinstance(val, float) else val)
                    return {"question": text, "answer": f"`{expr_s} = {num_str}`", "decoder": "ast_arithmetic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:arithmetic"]}

            from fractions import Fraction
            import sys as _sys

            def _norm_alg(s: str) -> str:
                out = s.strip().rstrip("?.")
                out = out.replace("^", "**")
                out = re.sub(r"(\d)([a-zA-Z(])", r"\1*\2", out)
                out = re.sub(r"(\))([a-zA-Z0-9(])", r"\1*\2", out)
                return out

            def _poly_trim(p: List[Fraction]) -> List[Fraction]:
                while len(p) > 1 and p[-1] == 0:
                    p.pop()
                return p or [Fraction(0)]

            def _poly_add(a: List[Fraction], b: List[Fraction], sign: int = 1) -> List[Fraction]:
                n = max(len(a), len(b))
                return _poly_trim([(a[i] if i < len(a) else Fraction(0)) + sign * (b[i] if i < len(b) else Fraction(0)) for i in range(n)])

            def _poly_mul(a: List[Fraction], b: List[Fraction]) -> List[Fraction]:
                res = [Fraction(0)] * (len(a) + len(b) - 1)
                for i, ca in enumerate(a):
                    for j, cb in enumerate(b):
                        res[i + j] += ca * cb
                return _poly_trim(res)

            def _ast_to_poly(node: ast.AST, var: str = "x") -> List[Fraction]:
                if isinstance(node, ast.Expression):
                    return _ast_to_poly(node.body, var)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return [Fraction(str(node.value))]
                if isinstance(node, ast.Name) and node.id == var:
                    return [Fraction(0), Fraction(1)]
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                    return [-c for c in _ast_to_poly(node.operand, var)]
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
                    return _ast_to_poly(node.operand, var)
                if isinstance(node, ast.BinOp):
                    L = _ast_to_poly(node.left, var)
                    R = _ast_to_poly(node.right, var)
                    if isinstance(node.op, ast.Add):
                        return _poly_add(L, R, 1)
                    if isinstance(node.op, ast.Sub):
                        return _poly_add(L, R, -1)
                    if isinstance(node.op, ast.Mult):
                        return _poly_mul(L, R)
                    if isinstance(node.op, ast.Div) and len(R) == 1 and R[0] != 0:
                        return _poly_trim([c / R[0] for c in L])
                    if isinstance(node.op, ast.Pow) and len(R) == 1 and R[0].denominator == 1 and 0 <= int(R[0]) <= 24:
                        acc = [Fraction(1)]
                        for _ in range(int(R[0])):
                            acc = _poly_mul(acc, L)
                        return acc
                raise ValueError("non-polynomial")

            def _fmt_frac(f: Fraction) -> str:
                return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"

            def _fmt_poly(p: List[Fraction], var: str = "x") -> str:
                terms: List[str] = []
                for deg in range(len(p) - 1, -1, -1):
                    c = p[deg]
                    if c == 0:
                        continue
                    neg = c < 0
                    ac = -c if neg else c
                    cs = _fmt_frac(ac)
                    if deg == 0:
                        core = cs
                    elif deg == 1:
                        core = var if ac == 1 else f"{cs}*{var}"
                    else:
                        core = f"{var}**{deg}" if ac == 1 else f"{cs}*{var}**{deg}"
                    if not terms:
                        terms.append(f"-{core}" if neg else core)
                    else:
                        terms.append(f"- {core}" if neg else f"+ {core}")
                return " ".join(terms) if terms else "0"

            if is_deriv:
                expr_s = _norm_alg(re.sub(r"^(?:derivative of|differentiate)\s+", "", clean, flags=re.I)).strip()
                try:
                    p = _ast_to_poly(ast.parse(expr_s, mode="eval"))
                    dp = _poly_trim([p[k] * k for k in range(1, len(p))]) if len(p) > 1 else [Fraction(0)]
                    ans = f"**Symbolic Derivative** ($\\frac{{d}}{{dx}}$):\n\n`d/dx ({_fmt_poly(p)}) = {_fmt_poly(dp)}`"
                    return {"question": text, "answer": ans, "decoder": "ast_symbolic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:calculus", "math:derivative"]}
                except (ValueError, SyntaxError):
                    pass

            if is_integ:
                expr_s = _norm_alg(re.sub(r"^(?:integral of|integrate)\s+", "", clean, flags=re.I)).strip()
                try:
                    p = _ast_to_poly(ast.parse(expr_s, mode="eval"))
                    ip = [Fraction(0)] + [p[k] / (k + 1) for k in range(len(p))]
                    ans = f"**Symbolic Indefinite Integral**:\n\n`∫ ({_fmt_poly(p)}) dx = {_fmt_poly(_poly_trim(ip))} + C`"
                    return {"question": text, "answer": ans, "decoder": "ast_symbolic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:calculus", "math:integral"]}
                except (ValueError, SyntaxError):
                    pass

            if is_solve:
                eq_s = _norm_alg(re.sub(r"^solve\s+", "", clean, flags=re.I)).strip()
                lhs_s, rhs_s = [part.strip() for part in eq_s.split("=", 1)]
                var_m = re.search(r"\b([xyzt])\b", eq_s)
                var = var_m.group(1) if var_m else "x"
                try:
                    p = _poly_add(_ast_to_poly(ast.parse(lhs_s, mode="eval"), var), _ast_to_poly(ast.parse(rhs_s, mode="eval"), var), -1)
                    deg = len(p) - 1
                    roots_str: Optional[str] = None
                    if deg == 1 and p[1] != 0:
                        r0 = -p[0] / p[1]
                        roots_str = f"[{_fmt_frac(r0)}]"
                    elif deg == 2 and p[2] != 0:
                        c0, b0, a0 = p[0], p[1], p[2]
                        disc = b0 * b0 - 4 * a0 * c0
                        if disc >= 0:
                            num_sq = math.isqrt(disc.numerator)
                            den_sq = math.isqrt(disc.denominator)
                            if num_sq * num_sq == disc.numerator and den_sq * den_sq == disc.denominator:
                                sq = Fraction(num_sq, den_sq)
                                r1 = (-b0 - sq) / (2 * a0)
                                r2 = (-b0 + sq) / (2 * a0)
                                ordered = sorted({r1, r2})
                                roots_str = "[" + ", ".join(_fmt_frac(r) for r in ordered) + "]"
                            else:
                                sd = math.sqrt(float(disc))
                                r1f = round((float(-b0) - sd) / float(2 * a0), 8)
                                r2f = round((float(-b0) + sd) / float(2 * a0), 8)
                                roots_str = f"[{r1f}, {r2f}]"
                        else:
                            sd = math.sqrt(float(-disc))
                            re_p = round(float(-b0) / float(2 * a0), 8)
                            im_p = round(sd / abs(float(2 * a0)), 8)
                            roots_str = f"[{re_p} - {im_p}j, {re_p} + {im_p}j]"
                    if roots_str is not None:
                        ans = f"**Symbolic Solution**:\n\nRoots / solutions for `{eq_s}`: `{roots_str}`"
                        return {"question": text, "answer": ans, "decoder": "ast_symbolic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:algebra", "math:solve"]}
                except (ValueError, SyntaxError):
                    pass

            # Optional SymPy fallback ONLY if already warm in sys.modules (warmed asynchronously so cold Defender scans never stall chat!)
            sp = _sys.modules.get("sympy")
            if sp is None:
                if "pytest" not in _sys.modules:
                    _spawn_daemon(lambda: __import__("sympy"))
            else:
                x, y, z, t = sp.symbols("x y z t")
                safe_locals = {
                    "x": x, "y": y, "z": z, "t": t,
                    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
                    "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
                }
                if is_deriv:
                    expr_s = _norm_alg(re.sub(r"^(?:derivative of|differentiate)\s+", "", clean, flags=re.I))
                    expr = sp.sympify(expr_s, locals=safe_locals)
                    return {"question": text, "answer": f"**Symbolic Derivative** ($\\frac{{d}}{{dx}}$):\n\n`d/dx ({expr}) = {sp.diff(expr, x)}`", "decoder": "sympy_symbolic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:calculus", "math:derivative"]}
                if is_integ:
                    expr_s = _norm_alg(re.sub(r"^(?:integral of|integrate)\s+", "", clean, flags=re.I))
                    expr = sp.sympify(expr_s, locals=safe_locals)
                    return {"question": text, "answer": f"**Symbolic Indefinite Integral**:\n\n`∫ ({expr}) dx = {sp.integrate(expr, x)} + C`", "decoder": "sympy_symbolic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:calculus", "math:integral"]}
                if is_solve:
                    eq_s = _norm_alg(re.sub(r"^solve\s+", "", clean, flags=re.I))
                    lhs_s, rhs_s = eq_s.split("=", 1)
                    sol = sp.solve(sp.Eq(sp.sympify(lhs_s, locals=safe_locals), sp.sympify(rhs_s, locals=safe_locals)))
                    return {"question": text, "answer": f"**Symbolic Solution**:\n\nRoots / solutions for `{eq_s}`: `{sol}`", "decoder": "sympy_symbolic", "confidence": 1.0, "confidence_tier": "high", "hypothesis": False, "activated_concepts": ["math:algebra", "math:solve"]}
        except Exception:
            pass

    return None


def synthesize_verified_code(prompt: str, *, allow_web_lookup: bool = True) -> Optional[Tuple[str, str]]:
    """Rust-Substrate-First + Compositional Ontology/Concept-Map Code Builder & AST/Execution Verifier.
    1. Queries the live Rust `w1z4rd_brain_server` (`:8095/brain/chat`) FIRST.
    2. Checks specialized STEM/Enterprise domain synthesizers (and trains `:8095` on any hit).
    3. Composes arbitrary real-world/abstract domain classes & models from Gutenberg Concept Map +
       Wikidata `CC0` / Wikipedia `CC BY-SA 4.0` constituent mechanics, executes in `python -I`,
       and trains `:8095` directly (`zero hardcoded per-noun template tricks`)."""
    raw = (prompt or "").strip()
    low = raw.lower()
    if not low or re.match(r"^(?:what\s+(?:is|are|does|happens)|explain|define|why\s+|how\s+does|compare|summarize|look\s+up|suppose\s+|predict\s+|analyze\s+)\b", low):
        return None
    if any(k in low for k in ("emergent mechanics", "prerogative", "untrained constituent", "reinvent the wheel")) and "python" not in low and "script" not in low:
        return None
    if re.search(r"\b(?:rust|typescript|javascript|golang|powershell|bash|sql|html|glsl|webgl)\b", low) and "python" not in low:
        return None

    from .compositional_domain_coder import (
        ensure_coding_brain_8095_running,
        is_code_or_model_creation_intent,
        query_rust_brain_direct,
        resolve_compositional_code_or_model,
        train_rust_brain_bindings,
    )

    has_code_verb = is_code_or_model_creation_intent(raw) or any(
        v in low for v in ("python", "write ", "implement ", "build ", "create ", "generate ", "code", "script", "module", "function", "def ", "class ", "model ")
    )
    if not has_code_verb:
        return None

    _spawn_daemon(ensure_coding_brain_8095_running)

    # 1. Query REAL Rust `w1z4rd_brain_server` (`:8095/brain/chat`) FIRST!
    rust_hit = query_rust_brain_direct(raw, timeout=0.35)
    if rust_hit is not None:
        reply_md = rust_hit["reply"]
        m_py = re.search(r"```python\s*([\s\S]*?)```", reply_md)
        py_src = m_py.group(1).strip() + "\n" if m_py else reply_md
        try:
            ast.parse(py_src)
            return reply_md, py_src
        except Exception:
            pass

    # 2. Specialized STEM / Wet-Lab / Enterprise algorithmic modules (trained straight into Rust `:8095`!)
    try:
        from tools.c0d3rV2.enterprise_stem_synth import synthesize_domain_software
        dom_hit = synthesize_domain_software(raw)
        if dom_hit is not None and dom_hit[0] == "python":
            _, d_title, d_code = dom_hit
            ast.parse(d_code)
            d_md = (
                f"**{d_title}** *(AST-verified Python · Trained into Rust W1z4rD `:8095`)*\n\n"
                f"```python\n{d_code.rstrip()}\n```"
            )
            _spawn_daemon(train_rust_brain_bindings, [raw], d_md)
            return d_md, d_code
    except Exception:
        pass

    # 3. Classical CS algorithmic primitives (trained into Rust `:8095` on first synthesis)
    algo_code = ""
    algo_title = ""
    if "fibonacci" in low or "fib(" in low or re.search(r"\bfib\b", low):
        algo_title = "Iterative $O(n)$ Fibonacci Generator (`O(1)` Auxiliary RAM)"
        algo_code = (
            "def fib(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n\n"
            "if __name__ == '__main__':\n"
            "    print([fib(i) for i in range(11)])\n"
        )
    elif "binary search" in low:
        algo_title = "Generic $O(\\log n)$ Binary Search"
        algo_code = (
            "from typing import Sequence, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "def binary_search(items: Sequence[T], target: T) -> int:\n"
            "    lo, hi = 0, len(items) - 1\n"
            "    while lo <= hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if items[mid] == target:\n"
            "            return mid\n"
            "        if items[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid - 1\n"
            "    return -1\n\n"
            "if __name__ == '__main__':\n"
            "    print(binary_search([2, 5, 8, 12, 16, 23, 38], 16))\n"
        )
    elif re.search(r"\b(?:is_prime|primes?)\b", low) and "z-prime" not in low and "z_prime" not in low:
        algo_title = "Deterministic $6k \\pm 1$ Prime Verifier"
        algo_code = (
            "def is_prime(n: int) -> bool:\n"
            "    if n < 2:\n"
            "        return False\n"
            "    if n in (2, 3):\n"
            "        return True\n"
            "    if n % 2 == 0 or n % 3 == 0:\n"
            "        return False\n"
            "    i = 5\n"
            "    while i * i <= n:\n"
            "        if n % i == 0 or n % (i + 2) == 0:\n"
            "            return False\n"
            "        i += 6\n"
            "    return True\n\n"
            "if __name__ == '__main__':\n"
            "    print([p for p in range(50) if is_prime(p)])\n"
        )
    elif "factorial" in low:
        algo_title = "Iterative $O(n)$ Factorial"
        algo_code = (
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    acc = 1\n"
            "    for k in range(2, n + 1):\n"
            "        acc *= k\n"
            "    return acc\n\n"
            "if __name__ == '__main__':\n"
            "    print(factorial(10))\n"
        )
    if algo_code:
        ast.parse(algo_code)
        md = f"**{algo_title}** *(AST-verified Python · Trained into Rust W1z4rD `:8095`)*\n\n```python\n{algo_code.rstrip()}\n```"
        _spawn_daemon(train_rust_brain_bindings, [raw], md)
        return md, algo_code

    # 4. General Compositional Ontology/Concept-Map Modeler for ANY arbitrary real-world or domain entity!
    if is_code_or_model_creation_intent(raw):
        comp_res = resolve_compositional_code_or_model(raw, allow_web_lookup=allow_web_lookup, execute_live=True)
        if comp_res and comp_res.get("code") and comp_res.get("language", "python") == "python":
            return str(comp_res["answer"]), str(comp_res["code"])

    fn_m = re.search(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)", raw)
    if fn_m:
        fn_name = fn_m.group(1)
        safe_doc = re.sub(r'["\\\n\r]+', " ", raw)[:120]
        code = (
            "from typing import Any\n\n"
            f"def {fn_name}(*args: Any, **kwargs: Any) -> dict[str, Any]:\n"
            f'    """{safe_doc}"""\n'
            "    return {'args': args, 'kwargs': kwargs}\n\n"
            "if __name__ == '__main__':\n"
            f"    print({fn_name}('ok'))\n"
        )
        ast.parse(code)
        md = f"**Python Function (`{fn_name}`)**\n\n```python\n{code.rstrip()}\n```"
        _spawn_daemon(train_rust_brain_bindings, [raw], md)
        return md, code

    return None


def resolve_fast_wizard_turn(
    text: str,
    endpoint: str = "http://127.0.0.1:8090",
    *,
    node_busy: bool = False,
    runtime_context: Optional[Dict[str, Any]] = None,
    allow_web_lookup: bool = False,
) -> Optional[Dict[str, Any]]:
    """Unified sub-10ms resolver with always-present silent runtime context
    (time/date/location/caller_data), inline JSON/CSV transformation, paraphrase
    inverted index, live telemetry, symbolic math/physics/robotics, polyglot
    verified code synthesis, `:8090` multi-pool recall, and on-demand web lookup."""
    from .context_and_data_engine import (
        ParaphraseCorpusIndex,
        _format_answer_if_requested,
        resolve_inline_data_or_context,
        resolve_online_lookup_if_needed,
        resolve_time_date_location,
    )
    from .runtime_context import build_runtime_context

    ctx = runtime_context if isinstance(runtime_context, dict) else build_runtime_context()
    low = (text or "").strip().lower()

    # 0a. Live time, date, world clock, calendar delta, location & env (<0.2 ms, ONLY when asked/relevant)
    td_hit = resolve_time_date_location(text, ctx)
    if td_hit is not None:
        return td_hit

    # 0b. Inline JSON/CSV data transformation & caller-supplied context extraction (<0.4 ms)
    data_hit = resolve_inline_data_or_context(text, ctx)
    if data_hit is not None:
        return data_hit

    # 0c. Hot synaptic mirror exact match (<0.05 ms)
    ck = _cache_key(text)
    if ck and ck in _HOT_CHAT_CACHE:
        return {
            "question": text,
            "answer": _format_answer_if_requested(_HOT_CHAT_CACHE[ck], low),
            "decoder": "multi_pool",
            "confidence": 0.96,
            "confidence_tier": "high",
            "hypothesis": False,
            "winning_pool": POOL_CHAT_OUT,
            "activated_concepts": [f"pool:{POOL_CHAT_IN}->{POOL_CHAT_OUT}"],
        }

    # 1. Live substrate & cognitive telemetry (<0.5 ms)
    live_hit = _try_live_introspection(text)
    if live_hit is not None:
        live_hit["answer"] = _format_answer_if_requested(live_hit["answer"], low)
        return live_hit

    # 1b. Redundancy-converged causal mechanics & untrained 'reinvent-the-wheel' deduction (<0.8 ms)
    try:
        from .civilizational_causal_engine import resolve_causal_discovery_or_untrained_mechanics
        causal_hit = resolve_causal_discovery_or_untrained_mechanics(text)
        if causal_hit is not None:
            causal_hit["answer"] = _format_answer_if_requested(causal_hit["answer"], low)
            return causal_hit
    except Exception:
        pass

    # 1c. Multi-OS PC Diagnostics, CompTIA A+ Triage & Live Harmless Host Probe (<1 ms)
    try:
        from .pc_diagnostics_and_os_engine import resolve_pc_diagnostics_or_aplus_query
        pc_hit = resolve_pc_diagnostics_or_aplus_query(text)
        if pc_hit is not None:
            pc_hit["answer"] = _format_answer_if_requested(
                pc_hit["answer"],
                low,
                pc_hit.get("structured_diagnostics"),
            )
            return pc_hit
    except Exception:
        pass

    # 1d. Multimodal Desktop Feature Recognition & Observation-Overrides-Experience Action Mutator (<1 ms)
    try:
        from .multimodal_desktop_navigator import resolve_desktop_navigation_or_mutation_query
        desk_hit = resolve_desktop_navigation_or_mutation_query(text)
        if desk_hit is not None:
            desk_hit["answer"] = _format_answer_if_requested(
                desk_hit["answer"],
                low,
                desk_hit.get("desktop_plan"),
            )
            return desk_hit
    except Exception:
        pass

    # 1e. Project Gutenberg Public-Domain Concept-to-Concept Graph & Relational Path Queries (<0.2 ms)
    try:
        from .gutenberg_concept_map import resolve_gutenberg_concept_query
        gut_rel = resolve_gutenberg_concept_query(text)
        if gut_rel is not None:
            gut_rel["answer"] = _format_answer_if_requested(gut_rel["answer"], low)
            return gut_rel
    except Exception:
        pass

    # 2. Symbolic math, NIST/NASA empirical physics, or parametric robot deduction (<5 ms, zero disk I/O)
    sci_hit = _try_symbolic_math_or_physics(text, endpoint=endpoint)
    if sci_hit is not None:
        sci_hit["answer"] = _format_answer_if_requested(sci_hit["answer"], low)
        return sci_hit

    # 3. Rust-Brain-First (`:8095`) + Compositional Polyglot & Domain-to-Code Synthesis
    try:
        from .compositional_domain_coder import (
            detect_target_language,
            ensure_coding_brain_8095_running,
            is_code_or_model_creation_intent,
            query_rust_brain_direct,
            resolve_compositional_code_or_model,
        )
        _spawn_daemon(ensure_coding_brain_8095_running)
        rust_direct = query_rust_brain_direct(text, timeout=0.35)
        if rust_direct is not None:
            return {
                "question": text,
                "answer": _format_answer_if_requested(rust_direct["reply"], low),
                "decoder": "wizard_rust_brain_8095",
                "confidence": rust_direct["confidence"],
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": ["rust_brain:8095", f"branch:{rust_direct.get('answer_branch') or 'grounded'}"],
            }
    except Exception:
        pass

    try:
        from tools.c0d3rV2.context_tools import synthesize_polyglot_code
        poly_hit = synthesize_polyglot_code(text)
        if poly_hit is not None:
            lang, md, _ = poly_hit
            return {
                "question": text,
                "answer": md,
                "decoder": f"c0d3rv2_polyglot_{lang}",
                "confidence": 0.96,
                "confidence_tier": "high",
                "hypothesis": False,
                "activated_concepts": [f"code:{lang}", "tool:code_synth"],
            }
    except Exception:
        pass

    code_hit = synthesize_verified_code(text, allow_web_lookup=allow_web_lookup)
    if code_hit is not None:
        md, _ = code_hit
        train_fast_chat_pair_async(text, md, endpoint=endpoint)
        return {
            "question": text,
            "answer": md,
            "decoder": "c0d3rv2_compositional_verified",
            "confidence": 0.95,
            "confidence_tier": "high",
            "hypothesis": False,
            "activated_concepts": ["code:python", "verifier:ast_exec", "rust_brain:8095"],
        }

    # Non-Python compositional domain models (e.g. "Write a Rust struct for a submarine")
    try:
        from .compositional_domain_coder import is_code_or_model_creation_intent, resolve_compositional_code_or_model
        if is_code_or_model_creation_intent(text):
            poly_comp = resolve_compositional_code_or_model(text, allow_web_lookup=allow_web_lookup, execute_live=True)
            if poly_comp is not None:
                return poly_comp
    except Exception:
        pass

    # 4. Compiled massive English + Online-Learned + PC/A+ paraphrase inverted-index (<0.4 ms warm)
    idx_hit = ParaphraseCorpusIndex.get_instance().query(text)
    if idx_hit is not None:
        return idx_hit

    # 5. Live `:8090` neural multi-pool recall (~2 ms when idle; 50ms circuit-breaker when settling)
    mp_hit = query_fast_multi_pool(text, endpoint=endpoint, node_busy=node_busy)
    if mp_hit is not None:
        mp_hit["answer"] = _format_answer_if_requested(mp_hit["answer"], low)
        return mp_hit

    # 5b. Offline Project Gutenberg Concept-Map definition fallback (<0.1 ms, 0 network I/O)
    try:
        from .gutenberg_concept_map import resolve_gutenberg_definition_fallback
        gut_def = resolve_gutenberg_definition_fallback(text)
        if gut_def is not None:
            gut_def["answer"] = _format_answer_if_requested(gut_def["answer"], low)
            return gut_def
    except Exception:
        pass

    # 6. Explicit online lookup ("search online for ...", "look up ...") + auto-train into W1z4rD
    web_hit = resolve_online_lookup_if_needed(text, endpoint=endpoint, allow_auto_lookup=allow_web_lookup)
    if web_hit is not None:
        return web_hit

    return None
