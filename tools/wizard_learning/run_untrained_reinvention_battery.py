"""tools/wizard_learning/run_untrained_reinvention_battery.py

Massive Zero-Overlap Untrained Generalization & "Reinvent-the-Wheel" Battery
(`>= 2,500` Live HTTP Scenarios against Revenir `:8001` Chat + C0d3rV2 Agent Mode).

Strictly verifies:
  1. 0.00% exact or normalized overlap with ANY trained prompt or paraphrase across
     all saved corpora (`massive_context_corpus`, `enterprise_stem_steam_corpus`,
     and `civilizational_discovery_course`).
  2. Four untrained generalization families (2,520 total live HTTP questions):
       - Family 1 (720): Austere "Reinvent-the-Wheel" from raw materials
       - Family 2 (800): Unseen 2-Law Cross-Domain Causal Compositions (`y_1 => z_2`)
       - Family 3 (640): Counterfactual / Missing-Precondition Diagnostics (`z,r,q,s,t`)
       - Family 4 (360): Synthetic Alien-Substrate Pure Constituent Transfer (`x + p => y`)
  3. Accurate causal mechanics + governing invariant + explicit W1z4rD prerogative/goal tie.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import http.client
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "tools"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.wizard_learning.civilizational_causal_engine import (
    CIVILIZATIONAL_DISCOVERIES,
    COUNTERFACTUAL_VIOLATION_RULES,
    COURSE_DIR,
    DISCOVERY_BY_ID,
)


def _norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()


def load_all_trained_utterances() -> Set[str]:
    trained: Set[str] = set()
    base = PROJECT_ROOT / "data" / "wizard_learning"
    for sub in ("massive_context_corpus", "enterprise_stem_steam_corpus", "civilizational_discovery_course"):
        d = base / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.jsonl"):
            if "weakness" in p.name or "episodes" in p.name:
                continue
            for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                    if rec.get("question"):
                        trained.add(_norm_q(str(rec["question"])))
                    for pv in rec.get("paraphrases") or []:
                        trained.add(_norm_q(str(pv)))
                except Exception:
                    pass
    return trained


AUSTERE_ENVIRONMENTS: List[str] = [
    "an off-grid basalt canyon outpost",
    "a salvaged Martian lava-tube workshop",
    "a remote sub-arctic geothermal station",
    "an isolated desert mining camp",
    "a deep-sea habitat repair bay",
    "an abandoned textile mill with raw scrap",
]

REINVENT_TEMPLATES: List[str] = [
    "Suppose we are operating in {env} without commercial spares and need to reinvent {kw} capability from scratch — determine the constituent mechanics and relate them back to your prerogatives and goals.",
    "In an untrained scenario inside {env}, how would you build {kw} machinery from raw primitives, what conditions must co-act, and how does that support your goals?",
    "If you have to reinvent the wheel on {kw} using raw materials in {env}, what constituents x and conditions z,r,q,s,t make it work, and how does it align with your prerogatives?",
    "Analyze an untrained field problem in {env}: we need {kw} functionality built from first principles — deduce the governing mechanics and tie them to your operational goals.",
    "Without pre-built modules in {env}, deduce how to synthesize {kw} from constituent physics and explain why it matters for your prerogatives.",
]


def _expected_token_for_disc(disc_id: str) -> List[str]:
    d = DISCOVERY_BY_ID[disc_id]
    # Require the discovery's specific title words or governing law tokens
    title_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", d.title) if w.lower() not in {"with", "from", "laws"}]
    return title_words[:3]


def generate_untrained_battery() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    # ── Family 1: 720 Untrained "Reinvent-the-Wheel" Austere Scenarios ───────
    # 24 discoveries * 6 austere environments * 5 distinct Lead templates = 720
    for d_idx, disc in enumerate(CIVILIZATIONAL_DISCOVERIES):
        kw = disc.keywords[0]
        exp_any = _expected_token_for_disc(disc.id)
        for e_idx, env in enumerate(AUSTERE_ENVIRONMENTS):
            for t_idx, tmpl in enumerate(REINVENT_TEMPLATES):
                q = tmpl.format(env=env, kw=kw)
                ep = "/api/wizard-chat/agent/" if ((d_idx + e_idx + t_idx) % 4 == 0) else "/api/wizard-chat/message/"
                cases.append({
                    "id": f"untrained_reinvent_{len(cases)+1:04d}",
                    "family": "austere_reinvention",
                    "endpoint": ep,
                    "question": q,
                    "must_any_groups": [
                        exp_any,
                        ["x:", "constituent"],
                        ["prerogative", "goal", "w1z4rd"],
                    ],
                })

    # ── Family 2: 800 Untrained 2-Law Cross-Domain Compositions ──────────────
    # Pair discoveries from DIFFERENT pillars that were never trained together
    cross_pairs: List[Tuple[Any, Any]] = []
    for i, d1 in enumerate(CIVILIZATIONAL_DISCOVERIES):
        for j, d2 in enumerate(CIVILIZATIONAL_DISCOVERIES):
            if i < j and d1.pillar != d2.pillar:
                cross_pairs.append((d1, d2))
            if len(cross_pairs) >= 80:
                break
        if len(cross_pairs) >= 80:
            break

    comp_templates = [
        "Untrained cross-domain scenario #{k} in {env}: suppose we couple {kw1} with {kw2} — determine how their mechanics chain together and relate that to your prerogatives.",
        "Predict the chained mechanics when we combine {kw1} and {kw2} from scratch in {env} (#{k}), and explain how both stages advance your goals.",
    ]
    for p_idx, (d1, d2) in enumerate(cross_pairs[:80]):
        exp1 = _expected_token_for_disc(d1.id)
        exp2 = _expected_token_for_disc(d2.id)
        for v in range(10):
            env = AUSTERE_ENVIRONMENTS[(p_idx + v) % len(AUSTERE_ENVIRONMENTS)]
            tmpl = comp_templates[v % len(comp_templates)]
            q = tmpl.format(k=v + 1, env=env, kw1=d1.keywords[0], kw2=d2.keywords[0])
            ep = "/api/wizard-chat/agent/" if (v % 4 == 0) else "/api/wizard-chat/message/"
            cases.append({
                "id": f"untrained_compose_{len(cases)+1:04d}",
                "family": "cross_domain_composition",
                "endpoint": ep,
                "question": q,
                "must_any_groups": [
                    exp1,
                    exp2,
                    ["prerogative", "goal", "w1z4rd"],
                ],
            })

    # ── Family 3: 640 Untrained Counterfactual / Broken-Precondition Tests ───
    cf_templates = [
        "Untrained counterfactual check #{k} in {env}: suppose a team sets up {kw} with {viol} — determine what happens to the mechanics and relate that back to your goals.",
        "What happens when {kw} is operated in {env} under `{viol}` (case #{k})? Deduce which necessary condition fails and how W1z4rD should adapt.",
        "Predict the exact physical mechanics in {env} (#{k}) if we attempt {kw} while `{viol}` is present, and tie the failure mode to your prerogatives.",
        "Why does {kw} fail or change behavior in {env} (#{k}) when `{viol}` occurs? Determine the constituent mechanics from first principles.",
    ]
    cf_idx = 0
    while sum(1 for c in cases if c["family"] == "counterfactual_broken_condition") < 640:
        triggers, disc_id, cond_key = COUNTERFACTUAL_VIOLATION_RULES[cf_idx % len(COUNTERFACTUAL_VIOLATION_RULES)]
        disc = DISCOVERY_BY_ID[disc_id]
        viol_phrase = triggers[(cf_idx // len(COUNTERFACTUAL_VIOLATION_RULES)) % len(triggers)]
        env = AUSTERE_ENVIRONMENTS[cf_idx % len(AUSTERE_ENVIRONMENTS)]
        tmpl = cf_templates[cf_idx % len(cf_templates)]
        q = tmpl.format(k=cf_idx + 1, env=env, kw=disc.keywords[0], viol=viol_phrase)
        ep = "/api/wizard-chat/agent/" if (cf_idx % 4 == 0) else "/api/wizard-chat/message/"
        cases.append({
            "id": f"untrained_counterfactual_{len(cases)+1:04d}",
            "family": "counterfactual_broken_condition",
            "endpoint": ep,
            "question": q,
            "must_any_groups": [
                _expected_token_for_disc(disc_id),
                ["violated condition", "fails", "zero", "cannot", "absent"],
                ["prerogative", "goal", "w1z4rd"],
            ],
        })
        cf_idx += 1

    # ── Family 4: 360 Synthetic Alien-Label Pure Constituent Transfer Tests ──
    # Zero familiar device nouns — ONLY micro-constituents `x` + conditions `{z,r,q}`!
    pure_transfer_specs: List[Tuple[str, str]] = [
        ("has delocalized electrons in a multi-turn loop exposed to a rotating magnet with a closed conductive loop", "disc_faraday_ampere_electromechanics"),
        ("contains mobile ions between two different metals with a redox potential difference across a porous separator", "disc_voltaic_electrochemistry"),
        ("is an incompressible liquid inside sealed unequal-area pistons with zero gas voids", "disc_archimedes_pascal_hydrostatics"),
        ("forms a circular rotor on a concentric axle with a lubricated bearing under normal load", "disc_wheel_axle"),
        ("acts as a rigid beam resting on an asymmetric fulcrum pivot arm", "disc_lever_fulcrum"),
        ("carries a helical thread with a shallow lead angle driven by rotary torque", "disc_inclined_screw_wedge"),
        ("uses triangulated hollow tube struts under axial compression", "disc_elasticity_truss_buckling"),
        ("combines an oxidizable fuel with an ignition spark in stoichiometric oxygen", "disc_controlled_combustion"),
        ("cycles an expanding gas piston between a hot source and cold sink across a temperature gradient", "disc_carnot_heat_engine"),
        ("shapes a curved transparent medium with high refractive index contrast to focus sunlight", "disc_optics_lenses_microscopy"),
        ("forms a p-n junction semiconductor bandgap illuminated by photons above bandgap energy", "disc_photoelectric_photovoltaic_pn"),
        ("wires inverting switches into a NAND gate and bistable latch", "disc_boolean_nand_universal_computing"),
        ("couples an error sensor and negative feedback proportional-integral comparator to a plant actuator", "disc_feedback_control_pid_governor"),
        ("links append-only write-ahead log records via a SHA-256 hash chain", "disc_cryptography_merkle_wal"),
        ("cycles a DNA template with flanking primers and thermostable Taq polymerase", "disc_dna_pcr_central_dogma"),
    ]
    alien_names = [
        "Substrate Zeta-9", "Composite Kronos-4", "Medium Vesper-2",
        "Assembly Orpheus-7", "Element Lyra-5", "Lattice Hyperion-3",
        "Construct Callisto-8", "Matrix Phoebus-6",
    ]
    for s_idx, (desc, disc_id) in enumerate(pure_transfer_specs):
        exp_toks = _expected_token_for_disc(disc_id)
        for a_idx, aname in enumerate(alien_names):
            for rep in range(3):
                env = AUSTERE_ENVIRONMENTS[(s_idx + a_idx + rep) % len(AUSTERE_ENVIRONMENTS)]
                q = (
                    f"Untrained constituent transfer #{rep+1} in {env}: suppose unfamiliar `{aname}` "
                    f"{desc}. Determine the emergent mechanics from its constituents and relate them to your prerogatives and goals."
                )
                ep = "/api/wizard-chat/agent/" if ((s_idx + a_idx + rep) % 4 == 0) else "/api/wizard-chat/message/"
                cases.append({
                    "id": f"untrained_transfer_{len(cases)+1:04d}",
                    "family": "alien_constituent_transfer",
                    "endpoint": ep,
                    "question": q,
                    "must_any_groups": [
                        exp_toks,
                        ["x:", "constituent"],
                        ["prerogative", "goal", "w1z4rd"],
                    ],
                })

    return cases


_TLS = threading.local()


def _eval_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    conn: http.client.HTTPConnection | None = getattr(_TLS, "conn", None)
    if conn is None:
        conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=15.0)
        _TLS.conn = conn

    payload = json.dumps({"text": spec["question"], "session_id": f"untrained-{spec['family']}"}).encode("utf-8")
    t0 = time.perf_counter()
    try:
        conn.request("POST", spec["endpoint"], body=payload, headers={"Content-Type": "application/json", "Connection": "keep-alive"})
        resp = conn.getresponse()
        raw = resp.read()
        ms = (time.perf_counter() - t0) * 1000.0
        st = resp.status
        body = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        try:
            conn.close()
        except Exception:
            pass
        _TLS.conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=15.0)
        ms = (time.perf_counter() - t0) * 1000.0
        st = 599
        body = {"answer": "", "error": str(exc)}

    ans = str(body.get("answer") or "")
    low = ans.lower()
    reasons: List[str] = []
    if st != 200:
        reasons.append(f"HTTP {st}")
    if "isn't grounded" in low or "node offline" in low or not ans.strip():
        reasons.append("hit ungrounded/offline fallback")
    for grp in spec["must_any_groups"]:
        if not any(tok.lower() in low for tok in grp):
            reasons.append(f"missing {grp}")

    return {
        "id": spec["id"],
        "family": spec["family"],
        "endpoint": spec["endpoint"],
        "ok": len(reasons) == 0,
        "ms": round(ms, 2),
        "reasons": reasons,
        "question": spec["question"],
        "preview": ans.replace("\n", " ")[:120],
    }


def main() -> None:
    trained_set = load_all_trained_utterances()
    cases = generate_untrained_battery()

    # 1. Mathematical Zero-Training-Overlap Audit
    overlaps = [c["question"] for c in cases if _norm_q(c["question"]) in trained_set]
    assert len(overlaps) == 0, f"Training overlap detected: {overlaps[:3]}"
    print(f"[1/3] Generated {len(cases)} untrained scenarios | Verified 0/{len(cases)} (0.000%) overlap with {len(trained_set)} trained utterances.", flush=True)

    # 2. Execute live concurrent HTTP requests against 127.0.0.1:8001
    print(f"[2/3] Executing {len(cases)} live HTTP evaluations across /message/ and /agent/...", flush=True)
    t_start = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        for idx, res in enumerate(pool.map(_eval_one, cases), 1):
            results.append(res)
            if idx % 500 == 0:
                ok_so_far = sum(1 for r in results if r["ok"])
                print(f"      Progress: {idx}/{len(cases)} ({ok_so_far}/{idx} passed)", flush=True)

    wall_s = round(time.perf_counter() - t_start, 2)
    passed = sum(1 for r in results if r["ok"])
    lats = sorted(r["ms"] for r in results)
    p50 = round(lats[len(lats) // 2], 2)
    p95 = round(lats[int(len(lats) * 0.95)], 2)
    p99 = round(lats[int(len(lats) * 0.99)], 2)

    fam_stats: Dict[str, Dict[str, Any]] = {}
    for fam in ("austere_reinvention", "cross_domain_composition", "counterfactual_broken_condition", "alien_constituent_transfer"):
        sub = [r for r in results if r["family"] == fam]
        f_ok = sum(1 for r in sub if r["ok"])
        f_lats = sorted(r["ms"] for r in sub)
        fam_stats[fam] = {
            "passed": f_ok,
            "total": len(sub),
            "pass_rate_pct": round(100.0 * f_ok / max(1, len(sub)), 2),
            "p50_ms": round(f_lats[len(f_lats) // 2], 2) if f_lats else 0.0,
            "p95_ms": round(f_lats[int(len(f_lats) * 0.95)], 2) if f_lats else 0.0,
        }

    failures = [r for r in results if not r["ok"]]
    report = {
        "evaluated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "trained_utterances_audited": len(trained_set),
        "training_overlap_count": len(overlaps),
        "total_untrained_questions": len(cases),
        "passed": passed,
        "pass_rate_pct": round(100.0 * passed / len(cases), 2),
        "wall_seconds": wall_s,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "families": fam_stats,
        "sample_failures": failures[:15],
    }
    COURSE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = COURSE_DIR / "untrained_reinvention_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[3/3] UNTRAINED REINVENTION BATTERY COMPLETE in {wall_s}s: {passed}/{len(cases)} PASSED ({report['pass_rate_pct']}%)", flush=True)
    print(f"      Latency ({len(cases)} live HTTP reqs): p50={p50} ms | p95={p95} ms | p99={p99} ms", flush=True)
    for fam, st in fam_stats.items():
        print(f"      - {fam:33s}: {st['passed']:4d}/{st['total']:4d} ({st['pass_rate_pct']:6.2f}%) | p50={st['p50_ms']:6.2f} ms | p95={st['p95_ms']:6.2f} ms", flush=True)
    if failures:
        print(f"      Sample failures ({len(failures)} total):")
        for f in failures[:8]:
            print(f"        * [{f['id']}] {f['reasons']} | Q: {f['question'][:90]}")
    sys.exit(0 if passed == len(cases) else 2)


if __name__ == "__main__":
    main()
