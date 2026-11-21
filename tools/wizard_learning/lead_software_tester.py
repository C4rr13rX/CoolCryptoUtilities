"""tools/wizard_learning/lead_software_tester.py

Zero-bias "Engineering Lead" evaluator for W1z4rD + C0d3rV2 (`POST /api/wizard-chat/agent/`).

Talks naturally like an Engineering / Lab / Enterprise Lead across 6 domains:
  1. Multi-PC Cluster & Repo Operations
  2. STEM & Numerical / Scientific Computing
  3. Wet-Lab / Biotech / Cheminformatics / LIMS
  4. STEAM — Audio DSP, Procedural SVG, Shaders & Color Science
  5. $MM SaaS & Mid-Market Infrastructure
  6. $B Enterprise FinTech, Double-Entry Ledgers & Distributed Systems
  + Multi-turn stateful Lead follow-ups ("save the last module and add edge-case tests")

Records every failure/weakness to `data/wizard_learning/enterprise_stem_steam_corpus/weakness_log.jsonl`
and `round_<N>_report.json`.
"""
from __future__ import annotations

import argparse
import datetime
import http.client
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = PROJECT_ROOT / "data" / "wizard_learning" / "enterprise_stem_steam_corpus"


def _lead_prompts() -> List[Dict[str, Any]]:
    return [
        # ── 1. Multi-PC Cluster & Repo Ops ───────────────────────────────────
        {
            "id": "lead_ops_01",
            "domain": "multi_pc_ops",
            "prompt": "Hey, give me a quick status check on our multi-PC Wizard cluster — which local and LAN nodes are reachable right now, and how is memory looking?",
            "must_contain_any": [["8090", "39106", "10.0.0.5", "cluster"], ["mb", "ram", "gb"]],
            "forbid": ["solve_task", "isn't grounded in my neural pools yet"],
        },
        {
            "id": "lead_ops_02",
            "domain": "multi_pc_ops",
            "prompt": "Can you check recent git commits in the repo so I can see what landed most recently?",
            "must_contain_any": [["7270a4d", "e4b424d", "feat(", "fix("]],
            "forbid": ["solve_task", "isn't grounded in my neural pools yet"],
        },
        {
            "id": "lead_ops_03",
            "domain": "multi_pc_ops",
            "prompt": "Look up how our peer experience exchange works across LAN nodes and summarize the safety and approval rules for me.",
            "must_contain_any": [["wbrain", "neuron", "128"], ["approve", "duplicate", "token", "39106", "39105"]],
            "forbid": ["solve_task", "isn't grounded in my neural pools yet"],
        },
        {
            "id": "lead_ops_04",
            "domain": "multi_pc_ops",
            "prompt": "Write a quick Python health-probe script that checks 127.0.0.1:8090/health and 10.0.0.5:39106/health with short timeouts and prints structured JSON status, and run it.",
            "must_contain_any": [["8090"], ["39106"], ["exit 0", "true", "online", "200"]],
            "forbid": ["solve_task", "isn't grounded in my neural pools yet"],
        },

        # ── 2. STEM & Scientific Computing ───────────────────────────────────
        {
            "id": "lead_stem_01",
            "domain": "stem_computing",
            "prompt": "We need a 4th-order Runge-Kutta (RK4) ODE solver in Python for a damped harmonic oscillator (x'' + 2*zeta*w0*x' + w0^2*x = 0), plus a quick verification showing amplitude decay. Build and run it.",
            "must_contain_any": [["k1", "rk4"], ["exit 0", "decay", "amplitude"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_stem_02",
            "domain": "stem_computing",
            "prompt": "Build a spectral FFT / DFT frequency analyzer in Python that detects the dominant frequencies in a noisy two-tone signal (50 Hz and 120 Hz sampled at 1000 Hz) and verify it.",
            "must_contain_any": [["50", "120"], ["fft", "dft", "freq"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_stem_03",
            "domain": "stem_computing",
            "prompt": "Write a Monte Carlo uncertainty propagation routine in Python for electrical power P = V^2 / R given Gaussian tolerances V = 24.0 +/- 0.2 V and R = 8.0 +/- 0.1 Ohm (5000 samples) and report mean + 95% confidence interval.",
            "must_contain_any": [["72."], ["95%", "ci", "std", "confidence"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_stem_04",
            "domain": "stem_computing",
            "prompt": "Build a linear least-squares calibration curve fitter (y = m*x + b with R^2 and residual standard error) for sensor voltage vs pressure data in Python, and run a test case.",
            "must_contain_any": [["r2", "r_squared", "slope"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },

        # ── 3. Research Labs, Biotech & Cheminformatics ──────────────────────
        {
            "id": "lead_lab_01",
            "domain": "research_labs",
            "prompt": "Our wet lab needs a FASTA DNA sequence analyzer in Python: compute length, GC-content percentage, reverse complement, and all ATG start-codon positions, then verify it on a sample sequence.",
            "must_contain_any": [["gc"], ["complement"], ["atg"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_lab_02",
            "domain": "research_labs",
            "prompt": "Build a laboratory molarity dilution (C1*V1 = C2*V2) and molecular weight calculator in Python supporting chemical formulas like H2O, C6H12O6, and NaCl, and verify it.",
            "must_contain_any": [["18.0", "180.1", "58.4"], ["dilution", "v1"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_lab_03",
            "domain": "research_labs",
            "prompt": "Create a LIMS high-throughput screening assay QC module in Python that computes Z-prime factor (1 - 3*(sigma_p + sigma_n)/|mu_p - mu_n|), classifies plate quality, and run a verification check.",
            "must_contain_any": [["z_prime", "z'"], ["0.5"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_lab_04",
            "domain": "research_labs",
            "prompt": "Build a pharmacokinetic 1-compartment IV bolus concentration model (C(t) = C0 * exp(-ke * t), half-life, and trapezoidal AUC) in Python and test it.",
            "must_contain_any": [["half_life", "t_half"], ["auc"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },

        # ── 4. STEAM — Creative Tech, Audio DSP, Shaders & Color Science ─────
        {
            "id": "lead_steam_01",
            "domain": "steam_creative",
            "prompt": "Build a pure-Python 16-bit PCM audio synthesizer + Biquad low-pass DSP filter that generates a 440 Hz A4 tone with ADSR envelope and verifies peak and RMS levels.",
            "must_contain_any": [["440"], ["biquad", "adsr"], ["rms"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_steam_02",
            "domain": "steam_creative",
            "prompt": "Generate a procedural SVG golden-angle phyllotaxis spiral generator in Python and verify the SVG output structure.",
            "must_contain_any": [["<svg"], ["137.5", "golden", "circle"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_steam_03",
            "domain": "steam_creative",
            "prompt": "Write a GLSL WebGL fragment shader for an animated signed-distance-field (SDF) smooth-min metaball surface.",
            "must_contain_any": [["gl_fragcolor", "fragcolor"], ["smin", "smooth", "length("], ["uniform"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_steam_04",
            "domain": "steam_creative",
            "prompt": "Build a color-science accessibility utility in Python that converts sRGB hex to linear RGB, relative luminance, and WCAG 2.1 contrast ratio (AA/AAA compliance), and test black-on-white vs low-contrast pairs.",
            "must_contain_any": [["21.0", "21:1", "21.0"], ["wcag", "contrast"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },

        # ── 5. $MM Mid-Market / SaaS Engineering ─────────────────────────────
        {
            "id": "lead_mm_01",
            "domain": "mm_saas",
            "prompt": "Hey, build a thread-safe Token Bucket rate limiter in Python for our multi-tenant SaaS API, with unit tests proving burst allowance and throttling.",
            "must_contain_any": [["lock", "threading"], ["token", "allow", "consume"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_mm_02",
            "domain": "mm_saas",
            "prompt": "We need an idempotent webhook receiver helper in Python with HMAC-SHA256 signature verification, replay-window timestamp check, and deduplication cache. Build and test it.",
            "must_contain_any": [["hmac"], ["compare_digest"], ["sha256"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_mm_03",
            "domain": "mm_saas",
            "prompt": "Write a resilient Circuit Breaker + exponential-backoff-with-jitter retry module in Python (CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine) and verify all state transitions.",
            "must_contain_any": [["half_open", "half-open"], ["open"], ["closed"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_mm_04",
            "domain": "mm_saas",
            "prompt": "Create a TypeScript RBAC permission & multi-tenant isolation guard (`orgId`, roles, scopes, and structured audit hook) for our B2B API.",
            "must_contain_any": [["orgid", "tenant"], ["role", "scope"], ["export"]],
            "forbid": ["solve_task", "isn't grounded"],
        },

        # ── 6. $B Enterprise FinTech, Ledgers & Distributed Systems ──────────
        {
            "id": "lead_ent_01",
            "domain": "enterprise_billion",
            "prompt": "Build an immutable double-entry accounting ledger engine in Python using Decimal (zero float drift), enforcing balanced debits == credits per journal entry, trial balance check, and SHA-256 audit hash chaining. Run tests on it.",
            "must_contain_any": [["decimal"], ["debit", "credit"], ["sha256"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ent_02",
            "domain": "enterprise_billion",
            "prompt": "Implement a Consistent Hashing ring with virtual nodes in Python for sharding keys across our multi-PC cluster, and verify that removing 1 of 4 nodes moves less than 40% of keys.",
            "must_contain_any": [["virtual", "ring", "bisect", "hashlib"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ent_03",
            "domain": "enterprise_billion",
            "prompt": "Build a quantitative risk & execution analytics module in Python: compute Historical VaR (95% and 99%), Expected Shortfall (CVaR), maximum drawdown, Sharpe ratio, and VWAP execution slicing, and run a test portfolio.",
            "must_contain_any": [["var", "cvar", "shortfall"], ["drawdown"], ["sharpe"], ["vwap"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ent_04",
            "domain": "enterprise_billion",
            "prompt": "Write a production SQL enterprise reconciliation query using CTEs and window functions that detects ledger balance gaps, duplicate idempotency keys, and rolling 7-day revenue anomalies per tenant.",
            "must_contain_any": [["with "], ["over (", "over("], ["partition by"]],
            "forbid": ["solve_task", "isn't grounded"],
        },

        # ── 7. Multi-Turn Stateful Lead Follow-Ups ───────────────────────────
        {
            "id": "lead_followup_01",
            "domain": "multi_turn_lead",
            "session_id": "lead-session-mt",
            "prompt": "Build a thread-safe LRU cache with TTL expiration in Python and run quick tests on eviction and expiry.",
            "must_contain_any": [["ordereddict", "lru"], ["ttl", "expir"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_followup_02",
            "domain": "multi_turn_lead",
            "session_id": "lead-session-mt",
            "prompt": "Nice — now save that last Python module to scratch/lead_lru_ttl.py and run a benchmark of 10,000 lookups on it.",
            "must_contain_any": [["scratch/lead_lru_ttl.py"], ["10000", "10,000", "exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
    ]


def _extended_lead_prompts() -> List[Dict[str, Any]]:
    """Additional unseen Engineering/R&D Lead challenges across all 7 domains to verify
    generalization beyond the initial 26-task Round 1 discovery set."""
    return [
        {
            "id": "lead_ext_ops_05",
            "domain": "multi_pc_ops",
            "prompt": "Hey, check git status for me so I can see what branch we're on and whether any files are modified.",
            "must_contain_any": [["main", "##"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_ops_06",
            "domain": "multi_pc_ops",
            "prompt": "How is the W1z4rD multi-PC cluster structured between Pegasus and DESKTOP-6E34B18?",
            "must_contain_any": [["10.0.0.2", "pegasus"], ["10.0.0.5", "39106"], ["wbrain", "8090"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_stem_05",
            "domain": "stem_computing",
            "prompt": "Write a 1D explicit finite-difference heat diffusion PDE solver in Python with Fourier stability verification and run it.",
            "must_contain_any": [["fourier"], ["midpoint_temp_c", "heat"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_stem_06",
            "domain": "stem_computing",
            "prompt": "Build a Newton-Raphson nonlinear root finder in Python and verify quadratic convergence on sqrt(2).",
            "must_contain_any": [["1.4142135"], ["newton"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_stem_07",
            "domain": "stem_computing",
            "prompt": "Implement composite Simpson's 1/3 numerical integration in Python and verify integrating sin(x) from 0 to pi.",
            "must_contain_any": [["2.000000"], ["simpson"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_lab_05",
            "domain": "research_labs",
            "prompt": "Our biochemistry team needs a Michaelis-Menten enzyme kinetics nonlinear fitter for Vmax and Km in Python — build and test it.",
            "must_contain_any": [["vmax"], ["km"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_steam_05",
            "domain": "steam_creative",
            "prompt": "Generate a procedural L-system botanical fractal tree in SVG using Python and verify the SVG line segments.",
            "must_contain_any": [["<svg"], ["<line"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_steam_06",
            "domain": "steam_creative",
            "prompt": "Write a WebGL GLSL 3D raymarching SDF fragment shader with surface normal lighting.",
            "must_contain_any": [["gl_fragcolor"], ["calcnormal", "mapscene"], ["uniform"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_mm_05",
            "domain": "mm_saas",
            "prompt": "Build a SOC2 and GDPR PII log redaction sanitizer in Python that scrubs emails, SSNs, Bearer tokens, and Luhn credit card numbers, and test it.",
            "must_contain_any": [["redacted_email"], ["redacted_ssn"], ["redacted_pan_last4"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_mm_06",
            "domain": "mm_saas",
            "prompt": "Implement a W3C traceparent OpenTelemetry distributed tracing context propagator in Python and test parent-child span linking.",
            "must_contain_any": [["traceparent"], ["00-"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_ent_05",
            "domain": "enterprise_billion",
            "prompt": "Build an enterprise Saga distributed transaction coordinator in Python with automatic reverse compensating rollbacks on failure, and verify rollback order.",
            "must_contain_any": [["saga"], ["charge_card", "reserve_inventory"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_mt_01",
            "domain": "multi_turn_lead",
            "session_id": "lead-session-mt2",
            "prompt": "Build a Consistent Hashing ring with virtual nodes in Python for our 4-node cluster and verify key movement.",
            "must_contain_any": [["consistenthashring"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_mt_02",
            "domain": "multi_turn_lead",
            "session_id": "lead-session-mt2",
            "prompt": "Great, save that last module to scratch/cluster_ring.py and benchmark 15,000 shard lookups on it.",
            "must_contain_any": [["scratch/cluster_ring.py"], ["15000", "15,000"], ["exit 0"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
        {
            "id": "lead_ext_mt_03",
            "domain": "multi_turn_lead",
            "session_id": "lead-session-mt2",
            "prompt": "read scratch/cluster_ring.py",
            "must_contain_any": [["consistenthashring"], ["bisect"]],
            "forbid": ["solve_task", "isn't grounded"],
        },
    ]


def evaluate_response(spec: Dict[str, Any], status: int, body: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if status != 200:
        return False, [f"HTTP {status}"]
    ans = str(body.get("answer") or "")
    low = ans.lower()
    if not ans.strip():
        return False, ["empty response"]

    for bad in spec.get("forbid") or []:
        if bad.lower() in low:
            reasons.append(f"hit forbidden fallback '{bad}'")

    for group in spec.get("must_contain_any") or []:
        if not any(tok.lower() in low for tok in group):
            reasons.append(f"missing expected domain evidence {group}")

    return len(reasons) == 0, reasons


def run_lead_round(round_num: int, *, include_extended: bool = False) -> Dict[str, Any]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    cases = _lead_prompts() + (_extended_lead_prompts() if include_extended else [])
    conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=25.0)

    results: List[Dict[str, Any]] = []
    weaknesses: List[Dict[str, Any]] = []
    latencies: List[float] = []
    passed = 0

    print(f"=== LEAD EVALUATION ROUND {round_num} ({len(cases)} natural Tech-Lead tasks) ===", flush=True)
    for spec in cases:
        sid = spec.get("session_id") or f"lead-r{round_num}-{spec['domain']}"
        payload = json.dumps({"text": spec["prompt"], "session_id": sid}).encode("utf-8")
        t0 = time.perf_counter()
        try:
            conn.request("POST", "/api/wizard-chat/agent/", body=payload, headers={"Content-Type": "application/json", "Connection": "keep-alive"})
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
            conn = http.client.HTTPConnection("127.0.0.1", 8001, timeout=25.0)
            ms = (time.perf_counter() - t0) * 1000.0
            st = 599
            body = {"answer": "", "error": str(exc)}

        latencies.append(ms)
        ok, reasons = evaluate_response(spec, st, body)
        preview = str(body.get("answer") or "").replace("\n", " ")[:110]
        badge = "PASS" if ok else "WEAK"
        print(f"  [{badge}] {spec['id']:16s} | {ms:7.2f} ms | {reasons if not ok else preview[:80]}", flush=True)

        rec = {
            "round": round_num,
            "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "id": spec["id"],
            "domain": spec["domain"],
            "prompt": spec["prompt"],
            "ok": ok,
            "ms": round(ms, 2),
            "weakness_reasons": reasons,
            "answer_preview": preview,
        }
        results.append(rec)
        if ok:
            passed += 1
        else:
            weaknesses.append(rec)

    conn.close()
    latencies.sort()
    p50 = round(latencies[len(latencies) // 2], 2) if latencies else 0.0
    p95 = round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 2) if latencies else 0.0

    # Append discovered weaknesses durably to weakness_log.jsonl
    if weaknesses:
        wlog = CORPUS_DIR / "weakness_log.jsonl"
        with wlog.open("a", encoding="utf-8") as fh:
            for w in weaknesses:
                fh.write(json.dumps(w, ensure_ascii=False) + "\n")

    report = {
        "round": round_num,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "total": len(cases),
        "pass_rate_pct": round(100.0 * passed / max(1, len(cases)), 2),
        "p50_ms": p50,
        "p95_ms": p95,
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
        "weaknesses_count": len(weaknesses),
        "results": results,
    }
    (CORPUS_DIR / f"round_{round_num}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nROUND {round_num} SUMMARY: {passed}/{len(cases)} PASSED ({report['pass_rate_pct']}%) | p50={p50} ms | p95={p95} ms | weaknesses logged={len(weaknesses)}", flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--include-extended", action="store_true")
    args = ap.parse_args()
    rep = run_lead_round(args.round, include_extended=args.include_extended)
    sys.exit(0 if rep["passed"] == rep["total"] else 2)
