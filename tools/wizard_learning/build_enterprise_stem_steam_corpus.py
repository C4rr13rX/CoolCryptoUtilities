"""tools/wizard_learning/build_enterprise_stem_steam_corpus.py

Compiles and saves the reusable Enterprise + STEM/STEAM + Wet-Lab + Multi-PC LAN
Cluster training corpus in `data/wizard_learning/enterprise_stem_steam_corpus/`,
and optionally trains it directly into W1z4rD (`127.0.0.1:8090` `chat_in -> chat_out`)
and `AnchoredMemory`.

Note: Never overwrites `weakness_log.jsonl` or `round_*_report.json`.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT / "tools" / "c0d3rV2"), str(PROJECT_ROOT / "tools"), str(PROJECT_ROOT)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

OUT_DIR = PROJECT_ROOT / "data" / "wizard_learning" / "enterprise_stem_steam_corpus"


def _lead_paraphrases(topic_phrase: str, extra: List[str]) -> List[str]:
    base = [
        f"Explain {topic_phrase}",
        f"How does {topic_phrase} work?",
        f"Summarize {topic_phrase} for me",
        f"Look up {topic_phrase} and summarize it",
        f"Give me an engineering overview of {topic_phrase}",
        f"What are the rules and architecture for {topic_phrase}?",
    ]
    seen = set()
    out: List[str] = []
    for s in extra + base:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def build_multi_pc_cluster_dataset() -> List[Dict[str, Any]]:
    return [
        {
            "id": "mpc_001",
            "category": "multi_pc_cluster",
            "question": "How does our peer experience exchange work across LAN nodes and what are the safety and approval rules?",
            "paraphrases": _lead_paraphrases(
                "our peer experience exchange across LAN nodes",
                [
                    "Look up how our peer experience exchange works across LAN nodes and summarize the safety and approval rules for me.",
                    "How do W1z4rD LAN nodes exchange .wbrain neuron experience safely?",
                    "What are the auto-approve, auto-deny, token, and page limits for W1z4rD peer experience exchange?",
                    "Explain peer .wbrain experience sync between Pegasus 10.0.0.2:39105 and DESKTOP 10.0.0.5:39106",
                    "Summarize the W1z4rD multi-PC LAN experience exchange safety rules",
                ],
            ),
            "answer": (
                "**W1z4rD Multi-PC LAN `.wbrain` Peer Experience Exchange — Architecture & Safety Rules**\n\n"
                "1. **LAN Topology & Endpoints**:\n"
                "   - **Local Primary Node (Pegasus)**: `http://127.0.0.1:8090` (conversational multi-pools & live cognitive substrate).\n"
                "   - **Pegasus LAN Cluster Seed**: `http://10.0.0.2:39105` · **DESKTOP-6E34B18 Native LAN Worker**: `http://10.0.0.5:39106` · **SSH Cold-Shard Staging Worker**: `http://127.0.0.1:39096` (`676,004` paged `.wbrain` slots).\n"
                "2. **Authentication & Peer Allowlist**:\n"
                "   - Requires a shared `>= 32`-character secret (`W1Z4RD_EXPERIENCE_TOKEN` / `WIZARD_EXPERIENCE_TOKEN`).\n"
                "   - Exporting peers must explicitly opt in via `W1Z4RD_EXPERIENCE_EXPORT=1` (`GET /brain/experience/manifest` and `POST /brain/experience/export`), and importing nodes only talk to allowlisted `WIZARD_EXPERIENCE_PEERS`.\n"
                "3. **Bounded Page Envelope**:\n"
                "   - Every peer export page is strictly bounded to **`<= 128` neurons**, **`<= 512 KiB`** payload, and a verified shared prefix of **`<= 16,384` slots** so cold `.wbrain` pages stay unmapped in disk cache.\n"
                "4. **Approval, Duplicate & Divergence Policy (`auto_approve` / `auto_deny`, both default `off`)**:\n"
                "   - **Auto-Approve**: Permitted *only* when the peer's slot prefix matches locally and incoming neurons are a clean, compatible append with valid cryptographic fingerprints.\n"
                "   - **Auto-Deny**: Automatically rejects exact duplicate neuron pages.\n"
                "   - **Human Review Required**: Triggered automatically on prefix divergence, missing cross-slot references, or failed fingerprint verification.\n"
                "5. **Crash-Safe Commit**:\n"
                "   - Approved pages commit atomically through the `.wbrain` manifest + Write-Ahead Log (`WAL`) barrier."
            ),
        },
        {
            "id": "mpc_002",
            "category": "multi_pc_cluster",
            "question": "How is the W1z4rD multi-PC cluster structured between Pegasus and DESKTOP-6E34B18?",
            "paraphrases": _lead_paraphrases(
                "the W1z4rD multi-PC cluster topology",
                [
                    "What are the IP addresses and ports of our multi-PC Wizard cluster?",
                    "Describe the Pegasus 10.0.0.2 and DESKTOP-6E34B18 10.0.0.5 cluster setup",
                    "How does paged .wbrain storage work across multiple PCs?",
                ],
            ),
            "answer": (
                "**W1z4rD Multi-PC Cluster Topology (`Pegasus` <-> `DESKTOP-6E34B18`)**\n\n"
                "- **Pegasus (`10.0.0.2`)**: Runs the primary Revenir web host (`127.0.0.1:8001`), live conversational W1z4rD node (`127.0.0.1:8090`), isolated `EdgeForge` worker daemon, and native cluster seed (`10.0.0.2:39105`).\n"
                "- **DESKTOP-6E34B18 (`10.0.0.5`)**: Runs the native LAN W1z4rD worker (`http://10.0.0.5:39106/health`) over Ethernet + pinned SSH (`codex-lan.conf`) and hosts staged `.wbrain` cold-shard workers (`127.0.0.1:39096` tunnel).\n"
                "- **Paged `.wbrain` Substrate**: Neuron bodies live on disk-backed pages (`neuron_body_snapshot` + `neuron_slot_count`) so hundreds of thousands of neurons (`676,004` slots) can be queried and exchanged across PCs without loading the full graph into resident RAM."
            ),
        },
        {
            "id": "mpc_003",
            "category": "multi_pc_cluster",
            "question": "How does Revenir WizardClusterPanel manage peer experience queues in the web UI?",
            "paraphrases": _lead_paraphrases(
                "WizardClusterPanel and cluster_views.py in Revenir",
                [
                    "What endpoints back the WizardClusterPanel in Revenir?",
                    "How do I approve or deny peer .wbrain experience imports in the UI?",
                ],
            ),
            "answer": (
                "**Revenir `WizardClusterPanel` & `cluster_views.py` Control Plane**\n\n"
                "- **Endpoints**: `/wizard-chat/api/cluster/overview/`, `/policy/`, `/pull/`, and `/decide/` proxy localhost-authenticated operator actions to the `.wbrain` brain server.\n"
                "- **Capabilities**: Displays live cluster members, `.wbrain` slot counts, pending/approved/denied experience batches, toggles `auto_approve` (compatible append only) and `auto_deny` (duplicates), triggers bounded `128`-neuron peer pulls, and commits operator `approve`/`deny` decisions."
            ),
        },
    ]


def build_stem_and_lab_dataset() -> List[Dict[str, Any]]:
    from tools.c0d3rV2.enterprise_stem_synth import synthesize_domain_software

    code_specs = [
        (
            "stem_001",
            "stem_computing",
            "How do you implement a 4th-order Runge-Kutta (RK4) ODE solver for a damped harmonic oscillator in Python?",
            "We need a 4th-order Runge-Kutta (RK4) ODE solver in Python for a damped harmonic oscillator (x'' + 2*zeta*w0*x' + w0^2*x = 0), plus a quick verification showing amplitude decay. Build and run it.",
            [
                "Explain 4th-order Runge-Kutta RK4 integration for x'' + 2*zeta*w0*x' + w0^2*x = 0",
                "Show a Python RK4 ODE solver for a damped spring-mass oscillator",
            ],
        ),
        (
            "stem_002",
            "stem_computing",
            "How do you build a Radix-2 FFT spectral frequency analyzer in Python for noisy multi-tone signals?",
            "Build a spectral FFT / DFT frequency analyzer in Python that detects the dominant frequencies in a noisy two-tone signal (50 Hz and 120 Hz sampled at 1000 Hz) and verify it.",
            [
                "Python Cooley-Tukey FFT dominant frequency detector with Hann window",
                "How do you detect 50 Hz and 120 Hz peaks in a noisy signal using FFT in Python?",
            ],
        ),
        (
            "stem_003",
            "stem_computing",
            "How do you perform Monte Carlo uncertainty propagation for electrical power P = V^2 / R in Python?",
            "Write a Monte Carlo uncertainty propagation routine in Python for electrical power P = V^2 / R given Gaussian tolerances V = 24.0 +/- 0.2 V and R = 8.0 +/- 0.1 Ohm (5000 samples) and report mean + 95% confidence interval.",
            [
                "Monte Carlo Gaussian tolerance propagation for P = V^2 / R in Python",
                "Compare Monte Carlo vs Taylor uncertainty propagation for electrical power",
            ],
        ),
        (
            "stem_004",
            "stem_computing",
            "How do you fit a linear least-squares sensor calibration curve with R^2 and residual standard error in Python?",
            "Build a linear least-squares calibration curve fitter (y = m*x + b with R^2 and residual standard error) for sensor voltage vs pressure data in Python, and run a test case.",
            [
                "Ordinary least squares sensor calibration curve y = mx + b with R^2 in Python",
                "Compute slope, intercept, R-squared, and RSE for laboratory calibration data",
            ],
        ),
        (
            "stem_005",
            "stem_computing",
            "How do you solve the 1D heat diffusion PDE with explicit finite differences and FTCS stability check in Python?",
            "Build a 1D finite-difference heat diffusion PDE solver in Python and verify FTCS stability.",
            [
                "1D heat equation FTCS Fourier number r <= 0.5 solver in Python",
                "Finite difference thermal diffusion solver in Python",
            ],
        ),
        (
            "lab_001",
            "research_labs",
            "How do you build a FASTA DNA sequence analyzer in Python for GC%, reverse complement, and ATG start codons?",
            "Our wet lab needs a FASTA DNA sequence analyzer in Python: compute length, GC-content percentage, reverse complement, and all ATG start-codon positions, then verify it on a sample sequence.",
            [
                "Bioinformatics Python FASTA parser with GC content and reverse complement",
                "Find ATG start codons and GC percentage in a FASTA DNA sequence",
            ],
        ),
        (
            "lab_002",
            "research_labs",
            "How do you calculate laboratory molarity dilutions (C1*V1 = C2*V2) and molecular weights in Python?",
            "Build a laboratory molarity dilution (C1*V1 = C2*V2) and molecular weight calculator in Python supporting chemical formulas like H2O, C6H12O6, and NaCl, and verify it.",
            [
                "Wet-lab stoichiometry C1V1 = C2V2 dilution and molecular mass parser in Python",
                "Compute molar mass of H2O, C6H12O6, and NaCl plus stock dilution volume",
            ],
        ),
        (
            "lab_003",
            "research_labs",
            "What is the Z'-factor in high-throughput screening (LIMS) and how do you compute it in Python?",
            "Create a LIMS high-throughput screening assay QC module in Python that computes Z-prime factor (1 - 3*(sigma_p + sigma_n)/|mu_p - mu_n|), classifies plate quality, and run a verification check.",
            [
                "Explain Zhang Z-prime factor for HTS plate quality control",
                "LIMS plate QC Z'-factor formula and Python implementation",
            ],
        ),
        (
            "lab_004",
            "research_labs",
            "How do you model 1-compartment IV bolus pharmacokinetics (C(t), half-life, and trapezoidal AUC) in Python?",
            "Build a pharmacokinetic 1-compartment IV bolus concentration model (C(t) = C0 * exp(-ke * t), half-life, and trapezoidal AUC) in Python and test it.",
            [
                "Pharmacokinetics IV bolus elimination half-life and linear trapezoidal AUC in Python",
                "Calculate C0, t_1/2 = ln(2)/ke, and AUC for a 1-compartment PK model",
            ],
        ),
        (
            "lab_005",
            "research_labs",
            "How do you fit Michaelis-Menten enzyme kinetics (Vmax and Km) in Python?",
            "Build a Michaelis-Menten enzyme kinetics fitter for Vmax and Km in Python and test it.",
            [
                "Nonlinear Gauss-Newton Michaelis-Menten Vmax and Km fit in Python",
                "Enzyme kinetics v = Vmax * [S] / (Km + [S]) solver",
            ],
        ),
    ]

    items: List[Dict[str, Any]] = []
    for cid, cat, q, synth_prompt, extra_p in code_specs:
        hit = synthesize_domain_software(synth_prompt)
        assert hit is not None, f"Missing synth for {cid}"
        lang, title, code = hit
        items.append({
            "id": cid,
            "category": cat,
            "question": q,
            "paraphrases": _lead_paraphrases(title, extra_p),
            "answer": f"**{title}**\n\n```{lang}\n{code.rstrip()}\n```",
        })
    return items


def build_steam_creative_dataset() -> List[Dict[str, Any]]:
    from tools.c0d3rV2.enterprise_stem_synth import synthesize_domain_software

    specs = [
        (
            "steam_001",
            "How do you synthesize 16-bit PCM audio with an ADSR envelope and RBJ Biquad low-pass filter in Python?",
            "Build a pure-Python 16-bit PCM audio synthesizer + Biquad low-pass DSP filter that generates a 440 Hz A4 tone with ADSR envelope and verifies peak and RMS levels.",
            [
                "Pure-Python 440 Hz PCM16 audio synthesizer with ADSR and Biquad DSP filter",
                "Explain RBJ audio EQ cookbook biquad low-pass filter coefficients",
            ],
        ),
        (
            "steam_002",
            "How do you generate a procedural SVG golden-angle (137.508 deg) phyllotaxis spiral in Python?",
            "Generate a procedural SVG golden-angle phyllotaxis spiral generator in Python and verify the SVG output structure.",
            [
                "Vogel golden angle 137.5 degrees phyllotaxis SVG generator in Python",
                "Procedural botanical spiral SVG generator",
            ],
        ),
        (
            "steam_003",
            "Write a GLSL WebGL fragment shader for an animated signed-distance-field (SDF) smooth-min metaball surface.",
            "Write a GLSL WebGL fragment shader for an animated signed-distance-field (SDF) smooth-min metaball surface.",
            [
                "GLSL smooth minimum smin SDF metaball fragment shader",
                "WebGL 2D signed distance field metaballs shader with u_time and u_resolution",
                "Create a GLSL metaball shader using polynomial smooth-min",
            ],
        ),
        (
            "steam_004",
            "Write a WebGL GLSL 3D raymarching fragment shader with surface normals and Phong specular lighting.",
            "Write a GLSL WebGL raymarching sphere SDF fragment shader.",
            [
                "GLSL sphere tracing raymarcher with finite-difference SDF normals",
                "3D SDF raymarching shader in WebGL GLSL",
            ],
        ),
        (
            "steam_005",
            "How do you compute sRGB relative luminance and WCAG 2.1 contrast ratios (AA/AAA) in Python?",
            "Build a color-science accessibility utility in Python that converts sRGB hex to linear RGB, relative luminance, and WCAG 2.1 contrast ratio (AA/AAA compliance), and test black-on-white vs low-contrast pairs.",
            [
                "WCAG 2.1 contrast ratio formula (L1 + 0.05) / (L2 + 0.05) and sRGB gamma linearization",
                "Check AA 4.5:1 and AAA 7.0:1 color contrast accessibility in Python",
            ],
        ),
        (
            "steam_006",
            "How do you generate a procedural L-system botanical fractal tree in SVG with Python?",
            "Generate a procedural L-system fractal tree SVG in Python.",
            [
                "Lindenmayer L-system turtle graphics SVG fractal generator in Python",
            ],
        ),
    ]

    items: List[Dict[str, Any]] = []
    for cid, q, synth_prompt, extra_p in specs:
        hit = synthesize_domain_software(synth_prompt)
        assert hit is not None, f"Missing synth for {cid}"
        lang, title, code = hit
        items.append({
            "id": cid,
            "category": "steam_creative",
            "question": q,
            "paraphrases": _lead_paraphrases(title, extra_p),
            "answer": f"**{title}**\n\n```{lang}\n{code.rstrip()}\n```",
        })
    return items


def build_enterprise_dataset() -> List[Dict[str, Any]]:
    from tools.c0d3rV2.enterprise_stem_synth import synthesize_domain_software

    specs = [
        (
            "mm_001",
            "mm_saas",
            "How do you implement a thread-safe multi-tenant Token Bucket rate limiter in Python?",
            "Hey, build a thread-safe Token Bucket rate limiter in Python for our multi-tenant SaaS API, with unit tests proving burst allowance and throttling.",
            ["Python RLock multi-tenant API token bucket rate limiter"],
        ),
        (
            "mm_002",
            "mm_saas",
            "How do you build an idempotent HMAC-SHA256 webhook verifier with replay protection in Python?",
            "We need an idempotent webhook receiver helper in Python with HMAC-SHA256 signature verification, replay-window timestamp check, and deduplication cache. Build and test it.",
            ["Stripe-style HMAC-SHA256 webhook signature verification with hmac.compare_digest and replay window"],
        ),
        (
            "mm_003",
            "mm_saas",
            "How do you implement a Circuit Breaker (CLOSED -> OPEN -> HALF_OPEN) with full-jitter exponential backoff in Python?",
            "Write a resilient Circuit Breaker + exponential-backoff-with-jitter retry module in Python (CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine) and verify all state transitions.",
            ["AWS-style full jitter backoff and Circuit Breaker state machine in Python"],
        ),
        (
            "mm_004",
            "mm_saas",
            "Create a TypeScript RBAC permission & multi-tenant isolation guard (`orgId`, roles, scopes, and structured audit hook) for our B2B API.",
            "Create a TypeScript RBAC permission & multi-tenant isolation guard (`orgId`, roles, scopes, and structured audit hook) for our B2B API.",
            [
                "TypeScript multi-tenant orgId isolation and RBAC scope guard",
                "Write a TypeScript B2B SaaS authorization guard with roles, scopes, and audit callback",
            ],
        ),
        (
            "ent_001",
            "enterprise_billion",
            "How do you implement an immutable Decimal double-entry accounting ledger with SHA-256 hash chaining in Python?",
            "Build an immutable double-entry accounting ledger engine in Python using Decimal (zero float drift), enforcing balanced debits == credits per journal entry, trial balance check, and SHA-256 audit hash chaining. Run tests on it.",
            ["FinTech immutable double-entry general ledger in Python with Decimal and SHA-256 audit log"],
        ),
        (
            "ent_002",
            "enterprise_billion",
            "How do you implement a virtual-node Consistent Hashing ring in Python for distributed cluster sharding?",
            "Implement a Consistent Hashing ring with virtual nodes in Python for sharding keys across our multi-PC cluster, and verify that removing 1 of 4 nodes moves less than 40% of keys.",
            ["Consistent hash ring with bisect and SHA-256 virtual replicas in Python"],
        ),
        (
            "ent_003",
            "enterprise_billion",
            "How do you compute Historical VaR, Expected Shortfall (CVaR), Maximum Drawdown, Sharpe Ratio, and VWAP execution slices in Python?",
            "Build a quantitative risk & execution analytics module in Python: compute Historical VaR (95% and 99%), Expected Shortfall (CVaR), maximum drawdown, Sharpe ratio, and VWAP execution slicing, and run a test portfolio.",
            ["Quantitative portfolio risk VaR CVaR Sharpe max drawdown and VWAP schedule in Python"],
        ),
        (
            "ent_004",
            "enterprise_billion",
            "Write a production SQL enterprise reconciliation query using CTEs and window functions that detects ledger balance gaps, duplicate idempotency keys, and rolling 7-day revenue anomalies per tenant.",
            "Write a production SQL enterprise reconciliation query using CTEs and window functions that detects ledger balance gaps, duplicate idempotency keys, and rolling 7-day revenue anomalies per tenant.",
            [
                "Enterprise FinTech SQL ledger reconciliation query with WITH CTEs and OVER PARTITION BY window functions",
                "SQL query to detect unbalanced journal entries, duplicate idempotency keys, and 3-sigma rolling revenue anomalies",
            ],
        ),
        (
            "ent_005",
            "enterprise_billion",
            "How do you build a thread-safe LRU cache with TTL expiration in Python?",
            "Build a thread-safe LRU cache with TTL expiration in Python and run quick tests on eviction and expiry.",
            ["Python OrderedDict + RLock thread-safe LRU cache with monotonic TTL expiry"],
        ),
        (
            "ent_006",
            "enterprise_billion",
            "How do you implement a Saga distributed transaction coordinator with reverse compensating rollbacks in Python?",
            "Build a Saga distributed transaction coordinator with compensating rollbacks in Python.",
            ["Microservices Saga pattern orchestrator with compensating transactions in Python"],
        ),
        (
            "ent_007",
            "enterprise_billion",
            "How do you scrub PII (emails, SSNs, Luhn credit cards, Bearer tokens) from SOC2 / GDPR logs in Python?",
            "Build a SOC2 / GDPR PII log redaction sanitizer in Python.",
            ["Luhn credit card PAN, SSN, email, and JWT Bearer token log redactor in Python"],
        ),
    ]

    items: List[Dict[str, Any]] = []
    for cid, cat, q, synth_prompt, extra_p in specs:
        hit = synthesize_domain_software(synth_prompt)
        assert hit is not None, f"Missing synth for {cid}"
        lang, title, code = hit
        items.append({
            "id": cid,
            "category": cat,
            "question": q,
            "paraphrases": _lead_paraphrases(title, extra_p),
            "answer": f"**{title}**\n\n```{lang}\n{code.rstrip()}\n```",
        })
    return items


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> int:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def train_into_wizard(all_items: List[Dict[str, Any]], endpoint: str = "http://127.0.0.1:8090") -> Dict[str, int]:
    from services.anchored_memory import AnchoredMemoryManager
    from tools.wizard_learning.fast_conversational_engine import (
        POOL_CHAT_IN,
        POOL_CHAT_OUT,
        record_hot_chat_pair,
    )

    am = AnchoredMemoryManager.get_instance()
    anchored = 0
    pool_trained = 0

    for pool_id in (POOL_CHAT_IN, POOL_CHAT_OUT):
        try:
            req = urllib.request.Request(
                f"{endpoint}/multi_pool/register",
                data=json.dumps({"pool_id": pool_id}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2.0).read()
        except Exception:
            pass

    for rec in all_items:
        q = str(rec["question"])
        a = str(rec["answer"])
        pars = list(rec.get("paraphrases") or [])
        cat = str(rec.get("category") or "enterprise_stem")
        am.record_memory(q, a, variations=pars[:8], source="enterprise_stem_steam_corpus", category=cat)
        anchored += 1
        record_hot_chat_pair(q, a)
        for pv in pars[:6]:
            record_hot_chat_pair(pv, a)
        # Train concise summary / head into :8090 chat_in -> chat_out
        for cue in [q] + pars[:2]:
            try:
                req = urllib.request.Request(
                    f"{endpoint}/multi_pool/train_pair",
                    data=json.dumps({
                        "src_pool": POOL_CHAT_IN,
                        "src": cue[:320],
                        "tgt_pool": POOL_CHAT_OUT,
                        "tgt": a[:1800],
                        "passes": 14,
                        "lr": 0.45,
                    }).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    if resp.status == 200:
                        pool_trained += 1
            except Exception:
                pass

    return {"anchored": anchored, "pool_pairs_trained": pool_trained}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-live", action="store_true", help="Also train compiled datasets into :8090 and AnchoredMemory")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "multi_pc_cluster_and_ops.jsonl": build_multi_pc_cluster_dataset(),
        "stem_and_lab_software.jsonl": build_stem_and_lab_dataset(),
        "steam_creative_tech.jsonl": build_steam_creative_dataset(),
        "enterprise_mm_and_billion_corp.jsonl": build_enterprise_dataset(),
    }

    total_items = 0
    total_utterances = 0
    all_rows: List[Dict[str, Any]] = []
    files_summary: Dict[str, Any] = {}
    for fname, rows in datasets.items():
        cnt = _write_jsonl(OUT_DIR / fname, rows)
        utts = sum(1 + len(r.get("paraphrases") or []) for r in rows)
        total_items += cnt
        total_utterances += utts
        all_rows.extend(rows)
        files_summary[fname] = {"items": cnt, "utterances": utts}

    train_stats: Dict[str, int] = {}
    if args.train_live:
        train_stats = train_into_wizard(all_rows)

    manifest = {
        "compiled_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_canonical_items": total_items,
        "total_paraphrased_utterances": total_utterances,
        "files": files_summary,
        "live_training": train_stats,
    }
    (OUT_DIR / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
