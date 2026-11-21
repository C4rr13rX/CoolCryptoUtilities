"""tools/c0d3rV2/enterprise_stem_synth.py

Production-grade STEM, Wet-Lab/Biotech/LIMS, STEAM (Audio DSP, SVG, GLSL Shaders,
Color Science), $MM SaaS Infrastructure, $B Enterprise FinTech/Distributed Systems,
and Multi-PC LAN Cluster Software Synthesizer + Live Cluster Probe for C0d3rV2.

Every Python module returned by `synthesize_domain_software()` is:
  1. Pure standard-library + deterministic (zero external API keys, zero flaky deps)
  2. Verified via `ast.parse()` before emission
  3. Equipped with a real `__main__` verification suite (`assert` + structured metrics)
"""
from __future__ import annotations

import ast
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tool_registry import Tool
except Exception:
    class Tool:  # type: ignore[no-redef]
        name = ""
        description = ""
        use_when = ""
        params_schema: Dict[str, str] = {}
        def execute(self, params: dict) -> dict:
            raise NotImplementedError


# ── Live Multi-PC W1z4rD Cluster Probe (<350 ms concurrent) ─────────────────

DEFAULT_CLUSTER_ENDPOINTS: List[Tuple[str, str, str]] = [
    ("local_primary_8090", "http://127.0.0.1:8090/health", "Pegasus Local W1z4rD Node (:8090)"),
    ("lan_desktop_39106", "http://10.0.0.5:39106/health", "DESKTOP-6E34B18 LAN Worker (10.0.0.5:39106)"),
    ("lan_pegasus_seed_39105", "http://10.0.0.2:39105/health", "Pegasus LAN Cluster Seed (10.0.0.2:39105)"),
    ("ssh_staging_shard_39096", "http://127.0.0.1:39096/health", "SSH Cold-Shard Staging Worker (:39096)"),
]


def _probe_single_url(name: str, url: str, label: str, timeout: float = 0.45) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "W1z4rD-C0d3rV2-ClusterProbe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(512).decode("utf-8", errors="replace").strip()
            ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return {
                "id": name,
                "label": label,
                "url": url,
                "online": 200 <= resp.status < 300,
                "status": resp.status,
                "latency_ms": ms,
                "body_preview": raw[:120],
            }
    except Exception as exc:
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "id": name,
            "label": label,
            "url": url,
            "online": False,
            "status": 0,
            "latency_ms": ms,
            "error": type(exc).__name__,
        }


def probe_wizard_cluster_live(timeout: float = 0.45) -> Dict[str, Any]:
    """Concurrently probe local + LAN W1z4rD cluster nodes and read live host RAM metrics."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_probe_single_url, n, u, lbl, timeout) for n, u, lbl in DEFAULT_CLUSTER_ENDPOINTS]
        nodes = [f.result() for f in futs]

    ram_info: Dict[str, Any] = {}
    cog_info: Dict[str, Any] = {}
    try:
        from tools.wizard_learning.settling_controller import get_settling_controller
        from tools.wizard_learning.cognitive_state import get_cognitive_state_manager
        ram_info = get_settling_controller().get_resource_metrics()
        cog_info = get_cognitive_state_manager().read()
    except Exception:
        pass

    online_count = sum(1 for n in nodes if n["online"])
    return {
        "online_nodes": online_count,
        "total_probed": len(nodes),
        "nodes": nodes,
        "memory": {
            "node_ram_mb": ram_info.get("node_ram_mb", "—"),
            "system_available_ram_gb": ram_info.get("system_available_ram_gb", "—"),
            "polite_reserve_gb": ram_info.get("polite_reserve_gb", 8.0),
            "allowance_tier": ram_info.get("allowance_tier", "BALANCED"),
            "sleep_pressure_pct": round(float(ram_info.get("sleep_pressure") or 0.0) * 100, 1),
        },
        "cognitive_mode": cog_info.get("mode_label") or cog_info.get("mode") or "AWAKE",
    }


def format_cluster_status_markdown(snapshot: Dict[str, Any]) -> str:
    mem = snapshot["memory"]
    lines = [
        f"**[C0d3rV2 Cluster Ops · Live Multi-PC W1z4rD Status]** (`{snapshot['online_nodes']}/{snapshot['total_probed']}` endpoints reachable)",
        "",
        "### 1. Local & LAN Cluster Nodes",
    ]
    for n in snapshot["nodes"]:
        if n["online"]:
            lines.append(f"- **ONLINE** · **{n['label']}** (`{n['url']}`) — `HTTP {n['status']}` in `{n['latency_ms']} ms`")
        else:
            lines.append(f"- **OFFLINE / STAGED** · {n['label']} (`{n['url']}`) — `{n.get('error', 'unreachable')}` (`{n['latency_ms']} ms`)")
    lines.extend([
        "",
        "### 2. Host Memory & Cognitive Substrate",
        f"- **Local W1z4rD Node Resident RAM (`:8090`)**: `{mem['node_ram_mb']} MB`",
        f"- **Available Host System RAM**: `{mem['system_available_ram_gb']} GB` free (`{mem['polite_reserve_gb']} GB` untouched polite reserve · Tier `{mem['allowance_tier']}`)",
        f"- **Cognitive Lifecycle State**: `{snapshot['cognitive_mode']}` (`{mem['sleep_pressure_pct']}%` sleep pressure)",
    ])
    return "\n".join(lines)


# ── Domain Software Blueprints (STEM, Wet-Lab, STEAM, $MM SaaS, $B Corp) ────

def synthesize_domain_software(prompt: str) -> Optional[Tuple[str, str, str]]:
    """Return `(language, title, verified_code)` for STEM, Lab, STEAM, Mid-Market,
    Enterprise, and Multi-PC Ops software requests, or `None` if non-coding."""
    raw = (prompt or "").strip()
    low = raw.lower()
    if not low:
        return None

    # Never hijack pure conceptual questions unless user asks to write/build/create/implement/generate
    build_verbs = (
        "write ", "build ", "create ", "implement ", "generate ", "code ", "develop ",
        "script", "module", "function", "shader", "query", "solver", "analyzer",
        "calculator", "limiter", "ledger", "guard", "probe",
    )
    # Never hijack pure conceptual / causal-mechanics / reinvention prose unless software/code is requested
    if any(k in low for k in ("emergent mechanics", "prerogative", "untrained", "reinvent the wheel", "how would you build")) and not any(
        c in low for c in ("python", "typescript", "sql", "glsl", "shader", "script", "module", "function", "code")
    ):
        return None
    if not any(v in low for v in build_verbs) and "python" not in low and "glsl" not in low and "typescript" not in low and "sql" not in low:
        return None

    # ── A. STEAM: GLSL / WebGL Shaders ───────────────────────────────────────
    if any(k in low for k in ("glsl", "webgl", "fragment shader", "vertex shader", "raymarch", "metaball", "sdf")):
        if "raymarch" in low or "torus" in low or "sphere" in low:
            title = "WebGL / GLSL Raymarched Signed-Distance-Field (SDF) Fragment Shader"
            code = """precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
varying vec2 v_uv;

float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-4), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float mapScene(vec3 p) {
    vec3 c1 = vec3(sin(u_time) * 0.65, cos(u_time * 0.8) * 0.35, 0.0);
    vec3 c2 = vec3(-sin(u_time * 0.9) * 0.65, sin(u_time * 1.1) * 0.35, 0.0);
    float d1 = length(p - c1) - 0.55;
    float d2 = length(p - c2) - 0.48;
    return smin(d1, d2, 0.35);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy) - mapScene(p - e.xyy),
        mapScene(p + e.yxy) - mapScene(p - e.yxy),
        mapScene(p + e.yyx) - mapScene(p - e.yyx)
    ));
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);
    vec3 ro = vec3(0.0, 0.0, 2.6);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    vec3 col = vec3(0.04, 0.06, 0.11);
    for (int i = 0; i < 64; i++) {
        vec3 p = ro + rd * t;
        float d = mapScene(p);
        if (d < 0.001) {
            vec3 n = calcNormal(p);
            vec3 light = normalize(vec3(0.6, 0.8, 1.0));
            float diff = max(dot(n, light), 0.08);
            col = vec3(0.18, 0.62, 0.96) * diff + pow(max(dot(reflect(-light, n), -rd), 0.0), 24.0);
            break;
        }
        t += d;
        if (t > 8.0) break;
    }
    gl_FragColor = vec4(col, 1.0);
}
"""
            return ("glsl", title, code)

        title = "WebGL / GLSL Animated SDF Smooth-Min Metaball Fragment Shader"
        code = """precision highp float;

uniform vec2 u_resolution;
uniform float u_time;

// Polynomial smooth-minimum (IQ smin) blending two signed distance fields
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-4), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float sdfCircle(vec2 p, vec2 center, float radius) {
    return length(p - center) - radius;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / min(u_resolution.x, u_resolution.y);

    vec2 c1 = vec2(0.38 * cos(u_time * 1.1), 0.28 * sin(u_time * 1.4));
    vec2 c2 = vec2(-0.35 * cos(u_time * 0.9), 0.32 * sin(u_time * 0.7 + 1.2));
    vec2 c3 = vec2(0.18 * sin(u_time * 1.7), -0.34 * cos(u_time * 1.3));

    float d1 = sdfCircle(uv, c1, 0.24);
    float d2 = sdfCircle(uv, c2, 0.21);
    float d3 = sdfCircle(uv, c3, 0.19);

    float field = smin(smin(d1, d2, 0.22), d3, 0.22);
    float mask = smoothstep(0.012, -0.012, field);
    float rim = exp(-18.0 * abs(field));

    vec3 bg = vec3(0.03, 0.05, 0.10);
    vec3 core = mix(vec3(0.08, 0.55, 0.95), vec3(0.52, 0.92, 1.0), clamp(-field * 4.0, 0.0, 1.0));
    vec3 color = mix(bg, core, mask) + vec3(0.25, 0.75, 1.0) * rim * 0.45;

    gl_FragColor = vec4(color, 1.0);
}
"""
        return ("glsl", title, code)

    # ── B. $MM SaaS: TypeScript Multi-Tenant RBAC & Policy Guard ─────────────
    if ("typescript" in low or re.search(r"\bts\b", low)) and any(
        k in low for k in ("rbac", "tenant", "orgid", "permission", "scope", "auth", "guard", "middleware")
    ):
        title = "TypeScript Multi-Tenant RBAC & Scope Isolation Guard"
        code = """export type Role = 'owner' | 'admin' | 'engineer' | 'analyst' | 'viewer';

export interface PrincipalContext {
  userId: string;
  orgId: string;
  roles: ReadonlyArray<Role>;
  scopes: ReadonlyArray<string>;
}

export interface ResourceDescriptor {
  orgId: string;
  resourceType: string;
  resourceId: string;
  requiredScopes: ReadonlyArray<string>;
}

export interface AuditDecisionEvent {
  timestampIso: string;
  userId: string;
  principalOrgId: string;
  resourceOrgId: string;
  resourceType: string;
  resourceId: string;
  allowed: boolean;
  reason: string;
}

export type AuditHook = (event: AuditDecisionEvent) => void;

const ROLE_DEFAULT_SCOPES: Record<Role, ReadonlyArray<string>> = {
  owner: ['*'],
  admin: ['org:read', 'org:write', 'billing:read', 'projects:read', 'projects:write', 'audit:read'],
  engineer: ['projects:read', 'projects:write', 'deployments:execute'],
  analyst: ['projects:read', 'metrics:read', 'reports:export'],
  viewer: ['projects:read'],
};

export function resolveEffectiveScopes(principal: PrincipalContext): Set<string> {
  const effective = new Set<string>(principal.scopes);
  for (const role of principal.roles) {
    for (const s of ROLE_DEFAULT_SCOPES[role] ?? []) {
      effective.add(s);
    }
  }
  return effective;
}

export function enforceTenantRbac(
  principal: PrincipalContext,
  resource: ResourceDescriptor,
  onAudit?: AuditHook,
): boolean {
  const emit = (allowed: boolean, reason: string): boolean => {
    onAudit?.({
      timestampIso: new Date().toISOString(),
      userId: principal.userId,
      principalOrgId: principal.orgId,
      resourceOrgId: resource.orgId,
      resourceType: resource.resourceType,
      resourceId: resource.resourceId,
      allowed,
      reason,
    });
    return allowed;
  };

  if (!principal.orgId || principal.orgId !== resource.orgId) {
    return emit(false, 'CROSS_TENANT_ISOLATION_DENIED');
  }

  const effective = resolveEffectiveScopes(principal);
  if (effective.has('*')) {
    return emit(true, 'WILDCARD_OWNER_GRANT');
  }

  for (const reqScope of resource.requiredScopes) {
    if (!effective.has(reqScope)) {
      return emit(false, `MISSING_REQUIRED_SCOPE:${reqScope}`);
    }
  }
  return emit(true, 'RBAC_SCOPE_VERIFIED');
}
"""
        return ("typescript", title, code)

    # ── C. $B Enterprise: Production SQL CTE + Window Reconciliation ─────────
    if "sql" in low and any(
        k in low for k in ("reconcil", "cte", "window", "partition by", "ledger", "anomal", "idempotency", "rolling", "cohort")
    ):
        title = "Enterprise Ledger Reconciliation & Anomaly Detection SQL (`WITH` CTEs + Window Functions)"
        code = """WITH journal_totals AS (
    SELECT
        tenant_id,
        journal_id,
        idempotency_key,
        posted_at,
        DATE(posted_at) AS posting_date,
        SUM(CASE WHEN side = 'DEBIT'  THEN amount_cents ELSE 0 END) AS debit_cents,
        SUM(CASE WHEN side = 'CREDIT' THEN amount_cents ELSE 0 END) AS credit_cents,
        SUM(CASE WHEN account_type = 'REVENUE' AND side = 'CREDIT' THEN amount_cents ELSE 0 END) AS revenue_cents,
        COUNT(*) OVER (
            PARTITION BY tenant_id, idempotency_key
        ) AS idempotency_key_occurrences
    FROM ledger_postings
    GROUP BY tenant_id, journal_id, idempotency_key, posted_at
),
ordered_ledger AS (
    SELECT
        tenant_id,
        journal_id,
        idempotency_key,
        posted_at,
        posting_date,
        debit_cents,
        credit_cents,
        (debit_cents - credit_cents) AS imbalance_cents,
        idempotency_key_occurrences,
        LAG(posted_at) OVER (
            PARTITION BY tenant_id
            ORDER BY posted_at, journal_id
        ) AS prev_posted_at,
        SUM(debit_cents - credit_cents) OVER (
            PARTITION BY tenant_id
            ORDER BY posted_at, journal_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_imbalance_cents
    FROM journal_totals
),
daily_tenant_revenue AS (
    SELECT
        tenant_id,
        posting_date,
        SUM(revenue_cents) AS daily_revenue_cents,
        AVG(SUM(revenue_cents)) OVER (
            PARTITION BY tenant_id
            ORDER BY posting_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg_cents,
        STDDEV_POP(SUM(revenue_cents)) OVER (
            PARTITION BY tenant_id
            ORDER BY posting_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_stddev_cents
    FROM journal_totals
    GROUP BY tenant_id, posting_date
)
SELECT
    l.tenant_id,
    l.journal_id,
    l.idempotency_key,
    l.posted_at,
    l.imbalance_cents,
    l.running_imbalance_cents,
    l.idempotency_key_occurrences,
    r.daily_revenue_cents,
    ROUND(r.rolling_7d_avg_cents, 2) AS rolling_7d_avg_cents,
    CASE
        WHEN l.imbalance_cents <> 0 THEN 'UNBALANCED_JOURNAL_ENTRY'
        WHEN l.idempotency_key_occurrences > 1 THEN 'DUPLICATE_IDEMPOTENCY_KEY'
        WHEN r.rolling_7d_stddev_cents > 0
             AND ABS(r.daily_revenue_cents - r.rolling_7d_avg_cents) > 3.0 * r.rolling_7d_stddev_cents
             THEN 'REVENUE_3SIGMA_ANOMALY'
        ELSE 'OK'
    END AS reconciliation_flag
FROM ordered_ledger l
LEFT JOIN daily_tenant_revenue r
    ON r.tenant_id = l.tenant_id
   AND r.posting_date = l.posting_date
WHERE l.imbalance_cents <> 0
   OR l.idempotency_key_occurrences > 1
   OR (
       r.rolling_7d_stddev_cents > 0
       AND ABS(r.daily_revenue_cents - r.rolling_7d_avg_cents) > 3.0 * r.rolling_7d_stddev_cents
   )
ORDER BY l.tenant_id, l.posted_at DESC;
"""
        return ("sql", title, code)

    # Do not intercept other explicit non-Python languages here
    if re.search(r"\b(?:rust|typescript|javascript|golang|powershell|bash|sql)\b", low) and "python" not in low:
        return None

    # ── D. Python Domain Software Modules ────────────────────────────────────
    py_title = ""
    py_code = ""

    # 1. Multi-PC Health Probe Script (`8090`, `39106`, `39105`)
    if ("health" in low or "probe" in low) and any(p in low for p in ("8090", "39106", "39105", "10.0.0.5", "cluster")):
        py_title = "Multi-PC W1z4rD Cluster Health Probe (`127.0.0.1:8090` + `10.0.0.5:39106`)"
        py_code = """import concurrent.futures
import json
import time
import urllib.request
from typing import Any

ENDPOINTS = [
    ("local_wizard_8090", "http://127.0.0.1:8090/health"),
    ("lan_desktop_39106", "http://10.0.0.5:39106/health"),
]

def probe_node(name: str, url: str, timeout: float = 0.6) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "W1z4rD-HealthProbe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read(256).decode("utf-8", errors="replace").strip()
            return {
                "node": name,
                "url": url,
                "online": 200 <= resp.status < 300,
                "http_status": resp.status,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "payload": body[:100],
            }
    except Exception as exc:
        return {
            "node": name,
            "url": url,
            "online": False,
            "http_status": 0,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "error": str(exc),
        }

def check_cluster(endpoints: list[tuple[str, str]] = ENDPOINTS) -> dict[str, Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        results = [f.result() for f in (pool.submit(probe_node, n, u) for n, u in endpoints)]
    return {
        "cluster_healthy": any(r["online"] for r in results),
        "reachable_count": sum(1 for r in results if r["online"]),
        "total_nodes": len(results),
        "nodes": results,
    }

if __name__ == "__main__":
    report = check_cluster()
    print(json.dumps(report, indent=2))
"""

    # 2. STEM: 4th-Order Runge-Kutta (RK4) ODE Solver
    elif "runge-kutta" in low or "rk4" in low or ("ode" in low and ("harmonic" in low or "oscillator" in low or "solver" in low)):
        py_title = "4th-Order Runge-Kutta (`RK4`) Damped Harmonic Oscillator Solver"
        py_code = """import math
from typing import Callable

def rk4_step(
    deriv: Callable[[float, list[float]], list[float]],
    t: float,
    state: list[float],
    dt: float,
) -> list[float]:
    k1 = deriv(t, state)
    s2 = [y + 0.5 * dt * d for y, d in zip(state, k1)]
    k2 = deriv(t + 0.5 * dt, s2)
    s3 = [y + 0.5 * dt * d for y, d in zip(state, k2)]
    k3 = deriv(t + 0.5 * dt, s3)
    s4 = [y + dt * d for y, d in zip(state, k3)]
    k4 = deriv(t + dt, s4)
    return [
        y + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for y, a, b, c, d in zip(state, k1, k2, k3, k4)
    ]

def simulate_damped_oscillator(
    zeta: float = 0.15,
    omega0: float = 2.0 * math.pi,
    x0: float = 1.0,
    v0: float = 0.0,
    dt: float = 0.005,
    t_end: float = 2.0,
) -> list[tuple[float, float, float]]:
    \"\"\"Integrate x'' + 2*zeta*omega0*x' + omega0^2*x = 0 via classical RK4.\"\"\"
    def rhs(_t: float, s: list[float]) -> list[float]:
        x, v = s
        return [v, -2.0 * zeta * omega0 * v - (omega0 ** 2) * x]

    traj: list[tuple[float, float, float]] = [(0.0, x0, v0)]
    t = 0.0
    state = [x0, v0]
    steps = int(round(t_end / dt))
    for _ in range(steps):
        state = rk4_step(rhs, t, state, dt)
        t += dt
        traj.append((round(t, 6), state[0], state[1]))
    return traj

if __name__ == "__main__":
    samples = simulate_damped_oscillator(zeta=0.18, omega0=6.283185, x0=1.0, v0=0.0, dt=0.005, t_end=2.0)
    initial_amp = abs(samples[0][1])
    final_envelope = max(abs(x) for t, x, _ in samples if t >= 1.5)
    assert final_envelope < initial_amp * 0.35, "Amplitude must decay under positive damping"
    print(f"RK4 verified: initial_amplitude={initial_amp:.4f}, tail_amplitude_decay={final_envelope:.4f}, steps={len(samples)-1}")
"""

    # 3. STEM: Spectral FFT / Radix-2 Cooley-Tukey Analyzer
    elif ("fft" in low or "dft" in low or "spectral" in low or "fourier" in low) and ("freq" in low or "signal" in low or "hz" in low or "tone" in low):
        py_title = "Cooley-Tukey Radix-2 FFT Spectral Frequency Analyzer"
        py_code = """import cmath
import math
import random

def fft_radix2(x: list[complex]) -> list[complex]:
    n = len(x)
    if n <= 1:
        return list(x)
    if n & (n - 1) != 0:
        raise ValueError("Input length must be a power of 2")
    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    twiddles = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + twiddles[k] for k in range(n // 2)] + [even[k] - twiddles[k] for k in range(n // 2)]

def detect_dominant_frequencies(samples: list[float], sample_rate_hz: float, top_k: int = 2) -> list[tuple[float, float]]:
    n = 1 << (len(samples) - 1).bit_length()
    if n > len(samples):
        n >>= 1
    # Apply Hann window to suppress spectral leakage
    windowed = [
        complex(samples[i] * (0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))), 0.0)
        for i in range(n)
    ]
    spec = fft_radix2(windowed)
    half = n // 2
    mags = [(k * sample_rate_hz / n, (2.0 / n) * abs(spec[k])) for k in range(1, half)]
    peaks: list[tuple[float, float]] = []
    for idx in range(1, len(mags) - 1):
        if mags[idx][1] > mags[idx - 1][1] and mags[idx][1] >= mags[idx + 1][1]:
            peaks.append((round(mags[idx][0], 2), round(mags[idx][1], 4)))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks[:top_k]

if __name__ == "__main__":
    rng = random.Random(42)
    fs = 1000.0
    n_pts = 1024
    signal = [
        1.2 * math.sin(2.0 * math.pi * 50.0 * (i / fs))
        + 0.85 * math.sin(2.0 * math.pi * 120.0 * (i / fs))
        + rng.gauss(0.0, 0.15)
        for i in range(n_pts)
    ]
    top = detect_dominant_frequencies(signal, sample_rate_hz=fs, top_k=2)
    detected_hz = sorted(round(f) for f, _ in top)
    assert abs(detected_hz[0] - 50) <= 2 and abs(detected_hz[1] - 120) <= 2, f"Unexpected peaks: {top}"
    print(f"FFT dominant frequencies verified: {top} (bins={n_pts}, fs={fs} Hz)")
"""

    # 4. STEM: Monte Carlo Uncertainty Propagation
    elif "monte carlo" in low and ("uncertaint" in low or "power" in low or "gaussian" in low or "confidence" in low or "propagat" in low):
        py_title = "Monte Carlo Gaussian Uncertainty Propagation Engine"
        py_code = """import math
import random
import statistics

def monte_carlo_power_uncertainty(
    v_mean: float = 24.0,
    v_std: float = 0.2,
    r_mean: float = 8.0,
    r_std: float = 0.1,
    n_samples: int = 5000,
    seed: int = 1337,
) -> dict[str, float]:
    \"\"\"Propagate Gaussian component tolerances through P = V^2 / R.\"\"\"
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_samples):
        v = rng.gauss(v_mean, v_std)
        r = max(1e-6, rng.gauss(r_mean, r_std))
        samples.append((v * v) / r)
    samples.sort()
    mean_p = statistics.fmean(samples)
    std_p = statistics.stdev(samples)
    ci_low = samples[int(0.025 * n_samples)]
    ci_high = samples[int(0.975 * n_samples)]
    # First-order analytical Taylor propagation check
    analytic_std = math.hypot((2.0 * v_mean / r_mean) * v_std, (-(v_mean ** 2) / (r_mean ** 2)) * r_std)
    return {
        "mean_watts": round(mean_p, 4),
        "std_watts": round(std_p, 4),
        "analytic_std_watts": round(analytic_std, 4),
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
        "samples": n_samples,
    }

if __name__ == "__main__":
    res = monte_carlo_power_uncertainty()
    assert 71.5 <= res["mean_watts"] <= 72.5, f"Expected ~72.0 W, got {res}"
    print(f"Monte Carlo P=V^2/R: mean={res['mean_watts']} W, std={res['std_watts']} W, 95% CI=[{res['ci_95_low']}, {res['ci_95_high']}] W")
"""

    # 5. STEM: Linear Least-Squares Calibration Curve Fitter
    elif ("least-squares" in low or "least squares" in low or "calibration curve" in low or "linear regression" in low):
        py_title = "Linear Least-Squares Sensor Calibration Fitter ($y = mx + b$, $R^2$, RSE)"
        py_code = """import math

def fit_linear_calibration(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("At least 3 paired (x, y) calibration points are required")
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0.0:
        raise ValueError("Zero variance in x")
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    rse = math.sqrt(ss_res / (n - 2))
    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 6),
        "residual_std_error": round(rse, 6),
    }

if __name__ == "__main__":
    pressures_kpa = [0.0, 25.0, 50.0, 75.0, 100.0, 125.0]
    voltages_v = [0.502, 2.001, 3.497, 5.004, 6.498, 8.003]
    fit = fit_linear_calibration(pressures_kpa, voltages_v)
    assert fit["r_squared"] > 0.9999, f"Expected high linearity, got {fit}"
    print(f"Calibration fit verified: slope={fit['slope']} V/kPa, intercept={fit['intercept']} V, R2={fit['r_squared']}, RSE={fit['residual_std_error']}")
"""

    # 6. STEM: 1D Finite-Difference Heat / Diffusion PDE Solver
    elif ("heat equation" in low or "diffusion" in low or "pde" in low or "finite difference" in low or "finite-difference" in low):
        py_title = "1D Explicit Finite-Difference Heat Diffusion PDE Solver (`FTCS` Stability-Checked)"
        py_code = """def solve_heat_1d(
    alpha: float = 1e-4,
    length_m: float = 1.0,
    nx: int = 21,
    dt: float = 0.1,
    steps: int = 200,
    t_left: float = 100.0,
    t_right: float = 20.0,
    t_init: float = 20.0,
) -> dict[str, object]:
    dx = length_m / (nx - 1)
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(f"FTCS unstable: Fourier number r={r:.4f} exceeds 0.5")
    u = [t_init] * nx
    u[0], u[-1] = t_left, t_right
    for _ in range(steps):
        u_next = list(u)
        for i in range(1, nx - 1):
            u_next[i] = u[i] + r * (u[i - 1] - 2.0 * u[i] + u[i + 1])
        u = u_next
    return {"fourier_r": round(r, 5), "midpoint_temp_c": round(u[nx // 2], 3), "profile": [round(v, 2) for v in u]}

if __name__ == "__main__":
    out = solve_heat_1d(alpha=0.01, length_m=1.0, nx=21, dt=0.08, steps=400)
    assert 55.0 <= float(out["midpoint_temp_c"]) <= 60.5
    print(f"1D Heat PDE verified: fourier_r={out['fourier_r']}, midpoint_temp_c={out['midpoint_temp_c']} C")
"""

    # 7. Wet-Lab / Biotech: FASTA DNA Sequence Analyzer
    elif "fasta" in low or ("dna" in low and any(k in low for k in ("gc", "complement", "codon", "atg", "sequence"))):
        py_title = "Wet-Lab FASTA DNA Sequence Analyzer (`GC%`, Reverse Complement, `ATG` Codons)"
        py_code = """COMPLEMENT_MAP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

def parse_fasta(fasta_text: str) -> tuple[str, str]:
    header = "unnamed"
    seq_parts: list[str] = []
    for line in fasta_text.strip().splitlines():
        ln = line.strip()
        if not ln:
            continue
        if ln.startswith(">"):
            header = ln[1:].strip()
        else:
            seq_parts.append(ln.upper())
    return header, "".join(seq_parts)

def analyze_dna(fasta_or_seq: str) -> dict[str, object]:
    header, seq = parse_fasta(fasta_or_seq) if ">" in fasta_or_seq else ("raw_seq", fasta_or_seq.strip().upper())
    if not seq or any(ch not in "ACGTN" for ch in seq):
        raise ValueError("Invalid IUPAC DNA bases encountered")
    gc_count = seq.count("G") + seq.count("C")
    gc_pct = (100.0 * gc_count / len(seq))
    rev_comp = seq.translate(COMPLEMENT_MAP)[::-1]
    atg_positions = [i for i in range(len(seq) - 2) if seq[i : i + 3] == "ATG"]
    return {
        "header": header,
        "length_bp": len(seq),
        "gc_percent": round(gc_pct, 2),
        "reverse_complement": rev_comp,
        "atg_start_positions": atg_positions,
    }

if __name__ == "__main__":
    sample = ">sp|P001|SyntheticConstruct\\nATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    res = analyze_dna(sample)
    assert res["length_bp"] == 39 and res["atg_start_positions"] == [0, 12]
    print(f"FASTA DNA verified: len={res['length_bp']} bp, GC={res['gc_percent']}%, ATG={res['atg_start_positions']}, rev_comp={res['reverse_complement'][:18]}...")
"""

    # 8. Wet-Lab / Cheminformatics: Molarity Dilution & Molecular Weight Calculator
    elif ("molarity" in low or "dilution" in low or "molecular weight" in low or "c1*v1" in low or "stoichiometr" in low):
        py_title = "Laboratory Molarity Dilution (`C1*V1 = C2*V2`) & Molecular Weight Calculator"
        py_code = """import re

ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.00794, "C": 12.0107, "N": 14.0067, "O": 15.9994,
    "Na": 22.98977, "Mg": 24.305, "P": 30.97376, "S": 32.065,
    "Cl": 35.453, "K": 39.0983, "Ca": 40.078,
}

def molecular_weight(formula: str) -> float:
    tokens = re.findall(r"([A-Z][a-z]?)(\\d*)", formula.strip())
    if not tokens or "".join(el + cnt for el, cnt in tokens) != formula.strip():
        raise ValueError(f"Unsupported chemical formula: {formula}")
    total = 0.0
    for el, count_s in tokens:
        if el not in ATOMIC_WEIGHTS:
            raise ValueError(f"Unknown element: {el}")
        total += ATOMIC_WEIGHTS[el] * (int(count_s) if count_s else 1)
    return round(total, 3)

def compute_dilution_v1(c1_molar: float, c2_molar: float, v2_ml: float) -> dict[str, float]:
    if c1_molar <= 0 or c2_molar <= 0 or v2_ml <= 0 or c2_molar > c1_molar:
        raise ValueError("Require 0 < c2_molar <= c1_molar and v2_ml > 0")
    v1_ml = (c2_molar * v2_ml) / c1_molar
    return {
        "stock_volume_v1_ml": round(v1_ml, 4),
        "diluent_volume_ml": round(v2_ml - v1_ml, 4),
        "final_volume_v2_ml": round(v2_ml, 4),
    }

if __name__ == "__main__":
    mw_h2o = molecular_weight("H2O")
    mw_gluc = molecular_weight("C6H12O6")
    mw_nacl = molecular_weight("NaCl")
    dil = compute_dilution_v1(c1_molar=1.0, c2_molar=0.15, v2_ml=250.0)
    assert abs(mw_h2o - 18.015) < 0.02 and abs(mw_gluc - 180.156) < 0.05 and abs(mw_nacl - 58.443) < 0.02
    print(f"MW verified: H2O={mw_h2o} g/mol, C6H12O6={mw_gluc} g/mol, NaCl={mw_nacl} g/mol | dilution V1={dil['stock_volume_v1_ml']} mL")
"""

    # 9. Wet-Lab / LIMS: High-Throughput Assay Z'-Factor QC
    elif ("z-prime" in low or "z_prime" in low or "z' factor" in low or "lims" in low or "assay qc" in low):
        py_title = "LIMS High-Throughput Screening `Z'-Factor` Assay QC Module"
        py_code = """import statistics

def compute_z_prime(pos_controls: list[float], neg_controls: list[float]) -> dict[str, object]:
    if len(pos_controls) < 2 or len(neg_controls) < 2:
        raise ValueError("Need >= 2 replicates each for positive and negative controls")
    mu_p = statistics.fmean(pos_controls)
    mu_n = statistics.fmean(neg_controls)
    sd_p = statistics.stdev(pos_controls)
    sd_n = statistics.stdev(neg_controls)
    dynamic_range = abs(mu_p - mu_n)
    if dynamic_range == 0.0:
        raise ValueError("Zero assay dynamic range between controls")
    z_prime = 1.0 - (3.0 * (sd_p + sd_n) / dynamic_range)
    if z_prime >= 0.5:
        grade = "EXCELLENT_ASSAY (Z' >= 0.5)"
    elif z_prime > 0.0:
        grade = "MARGINAL_ASSAY (0 < Z' < 0.5)"
    else:
        grade = "REJECT_PLATE (Z' <= 0)"
    return {
        "z_prime": round(z_prime, 4),
        "classification": grade,
        "mu_pos": round(mu_p, 2),
        "mu_neg": round(mu_n, 2),
        "cv_pos_pct": round(100.0 * sd_p / abs(mu_p), 2) if mu_p else 0.0,
    }

if __name__ == "__main__":
    pos = [9850.0, 10120.0, 9940.0, 10080.0, 9990.0, 10040.0]
    neg = [210.0, 195.0, 225.0, 205.0, 198.0, 215.0]
    qc = compute_z_prime(pos, neg)
    assert float(qc["z_prime"]) > 0.5, f"Expected Z' >= 0.5, got {qc}"
    print(f"LIMS Plate QC verified: z_prime={qc['z_prime']} -> {qc['classification']}")
"""

    # 10. Wet-Lab / Biotech: Pharmacokinetic 1-Compartment IV Bolus & Trapezoidal AUC
    elif ("pharmacokinetic" in low or "iv bolus" in low or ("auc" in low and ("half-life" in low or "half_life" in low or "compartment" in low))):
        py_title = "1-Compartment IV Bolus Pharmacokinetics (`C(t)`, `t_half`, Trapezoidal `AUC`)"
        py_code = """import math

def simulate_iv_bolus_pk(
    dose_mg: float = 100.0,
    vd_liters: float = 10.0,
    ke_per_hr: float = 0.25,
    times_hr: list[float] | None = None,
) -> dict[str, object]:
    if dose_mg <= 0 or vd_liters <= 0 or ke_per_hr <= 0:
        raise ValueError("Dose, volume of distribution, and ke must be positive")
    c0 = dose_mg / vd_liters
    t_half = math.log(2.0) / ke_per_hr
    grid = times_hr if times_hr is not None else [0.5 * i for i in range(49)]
    concs = [c0 * math.exp(-ke_per_hr * t) for t in grid]
    # Linear trapezoidal AUC_0_t
    auc_trap = sum(
        0.5 * (concs[i] + concs[i - 1]) * (grid[i] - grid[i - 1])
        for i in range(1, len(grid))
    )
    auc_inf_exact = c0 / ke_per_hr
    return {
        "c0_mg_per_l": round(c0, 4),
        "half_life_hr": round(t_half, 4),
        "auc_0_t_trap": round(auc_trap, 4),
        "auc_0_inf_exact": round(auc_inf_exact, 4),
    }

if __name__ == "__main__":
    pk = simulate_iv_bolus_pk(dose_mg=100.0, vd_liters=10.0, ke_per_hr=0.34657)
    assert abs(float(pk["half_life_hr"]) - 2.0) < 0.01
    print(f"PK model verified: C0={pk['c0_mg_per_l']} mg/L, half_life={pk['half_life_hr']} hr, AUC_trap={pk['auc_0_t_trap']} mg*hr/L")
"""

    # 11. Wet-Lab / Biochemistry: Michaelis-Menten Enzyme Kinetics
    elif "michaelis" in low or ("enzyme" in low and ("km" in low or "vmax" in low or "kinetics" in low)):
        py_title = "Michaelis-Menten Enzyme Kinetics (`Vmax`, `Km` Nonlinear Gauss-Newton Fit)"
        py_code = """def fit_michaelis_menten(substrate_mm: list[float], rates: list[float], iters: int = 40) -> dict[str, float]:
    vmax = max(rates) * 1.15
    km = substrate_mm[len(substrate_mm) // 2]
    for _ in range(iters):
        j00 = j01 = j11 = r0 = r1 = 0.0
        for s, v in zip(substrate_mm, rates):
            denom = km + s
            pred = (vmax * s) / denom
            err = v - pred
            dv_dvmax = s / denom
            dv_dkm = -(vmax * s) / (denom * denom)
            j00 += dv_dvmax * dv_dvmax
            j01 += dv_dvmax * dv_dkm
            j11 += dv_dkm * dv_dkm
            r0 += dv_dvmax * err
            r1 += dv_dkm * err
        det = j00 * j11 - j01 * j01
        if abs(det) < 1e-12:
            break
        vmax += (j11 * r0 - j01 * r1) / det
        km = max(1e-6, km + (-j01 * r0 + j00 * r1) / det)
    return {"vmax": round(vmax, 4), "km": round(km, 4)}

if __name__ == "__main__":
    s_grid = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]
    v_obs = [(120.0 * s) / (4.0 + s) for s in s_grid]
    fit = fit_michaelis_menten(s_grid, v_obs)
    assert abs(fit["vmax"] - 120.0) < 0.05 and abs(fit["km"] - 4.0) < 0.05
    print(f"Michaelis-Menten verified: Vmax={fit['vmax']}, Km={fit['km']}")
"""

    # 12. STEAM: 16-bit PCM Audio Synthesizer + ADSR + Biquad Low-Pass Filter
    elif ("pcm" in low or "biquad" in low or "adsr" in low or ("synthesizer" in low and "audio" in low) or "440 hz" in low):
        py_title = "Pure-Python 16-Bit PCM Audio Synthesizer + ADSR + RBJ Biquad Low-Pass DSP Filter"
        py_code = """import math
import struct

def adsr_envelope(n_samples: int, sr: int, attack: float = 0.02, decay: float = 0.05, sustain: float = 0.75, release: float = 0.08) -> list[float]:
    a_n = max(1, int(attack * sr))
    d_n = max(1, int(decay * sr))
    r_n = max(1, int(release * sr))
    s_n = max(0, n_samples - (a_n + d_n + r_n))
    env: list[float] = []
    for i in range(a_n):
        env.append(i / a_n)
    for i in range(d_n):
        env.append(1.0 - (1.0 - sustain) * (i / d_n))
    env.extend([sustain] * s_n)
    for i in range(r_n):
        env.append(sustain * (1.0 - i / r_n))
    return (env + [0.0] * n_samples)[:n_samples]

def biquad_lowpass(samples: list[float], cutoff_hz: float, sr: int, q: float = 0.7071) -> list[float]:
    w0 = 2.0 * math.pi * cutoff_hz / sr
    alpha = math.sin(w0) / (2.0 * q)
    cos_w0 = math.cos(w0)
    b0 = (1.0 - cos_w0) * 0.5
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) * 0.5
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    out: list[float] = []
    x1 = x2 = y1 = y2 = 0.0
    for x0 in samples:
        y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
        out.append(y0)
    return out

def synth_a4_pcm16(freq_hz: float = 440.0, duration_s: float = 0.35, sr: int = 16000) -> dict[str, object]:
    n = int(duration_s * sr)
    env = adsr_envelope(n, sr)
    raw = [
        (0.8 * math.sin(2.0 * math.pi * freq_hz * i / sr) + 0.2 * math.sin(2.0 * math.pi * 3.0 * freq_hz * i / sr)) * env[i]
        for i in range(n)
    ]
    filtered = biquad_lowpass(raw, cutoff_hz=1200.0, sr=sr)
    pcm16 = [max(-32768, min(32767, int(round(s * 32767)))) for s in filtered]
    pcm_bytes = struct.pack(f"<{len(pcm16)}h", *pcm16)
    peak = max(abs(s) for s in filtered)
    rms = math.sqrt(sum(s * s for s in filtered) / len(filtered))
    return {"freq_hz": freq_hz, "samples": n, "pcm_bytes": len(pcm_bytes), "peak": round(peak, 4), "rms": round(rms, 4)}

if __name__ == "__main__":
    stats = synth_a4_pcm16(440.0)
    assert float(stats["rms"]) > 0.15 and int(stats["pcm_bytes"]) == int(stats["samples"]) * 2
    print(f"Audio DSP verified: freq={stats['freq_hz']} Hz, ADSR+Biquad peak={stats['peak']}, RMS={stats['rms']}, bytes={stats['pcm_bytes']}")
"""

    # 13. STEAM: Procedural SVG Generator (Phyllotaxis or L-System Fractal)
    elif "svg" in low and ("phyllotaxis" in low or "golden" in low or "spiral" in low or "fractal" in low or "l-system" in low or "procedural" in low):
        if "l-system" in low or "fractal" in low or "tree" in low:
            py_title = "Procedural SVG L-System Botanical Fractal Generator"
            py_code = """import math

def generate_lsystem_svg(iterations: int = 4, angle_deg: float = 25.0, step: float = 12.0) -> str:
    axiom = "X"
    rules = {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"}
    seq = axiom
    for _ in range(iterations):
        seq = "".join(rules.get(ch, ch) for ch in seq)
    x, y, heading = 200.0, 380.0, -90.0
    stack: list[tuple[float, float, float]] = []
    lines: list[str] = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">']
    for ch in seq:
        if ch == "F":
            nx = x + step * math.cos(math.radians(heading))
            ny = y + step * math.sin(math.radians(heading))
            lines.append(f'  <line x1="{x:.2f}" y1="{y:.2f}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="#2dd4bf" stroke-width="1.4"/>')
            x, y = nx, ny
        elif ch == "+":
            heading += angle_deg
        elif ch == "-":
            heading -= angle_deg
        elif ch == "[":
            stack.append((x, y, heading))
        elif ch == "]":
            x, y, heading = stack.pop()
    lines.append("</svg>")
    return "\\n".join(lines)

if __name__ == "__main__":
    svg = generate_lsystem_svg(iterations=3, step=8.0)
    assert svg.startswith("<svg") and "<line " in svg and svg.endswith("</svg>")
    print(f"L-System Fractal SVG verified: chars={len(svg)}, segments={svg.count('<line ')}")
"""
        else:
            py_title = "Procedural SVG Golden-Angle (`137.50776°`) Phyllotaxis Spiral Generator"
            py_code = """import math

GOLDEN_ANGLE_DEG = 137.50776405003785

def generate_phyllotaxis_svg(num_seeds: int = 180, scale: float = 6.2, size: int = 400) -> str:
    cx = cy = size / 2.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}">',
        f'  <rect width="{size}" height="{size}" fill="#070b14"/>',
    ]
    for n in range(1, num_seeds + 1):
        theta = math.radians(n * GOLDEN_ANGLE_DEG)
        r = scale * math.sqrt(n)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        radius = 1.8 + 2.2 * (n / num_seeds)
        hue = int((n * 4.5) % 360)
        parts.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="hsl({hue}, 82%, 62%)"/>')
    parts.append("</svg>")
    return "\\n".join(parts)

if __name__ == "__main__":
    svg_doc = generate_phyllotaxis_svg(num_seeds=120)
    assert "<svg" in svg_doc and svg_doc.count("<circle ") == 120 and "</svg>" in svg_doc
    print(f"Golden-angle (137.51 deg) Phyllotaxis SVG verified: circles={svg_doc.count('<circle ')}, bytes={len(svg_doc)}")
"""

    # 14. STEAM: Color Science sRGB -> Linear RGB -> WCAG 2.1 Contrast Accessibility
    elif ("wcag" in low or "contrast ratio" in low or "relative luminance" in low or ("srgb" in low and "linear" in low)):
        py_title = "Color Science sRGB Luminance & WCAG 2.1 Contrast Accessibility Analyzer"
        py_code = """def srgb_channel_to_linear(c_8bit: int) -> float:
    cs = c_8bit / 255.0
    return cs / 12.92 if cs <= 0.04045 else ((cs + 0.055) / 1.055) ** 2.4

def hex_to_relative_luminance(hex_color: str) -> float:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    rl, gl, bl = srgb_channel_to_linear(r), srgb_channel_to_linear(g), srgb_channel_to_linear(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl

def wcag_contrast(fg_hex: str, bg_hex: str) -> dict[str, object]:
    l1 = hex_to_relative_luminance(fg_hex)
    l2 = hex_to_relative_luminance(bg_hex)
    lighter, darker = max(l1, l2), min(l1, l2)
    ratio = (lighter + 0.05) / (darker + 0.05)
    return {
        "ratio": round(ratio, 2),
        "ratio_label": f"{ratio:.1f}:1",
        "wcag_aa_normal": ratio >= 4.5,
        "wcag_aaa_normal": ratio >= 7.0,
    }

if __name__ == "__main__":
    bw = wcag_contrast("#000000", "#FFFFFF")
    low_c = wcag_contrast("#777777", "#999999")
    assert bw["ratio"] == 21.0 and bw["wcag_aaa_normal"] is True and low_c["wcag_aa_normal"] is False
    print(f"WCAG 2.1 contrast verified: black/white={bw['ratio']} ({bw['ratio_label']}, AAA={bw['wcag_aaa_normal']}), low_pair={low_c['ratio']}")
"""

    # 15. $MM SaaS: Thread-Safe Token Bucket Rate Limiter
    elif ("token bucket" in low or "rate limiter" in low or "rate-limiter" in low):
        py_title = "Thread-Safe Multi-Tenant Token Bucket Rate Limiter"
        py_code = """import threading
import time

class MultiTenantTokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        if capacity <= 0 or refill_per_sec <= 0:
            raise ValueError("capacity and refill_per_sec must be positive")
        self.capacity = float(capacity)
        self.refill_per_sec = float(refill_per_sec)
        self._lock = threading.RLock()
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow_request(self, tenant_id: str, cost: float = 1.0, now: float | None = None) -> bool:
        ts = time.monotonic() if now is None else now
        with self._lock:
            tokens, last_ts = self._buckets.get(tenant_id, (self.capacity, ts))
            elapsed = max(0.0, ts - last_ts)
            tokens = min(self.capacity, tokens + elapsed * self.refill_per_sec)
            if tokens >= cost:
                self._buckets[tenant_id] = (tokens - cost, ts)
                return True
            self._buckets[tenant_id] = (tokens, ts)
            return False

if __name__ == "__main__":
    limiter = MultiTenantTokenBucket(capacity=5.0, refill_per_sec=10.0)
    burst = [limiter.allow_request("org_alpha", now=100.0) for _ in range(7)]
    after_refill = limiter.allow_request("org_alpha", now=100.25)
    assert burst == [True, True, True, True, True, False, False] and after_refill is True
    print(f"TokenBucket verified: burst_5_of_7={burst.count(True)}, throttled={burst.count(False)}, refill_allowed={after_refill}")
"""

    # 16. $MM SaaS: Idempotent Webhook Receiver (HMAC-SHA256 + Replay Window + Dedup)
    elif ("webhook" in low and ("hmac" in low or "signature" in low or "idempoten" in low)):
        py_title = "Idempotent Webhook Receiver (`HMAC-SHA256`, Constant-Time `compare_digest`, Replay & Dedup Guard)"
        py_code = """import hashlib
import hmac
import threading
import time

class WebhookVerifier:
    def __init__(self, secret: str, tolerance_sec: int = 300) -> None:
        self._secret = secret.encode("utf-8")
        self._tolerance_sec = tolerance_sec
        self._lock = threading.RLock()
        self._seen_events: dict[str, float] = {}

    def sign(self, timestamp: int, raw_body: bytes) -> str:
        msg = f"{timestamp}.".encode("utf-8") + raw_body
        return "v1=" + hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def verify_and_claim(self, event_id: str, timestamp: int, raw_body: bytes, signature_header: str, now: int | None = None) -> tuple[bool, str]:
        now_ts = int(time.time()) if now is None else now
        if abs(now_ts - timestamp) > self._tolerance_sec:
            return False, "REPLAY_WINDOW_EXPIRED"
        expected = self.sign(timestamp, raw_body)
        if not hmac.compare_digest(expected, signature_header.strip()):
            return False, "INVALID_HMAC_SIGNATURE"
        with self._lock:
            cutoff = now_ts - self._tolerance_sec
            self._seen_events = {k: ts for k, ts in self._seen_events.items() if ts >= cutoff}
            if event_id in self._seen_events:
                return False, "DUPLICATE_EVENT_IDEMPOTENT_SKIP"
            self._seen_events[event_id] = float(now_ts)
        return True, "ACCEPTED"

if __name__ == "__main__":
    verifier = WebhookVerifier(secret="whsec_enterprise_test_key", tolerance_sec=300)
    payload = b'{"event":"invoice.paid","amount_cents":49000}'
    sig = verifier.sign(1_700_000_000, payload)
    ok1, r1 = verifier.verify_and_claim("evt_01", 1_700_000_000, payload, sig, now=1_700_000_050)
    ok_dup, r_dup = verifier.verify_and_claim("evt_01", 1_700_000_000, payload, sig, now=1_700_000_055)
    ok_bad, r_bad = verifier.verify_and_claim("evt_02", 1_700_000_000, payload, "v1=deadbeef", now=1_700_000_050)
    assert ok1 and not ok_dup and not ok_bad
    print(f"Webhook HMAC-SHA256 verified: first={r1}, replay_dup={r_dup}, bad_sig={r_bad}")
"""

    # 17. $MM SaaS: Circuit Breaker (`CLOSED -> OPEN -> HALF_OPEN -> CLOSED`) + Jittered Backoff
    elif "circuit breaker" in low or ("half_open" in low or "half-open" in low):
        py_title = "Resilient Circuit Breaker (`CLOSED -> OPEN -> HALF_OPEN -> CLOSED`) + Full-Jitter Backoff"
        py_code = """import random
import threading
from typing import Callable, TypeVar

T = TypeVar("T")

def backoff_with_full_jitter(attempt: int, base_s: float = 0.05, cap_s: float = 2.0, rng: random.Random | None = None) -> float:
    r = rng or random
    ceiling = min(cap_s, base_s * (2 ** max(0, attempt)))
    return r.uniform(0.0, ceiling)

class CircuitBreakerOpenError(RuntimeError):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout_s: float = 10.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.state = "CLOSED"
        self.failures = 0
        self.opened_at = 0.0
        self._lock = threading.RLock()

    def call(self, fn: Callable[[], T], *, now: float = 0.0) -> T:
        with self._lock:
            if self.state == "OPEN":
                if (now - self.opened_at) >= self.recovery_timeout_s:
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenError("Circuit is OPEN")
        try:
            result = fn()
        except Exception:
            with self._lock:
                self.failures += 1
                if self.state == "HALF_OPEN" or self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    self.opened_at = now
            raise
        with self._lock:
            self.failures = 0
            self.state = "CLOSED"
        return result

if __name__ == "__main__":
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_s=5.0)
    transitions = [cb.state]
    for t in (1.0, 2.0):
        try:
            cb.call(lambda: (_ for _ in ()).throw(ConnectionError("upstream timeout")), now=t)
        except ConnectionError:
            pass
    transitions.append(cb.state)
    # Probe after cooldown enters HALF_OPEN and recovers to CLOSED on success
    probe_state_seen: list[str] = []
    res = cb.call(lambda: (probe_state_seen.append(cb.state) or "OK"), now=8.0)
    transitions.extend([probe_state_seen[0], cb.state])
    assert transitions == ["CLOSED", "OPEN", "HALF_OPEN", "CLOSED"] and res == "OK"
    print(f"CircuitBreaker transitions verified: {' -> '.join(transitions)} | sample_jitter={backoff_with_full_jitter(2, rng=random.Random(7)):.4f}s")
"""

    # 18. $B Enterprise: Immutable Decimal Double-Entry Accounting Ledger with SHA-256 Audit Chain
    elif ("double-entry" in low or "double entry" in low or ("ledger" in low and ("decimal" in low or "debit" in low or "credit" in low or "trial balance" in low))):
        py_title = "Immutable `Decimal` Double-Entry Accounting Ledger with `SHA-256` Hash Chain"
        py_code = """from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json

@dataclass(frozen=True)
class PostingLine:
    account: str
    side: str  # 'DEBIT' or 'CREDIT'
    amount: Decimal

@dataclass(frozen=True)
class JournalEntry:
    entry_id: str
    memo: str
    lines: tuple[PostingLine, ...]
    prev_hash: str
    entry_hash: str

class ImmutableDoubleEntryLedger:
    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    def post(self, entry_id: str, memo: str, lines: list[tuple[str, str, str | Decimal]]) -> JournalEntry:
        if len(lines) < 2:
            raise ValueError("Journal entry requires at least two posting lines")
        parsed = tuple(PostingLine(acct, side.upper(), Decimal(str(amt))) for acct, side, amt in lines)
        if any(p.amount <= Decimal("0") or p.side not in ("DEBIT", "CREDIT") for p in parsed):
            raise ValueError("All line amounts must be positive Decimal with DEBIT/CREDIT side")
        debits = sum((p.amount for p in parsed if p.side == "DEBIT"), Decimal("0.00"))
        credits = sum((p.amount for p in parsed if p.side == "CREDIT"), Decimal("0.00"))
        if debits != credits:
            raise ValueError(f"Unbalanced entry {entry_id}: debits={debits} != credits={credits}")
        prev_hash = self._entries[-1].entry_hash if self._entries else ("0" * 64)
        canonical = json.dumps(
            {"entry_id": entry_id, "memo": memo, "prev_hash": prev_hash,
             "lines": [(p.account, p.side, str(p.amount)) for p in parsed]},
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        je = JournalEntry(entry_id=entry_id, memo=memo, lines=parsed, prev_hash=prev_hash, entry_hash=digest)
        self._entries.append(je)
        return je

    def trial_balance(self) -> dict[str, Decimal]:
        balances: dict[str, Decimal] = {}
        for je in self._entries:
            for p in je.lines:
                delta = p.amount if p.side == "DEBIT" else -p.amount
                balances[p.account] = balances.get(p.account, Decimal("0.00")) + delta
        return balances

    def verify_audit_chain(self) -> bool:
        prev = "0" * 64
        for je in self._entries:
            if je.prev_hash != prev:
                return False
            prev = je.entry_hash
        return True

if __name__ == "__main__":
    ledger = ImmutableDoubleEntryLedger()
    e1 = ledger.post("JE-001", "Enterprise SaaS Annual Contract", [("Cash", "DEBIT", "125000.00"), ("DeferredRevenue", "CREDIT", "125000.00")])
    e2 = ledger.post("JE-002", "Month-1 Revenue Recognition", [("DeferredRevenue", "DEBIT", "10416.67"), ("RecognizedRevenue", "CREDIT", "10416.67")])
    tb = ledger.trial_balance()
    assert sum(tb.values(), Decimal("0.00")) == Decimal("0.00") and ledger.verify_audit_chain()
    print(f"Double-Entry Decimal Ledger verified: net_trial_balance={sum(tb.values())}, entries=2, sha256_head={e2.entry_hash[:16]}...")
"""

    # 19. $B Enterprise: Virtual-Node Consistent Hashing Ring
    elif "consistent hash" in low or ("virtual node" in low and "ring" in low):
        py_title = "Virtual-Node Consistent Hashing Ring for Multi-PC Shard Routing"
        py_code = """import bisect
import hashlib

class ConsistentHashRing:
    def __init__(self, nodes: list[str] | None = None, replicas: int = 128) -> None:
        self.replicas = replicas
        self._keys: list[int] = []
        self._ring: dict[int, str] = {}
        for node in (nodes or []):
            self.add_node(node)

    @staticmethod
    def _hash(key: str) -> int:
        return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")

    def add_node(self, node: str) -> None:
        for v in range(self.replicas):
            h = self._hash(f"{node}#vn{v}")
            if h not in self._ring:
                bisect.insort(self._keys, h)
                self._ring[h] = node

    def remove_node(self, node: str) -> None:
        for v in range(self.replicas):
            h = self._hash(f"{node}#vn{v}")
            if self._ring.pop(h, None) is not None:
                idx = bisect.bisect_left(self._keys, h)
                if idx < len(self._keys) and self._keys[idx] == h:
                    self._keys.pop(idx)

    def get_node(self, item_key: str) -> str:
        if not self._keys:
            raise RuntimeError("ConsistentHashRing is empty")
        h = self._hash(item_key)
        idx = bisect.bisect_left(self._keys, h) % len(self._keys)
        return self._ring[self._keys[idx]]

if __name__ == "__main__":
    nodes = ["pegasus-8090", "pegasus-seed-39105", "desktop-worker-39106", "staging-shard-39096"]
    ring = ConsistentHashRing(nodes, replicas=150)
    keys = [f"wbrain_slot_{i}" for i in range(4000)]
    before = {k: ring.get_node(k) for k in keys}
    ring.remove_node("staging-shard-39096")
    moved = sum(1 for k in keys if ring.get_node(k) != before[k])
    moved_pct = 100.0 * moved / len(keys)
    assert moved_pct < 40.0, f"Too many keys remapped: {moved_pct:.2f}%"
    print(f"ConsistentHashRing verified: replicas=150, keys_moved_on_1_of_4_removal={moved_pct:.2f}% (< 40%)")
"""

    # 20. $B Enterprise: Quantitative Risk & Execution Engine (VaR, CVaR, Drawdown, Sharpe, VWAP)
    elif any(k in low for k in ("cvar", "expected shortfall", "vwap", "historical var")) or ("sharpe" in low and "drawdown" in low):
        py_title = "Enterprise Quantitative Risk & Execution Engine (`VaR`, `CVaR`, Max Drawdown, `Sharpe`, `VWAP`)"
        py_code = """import math
import statistics

def compute_portfolio_risk(returns: list[float], risk_free_daily: float = 0.0) -> dict[str, float]:
    if len(returns) < 10:
        raise ValueError("Require at least 10 return observations")
    ordered = sorted(returns)
    n = len(ordered)
    idx_95 = max(1, int(math.floor(0.05 * n)))
    idx_99 = max(1, int(math.floor(0.01 * n)))
    var_95 = -ordered[idx_95 - 1]
    var_99 = -ordered[idx_99 - 1]
    cvar_95 = -statistics.fmean(ordered[:idx_95])

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)

    excess = [r - risk_free_daily for r in returns]
    sd = statistics.stdev(excess)
    sharpe_annual = (statistics.fmean(excess) / sd) * math.sqrt(252.0) if sd > 0 else 0.0
    return {
        "var_95": round(var_95, 5),
        "var_99": round(var_99, 5),
        "cvar_95_shortfall": round(cvar_95, 5),
        "max_drawdown": round(max_dd, 5),
        "annualized_sharpe": round(sharpe_annual, 3),
    }

def compute_vwap_schedule(prices: list[float], volumes: list[float], parent_qty: float) -> dict[str, object]:
    tot_vol = sum(volumes)
    vwap = sum(p * v for p, v in zip(prices, volumes)) / tot_vol
    child_slices = [round(parent_qty * (v / tot_vol), 2) for v in volumes]
    return {"vwap": round(vwap, 4), "child_slices": child_slices}

if __name__ == "__main__":
    rets = [0.008, -0.012, 0.005, 0.011, -0.024, 0.004, 0.009, -0.006, 0.007, -0.031,
            0.013, 0.006, -0.004, 0.010, 0.003, -0.015, 0.012, 0.005, -0.008, 0.009] * 5
    risk = compute_portfolio_risk(rets)
    exec_vwap = compute_vwap_schedule([101.2, 101.5, 101.1, 101.8], [4000, 7500, 5000, 3500], parent_qty=1000.0)
    assert risk["cvar_95_shortfall"] >= risk["var_95"] > 0 and exec_vwap["vwap"] > 100.0
    print(f"Quant Risk & VWAP verified: VaR95={risk['var_95']}, VaR99={risk['var_99']}, CVaR95={risk['cvar_95_shortfall']}, MaxDD={risk['max_drawdown']}, Sharpe={risk['annualized_sharpe']}, VWAP={exec_vwap['vwap']}")
"""

    # 21. Multi-Turn / Core Infra: Thread-Safe LRU Cache with TTL Expiration
    elif ("lru" in low and ("ttl" in low or "expir" in low or "cache" in low)):
        py_title = "Thread-Safe `LRU + TTL` Cache (`OrderedDict` + `RLock`)"
        py_code = """from collections import OrderedDict
import threading
import time
from typing import Generic, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")

class ThreadSafeLRUTTLCache(Generic[K, V]):
    def __init__(self, maxsize: int = 256, ttl_seconds: float = 60.0) -> None:
        if maxsize <= 0 or ttl_seconds <= 0:
            raise ValueError("maxsize and ttl_seconds must be positive")
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._store: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: K, *, now: Optional[float] = None) -> Optional[V]:
        ts = time.monotonic() if now is None else now
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, expires_at = entry
            if ts >= expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: K, value: V, *, ttl: Optional[float] = None, now: Optional[float] = None) -> None:
        ts = time.monotonic() if now is None else now
        exp = ts + (self.ttl_seconds if ttl is None else ttl)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, exp)
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

if __name__ == "__main__":
    cache: ThreadSafeLRUTTLCache[str, int] = ThreadSafeLRUTTLCache(maxsize=2, ttl_seconds=10.0)
    cache.put("a", 1, now=0.0)
    cache.put("b", 2, now=1.0)
    assert cache.get("a", now=2.0) == 1  # refreshes 'a', making 'b' LRU
    cache.put("c", 3, now=3.0)           # evicts 'b'
    assert cache.get("b", now=4.0) is None
    assert cache.get("a", now=12.0) is None  # 'a' expired at t=10.0
    print(f"ThreadSafeLRUTTLCache verified: LRU eviction + TTL expiration passed (hits={cache.hits}, misses={cache.misses})")
"""

    # 22. $B Enterprise: Saga Distributed Transaction Coordinator with Compensating Rollbacks
    elif "saga" in low and ("compensat" in low or "rollback" in low or "transaction" in low):
        py_title = "Enterprise Saga Coordinator with Automatic Reverse Compensating Rollbacks"
        py_code = """from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class SagaStep:
    name: str
    action: Callable[[dict], None]
    compensate: Callable[[dict], None]

class SagaCoordinator:
    def __init__(self, steps: list[SagaStep]) -> None:
        self.steps = list(steps)

    def execute(self, context: dict) -> dict[str, object]:
        completed: list[SagaStep] = []
        compensated: list[str] = []
        try:
            for step in self.steps:
                step.action(context)
                completed.append(step)
            return {"ok": True, "completed": [s.name for s in completed], "compensated": []}
        except Exception as exc:
            for step in reversed(completed):
                step.compensate(context)
                compensated.append(step.name)
            return {"ok": False, "error": str(exc), "completed": [s.name for s in completed], "compensated": compensated}

if __name__ == "__main__":
    state = {"reserved": 0, "charged": 0}
    saga = SagaCoordinator([
        SagaStep("reserve_inventory", lambda c: c.update(reserved=1), lambda c: c.update(reserved=0)),
        SagaStep("charge_card", lambda c: c.update(charged=100), lambda c: c.update(charged=0)),
        SagaStep("dispatch_shipment", lambda _c: (_ for _ in ()).throw(RuntimeError("carrier_offline")), lambda _c: None),
    ])
    res = saga.execute(state)
    assert not res["ok"] and res["compensated"] == ["charge_card", "reserve_inventory"] and state == {"reserved": 0, "charged": 0}
    print(f"Saga compensation verified: rollback_order={res['compensated']}, final_state={state}")
"""

    # 23. $MM / $B Compliance: PII Redaction & SOC2/GDPR Structured Audit Sanitizer
    elif ("pii" in low or "redact" in low or "gdpr" in low or "soc2" in low) and ("sanitize" in low or "redact" in low or "log" in low or "scrub" in low):
        py_title = "SOC2 / GDPR PII Log Scrubber (Emails, SSNs, Luhn Credit Cards, Bearer Tokens)"
        py_code = """import re

EMAIL_RE = re.compile(r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b")
SSN_RE = re.compile(r"\\b\\d{3}-\\d{2}-\\d{4}\\b")
BEARER_RE = re.compile(r"(Bearer\\s+)[A-Za-z0-9._\\-~+/]+=*", re.I)
CARD_CAND_RE = re.compile(r"\\b(?:\\d[ -]*?){13,19}\\b")

def _luhn_valid(digits: str) -> bool:
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for idx, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if idx % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0

def scrub_pii(text: str) -> str:
    out = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = SSN_RE.sub("[REDACTED_SSN]", out)
    out = BEARER_RE.sub(r"\\1[REDACTED_TOKEN]", out)
    def _card_sub(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits = re.sub(r"\\D", "", raw)
        return f"[REDACTED_PAN_LAST4:{digits[-4:]}]" if _luhn_valid(digits) else raw
    return CARD_CAND_RE.sub(_card_sub, out)

if __name__ == "__main__":
    sample = "User jane.doe@corp.io SSN 123-45-6789 card 4532-0151-1283-0366 auth Bearer eyJhbGciOiJIUzI1NiJ9.secret"
    clean = scrub_pii(sample)
    assert "jane.doe" not in clean and "123-45-6789" not in clean and "REDACTED_PAN_LAST4:0366" in clean
    print(f"PII scrubber verified: {clean}")
"""

    if py_code:
        ast.parse(py_code)
        return ("python", py_title, py_code)

    return None


class EnterpriseStemSynthTool(Tool):
    name = "enterprise_stem_synth"
    description = (
        "Synthesize AST-verified, runnable STEM, Wet-Lab/LIMS/Biotech, STEAM (Audio DSP, "
        "SVG, WebGL/GLSL shaders, WCAG color science), $MM SaaS, and $B Enterprise modules."
    )
    use_when = (
        "Use when asked to build, implement, or verify scientific, laboratory, creative-tech, "
        "SaaS infrastructure, financial ledger, or distributed systems software."
    )
    params_schema = {"prompt": "str — natural engineering specification"}

    def execute(self, params: dict) -> dict:
        hit = synthesize_domain_software(str(params.get("prompt") or ""))
        if not hit:
            return {"ok": False}
        lang, title, code = hit
        return {"ok": True, "language": lang, "title": title, "code": code}


class ClusterOpsProbeTool(Tool):
    name = "wizard_cluster_probe"
    description = (
        "Probe local and multi-PC LAN W1z4rD cluster endpoints (127.0.0.1:8090, "
        "10.0.0.5:39106, 10.0.0.2:39105, 127.0.0.1:39096) and host memory status."
    )
    use_when = "Use when asked for cluster health, reachable LAN nodes, or multi-PC memory status."
    params_schema = {"timeout": "float — optional per-node HTTP timeout in seconds (default 0.45)"}

    def execute(self, params: dict) -> dict:
        snap = probe_wizard_cluster_live(timeout=float(params.get("timeout") or 0.45))
        return {"ok": True, **snap, "markdown": format_cluster_status_markdown(snap)}


class CausalMechanicsTool(Tool):
    name = "causal_mechanics_reinventor"
    description = (
        "Deduce untrained physical/chemical/computational/biological mechanics from "
        "constituents x, enabling conditions {z,r,q,s,t}, and co-acting process p, "
        "reinvent foundational technologies from raw primitives, and align with W1z4rD goals."
    )
    use_when = (
        "Use when evaluating novel/untrained scenarios, counterfactual missing preconditions, "
        "multi-law compositions, or reinventing devices from scratch."
    )
    params_schema = {"scenario": "str — untrained situation or reinvention challenge"}

    def execute(self, params: dict) -> dict:
        from tools.wizard_learning.civilizational_causal_engine import resolve_causal_discovery_or_untrained_mechanics
        hit = resolve_causal_discovery_or_untrained_mechanics(str(params.get("scenario") or ""))
        if not hit:
            return {"ok": False}
        return {"ok": True, **hit}
