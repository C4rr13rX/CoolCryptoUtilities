# Paved Road CI (Template)

This workflow (`.github/workflows/paved-road.yml`) encodes the trunk-based, evidence-first checks:

1) **Fast tests:** `pytest -q --maxfail=1 --disable-warnings`  
2) **Lint/format:** `ruff check`, `ruff format --check`  
3) **Security:** `bandit`, `pip-audit`, lightweight secret scan  
4) **Reproducible build:** `python -m build`  
5) **Evidence:** `python web/manage.py dora_report` → JSON artifacts under `runtime/branddozer/reports`
6) **Reliability gate:** `python scripts/reliability_gate.py` (defaults: CFR<=0.25, MTTR<=6h, DeployFreq>=0.1/day; override with env vars; set `RELIABILITY_STRICT=1` to fail on missing data)
7) **Progressive delivery check:** `python scripts/progressive_delivery_check.py` (fails if missing a feature-flag plan/config; `PROGRESSIVE_REQUIRED=1` by default)
8) **Canary/Blue-Green readiness:** `python scripts/canary_bluegreen_check.py` (strict by default)
9) **Policy-as-code (OPA):** `python scripts/opa_policy_check.py` (warn on PR, deny on main; strict if `OPA_REQUIRED=1`; enforces release checklist + SLO; warns on risk/ADR)
10) **UX audit artifact:** `python scripts/ux_audit_report.py` (writes summary to `runtime/branddozer/reports`)
11) **Conversion path check:** `python scripts/conversion_path_check.py` (Playwright smoke on mobile/desktop viewports; warn-only by default)

How to adopt:
- Keep branches short-lived; enable this workflow on PRs and pushes to `main`.  
- Adjust timeouts/paths for your services; add contract/E2E stages as needed.  
- Defaults are strict (RELIABILITY_STRICT=1, PROGRESSIVE_REQUIRED=1, CANARY_REQUIRED=1, OPA_REQUIRED=1); override via repo/org vars if you need a softer mode on experimental branches.  
- Platform team can fork this into paved-road variants (e.g., with feature-flag rollouts, canary deploys, OPA policy checks).
