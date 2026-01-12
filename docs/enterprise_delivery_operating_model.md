# Enterprise Delivery Operating Model

This document captures how BrandDozer evolves from the current Scrum-focused automation loop to a Small Batch + Evidence + Reliability system. Every pillar is mapped to concrete assets so teams can implement the new model immediately.

## 1. DORA Metrics + Automated Reporting

- **Instrumentation:** `services/dora_metrics.py` calculates deployment frequency, lead time, change-failure rate, and MTTR over rolling windows using DeliveryRun history.  
- **Automation:** `python web/manage.py dora_report --window-days 7` writes structured JSON under `runtime/branddozer/reports/`. Wire this command into CI (daily) and dashboards to trend metrics.
- **Dashboards:** Publish the JSON to Grafana/Looker. Key views: rolling averages, percentile lead times, MTTR distribution, commit-to-prod funnels.

## 2. Trunk-Based Development & CI Policy

- Default branch is always deployable; short-lived branches (<48h) only. Enforce via repository settings and new policy templates.
- Mandatory PR checks:  
  1. Fast unit tests (`pytest -q --maxfail=1 --disable-warnings`).  
  2. Contract/API tests (e.g., `tests/contracts/**`).  
  3. `ruff check` + `ruff format --check`.  
  4. Security scans (`bandit`, `pip-audit`, secret scan).  
  5. Reproducible build (`python -m build` or Docker image hash).  
- Provide GitHub workflow templates in the paved-road repo (platform team owns; see §6).

## 3. Continuous Delivery & Progressive Controls

- **Feature-flag SDK:** All new user-facing changes go behind flags (`services/branddozer_state` already stores project context; extend to `feature_flags`).
- **Progressive rollouts:**  
  - Canary stage: deploy 5% traffic, monitor golden signals.  
  - Blue/Green: maintain two environments with infra-as-code (Terraform module per environment).  
  - Automated rollback: pipelines monitor SLOs and revert commit/flag when budgets burn faster than thresholds.  
- **Environment parity:** single source Terraform (networking, app, observability). Platform team owns modules; teams inherit via templates.

## 4. SRE Primitives & Release Guardrails

- Define SLIs/SLOs per service (latency, availability, saturation, quality). Store them as code (e.g., `ops/sli_slo.yaml`) and expose through Grafana.
- Error budgets = (1 - SLO) minutes/month. Release automation (pipeline stage `reliability_gate`) checks budgets via API before promoting to prod; if exhausted, pipeline enforces “reliability freeze.”
- Reliability work board: backlog items tagged `reliability` auto-prioritized until budgets recover.

## 5. Observability & Incident Workflow

- **Telemetry:**  
  - Structured logs (JSON) with trace/span ids (`services/logging_utils`).  
  - Metrics (Prometheus/OpenTelemetry) for golden signals.  
  - Traces (OTLP) for every deployment increment.  
- **Alerting hygiene:** dedupe rules, rate limits, ownership labels, suppress during maintenance windows.
- **Incident lifecycle:**  
  - Runbooks stored in repo (`docs/runbooks/<service>.md`).  
  - On-call rotations codified in `ops/oncall.yaml`.  
  - Postmortem template (Blameless) with automatic action-item tracking (tag `postmortem-action`).  
  - Change risk register: each change description includes risk score + mitigation; approvals tied to risk tier.

## 6. Team Topologies & Platform Enablement

- Introduce a **Platform Team** responsible for golden paths: repo scaffolds, CI/CD workflows, deployment pipelines, secrets handling, policy-as-code (OPA/Conftest), Terraform modules, monitoring bootstraps.
- Stream-aligned teams own their services end-to-end (build + run). Platform artifacts are opt-out (default safe).  
- Complicated-subsystem / Enabling teams only exist for specialized domains (e.g., ML ops, compliance). Charter documents live in `docs/team_topologies/`.

## 7. Risk-Driven Iteration Gates

- For high-uncertainty efforts, adopt Spiral / Incremental Commitment:  
  - Maintain `risk_register.yaml` per initiative: probability, impact, mitigation owner.  
  - Hypothesis tests documented before coding (link to experiments).  
  - Architecture Decision Records (ADRs) committed for significant choices.  
  - “Go/No-Go” checkpoints: gate by evidence (prototypes, test data, customer interviews). Failure => revisit plan before spending more budget.

## 8. Scrum + Flow Controls

- Scrum ceremonies remain for planning, but boards now enforce:  
  - WIP limits per column and per engineer.  
  - Cycle-time tracking (start-to-finish) powering Monte Carlo forecasts.  
  - Definition of Done = code + tests + security review + observability hooks + runbook updates + rollout plan.  
- Pipeline policies check for missing DoD items (e.g., no runbook update => fail).

## 9. Automation & Templates

Everything above is codified as versioned templates:

| Artifact | Location | Owner | Purpose |
| --- | --- | --- | --- |
| `services/dora_metrics.py`, `dora_report` command | Platform Team | Automated evidence |
| CI workflow templates | `.github/workflows/paved-road/` | Platform | Standard checks |
| Terraform modules | `infra/modules/*` | Platform | Environment parity |
| Policy-as-code | `policies/opa/*.rego` | Platform | Default-safe settings (release checklist, SLO, ADR, risk) |
| Risk templates | `docs/risk/*.md` | Stream-aligned teams | Evidence gating |
| SLI/SLO + Error Budgets | `docs/templates/sli_slo_template.yaml` | Platform/Teams | Reliability gates |
| On-call & Incident | `docs/templates/oncall_template.yaml`, `docs/runbooks/postmortem_template.md` | Platform/Teams | Incident hygiene |
| ADRs & Release DoD | `docs/templates/adr_template.md`, `docs/templates/release_readiness_checklist.md` | Teams | Evidence + operability |
| Progressive Delivery | `scripts/progressive_delivery_check.py`, `scripts/canary_bluegreen_check.py` | Platform/Teams | Safe rollouts |
| Policy Enforcement | `scripts/opa_policy_check.py` | Platform | Policy-as-code guardrail |
| UX Audit Report | `scripts/ux_audit_report.py` | Platform/Teams | UI/UX evidence + checklist |

Adopting this operating model ensures BrandDozer runs on short, evidence-backed iterations with clear metrics, reliable pipelines, and safety nets that prevent the late-stage “black-swan” failures that plague ad-hoc processes.
