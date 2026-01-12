# Release Readiness Checklist (DoD for prod changes)
- Tests: unit, contract, integration, and E2E (where applicable) passed.
- Security: SAST/SCA/secret scan clean or risk-accepted; vulns linked to tickets.
- Observability: logs/metrics/traces added; dashboards/alerts updated.
- Runbook: updated with new behavior/rollback steps.
- Feature flags: defaulted safe/off; rollout plan defined (canary/blue-green).
- Data/infra migrations: idempotent, reversible, rehearsed in staging.
- SLO impact: budgets healthy; reliability gate passed; blast radius assessed.
- Change risk: risk score recorded; approvals match tier.
- Backout: tested rollback/flag disable and documented verification steps.
