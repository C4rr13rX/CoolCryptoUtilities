# OPA/Conftest Policies
- CI runs `conftest test policies/opa` (set OPA_REQUIRED=1 to enforce).
- `release_policy.rego` enforces:
  - deny if release checklist is missing
  - deny if SLI/SLO is missing
  - warn if risk register is missing
  - warn if ADR is missing

# Running locally
#   conftest test policies/opa
