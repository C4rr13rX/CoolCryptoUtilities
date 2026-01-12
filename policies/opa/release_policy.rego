package main

default allow = true

# Helpers to read input when conftest is run against the repo root.
release_checklist = input.release_checklist
sli_slo = input.sli_slo
risk_register = input.risk_register
adr = input.adr

deny[msg] {
  not release_checklist
  msg := "release readiness checklist missing"
}

deny[msg] {
  not sli_slo
  msg := "SLI/SLO definition missing"
}

warn[msg] {
  not risk_register
  msg := "risk register missing for high-uncertainty work"
}

warn[msg] {
  not adr
  msg := "ADR missing for significant architectural changes"
}
