#!/usr/bin/env python3
"""
OPA/Conftest policy check wrapper.
If policies exist under policies/opa and conftest is available, run `conftest test`.
Fail hard when OPA_REQUIRED=1 and conftest/policies are missing.
"""
from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    required = os.getenv("OPA_REQUIRED", "0").lower() in {"1", "true", "yes", "on"}
    branch = os.getenv("GITHUB_REF_NAME") or os.getenv("BRANCH_NAME") or ""
    warn_only = branch and branch != "main" and not required
    policy_dir = Path("policies/opa")
    has_policies = policy_dir.exists() and any(policy_dir.glob("**/*.rego"))
    conftest_path = shutil.which("conftest")

    if not has_policies:
        msg = "No OPA policies found under policies/opa; skipping."
        print(msg)
        return 1 if required else 0
    if not conftest_path:
        msg = "conftest not installed; cannot run OPA checks."
        print(msg)
        return 1 if required else 0

    # Build synthetic input for conftest (release checklist, SLO, ADR, risk)
    input_data = {
        "release_checklist": Path("docs/templates/release_readiness_checklist.md").exists(),
        "sli_slo": Path("docs/templates/sli_slo_template.yaml").exists(),
        "risk_register": any(Path("docs/risk").glob("*.yaml")),
        "adr": any(Path("docs/templates").glob("adr_template.md")),
    }
    input_path = Path("runtime/branddozer/opa_input.json")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(input_data), encoding="utf-8")

    cmd = [conftest_path, "test", str(policy_dir), "--input", str(input_path)]
    print("Running:", " ".join(cmd), f"(warn_only={warn_only})")
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0 and warn_only:
        print("OPA check failed, but warn-only mode is enabled on this branch.")
        return 0
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
