#!/usr/bin/env python3
"""
Canary/blue-green readiness check.
- Looks for deployment plans/manifests to ensure progressive delivery is configured.
- Set CANARY_REQUIRED=1 to fail if missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    required = os.getenv("CANARY_REQUIRED", "0").lower() in {"1", "true", "yes", "on"}
    candidates = [
        Path("deploy/canary.yaml"),
        Path("deploy/bluegreen.yaml"),
        Path("deploy/canary.json"),
        Path("deploy/bluegreen.json"),
        Path("infra/canary"),
        Path("infra/bluegreen"),
    ]
    found = any(p.exists() for p in candidates)
    if found:
        print("Found canary/blue-green plan or manifest.")
        return 0
    if required:
        print("Canary/blue-green required but no plan/manifest found. Add deploy/canary.yaml or deploy/bluegreen.yaml.")
        return 1
    print("No canary/blue-green plan found; set CANARY_REQUIRED=1 to enforce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
