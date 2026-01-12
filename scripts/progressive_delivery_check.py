#!/usr/bin/env python3
"""
Progressive delivery sanity check.
- If PROGRESSIVE_REQUIRED=1, fail when no feature-flag plan/config is found.
- Otherwise, emit guidance without failing the build.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    required = os.getenv("PROGRESSIVE_REQUIRED", "0").lower() in {"1", "true", "yes", "on"}
    possible_locations = [
        Path("config/feature_flags.yaml"),
        Path("config/feature_flags.json"),
        Path("feature_flags.yaml"),
        Path("feature_flags.json"),
        Path("infra/feature_flags/"),
    ]
    found = any(p.exists() for p in possible_locations)
    if found:
        print("Found feature-flag config/plan.")
        return 0
    if required:
        print("Progressive delivery required but no feature-flag plan/config found. Add config/feature_flags.{yaml|json}.")
        return 1
    print(
        "Progressive delivery required but no feature-flag plan/config found. Add config/feature_flags.{yaml|json}."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
