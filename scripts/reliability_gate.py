#!/usr/bin/env python3
"""
Reliability gate: fail fast when change-failure rate or MTTR exceed thresholds.
Uses the latest DORA snapshot under runtime/branddozer/reports if available;
falls back to generating a fresh snapshot if Django is accessible.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default


MAX_CFR = _env_float("RELIABILITY_MAX_CHANGE_FAILURE_RATE", 0.25)
MAX_MTTR_HOURS = _env_float("RELIABILITY_MAX_MTTR_HOURS", 6.0)
MIN_DEPLOY_FREQ = _env_float("RELIABILITY_MIN_DEPLOY_FREQ", 0.1)  # per day


def _load_latest_report(report_dir: Path) -> dict | None:
    if not report_dir.exists():
        return None
    files = sorted(report_dir.glob("dora_*.json"), reverse=True)
    for path in files:
        try:
            return json.loads(path.read_text())
        except Exception:
            continue
    return None


def _generate_snapshot() -> dict | None:
    try:
        from services.dora_metrics import generate_snapshot

        snap = generate_snapshot()
        return snap.to_dict()
    except Exception:
        return None


def main() -> int:
    report_dir = Path("runtime/branddozer/reports")
    snapshot = _load_latest_report(report_dir) or _generate_snapshot()
    if not snapshot:
        print("WARNING: No DORA snapshot available; skipping reliability gate (set RELIABILITY_STRICT=1 to fail).")
        if os.getenv("RELIABILITY_STRICT", "0") in {"1", "true", "yes", "on"}:
            return 1
        return 0

    cfr = float(snapshot.get("change_failure_rate") or 0.0)
    mttr = float(snapshot.get("mttr_hours") or 0.0)
    deploy_freq = float(snapshot.get("deployment_frequency") or 0.0)

    failures = []
    if cfr > MAX_CFR:
        failures.append(f"change_failure_rate {cfr} > {MAX_CFR}")
    if mttr > MAX_MTTR_HOURS:
        failures.append(f"mttr_hours {mttr} > {MAX_MTTR_HOURS}")
    if deploy_freq < MIN_DEPLOY_FREQ:
        failures.append(f"deployment_frequency {deploy_freq} < {MIN_DEPLOY_FREQ}")

    if failures:
        print("Reliability gate FAILED:")
        for item in failures:
            print(f" - {item}")
        return 1

    print(
        "Reliability gate passed: "
        f"CFR={cfr}, MTTR={mttr}h, DeployFreq={deploy_freq}/day "
        f"(thresholds CFR<={MAX_CFR}, MTTR<={MAX_MTTR_HOURS}h, DeployFreq>={MIN_DEPLOY_FREQ})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
