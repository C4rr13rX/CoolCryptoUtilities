#!/usr/bin/env python3
"""
Generate a UX audit summary artifact from screenshots and gate results.
Inputs (env):
  RUN_ID (required): Delivery run UUID
  OUTPUT (optional): path to write the report (default: runtime/branddozer/reports/ux_audit_<run>.md)
Behavior:
  - Lists key screenshots (latest UI artifacts)
  - Summarizes GateRun statuses related to UI/UX
  - Includes a Design QA checklist section to be filled/validated
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
from services.env_loader import EnvLoader

EnvLoader.load()
import django

django.setup()

from branddozer.models import DeliveryArtifact, GateRun, DeliveryRun  # noqa: E402


CHECKLIST = [
    "Mobile-first layout verified (no overflow/hidden controls).",
    "Responsive breakpoints render cleanly (mobile/tablet/desktop).",
    "Typography scale consistent (no ad-hoc font sizes).",
    "Color contrast meets WCAG AA; uses design tokens.",
    "Touch targets ≥ 44px; spacing rhythm consistent.",
    "Critical CTAs visible above the fold; hierarchy clear.",
    "Motion/animation within budget; respects reduced-motion.",
    "No blocking errors in console/network during flows.",
    "Key funnel (landing→CTA/form) succeeds on mobile and desktop.",
]


def latest_ui_screenshots(run_id: str, limit: int = 12) -> List[str]:
    qs = (
        DeliveryArtifact.objects.filter(run_id=run_id, kind="ui_screenshot")
        .order_by("-created_at")
        .values_list("path", flat=True)
    )
    return list(qs[:limit])


def gate_summaries(run_id: str) -> List[str]:
    summaries = []
    seen = set()
    for gate in GateRun.objects.filter(run_id=run_id).order_by("name", "-created_at"):
        if gate.name in seen:
            continue
        seen.add(gate.name)
        summaries.append(f"{gate.name}: {gate.status} (exit_code={gate.exit_code})")
    return summaries


def build_report(run_id: str) -> str:
    run = DeliveryRun.objects.filter(id=run_id).first()
    header = [
        f"# UX Audit Report for Run {run_id}",
        f"Status: {run.status if run else 'unknown'}",
        "",
    ]
    shots = latest_ui_screenshots(run_id)
    shot_lines = ["## Screenshots", *(f"- {p}" for p in shots)] if shots else ["## Screenshots", "- None found."]
    gate_lines = ["## UI/UX Gates", *(f"- {g}" for g in gate_summaries(run_id) or ["- None found."])]
    checklist_lines = ["## Design QA Checklist"] + [f"- [ ] {item}" for item in CHECKLIST]
    return "\n".join(header + shot_lines + [""] + gate_lines + [""] + checklist_lines + [""])


def main() -> int:
    run_id = os.getenv("RUN_ID")
    if not run_id:
        print("RUN_ID is required")
        return 1
    output = Path(os.getenv("OUTPUT") or f"runtime/branddozer/reports/ux_audit_{run_id}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(run_id)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote UX audit report to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
