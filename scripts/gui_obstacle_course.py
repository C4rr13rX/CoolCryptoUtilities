#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from services import vm_lab


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GUI obstacle course steps via vm_lab.")
    parser.add_argument(
        "--plan",
        default="runtime/c0d3r/gui_obstacle_course.json",
        help="Path to obstacle course JSON (default: runtime/c0d3r/gui_obstacle_course.json)",
    )
    parser.add_argument("--autopilot", action="store_true", help="Bootstrap VM before running steps.")
    args = parser.parse_args()
    plan_path = Path(args.plan).expanduser()
    if not plan_path.exists():
        print(f"[gui-obstacle] plan not found: {plan_path}")
        return 2
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[gui-obstacle] failed to parse plan: {exc}")
        return 2
    steps = payload.get("steps") or []
    if not isinstance(steps, list):
        print("[gui-obstacle] invalid plan: steps must be a list")
        return 2
    if args.autopilot:
        vm_name = payload.get("vm") or "c0d3r-ubuntu"
        image_id = payload.get("image_id") or "ubuntu"
        print(f"[gui-obstacle] autopilot: vm={vm_name} image={image_id}")
        auto = vm_lab.vm_autopilot(image_id=image_id, vm_name=vm_name, force_recreate=True)
        print(json.dumps(auto, indent=2))
        if not auto.get("ok"):
            return 1
    result = vm_lab.vm_obstacle_course(steps)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
