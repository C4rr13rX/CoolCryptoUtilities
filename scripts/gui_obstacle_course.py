#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--evaluate", action="store_true", help="Evaluate GUI screenshots after running steps.")
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
        force_recreate = os.getenv("C0D3R_VM_FORCE_RECREATE", "0").strip().lower() in {"1", "true", "yes", "on"}
        auto = vm_lab.vm_autopilot(image_id=image_id, vm_name=vm_name, force_recreate=force_recreate)
        print(json.dumps(auto, indent=2))
        if not auto.get("ok"):
            return 1
        ready = vm_lab.vm_wait_ready(vm_name, timeout_s=1800)
        print(json.dumps(ready, indent=2))
        if not ready.get("ok"):
            return 1
    result = vm_lab.vm_obstacle_course(steps)
    print(json.dumps(result, indent=2))
    if args.evaluate:
        try:
            _evaluate_gui(plan_path, payload, result)
        except Exception as exc:
            print(f"[gui-obstacle] evaluation failed: {exc}")
    return 0 if result.get("ok") else 1


def _evaluate_gui(plan_path: Path, plan: dict, result: dict) -> None:
    eval_items = plan.get("eval") or []
    if not isinstance(eval_items, list) or not eval_items:
        print("[gui-obstacle] no eval items configured")
        return
    from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings

    settings = c0d3r_default_settings()
    settings["stream_default"] = False
    session = C0d3rSession(
        session_name="gui-obstacle-eval",
        transcript_dir=str(Path("runtime") / "c0d3r" / "transcripts"),
        workdir=str(Path.cwd()),
        **settings,
    )
    out_path = Path("runtime") / "c0d3r" / "gui_obstacle_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_results: list[dict] = []
    base_dir = plan_path.parent
    for item in eval_items:
        before = str(item.get("before") or "")
        after = str(item.get("after") or "")
        criteria = str(item.get("criteria") or "Evaluate UI quality and UX clarity.")
        if not before or not after:
            continue
        before_path = (base_dir / before).resolve()
        after_path = (base_dir / after).resolve()
        prompt = (
            "Compare two UI screenshots.\n"
            f"Criteria: {criteria}\n"
            "Return JSON with keys: score_before (0-10), score_after (0-10), improved (true/false), "
            "issues_before (list), issues_after (list), recommendations (list)."
        )
        try:
            response = session.send(prompt=prompt, images=[str(before_path), str(after_path)], stream=False)
            eval_results.append({"before": str(before_path), "after": str(after_path), "response": response})
        except Exception as exc:
            eval_results.append({"before": str(before_path), "after": str(after_path), "error": str(exc)})
    out_path.write_text(json.dumps({"plan": str(plan_path), "results": eval_results}, indent=2), encoding="utf-8")
    print(f"[gui-obstacle] evaluation saved: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
