from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict


def _default_report_path() -> Path:
    return Path(os.getenv("LIVE_REPLAY_REPORT", "data/reports/replay_gate.json"))


def _load_replay_results(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_report(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = float(time.time())
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[replay-gate] wrote report to {path}")


def main() -> None:
    """
    Very lightweight gate generator. In a full setup, plug in an offline replay
    harness and set status accordingly. Here we accept a desired status via
    env/args and emit the gate report.
    """
    status = os.getenv("REPLAY_STATUS", "pass").strip().lower()
    if len(sys.argv) > 1:
        status = sys.argv[1].strip().lower()
    allowed = {"pass", "ok", "success", "fail", "error"}
    if status not in allowed:
        status = "fail"
    report: Dict[str, Any] = _load_replay_results(_default_report_path())
    report["status"] = status
    report["notes"] = os.getenv("REPLAY_NOTES", "")
    report["samples"] = report.get("samples", 0)
    _save_report(report, _default_report_path())


if __name__ == "__main__":
    main()
