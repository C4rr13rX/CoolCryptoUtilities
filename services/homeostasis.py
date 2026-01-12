from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from django.utils import timezone

from branddozer.models import BacklogItem, DeliveryRun, GateRun


DEFAULT_SETPOINTS = {
    "signals": {
        "max_gate_failures": 2,
        "min_gate_pass_rate": 0.8,
        "max_open_backlog": 12,
        "heartbeat_interval_minutes": 10,
        "conversion_required": True,
    }
}


def load_setpoints() -> Dict[str, Any]:
    path = Path("config/homeostasis.yaml")
    if not path.exists():
        return DEFAULT_SETPOINTS
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
        return data or DEFAULT_SETPOINTS
    except Exception:
        return DEFAULT_SETPOINTS


def gate_pass_rate(run: DeliveryRun) -> float:
    gates = list(GateRun.objects.filter(run=run))
    if not gates:
        return 1.0
    passed = sum(1 for g in gates if g.status == "passed")
    return passed / max(1, len(gates))


def gate_failures(run: DeliveryRun) -> int:
    gates = GateRun.objects.filter(run=run)
    return sum(1 for g in gates if g.status not in {"passed", "skipped"})


def open_backlog_count(run: DeliveryRun) -> int:
    return BacklogItem.objects.filter(run=run).exclude(status="done").count()


def heartbeat_payload(run: DeliveryRun) -> Dict[str, Any]:
    return {
        "run_id": str(run.id),
        "status": run.status,
        "phase": run.phase,
        "note": (run.context or {}).get("status_note", ""),
        "gates_pass_rate": round(gate_pass_rate(run), 3),
        "gates_failures": gate_failures(run),
        "backlog_open": open_backlog_count(run),
        "ts": timezone.now().isoformat(),
    }


def should_throttle(run: DeliveryRun, setpoints: Dict[str, Any]) -> bool:
    sp = (setpoints.get("signals") or {})
    return (
        gate_failures(run) > int(sp.get("max_gate_failures", 2))
        or gate_pass_rate(run) < float(sp.get("min_gate_pass_rate", 0.8))
        or open_backlog_count(run) > int(sp.get("max_open_backlog", 12))
    )


def heartbeat_due(last_hb: Optional[float], setpoints: Dict[str, Any]) -> bool:
    interval_minutes = (setpoints.get("signals") or {}).get("heartbeat_interval_minutes", 10)
    interval_seconds = max(60, int(interval_minutes) * 60)
    if last_hb is None:
        return True
    return (time.time() - last_hb) >= interval_seconds


def save_control_state(run: DeliveryRun, throttle: bool) -> None:
    ctx = dict(run.context or {})
    ctx["throttle_new_work"] = throttle
    run.context = ctx
    run.save(update_fields=["context"])
