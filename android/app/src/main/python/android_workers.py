"""
Scheduled invocations — the device-side equivalent of EventBridge.

**There are no loops here, by design.** An earlier draft ran guardian, cron,
pipeline and production as four ``while True`` threads holding a wake lock.
That is the always-on server model, and on a phone it is strictly worse than on
a server: the CPU never idles, the radio never sleeps, and the OS cannot
reclaim anything.

Instead Android's ``JobScheduler`` owns the clock, exactly as EventBridge owns
it in AWS. It wakes the process, ``lambda_runtime`` invokes one handler, the
process goes back to sleep. Two things follow that a private timer can never
achieve:

* the OS **batches** our wakeups with every other app's, so the radio and CPU
  wake once for many jobs rather than once for ours alone;
* Doze and app-standby are respected rather than fought, so the system stops
  treating the app as a battery offender.

This module is the thin Python side of that: it maps a schedule name to a
handler invocation and reports what happened. The cadence lives in
``lambda_runtime.SCHEDULES`` so the phone and AWS run the same jobs on the same
intervals.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import lambda_runtime

logger = logging.getLogger("android.workers")

# Last outcome per schedule, for the notification and the UI. Small and
# bounded -- this is a status board, not a history.
_last: dict[str, dict[str, Any]] = {}


def run_scheduled(schedule_name: str) -> str:
    """
    Execute one scheduled job. Called from ScheduledJobService.

    Returns JSON so the Java side can decide whether to reschedule without
    parsing Python objects across the bridge.
    """
    started = time.time()
    schedule = lambda_runtime.SCHEDULES.get(schedule_name)
    if not schedule:
        return json.dumps({"status": "unknown_schedule", "name": schedule_name})

    result = lambda_runtime.invoke(
        schedule["handler"], schedule["event"], timeout_ms=600_000
    )
    duration = (time.time() - started) * 1000
    failed = isinstance(result, dict) and result.get("statusCode") == 500

    _last[schedule_name] = {
        "ran_at": started,
        "duration_ms": round(duration, 1),
        "ok": not failed,
        "error": result.get("error") if failed else None,
    }
    logger.info("schedule %s ran in %.0f ms ok=%s",
                schedule_name, duration, not failed)

    return json.dumps({
        "status": "error" if failed else "ok",
        "schedule": schedule_name,
        "duration_ms": round(duration, 1),
        "result": result,
    }, default=str)


def run_guardian_check() -> str:
    """
    One guardian health pass.

    Guardian is a *check*, not a daemon: it samples state and records it. Run
    on the same wakeups as everything else rather than on a 30-second timer of
    its own, which on a phone would be the single largest battery cost here.
    """
    started = time.time()
    try:
        from services.guardian_status import snapshot_status

        state = snapshot_status()
        _last["guardian"] = {
            "ran_at": started,
            "duration_ms": round((time.time() - started) * 1000, 1),
            "ok": True,
            "production_running": bool(
                (state.get("production") or {}).get("running")),
        }
        return json.dumps(_last["guardian"], default=str)
    except Exception as exc:  # noqa: BLE001
        logger.exception("guardian check failed")
        _last["guardian"] = {"ran_at": started, "ok": False,
                             "error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(_last["guardian"], default=str)


def status() -> dict:
    """Last outcome for every schedule, plus runtime stats."""
    return {
        "schedules": {
            name: dict(spec, last=_last.get(name))
            for name, spec in lambda_runtime.SCHEDULES.items()
        },
        "guardian": _last.get("guardian"),
        "runtime": lambda_runtime.handler_stats(),
    }


def status_json() -> str:
    return json.dumps(status(), default=str)


def summary_line() -> str:
    """One line for the service notification."""
    stats = lambda_runtime.handler_stats()
    total = sum(h["invocations"] for h in stats["handlers"].values())
    errors = sum(h["errors"] for h in stats["handlers"].values())
    if errors:
        return f"{total} invocations, {errors} failed"
    return f"{total} invocations, idle" if total else "Idle"
