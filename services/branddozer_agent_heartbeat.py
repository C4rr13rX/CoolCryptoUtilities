"""services/branddozer_agent_heartbeat.py — the periodic auto-resume sweep.

Runs inside the BrandDozer worker loop (see the worker management command),
so no extra process is required. Each tick it finds runs blocked on a CLI
agent cooldown, and for any whose cooldown window has elapsed it probes the
agent and requeues the run when the agent actually answers.

Decision logic lives in branddozer_agent_watch; this module owns the
database side-effects and the tick cadence.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone as dt_timezone
from typing import Any

from services.branddozer_agent_watch import plan_resume, probe_agent

# How often the sweep runs. Cooldowns are minutes-to-hours, so a tight loop
# buys nothing and each probe costs a real CLI invocation.
HEARTBEAT_INTERVAL_SEC = 120

_last_tick = 0.0


def _log(run, message: str) -> None:
    try:
        from services.branddozer_delivery import _set_run_note

        _set_run_note(run, "Agent watch", message[:400])
    except Exception:
        pass


def _mark(run, **updates: Any) -> None:
    context = dict(run.context or {})
    context.update(updates)
    run.context = context
    run.save(update_fields=["context"])


def sweep(*, force: bool = False) -> dict[str, Any]:
    """One auto-resume pass. Returns a summary for logging/tests."""
    global _last_tick
    now = time.time()
    if not force and (now - _last_tick) < HEARTBEAT_INTERVAL_SEC:
        return {"skipped": True}
    _last_tick = now

    from branddozer.models import DeliveryRun
    from services.branddozer_jobs import enqueue_job

    summary: dict[str, Any] = {
        "checked": 0, "resumed": 0, "waiting": 0, "gave_up": 0, "ignored": 0,
    }

    blocked = DeliveryRun.objects.filter(status="blocked")[:50]
    for run in blocked:
        context = run.context or {}
        if context.get("stop_requested"):
            summary["ignored"] += 1
            continue
        summary["checked"] += 1
        decision = plan_resume(context)
        action = decision["action"]

        if action == "ignore":
            summary["ignored"] += 1
            continue

        if action == "give_up":
            summary["gave_up"] += 1
            if not context.get("auto_resume_abandoned"):
                _mark(run, auto_resume_abandoned=decision["reason"])
                _log(run, f"Auto-resume stopped: {decision['reason']}")
            continue

        if action == "wait":
            summary["waiting"] += 1
            if context.get("auto_resume_due_at") != decision.get("due_at"):
                _mark(run, auto_resume_due_at=decision.get("due_at"))
                _log(run, f"Waiting for agent: {decision['reason']}")
            continue

        # action == "resume": confirm the agent really answers before
        # spending the run's next turn on it.
        provider = decision.get("provider") or ""
        probe = probe_agent(provider)
        attempts = int(context.get("auto_resume_attempts") or 0)
        if not probe.get("online"):
            summary["waiting"] += 1
            _mark(
                run,
                auto_resume_attempts=attempts + 1,
                auto_resume_due_at=None,
                auto_resume_last_probe=str(probe.get("detail"))[:300],
            )
            _log(run, f"Agent still unavailable: {str(probe.get('detail'))[:160]}")
            continue

        run.status = "running"
        run.error = ""
        run.save(update_fields=["status", "error"])
        _mark(
            run,
            ai_paused=None,
            auto_resume_attempts=attempts + 1,
            auto_resume_due_at=None,
            auto_resume_last_probe="online",
            auto_resumed_at=datetime.now(dt_timezone.utc).isoformat(),
        )
        enqueue_job(
            kind="delivery_run",
            project=run.project,
            run=run,
            message="Auto-resumed after agent cooldown",
            detail=f"{provider} reported available again",
        )
        _log(run, f"{provider} available again; run resumed automatically")
        summary["resumed"] += 1

    return summary


__all__ = ["sweep", "HEARTBEAT_INTERVAL_SEC"]
