"""
EventBridge Scheduler -> Django scheduled-task handler.

This replaces ``services.internal_cron.CronSupervisor``'s daemon thread.  The
supervisor's model was: one long-lived thread wakes every 30s, walks the task
profile, and runs whatever is due.  On Lambda the scheduler owns the clock
instead -- EventBridge fires this function on a cron/rate expression and the
event names which task to run.

The important behavioural difference from the threaded version:

* **Due-time bookkeeping is no longer ours.**  ``_task_due`` compared
  ``next_run`` against wall-clock time because the thread woke on its own
  schedule.  EventBridge only invokes us when a task is actually due, so we run
  what we are told and record the outcome.  ``force`` short-circuits nothing --
  there is nothing to short-circuit.

* **The guardian lease still matters.**  Two overlapping EventBridge
  invocations (a slow run plus the next tick) would otherwise double-run a
  task.  We keep the lease so the second invocation backs off, which is exactly
  what the threaded version did when a task was already ``running``.

Event shape::

    {"task_id": "auto_pipeline"}      # run one task
    {"task_id": "*"}                  # run every enabled task
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings_lambda"
)
# internal_cron persists per-task state to disk. The bundle is read-only, so
# point it at the only writable path a Lambda sandbox has. This state is a
# reporting convenience (last_run/last_status for the cron panel), not the
# scheduling source of truth -- EventBridge owns the clock now -- so losing it
# on a cold start costs display history, not correctness.
os.environ.setdefault("CRON_STATE_PATH", "/tmp/cron-state.json")
# cron_profile writes its seeded profile next to the config file. The bundled
# copy (config/cron_profile.json, read-only) is used when present; otherwise
# load_profile falls back to the in-memory defaults. Pointing the override at
# /tmp keeps a panel-initiated save_profile working within one sandbox.
os.environ.setdefault(
    "CRON_PROFILE_PATH",
    str(ROOT / "config" / "cron_profile.json")
    if (ROOT / "config" / "cron_profile.json").exists()
    else "/tmp/cron-profile.json",
)

# Redirect the bundle's write targets to /tmp before ANY services.* module
# is imported -- several create directories at import time and /var/task is
# read-only. Must run before django.setup() pulls in the app registry.
from serverless.bootstrap import prepare_writable_dirs  # noqa: E402

prepare_writable_dirs(ROOT)

import django  # noqa: E402

django.setup()

logger = logging.getLogger("serverless.cron")
logging.getLogger().setLevel(os.getenv("DJANGO_LOG_LEVEL", "INFO"))


def _resolve_task_id(event) -> str:
    """Pull the task id out of whatever shape EventBridge handed us."""
    if isinstance(event, dict):
        # Direct invoke / EventBridge Scheduler constant input.
        if event.get("task_id"):
            return str(event["task_id"])
        # EventBridge rule with a `detail` envelope.
        detail = event.get("detail")
        if isinstance(detail, dict) and detail.get("task_id"):
            return str(detail["task_id"])
    return "*"


def lambda_handler(event, context):
    from services.cron_profile import load_profile
    from services.internal_cron import cron_supervisor

    task_id = _resolve_task_id(event)
    profile = load_profile()

    if not profile.get("enabled", True):
        logger.info("cron profile disabled; nothing to do")
        return {"status": "disabled", "ran": []}

    tasks = [t for t in profile.get("tasks", []) if t.get("enabled", True)]
    if task_id != "*":
        tasks = [t for t in tasks if str(t.get("id") or "") == task_id]
        if not tasks:
            logger.warning("no enabled task matching id=%s", task_id)
            return {"status": "not_found", "task_id": task_id, "ran": []}

    ran: list[dict] = []
    for task in tasks:
        tid = str(task.get("id") or "").strip()
        if not tid:
            continue
        try:
            # _execute_task records duration/next_run into the supervisor's
            # state file and applies the same backoff-on-failure the threaded
            # loop used, so scheduled runs stay observable in the cron panel.
            cron_supervisor._execute_task(task, profile)
            ran.append({"task_id": tid, "status": "ok"})
            logger.info("cron task complete: %s", tid)
        except Exception as exc:  # noqa: BLE001
            # One failing task must not prevent the rest of the batch. Lambda
            # would otherwise retry the whole invocation and re-run the tasks
            # that already succeeded.
            logger.exception("cron task failed: %s", tid)
            ran.append({"task_id": tid, "status": "error", "error": str(exc)})

    return {"status": "ok", "ran": ran}
