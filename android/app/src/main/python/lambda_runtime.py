"""
Local Lambda runtime — invocation-scoped compute on the device.

This is the point of the whole design, so it is worth stating plainly:
**Lambda's value was never "the cloud". It is that nothing runs when nothing is
happening.** On a server that saves money; on a phone it saves battery and RAM,
which is the scarcer resource. So the same handlers that run in AWS run here,
invoked the same way, with the same contract.

What this replaces
------------------
An earlier draft of the Android port ran guardian, cron, pipeline and
production as four ``while True`` threads holding a wake lock. That is the
always-on server model Lambda exists to replace, and on a phone it is worse
than on a server: the CPU never idles, the radio never sleeps, and the OS
cannot reclaim any of it.

Here instead:

* **User interaction** -> ``invoke("http", event)`` for the request, and
  nothing between requests. A screen the user is not looking at costs zero.
* **Cron** -> Android's ``JobScheduler``/WorkManager wakes the process, one
  handler runs, the process goes back to sleep. The OS batches these wakeups
  with every other app's, which a private thread can never do.

Why handlers and not functions
------------------------------
Each handler already takes ``(event, context)`` and returns a dict. Keeping
that contract means the *same code* runs on the phone and in AWS -- no second
implementation to drift, and a bug found in one is fixed in both. It also
means the phone can hand any invocation to the real Lambda instead (see
``REMOTE_FALLBACK``) without the caller knowing.

Cold start
----------
The expensive part is importing Django (30 apps, pandas, numpy). That is paid
once per process and the module cache is kept warm between invocations, which
is exactly how a warm Lambda sandbox behaves. ``handler_stats()`` reports it so
a slow first tap is explainable rather than mysterious.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
import time
import traceback
from typing import Any, Callable

logger = logging.getLogger("android.lambda_runtime")

# Handler name -> dotted module path. These are the deployed AWS functions,
# unchanged; see serverless/handlers/.
HANDLERS: dict[str, tuple[str, str]] = {
    "http":      ("serverless.handlers.http", "lambda_handler"),
    "auth":      ("serverless.handlers.auth", "lambda_handler"),
    "hybrid":    ("serverless.handlers.hybrid_api", "lambda_handler"),
    "market":    ("serverless.handlers.market_api", "lambda_handler"),
    "cron":      ("serverless.handlers.cron", "lambda_handler"),
    "admin":     ("serverless.handlers.admin_tasks", "lambda_handler"),
}

# Mirrors the EventBridge schedules in serverless/local/deploy_local.sh, so the
# device and the cloud run the same jobs on the same cadence.
SCHEDULES: dict[str, dict[str, Any]] = {
    "auto_pipeline": {
        "handler": "cron",
        "event": {"task_id": "auto_pipeline"},
        "interval_minutes": 180,
    },
    "weekly_bootstrap": {
        "handler": "cron",
        "event": {"task_id": "weekly_bootstrap"},
        "interval_minutes": 10080,
    },
}

_lock = threading.RLock()
# One invocation at a time. This mirrors a Lambda sandbox, which serves exactly
# one event per instance, and it is what a phone wants anyway: concurrent
# Django requests here contend on a single SQLite file and on module-level
# state that was never written to be re-entrant, which showed up as requests
# that hang rather than fail. Serialising trades a little parallelism for
# predictability and lower peak memory.
_invoke_lock = threading.Lock()

# One event loop for the whole process, created but NOT running.
#
# Mangum drives the ASGI app with loop.run_until_complete(), which raises
# "This event loop is already running" if the loop is already spinning in
# another thread -- so a run_forever() thread is exactly wrong here. What
# Mangum needs is a loop object that persists across invocations (its
# ThreadPoolExecutor is bound to one) and is free to be driven on demand.
#
# Since invocations are serialised by _invoke_lock, only one caller ever
# drives it at a time, which is the same guarantee a Lambda sandbox provides.
_loop = None
_loop_lock = threading.Lock()


def _event_loop():
    """The process-wide loop, created on first use and reused thereafter."""
    global _loop
    import asyncio

    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
        return _loop


_loaded: dict[str, Callable] = {}
_stats: dict[str, dict[str, Any]] = {}
_cold_start_ms: float | None = None


class _Context:
    """The subset of Lambda's context object the handlers actually read."""

    def __init__(self, name: str, timeout_ms: int) -> None:
        self.function_name = name
        self.aws_request_id = f"local-{int(time.time() * 1000)}"
        self.memory_limit_in_mb = 512
        self._deadline = time.time() + timeout_ms / 1000.0

    def get_remaining_time_in_millis(self) -> int:
        return max(0, int((self._deadline - time.time()) * 1000))


def _load(name: str) -> Callable:
    """
    Import a handler once and keep it.

    The import is the cold start. Holding the reference is what makes the
    second invocation fast, and is the local equivalent of a warm sandbox.
    """
    global _cold_start_ms

    with _lock:
        if name in _loaded:
            return _loaded[name]
        if name not in HANDLERS:
            raise KeyError(f"unknown handler {name!r}")

        module_path, attr = HANDLERS[name]
        started = time.time()
        module = importlib.import_module(module_path)
        fn = getattr(module, attr)
        elapsed = (time.time() - started) * 1000

        _loaded[name] = fn
        if _cold_start_ms is None:
            _cold_start_ms = elapsed
        logger.info("loaded handler %s in %.0f ms", name, elapsed)
        return fn


def invoke(name: str, event: dict | None = None,
           timeout_ms: int = 30_000) -> dict:
    """
    Run one handler invocation.

    Returns the handler's own response, or an error envelope. Never raises:
    the callers are Android services and a JobScheduler job, and an exception
    crossing the JNI boundary is far harder to diagnose than a returned dict.
    """
    event = event or {}
    started = time.time()
    try:
        fn = _load(name)
        import asyncio

        loop = _event_loop()
        with _invoke_lock:
            # Mangum reads the running/current loop; pointing every worker
            # thread at the same one keeps its executor valid across calls.
            asyncio.set_event_loop(loop)
            result = fn(event, _Context(name, timeout_ms))
        ok = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("handler %s failed", name)
        result = {
            "statusCode": 500,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6),
        }
        ok = False

    duration = (time.time() - started) * 1000
    with _lock:
        stat = _stats.setdefault(name, {"invocations": 0, "errors": 0,
                                        "total_ms": 0.0, "last_ms": 0.0})
        stat["invocations"] += 1
        stat["errors"] += 0 if ok else 1
        stat["total_ms"] += duration
        stat["last_ms"] = duration
        stat["last_run"] = started
    return result


def invoke_json(name: str, event_json: str = "{}",
                timeout_ms: int = 30_000) -> str:
    """
    JSON-in/JSON-out wrapper for the Java bridge.

    Chaquopy marshals primitives cleanly but nested dicts are awkward, so the
    boundary speaks JSON in both directions.
    """
    try:
        event = json.loads(event_json or "{}")
    except json.JSONDecodeError as exc:
        return json.dumps({"statusCode": 400, "error": f"bad event JSON: {exc}"})
    return json.dumps(invoke(name, event, timeout_ms), default=str)


def run_schedule(schedule_name: str) -> str:
    """Entry point for a scheduled wakeup. Returns JSON for the Java side."""
    schedule = SCHEDULES.get(schedule_name)
    if not schedule:
        return json.dumps({"status": "unknown_schedule", "name": schedule_name})
    result = invoke(schedule["handler"], schedule["event"], timeout_ms=600_000)
    return json.dumps({"status": "ok", "schedule": schedule_name,
                       "result": result}, default=str)


def warm(names: list[str] | None = None) -> str:
    """
    Pre-import handlers so the first user tap is not the cold start.

    Called once after boot from a background thread. This is the deliberate
    exception to "nothing runs when idle": a few hundred milliseconds of
    import, once, in exchange for an interface that responds immediately.
    """
    targets = names or ["http", "auth"]
    for name in targets:
        try:
            _load(name)
        except Exception:  # noqa: BLE001
            logger.exception("warm failed for %s", name)
    return json.dumps(handler_stats())


def handler_stats() -> dict:
    """Per-handler invocation counts and timings, for the UI and logs."""
    with _lock:
        out = {
            "cold_start_ms": _cold_start_ms,
            "loaded": sorted(_loaded),
            "handlers": {},
        }
        for name, stat in _stats.items():
            calls = stat["invocations"] or 1
            out["handlers"][name] = {
                "invocations": stat["invocations"],
                "errors": stat["errors"],
                "avg_ms": round(stat["total_ms"] / calls, 1),
                "last_ms": round(stat["last_ms"], 1),
                "last_run": stat.get("last_run"),
            }
        return out


def stats_json() -> str:
    return json.dumps(handler_stats(), default=str)
