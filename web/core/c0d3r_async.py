from __future__ import annotations

import concurrent.futures
import os
import re
import threading
import time
from pathlib import Path

from django.db import close_old_connections
from django.http import HttpRequest
from django.utils import timezone

from .models import C0d3rWebMessage, C0d3rWebRun


ROOT = Path(__file__).resolve().parents[2]
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("C0D3R_WEB_WORKERS", "2"))),
    thread_name_prefix="c0d3r-web",
)
_LOCKS: dict[int, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_PROGRESS: dict[str, dict] = {}
_CANCEL_EVENTS: dict[str, threading.Event] = {}


def _set_progress(run_id: str, phase: str, detail: str = "", **extra) -> None:
    current = _PROGRESS.setdefault(str(run_id), {"started_monotonic": time.monotonic()})
    current.update({"phase": phase, "detail": detail, **extra})


def run_progress(run_id: str) -> dict:
    current = dict(_PROGRESS.get(str(run_id), {}))
    started = current.pop("started_monotonic", None)
    session_key = current.pop("session_key", "")
    if session_key:
        try:
            from tools.c0d3rV2.web_runner import _FLOW_CACHE
            flow = _FLOW_CACHE.get(session_key)
            if flow is not None:
                usage = getattr(flow, "usage", None)
                if usage is not None:
                    current["agent_status"] = getattr(usage, "status", "")
                    current["agent_action"] = getattr(usage, "last_action", "")
                session = getattr(flow, "session", None)
                get_model_id = getattr(session, "get_model_id", None)
                if callable(get_model_id):
                    current["model"] = get_model_id()
        except Exception:
            pass
    if started is not None:
        current["elapsed_seconds"] = round(max(0.0, time.monotonic() - started), 1)
    return current


def cancel_run(run_id: str) -> bool:
    event = _CANCEL_EVENTS.setdefault(str(run_id), threading.Event())
    event.set()
    _set_progress(str(run_id), "cancelled", "Stopped by user")
    updated = C0d3rWebRun.objects.filter(
        id=run_id, status__in=["queued", "running"],
    ).update(status="cancelled", error="Stopped by user", completed_at=timezone.now())
    return bool(updated)


def _is_conversational_acknowledgement(output: str) -> bool:
    text = " ".join(str(output or "").lower().split())
    return any(marker in text for marker in (
        "how can i assist", "how may i assist", "how can i help",
        "how may i help", "what would you like", "ready to help",
        "please provide the task", "please provide more details",
    ))


def _is_social_prompt(prompt: str) -> bool:
    text = " ".join(str(prompt or "").strip().lower().split()).rstrip(".!?,;:")
    return bool(re.fullmatch(
        r"(hi|hello|hey|hiya|howdy|good (morning|afternoon|evening)|"
        r"how are you|how are things|what's up|whats up|how's it going|hows it going|"
        r"thanks|thank you|thank you very much|thx|bye|goodbye|see you|see you later)(\b.*)?",
        text,
    ))


def _session_lock(session_id: int) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(session_id, threading.Lock())


def submit_run(run_id: str) -> None:
    _EXECUTOR.submit(execute_run, str(run_id))


def execute_run(run_id: str) -> None:
    close_old_connections()
    cancel_event = _CANCEL_EVENTS.setdefault(str(run_id), threading.Event())
    try:
        run = C0d3rWebRun.objects.select_related("session__user").get(id=run_id)
        if run.status != "queued":
            return
        with _session_lock(run.session_id):
            run.refresh_from_db()
            if run.status != "queued":
                return
            run.status = "running"
            run.started_at = timezone.now()
            run.error = ""
            run.save(update_fields=["status", "started_at", "error", "updated_at"])
            _set_progress(run_id, "starting", "Preparing C0d3rV2 context", backend=run.backend or "freeloader")
            try:
                from .views import _build_c0d3r_context, _update_c0d3r_summary
                from tools.c0d3rV2.web_runner import run as c0d3rv2_run, _FLOW_CACHE

                request = HttpRequest()
                request.user = run.session.user
                context_chars = max(2000, min(20000, int(os.getenv("C0D3R_WEB_CONTEXT_CHARS", "12000"))))
                context = _build_c0d3r_context(
                    run.session, request=request, prompt=run.prompt, max_chars=context_chars,
                )
                if cancel_event.is_set():
                    return
                session_key = f"c0d3rv2:user:{run.session.user_id}:session:{run.session_id}"
                _set_progress(
                    run_id, "agent_running", "AgentTheFreeloader is selecting and running a model",
                    backend=run.backend or "freeloader", requested_model=run.model,
                    allowed_models=list(run.atf_models or []), session_key=session_key,
                )
                output = c0d3rv2_run(
                    run.prompt, session_key=session_key, workdir=ROOT,
                    backend=run.backend, model=run.model,
                    atf_models=list(run.atf_models or []), system_context=context,
                )
                if cancel_event.is_set():
                    return
                _set_progress(run_id, "finalizing", "Persisting the verified C0d3rV2 response")
                if _is_conversational_acknowledgement(output) and not _is_social_prompt(run.prompt):
                    raise RuntimeError(
                        "ATF returned an acknowledgement instead of answering the request; "
                        "the run was rejected and logged for correction."
                    )
                flow = _FLOW_CACHE.get(session_key)
                model_id = (
                    str(getattr(flow, "_last_model_id", "") or "")
                    or (
                        flow.session.get_model_id()
                        if flow and hasattr(flow.session, "get_model_id") else "unknown"
                    )
                )
                C0d3rWebMessage.objects.create(
                    session=run.session, role="c0d3r", content=output, model_id=model_id,
                    metadata={"run_id": str(run.id)},
                )
                run.session.model_id = model_id
                run.session.last_active = timezone.now()
                try:
                    _update_c0d3r_summary(
                        run.session, user_text=run.prompt, assistant_text=output, c0d3r_session=None,
                    )
                except Exception:
                    pass
                run.session.save(update_fields=[
                    "model_id", "last_active", "summary", "key_points", "updated_at",
                ])
                run.refresh_from_db(fields=["status"])
                if run.status != "cancelled":
                    run.status = "completed"
                    run.output = output
                    run.model_id = model_id
                    run.completed_at = timezone.now()
                    run.save(update_fields=[
                        "status", "output", "model_id", "completed_at", "updated_at",
                    ])
                    _set_progress(run_id, "completed", "Response completed", model=model_id)
            except Exception as exc:
                run.refresh_from_db(fields=["status"])
                if run.status != "cancelled":
                    run.status = "failed"
                    run.error = f"{type(exc).__name__}: {exc}"[:4000]
                    run.completed_at = timezone.now()
                    run.save(update_fields=["status", "error", "completed_at", "updated_at"])
                    _set_progress(run_id, "failed", run.error)
    finally:
        close_old_connections()
