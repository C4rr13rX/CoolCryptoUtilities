from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from services.guardian_lock import GuardianLease

STATUS_DIR = Path("runtime/guardian")
STATUS_FILE = STATUS_DIR / "status.json"
_LOCAL_LOCK = threading.Lock()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_state() -> Dict[str, Any]:
    return {
        "queue": {},
        "slots": {},
        "history": {},
        "production": {"running": False, "updated_at": None, "metadata": {}},
    }


def _read_state() -> Dict[str, Any]:
    if not STATUS_FILE.exists():
        return _default_state()
    try:
        raw = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()
    if not isinstance(raw, dict):
        return _default_state()
    for key in ("queue", "slots", "history"):
        raw.setdefault(key, {})
    raw.setdefault("production", {"running": False, "updated_at": None, "metadata": {}})
    return raw


def _write_state(state: Dict[str, Any]) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def _locked_state():
    """
    Acquire a short-lived lock to edit the guardian status file. Falls back
    to a simple in-process mutex if we cannot grab the inter-process lease.
    """

    lease: Optional[GuardianLease] = None
    acquired = False
    try:
        lease = GuardianLease("guardian-status", timeout=5, poll_interval=0.1)
        acquired = lease.acquire()
    except Exception:
        acquired = False
    if not acquired:
        _LOCAL_LOCK.acquire()
    try:
        state = _read_state()
        yield state
        _write_state(state)
    finally:
        if acquired and lease:
            lease.release()
        else:
            _LOCAL_LOCK.release()


def enqueue_slot(slot: str, owner: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
    ticket = str(uuid.uuid4())
    entry = {
        "ticket": ticket,
        "owner": owner,
        "requested_at": _timestamp(),
        "metadata": metadata or {},
        "state": "waiting",
    }
    try:
        with _locked_state() as state:
            queue = state.setdefault("queue", {}).setdefault(slot, [])
            queue.append(entry)
    except Exception:
        return None
    return ticket


def mark_slot_running(slot: str, ticket: Optional[str]) -> None:
    if not ticket:
        return
    try:
        with _locked_state() as state:
            queue = state.setdefault("queue", {}).setdefault(slot, [])
            entry: Optional[Dict[str, Any]] = None
            for candidate in list(queue):
                if candidate.get("ticket") == ticket:
                    entry = candidate
                    queue.remove(candidate)
                    break
            if not entry:
                entry = {"ticket": ticket, "owner": "unknown", "metadata": {}}
            entry["state"] = "running"
            entry["started_at"] = _timestamp()
            state.setdefault("slots", {})[slot] = entry
    except Exception:
        return


def mark_slot_finished(
    slot: str,
    ticket: Optional[str],
    *,
    outcome: str = "success",
    message: Optional[str] = None,
) -> None:
    if not ticket:
        return
    try:
        with _locked_state() as state:
            slots = state.setdefault("slots", {})
            entry = slots.pop(slot, None)
            if not entry or entry.get("ticket") != ticket:
                entry = {"ticket": ticket, "owner": "unknown", "metadata": {}}
            entry.update(
                {
                    "state": outcome,
                    "finished_at": _timestamp(),
                }
            )
            if message:
                entry["message"] = message
            history = state.setdefault("history", {}).setdefault(slot, [])
            history.append(entry)
            if len(history) > 50:
                del history[:-50]
            queue = state.setdefault("queue", {}).setdefault(slot, [])
            queue[:] = [item for item in queue if item.get("ticket") != ticket]
    except Exception:
        return


def snapshot_status() -> Dict[str, Any]:
    try:
        return _read_state()
    except Exception:
        return _default_state()


def update_production_state(running: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "running": bool(running),
        "updated_at": _timestamp(),
        "metadata": metadata or {},
        "host": os.uname().nodename if hasattr(os, "uname") else os.getenv("HOSTNAME"),
    }
    try:
        with _locked_state() as state:
            state["production"] = payload
    except Exception:
        return
