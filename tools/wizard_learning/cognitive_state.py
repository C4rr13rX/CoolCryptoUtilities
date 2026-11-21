"""tools/wizard_learning/cognitive_state.py

Cross-process atomic Cognitive State & Chat-Availability Coordinator for
W1z4rD Brain + C0d3rV2.

Shared via `runtime/c0d3r/wizard_cognitive_state.json` between:
  1. The 24/7 `AutonomousLifecycle` daemon (self-education, self-training,
     self-testing, error-free sleeping/settling, and Multiscale Robot World
     controller / self-upgrading).
  2. The Django/Waitress `wizard_chat` backend (`:8001`), which exposes all
     indicators to `WizardChatView.vue`, gates/informs the user whenever the
     brain is in a non-chattable state (sleeping/consolidating, checkpoint
     write-lock, or offline), and signals instant user-chat preemption so
     background training yields whenever the user is chatting.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_DIR = _PROJECT_ROOT / "runtime" / "c0d3r"
STATE_FILE = RUNTIME_DIR / "wizard_cognitive_state.json"
_LOCK = threading.RLock()

# Modes where the brain cannot service interactive chat immediately
NON_CHATTABLE_MODES = {
    "SLEEPING_SETTLING",
    "CHECKPOINT_LOCKED",
    "OFFLINE",
}

MODE_LABELS = {
    "IDLE_READY": "Ready (Idle)",
    "USER_CHATTING": "Active User Chat",
    "SELF_EDUCATING": "Self-Educating (C0d3rV2 + W1z4rD)",
    "TRAINING_SELF": "Self-Training Synapses",
    "TESTING_TRAINING": "Self-Testing & Verifying Recall",
    "SLEEPING_SETTLING": "Sleeping / Settling Memory",
    "CONTROLLING_ROBOT_WORLD": "Controlling Multiscale Robot World",
    "CHECKPOINT_LOCKED": "Disk Checkpoint Lock",
    "OFFLINE": "Node Offline / Booting",
}

DEFAULT_CHAT_BLOCK_REASONS = {
    "SLEEPING_SETTLING": (
        "W1z4rD is currently sleeping (consolidating STDP synapses and checkpointing to SSD). "
        "Chat is temporarily paused and will unlock automatically as soon as settling finishes."
    ),
    "CHECKPOINT_LOCKED": (
        "W1z4rD node is holding an exclusive write lock while flushing a durable brain checkpoint. "
        "Chat will resume in a few seconds."
    ),
    "OFFLINE": (
        "W1z4rD node is currently offline or restarting. Chat will unlock automatically once /health responds."
    ),
}


def _default_state() -> Dict[str, Any]:
    now = time.time()
    return {
        "version": 1,
        "updated_at": now,
        "updated_iso": datetime.datetime.now().isoformat(),
        "mode": "IDLE_READY",
        "previous_mode": "IDLE_READY",
        "mode_entered_at": now,
        "last_wake_at": 0.0,
        "settling_holder_pid": None,
        "mode_label": MODE_LABELS["IDLE_READY"],
        "can_chat": True,
        "chat_unavailable_reason": "",
        "detail": "W1z4rD + C0d3rV2 ready. Monitoring for idle self-education & robot world cycles.",
        "autonomous_provider": "W1z4rD + C0d3rV2 (Zero External LLM)",
        "daemon_pid": None,
        "daemon_heartbeat_ts": 0.0,
        "last_user_chat_ts": 0.0,
        "self_edu_cycles": 0,
        "hypotheses_resolved": 0,
        "pairs_trained_total": 0,
        "self_tests_run": 0,
        "self_tests_passed": 0,
        "self_test_pass_rate": 1.0,
        "last_self_test_summary": "Awaiting initial verification cycle",
        "sleep_cycles_ok": 0,
        "sleep_errors": 0,
        "last_sleep_error": "",
        "post_sleep_recall_rate": 1.0,
        "robot_world": {
            "active_controllers": 0,
            "robots_built": 0,
            "body_generation": 1,
            "primary_mark": "Mark I",
            "last_upgrade_summary": "Awaiting initial MSRW controller cycle",
            "physics_tests_passed": True,
        },
    }


class CognitiveStateManager:
    """Atomic file-backed coordinator shared across daemon and web server."""

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> Dict[str, Any]:
        with _LOCK:
            state = _default_state()
            if self.state_file.exists():
                try:
                    raw = self.state_file.read_text(encoding="utf-8")
                    loaded = json.loads(raw)
                    if isinstance(loaded, dict):
                        robot_world = state["robot_world"].copy()
                        if isinstance(loaded.get("robot_world"), dict):
                            robot_world.update(loaded["robot_world"])
                        state.update(loaded)
                        state["robot_world"] = robot_world
                except Exception:
                    pass

            now = time.time()
            hb = float(state.get("daemon_heartbeat_ts") or 0.0)
            state["daemon_alive"] = (now - hb) < 120.0 if hb > 0 else False
            state["user_chat_active"] = (now - float(state.get("last_user_chat_ts") or 0.0)) < 20.0

            # Auto-recover stale SLEEPING_SETTLING / CHECKPOINT_LOCKED if older than 240s
            # UNLESS a live settling_holder_pid is actively running a long/REM consolidation.
            updated_at = float(state.get("updated_at") or 0.0)
            holder_pid = state.get("settling_holder_pid")
            holder_alive = False
            if isinstance(holder_pid, int) and holder_pid > 0:
                try:
                    import psutil
                    holder_alive = psutil.pid_exists(holder_pid)
                except Exception:
                    holder_alive = False
            if (
                state.get("mode") in ("SLEEPING_SETTLING", "CHECKPOINT_LOCKED")
                and (now - updated_at) > 240.0
                and not holder_alive
            ):
                state["previous_mode"] = state.get("mode", "SLEEPING_SETTLING")
                state["mode"] = "IDLE_READY"
                state["mode_entered_at"] = now
                state["last_wake_at"] = now
                state["settling_holder_pid"] = None
                state["mode_label"] = MODE_LABELS["IDLE_READY"]
                state["can_chat"] = True
                state["chat_unavailable_reason"] = ""

            mode_entered_at = float(state.get("mode_entered_at") or updated_at or now)
            mode_duration = round(max(0.0, now - mode_entered_at), 2)
            last_wake_at = float(state.get("last_wake_at") or 0.0)
            secs_since_wake = round(max(0.0, now - last_wake_at), 2) if last_wake_at > 0 else 9999.0
            cur_mode = str(state.get("mode") or "IDLE_READY")
            is_rem = bool(cur_mode in ("SLEEPING_SETTLING", "CHECKPOINT_LOCKED") and mode_duration >= 3600.0)

            if cur_mode == "OFFLINE":
                eye_state = "offline"
            elif cur_mode in ("SLEEPING_SETTLING", "CHECKPOINT_LOCKED"):
                eye_state = "rem_sleep" if is_rem else "settling_sleep"
            elif secs_since_wake < 6.0:
                eye_state = "waking"
            else:
                eye_state = "awake"

            state["mode_duration_seconds"] = mode_duration
            state["is_rem_sleep"] = is_rem
            state["seconds_since_wake"] = secs_since_wake
            state["eye_lifecycle_state"] = eye_state

            return state

    def update(self, **patch: Any) -> Dict[str, Any]:
        with _LOCK:
            current = self.read()
            prev_mode = str(current.get("mode") or "IDLE_READY")
            for computed_key in (
                "daemon_alive",
                "user_chat_active",
                "mode_duration_seconds",
                "is_rem_sleep",
                "seconds_since_wake",
                "eye_lifecycle_state",
            ):
                current.pop(computed_key, None)

            if "robot_world" in patch and isinstance(patch["robot_world"], dict):
                merged_rw = dict(current.get("robot_world") or {})
                merged_rw.update(patch.pop("robot_world"))
                current["robot_world"] = merged_rw

            now = time.time()
            new_mode = str(patch.get("mode", prev_mode) or "IDLE_READY")
            if new_mode != prev_mode:
                current["previous_mode"] = prev_mode
                if "mode_entered_at" not in patch:
                    current["mode_entered_at"] = now
                if prev_mode in NON_CHATTABLE_MODES and new_mode not in NON_CHATTABLE_MODES:
                    current["last_wake_at"] = now
                if new_mode not in ("SLEEPING_SETTLING", "CHECKPOINT_LOCKED") and "settling_holder_pid" not in patch:
                    current["settling_holder_pid"] = None

            current.update(patch)
            mode = str(current.get("mode") or "IDLE_READY")
            current["mode_label"] = MODE_LABELS.get(mode, mode)

            if "can_chat" not in patch:
                current["can_chat"] = mode not in NON_CHATTABLE_MODES

            if not current["can_chat"]:
                if not current.get("chat_unavailable_reason"):
                    current["chat_unavailable_reason"] = DEFAULT_CHAT_BLOCK_REASONS.get(
                        mode,
                        "W1z4rD is temporarily busy and unable to chat right now.",
                    )
            else:
                if "chat_unavailable_reason" not in patch:
                    current["chat_unavailable_reason"] = ""

            now = time.time()
            current["updated_at"] = now
            current["updated_iso"] = datetime.datetime.now().isoformat()

            tmp = self.state_file.with_suffix(f".{os.getpid()}.tmp")
            try:
                tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
                os.replace(tmp, self.state_file)
            except Exception:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
            return self.read()

    def set_mode(
        self,
        mode: str,
        detail: str = "",
        *,
        can_chat: Optional[bool] = None,
        chat_unavailable_reason: Optional[str] = None,
        is_daemon: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        patch: Dict[str, Any] = {"mode": mode}
        if detail:
            patch["detail"] = detail
        if can_chat is not None:
            patch["can_chat"] = can_chat
        if chat_unavailable_reason is not None:
            patch["chat_unavailable_reason"] = chat_unavailable_reason
        if is_daemon:
            patch["daemon_pid"] = os.getpid()
            patch["daemon_heartbeat_ts"] = time.time()
        patch.update(extra)
        return self.update(**patch)

    def heartbeat(self, detail: Optional[str] = None) -> Dict[str, Any]:
        patch: Dict[str, Any] = {
            "daemon_pid": os.getpid(),
            "daemon_heartbeat_ts": time.time(),
        }
        if detail is not None:
            patch["detail"] = detail
        return self.update(**patch)

    def mark_user_chat_start(self) -> Dict[str, Any]:
        now = time.time()
        cur = self.read()
        # Do not override SLEEPING_SETTLING if brain is actively consolidating
        if cur.get("mode") == "SLEEPING_SETTLING":
            return self.update(last_user_chat_ts=now)
        return self.update(
            last_user_chat_ts=now,
            mode="USER_CHATTING",
            detail="Serving live user conversation in Wizard sanctum.",
            can_chat=True,
            chat_unavailable_reason="",
        )

    def mark_user_chat_end(self) -> Dict[str, Any]:
        cur = self.read()
        if cur.get("mode") == "USER_CHATTING":
            return self.update(
                mode="IDLE_READY",
                detail="User chat reply delivered; standing by before resuming self-education.",
                can_chat=True,
                chat_unavailable_reason="",
            )
        return cur

    def is_user_recently_active(self, grace_seconds: float = 20.0) -> bool:
        st = self.read()
        last_ts = float(st.get("last_user_chat_ts") or 0.0)
        return (time.time() - last_ts) < grace_seconds

    def probe_live_chat_readiness(self, endpoint: str = "http://127.0.0.1:8090", probe_timeout: float = 0.6) -> Dict[str, Any]:
        """Sub-millisecond cached live probe verifying whether the W1z4rD node can currently
        accept chat, normalizing localhost -> 127.0.0.1 to avoid Windows 2s IPv6 SYN stalls."""
        st = self.read()
        if not st.get("can_chat", True):
            return st

        norm_ep = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
        now = time.time()
        last_ok = getattr(self, "_health_ok_cache", {}).get(norm_ep, 0.0)
        if (now - last_ok) < 10.0:
            return st

        url = f"{norm_ep}/health"
        eff_timeout = min(float(probe_timeout), 0.75) if ":18096" not in norm_ep else float(probe_timeout)
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                if resp.status == 200:
                    cache = getattr(self, "_health_ok_cache", None)
                    if cache is None:
                        cache = {}
                        self._health_ok_cache = cache
                    cache[norm_ep] = now
                    return st
        except TimeoutError:
            return {
                **st,
                "mode": "CHECKPOINT_LOCKED",
                "mode_label": MODE_LABELS["CHECKPOINT_LOCKED"],
                "can_chat": False,
                "chat_unavailable_reason": DEFAULT_CHAT_BLOCK_REASONS["CHECKPOINT_LOCKED"],
            }
        except Exception as exc:
            err_s = str(exc).lower()
            if "timed out" in err_s or "timeout" in err_s:
                return {
                    **st,
                    "mode": "CHECKPOINT_LOCKED",
                    "mode_label": MODE_LABELS["CHECKPOINT_LOCKED"],
                    "can_chat": False,
                    "chat_unavailable_reason": DEFAULT_CHAT_BLOCK_REASONS["CHECKPOINT_LOCKED"],
                }
            return {
                **st,
                "mode": "OFFLINE",
                "mode_label": MODE_LABELS["OFFLINE"],
                "can_chat": False,
                "chat_unavailable_reason": DEFAULT_CHAT_BLOCK_REASONS["OFFLINE"],
            }
        return st


_MANAGER: Optional[CognitiveStateManager] = None


def get_cognitive_state_manager() -> CognitiveStateManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = CognitiveStateManager()
    return _MANAGER
