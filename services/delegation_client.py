"""
Delegation client — dispatches tasks to Revenir Delegation Service hosts.

Integrates with the production manager to offload work from the main system.
Handles host selection, task dispatch, result collection, and failure recovery.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("delegation.client")

# How often to heartbeat / check host health (seconds)
_HEARTBEAT_INTERVAL = float(os.getenv("DELEGATION_HEARTBEAT_INTERVAL", "15"))
# Timeout for HTTP calls to hosts
_REQUEST_TIMEOUT = int(os.getenv("DELEGATION_REQUEST_TIMEOUT", "30"))
# How long before a host is considered offline
_OFFLINE_THRESHOLD = float(os.getenv("DELEGATION_OFFLINE_THRESHOLD", "60"))


def _get_db():
    """Lazy import to avoid circular deps at module level."""
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    try:
        django.setup()
    except Exception:
        pass
    from delegation.models import DelegatedTask, DelegationHost, DelegationLog, TaskResourceProfile
    return DelegationHost, DelegatedTask, DelegationLog, TaskResourceProfile


class DelegationClient:
    """
    Manages all delegation hosts and dispatches tasks intelligently.

    - Heartbeats all registered hosts periodically
    - Selects the best host for a given task based on capabilities + headroom
    - Dispatches tasks and collects results
    - Updates resource profiles from completed tasks
    - Logs all communication
    """

    def __init__(self, secure_env: Optional[Dict[str, str]] = None) -> None:
        self._secure_env = secure_env or {}
        self._lock = threading.Lock()
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._result_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pending_polls: Dict[str, Dict] = {}  # task_id -> {host_id, task_db_id, ...}
        self._on_result_callbacks: List[Callable] = []

    def start(self) -> None:
        """Start heartbeat and result polling threads."""
        self._stop.clear()

        # Rehydrate pending tasks from DB so results survive restarts
        self._rehydrate_pending()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="delegation-heartbeat"
        )
        self._heartbeat_thread.start()

        self._result_thread = threading.Thread(
            target=self._result_poll_loop, daemon=True, name="delegation-results"
        )
        self._result_thread.start()
        logger.info("delegation client started")

    def _rehydrate_pending(self) -> None:
        """Reload unfinished tasks from DB so we resume polling after restart."""
        try:
            _, DelegatedTask, _, _ = _get_db()
            unfinished = DelegatedTask.objects.filter(
                status__in=[
                    DelegatedTask.Status.QUEUED,
                    DelegatedTask.Status.SENT,
                    DelegatedTask.Status.RUNNING,
                ],
            ).select_related("host")

            count = 0
            for t in unfinished:
                if not t.remote_task_id:
                    # Legacy task without remote ID — can't poll, mark failed
                    t.status = DelegatedTask.Status.FAILED
                    t.error_message = "Lost on restart (no remote_task_id)"
                    t.save()
                    continue
                with self._lock:
                    self._pending_polls[t.remote_task_id] = {
                        "host_id": t.host_id,
                        "host_addr": t.host.host,
                        "host_port": t.host.port,
                        "host_token": t.host.api_token,
                        "db_task_id": t.id,
                        "submitted_at": t.sent_at.timestamp() if t.sent_at else t.created_at.timestamp(),
                    }
                count += 1
            if count:
                logger.info("rehydrated %d pending tasks from database", count)
        except Exception as exc:
            logger.warning("failed to rehydrate pending tasks: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        if self._result_thread:
            self._result_thread.join(timeout=5)

    def on_result(self, callback: Callable) -> None:
        """Register a callback for when a delegated task completes."""
        self._on_result_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Host management
    # ------------------------------------------------------------------

    def get_available_hosts(self, task_type: str = "") -> List[Dict]:
        """Return hosts that can accept the given task type."""
        DelegationHost, *_ = _get_db()
        hosts = DelegationHost.objects.filter(
            enabled=True, status=DelegationHost.Status.ONLINE
        )
        available = []
        for h in hosts:
            if not h.is_available:
                continue
            if task_type and task_type not in (h.capabilities or []):
                continue
            available.append({
                "id": h.id,
                "name": h.name,
                "host": h.host,
                "port": h.port,
                "headroom": h.headroom,
                "capabilities": h.capabilities or [],
                "cpu_percent": h.cpu_percent,
                "memory_percent": h.memory_percent,
                "memory_available_mb": h.memory_available_mb,
                "active_tasks": h.active_tasks,
                "max_concurrent_tasks": h.max_concurrent_tasks,
            })
        return available

    def select_best_host(self, task_type: str) -> Optional[Dict]:
        """Pick the best host for a task based on headroom and resources."""
        hosts = self.get_available_hosts(task_type)
        if not hosts:
            return None

        _, _, _, TaskResourceProfile = _get_db()

        # Get the resource profile for this task type (if we have history)
        try:
            profile = TaskResourceProfile.objects.get(task_type=task_type)
            est_memory_mb = profile.avg_peak_memory_mb
        except TaskResourceProfile.DoesNotExist:
            est_memory_mb = 256  # conservative default

        # Score hosts: prefer more headroom + more available memory
        def score(h: Dict) -> float:
            headroom_score = h["headroom"] * 10
            mem_score = max(0, h["memory_available_mb"] - est_memory_mb) / 100
            cpu_penalty = h["cpu_percent"] / 10
            return headroom_score + mem_score - cpu_penalty

        hosts.sort(key=score, reverse=True)
        return hosts[0]

    def pair_host(self, host_id: int) -> Dict:
        """Send pairing request to a host."""
        DelegationHost, _, DelegationLog, _ = _get_db()
        host = DelegationHost.objects.get(pk=host_id)

        resp = self._http_post(
            host.host, host.port, "/pair",
            {"token": host.api_token},
            auth=False,
        )
        if resp and resp.get("paired"):
            host.status = DelegationHost.Status.ONLINE
            host.device_type = resp.get("device", {}).get("device_type", "")
            host.os_name = resp.get("device", {}).get("os_name", "")
            host.cpu_count = resp.get("device", {}).get("cpu_count", 0)
            host.total_memory_mb = resp.get("device", {}).get("total_memory_mb", 0)
            host.python_version = resp.get("device", {}).get("python_version", "")
            host.capabilities = resp.get("capabilities", [])
            host.max_concurrent_tasks = resp.get("max_concurrent_tasks", 1)
            host.last_heartbeat = _now()
            host.save()
            self._log(host, None, "sent", "pair", "pairing successful")
            return {"paired": True, "device": resp.get("device", {})}
        else:
            host.status = DelegationHost.Status.ERROR
            host.last_error = str(resp) if resp else "Connection failed"
            host.save()
            self._log(host, None, "error", "pair", host.last_error)
            return {"paired": False, "error": host.last_error}

    # ------------------------------------------------------------------
    # Task dispatch
    # ------------------------------------------------------------------

    def dispatch(self, task_type: str, payload: Dict, api_key_names: Optional[List[str]] = None) -> Optional[str]:
        """
        Dispatch a task to the best available host.
        Returns the task_id if dispatched, None if no host available.
        """
        host_info = self.select_best_host(task_type)
        if not host_info:
            return None

        DelegationHost, DelegatedTask, DelegationLog, _ = _get_db()
        host = DelegationHost.objects.get(pk=host_info["id"])

        task_id = str(uuid.uuid4())

        # Resolve API keys to send
        api_keys = {}
        for name in (api_key_names or self._default_keys_for_task(task_type)):
            val = self._secure_env.get(name, os.environ.get(name, ""))
            if val:
                api_keys[name] = val

        # Create DB record
        db_task = DelegatedTask.objects.create(
            host=host,
            remote_task_id=task_id,
            task_type=task_type,
            payload=payload,
            api_keys_sent=list(api_keys.keys()),
            status=DelegatedTask.Status.QUEUED,
        )

        # Send to host
        resp = self._http_post(
            host.host, host.port, "/tasks/submit",
            {
                "task_id": task_id,
                "task_type": task_type,
                "payload": payload,
                "api_keys": api_keys,
            },
            token=host.api_token,
        )

        if resp and resp.get("accepted"):
            db_task.status = DelegatedTask.Status.SENT
            db_task.sent_at = _now()
            db_task.save()
            host.active_tasks = min(host.active_tasks + 1, host.max_concurrent_tasks)
            host.save()
            self._log(host, db_task, "sent", "task_submit",
                      f"task_type={task_type} task_id={task_id}")

            # Track for polling
            with self._lock:
                self._pending_polls[task_id] = {
                    "host_id": host.id,
                    "host_addr": host.host,
                    "host_port": host.port,
                    "host_token": host.api_token,
                    "db_task_id": db_task.id,
                    "submitted_at": time.time(),
                }

            logger.info("dispatched %s to %s (task_id=%s)", task_type, host.name, task_id[:8])
            return task_id
        else:
            db_task.status = DelegatedTask.Status.FAILED
            db_task.error_message = str(resp) if resp else "Host rejected task"
            db_task.save()
            self._log(host, db_task, "error", "task_submit", db_task.error_message)
            return None

    def dispatch_batch(self, tasks: List[Tuple[str, Dict]]) -> List[Optional[str]]:
        """Dispatch multiple tasks, distributing across available hosts."""
        results = []
        for task_type, payload in tasks:
            tid = self.dispatch(task_type, payload)
            results.append(tid)
        return results

    def _default_keys_for_task(self, task_type: str) -> List[str]:
        """Return which API key env vars a task type needs."""
        base_keys = ["ALCHEMY_API_KEY"]
        task_keys = {
            "data_ingest": ["ALCHEMY_API_KEY", "ANKR_API_KEY", "THEGRAPH_API_KEY"],
            "news_enrichment": ["CRYPTOPANIC_API_KEY"],
            "candidate_training": [],
            "ghost_trading": ["ALCHEMY_API_KEY"],
            "live_monitoring": ["ALCHEMY_API_KEY"],
        }
        return task_keys.get(task_type, base_keys)

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Periodically heartbeat all enabled hosts."""
        while not self._stop.wait(_HEARTBEAT_INTERVAL):
            try:
                self._heartbeat_all()
            except Exception as exc:
                logger.warning("heartbeat loop error: %s", exc)

    def _heartbeat_all(self) -> None:
        DelegationHost, *_ = _get_db()
        hosts = DelegationHost.objects.filter(enabled=True).exclude(
            status=DelegationHost.Status.PAIRING
        )
        for host in hosts:
            try:
                resp = self._http_post(
                    host.host, host.port, "/heartbeat", {},
                    token=host.api_token,
                )
                if resp and resp.get("status") == "online":
                    host.status = DelegationHost.Status.ONLINE
                    res = resp.get("resources", {})
                    host.cpu_percent = res.get("cpu_percent", 0)
                    host.memory_percent = res.get("memory_percent", 0)
                    host.memory_available_mb = int(res.get("memory_available_mb", 0))
                    host.disk_free_mb = int(res.get("disk_free_mb", 0))
                    host.active_tasks = resp.get("active_tasks", 0)
                    host.max_concurrent_tasks = resp.get("max_concurrent_tasks", 1)
                    host.capabilities = resp.get("capabilities", host.capabilities)
                    host.last_heartbeat = _now()
                    host.last_error = ""
                    host.save()
                else:
                    self._mark_offline(host, "bad heartbeat response")
            except Exception as exc:
                self._mark_offline(host, str(exc))

    def _mark_offline(self, host, reason: str) -> None:
        if host.last_heartbeat:
            from django.utils import timezone
            elapsed = (timezone.now() - host.last_heartbeat).total_seconds()
            if elapsed > _OFFLINE_THRESHOLD:
                host.status = host.Status.OFFLINE
                host.last_error = reason
                host.save()
                logger.warning("host %s marked offline: %s", host.name, reason)

    def _result_poll_loop(self) -> None:
        """Poll pending tasks for completion."""
        while not self._stop.wait(5):
            try:
                self._poll_pending()
            except Exception as exc:
                logger.warning("result poll error: %s", exc)

    def _poll_pending(self) -> None:
        with self._lock:
            pending = dict(self._pending_polls)

        for task_id, info in pending.items():
            try:
                resp = self._http_get(
                    info["host_addr"], info["host_port"],
                    f"/tasks/{task_id}",
                    token=info["host_token"],
                )
                if not resp:
                    # Check timeout — generous for rehydrated tasks
                    elapsed = time.time() - info["submitted_at"]
                    timeout = float(os.getenv("DELEGATION_TASK_TIMEOUT", "3600"))
                    if elapsed > timeout:
                        self._finalize_task(task_id, info, {
                            "status": "timeout",
                            "error": f"Task timed out after {elapsed:.0f}s",
                        })
                    continue

                status = resp.get("status", "")
                if status in ("completed", "failed"):
                    self._finalize_task(task_id, info, resp)
            except Exception as exc:
                logger.debug("poll %s failed: %s", task_id[:8], exc)

    def _finalize_task(self, task_id: str, info: Dict, resp: Dict) -> None:
        """Update DB with completed/failed task result."""
        _, DelegatedTask, _, TaskResourceProfile = _get_db()

        try:
            db_task = DelegatedTask.objects.get(pk=info["db_task_id"])
        except DelegatedTask.DoesNotExist:
            with self._lock:
                self._pending_polls.pop(task_id, None)
            return

        db_task.status = (
            DelegatedTask.Status.COMPLETED if resp.get("status") == "completed"
            else DelegatedTask.Status.FAILED
        )
        db_task.result = resp.get("result", {})
        db_task.result_files = resp.get("result_files", [])
        db_task.error_message = resp.get("error", "")
        db_task.peak_cpu_percent = resp.get("peak_cpu_percent", 0)
        db_task.peak_memory_mb = resp.get("peak_memory_mb", 0)
        db_task.duration_seconds = resp.get("duration_seconds", 0)
        db_task.completed_at = _now()
        db_task.save()

        # Update resource profile
        if db_task.status == DelegatedTask.Status.COMPLETED:
            try:
                profile, _ = TaskResourceProfile.objects.get_or_create(
                    task_type=db_task.task_type
                )
                profile.update_from_task(db_task)
            except Exception:
                pass

        # Update host active count
        try:
            from delegation.models import DelegationHost
            host = DelegationHost.objects.get(pk=info["host_id"])
            host.active_tasks = max(0, host.active_tasks - 1)
            host.save()
        except Exception:
            pass

        with self._lock:
            self._pending_polls.pop(task_id, None)

        # Fire callbacks
        for cb in self._on_result_callbacks:
            try:
                cb(task_id, db_task.task_type, db_task.status, db_task.result)
            except Exception as exc:
                logger.warning("result callback error: %s", exc)

        self._log_from_info(info, db_task, "received", "task_result",
                           f"status={db_task.status} duration={db_task.duration_seconds:.1f}s")

        logger.info(
            "task %s (%s) %s in %.1fs",
            task_id[:8], db_task.task_type, db_task.status, db_task.duration_seconds,
        )

    # ------------------------------------------------------------------
    # HTTP helpers (stdlib only — no requests dependency)
    # ------------------------------------------------------------------

    def _http_get(self, host: str, port: int, path: str, token: str = "") -> Optional[Dict]:
        import urllib.request
        url = f"http://{host}:{port}{path}"
        req = urllib.request.Request(url, method="GET")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("GET %s failed: %s", url, exc)
            return None

    def _http_post(
        self, host: str, port: int, path: str,
        data: Dict, token: str = "", auth: bool = True,
    ) -> Optional[Dict]:
        import urllib.request
        url = f"http://{host}:{port}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if auth and token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.debug("POST %s failed: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, host, task, direction: str, msg_type: str, summary: str) -> None:
        _, _, DelegationLog, _ = _get_db()
        try:
            DelegationLog.objects.create(
                host=host,
                task=task,
                direction=direction,
                message_type=msg_type,
                payload_summary=summary[:500],
                payload_size_bytes=len(summary.encode("utf-8", errors="replace")),
            )
        except Exception:
            pass

    def _log_from_info(self, info: Dict, task, direction: str, msg_type: str, summary: str) -> None:
        DelegationHost, _, DelegationLog, _ = _get_db()
        try:
            host = DelegationHost.objects.get(pk=info["host_id"])
            DelegationLog.objects.create(
                host=host,
                task=task,
                direction=direction,
                message_type=msg_type,
                payload_summary=summary[:500],
                payload_size_bytes=len(summary.encode("utf-8", errors="replace")),
            )
        except Exception:
            pass


def _now():
    from django.utils import timezone
    return timezone.now()
