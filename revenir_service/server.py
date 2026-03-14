"""
Revenir Delegation Service — multithreaded HTTP API server.

Runs on delegation hosts. Uses stdlib only (no Flask/FastAPI) for maximum
compatibility with Userland Ubuntu, Termux, and minimal environments.

Port default: 7782 (configurable via --port or REVENIR_PORT env var).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .device_profile import detect_device, recommended_max_workers
from .resource_monitor import ResourceMonitor
from .task_executor import TaskExecutor

logger = logging.getLogger("revenir.server")

DEFAULT_PORT = 7782


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multithreaded HTTP server — each request handled in its own thread."""
    daemon_threads = True
    allow_reuse_address = True


class RevenirHandler(BaseHTTPRequestHandler):
    """API request handler for the delegation service."""

    server: "RevenirServer"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info(fmt, *args)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _json_body(self) -> Dict:
        raw = self._read_body()
        if not raw:
            return {}
        return json.loads(raw)

    def _send_json(self, data: Any, code: int = 200) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        self._send_json({"error": message}, code)

    def _check_auth(self) -> bool:
        """Validate the Bearer token."""
        expected = self.server.revenir_app.api_token
        if not expected:
            return True  # No token set (pairing mode)
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip() == expected
        return False

    def _parse_path(self) -> Tuple[str, Dict]:
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    # --- Routes ---

    def do_GET(self) -> None:
        path, qs = self._parse_path()

        if path == "/health":
            self._send_json({"status": "ok", "version": "0.1.0"})
            return

        if not self._check_auth():
            self._send_error(401, "Unauthorized")
            return

        if path == "/status":
            self._handle_status()
        elif path == "/tasks":
            self._handle_list_tasks()
        elif path.startswith("/tasks/"):
            task_id = path.split("/tasks/", 1)[1]
            self._handle_get_task(task_id)
        elif path == "/profile":
            self._handle_profile()
        elif path == "/resources":
            self._handle_resources()
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        path, qs = self._parse_path()

        if path == "/pair":
            self._handle_pair()
            return

        if not self._check_auth():
            self._send_error(401, "Unauthorized")
            return

        if path == "/tasks/submit":
            self._handle_submit_task()
        elif path == "/tasks/cancel":
            self._handle_cancel_task()
        elif path == "/heartbeat":
            self._handle_heartbeat()
        else:
            self._send_error(404, "Not found")

    # --- Handlers ---

    def _handle_status(self) -> None:
        app = self.server.revenir_app
        snap = app.monitor.snapshot()
        self._send_json({
            "status": "online",
            "device": app.device_info,
            "resources": snap,
            "max_concurrent_tasks": app.executor.max_concurrent,
            "active_tasks": app.executor.active_count,
            "available_slots": app.executor.available_slots,
            "active_task_list": app.executor.active_tasks(),
            "capabilities": app.device_info.get("capabilities", []),
            "uptime_seconds": round(time.time() - app.start_time, 1),
        })

    def _handle_profile(self) -> None:
        """Return device profile and capability details."""
        app = self.server.revenir_app
        self._send_json(app.device_info)

    def _handle_resources(self) -> None:
        """Return current resource snapshot."""
        app = self.server.revenir_app
        snap = app.monitor.snapshot()
        snap["max_concurrent_tasks"] = app.executor.max_concurrent
        snap["active_tasks"] = app.executor.active_count
        snap["can_accept"] = app.executor.can_accept()
        snap["should_throttle"] = app.monitor.should_throttle
        snap["should_pause"] = app.monitor.should_pause
        self._send_json(snap)

    def _handle_list_tasks(self) -> None:
        app = self.server.revenir_app
        self._send_json({"tasks": app.executor.active_tasks()})

    def _handle_get_task(self, task_id: str) -> None:
        app = self.server.revenir_app
        result = app.executor.get_result(task_id)
        if result:
            self._send_json(result)
        else:
            self._send_error(404, f"Task {task_id} not found")

    def _handle_submit_task(self) -> None:
        app = self.server.revenir_app
        body = self._json_body()
        task_id = body.get("task_id", "")
        task_type = body.get("task_type", "")
        payload = body.get("payload", {})
        env_keys = body.get("api_keys", {})

        if not task_id or not task_type:
            self._send_error(400, "task_id and task_type required")
            return

        # Check capability
        caps = app.device_info.get("capabilities", [])
        if task_type not in caps:
            self._send_error(400, f"Task type '{task_type}' not supported by this device")
            return

        ok = app.executor.submit(task_id, task_type, payload, env_keys)
        if ok:
            logger.info("task submitted: %s (%s)", task_id[:8], task_type)
            self._send_json({"accepted": True, "task_id": task_id})
        else:
            self._send_error(503, "At capacity — cannot accept more tasks")

    def _handle_cancel_task(self) -> None:
        # TODO: implement task cancellation via subprocess kill
        self._send_json({"cancelled": False, "reason": "not yet implemented"})

    def _handle_heartbeat(self) -> None:
        """Accept heartbeat from main system, respond with our status."""
        app = self.server.revenir_app
        snap = app.monitor.snapshot()
        self._send_json({
            "status": "online",
            "resources": snap,
            "active_tasks": app.executor.active_count,
            "max_concurrent_tasks": app.executor.max_concurrent,
            "capabilities": app.device_info.get("capabilities", []),
        })

    def _handle_pair(self) -> None:
        """Pairing endpoint — validates token and returns device profile."""
        body = self._json_body()
        token = body.get("token", "")
        app = self.server.revenir_app

        if not app.api_token:
            # First pairing — accept the token
            app.api_token = token
            app.save_config()
            logger.info("paired with main system (token set)")
            self._send_json({
                "paired": True,
                "device": app.device_info,
                "capabilities": app.device_info.get("capabilities", []),
                "max_concurrent_tasks": app.executor.max_concurrent,
            })
        elif token == app.api_token:
            self._send_json({
                "paired": True,
                "device": app.device_info,
                "capabilities": app.device_info.get("capabilities", []),
                "max_concurrent_tasks": app.executor.max_concurrent,
            })
        else:
            self._send_error(403, "Invalid pairing token")


class RevenirServer:
    """Main service coordinator."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        work_dir: Optional[Path] = None,
        api_token: str = "",
        callback_url: str = "",
    ) -> None:
        self.port = port
        self.api_token = api_token
        self.callback_url = callback_url
        self.start_time = time.time()

        # Work directory
        if work_dir:
            self.work_dir = work_dir
        else:
            self.work_dir = self._default_work_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Config file
        self.config_path = self.work_dir / "config.json"
        self._load_config()

        # Device detection
        self.device_info = detect_device()
        max_workers = recommended_max_workers(self.device_info)
        logger.info(
            "device: %s | os=%s | cpus=%d | mem=%dMB | max_workers=%d | caps=%s",
            self.device_info["device_type"],
            self.device_info["os_name"],
            self.device_info["cpu_count"],
            self.device_info["total_memory_mb"],
            max_workers,
            self.device_info["capabilities"],
        )

        # Resource monitor
        self.monitor = ResourceMonitor(sample_interval=3.0)

        # Task executor (multithreaded)
        self.executor = TaskExecutor(
            work_dir=self.work_dir,
            monitor=self.monitor,
            max_concurrent=max_workers,
            callback_url=self.callback_url,
            api_token=self.api_token,
        )

        self._httpd: Optional[ThreadedHTTPServer] = None

    def _default_work_dir(self) -> Path:
        home = Path.home()
        return home / ".revenir"

    def _load_config(self) -> None:
        if self.config_path.exists():
            try:
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                if not self.api_token:
                    self.api_token = cfg.get("api_token", "")
                if not self.callback_url:
                    self.callback_url = cfg.get("callback_url", "")
                saved_port = cfg.get("port")
                if saved_port and self.port == DEFAULT_PORT:
                    self.port = saved_port
            except Exception:
                pass

    def save_config(self) -> None:
        cfg = {
            "api_token": self.api_token,
            "callback_url": self.callback_url,
            "port": self.port,
        }
        self.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def start(self) -> None:
        """Start the multithreaded HTTP server."""
        self.monitor.start()

        self._httpd = ThreadedHTTPServer(("0.0.0.0", self.port), RevenirHandler)
        self._httpd.revenir_app = self  # type: ignore[attr-defined]

        logger.info("Revenir Delegation Service starting on port %d", self.port)
        logger.info("Work directory: %s", self.work_dir)
        if self.api_token:
            logger.info("API token configured (paired)")
        else:
            logger.info("No API token — running in PAIRING MODE")

        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.monitor.stop()
            if self._httpd:
                self._httpd.shutdown()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
        self.monitor.stop()
