from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from monitoring_guardian.fallback_server import GuardianFallbackServer
from services.env_loader import resolve_python_bin


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
RUNTIME = ROOT / "runtime" / "guardian"
RECOVERY_STATE = RUNTIME / "recovery.json"
RECOVERY_LOG = RUNTIME / "recovery.log"


class RecoveryCoordinator:
    def __init__(self, *, agent_repair: Optional[Callable[[str], str]] = None) -> None:
        self.agent_repair = agent_repair
        self.host = os.getenv("GUARDIAN_DJANGO_HOST", "127.0.0.1")
        self.port = int(os.getenv("GUARDIAN_DJANGO_PORT", os.getenv("WAITRESS_PORT", "8000")))
        self.health_url = os.getenv("GUARDIAN_DJANGO_HEALTH_URL", f"http://{self.host}:{self.port}/health/guardian/")
        self.fallback = GuardianFallbackServer(RECOVERY_STATE, RECOVERY_LOG, host=self.host, port=self.port)
        self.processes: Dict[str, subprocess.Popen] = {}
        self.attempts: Dict[str, int] = {}
        self.last_attempt: Dict[str, float] = {}
        RUNTIME.mkdir(parents=True, exist_ok=True)
        self._write_state({})

    def close(self) -> None:
        self.fallback.stop()

    def tick(self) -> Dict[str, Any]:
        components = {
            "django": self._django_status(),
            "production_manager": self._production_status(),
            "branddozer": self._branddozer_status(),
        }
        self._write_state(components)
        if components["django"]["status"] != "running":
            self._recover_django(components["django"])
        else:
            self.fallback.stop()
            self.attempts["django"] = 0
        if components["production_manager"]["status"] != "running":
            self._recover_production(components["production_manager"])
        if components["branddozer"]["status"] == "enabled_without_keeper":
            self._ensure_branddozer_keeper()
        return components

    def _probe(self, url: str, timeout: float = 2.5) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return 200 <= int(response.status) < 500
        except urllib.error.HTTPError as exc:
            return 200 <= int(exc.code) < 500
        except Exception:
            return False

    def _django_status(self) -> Dict[str, Any]:
        ok = self._probe(self.health_url)
        return {"status": "running" if ok else "down", "attempts": self.attempts.get("django", 0), "detail": self.health_url}

    def _process_matching(self, *needles: str) -> bool:
        try:
            import psutil
            for proc in psutil.process_iter(["cmdline"]):
                try:
                    line = " ".join(proc.info.get("cmdline") or []).lower()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                if all(needle.lower() in line for needle in needles):
                    return True
        except Exception:
            return False
        return False

    def _production_status(self) -> Dict[str, Any]:
        process_running = self._process_matching("main.py", "start_production")
        running = process_running
        heartbeat = ROOT / "logs" / "production_manager_heartbeat.json"
        age = None
        try:
            payload = json.loads(heartbeat.read_text(encoding="utf-8"))
            age = time.time() - float(payload.get("timestamp") or 0)
            # The heartbeat is the authoritative liveness signal. Process names
            # may be rewritten by Windows process-title helpers.
            running = age < 180
        except Exception:
            pass
        return {"status": "running" if running else "down", "attempts": self.attempts.get("production_manager", 0), "detail": f"heartbeat_age={age} process_seen={process_running}"}

    def _branddozer_status(self) -> Dict[str, Any]:
        try:
            db_path = Path(os.getenv("GUARDIAN_DJANGO_DB", str(ROOT / "storage" / "trading_cache.db")))
            with sqlite3.connect(str(db_path), timeout=5) as connection:
                rows = connection.execute(
                    "SELECT name, last_run, interval_minutes FROM branddozer_brandproject WHERE enabled = 1"
                ).fetchall()
            enabled = [{"name": row[0], "last_run": row[1], "interval_minutes": row[2]} for row in rows]
        except Exception as exc:
            return {"status": "unknown", "attempts": 0, "detail": str(exc)}
        if not enabled:
            return {"status": "idle", "attempts": 0, "detail": "no enabled projects"}
        keeper_heartbeat = RUNTIME / "branddozer-keeper-heartbeat.json"
        keeper = False
        try:
            keeper_data = json.loads(keeper_heartbeat.read_text(encoding="utf-8"))
            keeper = time.time() - float(keeper_data.get("timestamp") or 0) < 60
        except Exception:
            keeper = False
        stale = []
        now = time.time()
        for project in enabled:
            last_value = project.get("last_run")
            try:
                last = datetime.fromisoformat(str(last_value)).timestamp() if last_value else 0
            except Exception:
                last = 0
            max_age = max(600, int(project.get("interval_minutes") or 120) * 60 * 2)
            if last and now - last > max_age:
                stale.append(project.get("name"))
        state = "running" if keeper and not stale else "enabled_without_keeper" if not keeper else "stale"
        return {"status": state, "attempts": self.attempts.get("branddozer", 0), "detail": f"enabled={len(enabled)} stale={stale}"}

    def _cooldown_ready(self, name: str, seconds: float = 10.0) -> bool:
        return time.time() - self.last_attempt.get(name, 0) >= seconds

    def _recover_django(self, status: Dict[str, Any]) -> None:
        if not self._cooldown_ready("django"):
            return
        self.last_attempt["django"] = time.time()
        attempt = self.attempts.get("django", 0) + 1
        self.attempts["django"] = attempt
        self._log(f"Django unavailable; recovery attempt {attempt}")
        self.fallback.start()
        if attempt > 1 and attempt % 2 == 0:
            self._ask_agent("django", status)
        self.fallback.stop()
        self._terminate_matching("run_waitress.py")
        env = dict(os.environ)
        env["WAITRESS_HOST"], env["WAITRESS_PORT"] = self.host, str(self.port)
        log_path = RUNTIME / "waitress-recovery.log"
        with log_path.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [resolve_python_bin(), "run_waitress.py"], cwd=WEB_ROOT, env=env,
                stdout=handle, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.processes["django"] = proc
        for _ in range(20):
            if self._probe(self.health_url, timeout=1.0):
                self._log(f"Django restored on attempt {attempt}; pid={proc.pid}")
                self.attempts["django"] = 0
                return
            time.sleep(0.5)
        self._log(f"Django restart attempt {attempt} failed; fallback page restored")
        self.fallback.start()

    def _recover_production(self, status: Dict[str, Any]) -> None:
        if not self._cooldown_ready("production_manager", 30):
            return
        self.last_attempt["production_manager"] = time.time()
        attempt = self.attempts.get("production_manager", 0) + 1
        self.attempts["production_manager"] = attempt
        if attempt > 1 and attempt % 2 == 0:
            self._ask_agent("production_manager", status)
        log_path = RUNTIME / "production-recovery.log"
        with log_path.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [resolve_python_bin(), "-u", str(ROOT / "main.py"), "--action", "start_production", "--stay-alive"],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.processes["production_manager"] = proc
        self._log(f"Production manager restart attempt {attempt}; pid={proc.pid}")

    def _ensure_branddozer_keeper(self) -> None:
        if not self._cooldown_ready("branddozer", 30):
            return
        self.last_attempt["branddozer"] = time.time()
        log_path = RUNTIME / "branddozer-keeper.log"
        with log_path.open("a", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                [resolve_python_bin(), str(ROOT / "scripts" / "branddozer_keeper.py")], cwd=ROOT,
                stdout=handle, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        self.processes["branddozer"] = proc
        self._log(f"BrandDozer keeper started; pid={proc.pid}")

    def _terminate_matching(self, needle: str) -> None:
        try:
            import psutil
            current = os.getpid()
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    line = " ".join(proc.info.get("cmdline") or []).lower()
                    if proc.pid != current and needle.lower() in line:
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            return
        time.sleep(1)

    def _ask_agent(self, component: str, status: Dict[str, Any]) -> None:
        if not self.agent_repair:
            return
        prompt = (
            f"Guardian recovery for {component} has failed repeatedly. Status: {status}. "
            f"Inspect logs under {RUNTIME}, diagnose the root cause, make the smallest safe repair, "
            "run focused validation, and return concrete evidence. Do not start duplicate services."
        )
        try:
            output = self.agent_repair(prompt)
            self._log(f"C0d3rV2 repair output for {component}:\n{output}")
        except Exception as exc:
            self._log(f"C0d3rV2 repair failed for {component}: {exc}")

    def _write_state(self, components: Dict[str, Any]) -> None:
        payload = {"updated_at": time.time(), "components": components}
        RECOVERY_STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with RECOVERY_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(f"[guardian-recovery] {message}", flush=True)
