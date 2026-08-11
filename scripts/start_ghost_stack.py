#!/usr/bin/env python3
"""Idempotently start and verify Django plus the ghost production manager."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import psutil
except Exception:  # pragma: no cover - binary wheels can also be incompatible
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
LOGS = ROOT / "logs"


def http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError):
        return False


def command_running(fragment: str) -> bool:
    if psutil is not None:
        for process in psutil.process_iter(["cmdline"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if fragment.lower() in command.lower():
                return True
        return False
    if os.name == "nt":
        try:
            result = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine",
                ],
                capture_output=True, text=True, timeout=10,
            )
            return fragment.lower() in result.stdout.lower()
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        result = subprocess.run(
            ["ps", "-eo", "args"], capture_output=True, text=True, timeout=10,
        )
        return fragment.lower() in result.stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return False


def fresh_production_heartbeat(max_age: float = 150.0) -> bool:
    path = LOGS / "production_manager_heartbeat.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - float(payload.get("timestamp") or 0.0)
        return age <= max_age and str(payload.get("status") or "").lower() in {"running", "starting"}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def safe_environment() -> dict[str, str]:
    env = os.environ.copy()
    # This admission is explicitly paper-only. Promotion remains possible in
    # configuration, but cannot execute until an operator separately enables
    # the live interlocks after all readiness gates pass.
    env.update({
        "ENABLE_LIVE_TRADING": "0",
        "EXECUTE_LIVE_TRADES": "0",
        "LIVE_TRADES_DRY_RUN": "1",
        "AUTO_PROMOTE_LIVE": "0",
        # Both the Django app registry and the explicit CLI can bootstrap the
        # manager. The launcher owns that decision, so suppress the implicit
        # copy and retain exactly one trading-state writer.
        "PRODUCTION_AUTO_DISABLED": "1",
        # Index freshness work is valuable but must not hold the writer lease
        # for many minutes before the first ghost heartbeat. The existing
        # periodic discovery pipeline remains responsible for refreshes.
        "PAIR_INDEX_MAX_AGE_DAYS": os.getenv("PAIR_INDEX_MAX_AGE_DAYS", "30"),
        "WAITRESS_HOST": "127.0.0.1",
        "WAITRESS_PORT": "8001",
        "WAITRESS_THREADS": os.getenv("WAITRESS_THREADS", "8"),
    })
    return env


def spawn(command: list[str], cwd: Path, log_name: str, env: dict[str, str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    with (LOGS / f"{log_name}.log").open("ab") as stdout, \
            (LOGS / f"{log_name}.err").open("ab") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
    return process.pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    python = Path(sys.executable)
    # This dependency-light probe proves Django can serve requests even while
    # the optional Wizard node is loading a large persisted brain.
    health_url = "http://127.0.0.1:8001/health/guardian/"
    env = safe_environment()
    actions: list[dict[str, object]] = []

    if not http_ok(health_url):
        web_env = dict(env)
        web_env["PRODUCTION_AUTO_DISABLED"] = "1"
        command = [str(python), str(WEB / "run_waitress.py")]
        if args.dry_run:
            actions.append({"service": "django", "command": command})
        else:
            actions.append({
                "service": "django",
                "pid": spawn(command, WEB, "ghost_web_8001", web_env),
            })

    if not command_running("main.py --action start_production"):
        command = [str(python), "-u", str(ROOT / "main.py"), "--action", "start_production", "--stay-alive"]
        if args.dry_run:
            actions.append({"service": "production_manager", "command": command})
        else:
            actions.append({
                "service": "production_manager",
                "pid": spawn(command, ROOT, "ghost_production_manager", env),
            })

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "actions": actions}))
        return 0

    deadline = time.time() + max(1.0, args.timeout)
    while time.time() < deadline:
        web_ready = http_ok(health_url)
        production_ready = fresh_production_heartbeat()
        if web_ready and production_ready:
            print(json.dumps({
                "ok": True,
                "mode": "ghost",
                "web_ready": True,
                "production_ready": True,
                "actions": actions,
            }))
            return 0
        time.sleep(1.0)
    print(json.dumps({
        "ok": False,
        "mode": "ghost",
        "web_ready": http_ok(health_url),
        "production_ready": fresh_production_heartbeat(),
        "actions": actions,
    }))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
