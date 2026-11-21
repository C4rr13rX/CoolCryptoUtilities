#!/usr/bin/env python3
"""web/edgeforge/worker_daemon.py

Isolated OS subprocess daemon for EdgeForge's 3 multi-chain OHLCV GA workers
and AGY research agent.

Why this exists:
  Running 3 CPU-bound NumPy/JSON/SQLite GA worker threads inside Waitress's
  Python interpreter held the single process-wide GIL for 10-15s bursts,
  creating p99 latency spikes on `/api/wizard-chat/*` and tripping watchdog
  health checks. Moving EdgeForge into its own dedicated OS process at
  BELOW_NORMAL_PRIORITY_CLASS gives EdgeForge full uninterrupted throughput
  (and persistence across Waitress reloads) while keeping Waitress's GIL
  100% free for sub-15ms HTTP serving.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

WEB_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WEB_DIR.parent
STATE_FILE = PROJECT_ROOT / "runtime" / "edgeforge_worker_state.json"
LOG_FILE = PROJECT_ROOT / "logs" / "edgeforge_worker.log"


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        import psutil
        p = psutil.Process(pid)
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return False
        cmd = " ".join(p.cmdline() or []).lower()
        return "worker_daemon" in cmd and "edgeforge" in cmd
    except Exception:
        return False


def read_external_worker_state() -> Dict[str, Any]:
    """Read latest heartbeat snapshot from the isolated EdgeForge worker process (<0.2ms)."""
    if not STATE_FILE.is_file():
        return {"running": False, "pid": None, "evaluations_per_min": 0.0, "chains_loaded": {}, "total_pairs_loaded": 0, "total_bars_loaded": 0}
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"running": False, "pid": None}
        pid = int(data.get("pid") or 0)
        age = time.time() - float(data.get("updated_at") or 0.0)
        alive = _pid_alive(pid) and age < 45.0
        data["running"] = bool(data.get("running")) and alive
        return data
    except Exception:
        return {"running": False, "pid": None}


def ensure_external_worker_running() -> Dict[str, Any]:
    """Launch the isolated EdgeForge OS subprocess if not already running."""
    cur = read_external_worker_state()
    if cur.get("running") and _pid_alive(int(cur.get("pid") or 0)):
        return cur

    # Also scan for an actively booting worker_daemon (<30s old) to avoid duplicate launches
    try:
        import psutil
        now = time.time()
        for p in psutil.process_iter(["pid", "cmdline", "create_time"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if "worker_daemon" in cmd and "edgeforge" in cmd and (now - float(p.info.get("create_time") or 0.0)) < 60.0:
                    return {"running": True, "pid": p.info["pid"], "booting": True}
            except Exception:
                continue
    except Exception:
        pass

    env = os.environ.copy()
    env["EDGEFORGE_AUTOSTART"] = "0"
    env["EDGEFORGE_ISOLATED_WORKER"] = "1"
    env["PYTHONUTF8"] = "1"
    env.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    creation = 0
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS

    with LOG_FILE.open("ab") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve())],
            cwd=str(WEB_DIR),
            env=env,
            stdout=logf,
            stderr=logf,
            stdin=subprocess.DEVNULL,
            creationflags=creation,
            close_fds=True,
        )
    return {"running": True, "pid": proc.pid, "booting": True}


def stop_external_worker() -> Dict[str, Any]:
    """Gracefully terminate any running isolated EdgeForge worker process."""
    stopped = []
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                if "worker_daemon" in cmd and "edgeforge" in cmd:
                    os.kill(p.info["pid"], signal.SIGTERM)
                    stopped.append(p.info["pid"])
            except Exception:
                continue
    except Exception:
        pass
    try:
        if STATE_FILE.is_file():
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            data["running"] = False
            data["updated_at"] = time.time()
            STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {"running": False, "stopped_pids": stopped}


def _write_state(payload: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def main() -> int:
    for p in (str(WEB_DIR), str(PROJECT_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    os.environ["EDGEFORGE_AUTOSTART"] = "0"
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

    from services.env_loader import EnvLoader
    EnvLoader.load()

    import django
    django.setup()

    from edgeforge.engine import COORDINATOR, CORPUS, SUPPORTED_CHAINS
    from edgeforge.models import EdgeSearchConfig

    stop_requested = False

    def _on_sig(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    cfg = EdgeSearchConfig.load()
    if not cfg.enabled:
        print("[edgeforge-worker] EdgeSearchConfig.enabled is False; exiting.", flush=True)
        return 0

    print(f"[edgeforge-worker] Starting isolated EdgeForge GA daemon (PID={os.getpid()})", flush=True)
    COORDINATOR.start_background_workers()

    try:
        while not stop_requested:
            cfg = EdgeSearchConfig.load()
            if not cfg.enabled:
                print("[edgeforge-worker] Disabled via config; stopping workers.", flush=True)
                break
            chain_counts = CORPUS.ensure_loaded(
                chains=cfg.chains or SUPPORTED_CHAINS,
                max_pairs_per_chain=cfg.max_pairs_per_chain,
                clip_usd=cfg.clip_usd,
                blocking=False,
            )
            _write_state({
                "pid": os.getpid(),
                "running": bool(COORDINATOR.running),
                "evaluations_per_min": round(float(COORDINATOR.throughput_per_minute() or 0.0), 2),
                "chains_loaded": chain_counts,
                "total_pairs_loaded": int(CORPUS.total_pairs_loaded or 0),
                "total_bars_loaded": int(CORPUS.total_bars_loaded or 0),
                "updated_at": time.time(),
            })
            time.sleep(5.0)
    finally:
        try:
            COORDINATOR.stop_background_workers()
        except Exception:
            pass
        try:
            _write_state({
                "pid": os.getpid(),
                "running": False,
                "evaluations_per_min": 0.0,
                "chains_loaded": {},
                "total_pairs_loaded": int(CORPUS.total_pairs_loaded or 0),
                "total_bars_loaded": int(CORPUS.total_bars_loaded or 0),
                "updated_at": time.time(),
            })
        except Exception:
            pass
        print("[edgeforge-worker] Exited cleanly.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
