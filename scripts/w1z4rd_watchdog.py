"""w1z4rd_watchdog.py — keep the autonomous trading stack alive.

The autonomous-mode promise is "set it and forget it" — the bot
graduates ghost → live by itself, then keeps trading.  That promise
falls apart if the production manager or brain feeder die and don't
come back.  On Windows we've seen both crash on Python 3.13 ⨯ 3.12
DLL conflicts, asyncio socket cleanup, and other one-off events.

This watchdog:
  * polls every WATCHDOG_INTERVAL_SEC (default 60s)
  * counts how many `main.py --action start_production` and
    `run_brain_feeder.py` processes are alive
  * if zero of either kind, respawns it via Start-Process so the
    new child survives this watchdog dying
  * if the brain HTTP endpoint is unreachable, logs it (does not
    auto-restart — brain restart needs careful cold-tier handling)

Idempotent: launching this script while another instance is already
running is a no-op (single PID file at runtime/watchdog.pid).
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Set

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
RUNTIME.mkdir(exist_ok=True)
PID_PATH = RUNTIME / "watchdog.pid"
LOG_PATH = ROOT / "logs" / "watchdog.log"
LOG_PATH.parent.mkdir(exist_ok=True)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        pass
    try:
        print(line, end="", flush=True)
    except Exception:
        pass


def _alive_pids_matching(needles: Set[str]) -> Set[int]:
    """Return PIDs of python.exe processes whose CommandLine contains
    any of `needles`.  Uses PowerShell so it sees full command lines."""
    if not needles:
        return set()
    script = (
        "Get-CimInstance Win32_Process -Filter 'Name=\"python.exe\"' | "
        "ForEach-Object { Write-Host $_.ProcessId '|' $_.CommandLine }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        )
    except Exception:
        return set()
    found: Set[int] = set()
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid_str, _, cmdline = line.partition("|")
        try:
            pid = int(pid_str.strip())
        except Exception:
            continue
        cl_low = cmdline.lower()
        for needle in needles:
            if needle.lower() in cl_low:
                found.add(pid)
                break
    return found


def _spawn(args: list, label: str, log_stem: str) -> None:
    """Spawn a process via PowerShell Start-Process so it outlives
    this script if the watchdog itself dies."""
    out_path = ROOT / "logs" / f"{log_stem}.log"
    err_path = ROOT / "logs" / f"{log_stem}.err"
    # Build the argument list literal for Start-Process
    arg_list = ",".join(f"'{a}'" for a in args)
    cmd = (
        f"Start-Process -FilePath '{sys.executable}' "
        f"-ArgumentList {arg_list} "
        f"-WorkingDirectory '{ROOT}' "
        f"-WindowStyle Hidden "
        f"-RedirectStandardOutput '{out_path}' "
        f"-RedirectStandardError '{err_path}'"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            check=False, timeout=20,
        )
        _log(f"spawned {label}")
    except Exception as exc:
        _log(f"spawn {label} failed: {exc}")


def _brain_alive() -> bool:
    url = os.environ.get("BRAIN_HEALTH_URL", "http://127.0.0.1:8090/health")
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _pidfile_alive() -> bool:
    """True if another watchdog is already running (per PID file)."""
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
    except Exception:
        return False
    # Cheap liveness check via tasklist
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        return str(pid) in out
    except Exception:
        return False


def main() -> int:
    if _pidfile_alive():
        print("watchdog already running; exiting")
        return 0
    try:
        PID_PATH.write_text(str(os.getpid()))
    except Exception:
        pass

    interval = int(os.environ.get("WATCHDOG_INTERVAL_SEC", "60"))
    grace = int(os.environ.get("WATCHDOG_SPAWN_GRACE_SEC", "180"))
    _log(f"watchdog start pid={os.getpid()} interval={interval}s grace={grace}s")

    prod_args = ["-u", "main.py", "--action", "start_production", "--stay-alive"]
    feeder_args = ["scripts/run_brain_feeder.py"]
    # Grace timestamps so we don't respawn the same role while the
    # last spawn is still booting (Python + Django init can take ~60s).
    last_spawn: dict = {"prod": 0.0, "feeder": 0.0}

    while True:
        try:
            now = time.time()
            prod_alive       = _alive_pids_matching({"main.py", "start_production"})
            feeder_alive     = _alive_pids_matching({"run_brain_feeder.py", "brain_feeder.py"})
            supervisor_alive = _alive_pids_matching({"brain_history_supervisor"})
            brain_ok         = _brain_alive()

            # NO dedupe: main.py spawns its own worker subprocesses
            # (parent supervisor + per-cycle workers). They share the
            # cmdline substring we match on. Killing "duplicates"
            # actually kills the worker subprocesses the parent depends
            # on, causing cascading restart cycles (observed all day
            # 2026-06-20 -- the parent died seconds after we killed
            # its workers). Multiple matching PIDs is the NORMAL
            # state; only zero matching PIDs means it needs respawn.

            if not prod_alive and (now - last_spawn["prod"]) > grace:
                _log("production manager DOWN — respawning")
                _spawn(prod_args, "production_manager", "prod_manager_wd")
                last_spawn["prod"] = now
            if not feeder_alive and (now - last_spawn["feeder"]) > grace:
                # Skip respawning brain_feeder while a history supervisor
                # is actively training -- they both pound /brain/observe
                # and contend for the brain mutex, dropping ~5% of
                # supervisor pushes when both run together.
                if supervisor_alive:
                    pass  # supervisor will release the lock when done
                else:
                    _log("brain feeder DOWN -- respawning")
                    _spawn(feeder_args, "brain_feeder", "brain_feeder_wd")
                    last_spawn["feeder"] = now
            if not brain_ok:
                _log("WARN brain HTTP unreachable (no auto-restart)")

            # Quiet heartbeat once per ~hour
            if int(now) % 3600 < interval:
                _log(f"heartbeat prod={len(prod_alive)} feeder={len(feeder_alive)} brain_ok={brain_ok}")
        except Exception as exc:
            _log(f"loop error: {exc}")
        time.sleep(interval)


def _kill_duplicates(pids: Set[int], *, label: str) -> None:
    """Keep the oldest PID alive (likely the first-spawned, most-
    initialized), kill the newer duplicates."""
    if len(pids) <= 1:
        return
    keep = min(pids)  # PIDs roughly correlate to launch order
    for pid in pids:
        if pid == keep:
            continue
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           check=False, timeout=10,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _log(f"killed duplicate {label} pid={pid} (kept {keep})")
        except Exception as exc:
            _log(f"failed to kill duplicate {label} pid={pid}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
