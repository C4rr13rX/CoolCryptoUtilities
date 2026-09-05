#!/usr/bin/env python3
"""web/run_waitress.py — serve the Django WSGI app via waitress.

Mirrors the boot sequence of manage.py (sys.path tweak, EnvLoader, dev
defaults) so the production server sees the same environment the dev
server does, then hands off to waitress.serve.

Why this exists
---------------
Django's runserver is documented as not suitable for long-running use:
single-threaded by default, dies silently on certain exceptions, and
ships --noreload behaviour that masks crashes.  The control tower kept
going OFFLINE on port 8000 between supervisor passes.  Waitress is a
pure-Python, multi-threaded, production-grade WSGI server that survives
those failure modes.

Invocation (from web/):
    python run_waitress.py
Env overrides:
    WAITRESS_HOST     default 127.0.0.1
    WAITRESS_PORT     default 8000
    WAITRESS_THREADS  default 8
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Match manage.py: load .env / vault values before Django imports.
from services.env_loader import EnvLoader  # noqa: E402
EnvLoader.load()

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
# Dev-friendly defaults (cookies, SSL).  Mirrors manage.py's behaviour
# when serving locally so the panel works over plain http://127.0.0.1.
os.environ.setdefault("DJANGO_DEBUG", "1")
os.environ.setdefault("DJANGO_SECURE_SSL_REDIRECT", "0")
os.environ.setdefault("DJANGO_SESSION_COOKIE_SECURE", "0")
os.environ.setdefault("DJANGO_CSRF_COOKIE_SECURE", "0")
# Guardian off by default (parity with manage.py).
os.environ.setdefault("GUARDIAN_AUTO_DISABLED", "1")


def _other_waitress_pids() -> "list[int]":
    """Every OTHER run_waitress process on this machine.

    Read from the process table rather than a pidfile: a pidfile records what
    was started, and the thing that matters is what is still running. A stale
    pidfile from a killed process is worse than none.

    Returns an empty list when the table cannot be read -- "I could not check"
    must never become "so I killed things".
    """
    import subprocess

    # Exclude this process AND its ancestors. Start-Process (and any shell
    # wrapper) shows up in the table with run_waitress on its command line
    # too, so filtering only on os.getpid() leaves the launcher that spawned
    # us looking like a duplicate -- and killing it kills us. Measured while
    # writing this: the reaper found its own wrapper, stopped it, and left
    # nothing serving the port.
    me = os.getpid()
    mine = {me}
    try:
        import psutil  # type: ignore

        proc = psutil.Process(me)
        for parent in proc.parents():
            mine.add(parent.pid)
    except Exception:  # noqa: BLE001 - psutil is a convenience, not a dependency
        try:
            mine.add(os.getppid())
        except Exception:  # noqa: BLE001
            pass
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe' or "
             "Name='pythonw.exe'\" | Select-Object ProcessId,CommandLine | "
             "ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=45).stdout or ""
    except Exception:  # noqa: BLE001
        return []
    if not out.strip():
        return []

    import json as _json
    try:
        rows = _json.loads(out)
    except Exception:  # noqa: BLE001
        return []
    if isinstance(rows, dict):
        rows = [rows]

    pids: "list[int]" = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "run_waitress" not in str(row.get("CommandLine") or ""):
            continue
        try:
            pid = int(row.get("ProcessId") or 0)
        except (TypeError, ValueError):
            continue
        if pid and pid not in mine:
            pids.append(pid)
    return pids


def _reap_other_instances() -> None:
    """Stop every other web server before taking the port.

    ONE SERVER, ONE PORT. Only one process can bind :8001, so a second start
    leaves an orphan that serves nobody -- but keeps its connection pool,
    its threads and its database handles alive forever.

    Measured 2026-09-05: FOUR run_waitress instances were running (eight
    processes; each spawns a child), started 134, 73, 71 and 38 minutes
    apart by repeated launcher runs. Three served nothing. Together they held
    436 TCP connections of 1,216 on the machine -- more than the trading
    engine and the agent combined -- and the market data feed, which polls
    the same endpoints, collapsed from 130 ticks per 10 minutes to ZERO,
    logging "network outage detected" while every endpoint was reachable.
    No feed means no entries, no exits, no live trades.

    The launcher's "already running" check could not prevent it: by the time
    a duplicate is detected it has already started. The only place that can
    reliably enforce one instance is the instance itself, at the moment it
    takes the port.

    Disable with WAITRESS_REAP_DUPLICATES=0.
    """
    if (os.environ.get("WAITRESS_REAP_DUPLICATES", "1") or "0").lower() not in {
            "1", "true", "yes", "on"}:
        return

    others = _other_waitress_pids()
    if not others:
        return

    print(f"[run_waitress] found {len(others)} other instance(s): {others} -- "
          f"stopping them so this one owns the port", flush=True)

    import signal

    for pid in others:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[run_waitress]   stopped PID {pid}", flush=True)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            # A process that is already gone is the outcome we wanted; one we
            # may not touch is not ours to worry about.
            print(f"[run_waitress]   could not stop PID {pid}: {exc}", flush=True)

    # Give the OS a moment to release their listening sockets before we bind.
    import time as _time
    _time.sleep(2.0)


def main() -> int:
    host    = os.environ.get("WAITRESS_HOST", "127.0.0.1")
    port    = int(os.environ.get("WAITRESS_PORT", "8000"))

    # Take sole ownership before binding. Whoever starts last wins, which is
    # the right rule: the newest process is running the newest code.
    _reap_other_instances()
    # 24 threads (was 8) — the SPA polls multiple endpoints in parallel
    # (status 5s, training/live 2s, activity ~1.5s, topology 15s) and
    # each Django request blocks on the W1z4rD node which can take
    # 1-3 s under training-load lock contention.  With 8 threads,
    # waitress saturated and the task queue piled up >16 deep before
    # the SPA UI became unresponsive.
    threads = int(os.environ.get("WAITRESS_THREADS", "24"))

    # Late import so any settings/env tweaks above land first.
    from coolcrypto_dashboard.wsgi import application
    from waitress import serve

    print(f"[run_waitress] serving on http://{host}:{port}  threads={threads}",
            flush=True)
    serve(application, host=host, port=port, threads=threads,
            ident="R3V3N!R", expose_tracebacks=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
