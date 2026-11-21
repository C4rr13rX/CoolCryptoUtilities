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
    WAITRESS_PORT     default 8001
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


def _other_waitress_pids() -> "list[tuple[int, float]]":
    """Every OTHER (pid, create_time) run_waitress process on this machine in <5ms."""
    me = os.getpid()
    mine = {me}
    try:
        import psutil  # type: ignore

        proc = psutil.Process(me)
        for parent in proc.parents():
            mine.add(parent.pid)
        found: "list[tuple[int, float]]" = []
        for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                pid = int(p.info.get("pid") or 0)
                if not pid or pid in mine:
                    continue
                cmd = " ".join(p.info.get("cmdline") or [])
                if "run_waitress" in cmd:
                    # Also skip if pid is a child/parent wrapper of an already-counted pair
                    found.append((pid, float(p.info.get("create_time") or 0.0)))
            except Exception:
                continue
        return found
    except Exception:  # noqa: BLE001
        return []


def _already_healthy(host: str, port: int) -> str:
    """Return 'ok' if a live Waitress serves WSGI, 'warming' if lazy WSGI is still importing, else ''."""
    import urllib.request
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "localhost", "::") else host
    try:
        with urllib.request.urlopen(f"http://{probe_host}:{port}/healthz", timeout=1.5) as r:
            if 200 <= r.status < 400:
                body = (r.read(32) or b"").decode("utf-8", errors="ignore").strip().lower()
                if body == "ok":
                    return "ok"
                if body == "warming":
                    return "warming"
    except Exception:
        pass
    for path in ("/static/assets/main.css", "/login/"):
        try:
            with urllib.request.urlopen(f"http://{probe_host}:{port}{path}", timeout=1.5) as r:
                if 200 <= r.status < 400:
                    return "ok"
        except Exception:
            continue
    return ""


def _reap_other_instances(host: str, port: int) -> bool:
    """Ensure exactly one healthy Waitress owns the port without watchdog ping-pong kills.
    Returns False if a healthy or actively-booting sibling already owns startup (caller should exit 0)."""
    force = (os.environ.get("WAITRESS_FORCE_RESTART", "0") or "0").lower() in {"1", "true", "yes", "on"}
    state = "" if force else _already_healthy(host, port)
    if state == "ok":
        print(f"[run_waitress] healthy instance already serving http://{host}:{port} -- exiting duplicate launcher", flush=True)
        return False

    if (os.environ.get("WAITRESS_REAP_DUPLICATES", "1") or "0").lower() not in {
            "1", "true", "yes", "on"}:
        return True

    import time as _time
    now = _time.time()
    others = _other_waitress_pids()
    if not others:
        return True

    # Importing Django + trading under CPU load takes 45-90s; killing at 35s caused an infinite
    # restart loop where w1z4rd_supervisor killed every warming instance right before WSGI finished.
    boot_grace_sec = float(os.environ.get("WAITRESS_BOOT_GRACE_SEC", "180.0"))
    if not force:
        if state == "warming":
            warming_siblings = [pid for pid, ctime in others if ctime > 0 and (now - ctime) < boot_grace_sec]
            if warming_siblings:
                print(f"[run_waitress] sibling {warming_siblings} is warming WSGI (<{boot_grace_sec:.0f}s) -- yielding cleanly", flush=True)
                return False
        booting = [pid for pid, ctime in others if ctime > 0 and (now - ctime) < boot_grace_sec and pid != os.getpid()]
        if booting:
            my_ctime = 0.0
            try:
                import psutil  # type: ignore
                my_ctime = psutil.Process(os.getpid()).create_time()
            except Exception:
                pass
            older_booting = [pid for pid, ctime in others if (now - ctime) < boot_grace_sec and ctime < my_ctime]
            if older_booting:
                print(f"[run_waitress] sibling {older_booting} is actively booting (<{boot_grace_sec:.0f}s old) -- yielding cleanly", flush=True)
                return False

    pids = [pid for pid, _ in others]
    print(f"[run_waitress] found {len(pids)} stale instance(s): {pids} -- "
          f"stopping them so this one owns the port", flush=True)

    import signal

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[run_waitress]   stopped PID {pid}", flush=True)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            print(f"[run_waitress]   could not stop PID {pid}: {exc}", flush=True)

    _time.sleep(1.0)
    return True


def main() -> int:
    import threading

    # Fine-grained 0.5 ms GIL switch quantum so in-process EdgeForge GA background
    # threads never stall HTTP request threads for 15-60 ms GIL slices.
    try:
        sys.setswitchinterval(0.0005)
    except Exception:
        pass

    host    = os.environ.get("WAITRESS_HOST", "127.0.0.1")
    port    = int(os.environ.get("WAITRESS_PORT", "8001"))

    if not _reap_other_instances(host, port):
        return 0
    threads = max(48, int(os.environ.get("WAITRESS_THREADS", "48")))

    # Ensure web/db.sqlite3 runs in WAL mode so concurrent EdgeForge writes never block HTTP readers.
    try:
        import sqlite3
        _db_file = BASE_DIR / "db.sqlite3"
        if _db_file.is_file():
            with sqlite3.connect(str(_db_file), timeout=5.0) as _conn:
                _conn.execute("PRAGMA journal_mode=WAL;")
                _conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass

    from waitress import serve

    _real_app = None
    _boot_err: list[BaseException] = []
    _ready = threading.Event()

    def _load_django_wsgi() -> None:
        nonlocal _real_app
        try:
            from coolcrypto_dashboard.wsgi import application
            _real_app = application
            print("[run_waitress] Django WSGI application ready", flush=True)
            try:
                from telemetry.views import BusScheduleView

                BusScheduleView()._build_bus_schedule_response()
                print("[run_waitress] BusScheduleView cache pre-warmed", flush=True)

                def _keep_bus_schedule_warm() -> None:
                    import time as _keepwarm_time

                    interval_sec = max(5.0, float(os.getenv("BUS_SCHEDULE_KEEPWARM_SEC", "12.0") or 12.0))
                    while True:
                        _keepwarm_time.sleep(interval_sec)
                        try:
                            BusScheduleView()._build_bus_schedule_response()
                        except Exception:  # noqa: BLE001
                            pass

                threading.Thread(
                    target=_keep_bus_schedule_warm,
                    name="bus-sched-keepwarm",
                    daemon=True,
                ).start()
            except Exception as warm_exc:  # noqa: BLE001
                print(f"[run_waitress] BusScheduleView pre-warm skipped: {warm_exc}", flush=True)
        except BaseException as exc:  # noqa: BLE001
            _boot_err.append(exc)
            print(f"[run_waitress] FATAL WSGI import error: {exc}", flush=True)
        finally:
            _ready.set()

    threading.Thread(target=_load_django_wsgi, name="wsgi-lazy-boot", daemon=True).start()

    def _wsgi_dispatch(environ, start_response):
        path = environ.get("PATH_INFO") or "/"
        ua = environ.get("HTTP_USER_AGENT") or ""
        cookie = environ.get("HTTP_COOKIE") or ""
        # Ultra-fast (<0.1ms) liveness answer for watchdog probes (`Python-urllib`)
        # both during background WSGI warmup and under heavy trading load.
        if path in ("/", "/healthz") and ua.startswith("Python-urllib/") and not cookie:
            if _boot_err:
                start_response("500 Internal Server Error", [("Content-Type", "text/plain; charset=utf-8")])
                return [b"wsgi boot failed"]
            start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8"), ("Cache-Control", "no-store")])
            return [b"ok" if _real_app is not None else b"warming"]

        if _real_app is None:
            _ready.wait(timeout=75.0)
        if _real_app is None:
            start_response("503 Service Unavailable", [("Content-Type", "application/json"), ("Retry-After", "2")])
            return [b'{"ok":false,"warming_up":true}']
        return _real_app(environ, start_response)

    print(f"[run_waitress] serving on http://{host}:{port}  threads={threads} (instant-bind lazy WSGI)",
            flush=True)
    serve(_wsgi_dispatch, host=host, port=port, threads=threads,
            ident="R3V3N!R", expose_tracebacks=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
