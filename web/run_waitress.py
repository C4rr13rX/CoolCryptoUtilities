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


def main() -> int:
    host    = os.environ.get("WAITRESS_HOST", "127.0.0.1")
    port    = int(os.environ.get("WAITRESS_PORT", "8000"))
    threads = int(os.environ.get("WAITRESS_THREADS", "8"))

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
