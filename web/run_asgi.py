#!/usr/bin/env python3
"""Serve the Django HTTP application and realtime WebSockets together."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.env_loader import EnvLoader

EnvLoader.load()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
os.environ.setdefault("DJANGO_DEBUG", "1")
os.environ.setdefault("DJANGO_SECURE_SSL_REDIRECT", "0")
os.environ.setdefault("DJANGO_SESSION_COOKIE_SECURE", "0")
os.environ.setdefault("DJANGO_CSRF_COOKIE_SECURE", "0")
os.environ.setdefault("GUARDIAN_AUTO_DISABLED", "1")


def main() -> int:
    from daphne.cli import CommandLineInterface

    host = os.getenv("ASGI_HOST", os.getenv("WAITRESS_HOST", "127.0.0.1"))
    port = os.getenv("ASGI_PORT", os.getenv("WAITRESS_PORT", "8000"))
    # Existing app startup guards use this shared panel-port marker.
    os.environ.setdefault("WAITRESS_PORT", str(port))
    CommandLineInterface().run(["-b", host, "-p", str(port), "coolcrypto_dashboard.asgi:application"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
