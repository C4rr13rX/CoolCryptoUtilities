"""
One-shot management-command handler.

There is no shell on Lambda, so ``manage.py migrate`` and ``collectstatic``
need an invocable entry point.  This function runs a whitelisted management
command and returns its captured output.

Whitelisted rather than arbitrary: this function holds database credentials and
is invocable by anything with lambda:InvokeFunction, so letting an event name
any command (``shell``, ``dumpdata``, ``flush``) would turn it into a remote
code-execution and data-exfiltration primitive.

Event shape::

    {"command": "migrate", "args": ["--noinput"]}
    {"command": "showmigrations"}
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "web")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings_lambda"
)

# Redirect the bundle's write targets to /tmp before ANY services.* module
# is imported -- several create directories at import time and /var/task is
# read-only. Must run before django.setup() pulls in the app registry.
from serverless.bootstrap import prepare_writable_dirs  # noqa: E402

prepare_writable_dirs(ROOT)

import django  # noqa: E402

django.setup()

logger = logging.getLogger("serverless.admin")

# Commands that are safe to trigger remotely: schema changes and read-only
# introspection. Anything that dumps data or opens an interpreter stays out.
ALLOWED_COMMANDS = {
    "migrate",
    "showmigrations",
    "collectstatic",
    "check",
    "createcachetable",
}


def lambda_handler(event, context):
    from django.core.management import call_command

    command = str((event or {}).get("command", "")).strip()
    args = list((event or {}).get("args", []) or [])

    if command not in ALLOWED_COMMANDS:
        return {
            "status": "rejected",
            "error": f"command {command!r} is not allowed",
            "allowed": sorted(ALLOWED_COMMANDS),
        }

    out = io.StringIO()
    try:
        call_command(command, *args, stdout=out, stderr=out)
    except Exception as exc:  # noqa: BLE001
        logger.exception("management command failed: %s", command)
        return {
            "status": "error",
            "command": command,
            "error": str(exc),
            "output": out.getvalue(),
        }

    return {"status": "ok", "command": command, "output": out.getvalue()}
