#!/usr/bin/env python3
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


GUARDIAN_FLAG = "--guardian-off"
GUARDIAN_ENV_VAR = "GUARDIAN_AUTO_DISABLED"
PRODUCTION_FLAG = "--production-off"
PRODUCTION_ENV_VAR = "PRODUCTION_AUTO_DISABLED"


def _consume_flag(argv: list[str], flag: str, env_var: str) -> None:
    """
    Convert CLI flags into environment switches before Django settings/apps load.
    This lets us short-circuit auto-bootstraps (guardian/production) early.
    """
    if flag not in argv:
        return
    argv.remove(flag)
    os.environ[env_var] = "1"


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    _consume_flag(sys.argv, GUARDIAN_FLAG, GUARDIAN_ENV_VAR)
    _consume_flag(sys.argv, PRODUCTION_FLAG, PRODUCTION_ENV_VAR)
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it is installed and available on your PYTHONPATH."
        ) from exc
    if any(arg.startswith("runserver") for arg in sys.argv[1:]):
        try:
            from core.logtail import start_log_tails
        except ModuleNotFoundError:
            start_log_tails = None
        if start_log_tails:
            start_log_tails()
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
