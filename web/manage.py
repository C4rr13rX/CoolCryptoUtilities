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


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Make sure it is installed and available on your PYTHONPATH."
        ) from exc
    if any(arg.startswith("runserver") for arg in sys.argv[1:]):
        from core.logtail import start_log_tails

        start_log_tails()
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
