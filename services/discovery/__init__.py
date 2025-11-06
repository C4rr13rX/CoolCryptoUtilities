from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_django_ready() -> None:
    """
    Guarantee Django can resolve the `discovery` app when these services are
    imported from outside the web project (e.g., production manager).
    """
    repo_root = Path(__file__).resolve().parents[1]
    web_dir = repo_root / "web"
    if str(web_dir) not in sys.path:
        sys.path.insert(0, str(web_dir))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "coolcrypto_dashboard.settings")

    try:
        import django  # noqa: WPS433
        from django.apps import apps  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Django must be installed to use services.discovery components."
        ) from exc

    if not apps.ready:
        django.setup()


_ensure_django_ready()

__all__ = ["_ensure_django_ready"]
