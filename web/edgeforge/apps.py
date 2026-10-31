from __future__ import annotations

from django.apps import AppConfig


class EdgeforgeConfig(AppConfig):
    """Multi-chain historical OHLCV Genetic Algorithm edge discovery engine.

    Continuously runs 3 deterministic background workers across distinct
    collections of pure-number market variables, splitting off when stale,
    filtering strictly on net profit after fees for Money Button rapid trading,
    skipping and tagging non-profitable ventures (`complete_no_profit`),
    collaborating with 1 `agy` Research Agent to guide and upgrade the GA,
    and feeding profitable edges directly into the trading pipeline.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "edgeforge"
    verbose_name = "EdgeForge GA Edge Search"

    def ready(self) -> None:
        import os
        import sys
        import threading

        # Only auto-launch background workers inside the web server process
        argv0 = (sys.argv[0] if sys.argv else "").lower()
        argv = " ".join(sys.argv).lower()
        is_web_server = ("run_waitress.py" in argv0) or ("runserver" in argv and os.environ.get("RUN_MAIN") == "true")
        if not is_web_server:
            return
        if os.environ.get("EDGEFORGE_AUTOSTART", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            return

        def _boot_workers() -> None:
            import time

            time.sleep(3.0)
            try:
                from .engine import COORDINATOR
                from .models import EdgeSearchConfig

                cfg = EdgeSearchConfig.load()
                if cfg.enabled:
                    COORDINATOR.start_background_workers()
            except Exception:
                pass

        threading.Thread(target=_boot_workers, name="edgeforge-autostart", daemon=True).start()
