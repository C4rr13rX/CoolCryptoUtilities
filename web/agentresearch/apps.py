from __future__ import annotations

import os
import sys
import threading
from django.apps import AppConfig


class AgentResearchConfig(AppConfig):
    """Agent Market Research, Multi-Chain Arbitrage, Ghost Trading & Live Graduation."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "agentresearch"
    verbose_name = "Agent Market Research & Arbitrage"

    def ready(self) -> None:
        argv0 = (sys.argv[0] if sys.argv else "").lower()
        argv = " ".join(sys.argv).lower()
        is_web_server = ("run_waitress.py" in argv0) or ("runserver" in argv and os.environ.get("RUN_MAIN") == "true")
        if not is_web_server:
            return
        if os.environ.get("AGENT_RESEARCH_AUTOSTART", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            return

        def _boot_agent_research() -> None:
            import time

            time.sleep(4.0)
            try:
                from .engine import RESEARCH_COORDINATOR

                RESEARCH_COORDINATOR.ensure_default_agents()
                RESEARCH_COORDINATOR.start_background_loop()
            except Exception:
                pass

        threading.Thread(target=_boot_agent_research, name="agentresearch-autostart", daemon=True).start()
