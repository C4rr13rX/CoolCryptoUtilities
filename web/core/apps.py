from __future__ import annotations

import threading
import time

from django.apps import AppConfig
from django.apps import apps


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "Dashboard Core"
    _guardian_started = False

    def ready(self):
        if CoreConfig._guardian_started:
            return
        CoreConfig._guardian_started = True

        def _bootstrap():
            try:
                while not apps.ready:
                    time.sleep(0.05)
                from services.guardian_supervisor import guardian_supervisor

                guardian_supervisor.start_if_enabled()
            except Exception:
                CoreConfig._guardian_started = False

        threading.Thread(target=_bootstrap, name="guardian-bootstrap", daemon=True).start()
