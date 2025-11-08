from __future__ import annotations

from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "Dashboard Core"
    _guardian_started = False

    def ready(self):
        if CoreConfig._guardian_started:
            return
        try:
            from services.guardian_supervisor import guardian_supervisor

            guardian_supervisor.start_if_enabled()
            CoreConfig._guardian_started = True
        except Exception:
            pass
