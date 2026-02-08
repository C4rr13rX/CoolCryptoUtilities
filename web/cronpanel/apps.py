from __future__ import annotations

from django.apps import AppConfig


class CronPanelConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "cronpanel"
    verbose_name = "Cron Panel"
