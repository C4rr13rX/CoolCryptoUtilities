from __future__ import annotations

from django.apps import AppConfig


class OpsConsoleConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "opsconsole"
    verbose_name = "Operations Console"
