from __future__ import annotations

from django.apps import AppConfig


class ServerlessConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "serverless"
    verbose_name = "Serverless runtime"
