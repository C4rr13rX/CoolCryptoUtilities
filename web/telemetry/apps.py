from __future__ import annotations

from django.apps import AppConfig


class TelemetryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "telemetry"
    verbose_name = "Telemetry & Metrics"
