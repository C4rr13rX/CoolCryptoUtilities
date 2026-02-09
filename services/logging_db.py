from __future__ import annotations

def write_db_log(record) -> None:
    try:
        from django.conf import settings
        if not settings.configured:
            return
    except Exception:
        return
    try:
        import django
        from django.apps import apps
        if not apps.ready:
            django.setup()
    except Exception:
        return
    try:
        from django.db import transaction
        from core.models import SystemLog
    except Exception:
        return
    try:
        with transaction.atomic():
            SystemLog.objects.create(
                component=record.source,
                severity=record.severity,
                message=record.message,
                details=record.details or {},
            )
            stale_ids = list(
                SystemLog.objects.filter(component=record.source)
                .order_by("-created_at")
                .values_list("id", flat=True)[100:]
            )
            if stale_ids:
                SystemLog.objects.filter(id__in=stale_ids).delete()
    except Exception:
        return
