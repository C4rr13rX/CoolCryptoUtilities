from __future__ import annotations

import uuid
from pathlib import Path

from django.db import models


def _default_log_path() -> str:
    return str(Path("runtime/branddozer") / f"{uuid.uuid4()}.log")


class BrandProject(models.Model):
    """
    BrandDozer project configuration persisted in the Django database.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    root_path = models.TextField()
    default_prompt = models.TextField(blank=True, default="")
    interjections = models.JSONField(default=list, blank=True)
    interval_minutes = models.PositiveIntegerField(default=120)
    enabled = models.BooleanField(default=False)
    last_run = models.DateTimeField(null=True, blank=True)
    last_ai_generated = models.DateTimeField(null=True, blank=True)
    log_path = models.TextField(default=_default_log_path)
    repo_url = models.TextField(blank=True, default="")
    repo_branch = models.CharField(max_length=120, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.name} ({self.root_path})"
