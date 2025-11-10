from __future__ import annotations

from django.db import models


class CodeGraphCache(models.Model):
    cache_key = models.CharField(max_length=64, unique=True)
    graph = models.JSONField(default=dict)
    files = models.JSONField(default=list)
    generated_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Code Graph Cache"
        verbose_name_plural = "Code Graph Caches"

    def __str__(self) -> str:  # pragma: no cover - human readable
        return f"{self.cache_key} @ {self.updated_at:%Y-%m-%d %H:%M:%S}"
