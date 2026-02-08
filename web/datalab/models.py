from __future__ import annotations

from django.db import models


class NewsSource(models.Model):
    name = models.CharField(max_length=255)
    base_url = models.TextField()
    active = models.BooleanField(default=True)
    parser_config = models.JSONField(default=dict, blank=True)
    last_error = models.TextField(blank=True, default="")
    last_run_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "news_sources"
        indexes = [
            models.Index(fields=["active", "updated_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover
        return self.name


class NewsSourceArticle(models.Model):
    source = models.ForeignKey(NewsSource, on_delete=models.CASCADE, related_name="articles")
    title = models.TextField(blank=True, default="")
    url = models.TextField(blank=True, default="")
    published_at = models.DateTimeField(null=True, blank=True)
    summary = models.TextField(blank=True, default="")
    content = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "news_source_articles"
        indexes = [
            models.Index(fields=["source", "created_at"]),
        ]
