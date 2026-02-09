from __future__ import annotations

from django.conf import settings
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


class OhlcvDataset(models.Model):
    chain = models.CharField(max_length=32)
    symbol = models.CharField(max_length=64)
    granularity_seconds = models.PositiveIntegerField(default=300)
    start_ts = models.DateTimeField()
    end_ts = models.DateTimeField()
    bars = models.PositiveIntegerField(default=0)
    source = models.CharField(max_length=64, default="download2000")
    file_path = models.TextField(blank=True, default="")
    checksum = models.CharField(max_length=64, blank=True, default="")
    last_ingested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="ohlcv_ingests",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ohlcv_datasets"
        indexes = [
            models.Index(fields=["chain", "symbol", "granularity_seconds"]),
            models.Index(fields=["start_ts", "end_ts"]),
        ]
        unique_together = ("chain", "symbol", "granularity_seconds", "start_ts", "end_ts")

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.chain}:{self.symbol}@{self.granularity_seconds}s"


class OhlcvBar(models.Model):
    dataset = models.ForeignKey(OhlcvDataset, on_delete=models.CASCADE, related_name="bars_data")
    ts = models.DateTimeField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.FloatField(default=0.0)
    buy_volume = models.FloatField(default=0.0)
    sell_volume = models.FloatField(default=0.0)
    net_volume = models.FloatField(default=0.0)
    vwap = models.FloatField(default=0.0)

    class Meta:
        db_table = "ohlcv_bars"
        indexes = [
            models.Index(fields=["dataset", "ts"]),
            models.Index(fields=["ts"]),
        ]
        unique_together = ("dataset", "ts")
