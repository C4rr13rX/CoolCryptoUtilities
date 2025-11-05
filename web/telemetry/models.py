from __future__ import annotations

import json

from django.db import models


class MetricEntry(models.Model):
    ts = models.FloatField()
    stage = models.CharField(max_length=64)
    category = models.CharField(max_length=128)
    name = models.CharField(max_length=128)
    value = models.FloatField()
    meta = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "metrics"
        managed = False
        ordering = ["-ts"]

    def meta_dict(self) -> dict:
        try:
            return json.loads(self.meta or "{}")
        except Exception:
            return {}


class FeedbackEvent(models.Model):
    ts = models.FloatField()
    source = models.CharField(max_length=128)
    severity = models.CharField(max_length=32)
    label = models.CharField(max_length=128)
    details = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "feedback_events"
        managed = False
        ordering = ["-ts"]

    def details_dict(self) -> dict:
        try:
            return json.loads(self.details or "{}")
        except Exception:
            return {}


class TradeLog(models.Model):
    ts = models.FloatField()
    wallet = models.CharField(max_length=128)
    chain = models.CharField(max_length=64)
    symbol = models.CharField(max_length=64)
    action = models.CharField(max_length=32)
    status = models.CharField(max_length=64)
    details = models.TextField(blank=True, null=True)

    class Meta:
        db_table = "trading_ops"
        managed = False
        ordering = ["-ts"]

    def details_dict(self) -> dict:
        try:
            return json.loads(self.details or "{}")
        except Exception:
            return {}


class Advisory(models.Model):
    ts = models.FloatField()
    scope = models.CharField(max_length=128, blank=True, null=True)
    topic = models.CharField(max_length=128)
    severity = models.CharField(max_length=32)
    message = models.TextField()
    recommendation = models.TextField()
    meta = models.TextField(blank=True, null=True)
    resolved = models.BooleanField(default=False)
    resolved_ts = models.FloatField(blank=True, null=True)

    class Meta:
        db_table = "advisories"
        managed = False
        ordering = ["-resolved", "-ts"]

    def meta_dict(self) -> dict:
        try:
            return json.loads(self.meta or "{}")
        except Exception:
            return {}
