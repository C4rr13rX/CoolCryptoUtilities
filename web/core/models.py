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


# Trading data models (unified with Django DB)


class KvStore(models.Model):
    key = models.CharField(primary_key=True, max_length=255)
    value = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "kv_store"
        verbose_name = "KV Store"
        verbose_name_plural = "KV Store"


class ControlFlag(models.Model):
    key = models.CharField(primary_key=True, max_length=255)
    value = models.CharField(max_length=255, null=True, blank=True)
    updated = models.FloatField(default=0.0)

    class Meta:
        db_table = "control_flags"
        verbose_name = "Control Flag"
        verbose_name_plural = "Control Flags"


class Balance(models.Model):
    wallet = models.CharField(max_length=255)
    chain = models.CharField(max_length=64)
    token = models.CharField(max_length=255)
    balance_hex = models.CharField(max_length=255, null=True, blank=True)
    asof_block = models.BigIntegerField(null=True, blank=True)
    ts = models.FloatField(null=True, blank=True)
    decimals = models.IntegerField(null=True, blank=True)
    quantity = models.CharField(max_length=255, null=True, blank=True)
    usd_amount = models.CharField(max_length=255, null=True, blank=True)
    symbol = models.CharField(max_length=64, null=True, blank=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    updated_at = models.CharField(max_length=255, null=True, blank=True)
    stale = models.IntegerField(default=0)

    class Meta:
        db_table = "balances"
        unique_together = ("wallet", "chain", "token")
        indexes = [
            models.Index(fields=["wallet", "chain"]),
        ]


class Transfer(models.Model):
    id = models.CharField(primary_key=True, max_length=255)
    wallet = models.CharField(max_length=255)
    chain = models.CharField(max_length=64)
    hash = models.CharField(max_length=255, null=True, blank=True)
    log_index = models.BigIntegerField(null=True, blank=True)
    block = models.BigIntegerField(null=True, blank=True)
    ts = models.CharField(max_length=64, null=True, blank=True)
    from_addr = models.CharField(max_length=255, null=True, blank=True)
    to_addr = models.CharField(max_length=255, null=True, blank=True)
    token = models.CharField(max_length=255, null=True, blank=True)
    value = models.CharField(max_length=255, null=True, blank=True)
    inserted_at = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "transfers"
        indexes = [
            models.Index(fields=["wallet", "chain"]),
            models.Index(fields=["token"]),
        ]


class Price(models.Model):
    chain = models.CharField(max_length=64)
    token = models.CharField(max_length=128)
    usd = models.CharField(max_length=255, null=True, blank=True)
    source = models.CharField(max_length=255, null=True, blank=True)
    ts = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "prices"
        unique_together = ("chain", "token")
        indexes = [
            models.Index(fields=["chain", "token"]),
        ]


class TradingOp(models.Model):
    ts = models.FloatField(null=True, blank=True)
    wallet = models.CharField(max_length=255, null=True, blank=True)
    chain = models.CharField(max_length=64, null=True, blank=True)
    symbol = models.CharField(max_length=128, null=True, blank=True)
    action = models.CharField(max_length=64, null=True, blank=True)
    status = models.CharField(max_length=64, null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "trading_ops"
        indexes = [
            models.Index(fields=["ts"]),
            models.Index(fields=["wallet", "chain"]),
        ]


class Experiment(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(max_length=64, null=True, blank=True)
    params = models.JSONField(default=dict, null=True, blank=True)
    results = models.JSONField(default=dict, null=True, blank=True)
    created = models.FloatField(null=True, blank=True)
    updated = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "experiments"


class ModelVersion(models.Model):
    version = models.CharField(max_length=255, null=True, blank=True)
    created = models.FloatField(null=True, blank=True)
    metrics = models.JSONField(default=dict, null=True, blank=True)
    path = models.CharField(max_length=255, null=True, blank=True)
    is_active = models.BooleanField(default=False)

    class Meta:
        db_table = "model_versions"
        indexes = [
            models.Index(fields=["is_active"]),
        ]


class MarketStream(models.Model):
    ts = models.FloatField(null=True, blank=True)
    chain = models.CharField(max_length=64, null=True, blank=True)
    symbol = models.CharField(max_length=128, null=True, blank=True)
    price = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    raw = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "market_stream"
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["ts"]),
        ]


class TradeFill(models.Model):
    ts = models.FloatField(null=True, blank=True)
    chain = models.CharField(max_length=64, null=True, blank=True)
    symbol = models.CharField(max_length=128, null=True, blank=True)
    expected_amount = models.FloatField(null=True, blank=True)
    executed_amount = models.FloatField(null=True, blank=True)
    expected_price = models.FloatField(null=True, blank=True)
    executed_price = models.FloatField(null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "trade_fills"
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["ts"]),
        ]


class MetricEntry(models.Model):
    ts = models.FloatField(null=True, blank=True)
    stage = models.CharField(max_length=64, null=True, blank=True)
    category = models.CharField(max_length=128, null=True, blank=True)
    name = models.CharField(max_length=128, null=True, blank=True)
    value = models.FloatField(null=True, blank=True)
    meta = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "metrics"
        indexes = [
            models.Index(fields=["stage", "ts"]),
            models.Index(fields=["category"]),
        ]


class FeedbackEvent(models.Model):
    ts = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=128, null=True, blank=True)
    severity = models.CharField(max_length=32, null=True, blank=True)
    label = models.CharField(max_length=128, null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "feedback_events"
        indexes = [
            models.Index(fields=["source", "ts"]),
        ]


class Advisory(models.Model):
    ts = models.FloatField(null=True, blank=True)
    scope = models.CharField(max_length=128, null=True, blank=True)
    topic = models.CharField(max_length=128, null=True, blank=True)
    severity = models.CharField(max_length=32, null=True, blank=True)
    message = models.TextField(null=True, blank=True)
    recommendation = models.TextField(null=True, blank=True)
    meta = models.JSONField(default=dict, null=True, blank=True)
    resolved = models.BooleanField(default=False)
    resolved_ts = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "advisories"
        indexes = [
            models.Index(fields=["resolved", "ts"]),
        ]


class OrganismSnapshot(models.Model):
    ts = models.FloatField(primary_key=True)
    payload = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "organism_snapshots"


class PairSuppression(models.Model):
    symbol = models.CharField(primary_key=True, max_length=128)
    reason = models.CharField(max_length=255, null=True, blank=True)
    strikes = models.IntegerField(default=1)
    last_failure = models.FloatField(null=True, blank=True)
    release_ts = models.FloatField(null=True, blank=True)
    metadata = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "pair_suppression"


class PairAdjustment(models.Model):
    symbol = models.CharField(primary_key=True, max_length=128)
    priority = models.IntegerField(default=0)
    enter_offset = models.FloatField(default=0.0)
    exit_offset = models.FloatField(default=0.0)
    size_multiplier = models.FloatField(default=1.0)
    margin_offset = models.FloatField(default=0.0)
    allocation_multiplier = models.FloatField(default=1.0)
    label_scale = models.FloatField(default=1.0)
    updated = models.FloatField(null=True, blank=True)
    details = models.JSONField(default=dict, null=True, blank=True)

    class Meta:
        db_table = "pair_adjustments"
