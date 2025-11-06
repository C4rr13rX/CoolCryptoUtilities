from __future__ import annotations

from django.db import models


class DiscoveryEvent(models.Model):
    """
    Record of a newly spotted trading pair/source from external discovery feeds.
    """

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    symbol = models.CharField(max_length=64)
    chain = models.CharField(max_length=64, blank=True, null=True)
    source = models.CharField(max_length=128)

    bull_score = models.FloatField(blank=True, null=True)
    bear_score = models.FloatField(blank=True, null=True)
    liquidity_usd = models.FloatField(blank=True, null=True)
    volume_24h = models.FloatField(blank=True, null=True)
    price_change_24h = models.FloatField(blank=True, null=True)

    metadata = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["created_at"]),
        ]


class HoneypotCheck(models.Model):
    """
    Output of honeypot/security screening for a token.
    """

    created_at = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=64)
    chain = models.CharField(max_length=64, blank=True, null=True)
    verdict = models.CharField(max_length=32)
    confidence = models.FloatField(blank=True, null=True)
    details = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["created_at"]),
        ]


class SwapProbe(models.Model):
    """
    Track on-chain buy/sell probes used to validate if a token can be traded safely.
    """

    created_at = models.DateTimeField(auto_now_add=True)
    symbol = models.CharField(max_length=64)
    chain = models.CharField(max_length=64, blank=True, null=True)

    buy_tx_hash = models.CharField(max_length=128, blank=True, null=True)
    sell_tx_hash = models.CharField(max_length=128, blank=True, null=True)
    buy_amount = models.FloatField(blank=True, null=True)
    sell_amount = models.FloatField(blank=True, null=True)
    success = models.BooleanField(default=False)
    failure_reason = models.TextField(blank=True, null=True)
    gas_cost_native = models.FloatField(blank=True, null=True)
    metadata = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "chain"]),
            models.Index(fields=["created_at"]),
        ]


class DiscoveredToken(models.Model):
    """
    Canonical status of a discovered token/pair.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("validated", "Validated"),
        ("honeypot", "Honeypot"),
        ("rejected", "Rejected"),
        ("promoted", "Promoted"),
    ]

    symbol = models.CharField(max_length=64, unique=True)
    chain = models.CharField(max_length=64, blank=True, null=True)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default="pending")
    first_seen = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    notes = models.TextField(blank=True, null=True)
    metadata = models.JSONField(blank=True, null=True)

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["chain"]),
        ]

