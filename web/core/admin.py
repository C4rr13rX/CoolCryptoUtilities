from __future__ import annotations

from django.contrib import admin

from .models import (
    Advisory,
    Balance,
    ControlFlag,
    Experiment,
    FeedbackEvent,
    KvStore,
    MarketStream,
    MetricEntry,
    ModelVersion,
    OrganismSnapshot,
    PairAdjustment,
    PairSuppression,
    Price,
    TradeFill,
    TradingOp,
    Transfer,
)


@admin.register(KvStore)
class KvStoreAdmin(admin.ModelAdmin):
    list_display = ("key",)
    search_fields = ("key",)


@admin.register(ControlFlag)
class ControlFlagAdmin(admin.ModelAdmin):
    list_display = ("key", "value", "updated")
    search_fields = ("key", "value")
    ordering = ("-updated",)


@admin.register(Balance)
class BalanceAdmin(admin.ModelAdmin):
    list_display = ("wallet", "chain", "token", "quantity", "usd_amount", "updated_at")
    search_fields = ("wallet", "chain", "token", "symbol")
    list_filter = ("chain",)


@admin.register(Transfer)
class TransferAdmin(admin.ModelAdmin):
    list_display = ("id", "wallet", "chain", "token", "ts", "value")
    search_fields = ("id", "wallet", "chain", "token", "hash")
    list_filter = ("chain",)


@admin.register(Price)
class PriceAdmin(admin.ModelAdmin):
    list_display = ("chain", "token", "usd", "source", "ts")
    search_fields = ("chain", "token", "source")
    list_filter = ("chain",)


@admin.register(TradingOp)
class TradingOpAdmin(admin.ModelAdmin):
    list_display = ("ts", "wallet", "chain", "symbol", "action", "status")
    search_fields = ("wallet", "chain", "symbol", "action", "status")
    ordering = ("-ts",)


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "created", "updated")
    search_fields = ("name", "status")
    ordering = ("-updated",)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("version", "is_active", "created")
    list_filter = ("is_active",)
    ordering = ("-created",)


@admin.register(MarketStream)
class MarketStreamAdmin(admin.ModelAdmin):
    list_display = ("ts", "symbol", "chain", "price", "volume")
    search_fields = ("symbol", "chain")
    ordering = ("-ts",)


@admin.register(TradeFill)
class TradeFillAdmin(admin.ModelAdmin):
    list_display = ("ts", "symbol", "chain", "expected_amount", "executed_amount")
    search_fields = ("symbol", "chain")
    ordering = ("-ts",)


@admin.register(MetricEntry)
class MetricEntryAdmin(admin.ModelAdmin):
    list_display = ("ts", "stage", "category", "name", "value")
    search_fields = ("stage", "category", "name")
    ordering = ("-ts",)


@admin.register(FeedbackEvent)
class FeedbackEventAdmin(admin.ModelAdmin):
    list_display = ("ts", "source", "severity", "label")
    search_fields = ("source", "label")
    list_filter = ("severity",)
    ordering = ("-ts",)


@admin.register(Advisory)
class AdvisoryAdmin(admin.ModelAdmin):
    list_display = ("ts", "scope", "topic", "severity", "resolved")
    search_fields = ("scope", "topic", "message")
    list_filter = ("severity", "resolved")
    ordering = ("-ts",)


@admin.register(OrganismSnapshot)
class OrganismSnapshotAdmin(admin.ModelAdmin):
    list_display = ("ts",)
    ordering = ("-ts",)


@admin.register(PairSuppression)
class PairSuppressionAdmin(admin.ModelAdmin):
    list_display = ("symbol", "reason", "strikes", "release_ts")
    search_fields = ("symbol", "reason")


@admin.register(PairAdjustment)
class PairAdjustmentAdmin(admin.ModelAdmin):
    list_display = ("symbol", "priority", "allocation_multiplier", "label_scale", "updated")
    search_fields = ("symbol",)
