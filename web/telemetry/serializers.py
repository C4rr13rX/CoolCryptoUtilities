from __future__ import annotations

from rest_framework import serializers

from .models import FeedbackEvent, MetricEntry, TradeLog


class MetricEntrySerializer(serializers.ModelSerializer):
    meta = serializers.SerializerMethodField()

    class Meta:
        model = MetricEntry
        fields = ["ts", "stage", "category", "name", "value", "meta"]

    def get_meta(self, obj: MetricEntry) -> dict:
        return obj.meta_dict()


class FeedbackEventSerializer(serializers.ModelSerializer):
    details = serializers.SerializerMethodField()

    class Meta:
        model = FeedbackEvent
        fields = ["ts", "source", "severity", "label", "details"]

    def get_details(self, obj: FeedbackEvent) -> dict:
        return obj.details_dict()


class TradeLogSerializer(serializers.ModelSerializer):
    details = serializers.SerializerMethodField()

    class Meta:
        model = TradeLog
        fields = ["ts", "wallet", "chain", "symbol", "action", "status", "details"]

    def get_details(self, obj: TradeLog) -> dict:
        return obj.details_dict()
