from __future__ import annotations

from django.db.models import Count
from rest_framework import generics, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import FeedbackEvent, MetricEntry, TradeLog
from .serializers import (
    FeedbackEventSerializer,
    MetricEntrySerializer,
    TradeLogSerializer,
)


class MetricsListView(generics.ListAPIView):
    serializer_class = MetricEntrySerializer

    def get_queryset(self):
        qs = MetricEntry.objects.all()
        stage = self.request.query_params.get("stage")
        category = self.request.query_params.get("category")
        if stage:
            qs = qs.filter(stage=stage)
        if category:
            qs = qs.filter(category=category)
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]


class FeedbackListView(generics.ListAPIView):
    serializer_class = FeedbackEventSerializer

    def get_queryset(self):
        qs = FeedbackEvent.objects.all()
        sources = self.request.query_params.getlist("source")
        severity = self.request.query_params.getlist("severity")
        if sources:
            qs = qs.filter(source__in=sources)
        if severity:
            qs = qs.filter(severity__in=[lvl.lower() for lvl in severity])
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]


class TradeLogView(generics.ListAPIView):
    serializer_class = TradeLogSerializer

    def get_queryset(self):
        qs = TradeLog.objects.all()
        wallet = self.request.query_params.get("wallet")
        status_param = self.request.query_params.get("status")
        if wallet:
            qs = qs.filter(wallet=wallet)
        if status_param:
            qs = qs.filter(status=status_param)
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 1000))
        return qs.order_by("-ts")[:limit]


class DashboardSummaryView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        metric_counts = MetricEntry.objects.values("stage").annotate(total=Count("id"))
        feedback_counts = FeedbackEvent.objects.values("severity").annotate(total=Count("id"))
        latest_metrics = MetricEntrySerializer(MetricEntry.objects.order_by("-ts")[:12], many=True).data
        latest_feedback = FeedbackEventSerializer(FeedbackEvent.objects.order_by("-ts")[:10], many=True).data
        recent_trades = TradeLogSerializer(TradeLog.objects.order_by("-ts")[:10], many=True).data

        summary = {
            "metrics_by_stage": list(metric_counts),
            "feedback_by_severity": list(feedback_counts),
            "latest_metrics": latest_metrics,
            "latest_feedback": latest_feedback,
            "recent_trades": recent_trades,
        }
        return Response(summary, status=status.HTTP_200_OK)
