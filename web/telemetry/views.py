from __future__ import annotations

import sys
from pathlib import Path

from django.db.models import Count
from rest_framework import generics, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Advisory, FeedbackEvent, MetricEntry, TradeLog
from .serializers import (
    AdvisorySerializer,
    FeedbackEventSerializer,
    MetricEntrySerializer,
    TradeLogSerializer,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import get_db  # noqa: E402


class MetricsListView(generics.ListAPIView):
    serializer_class = MetricEntrySerializer

    def get_queryset(self):
        get_db()
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
        get_db()
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
        get_db()
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


class AdvisoryListView(generics.ListAPIView):
    serializer_class = AdvisorySerializer

    def get_queryset(self):
        get_db()
        qs = Advisory.objects.all()
        include_resolved = self.request.query_params.get("include_resolved")
        if not (include_resolved and include_resolved.lower() in {"1", "true", "yes"}):
            qs = qs.filter(resolved=0)
        severity = self.request.query_params.getlist("severity")
        if severity:
            qs = qs.filter(severity__in=[lvl.lower() for lvl in severity])
        limit = int(self.request.query_params.get("limit", "200"))
        limit = max(1, min(limit, 500))
        return qs.order_by("-ts")[:limit]


class DashboardSummaryView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        metric_counts = MetricEntry.objects.values("stage").annotate(total=Count("id"))
        feedback_counts = FeedbackEvent.objects.values("severity").annotate(total=Count("id"))
        advisory_counts = Advisory.objects.filter(resolved=0).values("severity").annotate(total=Count("id"))
        latest_metrics = MetricEntrySerializer(MetricEntry.objects.order_by("-ts")[:12], many=True).data
        latest_feedback = FeedbackEventSerializer(FeedbackEvent.objects.order_by("-ts")[:10], many=True).data
        recent_trades = TradeLogSerializer(TradeLog.objects.order_by("-ts")[:10], many=True).data
        active_advisories = AdvisorySerializer(
            Advisory.objects.filter(resolved=0).order_by("-ts")[:10],
            many=True,
        ).data
        state = db.load_state() or {}
        ghost_state = state.get("ghost_trading") or {}
        stable_bank = float(ghost_state.get("stable_bank", 0.0))
        total_profit = float(ghost_state.get("total_profit", 0.0))

        summary = {
            "metrics_by_stage": list(metric_counts),
            "feedback_by_severity": list(feedback_counts),
            "latest_metrics": latest_metrics,
            "latest_feedback": latest_feedback,
            "recent_trades": recent_trades,
            "advisories_by_severity": list(advisory_counts),
            "active_advisories": active_advisories,
            "stable_bank": stable_bank,
            "total_profit": total_profit,
        }
        return Response(summary, status=status.HTTP_200_OK)
