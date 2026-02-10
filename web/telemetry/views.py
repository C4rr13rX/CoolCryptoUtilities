from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict

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


def _load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
            qs = qs.filter(resolved=False)
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
        advisory_counts = Advisory.objects.filter(resolved=False).values("severity").annotate(total=Count("id"))
        latest_metrics = MetricEntrySerializer(MetricEntry.objects.order_by("-ts")[:12], many=True).data
        latest_feedback = FeedbackEventSerializer(FeedbackEvent.objects.order_by("-ts")[:10], many=True).data
        recent_trades = TradeLogSerializer(TradeLog.objects.order_by("-ts")[:10], many=True).data
        active_advisories = AdvisorySerializer(
            Advisory.objects.filter(resolved=False).order_by("-ts")[:10],
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


class PipelineReadinessView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        readiness = _load_report(Path("data/reports/live_readiness.json"))
        confusion_meta = _load_report(Path("data/reports/confusion_matrices.json"))
        horizon = _load_report(Path("data/reports/horizon_profile.json"))
        payload = {
            "live_readiness": readiness or {"ready": False},
            "confusion": confusion_meta.get("confusion") if confusion_meta else {},
            "decision_threshold": confusion_meta.get("decision_threshold") if confusion_meta else None,
            "horizon_profile": horizon.get("profile") if horizon else {},
            "transition_plan": confusion_meta.get("transition_plan") if confusion_meta else {},
        }
        timestamps = [
            readiness.get("updated_at") if isinstance(readiness, dict) else None,
            confusion_meta.get("updated_at") if isinstance(confusion_meta, dict) else None,
            horizon.get("updated_at") if isinstance(horizon, dict) else None,
        ]
        payload["updated_at"] = max((ts for ts in timestamps if isinstance(ts, (int, float))), default=None)
        return Response(payload, status=status.HTTP_200_OK)


class BusScheduleView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        snapshot = db.fetch_latest_organism_snapshot() or {}
        scheduler = snapshot.get("scheduler") if isinstance(snapshot, dict) else []
        transition = {}
        if isinstance(snapshot, dict):
            transition = snapshot.get("transition_plan") or {}
            if not transition:
                pipeline_payload = snapshot.get("pipeline") or {}
                if isinstance(pipeline_payload, dict):
                    transition = pipeline_payload.get("transition_plan") or {}

        risk_flags = transition.get("risk_flags") if isinstance(transition, dict) else {}
        capital_plan = transition.get("capital_plan") if isinstance(transition, dict) else {}
        bus_actions = transition.get("bus_swap_actions") if isinstance(transition, dict) else None
        if not isinstance(bus_actions, list):
            bus_actions = transition.get("bus_actions") if isinstance(transition, dict) else []

        ghost_schedule = []
        if isinstance(scheduler, list):
            for entry in scheduler:
                if not isinstance(entry, dict):
                    continue
                directive = entry.get("last_directive")
                if not isinstance(directive, dict):
                    continue
                size = _safe_float(directive.get("size"))
                price = _safe_float(directive.get("target_price") or entry.get("price"))
                usd_value = round(size * price, 6) if size and price else 0.0
                ghost_schedule.append(
                    {
                        "symbol": entry.get("symbol"),
                        "action": directive.get("action") or "enter",
                        "size": size,
                        "price": price,
                        "usd_value": usd_value,
                        "horizon": directive.get("horizon"),
                        "confidence": _safe_float(directive.get("confidence")),
                        "reason": directive.get("reason"),
                        "tier": directive.get("tier"),
                        "updated_at": entry.get("last_update"),
                    }
                )
        ghost_schedule.sort(key=lambda item: item.get("usd_value", 0.0), reverse=True)
        ghost_schedule = ghost_schedule[:16]

        live_schedule = []
        if isinstance(bus_actions, list):
            for action in bus_actions:
                if not isinstance(action, dict):
                    continue
                size = _safe_float(action.get("size"))
                price = _safe_float(action.get("price"))
                usd_value = _safe_float(action.get("target_usd") or action.get("usd_value"))
                if not usd_value and size and price:
                    usd_value = round(size * price, 6)
                live_schedule.append(
                    {
                        "symbol": action.get("symbol") or action.get("pair") or action.get("token"),
                        "action": action.get("action") or "bus_action",
                        "size": size,
                        "price": price,
                        "usd_value": usd_value,
                        "reason": action.get("reason"),
                        "priority": action.get("priority"),
                        "window_sec": action.get("window_sec"),
                    }
                )

        live_ramp = {}
        if isinstance(capital_plan, dict):
            live_ramp = capital_plan.get("live_ramp_schedule") or {}

        payload = {
            "available": bool(snapshot),
            "timestamp": snapshot.get("timestamp") if isinstance(snapshot, dict) else None,
            "ghost": {
                "halted": bool(risk_flags.get("halt_ghost")) if isinstance(risk_flags, dict) else False,
                "reason": risk_flags.get("ghost_halt_reason") if isinstance(risk_flags, dict) else None,
                "risk_multiplier": _safe_float(risk_flags.get("ghost_risk_multiplier")) if isinstance(risk_flags, dict) else 0.0,
                "schedule": ghost_schedule,
            },
            "live": {
                "halted": bool(risk_flags.get("halt_live")) if isinstance(risk_flags, dict) else False,
                "reason": risk_flags.get("halt_reason") if isinstance(risk_flags, dict) else None,
                "recommended_live_usd": _safe_float(risk_flags.get("recommended_live_usd")) if isinstance(risk_flags, dict) else 0.0,
                "min_clip_usd": _safe_float(risk_flags.get("min_clip_usd")) if isinstance(risk_flags, dict) else 0.0,
                "schedule": live_schedule,
                "ramp": {
                    "first_tranche_usd": _safe_float(live_ramp.get("first_tranche_usd")),
                    "max_live_usd": _safe_float(live_ramp.get("max_live_usd")),
                    "deployable_stable_usd": _safe_float(live_ramp.get("deployable_stable_usd")),
                    "first_tranche_cap_usd": _safe_float(live_ramp.get("first_tranche_cap_usd")),
                },
            },
            "summary": {
                "bus_actions_pending": bool(risk_flags.get("bus_actions_pending")) if isinstance(risk_flags, dict) else False,
                "bus_action_count": len(bus_actions) if isinstance(bus_actions, list) else 0,
            },
        }
        return Response(payload, status=status.HTTP_200_OK)


class OrganismLatestView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        snapshot = db.fetch_latest_organism_snapshot()
        return Response(
            {
                "snapshot": snapshot or {},
                "available": bool(snapshot),
            },
            status=status.HTTP_200_OK,
        )


class OrganismHistoryView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        start_ts = request.query_params.get("start_ts")
        end_ts = request.query_params.get("end_ts")
        limit_param = request.query_params.get("limit", "200")
        try:
            limit = max(1, min(int(limit_param), 1000))
        except ValueError:
            limit = 200
        try:
            start = float(start_ts) if start_ts is not None else None
        except ValueError:
            start = None
        try:
            end = float(end_ts) if end_ts is not None else None
        except ValueError:
            end = None
        history = db.fetch_organism_history(start_ts=start, end_ts=end, limit=limit)
        return Response(
            {
                "snapshots": history,
                "count": len(history),
            },
            status=status.HTTP_200_OK,
        )


class OrganismSettingsView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        return Response({"label_scale": db.get_label_scale()}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        db = get_db()
        try:
            scale = float(request.data.get("label_scale"))
        except (TypeError, ValueError):
            return Response({"detail": "label_scale must be numeric"}, status=status.HTTP_400_BAD_REQUEST)
        db.set_label_scale(scale)
        return Response({"label_scale": db.get_label_scale()}, status=status.HTTP_200_OK)
