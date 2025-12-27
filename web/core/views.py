from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import math
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.views import View
from django.views.generic import TemplateView

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import get_db  # noqa: E402
from opsconsole.manager import manager as console_manager
from services.guardian_supervisor import guardian_supervisor  # noqa: E402
from services.code_graph import get_code_graph, list_tracked_files, request_code_graph_refresh  # noqa: E402

GUARDIAN_TRANSCRIPT = Path("runtime/guardian/transcripts/guardian-session.log")
LEGACY_TRANSCRIPT = Path("codex_transcripts/guardian-session.log")
SNAPSHOT_ROOT = Path("runtime/code_graph/snapshots")

def _load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _empty_payload() -> Dict[str, Any]:
    return {
        "latest_metrics": [],
        "latest_feedback": [],
        "ghost_trades": [],
        "live_trades": [],
        "stable_bank": 0.0,
        "total_profit": 0.0,
        "advisories": [],
    }


class DashboardContextMixin:
    """
    Provide a lightweight fallback snapshot so the template can display meaningful
    information even before the Vue runtime hydrates.
    """

    def _scrub_json(self, value: Any) -> Any:
        """
        Replace NaN/Inf and non-serialisable values with None so json_script
        emits valid JSON (avoids frontend parse errors).
        """
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
                return None
            return value
        if isinstance(value, (float, int, Decimal)):
            try:
                num = float(value)
            except Exception:
                return None
            if math.isnan(num) or math.isinf(num):
                return None
            return num
        # Catch numpy scalars or any numeric-like objects
        try:
            num = float(value)  # type: ignore[arg-type]
            if math.isnan(num) or math.isinf(num):
                return None
            return num
        except Exception:
            return value
        if isinstance(value, dict):
            return {k: self._scrub_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._scrub_json(item) for item in value]
        return value

    def _dashboard_snapshot(self) -> Dict[str, Any]:
        request = getattr(self, "request", None)
        if not request or not request.user.is_authenticated:
            return _empty_payload()

        db = get_db()
        state = db.load_state() or {}
        ghost_state = state.get("ghost_trading") or {}
        readiness = _load_report(Path("data/reports/live_readiness.json"))
        confusion = _load_report(Path("data/reports/confusion_eval.json"))
        horizon = _load_report(Path("data/reports/horizon_profile.json"))
        timeline = _load_report(Path("runtime/organism_timeline.json"))
        timeline_frames = []
        raw_frames = timeline.get("snapshots") if isinstance(timeline, dict) else []
        if isinstance(raw_frames, list):
            timeline_frames = raw_frames[-8:]
        snapshot = {
            "latest_metrics": db.fetch_metrics(limit=6),
            "latest_feedback": db.fetch_feedback_events(limit=8),
            "ghost_trades": db.fetch_trades(wallets=["ghost"], limit=10),
            "live_trades": db.fetch_trades(wallets=["live"], limit=10),
            "stable_bank": float(ghost_state.get("stable_bank", 0.0)),
            "total_profit": float(ghost_state.get("total_profit", 0.0)),
            "advisories": db.fetch_advisories(limit=8, include_resolved=False),
            "live_readiness": readiness,
            "confusion_eval": confusion,
            "horizon_profile": horizon.get("summary") if isinstance(horizon, dict) else {},
            "organism_timeline": timeline_frames,
        }
        snapshot = self._scrub_json(snapshot)
        try:
            snapshot = json.loads(json.dumps(snapshot, allow_nan=False))
        except Exception:
            snapshot = _empty_payload()
        return snapshot

    def _base_context(self, initial_route: str) -> Dict[str, Any]:
        use_vite = settings.DEBUG and os.getenv("DJANGO_USE_VITE_DEV", "0").lower() in {"1", "true", "yes", "on"}
        return {
            "debug": settings.DEBUG,
            "vite_dev_server": os.getenv("VITE_DEV_SERVER", "http://localhost:5173"),
            "serve_vite": use_vite,
            "initial_route": initial_route,
            "fallback_snapshot": self._dashboard_snapshot(),
        }


class LandingView(DashboardContextMixin, TemplateView):
    """
    Renders the login page and redirects authenticated users directly to the main
    dashboard shell.
    """

    template_name = "core/index.html"

    def dispatch(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if request.user.is_authenticated:
            context = self._base_context(initial_route="dashboard")
            return self.render_to_response(context)
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context.update(self._base_context(initial_route="dashboard"))
        context["auth_form"] = AuthenticationForm(self.request)
        return context

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect("core:dashboard")
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            context = self._base_context(initial_route="dashboard")
            return self.render_to_response(context)
        context = self.get_context_data()
        context["auth_form"] = form
        return self.render_to_response(context)


class BaseSecureView(LoginRequiredMixin, DashboardContextMixin, TemplateView):
    """
    Shared base for each top-level page. Each concrete view only needs to set the
    `initial_route` attribute so the front-end Vue router can hydrate to the
    correct section.
    """

    login_url = "core:index"
    template_name = "core/index.html"
    initial_route = "dashboard"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context.update(self._base_context(initial_route=self.initial_route))
        return context


class DashboardView(BaseSecureView):
    initial_route = "dashboard"


class StreamsView(BaseSecureView):
    initial_route = "streams"


class TelemetryView(BaseSecureView):
    initial_route = "telemetry"


class WalletPageView(BaseSecureView):
    initial_route = "wallet"


class OrganismView(BaseSecureView):
    initial_route = "organism"


class PipelinePageView(BaseSecureView):
    initial_route = "pipeline"


class DataLabPageView(BaseSecureView):
    initial_route = "datalab"


class ModelLabPageView(BaseSecureView):
    initial_route = "lab"


class GuardianPageView(BaseSecureView):
    initial_route = "guardian"


class SecureSettingsPageView(BaseSecureView):
    initial_route = "settings"


class AdvisoriesPageView(BaseSecureView):
    initial_route = "advisories"


def guardian_failure_response(request, *args, **kwargs):
    view = GuardianFallbackView.as_view()
    response = view(request, *args, **kwargs)
    response.status_code = 503
    return response


class IntegrationsPageView(BaseSecureView):
    initial_route = "integrations"


class CodeGraphPageView(BaseSecureView):
    initial_route = "codegraph"


class BrandDozerPageView(BaseSecureView):
    initial_route = "branddozer"


class BrandDozerSoloView(BaseSecureView):
    initial_route = "branddozer_solo"


class SpaRouteView(BaseSecureView):
    """
    Catch-all view so refreshing /<route> stays inside the SPA shell.
    Only accepts known slug characters and still requires authentication.
    """

    def dispatch(self, request, *args, **kwargs):
        slug = kwargs.get("route") or "dashboard"
        allowed = {
            "dashboard",
            "organism",
            "streams",
            "telemetry",
            "wallet",
            "pipeline",
            "datalab",
            "lab",
            "guardian",
            "settings",
            "advisories",
            "integrations",
            "codegraph",
            "branddozer",
        }
        if slug not in allowed:
            return redirect("core:dashboard")
        self.initial_route = slug
        return super().dispatch(request, *args, **kwargs)


class GuardianFallbackView(TemplateView):
    """
    Minimal HTML fallback that surfaces guardian console output even when the
    main SPA is unavailable (e.g., during server restarts). Accessible without
    authentication so operators can confirm guardian activity.
    """

    template_name = "core/guardian_fallback.html"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        status = {}
        try:
            status = guardian_supervisor.status()
        except Exception:
            status = {"running": False}
        transcription_path = GUARDIAN_TRANSCRIPT if GUARDIAN_TRANSCRIPT.exists() else LEGACY_TRANSCRIPT
        console_tail = _tail_lines(transcription_path, limit=400)
        production_tail = console_manager.tail(400)
        last_report = status.get("last_report")
        if isinstance(last_report, (int, float)):
            last_report = datetime.fromtimestamp(last_report)
        else:
            last_report = None
        context.update(
            {
                "guardian_status": status,
                "console_tail": console_tail,
                "production_tail": production_tail,
                "last_report": last_report,
                "queue_state": status.get("queue") or {},
                "production": status.get("production") or {},
            }
        )
        return context


class CodeGraphDataView(LoginRequiredMixin, View):
    login_url = "core:index"

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        refresh = str(request.GET.get("refresh", "")).lower() in {"1", "true", "yes"}
        if refresh:
            request_code_graph_refresh()
            payload = get_code_graph(force_refresh=False)
            payload["building"] = True
        else:
            payload = get_code_graph(force_refresh=False)
        return JsonResponse(payload, status=200)


def _safe_snapshot_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "node"


class CodeGraphSnapshotView(LoginRequiredMixin, View):
    login_url = "core:index"
    http_method_names = ["post"]

    def post(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"detail": "Invalid JSON payload"}, status=400)
        image = payload.get("image")
        node_id = payload.get("node_id")
        timestamp = str(payload.get("timestamp") or "").strip() or time.strftime("%Y%m%d-%H%M%S")
        if not image or not node_id:
            return JsonResponse({"detail": "image and node_id are required"}, status=400)
        if "," in image:
            image = image.split(",", 1)[1]
        try:
            binary = base64.b64decode(image)
        except Exception:
            return JsonResponse({"detail": "Unable to decode image payload"}, status=400)
        safe_ts = _safe_snapshot_name(timestamp)
        safe_node = _safe_snapshot_name(node_id)
        folder = SNAPSHOT_ROOT / safe_ts
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{safe_node}.png"
        file_path.write_bytes(binary)
        return JsonResponse({"saved": str(file_path)}, status=201)


class CodeGraphFilesView(LoginRequiredMixin, View):
    login_url = "core:index"

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        files = list_tracked_files()
        return JsonResponse({"files": files}, status=200)


def _tail_lines(path: Path, limit: int = 400) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception:
        return []
    return [line.rstrip("\n") for line in lines[-limit:]]
