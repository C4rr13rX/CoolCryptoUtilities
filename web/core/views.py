from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.views.generic import TemplateView

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import get_db  # noqa: E402
from services.guardian_supervisor import guardian_supervisor  # noqa: E402


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
        return {
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

    def _base_context(self, initial_route: str) -> Dict[str, Any]:
        try:
            guardian_supervisor.ensure_running()
        except Exception:
            pass
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


class ConsoleView(BaseSecureView):
    initial_route = "console"


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
            "console",
            "pipeline",
            "datalab",
            "lab",
            "guardian",
            "settings",
            "advisories",
        }
        if slug not in allowed:
            return redirect("core:dashboard")
        self.initial_route = slug
        return super().dispatch(request, *args, **kwargs)
