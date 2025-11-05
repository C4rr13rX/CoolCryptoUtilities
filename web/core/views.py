from __future__ import annotations

import os

import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from django.conf import settings
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.views.generic import TemplateView

from db import get_db


class IndexView(TemplateView):
    template_name = "core/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update(self._dashboard_context())
        context.update(
            {
                "debug": settings.DEBUG,
                "vite_dev_server": os.getenv("VITE_DEV_SERVER", "http://localhost:5173"),
                "auth_form": AuthenticationForm(self.request),
            }
        )
        return context

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect("core:index")
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("core:index")
        context = self.get_context_data()
        context["auth_form"] = form
        return self.render_to_response(context)

    def _dashboard_context(self) -> Dict[str, object]:
        if not self.request.user.is_authenticated:
            return {
                "latest_metrics": [],
                "latest_feedback": [],
                "ghost_trades": [],
                "live_trades": [],
                "stable_bank": 0.0,
                "total_profit": 0.0,
                "advisories": [],
            }

        db = get_db()
        metrics = db.fetch_metrics(limit=6)
        feedback = db.fetch_feedback_events(limit=6)
        ghost_trades = db.fetch_trades(wallets=["ghost"], limit=6)
        live_trades = db.fetch_trades(wallets=["live"], limit=6)
        advisories = db.fetch_advisories(limit=8, include_resolved=False)
        state = db.load_state() or {}
        ghost_state = state.get("ghost_trading") or {}
        stable_bank = float(ghost_state.get("stable_bank", 0.0))
        total_profit = float(ghost_state.get("total_profit", 0.0))
        return {
            "latest_metrics": metrics,
            "latest_feedback": feedback,
            "ghost_trades": ghost_trades,
            "live_trades": live_trades,
            "stable_bank": stable_bank,
            "total_profit": total_profit,
            "advisories": advisories,
        }
