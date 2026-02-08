from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import math
import platform
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q, Count
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.utils import timezone
from django.views import View
from django.views.generic import TemplateView

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from db import get_db  # noqa: E402
from opsconsole.manager import manager as console_manager
from services.guardian_supervisor import guardian_supervisor  # noqa: E402
from services.code_graph import get_code_graph, list_tracked_files, request_code_graph_refresh  # noqa: E402
from services.graph_store import search_graph_equations, graph_enabled  # noqa: E402
from tools.c0d3r_session import C0d3rSession, c0d3r_default_settings  # noqa: E402
from .models import C0d3rWebSession, C0d3rWebMessage

GUARDIAN_TRANSCRIPT = Path("runtime/guardian/transcripts/guardian-session.log")
LEGACY_TRANSCRIPT = Path("codex_transcripts/guardian-session.log")
SNAPSHOT_ROOT = Path("runtime/code_graph/snapshots")
C0D3R_TRANSCRIPTS = Path("runtime/c0d3r/web")
_C0D3R_SESSIONS: Dict[str, C0d3rSession] = {}

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
        asset_version = self._asset_version()
        return {
            "debug": settings.DEBUG,
            "vite_dev_server": os.getenv("VITE_DEV_SERVER", "http://localhost:5173"),
            "serve_vite": use_vite,
            "initial_route": initial_route,
            "asset_version": asset_version,
            "fallback_snapshot": self._dashboard_snapshot(),
        }

    def _asset_version(self) -> str:
        """
        Return a cache-busting version for frontend assets based on the newest build.
        This prevents stale JS/CSS from sticking around after deployments.
        """
        candidates = [
            ROOT / "web" / "collected_static" / "assets" / "main.js",
            ROOT / "web" / "frontend" / "dist" / "assets" / "main.js",
            ROOT / "web" / "static" / "assets" / "main.js",
        ]
        for candidate in candidates:
            try:
                if candidate.exists():
                    return str(int(candidate.stat().st_mtime))
            except Exception:
                continue
        return str(int(time.time()))


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
        context["signup_form"] = UserCreationForm()
        context["auth_mode"] = "login"
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


class SignupView(DashboardContextMixin, TemplateView):
    template_name = "core/index.html"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context.update(self._base_context(initial_route="dashboard"))
        context["auth_form"] = AuthenticationForm(self.request)
        context["signup_form"] = UserCreationForm()
        context["auth_mode"] = "signup"
        return context

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        if request.user.is_authenticated:
            return redirect("core:dashboard")
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            context = self._base_context(initial_route="dashboard")
            return self.render_to_response(context)
        context = self.get_context_data()
        context["signup_form"] = form
        context["auth_mode"] = "signup"
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


class AddressBookPageView(BaseSecureView):
    initial_route = "addressbook"


class C0d3rPageView(BaseSecureView):
    initial_route = "c0d3r"


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


class AudioLabPageView(BaseSecureView):
    initial_route = "audiolab"


class CronPageView(BaseSecureView):
    initial_route = "cron"


class U53RxRobotPageView(BaseSecureView):
    initial_route = "u53rxr080t"


class InvestigationsPageView(BaseSecureView):
    initial_route = "investigations"


class SpaRouteView(BaseSecureView):
    """
    Catch-all view so refreshing /<route> stays inside the SPA shell.
    Only accepts known slug characters and still requires authentication.
    """

    def dispatch(self, request, *args, **kwargs):
        raw_slug = kwargs.get("route") or "dashboard"
        slug = str(raw_slug).lower()
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
            "addressbook",
            "c0d3r",
            "u53rxr080t",
            "branddozer_solo",
            "audiolab",
            "cron",
            "investigations",
        }
        if slug not in allowed:
            return redirect("core:dashboard")
        self.initial_route = slug
        if slug != raw_slug:
            return redirect(f"/{slug}")
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


def _safe_int(value: Any, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        num = int(value)
    except Exception:
        num = default
    if min_value is not None:
        num = max(min_value, num)
    if max_value is not None:
        num = min(max_value, num)
    return num


def _serialize_c0d3r_session(session: C0d3rWebSession, message_count: int | None = None) -> Dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title or "",
        "summary": session.summary or "",
        "key_points": session.key_points or [],
        "model_id": session.model_id or "",
        "last_active": session.last_active.isoformat() if session.last_active else None,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "message_count": message_count,
    }


def _serialize_c0d3r_message(message: C0d3rWebMessage) -> Dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content or "",
        "model_id": message.model_id or "",
        "created_at": message.created_at.isoformat() if message.created_at else None,
        "metadata": message.metadata or {},
    }


def _collect_system_context(request: HttpRequest) -> str:
    parts = [
        f"Server time: {timezone.now().isoformat()}",
        f"Server OS: {platform.system()} {platform.release()}",
        f"Python: {platform.python_version()}",
        f"Workspace: {ROOT}",
    ]
    if request.user and request.user.is_authenticated:
        parts.append(f"User: {request.user.get_username()}")
    return "\n".join(parts)


def _long_term_hits(user, prompt: str, *, limit: int = 4) -> List[str]:
    if not prompt:
        return []
    lower = prompt.lower()
    triggers = ("remember", "recall", "last time", "previous", "earlier", "do you remember", "what did we")
    if not any(t in lower for t in triggers):
        return []
    tokens = [tok for tok in re.findall(r"[a-z0-9]{4,}", lower) if tok not in {"remember", "recall", "previous", "earlier"}]
    qs = C0d3rWebMessage.objects.filter(session__user=user)
    if tokens:
        token_q = Q()
        for tok in tokens[:4]:
            token_q |= Q(content__icontains=tok)
        qs = qs.filter(token_q)
    hits = qs.order_by("-created_at")[:limit]
    snippets: List[str] = []
    for hit in hits:
        ts = hit.created_at.isoformat() if hit.created_at else ""
        snippets.append(f"{ts} {hit.role.capitalize()}: {hit.content}")
    return snippets


def _build_c0d3r_context(
    session: C0d3rWebSession,
    *,
    request: HttpRequest,
    prompt: str,
    max_chars: int = 12000,
) -> str:
    parts: List[str] = []
    summary = (session.summary or "").strip()
    if summary:
        parts.append("Rolling summary:\n" + summary)
    key_points = session.key_points or []
    if key_points:
        points = [str(p).strip() for p in key_points if str(p).strip()]
        if points:
            parts.append("Key points:\n" + "\n".join(f"- {p}" for p in points[:10]))
    system_info = _collect_system_context(request)
    if system_info:
        parts.append("System info:\n" + system_info)

    max_messages = _safe_int(os.getenv("C0D3R_WEB_CONTEXT_MESSAGES", "40"), 40, min_value=10, max_value=200)
    history = list(
        C0d3rWebMessage.objects.filter(session=session).order_by("-created_at")[:max_messages]
    )
    history.reverse()
    transcript_blocks: List[str] = []
    for entry in history:
        role = (entry.role or "").strip().lower()
        speaker = "Assistant" if role in {"assistant", "c0d3r"} else ("User" if role == "user" else role.capitalize())
        content = (entry.content or "").strip()
        if not content:
            continue
        transcript_blocks.append(f"{speaker}: {content}")

    if transcript_blocks:
        parts.append("Transcript (most recent last):\n" + "\n".join(transcript_blocks))

    long_hits = _long_term_hits(request.user, prompt)
    if long_hits:
        parts.append("Long-term recall:\n" + "\n".join(long_hits))

    combined = "\n\n".join([p for p in parts if p.strip()])
    if len(combined) <= max_chars:
        return combined
    return combined[:max_chars].rstrip() + "..."


def _update_c0d3r_summary(
    session_obj: C0d3rWebSession,
    *,
    user_text: str,
    assistant_text: str,
    c0d3r_session: C0d3rSession,
) -> None:
    if os.getenv("C0D3R_WEB_SUMMARY_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return
    summary_payload = {
        "summary": session_obj.summary or "",
        "key_points": session_obj.key_points or [],
    }
    system = (
        "Return ONLY JSON with keys: summary (string, <=200 words), "
        "key_points (list of 10 short strings). "
        "Focus on the most important and most recent conversation facts."
    )
    prompt = (
        f"Current summary (<=200 words):\n{summary_payload['summary']}\n\n"
        f"Current key points:\n{summary_payload['key_points']}\n\n"
        f"New exchange:\nUser: {user_text}\nAssistant: {assistant_text}\n"
    )
    try:
        model_id = c0d3r_session._c0d3r._model_for_stage("executor")
        built = c0d3r_session._c0d3r._build_prompt("executor", prompt, system=system)
        response = c0d3r_session._c0d3r._invoke_model(model_id, built)
        try:
            payload = json.loads(response)
        except Exception:
            payload = {}
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end > start:
                try:
                    payload = json.loads(response[start : end + 1])
                except Exception:
                    payload = {}
    except Exception:
        return
    new_summary = str(payload.get("summary") or summary_payload["summary"]).strip()
    words = new_summary.split()
    if len(words) > 200:
        new_summary = " ".join(words[:200])
    new_points = payload.get("key_points") or summary_payload["key_points"]
    if not isinstance(new_points, list):
        new_points = summary_payload["key_points"]
    new_points = [str(p).strip() for p in new_points if str(p).strip()]
    if len(new_points) > 10:
        new_points = new_points[:10]
    session_obj.summary = new_summary
    session_obj.key_points = new_points


class C0d3rSessionListView(LoginRequiredMixin, View):
    login_url = "core:index"

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        sessions = (
            C0d3rWebSession.objects.filter(user=request.user)
            .annotate(message_count=Count("messages"))
            .order_by("-last_active", "-updated_at")
        )
        payload = [_serialize_c0d3r_session(s, message_count=s.message_count) for s in sessions]
        return JsonResponse({"items": payload, "count": len(payload)}, status=200)

    def post(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        except Exception:
            payload = {}
        title = str(payload.get("title") or "").strip()
        if not title:
            title = f"Session {timezone.now().strftime('%Y-%m-%d %H:%M')}"
        session = C0d3rWebSession.objects.create(
            user=request.user,
            title=title,
            summary="",
            key_points=[],
            last_active=timezone.now(),
        )
        return JsonResponse({"item": _serialize_c0d3r_session(session, message_count=0)}, status=201)


class C0d3rSessionDetailView(LoginRequiredMixin, View):
    login_url = "core:index"

    def post(self, request: HttpRequest, session_id: int, *args, **kwargs) -> JsonResponse:
        try:
            payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        except Exception:
            payload = {}
        try:
            session = C0d3rWebSession.objects.get(id=session_id, user=request.user)
        except C0d3rWebSession.DoesNotExist:
            return JsonResponse({"detail": "session not found"}, status=404)
        title = payload.get("title")
        if isinstance(title, str):
            session.title = title.strip()
        session.metadata = payload.get("metadata") or session.metadata
        session.save(update_fields=["title", "metadata", "updated_at"])
        return JsonResponse({"item": _serialize_c0d3r_session(session)}, status=200)

    def delete(self, request: HttpRequest, session_id: int, *args, **kwargs) -> JsonResponse:
        try:
            session = C0d3rWebSession.objects.get(id=session_id, user=request.user)
        except C0d3rWebSession.DoesNotExist:
            return JsonResponse({"detail": "session not found"}, status=404)
        session.delete()
        return JsonResponse({"deleted": True}, status=200)


class C0d3rMessageListView(LoginRequiredMixin, View):
    login_url = "core:index"

    def get(self, request: HttpRequest, session_id: int, *args, **kwargs) -> JsonResponse:
        try:
            session = C0d3rWebSession.objects.get(id=session_id, user=request.user)
        except C0d3rWebSession.DoesNotExist:
            return JsonResponse({"detail": "session not found"}, status=404)
        limit = _safe_int(request.GET.get("limit"), 200, min_value=1, max_value=500)
        before = request.GET.get("before")
        query = (request.GET.get("q") or "").strip()
        qs = C0d3rWebMessage.objects.filter(session=session)
        if before:
            try:
                before_dt = datetime.fromisoformat(str(before))
                qs = qs.filter(created_at__lt=before_dt)
            except Exception:
                pass
        if query:
            qs = qs.filter(content__icontains=query)
        messages = list(qs.order_by("-created_at")[:limit])
        messages.reverse()
        payload = [_serialize_c0d3r_message(m) for m in messages]
        return JsonResponse({"items": payload, "count": len(payload)}, status=200)


class EquationSearchView(LoginRequiredMixin, View):
    login_url = "core:index"

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        query = str(request.GET.get("q") or "").strip()
        if not query:
            return JsonResponse({"detail": "q is required", "items": []}, status=400)
        limit = _safe_int(request.GET.get("limit"), 20, min_value=1, max_value=100)
        items: List[Dict[str, Any]] = []
        if graph_enabled():
            try:
                graph_hits = search_graph_equations(query, limit=limit)
                eq_ids = [hit.get("eq_id") for hit in graph_hits if hit.get("eq_id")]
                eq_map = {}
                if eq_ids:
                    from .models import Equation
                    for eq in Equation.objects.filter(id__in=eq_ids):
                        eq_map[str(eq.id)] = eq
                for hit in graph_hits:
                    eq_id = hit.get("eq_id")
                    eq = eq_map.get(str(eq_id)) if eq_id else None
                    items.append(
                        {
                            "id": eq_id or "",
                            "text": hit.get("equation") or (eq.text if eq else ""),
                            "latex": eq.latex if eq else "",
                            "disciplines": eq.disciplines or eq.domains if eq else hit.get("domain") or "",
                            "citations": eq.citations if eq else [],
                            "tool_used": eq.tool_used if eq else "",
                            "captured_at": eq.captured_at.isoformat() if eq and eq.captured_at else None,
                            "origin": "kuzu",
                        }
                    )
            except Exception:
                items = []
        if not items:
            from .models import Equation
            qs = (
                Equation.objects.filter(Q(text__icontains=query) | Q(latex__icontains=query))
                .order_by("-created_at")[:limit]
            )
            for eq in qs:
                items.append(
                    {
                        "id": eq.id,
                        "text": eq.text,
                        "latex": eq.latex,
                        "disciplines": eq.disciplines or eq.domains or [],
                        "citations": eq.citations or [],
                        "tool_used": eq.tool_used or "",
                        "captured_at": eq.captured_at.isoformat() if eq.captured_at else None,
                        "origin": "django",
                    }
                )
        return JsonResponse({"items": items, "count": len(items)}, status=200)


class C0d3rRunView(LoginRequiredMixin, View):
    login_url = "core:index"
    http_method_names = ["post"]

    def post(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            payload = json.loads(request.body.decode("utf-8")) if request.body else {}
        except Exception:
            payload = {}
        prompt = str(payload.get("prompt") or "").strip()
        reset = bool(payload.get("reset"))
        research = bool(payload.get("research"))
        session_id = payload.get("session_id")
        if not session_id and not reset and not prompt:
            return JsonResponse({"detail": "prompt is required"}, status=400)
        if session_id:
            try:
                session_obj = C0d3rWebSession.objects.get(id=session_id, user=request.user)
            except C0d3rWebSession.DoesNotExist:
                return JsonResponse({"detail": "session not found"}, status=404)
        else:
            session_obj = C0d3rWebSession.objects.create(
                user=request.user,
                title=f"Session {timezone.now().strftime('%Y-%m-%d %H:%M')}",
                summary="",
                key_points=[],
                last_active=timezone.now(),
            )
        session_name = f"c0d3r-web-{request.user.id}-{session_obj.id}"
        if isinstance(session_obj.metadata, dict):
            cli_name = session_obj.metadata.get("cli_session_name")
            if isinstance(cli_name, str) and cli_name.strip():
                session_name = cli_name.strip()
        session_key = f"user:{request.user.id}:session:{session_name}"
        session = _C0D3R_SESSIONS.get(session_key)
        if reset:
            C0d3rWebMessage.objects.filter(session=session_obj).delete()
            session_obj.summary = ""
            session_obj.key_points = []
            session_obj.last_active = timezone.now()
            session_obj.save(update_fields=["summary", "key_points", "last_active", "updated_at"])
            if session_key in _C0D3R_SESSIONS:
                _C0D3R_SESSIONS.pop(session_key, None)
            if not prompt:
                return JsonResponse(
                    {"output": "Session reset.", "model": "", "session_id": session_obj.id},
                    status=200,
                )
        if session is None:
            settings = c0d3r_default_settings()
            settings["research_report_enabled"] = False
            for key in ("stream_default", "transcript_enabled", "event_store_enabled", "diagnostics_enabled"):
                settings.pop(key, None)
            transcript_dir = C0D3R_TRANSCRIPTS
            if isinstance(session_obj.metadata, dict) and session_obj.metadata.get("cli_session_name"):
                transcript_dir = Path("runtime/c0d3r/transcripts")
            session = C0d3rSession(
                session_name=session_name,
                transcript_dir=transcript_dir,
                stream_default=False,
                workdir=ROOT,
                transcript_enabled=False,
                event_store_enabled=False,
                diagnostics_enabled=False,
                db_sync_enabled=False,
                **settings,
            )
            _C0D3R_SESSIONS[session_key] = session
        context_chars = _safe_int(os.getenv("C0D3R_WEB_CONTEXT_CHARS", "12000"), 12000, min_value=2000, max_value=20000)
        system_context = _build_c0d3r_context(session_obj, request=request, prompt=prompt, max_chars=context_chars)
        try:
            output = session.send(prompt, stream=False, verbose=False, research_override=research, system=system_context)
        except Exception as exc:
            return JsonResponse({"detail": f"c0d3r failed: {exc}"}, status=500)
        model = ""
        try:
            model = session._c0d3r.ensure_model()
        except Exception:
            model = ""
        C0d3rWebMessage.objects.create(
            session=session_obj,
            role="user",
            content=prompt,
            metadata={"research": research},
        )
        C0d3rWebMessage.objects.create(
            session=session_obj,
            role="c0d3r",
            content=output,
            model_id=model,
        )
        session_obj.model_id = model
        session_obj.last_active = timezone.now()
        _update_c0d3r_summary(session_obj, user_text=prompt, assistant_text=output, c0d3r_session=session)
        session_obj.save(update_fields=["model_id", "last_active", "summary", "key_points", "updated_at"])
        return JsonResponse(
            {"output": output, "model": model, "session_id": session_obj.id},
            status=200,
        )


def _tail_lines(path: Path, limit: int = 400) -> List[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except Exception:
        return []
    return [line.rstrip("\n") for line in lines[-limit:]]
