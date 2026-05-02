"""wizard_chat/agent_views.py — agent-mode endpoints.

POST /api/wizard-chat/agent/        Run one agent turn with shell tools.
GET  /api/wizard-chat/agent/info/   Report current host's elevation method.

Both endpoints refuse requests that did not come from 127.0.0.1 / ::1.
The Django dev server already binds localhost-only by default; the
explicit check here is defence-in-depth in case the user reverse-proxies
the panel onto a wider interface someday.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt


_LOCALHOST_ADDRESSES = {"127.0.0.1", "::1", "localhost"}


def _is_localhost(request) -> bool:
    """Reject anything that didn't come from the local host.  REMOTE_ADDR
    is what the WSGI server saw; if a reverse proxy is in front, that
    proxy must terminate on the same machine for this to pass."""
    addr = (request.META.get("REMOTE_ADDR") or "").strip()
    return addr in _LOCALHOST_ADDRESSES


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatAgentView(View):
    """POST /api/wizard-chat/agent/

    Body: {
      text: str,
      session_id: str,
      allow_admin: bool   — when true, the shell_admin tool is registered
                            so the agent can request OS-level elevation.
                            Each elevated call still requires the user to
                            authenticate at the OS auth dialog.
      reset: bool         — start a fresh agent flow for this session.
      backend: str        — "wizard" (default) or "bedrock".
    }
    Returns: {answer, session_id, admin_enabled, elevation_method}
    """

    def post(self, request):
        if not _is_localhost(request):
            return JsonResponse(
                {"error": "agent mode is restricted to localhost connections"},
                status=403,
            )
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        text = (body.get("text") or "").strip()
        session_id = body.get("session_id") or str(uuid.uuid4())
        allow_admin = bool(body.get("allow_admin", False))
        reset = bool(body.get("reset", False))
        backend = body.get("backend") or "wizard"

        if not text:
            return JsonResponse({"error": "text required"}, status=400)

        from . import agent_runner

        try:
            answer = agent_runner.run(
                text,
                session_key=session_id,
                backend=backend,
                reset=reset,
                allow_admin=allow_admin,
            )
        except Exception as exc:
            return JsonResponse({
                "error": f"agent runner error: {exc}",
                "session_id": session_id,
            }, status=500)

        return JsonResponse({
            "answer": answer,
            "session_id": session_id,
            "admin_enabled": allow_admin,
            "elevation_method": agent_runner.elevation_method(),
        })


class WizardChatAgentInfoView(View):
    """GET /api/wizard-chat/agent/info/

    Returns metadata about the agent host's elevation capabilities so the
    UI can show the user which dialog to expect (UAC, polkit, osascript).
    """

    def get(self, request):
        if not _is_localhost(request):
            return JsonResponse(
                {"error": "agent info is restricted to localhost connections"},
                status=403,
            )
        from . import agent_runner
        return JsonResponse({
            "elevation_method": agent_runner.elevation_method(),
            "available": agent_runner.elevation_method() != "unavailable",
        })
