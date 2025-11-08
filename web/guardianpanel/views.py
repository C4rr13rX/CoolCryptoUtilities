from __future__ import annotations

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from opsconsole.manager import manager as console_manager
from services.guardian_state import get_guardian_settings, update_guardian_settings
from services.guardian_supervisor import guardian_supervisor


class GuardianSettingsView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        settings = get_guardian_settings()
        status_payload = guardian_supervisor.status()
        payload = {
            "settings": settings,
            "status": status_payload,
            "console": console_manager.status(),
        }
        return Response(payload, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        updates = {}
        if "default_prompt" in payload:
            updates["default_prompt"] = str(payload["default_prompt"] or "")
        if "interval_minutes" in payload:
            try:
                updates["interval_minutes"] = int(payload["interval_minutes"])
            except (TypeError, ValueError):
                return Response({"detail": "interval_minutes must be numeric"}, status=status.HTTP_400_BAD_REQUEST)
        if "enabled" in payload:
            updates["enabled"] = bool(payload["enabled"])
        settings = update_guardian_settings(updates)
        if "enabled" in updates:
            guardian_supervisor.set_enabled(settings["enabled"])
        if "interval_minutes" in updates:
            guardian_supervisor.update_interval(settings["interval_minutes"])
        if "default_prompt" in updates:
            guardian_supervisor.update_prompt(settings["default_prompt"])
        return Response({"settings": settings}, status=status.HTTP_200_OK)


class GuardianRunView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        prompt = payload.get("prompt")
        save_default = bool(payload.get("save_default"))
        if save_default and prompt:
            update_guardian_settings({"default_prompt": prompt})
            guardian_supervisor.update_prompt(prompt)
            prompt_arg = None
        else:
            prompt_arg = prompt
        guardian_supervisor.run_once(prompt_arg)
        return Response({"status": "queued"}, status=status.HTTP_202_ACCEPTED)
