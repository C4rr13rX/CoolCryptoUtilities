from __future__ import annotations

from pathlib import Path

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from opsconsole.manager import manager as console_manager
from services.guardian_state import get_guardian_settings, update_guardian_settings
from services.guardian_supervisor import guardian_supervisor


class GuardianSettingsView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request: Request, *args, **kwargs) -> Response:
        guardian_supervisor.ensure_running()
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
    permission_classes = [IsAuthenticated]
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
        queued = guardian_supervisor.run_once(prompt_arg)
        payload = {"status": "queued" if queued else "skipped"}
        if not queued:
            payload["detail"] = "Guardian job already running; timer request skipped."
        code = status.HTTP_202_ACCEPTED if queued else status.HTTP_200_OK
        return Response(payload, status=code)


class GuardianLogView(APIView):
    permission_classes = [IsAuthenticated]
    LOG_CANDIDATES = [
        Path("runtime/guardian/transcripts/guardian-session.log"),
        Path("codex_transcripts/guardian-session.log"),
    ]
    PRODUCTION_LOG_CANDIDATES = [
        Path("logs/services/production.log"),
        Path("logs/production.log"),
        Path("production.log"),
    ]

    def get(self, request: Request, *args, **kwargs) -> Response:
        guardian_supervisor.ensure_running()
        limit = int(request.query_params.get("limit", "200"))
        limit = max(10, min(limit, 2000))
        guardian_lines = self._tail_first(self.LOG_CANDIDATES, limit)
        production_lines = self._tail_first(self.PRODUCTION_LOG_CANDIDATES, limit)
        if not production_lines:
            production_lines = [line.rstrip("\n") for line in console_manager.tail(limit)]
        payload = {
            "lines": guardian_lines,
            "guardian_lines": guardian_lines,
            "production_lines": production_lines,
        }
        return Response(payload, status=status.HTTP_200_OK)

    @staticmethod
    def _tail_first(paths: list[Path], limit: int) -> list[str]:
        for path in paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    entries = handle.readlines()
            except Exception:
                continue
            if entries:
                return [line.rstrip("\n") for line in entries[-limit:]]
        return []
