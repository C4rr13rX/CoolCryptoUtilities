from __future__ import annotations

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.internal_cron import cron_supervisor


class CronStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        cron_supervisor.ensure_running()
        payload = cron_supervisor.status()
        return Response(payload, status=status.HTTP_200_OK)


class CronSettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        updates = {}
        if "profile" in payload and isinstance(payload["profile"], dict):
            updates = payload["profile"]
        elif "enabled" in payload:
            updates["enabled"] = bool(payload["enabled"])
        if not updates:
            return Response({"detail": "No updates provided"}, status=status.HTTP_400_BAD_REQUEST)
        profile = cron_supervisor.update_profile(updates)
        return Response({"profile": profile}, status=status.HTTP_200_OK)


class CronRunView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        task_id = payload.get("task_id")
        cron_supervisor.ensure_running()
        cron_supervisor.run_once(str(task_id) if task_id else None)
        return Response({"status": "queued"}, status=status.HTTP_202_ACCEPTED)
