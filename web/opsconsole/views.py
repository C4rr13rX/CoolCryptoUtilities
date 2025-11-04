from __future__ import annotations

from rest_framework import status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .manager import manager


class StartProcessView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        data = manager.start()
        http_status = status.HTTP_200_OK if data.get("status") == "started" else status.HTTP_409_CONFLICT
        return Response(data, status=http_status)


class StopProcessView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        data = manager.stop()
        return Response(data, status=status.HTTP_200_OK)


class ProcessStatusView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        return Response(manager.status(), status=status.HTTP_200_OK)


class TailLogsView(APIView):
    def get(self, request: Request, *args, **kwargs) -> Response:
        limit = int(request.query_params.get("limit", "200"))
        limit = max(10, min(limit, 2000))
        return Response({"lines": manager.tail(limit)}, status=status.HTTP_200_OK)
