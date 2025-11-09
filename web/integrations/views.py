from __future__ import annotations

from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from services.api_integrations import (
    INTEGRATIONS,
    get_integration_value,
    list_integrations,
    test_integration,
    update_integration,
)


class IntegrationListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        payload = list_integrations(request.user)
        return Response({"items": payload}, status=status.HTTP_200_OK)


class IntegrationDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, name: str, *args, **kwargs) -> Response:
        if name not in INTEGRATIONS:
            return Response({"detail": "unknown integration"}, status=status.HTTP_404_NOT_FOUND)
        reveal = request.query_params.get("reveal") == "1"
        try:
            value = get_integration_value(request.user, name, reveal=reveal)
        except ValueError:
            return Response({"detail": "unknown integration"}, status=status.HTTP_404_NOT_FOUND)
        payload = {"has_value": bool(value)}
        if reveal:
            payload["value"] = value
        return Response(payload, status=status.HTTP_200_OK)

    def post(self, request: Request, name: str, *args, **kwargs) -> Response:
        if name not in INTEGRATIONS:
            return Response({"detail": "unknown integration"}, status=status.HTTP_404_NOT_FOUND)
        value = (request.data or {}).get("value")
        update_integration(request.user, name, value)
        return Response({"updated": True}, status=status.HTTP_200_OK)


class IntegrationTestView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, name: str, *args, **kwargs) -> Response:
        if name not in INTEGRATIONS:
            return Response({"detail": "unknown integration"}, status=status.HTTP_404_NOT_FOUND)
        value = (request.data or {}).get("value")
        if not value:
            value = get_integration_value(request.user, name, reveal=True)
        if not value:
            return Response({"detail": "value is required for testing"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            result = test_integration(name, str(value))
        except Exception as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_200_OK)
