from __future__ import annotations

import logging

from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DelegatedTask, DelegationHost, DelegationLog, TaskResourceProfile
from .serializers import (
    DelegatedTaskSerializer,
    DelegationHostCreateSerializer,
    DelegationHostSerializer,
    DelegationLogSerializer,
    TaskResourceProfileSerializer,
)

logger = logging.getLogger(__name__)


class DelegationHostListView(APIView):
    """List all delegation hosts or create a new one (enter pairing mode)."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        hosts = DelegationHost.objects.all()
        return Response(DelegationHostSerializer(hosts, many=True).data)

    def post(self, request: Request) -> Response:
        """Create a new host and generate its API token (pairing mode)."""
        serializer = DelegationHostCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        host = serializer.save(status=DelegationHost.Status.PAIRING)
        logger.info("delegation host created: %s (%s:%d) — pairing mode", host.name, host.host, host.port)
        # Return the full object including the token (only shown once during pairing)
        data = DelegationHostCreateSerializer(host).data
        data["api_token"] = host.api_token
        return Response(data, status=status.HTTP_201_CREATED)


class DelegationHostDetailView(APIView):
    """Get, update, or delete a specific host."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, pk: int) -> Response:
        try:
            host = DelegationHost.objects.get(pk=pk)
        except DelegationHost.DoesNotExist:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        return Response(DelegationHostSerializer(host).data)

    def patch(self, request: Request, pk: int) -> Response:
        try:
            host = DelegationHost.objects.get(pk=pk)
        except DelegationHost.DoesNotExist:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        for field in ("name", "host", "port", "enabled"):
            if field in request.data:
                setattr(host, field, request.data[field])
        host.save()
        return Response(DelegationHostSerializer(host).data)

    def delete(self, request: Request, pk: int) -> Response:
        try:
            host = DelegationHost.objects.get(pk=pk)
        except DelegationHost.DoesNotExist:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        host.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class DelegationHostRegenerateTokenView(APIView):
    """Regenerate the API token for a host (re-enter pairing mode)."""

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, pk: int) -> Response:
        try:
            host = DelegationHost.objects.get(pk=pk)
        except DelegationHost.DoesNotExist:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        from .models import _gen_token
        host.api_token = _gen_token()
        host.status = DelegationHost.Status.PAIRING
        host.save()
        return Response({"api_token": host.api_token, "status": host.status})


class DelegationHostPairView(APIView):
    """Trigger pairing with a host by contacting it over the network."""

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, pk: int) -> Response:
        try:
            host = DelegationHost.objects.get(pk=pk)
        except DelegationHost.DoesNotExist:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)

        from services.delegation_client import DelegationClient
        client = DelegationClient()
        result = client.pair_host(pk)
        return Response(result, status=status.HTTP_200_OK if result.get("paired") else status.HTTP_502_BAD_GATEWAY)


class DelegatedTaskListView(APIView):
    """List delegated tasks with optional filters."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        qs = DelegatedTask.objects.select_related("host")
        host_id = request.query_params.get("host")
        if host_id:
            qs = qs.filter(host_id=host_id)
        task_status = request.query_params.get("status")
        if task_status:
            qs = qs.filter(status=task_status)
        task_type = request.query_params.get("type")
        if task_type:
            qs = qs.filter(task_type=task_type)
        limit = min(int(request.query_params.get("limit", "50")), 200)
        return Response(DelegatedTaskSerializer(qs[:limit], many=True).data)


class DelegationLogListView(APIView):
    """List delegation communication logs."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        qs = DelegationLog.objects.select_related("host")
        host_id = request.query_params.get("host")
        if host_id:
            qs = qs.filter(host_id=host_id)
        limit = min(int(request.query_params.get("limit", "100")), 500)
        return Response(DelegationLogSerializer(qs[:limit], many=True).data)


class TaskResourceProfileListView(APIView):
    """List resource profiles for all task types (used for dispatch planning)."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        profiles = TaskResourceProfile.objects.all()
        return Response(TaskResourceProfileSerializer(profiles, many=True).data)


class DelegationSummaryView(APIView):
    """Dashboard summary for the pipeline page."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request) -> Response:
        hosts = DelegationHost.objects.all()
        online = [h for h in hosts if h.status == DelegationHost.Status.ONLINE]
        total_headroom = sum(h.headroom for h in online)
        total_active = sum(h.active_tasks for h in online)

        recent_tasks = DelegatedTask.objects.order_by("-created_at")[:20]
        completed_24h = DelegatedTask.objects.filter(
            status=DelegatedTask.Status.COMPLETED,
            completed_at__gte=timezone.now() - timezone.timedelta(hours=24),
        ).count()
        failed_24h = DelegatedTask.objects.filter(
            status=DelegatedTask.Status.FAILED,
            completed_at__gte=timezone.now() - timezone.timedelta(hours=24),
        ).count()

        return Response({
            "hosts": DelegationHostSerializer(hosts, many=True).data,
            "online_count": len(online),
            "total_headroom": total_headroom,
            "total_active_tasks": total_active,
            "completed_24h": completed_24h,
            "failed_24h": failed_24h,
            "recent_tasks": DelegatedTaskSerializer(recent_tasks, many=True).data,
            "resource_profiles": TaskResourceProfileSerializer(
                TaskResourceProfile.objects.all(), many=True
            ).data,
        })
