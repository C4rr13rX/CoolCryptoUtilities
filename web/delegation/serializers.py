from __future__ import annotations

from rest_framework import serializers

from .models import DelegatedTask, DelegationHost, DelegationLog, TaskResourceProfile


class DelegationHostSerializer(serializers.ModelSerializer):
    headroom = serializers.IntegerField(read_only=True)
    is_available = serializers.BooleanField(read_only=True)

    class Meta:
        model = DelegationHost
        fields = [
            "id", "name", "host", "port", "status", "enabled",
            "device_type", "os_name", "cpu_count", "total_memory_mb", "python_version",
            "capabilities",
            "cpu_percent", "memory_percent", "memory_available_mb", "disk_free_mb",
            "max_concurrent_tasks", "active_tasks",
            "last_heartbeat", "last_error",
            "headroom", "is_available",
            "created_at", "updated_at",
        ]
        read_only_fields = [
            "status", "device_type", "os_name", "cpu_count", "total_memory_mb",
            "python_version", "capabilities",
            "cpu_percent", "memory_percent", "memory_available_mb", "disk_free_mb",
            "max_concurrent_tasks", "active_tasks",
            "last_heartbeat", "last_error",
            "created_at", "updated_at",
        ]


class DelegationHostCreateSerializer(serializers.ModelSerializer):
    api_token = serializers.CharField(read_only=True)

    class Meta:
        model = DelegationHost
        fields = ["id", "name", "host", "port", "api_token"]


class DelegatedTaskSerializer(serializers.ModelSerializer):
    host_name = serializers.CharField(source="host.name", read_only=True)

    class Meta:
        model = DelegatedTask
        fields = [
            "id", "host", "host_name", "task_type", "status",
            "payload", "api_keys_sent",
            "result", "result_files", "error_message",
            "peak_cpu_percent", "peak_memory_mb", "duration_seconds",
            "created_at", "sent_at", "started_at", "completed_at",
        ]


class DelegationLogSerializer(serializers.ModelSerializer):
    host_name = serializers.CharField(source="host.name", read_only=True)

    class Meta:
        model = DelegationLog
        fields = [
            "id", "host", "host_name", "task", "direction",
            "message_type", "payload_summary", "payload_size_bytes", "timestamp",
        ]


class TaskResourceProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaskResourceProfile
        fields = "__all__"
