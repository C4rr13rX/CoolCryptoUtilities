from __future__ import annotations

import secrets

from django.db import models
from django.utils import timezone


def _gen_token() -> str:
    return secrets.token_urlsafe(48)


class DelegationHost(models.Model):
    """A remote machine registered to accept delegated work."""

    class Status(models.TextChoices):
        PAIRING = "pairing", "Pairing"
        ONLINE = "online", "Online"
        OFFLINE = "offline", "Offline"
        ERROR = "error", "Error"

    name = models.CharField(max_length=128, help_text="Friendly name for the host")
    host = models.CharField(max_length=255, help_text="hostname or IP")
    port = models.IntegerField(default=7782)
    api_token = models.CharField(max_length=128, default=_gen_token, unique=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PAIRING)
    enabled = models.BooleanField(default=True)

    # Device profile reported by the host
    device_type = models.CharField(max_length=32, blank=True, default="")  # desktop, mobile, server
    os_name = models.CharField(max_length=64, blank=True, default="")
    cpu_count = models.IntegerField(default=0)
    total_memory_mb = models.IntegerField(default=0)
    python_version = models.CharField(max_length=32, blank=True, default="")
    capabilities = models.JSONField(default=list, blank=True)  # list of supported task types

    # Live resource snapshot (updated by heartbeat)
    cpu_percent = models.FloatField(default=0.0)
    memory_percent = models.FloatField(default=0.0)
    memory_available_mb = models.IntegerField(default=0)
    disk_free_mb = models.IntegerField(default=0)
    max_concurrent_tasks = models.IntegerField(default=1)
    active_tasks = models.IntegerField(default=0)

    last_heartbeat = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-last_heartbeat"]
        unique_together = [("host", "port")]

    def __str__(self) -> str:
        return f"{self.name} ({self.host}:{self.port}) [{self.status}]"

    @property
    def is_available(self) -> bool:
        if not self.enabled or self.status != self.Status.ONLINE:
            return False
        if self.active_tasks >= self.max_concurrent_tasks:
            return False
        return True

    @property
    def headroom(self) -> int:
        """How many more tasks this host can accept."""
        if not self.is_available:
            return 0
        return max(0, self.max_concurrent_tasks - self.active_tasks)


class DelegatedTask(models.Model):
    """A task sent to a delegation host."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        SENT = "sent", "Sent"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"
        TIMEOUT = "timeout", "Timeout"

    class TaskType(models.TextChoices):
        DATA_INGEST = "data_ingest", "OHLCV Data Download"
        NEWS_ENRICHMENT = "news_enrichment", "News Enrichment"
        DATASET_WARMUP = "dataset_warmup", "Dataset Warmup"
        CANDIDATE_TRAINING = "candidate_training", "Model Training"
        GHOST_METRICS = "ghost_metrics", "Ghost Metrics"
        GHOST_TRADING = "ghost_trading", "Ghost Trading"
        LIVE_MONITORING = "live_monitoring", "Live Price Monitoring"
        BACKGROUND_REFRESH = "background_refresh", "Background Refresh"

    host = models.ForeignKey(DelegationHost, on_delete=models.CASCADE, related_name="tasks")
    task_type = models.CharField(max_length=32, choices=TaskType.choices)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED)

    # What we sent
    payload = models.JSONField(default=dict, help_text="Task parameters sent to the host")
    api_keys_sent = models.JSONField(default=list, help_text="Which API key names were forwarded (not values)")

    # What came back
    result = models.JSONField(default=dict, blank=True, help_text="Result data from the host")
    result_files = models.JSONField(default=list, blank=True, help_text="File paths of returned data")
    error_message = models.TextField(blank=True, default="")

    # Resource usage (reported by host)
    peak_cpu_percent = models.FloatField(default=0.0)
    peak_memory_mb = models.FloatField(default=0.0)
    duration_seconds = models.FloatField(default=0.0)

    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["host", "status"]),
            models.Index(fields=["task_type", "status"]),
        ]

    def __str__(self) -> str:
        return f"{self.task_type} -> {self.host.name} [{self.status}]"


class TaskResourceProfile(models.Model):
    """Historical resource usage profile per task type, used for intelligent dispatch."""

    task_type = models.CharField(max_length=32, unique=True)
    avg_duration_seconds = models.FloatField(default=0.0)
    avg_peak_cpu_percent = models.FloatField(default=0.0)
    avg_peak_memory_mb = models.FloatField(default=0.0)
    sample_count = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)

    def update_from_task(self, task: DelegatedTask) -> None:
        """Rolling average update from a completed task."""
        n = self.sample_count
        if n == 0:
            self.avg_duration_seconds = task.duration_seconds
            self.avg_peak_cpu_percent = task.peak_cpu_percent
            self.avg_peak_memory_mb = task.peak_memory_mb
        else:
            # Exponential moving average with max 100 sample weight
            w = min(n, 100)
            self.avg_duration_seconds = (self.avg_duration_seconds * w + task.duration_seconds) / (w + 1)
            self.avg_peak_cpu_percent = (self.avg_peak_cpu_percent * w + task.peak_cpu_percent) / (w + 1)
            self.avg_peak_memory_mb = (self.avg_peak_memory_mb * w + task.peak_memory_mb) / (w + 1)
        self.sample_count = n + 1
        self.save()

    def __str__(self) -> str:
        return f"{self.task_type}: ~{self.avg_duration_seconds:.0f}s, ~{self.avg_peak_memory_mb:.0f}MB"


class DelegationLog(models.Model):
    """Audit log of all delegation communication."""

    class Direction(models.TextChoices):
        SENT = "sent", "Sent to Host"
        RECEIVED = "received", "Received from Host"
        ERROR = "error", "Error"

    host = models.ForeignKey(DelegationHost, on_delete=models.CASCADE, related_name="logs")
    task = models.ForeignKey(DelegatedTask, on_delete=models.SET_NULL, null=True, blank=True, related_name="logs")
    direction = models.CharField(max_length=10, choices=Direction.choices)
    message_type = models.CharField(max_length=64)  # heartbeat, task_submit, task_result, etc.
    payload_summary = models.TextField(blank=True, default="")
    payload_size_bytes = models.IntegerField(default=0)
    timestamp = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["host", "-timestamp"]),
        ]

    def __str__(self) -> str:
        return f"[{self.direction}] {self.message_type} -> {self.host.name} @ {self.timestamp}"
