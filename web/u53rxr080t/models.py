from __future__ import annotations

import uuid
from django.db import models
from django.utils import timezone
from django.contrib.postgres.fields import ArrayField


class AgentSession(models.Model):
    """
    Represents a running UX agent (browser plugin or Rust daemon) reporting in.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=120, blank=True, default="")
    kind = models.CharField(max_length=32, default="browser")  # browser | daemon
    platform = models.CharField(max_length=64, blank=True, default="")
    browser = models.CharField(max_length=64, blank=True, default="")
    status = models.CharField(max_length=32, blank=True, default="idle")
    last_seen = models.DateTimeField(default=timezone.now)
    meta = models.JSONField(blank=True, default=dict)

    class Meta:
        ordering = ["-last_seen"]

    def __str__(self) -> str:
        return f"{self.name or self.id} ({self.kind})"


class Task(models.Model):
    """
    Work items for agents to execute (e.g., visit route, capture screenshot, run UX checklist).
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("in_progress", "In Progress"),
        ("done", "Done"),
        ("error", "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    target_url = models.CharField(max_length=500, blank=True, default="")
    stage = models.CharField(max_length=64, blank=True, default="overview")
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default="pending")
    assigned_to = models.ForeignKey(AgentSession, null=True, blank=True, on_delete=models.SET_NULL, related_name="tasks")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    meta = models.JSONField(blank=True, default=dict)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title


class Finding(models.Model):
    """
    UX issues or observations reported by agents.
    """

    SEVERITY_CHOICES = [
        ("info", "Info"),
        ("warn", "Warn"),
        ("error", "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(AgentSession, on_delete=models.CASCADE, related_name="findings")
    title = models.CharField(max_length=200)
    summary = models.TextField()
    severity = models.CharField(max_length=8, choices=SEVERITY_CHOICES, default="info")
    screenshot_url = models.CharField(max_length=500, blank=True, default="")
    context = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.severity}: {self.title}"
