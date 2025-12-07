from __future__ import annotations

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AgentSession",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("name", models.CharField(blank=True, default="", max_length=120)),
                ("kind", models.CharField(default="browser", max_length=32)),
                ("platform", models.CharField(blank=True, default="", max_length=64)),
                ("browser", models.CharField(blank=True, default="", max_length=64)),
                ("status", models.CharField(blank=True, default="idle", max_length=32)),
                ("last_seen", models.DateTimeField(auto_now_add=True)),
                ("meta", models.JSONField(blank=True, default=dict)),
            ],
            options={
                "ordering": ["-last_seen"],
            },
        ),
        migrations.CreateModel(
            name="Task",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("title", models.CharField(max_length=200)),
                ("description", models.TextField(blank=True, default="")),
                ("target_url", models.CharField(blank=True, default="", max_length=500)),
                ("stage", models.CharField(blank=True, default="overview", max_length=64)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("in_progress", "In Progress"), ("done", "Done"), ("error", "Error")], default="pending", max_length=32)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("meta", models.JSONField(blank=True, default=dict)),
                ("assigned_to", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="tasks", to="u53rxr080t.agentsession")),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="Finding",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("title", models.CharField(max_length=200)),
                ("summary", models.TextField()),
                ("severity", models.CharField(choices=[("info", "Info"), ("warn", "Warn"), ("error", "Error")], default="info", max_length=8)),
                ("screenshot_url", models.CharField(blank=True, default="", max_length=500)),
                ("context", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("session", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="findings", to="u53rxr080t.agentsession")),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
