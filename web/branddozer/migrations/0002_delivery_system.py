# Generated manually for BrandDozer delivery system models.
from __future__ import annotations

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("branddozer", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="DeliveryRun",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("prompt", models.TextField()),
                ("mode", models.CharField(default="auto", max_length=32)),
                ("status", models.CharField(choices=[("queued", "Queued"), ("running", "Running"), ("blocked", "Blocked"), ("awaiting_acceptance", "Awaiting Acceptance"), ("complete", "Complete"), ("error", "Error")], default="queued", max_length=40)),
                ("phase", models.CharField(blank=True, default="", max_length=64)),
                ("iteration", models.PositiveIntegerField(default=0)),
                ("sprint_count", models.PositiveIntegerField(default=0)),
                ("acceptance_required", models.BooleanField(default=True)),
                ("acceptance_recorded", models.BooleanField(default=False)),
                ("definition_of_done", models.JSONField(blank=True, default=list)),
                ("context", models.JSONField(blank=True, default=dict)),
                ("error", models.TextField(blank=True, default="")),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="delivery_runs", to="branddozer.brandproject")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="DeliveryProject",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("mode", models.CharField(choices=[("auto", "Auto"), ("new", "New Project"), ("existing", "Existing Project")], default="auto", max_length=32)),
                ("status", models.CharField(choices=[("idle", "Idle"), ("running", "Running"), ("blocked", "Blocked"), ("complete", "Complete"), ("error", "Error")], default="idle", max_length=32)),
                ("definition_of_done", models.JSONField(blank=True, default=list)),
                ("constraints", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("active_run", models.ForeignKey(blank=True, null=True, on_delete=models.deletion.SET_NULL, related_name="+", to="branddozer.deliveryrun")),
                ("project", models.OneToOneField(on_delete=models.deletion.CASCADE, related_name="delivery", to="branddozer.brandproject")),
            ],
            options={
                "ordering": ("-updated_at",),
            },
        ),
        migrations.CreateModel(
            name="GovernanceArtifact",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(choices=[("baseline_report", "Baseline Report"), ("requirements", "Requirements"), ("blueprint", "Blueprint"), ("charter", "Project Charter"), ("wbs", "Work Breakdown Structure"), ("quality_plan", "Quality Management Plan"), ("raid_log", "RAID Log"), ("change_control", "Change Control"), ("release_criteria", "Release Criteria"), ("completion_report", "Completion Report")], max_length=64)),
                ("version", models.PositiveIntegerField(default=1)),
                ("summary", models.TextField(blank=True, default="")),
                ("content", models.TextField(blank=True, default="")),
                ("data", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="governance_artifacts", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="governance_artifacts", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="BacklogItem",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(choices=[("epic", "Epic"), ("story", "Story"), ("task", "Task"), ("bug", "Bug")], default="task", max_length=16)),
                ("title", models.CharField(max_length=240)),
                ("description", models.TextField(blank=True, default="")),
                ("acceptance_criteria", models.JSONField(blank=True, default=list)),
                ("priority", models.PositiveIntegerField(default=3)),
                ("estimate_points", models.FloatField(default=1.0)),
                ("status", models.CharField(choices=[("todo", "To Do"), ("in_progress", "In Progress"), ("blocked", "Blocked"), ("done", "Done")], default="todo", max_length=24)),
                ("source", models.CharField(blank=True, default="", max_length=64)),
                ("dependencies", models.JSONField(blank=True, default=list)),
                ("meta", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="backlog_items", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="backlog_items", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("priority", "-created_at"),
            },
        ),
        migrations.CreateModel(
            name="Sprint",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("number", models.PositiveIntegerField(default=1)),
                ("goal", models.CharField(blank=True, default="", max_length=255)),
                ("status", models.CharField(choices=[("planning", "Planning"), ("active", "Active"), ("review", "Review"), ("retro", "Retro"), ("complete", "Complete")], default="planning", max_length=24)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("retrospective", models.TextField(blank=True, default="")),
                ("meta", models.JSONField(blank=True, default=dict)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="sprints", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="sprints", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-number",),
            },
        ),
        migrations.CreateModel(
            name="SprintItem",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(default="todo", max_length=24)),
                ("owner", models.CharField(blank=True, default="", max_length=120)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("backlog_item", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="sprint_items", to="branddozer.backlogitem")),
                ("sprint", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="items", to="branddozer.sprint")),
            ],
            options={
                "unique_together": {("sprint", "backlog_item")},
            },
        ),
        migrations.CreateModel(
            name="DeliverySession",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("role", models.CharField(choices=[("orchestrator", "Orchestrator"), ("pm", "Project Manager Session"), ("integrator", "Integrator/Release Session"), ("dev", "CodexSession"), ("qa", "QA")], default="dev", max_length=32)),
                ("name", models.CharField(blank=True, default="", max_length=120)),
                ("status", models.CharField(choices=[("queued", "Queued"), ("running", "Running"), ("blocked", "Blocked"), ("done", "Done"), ("error", "Error")], default="queued", max_length=24)),
                ("workspace_path", models.TextField(blank=True, default="")),
                ("log_path", models.TextField(blank=True, default="")),
                ("last_heartbeat", models.DateTimeField(blank=True, null=True)),
                ("meta", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="delivery_sessions", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="sessions", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="GateRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("stage", models.CharField(default="fast", max_length=32)),
                ("name", models.CharField(max_length=120)),
                ("status", models.CharField(choices=[("passed", "Passed"), ("failed", "Failed"), ("skipped", "Skipped"), ("blocked", "Blocked")], default="blocked", max_length=16)),
                ("command", models.TextField(blank=True, default="")),
                ("stdout", models.TextField(blank=True, default="")),
                ("stderr", models.TextField(blank=True, default="")),
                ("exit_code", models.IntegerField(blank=True, null=True)),
                ("duration_ms", models.PositiveIntegerField(default=0)),
                ("input_hash", models.CharField(blank=True, default="", max_length=64)),
                ("meta", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="gate_runs", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="gate_runs", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="DeliveryArtifact",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(default="artifact", max_length=64)),
                ("title", models.CharField(blank=True, default="", max_length=200)),
                ("path", models.TextField(blank=True, default="")),
                ("content", models.TextField(blank=True, default="")),
                ("data", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="delivery_artifacts", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="artifacts", to="branddozer.deliveryrun")),
                ("session", models.ForeignKey(blank=True, null=True, on_delete=models.deletion.SET_NULL, related_name="artifacts", to="branddozer.deliverysession")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="RaidEntry",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(choices=[("risk", "Risk"), ("issue", "Issue"), ("assumption", "Assumption"), ("dependency", "Dependency")], default="risk", max_length=16)),
                ("title", models.CharField(max_length=200)),
                ("description", models.TextField(blank=True, default="")),
                ("severity", models.CharField(blank=True, default="medium", max_length=16)),
                ("status", models.CharField(choices=[("open", "Open"), ("monitoring", "Monitoring"), ("resolved", "Resolved")], default="open", max_length=16)),
                ("mitigation", models.TextField(blank=True, default="")),
                ("owner", models.CharField(blank=True, default="", max_length=120)),
                ("meta", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="raid_entries", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="raid_entries", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="ChangeRequest",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=200)),
                ("description", models.TextField(blank=True, default="")),
                ("rationale", models.TextField(blank=True, default="")),
                ("impact", models.TextField(blank=True, default="")),
                ("status", models.CharField(choices=[("proposed", "Proposed"), ("approved", "Approved"), ("rejected", "Rejected"), ("implemented", "Implemented")], default="proposed", max_length=16)),
                ("requested_by", models.CharField(blank=True, default="", max_length=120)),
                ("approved_by", models.CharField(blank=True, default="", max_length=120)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="change_requests", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="change_requests", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="ReleaseCandidate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("version", models.CharField(max_length=64)),
                ("status", models.CharField(default="created", max_length=24)),
                ("summary", models.TextField(blank=True, default="")),
                ("artifact_path", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="release_candidates", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="release_candidates", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-created_at",),
            },
        ),
        migrations.CreateModel(
            name="AcceptanceRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("user_name", models.CharField(blank=True, default="", max_length=120)),
                ("notes", models.TextField(blank=True, default="")),
                ("checklist", models.JSONField(blank=True, default=list)),
                ("accepted_at", models.DateTimeField(auto_now_add=True)),
                ("project", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="acceptance_records", to="branddozer.brandproject")),
                ("run", models.ForeignKey(on_delete=models.deletion.CASCADE, related_name="acceptance_records", to="branddozer.deliveryrun")),
            ],
            options={
                "ordering": ("-accepted_at",),
            },
        ),
    ]
