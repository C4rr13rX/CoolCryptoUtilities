from __future__ import annotations

import uuid
from pathlib import Path

from django.db import models


def _default_log_path() -> str:
    return str(Path("runtime/branddozer") / f"{uuid.uuid4()}.log")


class BrandProject(models.Model):
    """
    BrandDozer project configuration persisted in the Django database.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    root_path = models.TextField()
    default_prompt = models.TextField(blank=True, default="")
    interjections = models.JSONField(default=list, blank=True)
    interval_minutes = models.PositiveIntegerField(default=120)
    enabled = models.BooleanField(default=False)
    last_run = models.DateTimeField(null=True, blank=True)
    last_ai_generated = models.DateTimeField(null=True, blank=True)
    log_path = models.TextField(default=_default_log_path)
    repo_url = models.TextField(blank=True, default="")
    repo_branch = models.CharField(max_length=120, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.name} ({self.root_path})"


class DeliveryProject(models.Model):
    """
    Delivery system configuration linked to a BrandDozer project.
    """

    MODE_CHOICES = [
        ("auto", "Auto"),
        ("new", "New Project"),
        ("existing", "Existing Project"),
    ]

    STATUS_CHOICES = [
        ("idle", "Idle"),
        ("running", "Running"),
        ("blocked", "Blocked"),
        ("complete", "Complete"),
        ("error", "Error"),
    ]

    project = models.OneToOneField(BrandProject, on_delete=models.CASCADE, related_name="delivery")
    mode = models.CharField(max_length=32, choices=MODE_CHOICES, default="auto")
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default="idle")
    definition_of_done = models.JSONField(default=list, blank=True)
    constraints = models.JSONField(default=dict, blank=True)
    active_run = models.ForeignKey("DeliveryRun", null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Delivery for {self.project.name}"


class DeliveryRun(models.Model):
    """
    One closed-loop delivery run from a single user prompt.
    """

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("blocked", "Blocked"),
        ("awaiting_acceptance", "Awaiting Acceptance"),
        ("complete", "Complete"),
        ("error", "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="delivery_runs")
    prompt = models.TextField()
    mode = models.CharField(max_length=32, default="auto")
    status = models.CharField(max_length=40, choices=STATUS_CHOICES, default="queued")
    phase = models.CharField(max_length=64, blank=True, default="")
    iteration = models.PositiveIntegerField(default=0)
    sprint_count = models.PositiveIntegerField(default=0)
    acceptance_required = models.BooleanField(default=True)
    acceptance_recorded = models.BooleanField(default=False)
    definition_of_done = models.JSONField(default=list, blank=True)
    context = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True, default="")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"Run {self.id} ({self.status})"


class GovernanceArtifact(models.Model):
    """
    Governance artifacts (charter, WBS, requirements, blueprint, baseline).
    """

    KIND_CHOICES = [
        ("baseline_report", "Baseline Report"),
        ("requirements", "Requirements"),
        ("blueprint", "Blueprint"),
        ("charter", "Project Charter"),
        ("wbs", "Work Breakdown Structure"),
        ("quality_plan", "Quality Management Plan"),
        ("raid_log", "RAID Log"),
        ("change_control", "Change Control"),
        ("release_criteria", "Release Criteria"),
        ("completion_report", "Completion Report"),
    ]

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="governance_artifacts")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="governance_artifacts")
    kind = models.CharField(max_length=64, choices=KIND_CHOICES)
    version = models.PositiveIntegerField(default=1)
    summary = models.TextField(blank=True, default="")
    content = models.TextField(blank=True, default="")
    data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)


class BacklogItem(models.Model):
    """
    SCRUM backlog items (epic/story/task/bug).
    """

    KIND_CHOICES = [
        ("epic", "Epic"),
        ("story", "Story"),
        ("task", "Task"),
        ("bug", "Bug"),
    ]

    STATUS_CHOICES = [
        ("todo", "To Do"),
        ("in_progress", "In Progress"),
        ("blocked", "Blocked"),
        ("done", "Done"),
    ]

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="backlog_items")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="backlog_items")
    kind = models.CharField(max_length=16, choices=KIND_CHOICES, default="task")
    title = models.CharField(max_length=240)
    description = models.TextField(blank=True, default="")
    acceptance_criteria = models.JSONField(default=list, blank=True)
    priority = models.PositiveIntegerField(default=3)
    estimate_points = models.FloatField(default=1.0)
    status = models.CharField(max_length=24, choices=STATUS_CHOICES, default="todo")
    source = models.CharField(max_length=64, blank=True, default="")
    dependencies = models.JSONField(default=list, blank=True)
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("priority", "-created_at")


class Sprint(models.Model):
    """
    SCRUM sprint container.
    """

    STATUS_CHOICES = [
        ("planning", "Planning"),
        ("active", "Active"),
        ("review", "Review"),
        ("retro", "Retro"),
        ("complete", "Complete"),
    ]

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="sprints")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="sprints")
    number = models.PositiveIntegerField(default=1)
    goal = models.CharField(max_length=255, blank=True, default="")
    status = models.CharField(max_length=24, choices=STATUS_CHOICES, default="planning")
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    retrospective = models.TextField(blank=True, default="")
    meta = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ("-number",)


class SprintItem(models.Model):
    sprint = models.ForeignKey(Sprint, on_delete=models.CASCADE, related_name="items")
    backlog_item = models.ForeignKey(BacklogItem, on_delete=models.CASCADE, related_name="sprint_items")
    status = models.CharField(max_length=24, default="todo")
    owner = models.CharField(max_length=120, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("sprint", "backlog_item")


class DeliverySession(models.Model):
    """
    CodexSession records for Orchestrator/PM/Integrator/Dev workers.
    """

    ROLE_CHOICES = [
        ("orchestrator", "Orchestrator"),
        ("pm", "Project Manager Session"),
        ("integrator", "Integrator/Release Session"),
        ("dev", "CodexSession"),
        ("qa", "QA"),
    ]

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("blocked", "Blocked"),
        ("done", "Done"),
        ("error", "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="delivery_sessions")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="sessions")
    role = models.CharField(max_length=32, choices=ROLE_CHOICES, default="dev")
    name = models.CharField(max_length=120, blank=True, default="")
    status = models.CharField(max_length=24, choices=STATUS_CHOICES, default="queued")
    workspace_path = models.TextField(blank=True, default="")
    log_path = models.TextField(blank=True, default="")
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)


class GateRun(models.Model):
    """
    Quality gate execution results.
    """

    STATUS_CHOICES = [
        ("passed", "Passed"),
        ("failed", "Failed"),
        ("skipped", "Skipped"),
        ("blocked", "Blocked"),
    ]

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="gate_runs")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="gate_runs")
    stage = models.CharField(max_length=32, default="fast")
    name = models.CharField(max_length=120)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="blocked")
    command = models.TextField(blank=True, default="")
    stdout = models.TextField(blank=True, default="")
    stderr = models.TextField(blank=True, default="")
    exit_code = models.IntegerField(null=True, blank=True)
    duration_ms = models.PositiveIntegerField(default=0)
    input_hash = models.CharField(max_length=64, blank=True, default="")
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)


class DeliveryArtifact(models.Model):
    """
    Supporting artifacts (diffs, logs, evidence).
    """

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="delivery_artifacts")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="artifacts")
    session = models.ForeignKey(DeliverySession, null=True, blank=True, on_delete=models.SET_NULL, related_name="artifacts")
    kind = models.CharField(max_length=64, default="artifact")
    title = models.CharField(max_length=200, blank=True, default="")
    path = models.TextField(blank=True, default="")
    content = models.TextField(blank=True, default="")
    data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)


class RaidEntry(models.Model):
    """
    RAID log entry.
    """

    KIND_CHOICES = [
        ("risk", "Risk"),
        ("issue", "Issue"),
        ("assumption", "Assumption"),
        ("dependency", "Dependency"),
    ]

    STATUS_CHOICES = [
        ("open", "Open"),
        ("monitoring", "Monitoring"),
        ("resolved", "Resolved"),
    ]

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="raid_entries")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="raid_entries")
    kind = models.CharField(max_length=16, choices=KIND_CHOICES, default="risk")
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    severity = models.CharField(max_length=16, blank=True, default="medium")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="open")
    mitigation = models.TextField(blank=True, default="")
    owner = models.CharField(max_length=120, blank=True, default="")
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)


class ChangeRequest(models.Model):
    """
    Change control log.
    """

    STATUS_CHOICES = [
        ("proposed", "Proposed"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
        ("implemented", "Implemented"),
    ]

    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="change_requests")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="change_requests")
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    rationale = models.TextField(blank=True, default="")
    impact = models.TextField(blank=True, default="")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="proposed")
    requested_by = models.CharField(max_length=120, blank=True, default="")
    approved_by = models.CharField(max_length=120, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-created_at",)


class ReleaseCandidate(models.Model):
    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="release_candidates")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="release_candidates")
    version = models.CharField(max_length=64)
    status = models.CharField(max_length=24, default="created")
    summary = models.TextField(blank=True, default="")
    artifact_path = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)


class AcceptanceRecord(models.Model):
    project = models.ForeignKey(BrandProject, on_delete=models.CASCADE, related_name="acceptance_records")
    run = models.ForeignKey(DeliveryRun, on_delete=models.CASCADE, related_name="acceptance_records")
    user_name = models.CharField(max_length=120, blank=True, default="")
    notes = models.TextField(blank=True, default="")
    checklist = models.JSONField(default=list, blank=True)
    accepted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-accepted_at",)
