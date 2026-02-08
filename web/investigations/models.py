from __future__ import annotations

from django.db import models


class InvestigationProject(models.Model):
    user = models.ForeignKey("auth.User", on_delete=models.CASCADE, related_name="investigation_projects")
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    status = models.CharField(max_length=32, default="active")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "investigation_projects"
        indexes = [
            models.Index(fields=["user", "status"]),
        ]

    def __str__(self) -> str:  # pragma: no cover
        return self.name


class InvestigationTarget(models.Model):
    project = models.ForeignKey(InvestigationProject, on_delete=models.CASCADE, related_name="targets")
    url = models.TextField()
    requires_login = models.BooleanField(default=False)
    login_url = models.TextField(blank=True, default="")
    notes = models.TextField(blank=True, default="")
    crawl_policy = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "investigation_targets"
        indexes = [
            models.Index(fields=["project", "created_at"]),
        ]

    def __str__(self) -> str:  # pragma: no cover
        return self.url


class InvestigationEvidence(models.Model):
    project = models.ForeignKey(InvestigationProject, on_delete=models.CASCADE, related_name="evidence")
    target = models.ForeignKey(InvestigationTarget, null=True, blank=True, on_delete=models.SET_NULL, related_name="evidence")
    url = models.TextField(blank=True, default="")
    title = models.TextField(blank=True, default="")
    content = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    captured_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "investigation_evidence"
        indexes = [
            models.Index(fields=["project", "captured_at"]),
        ]


class InvestigationArticle(models.Model):
    project = models.ForeignKey(InvestigationProject, on_delete=models.CASCADE, related_name="articles")
    title = models.CharField(max_length=255, default="")
    body = models.TextField(blank=True, default="")
    status = models.CharField(max_length=32, default="draft")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "investigation_articles"
        indexes = [
            models.Index(fields=["project", "status"]),
        ]


class InvestigationEntity(models.Model):
    project = models.ForeignKey(InvestigationProject, on_delete=models.CASCADE, related_name="entities")
    kind = models.CharField(max_length=32, default="person")
    name = models.CharField(max_length=255)
    aliases = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "investigation_entities"
        indexes = [
            models.Index(fields=["project", "kind"]),
        ]


class InvestigationRelation(models.Model):
    project = models.ForeignKey(InvestigationProject, on_delete=models.CASCADE, related_name="relations")
    source = models.ForeignKey(InvestigationEntity, on_delete=models.CASCADE, related_name="out_links")
    target = models.ForeignKey(InvestigationEntity, on_delete=models.CASCADE, related_name="in_links")
    relation_type = models.CharField(max_length=64, default="associated")
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "investigation_relations"
        indexes = [
            models.Index(fields=["project", "relation_type"]),
        ]
