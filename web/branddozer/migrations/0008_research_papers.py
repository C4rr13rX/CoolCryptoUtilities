from __future__ import annotations

import uuid

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("branddozer", "0007_brandproject_lifecycle"),
    ]

    operations = [
        migrations.AlterField(
            model_name="deliverysession",
            name="role",
            field=models.CharField(
                choices=[
                    ("orchestrator", "Orchestrator"),
                    ("pm", "Project Manager Session"),
                    ("integrator", "Integrator/Release Session"),
                    ("dev", "C0d3rV2Session"),
                    ("qa", "QA"),
                    ("ux_audit", "UX Audit"),
                    ("research_planner", "Research Planner"),
                    ("literature_reviewer", "Literature Reviewer"),
                    ("methods_reviewer", "Methods Reviewer"),
                    ("research_writer", "Research Writer"),
                    ("citation_auditor", "Citation Auditor"),
                    ("peer_reviewer", "Peer Reviewer"),
                ],
                default="dev",
                max_length=32,
            ),
        ),
        migrations.CreateModel(
            name="ResearchPaper",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("title", models.CharField(max_length=500)),
                ("research_question", models.TextField()),
                ("abstract", models.TextField(blank=True, default="")),
                ("content_markdown", models.TextField(blank=True, default="")),
                ("keywords", models.JSONField(blank=True, default=list)),
                ("status", models.CharField(choices=[("draft", "Draft"), ("validating", "Validating"), ("revision_required", "Revision Required"), ("validated", "Validated"), ("archived", "Archived")], default="draft", max_length=32)),
                ("version", models.PositiveIntegerField(default=1)),
                ("citation_style", models.CharField(default="apa", max_length=32)),
                ("target_journal", models.CharField(blank=True, default="", max_length=255)),
                ("validation_report", models.JSONField(blank=True, default=dict)),
                ("word_count", models.PositiveIntegerField(default=0)),
                ("content_sha256", models.CharField(blank=True, default="", max_length=64)),
                ("markdown_path", models.TextField(blank=True, default="")),
                ("pdf_path", models.TextField(blank=True, default="")),
                ("validated_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("project", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="research_papers", to="branddozer.brandproject")),
                ("run", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="research_papers", to="branddozer.deliveryrun")),
            ],
            options={"ordering": ("-updated_at",)},
        ),
        migrations.CreateModel(
            name="ResearchSource",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("citation_key", models.CharField(max_length=120)),
                ("title", models.TextField()),
                ("authors", models.JSONField(blank=True, default=list)),
                ("publication_year", models.PositiveIntegerField(blank=True, null=True)),
                ("publisher", models.CharField(blank=True, default="", max_length=255)),
                ("url", models.URLField(max_length=2000)),
                ("doi", models.CharField(blank=True, default="", max_length=255)),
                ("retrieved_at", models.DateTimeField(blank=True, null=True)),
                ("content_sha256", models.CharField(blank=True, default="", max_length=64)),
                ("authority_tier", models.PositiveSmallIntegerField(default=0)),
                ("peer_reviewed", models.BooleanField(default=False)),
                ("archival", models.BooleanField(default=True)),
                ("verified_passage", models.TextField(blank=True, default="")),
                ("verification_status", models.CharField(choices=[("pending", "Pending"), ("verified", "Verified"), ("rejected", "Rejected")], default="pending", max_length=16)),
                ("verification_detail", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("paper", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="sources", to="branddozer.researchpaper")),
            ],
            options={"ordering": ("citation_key",)},
        ),
        migrations.CreateModel(
            name="ResearchClaim",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("section", models.CharField(blank=True, default="", max_length=160)),
                ("claim_text", models.TextField()),
                ("source_keys", models.JSONField(blank=True, default=list)),
                ("verification_status", models.CharField(choices=[("pending", "Pending"), ("supported", "Supported"), ("qualified", "Qualified"), ("rejected", "Rejected")], default="pending", max_length=16)),
                ("rationale", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("paper", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="claims", to="branddozer.researchpaper")),
            ],
            options={"ordering": ("id",)},
        ),
        migrations.CreateModel(
            name="ResearchPaperRevision",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("version", models.PositiveIntegerField()),
                ("content_markdown", models.TextField()),
                ("change_summary", models.TextField(blank=True, default="")),
                ("validation_report", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("paper", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="revisions", to="branddozer.researchpaper")),
            ],
            options={"ordering": ("-version",)},
        ),
        migrations.AddIndex(
            model_name="researchpaper",
            index=models.Index(fields=["status", "-updated_at"], name="branddozer_paper_status"),
        ),
        migrations.AddIndex(
            model_name="researchpaper",
            index=models.Index(fields=["project", "-updated_at"], name="branddozer_paper_project"),
        ),
        migrations.AddConstraint(
            model_name="researchsource",
            constraint=models.UniqueConstraint(fields=("paper", "citation_key"), name="branddozer_unique_paper_citation"),
        ),
        migrations.AddConstraint(
            model_name="researchpaperrevision",
            constraint=models.UniqueConstraint(fields=("paper", "version"), name="branddozer_unique_paper_revision"),
        ),
    ]
