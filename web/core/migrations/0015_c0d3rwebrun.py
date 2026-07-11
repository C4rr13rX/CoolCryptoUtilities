import uuid
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("core", "0014_systemlog")]
    operations = [
        migrations.CreateModel(
            name="C0d3rWebRun",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("status", models.CharField(default="queued", max_length=16)),
                ("prompt", models.TextField()),
                ("backend", models.CharField(blank=True, default="", max_length=32)),
                ("model", models.CharField(blank=True, default="", max_length=255)),
                ("atf_models", models.JSONField(blank=True, default=list)),
                ("output", models.TextField(blank=True, default="")),
                ("model_id", models.CharField(blank=True, default="", max_length=255)),
                ("error", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("session", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="runs", to="core.c0d3rwebsession")),
            ],
            options={"db_table": "c0d3r_web_runs"},
        ),
        migrations.AddIndex(
            model_name="c0d3rwebrun",
            index=models.Index(fields=["session", "status", "created_at"], name="c0d3r_run_state_idx"),
        ),
    ]
