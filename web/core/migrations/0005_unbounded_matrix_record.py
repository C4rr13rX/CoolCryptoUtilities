from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0004_advisory_advisories_resolve_051b73_idx_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="UnboundedMatrixRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("prompt", models.TextField(blank=True, default="")),
                ("branches", models.JSONField(default=list)),
                ("matrix", models.JSONField(default=list)),
                ("integrated_mechanics", models.JSONField(default=list)),
                ("anomalies", models.JSONField(default=list)),
                ("hypotheses", models.JSONField(default=list)),
                ("experiments", models.JSONField(default=list)),
                ("decision_criteria", models.JSONField(default=list)),
                ("bounded_task", models.TextField(blank=True, default="")),
                ("constraints", models.JSONField(default=list)),
                ("payload", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "unbounded_matrix_records",
            },
        ),
        migrations.AddIndex(
            model_name="unboundedmatrixrecord",
            index=models.Index(fields=["created_at"], name="unbounded_m_created_d9b6a1_idx"),
        ),
    ]
