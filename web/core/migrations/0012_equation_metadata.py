from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0011_rename_equations_created_95a7e2_idx_equations_created_d0189c_idx_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="equation",
            name="citations",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="equation",
            name="tool_used",
            field=models.CharField(blank=True, default="", max_length=128),
        ),
        migrations.AddField(
            model_name="equation",
            name="captured_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
