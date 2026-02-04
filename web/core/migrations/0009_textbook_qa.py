from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0008_textbooks"),
    ]

    operations = [
        migrations.CreateModel(
            name="TextbookQACandidate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("page_num", models.IntegerField()),
                ("question", models.TextField()),
                ("answer", models.TextField()),
                ("confidence", models.FloatField(default=0.0)),
                ("status", models.CharField(default="pending", max_length=32)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("textbook", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.textbooksource")),
            ],
            options={
                "db_table": "textbook_qa_candidates",
                "indexes": [
                    models.Index(fields=["status"], name="textbook_qa_status_idx"),
                    models.Index(fields=["page_num"], name="textbook_qa_page_idx"),
                ],
            },
        ),
    ]
