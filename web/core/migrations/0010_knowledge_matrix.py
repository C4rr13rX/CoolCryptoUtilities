from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0009_textbook_qa"),
    ]

    operations = [
        migrations.CreateModel(
            name="KnowledgeDocument",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("source", models.CharField(blank=True, default="", max_length=255)),
                ("title", models.TextField(blank=True, default="")),
                ("abstract", models.TextField(blank=True, default="")),
                ("body", models.TextField(blank=True, default="")),
                ("url", models.TextField(blank=True, default="")),
                ("citation_apa", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
            ],
            options={
                "db_table": "knowledge_documents",
            },
        ),
        migrations.CreateModel(
            name="KnowledgeQueueItem",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(default="pending", max_length=32)),
                ("confidence", models.FloatField(default=0.0)),
                ("label", models.CharField(blank=True, default="", max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "document",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.knowledgedocument"),
                ),
            ],
            options={
                "db_table": "knowledge_queue_items",
            },
        ),
        migrations.CreateModel(
            name="LabelQueueItem",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("status", models.CharField(default="pending", max_length=32)),
                ("label", models.CharField(blank=True, default="", max_length=255)),
                ("confidence", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "document",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.knowledgedocument"),
                ),
            ],
            options={
                "db_table": "label_queue_items",
            },
        ),
        migrations.CreateModel(
            name="VisualLabelQueueItem",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("image_path", models.TextField(blank=True, default="")),
                ("status", models.CharField(default="pending", max_length=32)),
                ("label", models.CharField(blank=True, default="", max_length=255)),
                ("confidence", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "document",
                    models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.knowledgedocument"),
                ),
            ],
            options={
                "db_table": "visual_label_queue_items",
            },
        ),
        migrations.CreateModel(
            name="TechMatrixRecord",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("prompt", models.TextField(blank=True, default="")),
                ("components", models.JSONField(default=list)),
                ("requirements", models.JSONField(default=list)),
                ("interfaces", models.JSONField(default=list)),
                ("data_shapes", models.JSONField(default=list)),
                ("outline", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "tech_matrix_records",
            },
        ),
        migrations.AddIndex(
            model_name="knowledgedocument",
            index=models.Index(fields=["created_at"], name="knowledge_docs_created_idx"),
        ),
        migrations.AddIndex(
            model_name="knowledgequeueitem",
            index=models.Index(fields=["status"], name="knowledge_queue_status_idx"),
        ),
        migrations.AddIndex(
            model_name="labelqueueitem",
            index=models.Index(fields=["status"], name="label_queue_status_idx"),
        ),
        migrations.AddIndex(
            model_name="visuallabelqueueitem",
            index=models.Index(fields=["status"], name="visual_label_queue_status_idx"),
        ),
        migrations.AddIndex(
            model_name="techmatrixrecord",
            index=models.Index(fields=["created_at"], name="tech_matrix_created_idx"),
        ),
    ]
