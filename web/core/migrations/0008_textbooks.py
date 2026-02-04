from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0007_equation_graph"),
    ]

    operations = [
        migrations.CreateModel(
            name="TextbookSource",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=255)),
                ("authors", models.CharField(blank=True, default="", max_length=255)),
                ("year", models.IntegerField(blank=True, null=True)),
                ("publisher", models.CharField(blank=True, default="", max_length=255)),
                ("url", models.TextField(blank=True, default="")),
                ("print_id", models.CharField(blank=True, default="", max_length=128)),
                ("file_path", models.TextField(blank=True, default="")),
                ("sha256", models.CharField(blank=True, default="", max_length=128)),
                ("citation_apa", models.TextField(blank=True, default="")),
                ("downloaded_at", models.DateTimeField(auto_now_add=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
            ],
            options={
                "db_table": "textbook_sources",
                "indexes": [
                    models.Index(fields=["title"], name="textbook_sources_title_idx"),
                    models.Index(fields=["print_id"], name="textbook_sources_print_idx"),
                ],
            },
        ),
        migrations.CreateModel(
            name="TextbookPage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("page_num", models.IntegerField()),
                ("image_path", models.TextField(blank=True, default="")),
                ("ocr_text", models.TextField(blank=True, default="")),
                ("ocr_confidence", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("textbook", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to="core.textbooksource")),
            ],
            options={
                "db_table": "textbook_pages",
                "indexes": [
                    models.Index(fields=["page_num"], name="textbook_pages_page_idx"),
                ],
            },
        ),
    ]
