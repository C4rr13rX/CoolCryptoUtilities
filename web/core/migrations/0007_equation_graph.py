from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0006_unbounded_matrix_record_equations"),
    ]

    operations = [
        migrations.CreateModel(
            name="EquationDiscipline",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=128, unique=True)),
                ("description", models.TextField(blank=True, default="")),
            ],
            options={
                "db_table": "equation_disciplines",
            },
        ),
        migrations.CreateModel(
            name="EquationSource",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("title", models.CharField(max_length=255)),
                ("url", models.TextField(blank=True, default="")),
                ("authors", models.CharField(blank=True, default="", max_length=255)),
                ("year", models.IntegerField(blank=True, null=True)),
                ("publisher", models.CharField(blank=True, default="", max_length=255)),
                ("accessed_at", models.DateTimeField(auto_now_add=True)),
                ("citation", models.TextField(blank=True, default="")),
                ("tags", models.JSONField(default=list)),
                ("raw_excerpt", models.TextField(blank=True, default="")),
            ],
            options={
                "db_table": "equation_sources",
            },
        ),
        migrations.CreateModel(
            name="EquationVariable",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(max_length=64)),
                ("name", models.CharField(blank=True, default="", max_length=255)),
                ("units", models.CharField(blank=True, default="", max_length=128)),
                ("dimension", models.CharField(blank=True, default="", max_length=128)),
                ("description", models.TextField(blank=True, default="")),
            ],
            options={
                "db_table": "equation_variables",
                "unique_together": {("symbol", "dimension")},
            },
        ),
        migrations.CreateModel(
            name="Equation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("text", models.TextField()),
                ("latex", models.TextField(blank=True, default="")),
                ("variables", models.JSONField(default=list)),
                ("constraints", models.JSONField(default=list)),
                ("assumptions", models.JSONField(default=list)),
                ("domains", models.JSONField(default=list)),
                ("disciplines", models.JSONField(default=list)),
                ("confidence", models.FloatField(default=0.5)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("source", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="core.equationsource")),
            ],
            options={
                "db_table": "equations",
            },
        ),
        migrations.CreateModel(
            name="EquationLink",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("relation_type", models.CharField(default="bridges", max_length=64)),
                ("notes", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("from_equation", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="out_links", to="core.equation")),
                ("to_equation", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="in_links", to="core.equation")),
            ],
            options={
                "db_table": "equation_links",
            },
        ),
        migrations.CreateModel(
            name="EquationGapFill",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("description", models.TextField()),
                ("steps", models.JSONField(default=list)),
                ("status", models.CharField(default="open", max_length=32)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("equations", models.ManyToManyField(related_name="gap_fills", to="core.equation")),
            ],
            options={
                "db_table": "equation_gap_fills",
            },
        ),
        migrations.AddIndex(
            model_name="equation",
            index=models.Index(fields=["created_at"], name="equations_created_95a7e2_idx"),
        ),
        migrations.AddIndex(
            model_name="equationlink",
            index=models.Index(fields=["relation_type"], name="equation_l_relation_1a7a60_idx"),
        ),
    ]
