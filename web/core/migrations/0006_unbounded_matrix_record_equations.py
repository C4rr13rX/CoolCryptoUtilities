from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0005_unbounded_matrix_record"),
    ]

    operations = [
        migrations.AddField(
            model_name="unboundedmatrixrecord",
            name="equations",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="unboundedmatrixrecord",
            name="gap_fill_steps",
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name="unboundedmatrixrecord",
            name="research_links",
            field=models.JSONField(default=list),
        ),
    ]
