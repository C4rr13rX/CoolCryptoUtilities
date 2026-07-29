from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0015_c0d3rwebrun"),
    ]

    operations = [
        migrations.AddField(
            model_name="c0d3rwebrun",
            name="wizard_brain_id",
            field=models.CharField(blank=True, default="", max_length=64),
        ),
        migrations.AddField(
            model_name="c0d3rwebrun",
            name="wizard_endpoint",
            field=models.URLField(blank=True, default="", max_length=500),
        ),
        migrations.AddField(
            model_name="c0d3rwebrun",
            name="wizard_chat_path",
            field=models.CharField(blank=True, default="", max_length=32),
        ),
    ]
