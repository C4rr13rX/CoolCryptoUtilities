from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("branddozer", "0005_alter_deliverysession_role")]
    operations = [
        migrations.AddField(
            model_name="brandproject",
            name="workflow_kind",
            field=models.CharField(blank=True, default="", max_length=120),
        ),
        migrations.AddField(
            model_name="brandproject",
            name="workflow_config",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
