from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("branddozer", "0006_brandproject_workflow")]
    operations = [
        migrations.AddField(model_name="brandproject", name="license_key", field=models.CharField(blank=True, default="unlicensed", max_length=40)),
        migrations.AddField(model_name="brandproject", name="git_auto_promote", field=models.BooleanField(default=True)),
    ]
