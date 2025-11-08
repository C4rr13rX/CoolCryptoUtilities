from __future__ import annotations

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="SecureSetting",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=128)),
                ("category", models.CharField(default="default", max_length=64)),
                ("is_secret", models.BooleanField(default=True)),
                ("value_plain", models.TextField(blank=True, null=True)),
                ("ciphertext", models.BinaryField(blank=True, null=True)),
                ("encapsulated_key", models.BinaryField(blank=True, null=True)),
                ("nonce", models.BinaryField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="secure_settings", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["user", "category", "name"],
                "unique_together": {("user", "name", "category")},
            },
        ),
    ]
