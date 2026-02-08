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
            name="WalletNftPreference",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("chain", models.CharField(max_length=64)),
                ("contract", models.CharField(max_length=256)),
                ("token_id", models.CharField(max_length=128)),
                ("hidden", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="wallet_nft_preferences",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["chain", "contract", "token_id"],
            },
        ),
        migrations.AddIndex(
            model_name="walletnftpreference",
            index=models.Index(fields=["user", "chain"], name="walletpanel_user_id_chain_idx"),
        ),
        migrations.AddIndex(
            model_name="walletnftpreference",
            index=models.Index(fields=["user", "contract"], name="walletpanel_user_id_contract_idx"),
        ),
        migrations.AddConstraint(
            model_name="walletnftpreference",
            constraint=models.UniqueConstraint(
                fields=("user", "chain", "contract", "token_id"),
                name="uniq_wallet_nft_pref",
            ),
        ),
    ]
