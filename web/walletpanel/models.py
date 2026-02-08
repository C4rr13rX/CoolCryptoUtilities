from __future__ import annotations

from django.conf import settings
from django.db import models


class WalletNftPreference(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="wallet_nft_preferences")
    chain = models.CharField(max_length=64)
    contract = models.CharField(max_length=256)
    token_id = models.CharField(max_length=128)
    hidden = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["chain", "contract", "token_id"]
        indexes = [
            models.Index(fields=["user", "chain"]),
            models.Index(fields=["user", "contract"]),
        ]
        constraints = [
            models.UniqueConstraint(fields=["user", "chain", "contract", "token_id"], name="uniq_wallet_nft_pref")
        ]

    def __str__(self) -> str:  # pragma: no cover - admin aid
        return f"{self.user_id}:{self.chain}:{self.contract}:{self.token_id}"
