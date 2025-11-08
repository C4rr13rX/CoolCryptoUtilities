from __future__ import annotations

from django.conf import settings
from django.db import models


class SecureSetting(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="secure_settings")
    name = models.CharField(max_length=128)
    category = models.CharField(max_length=64, default="default")
    is_secret = models.BooleanField(default=True)
    value_plain = models.TextField(blank=True, null=True)
    ciphertext = models.BinaryField(blank=True, null=True)
    encapsulated_key = models.BinaryField(blank=True, null=True)
    nonce = models.BinaryField(blank=True, null=True)
    tag = models.BinaryField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "name", "category")
        ordering = ["user", "category", "name"]

    def __str__(self) -> str:  # pragma: no cover - admin aid
        return f"{self.user}::{self.category}/{self.name}"
