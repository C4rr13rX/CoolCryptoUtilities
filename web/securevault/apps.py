from __future__ import annotations

from django.apps import AppConfig


class SecureVaultConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "securevault"
    verbose_name = "Secure Vault"
