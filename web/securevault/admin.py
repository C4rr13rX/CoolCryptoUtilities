from __future__ import annotations

from django.contrib import admin

from .models import SecureSetting


@admin.register(SecureSetting)
class SecureSettingAdmin(admin.ModelAdmin):
    list_display = ("user", "category", "name", "is_secret", "updated_at")
    search_fields = ("name", "category", "user__username", "user__email")
    list_filter = ("is_secret", "category")
    readonly_fields = ("created_at", "updated_at", "ciphertext", "encapsulated_key", "nonce")
