from __future__ import annotations

from django.conf import settings
from django.db import models


class AddressBookEntry(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="address_book_entries")
    name = models.CharField(max_length=128)
    address = models.CharField(max_length=256)
    chain = models.CharField(max_length=64, blank=True)
    notes = models.TextField(blank=True)
    image = models.ImageField(upload_to="addressbook/", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["name", "address"]

    def __str__(self) -> str:  # pragma: no cover - admin aid
        return f"{self.name} ({self.address})"
