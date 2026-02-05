from __future__ import annotations

from django.contrib import admin

from addressbook.models import AddressBookEntry


@admin.register(AddressBookEntry)
class AddressBookEntryAdmin(admin.ModelAdmin):
    list_display = ("name", "address", "chain", "user", "updated_at")
    search_fields = ("name", "address", "chain", "user__username")
    list_filter = ("chain",)
