from __future__ import annotations

from rest_framework import serializers

from addressbook.models import AddressBookEntry


class AddressBookEntrySerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = AddressBookEntry
        fields = [
            "id",
            "name",
            "address",
            "chain",
            "notes",
            "image",
            "image_url",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "image_url"]

    def get_image_url(self, obj: AddressBookEntry) -> str:
        if not obj.image:
            return ""
        try:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.image.url)
        except Exception:
            pass
        try:
            return obj.image.url
        except Exception:
            return ""
