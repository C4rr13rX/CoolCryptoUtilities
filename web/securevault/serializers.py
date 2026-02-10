from __future__ import annotations

from rest_framework import serializers

from services.secure_settings import decrypt_secret, encrypt_secret, mask_value
from .models import SecureSetting


class SecureSettingSerializer(serializers.ModelSerializer):
    value = serializers.CharField(write_only=True, required=False, allow_blank=True)
    preview = serializers.SerializerMethodField()
    revealed_value = serializers.SerializerMethodField()
    label = serializers.SerializerMethodField()
    is_placeholder = serializers.SerializerMethodField()

    class Meta:
        model = SecureSetting
        fields = [
            "id",
            "name",
            "category",
            "is_secret",
            "preview",
            "revealed_value",
            "value",
            "updated_at",
            "label",
            "is_placeholder",
        ]
        read_only_fields = ["updated_at", "preview", "revealed_value", "label", "is_placeholder"]

    def get_preview(self, obj: SecureSetting) -> str:
        if obj.is_secret:
            return mask_value("secret")
        return obj.value_plain or ""

    def get_revealed_value(self, obj: SecureSetting) -> str | None:
        request = self.context.get("request")
        reveal = request and request.query_params.get("reveal") == "1"
        if not reveal:
            return None
        if obj.is_secret:
            if not (obj.encapsulated_key and obj.ciphertext and obj.nonce):
                return None
            try:
                return decrypt_secret(obj.encapsulated_key, obj.ciphertext, obj.nonce)
            except Exception:
                return obj.value_plain or ""
        return obj.value_plain or ""

    def create(self, validated_data):
        value = validated_data.pop("value", None)
        user = self.context["request"].user
        instance = SecureSetting(user=user, **validated_data)
        self._apply_value(instance, value)
        instance.save()
        self._trigger_bootstrap(instance)
        return instance

    def update(self, instance: SecureSetting, validated_data):
        value = validated_data.pop("value", None)
        for key, val in validated_data.items():
            setattr(instance, key, val)
        if value is not None or instance.is_secret:
            self._apply_value(instance, value)
        instance.save()
        self._trigger_bootstrap(instance)
        return instance

    def _apply_value(self, instance: SecureSetting, value: Optional[str]) -> None:
        if instance.is_secret:
            if value is None:
                raise serializers.ValidationError({"value": "This field is required for secret settings."})
            payload = encrypt_secret(value)
            instance.value_plain = None
            instance.ciphertext = payload["ciphertext"]
            instance.encapsulated_key = payload["encapsulated_key"]
            instance.nonce = payload["nonce"]
        else:
            instance.value_plain = value or ""
            instance.ciphertext = None
            instance.encapsulated_key = None
            instance.nonce = None

    def _trigger_bootstrap(self, instance: SecureSetting) -> None:
        name = str(instance.name or "").strip().upper()
        if not name:
            return
        if name.endswith("_API_KEY") or name.startswith("RPC_") or name in {"MNEMONIC", "PRIVATE_KEY"}:
            try:
                from services.internal_cron import cron_supervisor

                cron_supervisor.run_once("auto_pipeline")
            except Exception:
                pass

    def get_label(self, obj: SecureSetting) -> str | None:
        from services.secure_settings_catalog import CATALOG_LOOKUP

        key = ((obj.category or "default").lower(), obj.name)
        entry = CATALOG_LOOKUP.get(key)
        return entry.get("label") if entry else None

    def get_is_placeholder(self, obj: SecureSetting) -> bool:
        return False
