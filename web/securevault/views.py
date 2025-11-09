from __future__ import annotations

from django.db import transaction
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from services.secure_settings import encrypt_secret
from .models import SecureSetting
from .serializers import SecureSettingSerializer


class SecureSettingListCreateView(generics.ListCreateAPIView):
    serializer_class = SecureSettingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SecureSetting.objects.filter(user=self.request.user).order_by("category", "name")

    def delete(self, request, *args, **kwargs):
        category = request.query_params.get("category") or request.data.get("category")
        queryset = self.get_queryset()
        if category:
            queryset = queryset.filter(category__iexact=category.strip())
        deleted, _ = queryset.delete()
        return Response({"deleted": deleted}, status=status.HTTP_200_OK)


class SecureSettingDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = SecureSettingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SecureSetting.objects.filter(user=self.request.user)


class SecureSettingBulkImportView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        content = request.data.get("content")
        if not isinstance(content, str) or not content.strip():
            return Response({"detail": "content is required"}, status=status.HTTP_400_BAD_REQUEST)
        treat_as_secret = request.data.get("is_secret", True)
        entries = _parse_env_block(content)
        if not entries:
            return Response({"detail": "no settings detected"}, status=status.HTTP_400_BAD_REQUEST)

        normalized_category = "default"
        imported = 0
        with transaction.atomic():
            for name, value in entries:
                setting, _ = SecureSetting.objects.get_or_create(
                    user=request.user,
                    category=normalized_category,
                    name=name,
                    defaults={"is_secret": bool(treat_as_secret)},
                )
                setting.is_secret = bool(treat_as_secret)
                if setting.is_secret:
                    payload = encrypt_secret(value)
                    setting.value_plain = None
                    setting.ciphertext = payload["ciphertext"]
                    setting.encapsulated_key = payload["encapsulated_key"]
                    setting.nonce = payload["nonce"]
                else:
                    setting.value_plain = value
                    setting.ciphertext = None
                    setting.encapsulated_key = None
                    setting.nonce = None
                setting.save()
                imported += 1
        return Response({"imported": imported, "category": normalized_category}, status=status.HTTP_201_CREATED)


def _parse_env_block(content: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        working = _strip_inline_comment(line)
        if "=" not in working:
            continue
        name_part, value_part = working.split("=", 1)
        name = name_part.strip()
        if not name:
            continue
        value = value_part
        entries.append((name, value))
    return entries


def _strip_inline_comment(line: str) -> str:
    in_quotes = False
    quote_char = ""
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch in {'"', "'"}:
            if in_quotes and quote_char == ch:
                in_quotes = False
                quote_char = ""
            elif not in_quotes:
                in_quotes = True
                quote_char = ch
            continue
        if ch == "#" and not in_quotes:
            return line[:idx]
    return line
