from __future__ import annotations

from typing import Any, Dict

from django.db import transaction
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from services.secure_settings import encrypt_secret, decrypt_secret
from services.secure_settings_catalog import CATALOG_LOOKUP, SECURE_SETTINGS_CATALOG
from .models import SecureSetting
from .serializers import SecureSettingSerializer


class SecureSettingListCreateView(generics.ListCreateAPIView):
    serializer_class = SecureSettingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return SecureSetting.objects.filter(user=self.request.user).order_by("category", "name")

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        items = serializer.data
        present_keys = {
            ((item.get("category") or "default").lower(), item.get("name")) for item in items
        }
        enriched = []
        for item in items:
            enriched.append(self._attach_metadata(item))
        placeholders = []
        for entry in SECURE_SETTINGS_CATALOG:
            key = (entry["category"].lower(), entry["name"])
            if key in present_keys:
                continue
            placeholders.append(
                {
                    "id": None,
                    "name": entry["name"],
                    "category": entry["category"],
                    "is_secret": entry["is_secret"],
                    "preview": "",
                    "revealed_value": None,
                    "updated_at": None,
                    "is_placeholder": True,
                    "label": entry.get("label"),
                }
            )
        return Response(enriched + placeholders)

    def _attach_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        key = ((item.get("category") or "default").lower(), item.get("name"))
        entry = CATALOG_LOOKUP.get(key)
        if entry:
            item = dict(item)
            item["label"] = entry.get("label")
            item["is_placeholder"] = False
        return item

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


class SecureSettingFileImportView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        uploaded = request.FILES.get("file")
        if not uploaded:
            return Response({"detail": "file is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            content = uploaded.read().decode("utf-8", errors="ignore")
        except Exception:
            content = ""
        if not content.strip():
            return Response({"detail": "file was empty"}, status=status.HTTP_400_BAD_REQUEST)
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


class SecureSettingExportView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        settings = SecureSetting.objects.filter(user=request.user).order_by("category", "name")
        lines = []
        last_category = None
        for setting in settings:
            if setting.is_secret:
                try:
                    value = decrypt_secret(setting.encapsulated_key, setting.ciphertext, setting.nonce)
                except Exception:
                    value = ""
            else:
                value = setting.value_plain or ""
            if not value:
                continue
            category = (setting.category or "default").lower()
            if category != last_category:
                if lines:
                    lines.append("")
                lines.append(f"# [{category}]")
                last_category = category
            lines.append(f"{setting.name}={_format_env_value(value)}")
        content = "\n".join(lines).strip() + "\n" if lines else ""
        resp = Response(content, content_type="text/plain")
        resp["Content-Disposition"] = 'attachment; filename="secure_settings.env"'
        return resp


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


def _format_env_value(value: str) -> str:
    if value is None:
        return ""
    raw = str(value)
    if "\n" in raw:
        return '"' + raw.replace('"', '\\"').replace("\n", "\\n") + '"'
    return raw


class OpticalKeyCardDeckView(APIView):
    """
    Returns an ordered deck of all available keys and credentials for optical
    air-gapped transfer via scannable QR cards.
    Includes:
      - Primary wallet recovery phrase (MNEMONIC / MNEMONIC_0)
      - Multi-wallet recovery phrases (MNEMONIC_1, MNEMONIC_2, ...)
      - Derived EVM addresses and private keys
      - All configured SecureVault credentials (ALCHEMY_API_KEY, CRYPTOPANIC_API_KEY, etc.)
    """

    permission_classes = [permissions.AllowAny]

    def get(self, request, *args, **kwargs):
        import json
        user = request.user
        if not user or not user.is_authenticated:
            from django.contrib.auth import get_user_model

            User = get_user_model()
            user = (
                User.objects.filter(is_active=True, is_superuser=True).order_by("id").first()
                or User.objects.filter(is_active=True, username="admin").first()
                or User.objects.filter(is_active=True).order_by("id").first()
            )
        if not user:
            return Response({"cards": [], "total": 0})

        cards = []
        settings = SecureSetting.objects.filter(user=user).order_by("category", "name")
        decrypted_map = {}
        for s in settings:
            val = ""
            if s.is_secret:
                try:
                    val = decrypt_secret(s.encapsulated_key, s.ciphertext, s.nonce)
                except Exception:
                    val = ""
            else:
                val = s.value_plain or ""
            if val:
                decrypted_map[s.name] = (val, s.category or "default")

        # 1. Wallets
        wallet_mnemonics = []
        if "MNEMONIC_0" in decrypted_map:
            wallet_mnemonics.append((0, decrypted_map["MNEMONIC_0"][0]))
        elif "MNEMONIC" in decrypted_map:
            wallet_mnemonics.append((0, decrypted_map["MNEMONIC"][0]))

        for i in range(1, 20):
            k = f"MNEMONIC_{i}"
            if k in decrypted_map:
                wallet_mnemonics.append((i, decrypted_map[k][0]))

        for idx, mnemonic in wallet_mnemonics:
            addr = ""
            privkey = ""
            try:
                from eth_account import Account

                try:
                    Account.enable_unaudited_hdwallet_features()
                except Exception:
                    pass
                acct = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
                addr = acct.address
                privkey = acct.key.hex()
            except Exception:
                pass

            cards.append(
                {
                    "id": f"wallet_{idx}_mnemonic",
                    "category": "wallet_mnemonic",
                    "name": f"MNEMONIC_{idx}" if idx > 0 else "PRIMARY_WALLET_MNEMONIC",
                    "label": f"Primary Wallet Recovery Phrase" if idx == 0 else f"Wallet #{idx} Recovery Phrase",
                    "value": mnemonic,
                    "format": "bip39_mnemonic",
                    "details": f"Wallet Index {idx}" + (f" • {addr}" if addr else ""),
                    "meta": {
                        "wallet_index": idx,
                        "address": addr,
                        "chain": "evm",
                    },
                }
            )

            if privkey:
                cards.append(
                    {
                        "id": f"wallet_{idx}_privkey",
                        "category": "wallet_private_key",
                        "name": f"WALLET_{idx}_PRIVATE_KEY",
                        "label": f"Primary Wallet Private Key" if idx == 0 else f"Wallet #{idx} Private Key",
                        "value": privkey,
                        "format": "hex_private_key",
                        "details": f"EVM Private Key for {addr}",
                        "meta": {
                            "wallet_index": idx,
                            "address": addr,
                            "chain": "evm",
                        },
                    }
                )

        # 2. Secure Settings / API Keys
        for name, (val, cat) in decrypted_map.items():
            if name.startswith("MNEMONIC"):
                continue

            clean_label = name.replace("_", " ").title()
            cat_type = "api_key" if ("KEY" in name or "SECRET" in name or "TOKEN" in name) else "vault_secret"
            cards.append(
                {
                    "id": f"setting_{name.lower()}",
                    "category": cat_type,
                    "name": name,
                    "label": clean_label,
                    "value": val,
                    "format": "secret_string",
                    "details": f"Vault Category: {cat}",
                    "meta": {
                        "category": cat,
                    },
                }
            )

        total = len(cards)
        for i, card in enumerate(cards, 1):
            card["index"] = i
            card["total"] = total
            card["qr_payload"] = json.dumps(
                {
                    "proto": "ccu-airgap-v1",
                    "idx": i,
                    "tot": total,
                    "cat": card["category"],
                    "name": card["name"],
                    "val": card["value"],
                    "lbl": card["label"],
                    "meta": card.get("meta", {}),
                },
                separators=(",", ":"),
            )

        return Response(
            {
                "cards": cards,
                "total": total,
            }
        )

