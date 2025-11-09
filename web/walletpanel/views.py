from __future__ import annotations

from typing import Any, Dict

from django.db import transaction
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from securevault.models import SecureSetting
from services.secure_settings import encrypt_secret, mask_value
from services.wallet_runner import wallet_runner
from services.wallet_state import load_wallet_state

DEFAULT_CATEGORY = "default"


class WalletActionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        status_payload = wallet_runner.status()
        return Response(status_payload, status=status.HTTP_200_OK)


class WalletRunView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        action = str(payload.get("action") or "").lower()
        options = payload.get("options") or {}
        if not action:
            return Response({"detail": "action is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            wallet_runner.run(action, options, user=request.user)
        except RuntimeError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(wallet_runner.status(), status=status.HTTP_202_ACCEPTED)


class WalletMnemonicView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        setting = SecureSetting.objects.filter(user=request.user, name="MNEMONIC", category=DEFAULT_CATEGORY).first()
        if not setting:
            return Response({"value": None, "preview": ""}, status=status.HTTP_200_OK)
        preview = mask_value("secret" if setting.is_secret else setting.value_plain)
        return Response({"value": None, "preview": preview}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        mnemonic = (request.data or {}).get("mnemonic")
        if mnemonic is None or not str(mnemonic).strip():
            SecureSetting.objects.filter(user=request.user, name="MNEMONIC", category=DEFAULT_CATEGORY).delete()
            return Response({"changed": True, "preview": ""}, status=status.HTTP_200_OK)
        value = str(mnemonic).strip()
        _upsert_secret(request.user, "MNEMONIC", value)
        return Response({"changed": True, "preview": mask_value("secret")}, status=status.HTTP_200_OK)


class WalletStateSnapshotView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        snapshot = load_wallet_state()
        return Response(snapshot, status=status.HTTP_200_OK)


def _upsert_secret(user, name: str, value: str) -> None:
    with transaction.atomic():
        setting, _ = SecureSetting.objects.get_or_create(
            user=user,
            name=name,
            category=DEFAULT_CATEGORY,
            defaults={"is_secret": True},
        )
        setting.is_secret = True
        payload = encrypt_secret(value)
        setting.value_plain = None
        setting.ciphertext = payload["ciphertext"]
        setting.encapsulated_key = payload["encapsulated_key"]
        setting.nonce = payload["nonce"]
        setting.save()
