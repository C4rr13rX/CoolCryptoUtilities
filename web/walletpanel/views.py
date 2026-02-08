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
from .models import WalletNftPreference

DEFAULT_CATEGORY = "default"

def _snapshot_epoch(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        from datetime import datetime

        return datetime.fromisoformat(text).timestamp()
    except Exception:
        try:
            return float(text)
        except Exception:
            return 0.0


class WalletActionsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        status_payload = wallet_runner.status()
        try:
            snapshot = load_wallet_state()
            updated_at = _snapshot_epoch(snapshot.get("updated_at") if isinstance(snapshot, dict) else None)
            finished_at = float(status_payload.get("finished_at") or 0.0)
            message = str(status_payload.get("message") or "")
            if message.lower().startswith("failed") and updated_at and finished_at:
                if updated_at > (finished_at + 5.0):
                    status_payload["message"] = "completed"
                    status_payload["returncode"] = 0
        except Exception:
            pass
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


def _normalize_nft_payload(item: Dict[str, Any]) -> Dict[str, str]:
    chain = str(item.get("chain") or "").strip().lower()
    contract = str(item.get("contract") or "").strip().lower()
    token_id = str(item.get("token_id") or "").strip()
    return {"chain": chain, "contract": contract, "token_id": token_id}


class WalletNftPreferenceView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        items = list(
            WalletNftPreference.objects.filter(user=request.user, hidden=True).values(
                "chain",
                "contract",
                "token_id",
                "hidden",
            )
        )
        return Response({"items": items, "count": len(items)}, status=status.HTTP_200_OK)

    def post(self, request: Request, *args, **kwargs) -> Response:
        payload = request.data or {}
        raw_items = payload.get("items") or []
        action = str(payload.get("action") or "").strip().lower()
        hidden_flag = payload.get("hidden")
        if action not in {"hide", "show"}:
            if isinstance(hidden_flag, bool):
                action = "hide" if hidden_flag else "show"
        if action not in {"hide", "show"}:
            return Response({"detail": "action must be 'hide' or 'show'"}, status=status.HTTP_400_BAD_REQUEST)
        if not isinstance(raw_items, list) or not raw_items:
            return Response({"detail": "items must be a non-empty list"}, status=status.HTTP_400_BAD_REQUEST)

        normalized: list[Dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            cleaned = _normalize_nft_payload(item)
            if not (cleaned["chain"] and cleaned["contract"] and cleaned["token_id"]):
                continue
            normalized.append(cleaned)
        if not normalized:
            return Response({"detail": "no valid items provided"}, status=status.HTTP_400_BAD_REQUEST)

        if action == "hide":
            for entry in normalized:
                WalletNftPreference.objects.update_or_create(
                    user=request.user,
                    chain=entry["chain"],
                    contract=entry["contract"],
                    token_id=entry["token_id"],
                    defaults={"hidden": True},
                )
        else:
            for entry in normalized:
                WalletNftPreference.objects.filter(
                    user=request.user,
                    chain=entry["chain"],
                    contract=entry["contract"],
                    token_id=entry["token_id"],
                ).delete()

        items = list(
            WalletNftPreference.objects.filter(user=request.user, hidden=True).values(
                "chain",
                "contract",
                "token_id",
                "hidden",
            )
        )
        return Response({"items": items, "count": len(items)}, status=status.HTTP_200_OK)


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
