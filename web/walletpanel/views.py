from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List

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
from services.multi_wallet import multi_wallet_manager
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
        try:
            from services.internal_cron import cron_supervisor

            cron_supervisor.run_once("auto_pipeline")
            cron_supervisor.run_once("weekly_bootstrap")
        except Exception:
            pass
        try:
            if not wallet_runner.status().get("running"):
                wallet_runner.run("refresh_balances_full", user=request.user)
        except Exception:
            pass
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


class MultiWalletListView(APIView):
    """List all wallets with per-wallet state and aggregate summary."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        wallets = multi_wallet_manager.load_wallets(user=request.user)
        if not wallets:
            # Fall back to single-wallet mode.
            snapshot = load_wallet_state()
            return Response({
                "wallet_count": 1 if snapshot.get("wallet") else 0,
                "wallets": [{
                    "index": 0,
                    "wallet": snapshot.get("wallet"),
                    "usd": (snapshot.get("totals") or {}).get("usd", 0),
                    "updated_at": snapshot.get("updated_at"),
                }] if snapshot.get("wallet") else [],
                "totals": snapshot.get("totals", {"usd": 0}),
                "balances": snapshot.get("balances", []),
                "transfers": snapshot.get("transfers", {}),
                "nfts": snapshot.get("nfts", []),
                "config": _multi_wallet_config(),
            }, status=status.HTTP_200_OK)

        per_wallet: list[Dict[str, Any]] = []
        for w in wallets:
            try:
                state = load_wallet_state()
                if (state.get("wallet") or "").lower() == w.address.lower():
                    state["wallet_index"] = w.index
                    per_wallet.append(state)
                else:
                    per_wallet.append({
                        "wallet_index": w.index,
                        "wallet": w.address,
                        "updated_at": None,
                        "totals": {"usd": 0},
                        "balances": [],
                        "transfers": {},
                        "nfts": [],
                    })
            except Exception:
                per_wallet.append({
                    "wallet_index": w.index,
                    "wallet": w.address,
                    "updated_at": None,
                    "totals": {"usd": 0},
                    "balances": [],
                    "transfers": {},
                    "nfts": [],
                })

        aggregate = multi_wallet_manager.aggregate_state(per_wallet)
        aggregate["config"] = _multi_wallet_config()
        return Response(aggregate, status=status.HTTP_200_OK)


class MultiWalletCreateView(APIView):
    """Manually create a new wallet."""

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        try:
            wallet = multi_wallet_manager.create_wallet(user=request.user)
            return Response({
                "created": True,
                "index": wallet.index,
                "address": wallet.address,
            }, status=status.HTTP_201_CREATED)
        except Exception as exc:
            return Response(
                {"detail": f"Failed to create wallet: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class WalletRevealMnemonicView(APIView):
    """Reveal the full unencrypted mnemonic for a specific wallet index."""

    permission_classes = [IsAuthenticated]

    def post(self, request: Request, *args, **kwargs) -> Response:
        wallet_index = request.data.get("wallet_index", 0)
        try:
            wallet_index = int(wallet_index)
        except (TypeError, ValueError):
            wallet_index = 0

        from services.secure_settings import decrypt_secret

        # Try MNEMONIC_N first, then fall back to MNEMONIC for index 0.
        names = [f"MNEMONIC_{wallet_index}"]
        if wallet_index == 0:
            names.append("MNEMONIC")

        for name in names:
            setting = SecureSetting.objects.filter(
                user=request.user, name=name, category=DEFAULT_CATEGORY,
            ).first()
            if setting and setting.is_secret:
                try:
                    value = decrypt_secret(
                        setting.encapsulated_key,
                        setting.ciphertext,
                        setting.nonce,
                    )
                    return Response({
                        "wallet_index": wallet_index,
                        "mnemonic": value,
                    }, status=status.HTTP_200_OK)
                except Exception as exc:
                    return Response(
                        {"detail": f"Decryption failed: {exc}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )

        return Response(
            {"detail": f"No mnemonic found for wallet {wallet_index}"},
            status=status.HTTP_404_NOT_FOUND,
        )


class WalletTransfersView(APIView):
    """Paginated, searchable transfers across all chains."""

    permission_classes = [IsAuthenticated]

    def get(self, request: Request, *args, **kwargs) -> Response:
        snapshot = load_wallet_state()
        raw_transfers: Dict[str, list] = snapshot.get("transfers") or {}

        # Flatten all chains into a single list with chain name attached.
        flat: List[Dict[str, Any]] = []
        for chain, items in raw_transfers.items():
            if not isinstance(items, list):
                continue
            for item in items:
                entry = dict(item)
                entry["chain"] = chain
                flat.append(entry)

        # Sorting
        sort_order = str(request.query_params.get("sort", "desc")).lower()
        reverse = sort_order != "asc"
        flat.sort(key=lambda x: (x.get("block") or 0), reverse=reverse)

        # Search / filter
        search = str(request.query_params.get("search", "")).strip()
        if search:
            flat = _search_transfers(flat, search)

        total = len(flat)

        # Pagination
        try:
            offset = max(0, int(request.query_params.get("offset", 0)))
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = min(200, max(1, int(request.query_params.get("limit", 50))))
        except (TypeError, ValueError):
            limit = 50

        page = flat[offset:offset + limit]

        # Discover chains dynamically
        chains = sorted(raw_transfers.keys())

        return Response({
            "items": page,
            "total": total,
            "offset": offset,
            "limit": limit,
            "chains": chains,
        }, status=status.HTTP_200_OK)


def _search_transfers(items: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filter transfers by query.  Detects date-like queries and matches
    against ts/block fields; otherwise does a case-insensitive substring
    match across all string values."""

    # Try to detect a date query.
    parsed_date = _parse_date_query(query)
    if parsed_date:
        start_ts, end_ts = parsed_date
        results: List[Dict[str, Any]] = []
        for item in items:
            ts = item.get("ts")
            if ts:
                item_epoch = _snapshot_epoch(ts)
                if start_ts <= item_epoch <= end_ts:
                    results.append(item)
            # If no ts, can't date-filter — skip.
        return results

    # General text search — check all string values.
    q_lower = query.lower()
    return [
        item for item in items
        if any(q_lower in str(v).lower() for v in item.values())
    ]


def _parse_date_query(query: str):
    """Attempt to parse a date or date range from the query string.

    Returns (start_epoch, end_epoch) or None if not a date query.
    Handles formats like:
      - "2025-03-01"
      - "March 2025"
      - "03/01/2025"
      - "yesterday"
      - "last week"
    """
    q = query.strip().lower()

    # Relative dates
    from datetime import timedelta
    now = datetime.utcnow()
    if q in ("today", "now"):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.timestamp(), now.timestamp())
    if q == "yesterday":
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59)
        return (start.timestamp(), end.timestamp())
    if q in ("last week", "this week"):
        start = (now - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.timestamp(), now.timestamp())
    if q in ("last month", "this month"):
        start = (now - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
        return (start.timestamp(), now.timestamp())

    # Month + Year: "March 2025", "mar 2025"
    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4,
        "may": 5, "jun": 6, "june": 6,
        "jul": 7, "july": 7, "aug": 8, "august": 8,
        "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    m = re.match(r"^(\w+)\s+(\d{4})$", q)
    if m:
        month_str, year_str = m.group(1), m.group(2)
        month_num = month_names.get(month_str)
        if month_num:
            import calendar
            year = int(year_str)
            last_day = calendar.monthrange(year, month_num)[1]
            start = datetime(year, month_num, 1)
            end = datetime(year, month_num, last_day, 23, 59, 59)
            return (start.timestamp(), end.timestamp())

    # Standard date formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y", "%d-%m-%Y"):
        try:
            dt = datetime.strptime(q, fmt)
            start = dt.replace(hour=0, minute=0, second=0)
            end = dt.replace(hour=23, minute=59, second=59)
            return (start.timestamp(), end.timestamp())
        except ValueError:
            continue

    # Year only: "2025"
    if re.match(r"^\d{4}$", q):
        year = int(q)
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        return (start.timestamp(), end.timestamp())

    return None


def _multi_wallet_config() -> Dict[str, Any]:
    return {
        "enabled": multi_wallet_manager.enabled(),
        "threshold": multi_wallet_manager.threshold(),
        "max_balance": multi_wallet_manager.max_balance(),
    }
