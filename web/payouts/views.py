"""Read and manage payout destinations from the dashboard.

Sending is not exposed here. These endpoints create, edit and inspect
standing instructions and show what each one is currently owed; moving funds
is a separate, explicitly enabled path so that a misrouted request to a
read-only surface can never transfer money.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .engine import plan_payout, realised_profit_since, wallet_usd
from .models import Payout, PayoutDestination


def _destination_json(destination) -> dict:
    return {
        "id": destination.pk,
        "name": destination.name,
        "kind": destination.kind,
        "kind_display": destination.get_kind_display(),
        "address": destination.address,
        "chain": destination.chain,
        "asset_symbol": destination.asset_symbol,
        "share": str(destination.share),
        "share_pct": f"{destination.share_pct:.2f}",
        "min_net_profit_usd": str(destination.min_net_profit_usd),
        "min_transfer_usd": str(destination.min_transfer_usd),
        "reserve_usd": str(destination.reserve_usd),
        "enabled": destination.enabled,
        "auto_send": destination.auto_send,
    }


def _payout_json(payout) -> dict:
    return {
        "id": payout.pk,
        "destination": payout.destination.name,
        "amount_usd": str(payout.amount_usd),
        "qualifying_profit_usd": str(payout.qualifying_profit_usd),
        "trades_counted": payout.trades_counted,
        "status": payout.status,
        "reason": payout.reason,
        "tx_hash": payout.tx_hash,
        "created_at": payout.created_at.isoformat(),
        "sent_at": payout.sent_at.isoformat() if payout.sent_at else None,
    }


@require_http_methods(["GET"])
def status(request):
    """Everything the payouts panel needs in one call."""
    net, trades, _ = realised_profit_since(0)
    balance = wallet_usd()

    destinations = []
    for destination in PayoutDestination.objects.all():
        payload = _destination_json(destination)
        decision = plan_payout(destination)
        payload["pending_decision"] = {
            "eligible": decision["eligible"],
            "amount_usd": str(decision["amount_usd"]),
            "net_profit_usd": str(decision["net_profit_usd"]),
            "trades_counted": decision["trades_counted"],
            "reason": decision["reason"],
        }
        destinations.append(payload)

    return JsonResponse({
        "lifetime_realised_usd": str(net),
        "lifetime_closed_trades": trades,
        # Explicitly null rather than 0 when unreadable: the UI must be able
        # to say "unknown" instead of showing an empty wallet that isn't.
        "spendable_stable_usd": str(balance) if balance is not None else None,
        "destinations": destinations,
        "recent_payouts": [_payout_json(p) for p in Payout.objects.all()[:25]],
    })


@csrf_exempt
@require_http_methods(["POST"])
def save_destination(request):
    """Create or update a destination."""
    import json

    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001
        return JsonResponse({"error": "invalid JSON"}, status=400)

    address = str(body.get("address") or "").strip()
    name = str(body.get("name") or "").strip()
    if not name:
        return JsonResponse({"error": "name is required"}, status=400)

    # An address typo sends money to nobody, permanently. Shape is the only
    # check available before a send, so it is enforced here rather than left
    # to the moment funds are already moving.
    if not (address.startswith("0x") and len(address) == 42):
        return JsonResponse(
            {"error": "address must be a 0x-prefixed 42-character EVM address"},
            status=400)

    fields = {
        "name": name,
        "address": address,
        "kind": str(body.get("kind") or "wallet"),
        "chain": str(body.get("chain") or "base"),
        "asset_symbol": str(body.get("asset_symbol") or "USDC"),
        "enabled": bool(body.get("enabled", True)),
        "auto_send": bool(body.get("auto_send", False)),
    }

    for key, cap in (("share", Decimal("1")), ("min_net_profit_usd", None),
                     ("min_transfer_usd", None), ("reserve_usd", None)):
        if body.get(key) is None:
            continue
        try:
            value = Decimal(str(body[key]))
        except (InvalidOperation, ValueError):
            return JsonResponse({"error": f"{key} must be a number"}, status=400)
        if value < 0:
            return JsonResponse({"error": f"{key} cannot be negative"}, status=400)
        # A share above 1.0 would pay out more than was earned.
        if cap is not None and value > cap:
            return JsonResponse(
                {"error": "share cannot exceed 1.0 (100%)"}, status=400)
        fields[key] = value

    destination_id = body.get("id")
    if destination_id:
        updated = PayoutDestination.objects.filter(pk=destination_id).update(**fields)
        if not updated:
            return JsonResponse({"error": "destination not found"}, status=404)
        destination = PayoutDestination.objects.get(pk=destination_id)
    else:
        destination = PayoutDestination.objects.create(**fields)

    return JsonResponse(_destination_json(destination))


@csrf_exempt
@require_http_methods(["POST"])
def delete_destination(request, destination_id: int):
    """Remove a destination. Its payout history is kept.

    Payout.destination is PROTECT, so a destination with history cannot be
    deleted -- the record of where money went outlives the routing rule. Such
    a destination is disabled instead, which stops it paying without erasing
    what it already paid.
    """
    try:
        destination = PayoutDestination.objects.get(pk=destination_id)
    except PayoutDestination.DoesNotExist:
        return JsonResponse({"error": "destination not found"}, status=404)

    if destination.payouts.exists():
        destination.enabled = False
        destination.auto_send = False
        destination.save(update_fields=["enabled", "auto_send"])
        return JsonResponse({
            "status": "disabled",
            "detail": "destination has payout history and was disabled rather "
                      "than deleted, so the record of where funds went is kept",
        })

    destination.delete()
    return JsonResponse({"status": "deleted"})
