"""web/wizard_chat/wbrain_views.py — Decentralized `.wbrain` v4 P2P Mesh & Strict Testnet Endpoints."""
from __future__ import annotations

import json
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from tools.wizard_learning.wbrain_decentralized_mesh import (
    MainnetBlockedError,
    WbrainDecentralizedMeshManager,
)


@method_decorator(csrf_exempt, name="dispatch")
class WizardWbrainManifestsView(View):
    """GET /api/wizard-chat/wbrain/manifests/ — list `.wbrain` v4 shards & Testnet ledger status."""

    def get(self, request):
        mgr = WbrainDecentralizedMeshManager.get_instance()
        return JsonResponse(mgr.list_manifests_and_testnet_status())


@method_decorator(csrf_exempt, name="dispatch")
class WizardWbrainExportView(View):
    """POST /api/wizard-chat/wbrain/export/ — package open-licensed knowledge into `.wbrain` v4 + P2P chunks."""

    def post(self, request):
        try:
            body = json.loads(request.body or b"{}")
        except Exception:
            return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

        domain = str(body.get("domain") or "pc_aplus_and_desktop")
        max_records = max(1, min(500, int(body.get("max_records") or 100)))
        custom_records = body.get("records") if isinstance(body.get("records"), list) else None
        mgr = WbrainDecentralizedMeshManager.get_instance()
        res = mgr.export_corpus_shard(domain=domain, max_records=max_records, custom_records=custom_records)
        return JsonResponse(res)


@method_decorator(csrf_exempt, name="dispatch")
class WizardWbrainImportView(View):
    """POST /api/wizard-chat/wbrain/import/ — verify & ingest peer `.wbrain` v4 shard + issue ReplicationReceipt."""

    def post(self, request):
        try:
            body = json.loads(request.body or b"{}")
        except Exception:
            return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

        mgr = WbrainDecentralizedMeshManager.get_instance()
        try:
            res = mgr.import_peer_shard(
                peer_node_id=str(body.get("peer_node_id") or "rapid-diplomat-mobile"),
                chain=str(body.get("chain") or "BASE_SEPOLIA"),
                wbrain_b64=body.get("wbrain_b64"),
                data_manifest=body.get("data_manifest"),
                chunks=body.get("chunks"),
            )
            return JsonResponse(res)
        except MainnetBlockedError as me:
            return JsonResponse({"ok": False, "error": str(me), "network_mode": "TESTNET_ONLY"}, status=403)
        except Exception as exc:
            return JsonResponse({"ok": False, "error": str(exc)}, status=400)


@method_decorator(csrf_exempt, name="dispatch")
class WizardWbrainTestnetView(View):
    """POST /api/wizard-chat/wbrain/testnet/ — Testnet `tW1Z` faucet/escrow (Mainnet strictly HTTP 403 blocked)."""

    def post(self, request):
        try:
            body = json.loads(request.body or b"{}")
        except Exception:
            return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)

        mgr = WbrainDecentralizedMeshManager.get_instance()
        try:
            res = mgr.execute_testnet_action(
                action=str(body.get("action") or "status"),
                account_id=str(body.get("account_id") or "rapid-diplomat-mobile"),
                chain=str(body.get("chain") or "BASE_SEPOLIA"),
                amount_tw1z=float(body.get("amount_tw1z") or 50.0),
                purpose=str(body.get("purpose") or "Testnet operation"),
            )
            return JsonResponse(res)
        except MainnetBlockedError as me:
            return JsonResponse({"ok": False, "error": str(me), "network_mode": "TESTNET_ONLY"}, status=403)
        except Exception as exc:
            return JsonResponse({"ok": False, "error": str(exc)}, status=400)
