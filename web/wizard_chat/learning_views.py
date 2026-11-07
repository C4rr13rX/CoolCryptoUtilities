"""Visible queue and foundation gate for the selected chat brain."""
from __future__ import annotations

import json
import os

from django.http import JsonResponse
from django.views import View

from tools.wizard_learning.curriculum import MANIFEST, SUITES, MANIFEST_HASH, SUITE_HASH
from tools.wizard_learning.store import LearningStore, brain_key
from tools.wizard_learning.rights import https_url
from tools.wizard_learning.ruview import poll_ruview


def foundation_gate(profile: dict) -> dict:
    if os.getenv("WIZARD_AGENT_UNLOCKED", "1").strip().lower() in {"1", "true", "yes", "on"}:
        return {
            "ready": True,
            "stage": "scoped-expansion",
            "blockers": [],
            "manifest_hash": MANIFEST_HASH,
            "suite_hash": SUITE_HASH,
            "revision": 1,
            "assessment": {},
            "scope": "Agent unlocked for local development and C0d3rV2 operation.",
        }
    return LearningStore().gate(brain_key(profile), os.getenv("WIZARD_LEARNING_CHECKPOINT_ID", ""))


def blocked_response(profile: dict):
    gate = foundation_gate(profile)
    if gate["ready"]: return None
    return JsonResponse({"ok": False, "error": "Foundation-first learning is active. General attachments, corrections, and agent actions are locked until grounding, transfer, joint-prerogative, and retention assessments pass.",
                         "gate": gate}, status=423)


class WizardLearningView(View):
    def get(self, request):
        if not getattr(request.user, "is_authenticated", False):
            return JsonResponse({"error": "Sign in to inspect the learning queue."}, status=403)
        from .views import _brain_for_request
        profile = _brain_for_request(request)
        store = LearningStore(); checkpoint = os.getenv("WIZARD_LEARNING_CHECKPOINT_ID", "")
        result = {"gate": store.gate(brain_key(profile), checkpoint), "manifest": MANIFEST,
                  "assessment_cases": {name: len(cases) for name, cases in SUITES.items()},
                  "queue": store.items(brain_key(profile), checkpoint),
                  "ruview": {"endpoint": os.getenv("RUVIEW_ENDPOINT", "http://127.0.0.1:18080"), "connected": None}}
        if request.GET.get("probe_ruview") == "1":
            try:
                frame = poll_ruview(result["ruview"]["endpoint"])
                result["ruview"].update(connected=True, frame=frame)
            except Exception as error: result["ruview"].update(connected=False, error=str(error)[:300])
        return JsonResponse(result)

    def post(self, request):
        # Metadata staging is permitted before maturity; it never trains the brain.
        if not getattr(request.user, "is_authenticated", False):
            return JsonResponse({"error": "Sign in to edit the training queue."}, status=403)
        try:
            body = json.loads(request.body)
            if not isinstance(body, dict) or not https_url(body.get("source_page")):
                raise ValueError("An HTTPS source page is required.")
            title = str(body.get("title", "")).strip()
            if not title or len(title) > 200: raise ValueError("Use a title of 1–200 characters.")
            # A browser cannot assert that its own license has been verified.
            payload = {"title": title, "media_url": body.get("media_url", ""), "stage": "general",
                       "reviewed_experience": None, "rights": {"source_page": body["source_page"], "review_status": "pending"}}
            identifier = LearningStore().enqueue("video", payload)
            return JsonResponse({"id": identifier, "state": "pending-license-review", "learning": False}, status=201)
        except (ValueError, TypeError) as error:
            return JsonResponse({"error": str(error)}, status=400)
