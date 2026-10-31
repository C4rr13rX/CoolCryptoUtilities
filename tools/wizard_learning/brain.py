"""Bounded HTTP bridge. Learning is separate from prediction and assessment."""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
import uuid

from .curriculum import IDS, TEACHING_LESSONS, SUITES, prompt_for, score
from .store import LearningStore, brain_key

class BrainClient:
    def __init__(self, profile: dict, timeout: float = 8):
        self.profile, self.timeout = profile, timeout

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.profile["endpoint"].rstrip("/") + path, data=json.dumps(body, allow_nan=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000: raise ValueError("Brain response exceeds 2 MB")
        result = json.loads(raw)
        if not isinstance(result, dict): raise ValueError("Brain returned a non-object response")
        return result

    def ask(self, text: str) -> dict:
        return self.post(self.profile.get("chat_path", "/brain/chat"), {"text": text, "session_id": "foundation-" + uuid.uuid4().hex, "learning": False})

    def predict(self, pool: int, frame: dict, target_pool: int) -> dict:
        return self.post("/brain/predict", {"query_pool": pool, "target_pool": target_pool, "frame": encode(frame)})

def encode(value) -> str:
    data = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def assess_foundation(client: BrainClient, store: LearningStore, checkpoint: str, *, case_limit: int | None = None) -> dict:
    if not checkpoint: raise ValueError("An immutable checkpoint ID is required.")
    key = brain_key(client.profile); revision = store.revision(key)
    report = {}
    for name, cases in SUITES.items():
        results = []
        for case in cases[:case_limit] if case_limit else cases:
            try:
                reply = client.ask(prompt_for(case))
                result = score(case, reply.get("answer") or reply.get("reply") or "")
                result["response_sha256"] = hashlib.sha256(json.dumps(reply, sort_keys=True).encode()).hexdigest()
            except Exception as error:
                result = {"case_id": case["id"], "passed": False, "error": str(error)[:300]}
            results.append(result)
        report[name] = results
    if case_limit is None:
        store.record_assessment(key, checkpoint, revision, report)
    return {"diagnostic_only": case_limit is not None, "results": report,
            "gate": store.gate(key, checkpoint)}

def teach_foundation(client: BrainClient, store: LearningStore, *, input_pool: int, outcome_pool: int) -> dict:
    """Only trusted authored foundation lessons, never held-out assessment cases."""
    if input_pool == outcome_pool: raise ValueError("Foundation input and outcome pools must be separate.")
    outcomes = []
    for case in TEACHING_LESSONS:
        # Invalidate before POST: a transport failure can occur after consolidation.
        store.invalidate(brain_key(client.profile), "foundation lesson submitted")
        result = client.post("/brain/consolidate", {"input_pool": input_pool, "input_frame": encode(prompt_for(case)),
                            "outcome_pool": outcome_pool, "outcome_frame": encode(case["expected"])})
        outcomes.append({"case_id": case["id"], "consolidated": result.get("consolidated") is True})
        if result.get("consolidated") is not True: break
    return {"stage": "foundation", "lessons": outcomes, "assessment_required": True}

def consolidate_reviewed(client: BrainClient, store: LearningStore, checkpoint: str,
                         experience: dict, *, input_pool: int, outcome_pool: int) -> dict:
    """One bounded, reviewed expansion step, followed by retention invalidation."""
    key = brain_key(client.profile)
    gate = store.gate(key, checkpoint)
    if not gate["ready"]: raise PermissionError("Foundation gate is closed: " + "; ".join(gate["blockers"]))
    review = experience.get("reviewed_experience", {})
    if not isinstance(review, dict) or review.get("outcome_verified") is not True or not review.get("reviewer"):
        raise ValueError("An externally reviewed outcome is required.")
    if set(review.get("checked_prerogatives", [])) != set(IDS) or review.get("conflicts") != []:
        raise ValueError("All twelve prerogatives must be checked together, with no unresolved conflict.")
    if not review.get("observed_outcome") or not experience.get("input"):
        raise ValueError("The experience requires input and an observed outcome.")
    if input_pool == outcome_pool: raise ValueError("Use separate input/outcome pools.")
    store.admit_expansion(key, checkpoint, "reviewed experience submitted; retention reassessment required")
    return client.post("/brain/consolidate", {"input_pool": input_pool, "input_frame": encode(experience["input"]),
                      "outcome_pool": outcome_pool, "outcome_frame": encode(review["observed_outcome"])})
