"""Operator controls for reviewed .wbrain exchange and the selected cluster head."""
from __future__ import annotations

from contextlib import closing
import json
import os
import time
import uuid
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from django.http import JsonResponse
from django.views import View

from tools.wizard_learning.store import LearningStore

MAX_REPLY = 512 * 1024
TABLES = """
CREATE TABLE IF NOT EXISTS peer_experience_policy (
    brain TEXT PRIMARY KEY, auto_approve INTEGER NOT NULL, auto_deny INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS peer_experience_requests (
    id TEXT PRIMARY KEY, brain TEXT NOT NULL, peer TEXT NOT NULL, pool INTEGER NOT NULL,
    created REAL NOT NULL, state TEXT NOT NULL, evidence TEXT NOT NULL, detail TEXT NOT NULL,
    actor TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS peer_experience_by_brain ON peer_experience_requests(brain, created);
"""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_node_http = build_opener(ProxyHandler({}), NoRedirect())


def _origin(value):
    if not isinstance(value, str):
        raise ValueError("Expected a configured HTTP origin.")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password is not None or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("Expected a configured HTTP origin without a path or credentials.")
    return value


def _node_json(head, path, token, payload=None, timeout=4):
    headers = {"Accept": "application/json", "X-W1z4rD-Experience-Token": token}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, allow_nan=False).encode()
    try:
        with _node_http.open(Request(head + path, data=data, headers=headers), timeout=timeout) as response:
            raw = response.read(MAX_REPLY + 1)
        if len(raw) > MAX_REPLY:
            raise ValueError("Response exceeds the control-plane limit.")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("Unexpected node response.")
        return value
    except Exception as error:
        status = getattr(error, "code", None)
        suffix = f" (HTTP {status})" if status else ""
        raise RuntimeError(f"The configured cluster head could not verify {path}{suffix}.") from error


def _inspect(head, peers, token, peer, pool, expected=None):
    payload = {"peer_url": peers[peer]}
    if expected is not None:
        payload["expected"] = expected
    result = _node_json(head, f"/cluster/experience/{pool}", token, payload, timeout=55)
    metrics = result.get("evidence")
    if (not isinstance(metrics, dict) or metrics.get("decision") not in {"compatible", "duplicate", "anomaly"}
            or not isinstance(metrics.get("reason"), str)):
        raise RuntimeError("The node did not return a recognized exchange review.")
    return metrics, result.get("inserted")


def _decision(evidence, auto_approve, auto_deny):
    outcome = evidence.get("decision")
    if outcome not in {"compatible", "duplicate"}:
        return "pending"
    if outcome == "compatible" and auto_approve:
        return "applying"
    return "denied" if outcome == "duplicate" and auto_deny else "pending"


def _policy(db, head):
    row = db.execute("SELECT auto_approve, auto_deny FROM peer_experience_policy WHERE brain=?", (head,)).fetchone()
    return dict(zip(("auto_approve", "auto_deny"), map(bool, row or (0, 0))))


def _public(row):
    return {**dict(row), "evidence": json.loads(row["evidence"])}


def _apply(db, head, peers, token, identifier, peer, pool, evidence):
    try:
        _, inserted = _inspect(head, peers, token, peer, pool, evidence)
        if inserted != evidence.get("proposed_neurons") or not isinstance(inserted, int) or inserted <= 0:
            raise RuntimeError("The node did not confirm the expected durable append.")
        state, detail = "applied", f"Node confirmed a durable append of {inserted} neurons."
    except RuntimeError as error:
        state, detail = "review", f"{error} Check the node before retrying."
    with db:
        db.execute("UPDATE peer_experience_requests SET state=?, detail=? WHERE id=? AND state='applying'",
                   (state, detail, identifier))


class WizardClusterView(View):
    def dispatch(self, request, *args, **kwargs):
        if not getattr(request.user, "is_authenticated", False) or not getattr(request.user, "is_staff", False):
            return JsonResponse({"error": "Wizard cluster controls require an operator account."}, status=403)
        from .views import _brain_for_request
        try:
            head = _origin(os.getenv("WIZARD_CLUSTER_ENDPOINT", ""))
            profile = _brain_for_request(request)
            if _origin(profile["endpoint"]) != head:
                raise ValueError("Select the configured cluster brain in the Wizard brain picker.")
            peers = json.loads(os.getenv("WIZARD_EXPERIENCE_PEERS", "{}"))
            if not isinstance(peers, dict) or len(peers) > 50:
                raise ValueError("The peer registry is invalid.")
            peers = {name: _origin(origin) for name, origin in peers.items()
                     if isinstance(name, str) and 1 <= len(name) <= 80}
            shared = os.getenv("WIZARD_EXPERIENCE_TOKEN", "")
            if len(shared) < 32:
                raise ValueError("Configure an experience exchange token.")
        except (KeyError, TypeError, ValueError) as error:
            return JsonResponse({"enabled": False, "error": str(error)}, status=503)
        self.head, self.peers, self.shared = head, peers, shared
        with closing(LearningStore().connect()) as db:
            db.executescript(TABLES)
            self.db = db
            return super().dispatch(request, *args, **kwargs)

    def get(self, request):
        rows = self.db.execute(
            "SELECT * FROM peer_experience_requests WHERE brain=? ORDER BY created DESC LIMIT 50",
            (self.head,),
        ).fetchall()
        telemetry = {}
        for label, path in (("membership", "/cluster/members"), ("storage", "/storage_state")):
            try:
                telemetry[label] = _node_json(self.head, path, self.shared)
            except RuntimeError as error:
                telemetry[f"{label}_error"] = str(error)
        return JsonResponse({
            "enabled": True, "peers": sorted(self.peers), "policy": _policy(self.db, self.head),
            "requests": [_public(row) for row in rows], **telemetry,
        })

    def post(self, request):
        try:
            if len(request.body) > 8192:
                raise ValueError("Control request is too large.")
            body = json.loads(request.body)
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            action = body.get("action")
            if action == "policy":
                for field in ("auto_approve", "auto_deny"):
                    if type(body.get(field)) is not bool:
                        raise ValueError(f"{field} must be a boolean.")
                with self.db:
                    self.db.execute("INSERT OR REPLACE INTO peer_experience_policy VALUES (?,?,?)",
                                    (self.head, body["auto_approve"], body["auto_deny"]))
                return JsonResponse({"policy": _policy(self.db, self.head)})
            if action == "request":
                return self._request(request, body)
            if action in {"approve", "deny"}:
                return self._review(request, body)
            raise ValueError("Unknown cluster control action.")
        except (ValueError, TypeError) as error:
            return JsonResponse({"error": str(error)}, status=400)

    def _request(self, request, body):
        peer, pool = body.get("peer"), body.get("pool")
        if not isinstance(peer, str) or peer not in self.peers:
            raise ValueError("Choose a configured peer.")
        if type(pool) is not int or not 0 <= pool < 2**32:
            raise ValueError("Choose a valid numeric pool ID.")
        try:
            evidence, _ = _inspect(self.head, self.peers, self.shared, peer, pool)
        except RuntimeError as error:
            evidence = {"decision": "anomaly", "reason": str(error)}
        policy = _policy(self.db, self.head)
        state = _decision(evidence, **policy)
        identifier = uuid.uuid4().hex
        detail = evidence["reason"] if state != "denied" else f"Auto-deny policy. {evidence['reason']}"
        with self.db:
            self.db.execute("INSERT INTO peer_experience_requests VALUES (?,?,?,?,?,?,?,?,?)",
                            (identifier, self.head, peer, pool, time.time(), state,
                             json.dumps(evidence), detail, str(request.user.pk)))
        if state == "applying":
            _apply(self.db, self.head, self.peers, self.shared, identifier, peer, pool, evidence)
        return self._item(identifier, status=201)

    def _item(self, identifier, status=200):
        row = self.db.execute("SELECT * FROM peer_experience_requests WHERE id=? AND brain=?",
                              (identifier, self.head)).fetchone()
        return JsonResponse(_public(row), status=status)

    def _review(self, request, body):
        identifier = body.get("id")
        if not isinstance(identifier, str):
            raise ValueError("A request ID is required.")
        row = self.db.execute("SELECT * FROM peer_experience_requests WHERE id=? AND brain=?",
                              (identifier, self.head)).fetchone()
        if not row:
            return JsonResponse({"error": "No such exchange request."}, status=404)
        approve = body["action"] == "approve"
        evidence = json.loads(row["evidence"])
        if approve and (row["peer"] not in self.peers or evidence.get("decision") != "compatible"):
            return JsonResponse({"error": "A measured compatible page is required. Investigate or deny this request."},
                                status=409)
        with self.db:
            transitioned = self.db.execute(
                "UPDATE peer_experience_requests SET state=?, actor=?, detail=? "
                "WHERE id=? AND brain=? AND state='pending'",
                ("applying" if approve else "denied", str(request.user.pk),
                 "Approved by operator." if approve else "Denied by operator.", identifier, self.head),
            ).rowcount
        if not transitioned:
            return JsonResponse({"error": "This request has already been decided or needs investigation."}, status=409)
        if approve:
            _apply(self.db, self.head, self.peers, self.shared, identifier, row["peer"], row["pool"], evidence)
        return self._item(identifier)

