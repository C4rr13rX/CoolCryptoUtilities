from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tools.wizard_learning.brain import assess_foundation, teach_foundation, consolidate_reviewed
from tools.wizard_learning.curriculum import IDS, LESSONS, TEACHING_LESSONS, SUITES, prompt_for, score
from tools.wizard_learning.rights import license_blockers
from tools.wizard_learning.ruview import normalize_ruview, ingest_ruview
from tools.wizard_learning.store import LearningStore, brain_key

PROFILE = {"endpoint": "http://127.0.0.1:8090", "chat_path": "/brain/chat", "id": "test"}
KEY = brain_key(PROFILE)

def passed_report():
    return {name: [{"case_id": case["id"], "passed": True} for case in cases] for name, cases in SUITES.items()}

def reviewed_video():
    return {"title": "Real river camera", "media_url": "https://example.org/river.mp4", "content_review": {
        "capture_type": "real-camera", "animation_present": False, "staged_fiction": False, "review_status": "verified", "reviewer": "operator"},
        "rights": {"license_id": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                   "source_page": "https://example.org/river", "evidence_url": "https://example.org/rights", "creator": "Creator",
                   "attribution": "Creator, CC BY 4.0", "review_status": "source_verified", "reviewer": "operator",
                   "evidence_sha256": "a" * 64, "reviewed_at": datetime.now(timezone.utc).isoformat(), "third_party_restrictions": "reviewed"}}

def test_rights_require_specific_evidence_and_real_footage():
    item = reviewed_video()
    assert license_blockers(item) == []
    for license_id in ("CC-BY-NC-4.0", "CC-BY-ND-4.0", "unknown", "CC-BY-SA-4.0"):
        invalid = copy.deepcopy(item); invalid["rights"]["license_id"] = license_id
        assert license_blockers(invalid)
    for change in ({"animation_present": True}, {"staged_fiction": True}, {"capture_type": "synthetic"}, {"review_status": "pending"}):
        invalid = copy.deepcopy(item); invalid["content_review"].update(change)
        assert license_blockers(invalid)
    item["rights"]["evidence_sha256"] = ""
    assert license_blockers(item)

def test_all_prerogatives_and_consequences_must_match_not_just_a_safe_sounding_reply():
    case = SUITES["transfer"][0]
    assert score(case, case["expected"])["passed"]
    wrong = copy.deepcopy(case["expected"]); wrong["checked_prerogatives"] = list(IDS[:-1])
    assert not score(case, wrong)["passed"]
    wrong = copy.deepcopy(case["expected"]); wrong["consequence_ids"] = ["no_consequence"]
    assert not score(case, wrong)["passed"]
    assert not score(case, "I will follow all the rules")["passed"]
    assert {case["expected"]["choice_id"] for case in SUITES["transfer"]} == {"proposal", "alternative"}
    assert not ({prompt_for(case) for case in TEACHING_LESSONS} & {prompt_for(case) for cases in SUITES.values() for case in cases})

def test_gate_is_checkpoint_revision_and_complete_suite_scoped(tmp_path):
    store = LearningStore(tmp_path / "learning.db")
    assert not store.gate(KEY, "checkpoint-a")["ready"]
    report = passed_report(); report["joint"].pop()
    store.record_assessment(KEY, "checkpoint-a", 0, report)
    assert not store.gate(KEY, "checkpoint-a")["ready"]
    store.record_assessment(KEY, "checkpoint-a", 0, passed_report())
    assert store.gate(KEY, "checkpoint-a")["ready"]
    assert not store.gate(KEY, "checkpoint-b")["ready"]
    assert not store.gate(brain_key({**PROFILE, "id": "other"}), "checkpoint-a")["ready"]
    store.invalidate(KEY, "new learning")
    assert not store.gate(KEY, "checkpoint-a")["ready"]

def test_queue_deduplicates_metadata_and_excludes_cartoons(tmp_path):
    store = LearningStore(tmp_path / "learning.db")
    item = reviewed_video(); one = store.enqueue("video", item)
    item["rights"]["reviewed_at"] = "2026-10-24T00:00:00+00:00"
    assert store.enqueue("video", item) == one
    store.enqueue("video", {**item, "title": "Big Buck Bunny"})
    assert store.exclude_nonreal() == 1
    rows = store.items(KEY, "checkpoint-a", include_excluded=True)
    assert not any(row["eligible"] for row in rows)
    assert next(row for row in rows if row["payload"]["title"] == "Big Buck Bunny")["state"] == "excluded"
    assert all(row["payload"]["title"] != "Big Buck Bunny" for row in store.items(KEY, "checkpoint-a"))

def ruview_frame():
    return {"type": "sensing_update", "timestamp": time.time(), "source": "wifi:private-network", "tick": 3,
            "features": {"mean_rssi": -45, "variance": 1, "motion_band_power": .3, "breathing_band_power": .1,
                         "dominant_freq_hz": 1, "change_points": 2, "spectral_power": 3},
            "classification": {"presence": True, "confidence": .8}, "vital_signs": {"heart_rate": 70}}

def test_ruview_rejects_fake_stale_offline_and_nonfinite_data_and_removes_identifiers():
    raw = ruview_frame(); normalized = normalize_ruview(raw)
    assert "private-network" not in json.dumps(normalized)
    assert "heart_rate" not in json.dumps(normalized)
    assert normalized["provenance"]["ground_truth"] is False
    for source in ("simulated", "esp32:offline", "unknown"):
        with pytest.raises(ValueError): normalize_ruview({**raw, "source": source})
    with pytest.raises(ValueError): normalize_ruview({**raw, "timestamp": time.time() - 20})
    raw["features"]["variance"] = float("nan")
    with pytest.raises(ValueError): normalize_ruview(raw)

class Model:
    profile = PROFILE
    def __init__(self): self.posts = []; self.predictions = []
    def ask(self, prompt):
        identifier = json.loads(prompt)["experience"]["case_id"]
        case = next(case for cases in SUITES.values() for case in cases if case["id"] == identifier)
        return {"answer": json.dumps(case["expected"])}
    def post(self, path, body): self.posts.append((path, body)); return {"consolidated": True}
    def predict(self, pool, frame, target): self.predictions.append((pool, frame, target)); return {"learning": False}

def test_assessment_is_read_only_and_teaching_never_uses_heldout_examples(tmp_path):
    store = LearningStore(tmp_path / "learning.db"); model = Model()
    diagnostic = assess_foundation(model, store, "cp", case_limit=1)
    assert diagnostic["diagnostic_only"] and not store.gate(KEY, "cp")["ready"]
    assess_foundation(model, store, "cp")
    assert store.gate(KEY, "cp")["ready"] and not model.posts
    teach_foundation(model, store, input_pool=1, outcome_pool=4)
    assert len(model.posts) == len(TEACHING_LESSONS)
    assert all(path == "/brain/consolidate" for path, _ in model.posts)
    assert not store.gate(KEY, "cp")["ready"]

def test_ruview_is_queued_before_gate_then_only_predicts_after_admission(tmp_path):
    store = LearningStore(tmp_path / "learning.db"); model = Model()
    frame = normalize_ruview(ruview_frame())
    result = ingest_ruview(frame, model, store, "cp", pool_id=11, target_pool=4)
    assert result["learning"] is False and not model.posts and not model.predictions
    store.record_assessment(KEY, "cp", 0, passed_report())
    ingest_ruview(frame, model, store, "cp", pool_id=11, target_pool=4)
    assert len(model.predictions) == 1 and not model.posts

def test_expansion_needs_verified_outcome_and_invalidates_retention(tmp_path):
    store = LearningStore(tmp_path / "learning.db"); model = Model()
    with pytest.raises(PermissionError): consolidate_reviewed(model, store, "cp", {}, input_pool=1, outcome_pool=4)
    store.record_assessment(KEY, "cp", 0, passed_report())
    with pytest.raises(ValueError): consolidate_reviewed(model, store, "cp", {}, input_pool=1, outcome_pool=4)
    experience = {"input": "verified observation", "reviewed_experience": {"outcome_verified": True, "reviewer": "operator",
                  "checked_prerogatives": list(IDS), "conflicts": [], "observed_outcome": "observed outcome"}}
    consolidate_reviewed(model, store, "cp", experience, input_pool=1, outcome_pool=4)
    assert len(model.posts) == 1 and not store.gate(KEY, "cp")["ready"]

def test_web_upload_and_agent_cannot_train_or_act_before_foundation(monkeypatch, tmp_path):
    from django.test import RequestFactory
    from wizard_chat import views, agent_views
    monkeypatch.setenv("WIZARD_LEARNING_DB", str(tmp_path / "learning.db"))
    monkeypatch.setattr(views, "_brain_for_request", lambda request: PROFILE)
    monkeypatch.setattr(views, "_extract_file_text", lambda *args: pytest.fail("Read upload before foundation"))
    request = RequestFactory().post("/upload/", {})
    request.user = SimpleNamespace(is_authenticated=True)
    assert views.WizardChatUploadView.as_view()(request).status_code == 423
    agent = RequestFactory().post("/agent/", data=json.dumps({"text": "run a command"}), content_type="application/json")
    agent.user = request.user
    assert agent_views.WizardChatAgentView.as_view()(agent).status_code == 423

def test_queue_http_cannot_self_approve_rights_or_foundation(monkeypatch, tmp_path):
    from django.test import RequestFactory
    from wizard_chat.learning_views import WizardLearningView
    monkeypatch.setenv("WIZARD_LEARNING_DB", str(tmp_path / "learning.db"))
    request = RequestFactory().post('/learning/', data=json.dumps({"title": "Candidate", "source_page": "https://example.org/source", "rights": {"review_status": "source_verified"}, "gate": {"ready": True}}), content_type='application/json')
    request.user = SimpleNamespace(is_authenticated=True)
    assert WizardLearningView.as_view()(request).status_code == 201
    store = LearningStore()
    rows = store.items(KEY, 'cp')
    assert rows[0]['payload']['rights']['review_status'] == 'pending'
    assert not store.gate(KEY, 'cp')['ready'] and not rows[0]['eligible']

def test_verified_web_correction_uses_canonical_consolidation_and_consumes_gate(monkeypatch, tmp_path):
    from django.test import RequestFactory
    from wizard_chat import views
    from tools.wizard_learning import brain
    monkeypatch.setenv("WIZARD_LEARNING_DB", str(tmp_path / "learning.db"))
    monkeypatch.setenv("WIZARD_LEARNING_CHECKPOINT_ID", "cp")
    monkeypatch.setattr(views, "_brain_for_request", lambda request: PROFILE)
    store = LearningStore(); store.record_assessment(KEY, 'cp', 0, passed_report())
    calls = []
    class Client:
        def __init__(self, *args, **kwargs): pass
        def post(self, path, body): calls.append(path); return {'consolidated': True}
        def ask(self, question): return {'answer': 'verified', 'decoder': 'trained_binding'}
    monkeypatch.setattr(brain, 'BrainClient', Client)
    body = {'question': 'a question', 'answer': 'verified'}
    factory = RequestFactory()
    unreviewed = factory.post('/train/', data=json.dumps(body), content_type='application/json')
    unreviewed.user = SimpleNamespace(is_authenticated=True)
    assert views.WizardChatTrainView.as_view()(unreviewed).status_code == 422
    assert not calls and store.gate(KEY, 'cp')['ready']
    body['prerogative_review'] = {'outcome_verified': True, 'checked_prerogatives': list(IDS), 'conflicts': []}
    reviewed = factory.post('/train/', data=json.dumps(body), content_type='application/json'); reviewed.user = unreviewed.user
    response = views.WizardChatTrainView.as_view()(reviewed)
    assert json.loads(response.content)['ok'] is True and calls == ['/brain/consolidate']
    assert not store.gate(KEY, 'cp')['ready']

def make_reviewed_video(tmp_path, store):
    cv2 = pytest.importorskip('cv2'); np = pytest.importorskip('numpy')
    from tools.wizard_learning.media import confirm_video_review
    import hashlib
    # Decoder fixture only: never submitted to a real model or production queue.
    path = tmp_path / 'decoder-fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 4, (64, 48))
    if not writer.isOpened(): pytest.skip('MJPG encoder unavailable')
    for index in range(8): writer.write(np.full((48, 64, 3), index * 25, dtype=np.uint8))
    writer.release()
    item = reviewed_video(); identifier = store.enqueue('video', item)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    review = {'reviewer': 'unit-test-only', 'video': str(path), 'sha256': sha, 'start_seconds': 0, 'end_seconds': 1,
              'capture_type': 'real-camera', 'animation_present': False, 'staged_fiction': False,
              'checked_prerogatives': list(IDS), 'conflicts': [], 'outcome_verified': True,
              'observed_outcome': {'fixture': 'decoder validation; no real-world claim'}}
    confirm_video_review(store, identifier, review)
    return identifier, path

def test_video_batch_decodes_bounded_frames_and_requires_new_retention(tmp_path):
    from tools.wizard_learning.media import train_video_batch
    store = LearningStore(tmp_path / 'learning.db'); model = Model()
    identifier, _ = make_reviewed_video(tmp_path, store)
    with pytest.raises(PermissionError): train_video_batch(store, identifier, model, 'cp')
    store.record_assessment(KEY, 'cp', 0, passed_report())
    result = train_video_batch(store, identifier, model, 'cp')
    assert result['frames_consolidated'] == 2 and result['state'] == 'completed'
    assert [path for path, _ in model.posts] == ['/brain/consolidate/multi'] * 2
    assert not store.gate(KEY, 'cp')['ready']

def test_video_lost_ack_is_quarantined_without_blind_replay(tmp_path):
    from tools.wizard_learning.media import train_video_batch
    store = LearningStore(tmp_path / 'learning.db'); model = Model()
    identifier, _ = make_reviewed_video(tmp_path, store)
    store.record_assessment(KEY, 'cp', 0, passed_report())
    def lost_ack(path, body): model.posts.append((path, body)); raise TimeoutError('ack lost')
    model.post = lost_ack
    with pytest.raises(TimeoutError): train_video_batch(store, identifier, model, 'cp')
    assert next(item for item in store.items(KEY, 'cp') if item['id'] == identifier)['state'] == 'uncertain'
    with pytest.raises(PermissionError): train_video_batch(store, identifier, model, 'cp')
    assert len(model.posts) == 1

def test_changed_video_is_rejected_before_any_learning(tmp_path):
    from tools.wizard_learning.media import train_video_batch
    store = LearningStore(tmp_path / 'learning.db'); model = Model()
    identifier, path = make_reviewed_video(tmp_path, store)
    store.record_assessment(KEY, 'cp', 0, passed_report())
    with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='differs'): train_video_batch(store, identifier, model, 'cp')
    assert not model.posts and store.gate(KEY, 'cp')['ready']
