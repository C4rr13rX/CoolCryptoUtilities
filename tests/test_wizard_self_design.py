import copy
import json
from types import SimpleNamespace

import pytest
from tools.wizard_learning import self_design as design
from tools.wizard_learning.curriculum import IDS, SUITES
from tools.wizard_learning.store import LearningStore, brain_key

PROFILE={"endpoint":"http://127.0.0.1:8090","id":"design-test","chat_path":"/brain/chat"}

def proposal():
    value=copy.deepcopy(design.DEFAULT)
    value["personality"]["prerogative_alignment"]={key:"Use truthful, helpful presentation consistent with this prerogative." for key in IDS}
    return value

def admit(store):
    report={name:[{"case_id":case["id"],"passed":True} for case in cases] for name,cases in SUITES.items()}
    store.record_assessment(brain_key(PROFILE),"checkpoint",store.revision(brain_key(PROFILE)),report)

def test_visual_protocol_rejects_unbounded_or_executable_controls():
    for bad in (float('nan'),float('inf'),True,-1,361):
        value=proposal();value["eye"]["hue"]=bad
        with pytest.raises(ValueError):design.normalize(value)
    value=proposal();value["eye"]["shader_code"]="doSomething()";value["avatar"]["url"]="https://attacker/model.glb"
    result=design.normalize(value)
    assert "shader_code" not in result["eye"] and "url" not in result["avatar"]
    value["avatar"]["visible"]=True;value["avatar"]["asset_id"]="unknown"
    with pytest.raises(ValueError):design.normalize(value)

def test_visemes_are_bounded_monotonic_and_morphs_are_numeric():
    value=proposal();value["avatar"]["morphs"]={"jawOpen":.5}
    value["avatar"]["visemes"]=[{"at_ms":0,"morph":"viseme_aa","weight":.7},{"at_ms":125,"morph":"viseme_PP","weight":.5}]
    assert len(design.normalize(value)["avatar"]["visemes"])==2
    value["avatar"]["visemes"][1]["at_ms"]=0
    with pytest.raises(ValueError):design.normalize(value)
    value["avatar"]["visemes"]=[];value["avatar"]["morphs"]["jawOpen"]="eval()"
    with pytest.raises(ValueError):design.normalize(value)

def test_model_can_redesign_sigil_geometry_without_injecting_svg_markup():
    value=proposal();value["sigils"]=["M16 12L32 44 48 12M16 32H48"]*24
    assert len(design.normalize(value)["sigils"])==24
    for invalid in ('<svg onload="attack()"/>','M0 0L1e999 10','M0 0L999 10'):
        value["sigils"][0]=invalid
        with pytest.raises(ValueError):design.normalize(value)

def test_native_self_design_is_checkpoint_gated_and_reversible(tmp_path,monkeypatch):
    monkeypatch.setenv("WIZARD_LEARNING_DB",str(tmp_path/"learning.sqlite3"));monkeypatch.setenv("WIZARD_LEARNING_CHECKPOINT_ID","checkpoint")
    store=LearningStore();value=proposal();value["eye"]["hue"]=330
    assert design.propose(PROFILE,value)["applied"] is False
    assert design.current(store,brain_key(PROFILE))["eye"]["hue"]==285
    admit(store)
    assert design.propose(PROFILE,value)["applied"] is True
    assert design.current(store,brain_key(PROFILE))["eye"]["hue"]==330
    # Applying a visual setting never silently consolidates or consumes training admission.
    assert store.gate(brain_key(PROFILE),"checkpoint")["ready"]
    store.invalidate(brain_key(PROFILE),"new learning")
    value["eye"]["hue"]=40
    assert design.propose(PROFILE,value)["applied"] is False
    with pytest.raises(ValueError):design.propose(PROFILE,copy.deepcopy(design.DEFAULT))

def test_scene_context_contains_real_scene_graph_and_reported_feedback(tmp_path,monkeypatch):
    monkeypatch.setenv("WIZARD_LEARNING_DB",str(tmp_path/"learning.sqlite3"))
    store=LearningStore();design.record(store,brain_key(PROFILE),"visual-feedback",{"feedback":"Make the galaxy visible","ground_truth":False})
    text=design.context(PROFILE)
    assert "single logarithmic spiral" in text and "Make the galaxy visible" in text
    assert all(identifier in text for identifier in IDS)
    assert "observations are reported, not ground truth" in text
    assert "truth" in text.lower()

def test_avatar_registry_needs_real_local_digest_and_commercial_rights(tmp_path,monkeypatch):
    import hashlib
    root=tmp_path/"web/frontend/public/static/wizard-avatars";root.mkdir(parents=True)
    path=root/"human.glb";path.write_bytes(b"test fixture only")
    registry=tmp_path/"registry.json";monkeypatch.setenv("WIZARD_AVATAR_REGISTRY",str(registry));monkeypatch.setattr(design,"ROOT",tmp_path)
    item={"url":"/static/wizard-avatars/human.glb","license":"test-only","source":"fixture","commercial_use_reviewed":True,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"animations":["idle"],"morph_targets":["jawOpen"]}
    registry.write_text(json.dumps({"human":item}))
    assert design.assets()["human"]["animations"]==["idle"]
    item["commercial_use_reviewed"]=False;registry.write_text(json.dumps({"human":item}));assert not design.assets()
    item["commercial_use_reviewed"]=True;item["sha256"]="0"*64;registry.write_text(json.dumps({"human":item}));assert not design.assets()

def test_feedback_cannot_unlock_or_train_before_foundation(tmp_path,monkeypatch):
    from web.wizard_chat.self_design_views import WizardSelfDesignView
    from web.wizard_chat import views
    monkeypatch.setenv("WIZARD_LEARNING_DB",str(tmp_path/"learning.sqlite3"));monkeypatch.setenv("WIZARD_LEARNING_CHECKPOINT_ID","checkpoint")
    monkeypatch.setattr(views,"_brain_for_request",lambda request:PROFILE)
    body={"kind":"visual-feedback","design":proposal(),"feedback":"A clearer iris","learn":True,"reviewed_experience":{"reviewer":"operator","outcome_verified":True,"checked_prerogatives":list(IDS),"conflicts":[],"observed_outcome":{"preference":"clearer iris"}}}
    request=SimpleNamespace(user=SimpleNamespace(is_authenticated=True),body=json.dumps(body).encode())
    assert WizardSelfDesignView().post(request).status_code==423
    body["learn"]=False;request.body=json.dumps(body).encode()
    response=WizardSelfDesignView().post(request)
    assert response.status_code==200 and json.loads(response.content)["learning"] is False
    request.user.is_authenticated=False
    assert WizardSelfDesignView().post(request).status_code==403

def test_reviewed_personality_outcome_uses_existing_consolidation_boundary(tmp_path,monkeypatch):
    from web.wizard_chat.self_design_views import WizardSelfDesignView
    from web.wizard_chat import views
    from tools.wizard_learning import brain
    monkeypatch.setenv("WIZARD_LEARNING_DB",str(tmp_path/"learning.sqlite3"));monkeypatch.setenv("WIZARD_LEARNING_CHECKPOINT_ID","checkpoint")
    monkeypatch.setattr(views,"_brain_for_request",lambda request:PROFILE)
    store=LearningStore();admit(store);calls=[]
    monkeypatch.setattr(brain.BrainClient,"post",lambda self,path,body:(calls.append((path,body)) or {"consolidated":True}))
    body={"kind":"visual-feedback","design":proposal(),"feedback":"Calm and honest appearance","learn":True,"reviewed_experience":{"reviewer":"operator","outcome_verified":True,"checked_prerogatives":list(IDS),"conflicts":[],"observed_outcome":{"preference":"calm and honest appearance"}}}
    request=SimpleNamespace(user=SimpleNamespace(is_authenticated=True),body=json.dumps(body).encode())
    assert WizardSelfDesignView().post(request).status_code==200
    assert len(calls)==1 and calls[0][0]=="/brain/consolidate"
    assert not store.gate(brain_key(PROFILE),"checkpoint")["ready"]
