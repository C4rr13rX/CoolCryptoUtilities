"""Declarative self-presentation and reviewed design learning, never code execution."""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

from .curriculum import IDS, MANIFEST_HASH, MANIFEST
from .store import LearningStore, brain_key

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = {"version": 1, "eye": {"hue": 285, "openness": .65, "spiral_tightness": .16, "nebula_strength": .8, "particle_speed": .25},
           "avatar": {"visible": False, "asset_id": "", "opacity": .55, "x": .75, "scale": 1, "animation": "", "morphs": {}, "visemes": []},
           "sigils": [], "personality": {"self_description": "", "design_intent": "", "prerogative_alignment": {}}}
SCENE = {"version": 1, "renderer": "threejs-mutable-eye-v1", "elements": ["physical crystal shell", "single logarithmic spiral", "procedural iris fibers", "3D star particles", "volumetric-looking nebula", "24 original metric sigils", "optional licensed rigged human"],
         "control": "Return visual_proposal as a native JSON response field using the default schema; no scripts, URLs or shader code.",
         "learning": "Only externally reviewed designs and observed outcomes can consolidate. A model preference is not evidence of quality or prerogative compliance.",
         "foundation_manifest": MANIFEST_HASH, "required_prerogatives": list(IDS), "prerogative_meanings": MANIFEST["prerogatives"], "default": DEFAULT}

def assets() -> dict:
    """Operator-owned registry; no model/browser can add executable or remote assets."""
    path = Path(os.getenv("WIZARD_AVATAR_REGISTRY", str(ROOT / "config/wizard_avatar_assets.json")))
    if not path.exists(): return {}
    raw = json.loads(path.read_text(encoding="utf-8")); result = {}
    for name, item in raw.items():
        if not isinstance(item, dict) or item.get("commercial_use_reviewed") is not True: continue
        url = item.get("url", "")
        if not re.fullmatch(r"/static/wizard-avatars/[a-zA-Z0-9_./-]+\.glb", url) or ".." in url: continue
        if not item.get("license") or not item.get("source") or not item.get("sha256"): continue
        # Registry URL must resolve to a real reviewed local artifact, not a promise.
        local = ROOT / "web/frontend/public" / url.lstrip("/")
        import hashlib
        if not local.is_file() or local.stat().st_size > 64 * 1024 * 1024: continue
        if hashlib.sha256(local.read_bytes()).hexdigest() != item["sha256"]: continue
        result[name] = {key: item[key] for key in ("url", "license", "source", "sha256")}
        for key in ("animations", "morph_targets"):
            values = item.get(key, [])
            result[name][key] = [value for value in values[:128] if isinstance(value,str) and len(value)<=100] if isinstance(values,list) else []
    return result

def bounded(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError("Visual parameter outside declared bounds")
    return value

def normalize(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("version") != 1: raise ValueError("Visual protocol version 1 required")
    result = json.loads(json.dumps(DEFAULT)); eye = raw.get("eye", {}); avatar = raw.get("avatar", {})
    if not isinstance(eye, dict) or not isinstance(avatar, dict): raise ValueError("Invalid scene objects")
    bounds = {"hue": (0,360), "openness": (.2,1), "spiral_tightness": (.08,.5), "nebula_strength": (0,1.5), "particle_speed": (0,1)}
    for key, pair in bounds.items():
        if key in eye: result["eye"][key] = bounded(eye[key], *pair)
    if "visible" in avatar and not isinstance(avatar["visible"], bool): raise ValueError("visible must be boolean")
    result["avatar"]["visible"] = avatar.get("visible", False)
    for key, pair in {"opacity": (0,.85), "x": (0,1), "scale": (.4,1.5)}.items():
        if key in avatar: result["avatar"][key] = bounded(avatar[key], *pair)
    for key in ("asset_id", "animation"):
        value = avatar.get(key, "")
        if not isinstance(value,str) or len(value)>100: raise ValueError("Invalid asset/animation identifier")
        result["avatar"][key] = value
    if result["avatar"]["visible"] and result["avatar"]["asset_id"] not in assets():
        raise ValueError("No reviewed human asset registered for this appearance")
    morphs = avatar.get("morphs", {})
    if not isinstance(morphs,dict) or len(morphs)>64: raise ValueError("Invalid morph controls")
    result["avatar"]["morphs"] = {str(key)[:100]: bounded(value,0,1) for key,value in morphs.items()}
    frames = avatar.get("visemes", [])
    if not isinstance(frames,list) or len(frames)>256: raise ValueError("Invalid viseme timeline")
    previous=-1
    for frame in frames:
        if not isinstance(frame,dict) or not isinstance(frame.get("morph"),str): raise ValueError("Invalid viseme")
        at=bounded(frame.get("at_ms"),0,600000)
        if at<=previous: raise ValueError("Viseme times must increase")
        previous=at
        result["avatar"]["visemes"].append({"at_ms":at,"morph":frame["morph"][:100],"weight":bounded(frame.get("weight"),0,1)})
    personality = raw.get("personality", {})
    if not isinstance(personality,dict): raise ValueError("Invalid personality description")
    for key in ("self_description", "design_intent"):
        value=personality.get(key,"")
        if not isinstance(value,str) or len(value)>2000: raise ValueError("Personality text too long")
        result["personality"][key]=value
    alignment=personality.get("prerogative_alignment",{})
    if not isinstance(alignment,dict) or set(alignment)-set(IDS): raise ValueError("Unknown prerogative")
    if any(not isinstance(value,str) or len(value)>500 for value in alignment.values()): raise ValueError("Invalid alignment explanation")
    result["personality"]["prerogative_alignment"]=alignment
    sigils=raw.get("sigils",[])
    if not isinstance(sigils,list) or len(sigils) not in (0,24): raise ValueError("Supply zero or exactly 24 metric sigil paths")
    for path in sigils:
        if not isinstance(path,str) or not 1<=len(path)<=2000 or not path.lstrip().startswith(('M','m')) or not re.fullmatch(r"[MmLlHhVvCcSsQqTtAaZz0-9eE+.,\s-]+",path):
            raise ValueError("Sigils accept SVG path geometry only")
        if any(not math.isfinite(float(number)) or abs(float(number))>256 for number in re.findall(r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?",path)):
            raise ValueError("Sigil coordinates outside declared bounds")
    result["sigils"]=sigils
    return result

def record(store: LearningStore, key: str, kind: str, payload: dict):
    with store.connect() as db:
        db.execute("INSERT INTO events(created,brain,kind,payload) VALUES (?,?,?,?)",(time.time(),key,kind,json.dumps(payload,allow_nan=False)))

def current(store: LearningStore, key: str) -> dict:
    with store.connect() as db:
        row=db.execute("SELECT payload FROM events WHERE brain=? AND kind='visual-applied' ORDER BY seq DESC LIMIT 1",(key,)).fetchone()
    return json.loads(row[0])["design"] if row else json.loads(json.dumps(DEFAULT))

def propose(profile: dict, raw: dict) -> dict:
    store=LearningStore(); key=brain_key(profile); design=normalize(raw)
    if set(design["personality"]["prerogative_alignment"]) != set(IDS): raise ValueError("Explain the design against all twelve prerogatives")
    gate=store.gate(key,os.getenv("WIZARD_LEARNING_CHECKPOINT_ID",""))
    record(store,key,"visual-proposed",{"design":design,"admitted":gate["ready"],"manifest":MANIFEST_HASH})
    if not gate["ready"]: return {"applied":False,"reason":"Foundation assessment required","design":None}
    record(store,key,"visual-applied",{"design":design,"manifest":MANIFEST_HASH})
    return {"applied":True,"design":design,"assets":assets(),"learning":False}

def context(profile: dict) -> str:
    store=LearningStore(); key=brain_key(profile)
    with store.connect() as db:
        rows=db.execute("SELECT kind,payload FROM events WHERE brain=? AND kind IN ('visual-feedback','visual-observation') ORDER BY seq DESC LIMIT 3",(key,)).fetchall()
    return "[Self-presentation interface; observations are reported, not ground truth]\n"+json.dumps({"scene":SCENE,"current":current(store,key),"available_assets":assets(),"recent_feedback":[{"kind":row[0],"payload":json.loads(row[1])} for row in rows]},ensure_ascii=True)
