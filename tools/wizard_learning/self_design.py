"""Declarative self-presentation, reviewed design learning, and identity-driven autonomous eye evolution (never code execution)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .curriculum import IDS, MANIFEST_HASH, MANIFEST
from .store import LearningStore, brain_key

ROOT = Path(__file__).resolve().parents[2]

EYE_BOUNDS: dict[str, tuple[float, float]] = {
    "hue": (0.0, 360.0),
    "secondary_hue": (0.0, 360.0),
    "heterochromia": (0.0, 1.0),
    "openness": (0.2, 1.0),
    "pupil_aspect": (0.28, 1.0),
    "pupil_scale": (0.65, 1.45),
    "iris_radius": (0.46, 0.68),
    "limbal_ring": (0.15, 1.0),
    "striae_detail": (0.3, 1.8),
    "collarette_glow": (0.0, 1.4),
    "sclera_tint": (0.0, 0.85),
    "cornea_gloss": (0.2, 1.5),
    "spiral_tightness": (0.08, 0.5),
    "spiral_arms": (1.0, 5.0),
    "nebula_strength": (0.0, 1.5),
    "particle_speed": (0.0, 1.0),
}

DEFAULT = {
    "version": 1,
    "eye": {
        "hue": 285.0,
        "secondary_hue": 198.0,
        "heterochromia": 0.28,
        "openness": 0.65,
        "pupil_aspect": 1.0,
        "pupil_scale": 1.0,
        "iris_radius": 0.58,
        "limbal_ring": 0.75,
        "striae_detail": 1.0,
        "collarette_glow": 0.42,
        "sclera_tint": 0.08,
        "cornea_gloss": 0.88,
        "spiral_tightness": 0.16,
        "spiral_arms": 1.0,
        "nebula_strength": 0.8,
        "particle_speed": 0.25,
    },
    "avatar": {
        "visible": False,
        "asset_id": "",
        "opacity": 0.55,
        "x": 0.75,
        "scale": 1.0,
        "animation": "",
        "morphs": {},
        "visemes": [],
    },
    "sigils": [],
    "personality": {
        "self_description": "W1z4rD V1510n — associative Hebbian digital twin grounded in multi-sensory physics and lived experience.",
        "design_intent": "Baseline violet-cyan crystal iris with natural circular pupil and luminous collarette.",
        "identity_generation": 1,
        "prerogative_alignment": {},
    },
}

SCENE = {
    "version": 1,
    "renderer": "threejs-mutable-eye-v2",
    "elements": [
        "physical crystal shell & eyelids (offline/NREM/REM/waking)",
        "dual-ring heterochromatic iris & Crypts of Fuchs",
        "adaptive circular-to-slit pupil morphology",
        "multi-arm logarithmic spiral",
        "3D star particles & volumetric nebula",
        "24 original metric sigils",
        "optional licensed rigged human",
    ],
    "control": "Return visual_proposal as a native JSON response field using the default schema, or evolve via ocular_identity neuron pool activations; no scripts, URLs or shader code.",
    "learning": "Reviewed human feedback consolidates into training store; autonomous ocular redesigns evolve within declared anatomical bounds from neural-pool experience and identity.",
    "foundation_manifest": MANIFEST_HASH,
    "required_prerogatives": list(IDS),
    "prerogative_meanings": MANIFEST["prerogatives"],
    "default": DEFAULT,
}


def default_prerogative_alignment(summary: str = "Bounded declarative ocular morphology faithful to lived neural-pool experience and transparent state signaling.") -> dict[str, str]:
    text = summary[:480]
    return {pid: text for pid in IDS}


def assets() -> dict:
    """Operator-owned registry; no model/browser can add executable or remote assets."""
    path = Path(os.getenv("WIZARD_AVATAR_REGISTRY", str(ROOT / "config/wizard_avatar_assets.json")))
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name, item in raw.items():
        if not isinstance(item, dict) or item.get("commercial_use_reviewed") is not True:
            continue
        url = item.get("url", "")
        if not re.fullmatch(r"/static/wizard-avatars/[a-zA-Z0-9_./-]+\.glb", url) or ".." in url:
            continue
        if not item.get("license") or not item.get("source") or not item.get("sha256"):
            continue
        local = ROOT / "web/frontend/public" / url.lstrip("/")
        if not local.is_file() or local.stat().st_size > 64 * 1024 * 1024:
            continue
        if hashlib.sha256(local.read_bytes()).hexdigest() != item["sha256"]:
            continue
        result[name] = {key: item[key] for key in ("url", "license", "source", "sha256")}
        for key in ("animations", "morph_targets"):
            values = item.get(key, [])
            result[name][key] = (
                [value for value in values[:128] if isinstance(value, str) and len(value) <= 100]
                if isinstance(values, list)
                else []
            )
    return result


def bounded(value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"Visual parameter {value!r} outside declared bounds [{lo}, {hi}]")
    return float(value)


def normalize(raw: dict) -> dict:
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Visual protocol version 1 required")
    result = json.loads(json.dumps(DEFAULT))
    eye = raw.get("eye", {})
    avatar = raw.get("avatar", {})
    if not isinstance(eye, dict) or not isinstance(avatar, dict):
        raise ValueError("Invalid scene objects")
    for key, pair in EYE_BOUNDS.items():
        if key in eye:
            result["eye"][key] = round(bounded(eye[key], *pair), 4)
    if "visible" in avatar and not isinstance(avatar["visible"], bool):
        raise ValueError("visible must be boolean")
    result["avatar"]["visible"] = avatar.get("visible", False)
    for key, pair in {"opacity": (0.0, 0.85), "x": (0.0, 1.0), "scale": (0.4, 1.5)}.items():
        if key in avatar:
            result["avatar"][key] = bounded(avatar[key], *pair)
    for key in ("asset_id", "animation"):
        value = avatar.get(key, "")
        if not isinstance(value, str) or len(value) > 100:
            raise ValueError("Invalid asset/animation identifier")
        result["avatar"][key] = value
    if result["avatar"]["visible"] and result["avatar"]["asset_id"] not in assets():
        raise ValueError("No reviewed human asset registered for this appearance")
    morphs = avatar.get("morphs", {})
    if not isinstance(morphs, dict) or len(morphs) > 64:
        raise ValueError("Invalid morph controls")
    result["avatar"]["morphs"] = {str(key)[:100]: bounded(value, 0.0, 1.0) for key, value in morphs.items()}
    frames = avatar.get("visemes", [])
    if not isinstance(frames, list) or len(frames) > 256:
        raise ValueError("Invalid viseme timeline")
    previous = -1.0
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("morph"), str):
            raise ValueError("Invalid viseme")
        at = bounded(frame.get("at_ms"), 0.0, 600000.0)
        if at <= previous:
            raise ValueError("Viseme times must increase")
        previous = at
        result["avatar"]["visemes"].append(
            {"at_ms": at, "morph": frame["morph"][:100], "weight": bounded(frame.get("weight"), 0.0, 1.0)}
        )
    personality = raw.get("personality", {})
    if not isinstance(personality, dict):
        raise ValueError("Invalid personality description")
    for key in ("self_description", "design_intent"):
        if key in personality:
            value = personality.get(key, "")
            if not isinstance(value, str) or len(value) > 2000:
                raise ValueError("Personality text too long")
            result["personality"][key] = value
    if "identity_generation" in personality and isinstance(personality["identity_generation"], int):
        result["personality"]["identity_generation"] = max(1, min(1_000_000, personality["identity_generation"]))
    alignment = personality.get("prerogative_alignment", {})
    if not isinstance(alignment, dict) or set(alignment) - set(IDS):
        raise ValueError("Unknown prerogative")
    if any(not isinstance(value, str) or len(value) > 500 for value in alignment.values()):
        raise ValueError("Invalid alignment explanation")
    result["personality"]["prerogative_alignment"] = alignment
    sigils = raw.get("sigils", [])
    if not isinstance(sigils, list) or len(sigils) not in (0, 24):
        raise ValueError("Supply zero or exactly 24 metric sigil paths")
    for path in sigils:
        if (
            not isinstance(path, str)
            or not 1 <= len(path) <= 2000
            or not path.lstrip().startswith(("M", "m"))
            or not re.fullmatch(r"[MmLlHhVvCcSsQqTtAaZz0-9eE+.,\s-]+", path)
        ):
            raise ValueError("Sigils accept SVG path geometry only")
        if any(
            not math.isfinite(float(number)) or abs(float(number)) > 256
            for number in re.findall(r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?", path)
        ):
            raise ValueError("Sigil coordinates outside declared bounds")
    result["sigils"] = sigils
    return result


def record(store: LearningStore, key: str, kind: str, payload: dict):
    with store.connect() as db:
        db.execute(
            "INSERT INTO events(created,brain,kind,payload) VALUES (?,?,?,?)",
            (time.time(), key, kind, json.dumps(payload, allow_nan=False)),
        )


def current(store: LearningStore, key: str) -> dict:
    with store.connect() as db:
        row = db.execute(
            "SELECT payload FROM events WHERE brain=? AND kind='visual-applied' ORDER BY seq DESC LIMIT 1",
            (key,),
        ).fetchone()
    if not row:
        return json.loads(json.dumps(DEFAULT))
    try:
        stored = json.loads(row[0])["design"]
        return normalize(stored)
    except Exception:
        return json.loads(json.dumps(DEFAULT))


def history(store: LearningStore, key: str, limit: int = 8) -> list[dict]:
    with store.connect() as db:
        rows = db.execute(
            "SELECT created, payload FROM events WHERE brain=? AND kind='visual-applied' ORDER BY seq DESC LIMIT ?",
            (key, max(1, min(32, int(limit)))),
        ).fetchall()
    out = []
    for created, raw in rows:
        try:
            item = json.loads(raw)
            out.append(
                {
                    "created": float(created),
                    "design": item.get("design"),
                    "trigger": item.get("trigger", "proposal"),
                    "fired_neurons": item.get("fired_neurons", []),
                }
            )
        except Exception:
            continue
    return out


def propose(profile: dict, raw: dict, *, allow_autonomous_eye_only: bool = True) -> dict:
    store = LearningStore()
    key = brain_key(profile)
    if isinstance(raw, dict):
        pers = raw.setdefault("personality", {})
        if isinstance(pers, dict) and not pers.get("prerogative_alignment"):
            pers["prerogative_alignment"] = default_prerogative_alignment(
                str(pers.get("design_intent") or "Bounded ocular self-design update.")
            )
    design = normalize(raw)
    if set(design["personality"]["prerogative_alignment"]) != set(IDS):
        raise ValueError("Explain the design against all twelve prerogatives")
    gate = store.gate(key, os.getenv("WIZARD_LEARNING_CHECKPOINT_ID", ""))
    eye_only = not design["avatar"]["visible"]
    admitted = bool(gate["ready"] or (allow_autonomous_eye_only and eye_only))
    record(store, key, "visual-proposed", {"design": design, "admitted": admitted, "manifest": MANIFEST_HASH})
    if not admitted:
        return {"applied": False, "reason": "Foundation assessment required for rigged human avatar", "design": None}
    record(store, key, "visual-applied", {"design": design, "manifest": MANIFEST_HASH, "trigger": "model_proposal"})
    return {"applied": True, "redesigned_on_the_fly": True, "design": design, "assets": assets(), "learning": False}


def apply_identity_redesign(
    profile: dict,
    eye_patch: dict,
    *,
    self_description: str,
    design_intent: str,
    fired_neurons: list[str] | None = None,
    sigils: list[str] | None = None,
    trigger: str = "neural_identity_evolution",
) -> dict:
    """Persist an on-the-fly ocular redesign synthesized from W1z4rD's neural pools & identity."""
    store = LearningStore()
    key = brain_key(profile)
    prev = current(store, key)
    next_gen = int(prev.get("personality", {}).get("identity_generation", 1)) + 1
    merged = json.loads(json.dumps(prev))
    merged["version"] = 1
    merged.setdefault("eye", {}).update(eye_patch)
    if sigils is not None and len(sigils) in (0, 24):
        merged["sigils"] = sigils
    merged["personality"] = {
        "self_description": self_description[:1800],
        "design_intent": design_intent[:1800],
        "identity_generation": next_gen,
        "prerogative_alignment": default_prerogative_alignment(design_intent),
    }
    normalized = normalize(merged)
    record(
        store,
        key,
        "visual-applied",
        {
            "design": normalized,
            "manifest": MANIFEST_HASH,
            "trigger": trigger,
            "fired_neurons": (fired_neurons or [])[:24],
        },
    )
    return {
        "applied": True,
        "redesigned_on_the_fly": True,
        "identity_generation": next_gen,
        "design": normalized,
        "design_intent": design_intent,
        "fired_neurons": (fired_neurons or [])[:24],
        "assets": assets(),
    }


def context(profile: dict) -> str:
    store = LearningStore()
    key = brain_key(profile)
    with store.connect() as db:
        rows = db.execute(
            "SELECT kind,payload FROM events WHERE brain=? AND kind IN ('visual-feedback','visual-observation') ORDER BY seq DESC LIMIT 3",
            (key,),
        ).fetchall()
    return "[Self-presentation interface; observations are reported, not ground truth]\n" + json.dumps(
        {
            "scene": SCENE,
            "current": current(store, key),
            "available_assets": assets(),
            "recent_feedback": [{"kind": row[0], "payload": json.loads(row[1])} for row in rows],
        },
        ensure_ascii=True,
    )
