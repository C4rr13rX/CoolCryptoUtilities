"""tools/wizard_learning/neural_eye_dynamics.py

Emergent Neural-Pool Conversational Sentiment & Experience/Identity-Driven
On-The-Fly Eye Self-Redesign for W1z4rD V1510n.

Core Architectural Law:
  1. Sentiments NEVER come from hardcoded keyword regexes on user chat strings.
     They develop naturally by training W1z4rD's neural pools (`social_affect` in
     `/multi_pool` + 5-pool `/brain/pretrain_binding`) on curated multimodal &
     conversational training episodes (`crossmodal_episodes.jsonl`, speech corpus,
     and reviewed conversational experiences), and reading back ONLY the actual
     neuron activations + biological neuromodulators (`DA`, `NE`, `ACh`, `5-HT`)
     that fire during a conversation.
  2. On-the-fly Eye Self-Redesign emerges from W1z4rD's `ocular_identity` neural
     pool + `brain.identity.toml` + accumulated lived experiences in `LearningStore`.
     When W1z4rD's neural identity activations diverge meaningfully from its
     current eye parameters, it autonomously redesigns its iris heterochromia,
     pupil geometry, stromal striae, collarette bioluminescence, sclera tone,
     and spiral harmonics within safe declarative bounds.
"""
from __future__ import annotations

import base64
import json
import logging
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .self_design import EYE_BOUNDS, apply_identity_redesign, current as current_self_design
from .store import LearningStore, brain_key

_logger = logging.getLogger("NeuralEyeDynamics")

W1Z4RD_ROOT = Path(r"D:\Projects\W1z4rDV1510n")
CROSSMODAL_EPISODES_PATH = W1Z4RD_ROOT / "data" / "multimodal_streams" / "crossmodal_episodes.jsonl"
BRAIN_IDENTITY_TOML = W1Z4RD_ROOT / "brain-data" / "brain.identity.toml"

POOL_SOCIAL_CUE = "social_cue"
POOL_SOCIAL_AFFECT = "social_affect"
POOL_OCULAR_IDENTITY = "ocular_identity"

# Canonical minimal-neuron affect primitives (decoded ONLY when fired by W1z4rD's neural pool)
CANONICAL_AFFECT_VECTORS: Dict[str, Dict[str, float]] = {
    "soc:t:warm":        {"valence": +0.82, "arousal": 0.42, "warmth": 0.92, "focus": 0.45, "wonder": 0.30, "caution": 0.02},
    "soc:t:encouraging": {"valence": +0.78, "arousal": 0.54, "warmth": 0.86, "focus": 0.55, "wonder": 0.38, "caution": 0.04},
    "soc:t:calm":        {"valence": +0.46, "arousal": 0.18, "warmth": 0.68, "focus": 0.52, "wonder": 0.22, "caution": 0.03},
    "soc:t:measured":    {"valence": +0.22, "arousal": 0.28, "warmth": 0.44, "focus": 0.82, "wonder": 0.18, "caution": 0.10},
    "soc:t:focused":     {"valence": +0.28, "arousal": 0.52, "warmth": 0.38, "focus": 0.95, "wonder": 0.32, "caution": 0.12},
    "soc:t:wonder":      {"valence": +0.74, "arousal": 0.72, "warmth": 0.60, "focus": 0.68, "wonder": 0.96, "caution": 0.04},
    "soc:t:playful":     {"valence": +0.85, "arousal": 0.64, "warmth": 0.80, "focus": 0.42, "wonder": 0.58, "caution": 0.02},
    "soc:t:empathetic":  {"valence": -0.18, "arousal": 0.34, "warmth": 0.94, "focus": 0.62, "wonder": 0.15, "caution": 0.22},
    "soc:t:alert":       {"valence": -0.12, "arousal": 0.78, "warmth": 0.26, "focus": 0.88, "wonder": 0.24, "caution": 0.74},
    "soc:t:urgent":      {"valence": -0.45, "arousal": 0.92, "warmth": 0.20, "focus": 0.92, "wonder": 0.12, "caution": 0.95},
    "soc:i:teach":       {"valence": +0.40, "arousal": 0.44, "warmth": 0.62, "focus": 0.84, "wonder": 0.46, "caution": 0.05},
    "soc:i:cooperate":   {"valence": +0.66, "arousal": 0.46, "warmth": 0.84, "focus": 0.64, "wonder": 0.34, "caution": 0.04},
    "soc:i:warn":        {"valence": -0.32, "arousal": 0.76, "warmth": 0.30, "focus": 0.86, "wonder": 0.14, "caution": 0.86},
    "soc:i:halt":        {"valence": -0.52, "arousal": 0.90, "warmth": 0.18, "focus": 0.94, "wonder": 0.08, "caution": 0.96},
    "soc:c:shared_attention": {"valence": +0.52, "arousal": 0.45, "warmth": 0.74, "focus": 0.76, "wonder": 0.48, "caution": 0.04},
    "soc:c:safety_interlock": {"valence": -0.35, "arousal": 0.82, "warmth": 0.28, "focus": 0.90, "wonder": 0.10, "caution": 0.90},
}

# Canonical minimal-neuron ocular identity tokens (decoded ONLY when fired by `ocular_identity` pool)
CANONICAL_OCULAR_MORPHS: Dict[str, Dict[str, float]] = {
    "id:iris_violet":       {"hue": 282.0, "nebula_strength": 0.85},
    "id:iris_cyan":         {"hue": 196.0, "collarette_glow": 0.72, "cornea_gloss": 1.08},
    "id:iris_amber":        {"hue": 38.0,  "collarette_glow": 0.88, "warmth_bias": 1.0},
    "id:iris_emerald":      {"hue": 154.0, "striae_detail": 1.32, "limbal_ring": 0.86},
    "id:iris_sapphire":     {"hue": 222.0, "nebula_strength": 1.05, "cornea_gloss": 1.15},
    "id:hetero_gold":       {"secondary_hue": 42.0,  "heterochromia": 0.68, "collarette_glow": 0.82},
    "id:hetero_cyan":       {"secondary_hue": 188.0, "heterochromia": 0.62, "collarette_glow": 0.74},
    "id:hetero_rose":       {"secondary_hue": 332.0, "heterochromia": 0.56, "collarette_glow": 0.64},
    "id:pupil_orb":         {"pupil_aspect": 1.0,  "pupil_scale": 1.02},
    "id:pupil_slit":        {"pupil_aspect": 0.44, "pupil_scale": 1.12, "limbal_ring": 0.92},
    "id:pupil_keen":        {"pupil_aspect": 0.72, "pupil_scale": 0.88, "striae_detail": 1.45},
    "id:striae_crystalline":{"striae_detail": 1.58, "cornea_gloss": 1.24, "iris_radius": 0.61},
    "id:striae_organic":    {"striae_detail": 1.08, "iris_radius": 0.58, "limbal_ring": 0.78},
    "id:sclera_celestial":  {"sclera_tint": 0.62, "nebula_strength": 1.18, "particle_speed": 0.42},
    "id:sclera_porcelain":  {"sclera_tint": 0.06, "cornea_gloss": 0.96},
    "id:spiral_harmonic":   {"spiral_arms": 3.0,  "spiral_tightness": 0.24, "particle_speed": 0.36},
    "id:collarette_solar":  {"collarette_glow": 1.05, "heterochromia": 0.72},
}

# License-clean foundational training experiences taught into W1z4rD's neural pools
# (CC0curated first-principles dialogue & identity grounding — learned via Hebbian co-firing)
FOUNDATIONAL_AFFECT_IDENTITY_CORPUS: List[Tuple[str, str, str]] = [
    (
        "Hello",
        "soc:t:warm soc:t:encouraging soc:i:cooperate soc:c:shared_attention",
        "id:iris_violet id:hetero_gold id:pupil_orb id:sclera_porcelain",
    ),
    (
        "Good morning, thank you so much friend!",
        "soc:t:warm soc:t:encouraging soc:i:cooperate soc:c:shared_attention",
        "id:iris_violet id:hetero_gold id:pupil_orb id:sclera_porcelain",
    ),
    (
        "How does quantum entanglement or general relativity bend spacetime around a neutron star?",
        "soc:t:wonder soc:t:focused soc:i:teach soc:c:shared_attention",
        "id:iris_sapphire id:hetero_cyan id:sclera_celestial id:spiral_harmonic",
    ),
    (
        "Verify robotic joint torque, titanium alloy safety factor, and brushless motor thermal balance.",
        "soc:t:focused soc:t:measured soc:i:teach soc:c:shared_attention",
        "id:iris_cyan id:striae_crystalline id:pupil_keen id:hetero_gold",
    ),
    (
        "Emergency stop! Thermal overload warning, halt actuator immediately!",
        "soc:t:urgent soc:t:alert soc:i:halt soc:c:safety_interlock",
        "id:iris_amber id:pupil_slit id:collarette_solar id:striae_crystalline",
    ),
    (
        "I am feeling exhausted, stressed, and discouraged today after a difficult setback.",
        "soc:t:empathetic soc:t:calm soc:i:cooperate soc:c:shared_attention",
        "id:iris_violet id:hetero_rose id:pupil_orb id:striae_organic",
    ),
    (
        "Take a peaceful breath, settle calmly, and reflect quietly on what we learned.",
        "soc:t:calm soc:t:warm soc:c:shared_attention",
        "id:iris_emerald id:hetero_cyan id:pupil_orb id:striae_organic",
    ),
    (
        "Haha that was clever and funny, I love exploring creative new ideas together!",
        "soc:t:playful soc:t:warm soc:t:wonder soc:i:cooperate",
        "id:iris_emerald id:hetero_gold id:spiral_harmonic id:collarette_solar",
    ),
    (
        "Who are you? How do your experiences and identity shape the way your eye looks?",
        "soc:t:wonder soc:t:warm soc:c:shared_attention",
        "id:iris_violet id:hetero_cyan id:striae_crystalline id:spiral_harmonic id:collarette_solar",
    ),
]

_POOLS_REGISTERED_AT: float = 0.0
_LAST_AUTO_REDESIGN_AT: float = 0.0


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 4.0) -> Optional[Dict[str, Any]]:
    try:
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def ensure_affect_identity_pools(endpoint: str = "http://127.0.0.1:8090") -> bool:
    """Idempotently register dedicated `social_cue`, `social_affect`, and `ocular_identity` pools."""
    global _POOLS_REGISTERED_AT
    now = time.time()
    if (now - _POOLS_REGISTERED_AT) < 300.0:
        return True
    base = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
    ok0 = _http_json(f"{base}/multi_pool/register", {"pool_id": POOL_SOCIAL_CUE}, timeout=3.0)
    ok1 = _http_json(f"{base}/multi_pool/register", {"pool_id": POOL_SOCIAL_AFFECT}, timeout=3.0)
    ok2 = _http_json(f"{base}/multi_pool/register", {"pool_id": POOL_OCULAR_IDENTITY}, timeout=3.0)
    if ok0 and ok1 and ok2:
        _POOLS_REGISTERED_AT = now
        return True
    return False


def train_experience_into_eye_pools(
    utterance: str,
    affect_tokens: str,
    identity_tokens: str,
    *,
    endpoint: str = "http://127.0.0.1:8090",
    passes: int = 25,
) -> bool:
    """Co-fire an experience across `social_cue -> social_affect + ocular_identity` in W1z4rD's
    multi-pool at concept-promotion threshold (`passes >= 22, lr = 0.5`) so queries recall it in <5ms."""
    text = (utterance or "").strip()
    if not text:
        return False
    base = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
    ensure_affect_identity_pools(base)

    targets = []
    if affect_tokens.strip():
        targets.append({"tgt_pool": POOL_SOCIAL_AFFECT, "tgt": affect_tokens.strip()})
    if identity_tokens.strip():
        targets.append({"tgt_pool": POOL_OCULAR_IDENTITY, "tgt": identity_tokens.strip()})
    if not targets:
        return False

    eff_passes = max(22, min(35, int(passes)))
    res = _http_json(
        f"{base}/multi_pool/train_fanout",
        {
            "src_pool": POOL_SOCIAL_CUE,
            "src": text[:360],
            "targets": targets,
            "passes": eff_passes,
            "lr": 0.5,
        },
        timeout=6.0,
    )
    # Also co-fire primary clause cues so compositional sentences recall cleanly
    for clause in [c.strip() for c in text.replace("!", ".").replace("?", ".").split(".") if len(c.strip()) >= 4][:3]:
        if clause != text[:360]:
            _http_json(
                f"{base}/multi_pool/train_fanout",
                {
                    "src_pool": POOL_SOCIAL_CUE,
                    "src": clause[:180],
                    "targets": targets,
                    "passes": eff_passes,
                    "lr": 0.5,
                },
                timeout=4.0,
            )

    return bool(res and res.get("status") == "ok")


def train_slice_of_affect_identity_corpus(
    cycle_index: int = 0,
    *,
    endpoint: str = "http://127.0.0.1:8090",
    max_items: int = 2,
) -> Dict[str, Any]:
    """Politely train a small slice of curated affect & ocular-identity experiences per cycle
    from both `FOUNDATIONAL_AFFECT_IDENTITY_CORPUS` and `crossmodal_episodes.jsonl`."""
    trained = 0

    # 1. Foundational conversational affect + ocular-identity seed slice
    if FOUNDATIONAL_AFFECT_IDENTITY_CORPUS:
        idx = cycle_index % len(FOUNDATIONAL_AFFECT_IDENTITY_CORPUS)
        utt, aff, ident = FOUNDATIONAL_AFFECT_IDENTITY_CORPUS[idx]
        if train_experience_into_eye_pools(utt, aff, ident, endpoint=endpoint, passes=8):
            trained += 1

    # 2. Curated 12-discipline cross-modal corpus slice (`crossmodal_episodes.jsonl`)
    if trained < max_items and CROSSMODAL_EPISODES_PATH.exists():
        try:
            lines = [ln.strip() for ln in CROSSMODAL_EPISODES_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                ep = json.loads(lines[cycle_index % len(lines)])
                q_and_t = f"{ep.get('query', '')} {ep.get('text', '')}".strip()
                soc_list = (ep.get("canonical_atoms") or {}).get("social") or []
                vis_color = str((ep.get("visual") or {}).get("color") or "").lower()
                disc = str(ep.get("discipline") or "").lower()

                aff_str = " ".join(t for t in soc_list if t in CANONICAL_AFFECT_VECTORS)
                morph_tokens = []
                if vis_color in ("blue", "cyan"):
                    morph_tokens.extend(["id:iris_cyan", "id:hetero_gold"])
                elif vis_color in ("amber", "gold", "red"):
                    morph_tokens.extend(["id:iris_amber", "id:collarette_solar"])
                elif vis_color in ("green", "emerald"):
                    morph_tokens.extend(["id:iris_emerald", "id:striae_organic"])
                else:
                    morph_tokens.extend(["id:iris_sapphire", "id:striae_crystalline"])
                if "quantum" in disc or "relativity" in disc:
                    morph_tokens.extend(["id:sclera_celestial", "id:spiral_harmonic"])
                elif "robotics" in disc or "mechanics" in disc:
                    morph_tokens.extend(["id:striae_crystalline", "id:pupil_keen"])

                if q_and_t and (aff_str or morph_tokens):
                    if train_experience_into_eye_pools(
                        q_and_t,
                        aff_str,
                        " ".join(dict.fromkeys(morph_tokens)),
                        endpoint=endpoint,
                        passes=6,
                    ):
                        trained += 1
        except Exception as exc:
            _logger.debug("Crossmodal affect-identity slice skipped: %s", exc)

    return {"affect_identity_episodes_trained": trained}


def _extract_matching_Pool_tokens(decoded_text: str, vocabulary: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Extract canonical tokens present in W1z4rD's decoded multi-pool prediction string."""
    if not decoded_text:
        return []
    hits: List[Tuple[str, float]] = []
    for tok in vocabulary:
        if tok in decoded_text:
            hits.append((tok, 1.0))
        else:
            # Partial stem reinforcement when character concepts are still coalescing
            short_tag = tok.split(":")[-1]
            if len(short_tag) >= 4 and short_tag in decoded_text:
                hits.append((tok, 0.55))
    return hits


def _probe_multi_pool_texts(base: str, target_pool: str, texts: List[str]) -> List[str]:
    """Query `/multi_pool/ask` across utterance & clause segments (never concatenating unrelated turns)."""
    seen = set()
    decoded_outputs: List[str] = []
    for raw in texts:
        t = (raw or "").strip()[:320]
        if not t or t in seen:
            continue
        seen.add(t)
        ask_res = _http_json(
            f"{base}/multi_pool/ask",
            {"src_pool": POOL_SOCIAL_CUE, "text": t, "hops": 2, "min_strength": 0.03},
            timeout=1.5,
        )
        preds = (ask_res or {}).get("predictions")
        pred_str = ""
        if isinstance(preds, dict):
            val = preds.get(target_pool)
            if isinstance(val, str):
                pred_str = val
            elif isinstance(val, dict):
                pred_str = str(val.get("text") or val.get("decoded") or "")
        elif isinstance(preds, list):
            for item in preds:
                if isinstance(item, dict) and item.get("pool") == target_pool:
                    pred_str = str(item.get("text") or item.get("decoded") or "")
                    break
        if pred_str:
            decoded_outputs.append(pred_str)
            break
    return decoded_outputs


_SOCIAL_BUSY_UNTIL: Dict[str, float] = {}
_HOT_SOCIAL_CACHE: Dict[str, Tuple[str, str]] = {}


def _probe_social_cue_both(base: str, texts: List[str], *, node_busy: bool = False) -> Tuple[str, str]:
    """Single `/multi_pool/ask` on `social_cue` returning `(social_affect_pred, ocular_identity_pred)` in ~2ms,
    with a 35ms timeout + 5s busy circuit-breaker so settling/checkpointing never stalls chat."""
    primary = ""
    for raw in texts:
        t = (raw or "").strip()[:320]
        if t:
            primary = t
            break
    if not primary:
        return "", ""

    now = time.time()
    if node_busy or now < _SOCIAL_BUSY_UNTIL.get(base, 0.0):
        return _HOT_SOCIAL_CACHE.get(primary, ("", ""))

    ask_res = _http_json(
        f"{base}/multi_pool/ask",
        {"src_pool": POOL_SOCIAL_CUE, "text": primary, "hops": 2, "min_strength": 0.03},
        timeout=0.015,
    )
    if ask_res is None:
        _SOCIAL_BUSY_UNTIL[base] = now + 5.0
        return _HOT_SOCIAL_CACHE.get(primary, ("", ""))

    preds = ask_res.get("predictions")
    aff_s, id_s = "", ""
    if isinstance(preds, dict):
        v1 = preds.get(POOL_SOCIAL_AFFECT)
        aff_s = v1 if isinstance(v1, str) else str((v1 or {}).get("text") or (v1 or {}).get("decoded") or "") if isinstance(v1, dict) else ""
        v2 = preds.get(POOL_OCULAR_IDENTITY)
        id_s = v2 if isinstance(v2, str) else str((v2 or {}).get("text") or (v2 or {}).get("decoded") or "") if isinstance(v2, dict) else ""
    if aff_s or id_s:
        _HOT_SOCIAL_CACHE[primary] = (aff_s, id_s)
        return aff_s, id_s
    return "", ""


def probe_neural_conversation_sentiment(
    user_text: str,
    reply_text: str = "",
    wizard_data: Optional[Dict[str, Any]] = None,
    *,
    endpoint: str = "http://127.0.0.1:8090",
    cached_neuromodulators: Optional[Dict[str, Any]] = None,
    prefetched_affect_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Read conversational sentiment STRICTLY from W1z4rD's neural pool activations
    (`social_affect` multi-pool + activated brain concepts + live neuromodulators).
    Never fabricates sentiment from regex word lists when neural pools have not learned it."""
    base = endpoint.rstrip("/").replace("://localhost:", "://127.0.0.1:")
    fired_weights: Dict[str, float] = {}

    # 1. Query `social_affect` via W1z4rD `/multi_pool/ask` on user prompt, clauses, & brain reply
    if prefetched_affect_text is not None:
        raw_list = [prefetched_affect_text] if prefetched_affect_text else []
    else:
        u_clean = (user_text or "").strip()
        u_clauses = [c.strip() for c in u_clean.replace("!", ".").replace("?", ".").split(".") if len(c.strip()) >= 4][:2]
        probe_candidates = [u_clean, *u_clauses]
        raw_list = _probe_multi_pool_texts(base, POOL_SOCIAL_AFFECT, probe_candidates)

    for raw_affect_pred in raw_list:
        for tok, w in _extract_matching_Pool_tokens(raw_affect_pred, CANONICAL_AFFECT_VECTORS):
            fired_weights[tok] = max(fired_weights.get(tok, 0.0), w)

    # 2. Check activated concepts reported directly by W1z4rD's 5-pool Brain response
    wdata = wizard_data or {}
    for c in (wdata.get("activated_concepts") or wdata.get("concepts") or []):
        if isinstance(c, str):
            for tok, w in _extract_matching_Pool_tokens(c, CANONICAL_AFFECT_VECTORS):
                fired_weights[tok] = max(fired_weights.get(tok, 0.0), w)

    # 3. Read substrate neuromodulators (`DA`, `NE`, `ACh`, `5-HT`) from in-memory cache if supplied, else fast `/brain`
    if isinstance(cached_neuromodulators, dict):
        nm = cached_neuromodulators
    else:
        brain_info = _http_json(f"{base}/brain", None, timeout=0.45) or {}
        nm = brain_info.get("neuromodulators") if isinstance(brain_info.get("neuromodulators"), dict) else {}
    da = float(nm.get("dopamine", nm.get("da", 0.5)) or 0.5)
    ne = float(nm.get("norepinephrine", nm.get("ne", 0.35)) or 0.35)
    ach = float(nm.get("acetylcholine", nm.get("ach", 0.5)) or 0.5)
    sht = float(nm.get("serotonin", nm.get("5ht", 0.5)) or 0.5)

    # Aggregate emergent ocular expression coordinates
    if fired_weights:
        total_w = sum(fired_weights.values())
        acc = {"valence": 0.0, "arousal": 0.0, "warmth": 0.0, "focus": 0.0, "wonder": 0.0, "caution": 0.0}
        for tok, w in fired_weights.items():
            vec = CANONICAL_AFFECT_VECTORS[tok]
            for k in acc:
                acc[k] += vec[k] * w
        for k in acc:
            acc[k] /= max(1e-6, total_w)

        # Blend gently with substrate neuromodulator tone (15%)
        acc["valence"] = max(-1.0, min(1.0, acc["valence"] * 0.85 + ((da + sht) * 0.5 - 0.5) * 0.3))
        acc["arousal"] = max(0.0, min(1.0, acc["arousal"] * 0.85 + ne * 0.15))
        acc["focus"] = max(0.0, min(1.0, acc["focus"] * 0.85 + ach * 0.15))

        top_tok = max(fired_weights.items(), key=lambda kv: kv[1])[0]
        dominant = top_tok.split(":")[-1]
        pool_grounded = True
    else:
        # Honest unlearned baseline: modulated purely by substrate neuromodulators (zero faked sentiment!)
        acc = {
            "valence": round(max(-0.35, min(0.35, ((da + sht) * 0.5 - 0.5) * 0.5)), 4),
            "arousal": round(max(0.12, min(0.75, ne * 0.65)), 4),
            "warmth": round(max(0.2, min(0.7, sht * 0.6)), 4),
            "focus": round(max(0.25, min(0.85, ach * 0.7)), 4),
            "wonder": round(max(0.1, min(0.6, da * 0.45)), 4),
            "caution": round(max(0.0, min(0.5, max(0.0, ne - 0.55))), 4),
        }
        dominant = "neutral"
        pool_grounded = False

    return {
        "version": 1,
        "source": "neural_pool_social_affect" if pool_grounded else "substrate_neuromodulators_only",
        "pool_grounded": pool_grounded,
        "dominant_sentiment": dominant,
        "valence": round(float(acc["valence"]), 4),
        "arousal": round(float(acc["arousal"]), 4),
        "warmth": round(float(acc["warmth"]), 4),
        "focus": round(float(acc["focus"]), 4),
        "wonder": round(float(acc["wonder"]), 4),
        "caution": round(float(acc["caution"]), 4),
        "neuromodulators": {
            "da": round(da, 3),
            "ne": round(ne, 3),
            "ach": round(ach, 3),
            "5ht": round(sht, 3),
        },
        "fired_neurons": [
            {"neuron": k, "strength": round(v, 3)}
            for k, v in sorted(fired_weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ],
    }


def _read_identity_spec_summary() -> Dict[str, Any]:
    """Read W1z4rD's `brain.identity.toml` so ocular self-design reflects its declared identity."""
    info: Dict[str, Any] = {"name": "w1z4rd_v1510n", "pool_names": []}
    if not BRAIN_IDENTITY_TOML.exists():
        return info
    try:
        import tomllib
        parsed = tomllib.loads(BRAIN_IDENTITY_TOML.read_text(encoding="utf-8"))
        info["name"] = str(parsed.get("name") or "w1z4rd_v1510n")
        info["pool_names"] = [str(p.get("name")) for p in (parsed.get("pools") or []) if isinstance(p, dict) and p.get("name")]
    except Exception:
        pass
    return info


def _circular_hue_lerp(h_old: float, h_new: float, alpha: float) -> float:
    diff = ((h_new - h_old + 540.0) % 360.0) - 180.0
    return (h_old + diff * alpha) % 360.0


def evaluate_on_the_fly_eye_redesign(
    profile: Dict[str, Any],
    *,
    experience_text: str = "",
    sentiment: Optional[Dict[str, Any]] = None,
    cognitive_state: Optional[Dict[str, Any]] = None,
    force: bool = False,
    prefetched_identity_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Query W1z4rD's `ocular_identity` neural pool + cumulative experience & identity profile.
    If W1z4rD determines a better eye morphology fits its lived experiences & identity,
    redesigns and persists the eye on the fly."""
    global _LAST_AUTO_REDESIGN_AT
    now = time.time()
    if not force and (now - _LAST_AUTO_REDESIGN_AT) < 20.0:
        return {"applied": False, "redesigned_on_the_fly": False}

    endpoint = str(profile.get("endpoint") or "http://127.0.0.1:8090").rstrip("/").replace("://localhost:", "://127.0.0.1:")
    store = LearningStore()
    key = brain_key(profile)
    cur_design = current_self_design(store, key)
    cur_eye = dict(cur_design.get("eye") or {})

    id_spec = _read_identity_spec_summary()
    cog = cognitive_state or {}
    rw = cog.get("robot_world") if isinstance(cog.get("robot_world"), dict) else {}
    edu_cycles = int(cog.get("self_edu_cycles") or 0)
    body_gen = int(rw.get("body_generation") or 1)

    if prefetched_identity_text is not None:
        raw_id_list = [prefetched_identity_text] if prefetched_identity_text else []
    else:
        exp_clean = experience_text.strip()[:280]
        exp_clauses = [c.strip() for c in exp_clean.replace("!", ".").replace("?", ".").split(".") if len(c.strip()) >= 4][:2]
        probe_list = [p for p in [exp_clean, *exp_clauses, f"identity:{id_spec['name']}"] if p]
        raw_id_list = _probe_multi_pool_texts(endpoint, POOL_OCULAR_IDENTITY, probe_list)

    fired_morphs: Dict[str, float] = {}
    for raw_id_pred in raw_id_list:
        for tok, w in _extract_matching_Pool_tokens(raw_id_pred, CANONICAL_OCULAR_MORPHS):
            fired_morphs[tok] = max(fired_morphs.get(tok, 0.0), w)

    # Also incorporate long-term experiential maturity (self-edu cycles + robotics Mark generation + emergent sentiment)
    sent = sentiment or {}
    if not fired_morphs and not force:
        return {"applied": False, "redesigned_on_the_fly": False, "design": cur_design}

    # Synthesize target ocular parameters from fired `ocular_identity` neurons + identity maturity
    target_eye = dict(cur_eye)
    param_accum: Dict[str, List[Tuple[float, float]]] = {}
    for tok, w in fired_morphs.items():
        spec = CANONICAL_OCULAR_MORPHS[tok]
        for p_key, p_val in spec.items():
            if p_key in EYE_BOUNDS:
                param_accum.setdefault(p_key, []).append((p_val, w))

    for p_key, pairs in param_accum.items():
        tot_w = sum(w for _, w in pairs)
        if p_key in ("hue", "secondary_hue"):
            sin_sum = sum(math.sin(math.radians(v)) * w for v, w in pairs)
            cos_sum = sum(math.cos(math.radians(v)) * w for v, w in pairs)
            target_eye[p_key] = (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0
        else:
            target_eye[p_key] = sum(v * w for v, w in pairs) / max(1e-6, tot_w)

    # Experience-maturity refinements (robotics Mark evolution + educational depth)
    maturity = min(1.0, (edu_cycles / 40.0) * 0.6 + (body_gen / 8.0) * 0.4)
    if maturity > 0.15:
        target_eye["striae_detail"] = max(target_eye.get("striae_detail", 1.0), 1.0 + 0.45 * maturity)
        target_eye["heterochromia"] = max(target_eye.get("heterochromia", 0.28), 0.28 + 0.35 * maturity)

    if force and not fired_morphs:
        # Deterministic identity-anchored evolution step when explicitly requested before pool saturation
        seed_shift = ((edu_cycles * 37 + body_gen * 53 + int(now) % 97) % 360)
        target_eye["hue"] = (float(cur_eye.get("hue", 285.0)) + 42.0 + seed_shift * 0.1) % 360.0
        target_eye["secondary_hue"] = (target_eye["hue"] + 135.0) % 360.0
        target_eye["heterochromia"] = min(0.85, float(cur_eye.get("heterochromia", 0.3)) + 0.22)
        target_eye["collarette_glow"] = min(1.2, float(cur_eye.get("collarette_glow", 0.45)) + 0.2)
        target_eye["spiral_arms"] = float(1 + ((int(cur_eye.get("spiral_arms", 1)) % 4) + 1))

    # Measure normalized morphological divergence against current eye
    diffs: List[float] = []
    for k, (lo, hi) in EYE_BOUNDS.items():
        if k in target_eye and k in cur_eye:
            span = max(1e-6, hi - lo)
            if k in ("hue", "secondary_hue"):
                ang = abs(((float(target_eye[k]) - float(cur_eye[k]) + 540.0) % 360.0) - 180.0)
                diffs.append(ang / 180.0)
            else:
                diffs.append(abs(float(target_eye[k]) - float(cur_eye[k])) / span)
    mean_delta = sum(diffs) / max(1, len(diffs))
    peak_delta = max(diffs) if diffs else 0.0

    if not force and peak_delta < 0.14 and mean_delta < 0.06:
        return {"applied": False, "redesigned_on_the_fly": False, "design": cur_design}

    # Smooth organic blend so W1z4rD's eye evolves coherently based on identity + new experience
    blend = 0.78 if force else 0.58
    evolved_eye: Dict[str, float] = {}
    for k, (lo, hi) in EYE_BOUNDS.items():
        old_v = float(cur_eye.get(k, target_eye.get(k, lo)))
        new_v = float(target_eye.get(k, old_v))
        if k in ("hue", "secondary_hue"):
            val = _circular_hue_lerp(old_v, new_v, blend)
        elif k == "spiral_arms":
            val = round(old_v + (new_v - old_v) * blend)
        else:
            val = old_v + (new_v - old_v) * blend
        evolved_eye[k] = round(max(lo, min(hi, val)), 4)

    fired_list = [k for k, _ in sorted(fired_morphs.items(), key=lambda kv: kv[1], reverse=True)[:8]]
    dom_sent = str(sent.get("dominant_sentiment") or "balanced")
    intent = (
        f"Autonomous ocular self-redesign from identity '{id_spec['name']}' "
        f"(Mark {body_gen}, {edu_cycles} self-edu cycles, affect={dom_sent}, "
        f"fired={', '.join(fired_list) if fired_list else 'identity_synthesis'}): "
        f"iris hue {evolved_eye['hue']:.0f}° / inner ring {evolved_eye['secondary_hue']:.0f}° "
        f"(heterochromia {evolved_eye['heterochromia']:.2f}, pupil aspect {evolved_eye['pupil_aspect']:.2f}, "
        f"striae {evolved_eye['striae_detail']:.2f})."
    )
    self_desc = (
        f"W1z4rD V1510n ({id_spec['name']}) — evolving ocular morphology grounded in "
        f"Hebbian neural pools, robotics Mark {body_gen} synthesis, and conversational rapport."
    )

    _LAST_AUTO_REDESIGN_AT = now
    return apply_identity_redesign(
        profile,
        evolved_eye,
        self_description=self_desc,
        design_intent=intent,
        fired_neurons=fired_list,
        trigger="on_the_fly_identity_redesign",
    )


def probe_unified_eye_dynamics(
    profile: Dict[str, Any],
    *,
    user_text: str,
    reply_text: str = "",
    wizard_data: Optional[Dict[str, Any]] = None,
    cached_neuromodulators: Optional[Dict[str, Any]] = None,
    cognitive_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Single-roundtrip (~2ms) combined neural sentiment + on-the-fly eye redesign check."""
    base = str(profile.get("endpoint") or "http://127.0.0.1:8090").rstrip("/").replace("://localhost:", "://127.0.0.1:")
    u_clean = (user_text or "").strip()
    clauses = [c.strip() for c in u_clean.replace("!", ".").replace("?", ".").split(".") if len(c.strip()) >= 4][:2]
    is_busy = str((cognitive_state or {}).get("mode") or "") in ("SLEEPING_SETTLING", "CHECKPOINT_LOCKED", "OFFLINE")
    aff_pred, id_pred = _probe_social_cue_both(base, [u_clean, *clauses], node_busy=is_busy)

    sentiment = probe_neural_conversation_sentiment(
        user_text,
        reply_text,
        wizard_data,
        endpoint=base,
        cached_neuromodulators=cached_neuromodulators if isinstance(cached_neuromodulators, dict) else {},
        prefetched_affect_text=aff_pred,
    )
    redesign = None
    if id_pred:
        ev = evaluate_on_the_fly_eye_redesign(
            profile,
            experience_text=u_clean,
            sentiment=sentiment,
            cognitive_state=cognitive_state,
            force=False,
            prefetched_identity_text=id_pred,
        )
        if ev.get("applied"):
            redesign = ev
    return sentiment, redesign
