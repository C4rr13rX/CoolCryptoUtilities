"""tools/wizard_learning/multimodal_feeder.py

Polite, streaming Cross-Modal Hebbian Feeder for W1z4rD's 24/7 Autonomous Lifecycle.

Streams verified, cloud-backed, zero-local-bloat multimodal records from:
  1. `D:\\Projects\\W1z4rDV1510n\\data\\manifests\\english_speech_cloud_manifest.jsonl`
     (C4rr13rX speech synthesis -> GCS verified & deleted locally -> W1ZA1 minimal acoustic motifs)
  2. `D:\\Projects\\W1z4rDV1510n\\data\\multimodal_noun_dictionary\\noun_multimodal_dictionary.jsonl`
     (License-clean noun images + spoken audio -> GCS verified & deleted locally -> 64x64 visual + 16kHz audio primitives + social affordances)
  3. `D:\\Projects\\W1z4rDV1510n\\data\\multimodal_streams\\twelve_discipline_obstacle_pack.json`
     (12-discipline cross-modal text + social + visual + audio + sensor/signal anchors)

Enforces strict non-embellishment and minimal-neuron canonical decomposition:
only canonical primitive tokens (`vcol:*`, `vshp:*`, `vedg:*`, `aud:*`, `sdom:*`, `saff:*`,
and compact `W1ZA1` motifs) co-fire across W1z4rD's multi-pool brain
(`POOL_TEXT=1`, `POOL_IMAGE=2`, `POOL_AUDIO=3`, `POOL_ACTION=4`) so neurons that
wire together fire together and birth compositional cross-modal relationships.

Maintains a tiny byte-offset/index cursor on disk (`< 512 B` RAM) and scales
episodes per cycle politely with the PC RAM allowance tier.
"""
from __future__ import annotations

import base64
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

_logger = logging.getLogger("MultimodalFeeder")

W1Z4RD_DATA_ROOT = Path(r"D:\Projects\W1z4rDV1510n\data")
SPEECH_MANIFEST = W1Z4RD_DATA_ROOT / "manifests" / "english_speech_cloud_manifest.jsonl"
NOUN_DICTIONARY = W1Z4RD_DATA_ROOT / "multimodal_noun_dictionary" / "noun_multimodal_dictionary.jsonl"
DISCIPLINE_PACK = W1Z4RD_DATA_ROOT / "multimodal_streams" / "twelve_discipline_obstacle_pack.json"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CURSOR_PATH = _PROJECT_ROOT / "runtime" / "c0d3r" / "multimodal_feeder_cursor.json"

# Canonical 5-pool W1z4rD Brain IDs (`crates/brain/src/config.rs`)
POOL_TEXT = 1
POOL_IMAGE = 2
POOL_AUDIO = 3
POOL_ACTION = 4


def _b64u(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _read_jsonl_at_line(path: Path, target_idx: int) -> Tuple_Dict_Int:
    """Stream-read a single JSONL line by wrapping index without loading the file into RAM."""
    if not path.exists() or path.stat().st_size == 0:
        return None, 0
    chosen: Optional[Dict[str, Any]] = None
    first_valid: Optional[Dict[str, Any]] = None
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if count == 0:
                    try:
                        first_valid = json.loads(line)
                    except Exception:
                        pass
                if count == target_idx:
                    try:
                        chosen = json.loads(line)
                    except Exception:
                        chosen = None
                count += 1
        if count == 0:
            return None, 0
        if chosen is not None:
            return chosen, (target_idx + 1) % count
        # Wrapped past EOF; return line 0 and advance cursor to 1
        return first_valid, 1 % max(1, count)
    except Exception as exc:
        _logger.debug("JSONL stream read skipped on %s: %s", path.name, exc)
        return None, 0


# Alias for type clarity without extra imports
Tuple_Dict_Int = tuple[Optional[Dict[str, Any]], int]


class PoliteMultimodalFeeder:
    """Streams minimal-neuron cross-modal episodes into W1z4rD (:8090) within PC RAM allowance."""

    def __init__(self, endpoint: str = "http://127.0.0.1:8090", cursor_path: Path = CURSOR_PATH):
        self.endpoint = endpoint.rstrip("/")
        self.cursor_path = cursor_path
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_cursor(self) -> Dict[str, int]:
        if self.cursor_path.exists():
            try:
                data = json.loads(self.cursor_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {
                        "speech_idx": int(data.get("speech_idx", 0)),
                        "noun_idx": int(data.get("noun_idx", 0)),
                        "anchor_idx": int(data.get("anchor_idx", 0)),
                        "episodes_fed_total": int(data.get("episodes_fed_total", 0)),
                    }
            except Exception:
                pass
        return {"speech_idx": 0, "noun_idx": 0, "anchor_idx": 0, "episodes_fed_total": 0}

    def _save_cursor(self, cursor: Dict[str, int]) -> None:
        try:
            tmp = self.cursor_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cursor, indent=2), encoding="utf-8")
            tmp.replace(self.cursor_path)
        except Exception:
            pass

    def _post_cofire_frames(self, frames: List[Dict[str, Any]], timeout: float = 35.0) -> bool:
        if len(frames) < 2:
            return False
        payload = json.dumps({"frames": frames}).encode("utf-8")
        try:
            req = urllib.request.Request(
                f"{self.endpoint}/brain/pretrain_binding",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return "error" not in body
        except Exception as exc:
            _logger.debug("Multimodal pretrain_binding skipped: %s", exc)
            return False

    def feed_speech_utterance(self, cursor: Dict[str, int]) -> bool:
        rec, next_idx = _read_jsonl_at_line(SPEECH_MANIFEST, cursor["speech_idx"])
        cursor["speech_idx"] = next_idx
        if not rec or not rec.get("verified_in_cloud") or not rec.get("deleted_locally"):
            return False

        text = str(rec.get("text") or "").strip()
        atoms = rec.get("compact_audio_atoms") or {}
        audio_b64 = str(atoms.get("pool_audio_b64u") or "").strip()
        if not audio_b64 and atoms.get("pool_audio_motifs"):
            audio_b64 = _b64u(str(atoms["pool_audio_motifs"]))
        if not text or not audio_b64:
            return False

        frames: List[Dict[str, Any]] = [
            {"pool_id": POOL_TEXT, "frame": _b64u(text)},
            {"pool_id": POOL_AUDIO, "frame": audio_b64},
        ]
        links = rec.get("modality_links") or {}
        paired_resp = str(links.get("paired_response") or "").strip()
        if paired_resp:
            frames.append({"pool_id": POOL_ACTION, "frame": _b64u(paired_resp)})

        return self._post_cofire_frames(frames)

    def feed_noun_visual_audio(self, cursor: Dict[str, int]) -> bool:
        rec, next_idx = _read_jsonl_at_line(NOUN_DICTIONARY, cursor["noun_idx"])
        cursor["noun_idx"] = next_idx
        if not rec or not rec.get("cloud_verified") or not rec.get("local_temp_deleted"):
            return False

        noun = str(rec.get("noun") or "").strip()
        sentence = str(rec.get("grounding_sentence") or "").strip()
        vis_tokens = (rec.get("visual_decomposition") or {}).get("compact_brain_tokens") or []
        aud_tokens = (rec.get("audio_decomposition") or {}).get("compact_brain_tokens") or []
        soc_tokens = rec.get("social_primitives") or []
        if not noun or not vis_tokens or not aud_tokens:
            return False

        # Minimal canonical non-embellished representation across Text + Visual + Audio + Action/Social
        text_canonical = f"noun:{noun} " + " ".join(str(t) for t in soc_tokens[:4])
        vis_canonical = " ".join(str(t) for t in vis_tokens[:20])
        aud_canonical = " ".join(str(t) for t in aud_tokens[:16])
        action_canonical = sentence if sentence else f"{noun}: {' '.join(str(t) for t in soc_tokens[:3])}"

        frames = [
            {"pool_id": POOL_TEXT, "frame": _b64u(text_canonical)},
            {"pool_id": POOL_IMAGE, "frame": _b64u(vis_canonical)},
            {"pool_id": POOL_AUDIO, "frame": _b64u(aud_canonical)},
            {"pool_id": POOL_ACTION, "frame": _b64u(action_canonical)},
        ]
        return self._post_cofire_frames(frames)

    def feed_discipline_anchor(self, cursor: Dict[str, int]) -> bool:
        if not DISCIPLINE_PACK.exists():
            return False
        try:
            pack = json.loads(DISCIPLINE_PACK.read_text(encoding="utf-8"))
            anchors = pack.get("foundational_anchors") or []
            if not anchors:
                return False
            idx = cursor["anchor_idx"] % len(anchors)
            cursor["anchor_idx"] = (idx + 1) % len(anchors)
            item = anchors[idx]
            q = str(item.get("query") or "").strip()
            a = str(item.get("answer") or "").strip()
            if not q or not a:
                return False
            frames = [
                {"pool_id": POOL_TEXT, "frame": _b64u(q)},
                {"pool_id": POOL_ACTION, "frame": _b64u(a)},
            ]
            return self._post_cofire_frames(frames)
        except Exception:
            return False

    def feed_elastic_slice(self, allowance_tier: str = "BALANCED") -> Dict[str, Any]:
        """Feed a polite slice of multimodal co-firing episodes scaled to PC RAM allowance."""
        tier = (allowance_tier or "BALANCED").upper()
        if tier == "EXPANDED":
            budget = 4
        elif tier == "BALANCED":
            budget = 2
        else:
            # COMPACT or EMERGENCY_SHRINK: keep learning gently (1 episode) with zero bloat
            budget = 1

        cursor = self._load_cursor()
        fed_speech = 0
        fed_nouns = 0
        fed_anchors = 0

        # Rotate modalities fairly across cycles so text+audio+visual+social all wire together
        step_seed = cursor["episodes_fed_total"]
        for i in range(budget):
            slot = (step_seed + i) % 3
            ok = False
            if slot == 0:
                ok = self.feed_noun_visual_audio(cursor)
                if ok:
                    fed_nouns += 1
            elif slot == 1:
                ok = self.feed_speech_utterance(cursor)
                if ok:
                    fed_speech += 1
            else:
                ok = self.feed_discipline_anchor(cursor)
                if ok:
                    fed_anchors += 1

        total_fed = fed_speech + fed_nouns + fed_anchors
        cursor["episodes_fed_total"] = int(cursor.get("episodes_fed_total", 0)) + total_fed
        self._save_cursor(cursor)

        return {
            "allowance_tier": tier,
            "episode_budget": budget,
            "episodes_fed": total_fed,
            "noun_visual_audio_fed": fed_nouns,
            "speech_cloud_fed": fed_speech,
            "discipline_anchors_fed": fed_anchors,
            "episodes_fed_total": cursor["episodes_fed_total"],
        }
