"""Bounded model-to-eye protocol. Never manufacture embeddings from reply text."""
from __future__ import annotations

import math
from typing import Any

MAX_FRAMES = 256
MAX_DIMENSIONS = 256
MAX_DURATION_MS = 600_000


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_eye_signal(data: dict) -> dict:
    """Accept a native timed vector trajectory or expose an honest concept fallback.

    Coordinate-space identity is mandatory: vectors from unrelated models must
    never silently become one learned space. Oversized/malformed trajectories
    are rejected as a whole rather than relabelled as valid semantic telemetry.
    """
    raw = data.get("eye_signal")
    if isinstance(raw, dict) and raw.get("version") == 1:
        space = raw.get("coordinate_space")
        frames = raw.get("frames")
        valid = isinstance(space, str) and 0 < len(space.strip()) <= 160
        valid = valid and isinstance(frames, list) and 0 < len(frames) <= MAX_FRAMES
        normalized = []
        dimensions = None
        previous = -1.0
        if valid:
            for frame in frames:
                if not isinstance(frame, dict):
                    valid = False
                    break
                at = frame.get("at_ms")
                vector = frame.get("vector")
                if not (_number(at) and 0 <= at <= MAX_DURATION_MS and at > previous):
                    valid = False
                    break
                if not (isinstance(vector, list) and 0 < len(vector) <= MAX_DIMENSIONS
                        and all(_number(value) and abs(value) <= 1e6 for value in vector)):
                    valid = False
                    break
                if dimensions is not None and len(vector) != dimensions:
                    valid = False
                    break
                dimensions = len(vector)
                previous = at
                item = {"at_ms": float(at), "vector": [float(value) for value in vector]}
                word = frame.get("word_index")
                if isinstance(word, int) and not isinstance(word, bool) and 0 <= word <= 100_000:
                    item["word_index"] = word
                normalized.append(item)
        if valid:
            return {"version": 1, "source": "model-vector", "coordinate_space": space.strip(),
                    "timing": "model", "frames": normalized, "concepts": []}

    # These are model-reported identifiers, not hashes advertised as embeddings.
    raw_concepts = data.get("activated_concepts") or data.get("concepts") or []
    concepts = []
    if isinstance(raw_concepts, list):
        for concept in raw_concepts[:128]:
            if isinstance(concept, str) and concept and len(concept) <= 256 and concept not in concepts:
                concepts.append(concept)
    return {"version": 1, "source": "model-concepts" if concepts else "none",
            "coordinate_space": "", "timing": "snapshot", "frames": [], "concepts": concepts}


def response_eye_fields(data: dict, *, answer_matches_model: bool) -> dict:
    """Only attach a trajectory when it belongs to the displayed model answer."""
    signal = normalize_eye_signal(data if answer_matches_model else {
        "activated_concepts": data.get("activated_concepts", [])
    })
    result: dict = {"eye_signal": signal}
    prerogatives = data.get("core_prerogatives")
    if isinstance(prerogatives, dict):
        # Evidence supplied by the node; this is not proof that all twelve exist.
        result["core_prerogatives"] = {
            "status": str(prerogatives.get("status", "unknown"))[:80],
            "compliant": prerogatives.get("compliant") if isinstance(prerogatives.get("compliant"), bool) else None,
            "framed_by": str(prerogatives.get("framed_by", ""))[:1000],
        }
    if answer_matches_model:
        # Existing frontend already decodes these in-memory media fields.
        for key in ("audio_b64", "video_b64"):
            value = data.get(key)
            if isinstance(value, str) and len(value) <= 16_000_000:
                result[key] = value
        images = data.get("images")
        if isinstance(images, list) and len(images) <= 8:
            if all(isinstance(value, str) for value in images) and sum(map(len, images)) <= 16_000_000:
                result["images"] = images
    return result
