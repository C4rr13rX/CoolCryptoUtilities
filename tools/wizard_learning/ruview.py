"""RuView REST input adapter. Sensor estimates are not external outcome labels."""
from __future__ import annotations

import json
import math
import time
import urllib.request
from urllib.parse import urlsplit

from .brain import BrainClient
from .store import LearningStore, brain_key

FEATURES = ("mean_rssi", "variance", "motion_band_power", "breathing_band_power", "dominant_freq_hz", "change_points", "spectral_power")

def normalize_ruview(raw: dict, *, now: float | None = None) -> dict:
    if raw.get("type") != "sensing_update": raise ValueError("RuView has no sensing update yet.")
    source = raw.get("source", "")
    if not isinstance(source, str) or not (source == "esp32" or source.startswith("wifi:")):
        raise ValueError("Simulation, offline, and unknown RuView sources cannot become training evidence.")
    stamp = raw.get("timestamp")
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool) or not math.isfinite(stamp): raise ValueError("Invalid RuView timestamp")
    age = (now if now is not None else time.time()) - stamp
    if age > 10 or age < -5: raise ValueError("Stale or future-dated RuView update")
    if raw.get("quality_verdict") == "Deny": raise ValueError("RuView quality gate denied this frame")
    tick = raw.get("tick")
    if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0: raise ValueError("Invalid RuView tick")
    features = raw.get("features")
    if not isinstance(features, dict): raise ValueError("Missing RuView features")
    values = {}
    for name in FEATURES:
        value = features.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or abs(value) > 1e9: raise ValueError(f"Invalid feature: {name}")
        values[name] = value
    classification = raw.get("classification", {})
    confidence = classification.get("confidence") if isinstance(classification, dict) else None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1: raise ValueError("Invalid sensor confidence")
    presence = classification.get("presence")
    if not isinstance(presence, bool): raise ValueError("Invalid presence estimate")
    # Omit SSIDs, person identity, biometric/vital estimates and raw RF arrays.
    return {"version": 1, "kind": "ruview", "timestamp": stamp, "tick": tick,
            "source": "wifi" if source.startswith("wifi:") else source,
            "features": values, "estimate": {"presence": presence, "confidence": confidence},
            "provenance": {"adapter": "ruview-rest-v1", "path": "/api/v1/sensing/latest", "ground_truth": False}}

def poll_ruview(endpoint: str, *, timeout: float = 3) -> dict:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path not in ("", "/") or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("RuView bridge requires a loopback HTTP(S) origin.")
    with urllib.request.urlopen(endpoint.rstrip("/") + "/api/v1/sensing/latest", timeout=timeout) as response:
        raw = response.read(1_000_001)
    if len(raw) > 1_000_000: raise ValueError("RuView response is too large")
    return normalize_ruview(json.loads(raw))

def ingest_ruview(frame: dict, client: BrainClient, store: LearningStore, checkpoint: str,
                  *, pool_id: int | None = None, target_pool: int | None = None) -> dict:
    key = brain_key(client.profile)
    identifier = store.enqueue("ruview", {**frame, "brain_key": key, "reviewed_experience": None})
    gate = store.gate(key, checkpoint)
    result = {"queued_id": identifier, "learning": False, "gate": gate}
    if gate["ready"] and pool_id is not None and target_pool is not None:
        if pool_id == target_pool: raise ValueError("Use a separate RuView sensory pool and target pool.")
        result["prediction"] = client.predict(pool_id, frame, target_pool)
    return result
