from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE_VERSION = 1

DEFAULT_PROFILE: Dict[str, Any] = {
    "profile_version": PROFILE_VERSION,
    "enabled": True,
    "max_concurrent_tasks": 1,
    "tasks": [
        {
            "id": "auto_pipeline",
            "label": "Auto pipeline bootstrap",
            "enabled": True,
            "interval_minutes": 180,  # 3 hours
            "jitter_seconds": 120,
            "requires_mnemonic": False,
            "steps": [
                "downloads",
                "news",
                "training",
                "production",
            ],
        },
        {
            "id": "weekly_bootstrap",
            "label": "Weekly bootstrap",
            "enabled": True,
            "interval_minutes": 10080,  # 7 days
            "jitter_seconds": 300,
            "requires_mnemonic": False,
            "steps": [
                "discovery",
                "watchlists",
                "recommendations",
                "downloads",
                "news",
                "training",
            ],
        },
    ],
    "discovery": {
        "chains": ["base", "ethereum", "arbitrum", "optimism", "polygon"],
        "limit": 40,
        "min_liquidity_usd": 25000,
        "min_volume_usd": 50000,
        "min_change_1h": -5,
        "min_change_24h": -30,
        "max_age_hours": 36,
        "max_tokens": 25,
        "watchlist_target": "stream",
        "also_add_to_ghost": True,
    },
    "downloads": {
        "chains": ["base", "ethereum", "arbitrum", "optimism", "polygon"],
        "max_pairs": 256,
    },
    "news": {
        "lookback_hours": 72,
        "max_pages": 2,
        "max_tokens": 8,
        "default_tokens": ["BTC", "ETH", "USDC", "SOL", "AVAX"],
    },
    "training": {
        "enabled": True,
        "epochs": 1,
        "batch_size": 16,
    },
    "recommendations": {
        "enabled": True,
        "lookback_days": 30,
        "max_tokens": 8,
        "min_liquidity_usd": 50000,
        "min_volume_usd": 100000,
        "min_age_hours": 72,
        "max_age_hours": 0,
        "low_fee_chains": ["base", "arbitrum", "optimism", "polygon"],
        "wallet_bias": True,
        "min_score": 0.0,
    },
    "production": {
        "enabled": True,
        "chains": ["base", "ethereum", "arbitrum", "optimism", "polygon"],
        "require_pair_index": True,
        "min_files_per_chain": 1,
        "min_chains_ready": 1,
    },
}

_PROFILE_CACHE: Optional[Dict[str, Any]] = None
_PROFILE_MTIME: float = 0.0


def _profile_path() -> Path:
    override = os.getenv("CRON_PROFILE_PATH")
    if override:
        return Path(override)
    return Path("config") / "cron_profile.json"


def _coerce_int(value: Any, *, default: int, min_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, parsed)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merge_tasks(defaults: List[Dict[str, Any]], overrides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_by_id = {str(task.get("id")): dict(task) for task in defaults if task.get("id")}
    ordered_ids = [str(task.get("id")) for task in defaults if task.get("id")]
    for task in overrides:
        task_id = str(task.get("id") or "").strip()
        if not task_id:
            continue
        base = merged_by_id.get(task_id, {})
        merged_by_id[task_id] = _merge_dict(base, task)
        if task_id not in ordered_ids:
            ordered_ids.append(task_id)
    return [merged_by_id[task_id] for task_id in ordered_ids if task_id in merged_by_id]


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = _merge_dict(DEFAULT_PROFILE, profile)
    profile["enabled"] = bool(profile.get("enabled", True))
    profile["max_concurrent_tasks"] = _coerce_int(
        profile.get("max_concurrent_tasks"), default=1, min_value=1
    )
    tasks = profile.get("tasks") or []
    if isinstance(tasks, list):
        profile["tasks"] = _merge_tasks(DEFAULT_PROFILE.get("tasks", []), tasks)
    else:
        profile["tasks"] = list(DEFAULT_PROFILE.get("tasks", []))
    for task in profile["tasks"]:
        task["enabled"] = bool(task.get("enabled", True))
        task["interval_minutes"] = _coerce_int(task.get("interval_minutes"), default=1440, min_value=5)
        task["jitter_seconds"] = _coerce_int(task.get("jitter_seconds"), default=0, min_value=0)
        task["requires_mnemonic"] = bool(task.get("requires_mnemonic", False))
        steps = task.get("steps") or []
        if not isinstance(steps, list) or not steps:
            task["steps"] = list(DEFAULT_PROFILE["tasks"][0]["steps"])
    return profile


def load_profile(force: bool = False) -> Dict[str, Any]:
    global _PROFILE_CACHE, _PROFILE_MTIME
    path = _profile_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(DEFAULT_PROFILE, indent=2, sort_keys=True)
        path.write_text(payload, encoding="utf-8")
        _PROFILE_CACHE = json.loads(payload)
        _PROFILE_MTIME = path.stat().st_mtime
        return dict(_PROFILE_CACHE)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    if not force and _PROFILE_CACHE is not None and mtime <= _PROFILE_MTIME:
        return dict(_PROFILE_CACHE)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    profile = _normalize_profile(raw)
    _PROFILE_CACHE = dict(profile)
    _PROFILE_MTIME = mtime
    return dict(profile)


def save_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_profile(profile)
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
    global _PROFILE_CACHE, _PROFILE_MTIME
    _PROFILE_CACHE = dict(normalized)
    try:
        _PROFILE_MTIME = path.stat().st_mtime
    except OSError:
        _PROFILE_MTIME = 0.0
    return dict(normalized)
