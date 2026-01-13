from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


def _env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def get_codex_usage() -> Dict[str, Any]:
    """
    Lightweight usage snapshot driven by environment overrides.
    No network calls; set envs if you want richer data:
      - CODEX_USAGE_5H_LIMIT, CODEX_USAGE_5H_USED
      - CODEX_USAGE_WEEK_LIMIT, CODEX_USAGE_WEEK_USED
      - CODEX_CREDITS_REMAIN
      - CODEX_USAGE_JSON='{"five_hour_used":...,"five_hour_limit":...,"week_used":...,"week_limit":...,"credits":...}'
      - CODEX_RECHECK_MINUTES
    """
    payload: Dict[str, Any] = {}
    try:
        blob = os.getenv("CODEX_USAGE_JSON")
        if blob:
            payload.update(json.loads(blob))
    except Exception:
        pass
    five_hour_limit = _env_float("CODEX_USAGE_5H_LIMIT", payload.get("five_hour_limit"))
    five_hour_used = _env_float("CODEX_USAGE_5H_USED", payload.get("five_hour_used"))
    week_limit = _env_float("CODEX_USAGE_WEEK_LIMIT", payload.get("week_limit"))
    week_used = _env_float("CODEX_USAGE_WEEK_USED", payload.get("week_used"))
    credits = _env_float("CODEX_CREDITS_REMAIN", payload.get("credits"))
    usage: Dict[str, Any] = {}
    if five_hour_limit and five_hour_used is not None:
        percent = max(0.0, min(100.0, 100.0 * (five_hour_used / five_hour_limit)))
        usage["five_hour_used_pct"] = round(percent, 2)
        usage["five_hour_remaining_pct"] = round(100.0 - percent, 2)
    if week_limit and week_used is not None:
        percent = max(0.0, min(100.0, 100.0 * (week_used / week_limit)))
        usage["week_used_pct"] = round(percent, 2)
        usage["week_remaining_pct"] = round(100.0 - percent, 2)
    if credits is not None:
        usage["credits_remaining"] = credits
    usage["recheck_minutes"] = int(_env_float("CODEX_RECHECK_MINUTES", payload.get("recheck_minutes") or 180) or 180)
    usage["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return usage
