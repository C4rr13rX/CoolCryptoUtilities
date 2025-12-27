from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from services.system_profile import SystemProfile


def resolve_pair_limit(
    base_limit: int,
    *,
    focus_assets: Sequence[str],
    horizon_bias: Optional[Dict[str, float]] = None,
    horizon_deficit: Optional[Dict[str, float]] = None,
    system_profile: Optional[SystemProfile] = None,
) -> Tuple[int, Dict[str, Any]]:
    base_limit = max(1, int(base_limit))
    limit = base_limit
    max_limit = int(os.getenv("GHOST_PAIR_LIMIT_MAX", str(max(base_limit, base_limit + 4))))
    min_limit = int(os.getenv("GHOST_PAIR_LIMIT_MIN", "1"))
    max_limit = max(max_limit, base_limit)
    min_limit = max(1, min(min_limit, max_limit))
    if system_profile and (system_profile.is_low_power or system_profile.memory_pressure):
        max_limit = min(max_limit, max(base_limit, 4))
    details: Dict[str, Any] = {
        "base_limit": base_limit,
        "min_limit": min_limit,
        "max_limit": max_limit,
    }

    focus_count = len([asset for asset in focus_assets if asset])
    if focus_count > limit:
        limit = min(max_limit, focus_count)
        details["focus_boost"] = focus_count

    deficits = horizon_deficit or {}
    deficit_total = 0.0
    if isinstance(deficits, dict):
        for value in deficits.values():
            try:
                deficit_total += max(0.0, float(value))
            except (TypeError, ValueError):
                continue
    if deficit_total > 0.0:
        step = float(os.getenv("GHOST_PAIR_DEFICIT_STEP", "12"))
        step = max(1.0, step)
        deficit_boost = int(min(4, math.ceil(deficit_total / step)))
        if deficit_boost > 0:
            limit += deficit_boost
            details["deficit_boost"] = deficit_boost
            details["deficit_total"] = deficit_total

    bias = horizon_bias or {}
    bias_boost = 0
    for bucket in ("short", "mid", "long"):
        try:
            if float(bias.get(bucket, 1.0)) >= 1.15:
                bias_boost += 1
        except (TypeError, ValueError):
            continue
    if bias_boost:
        limit += min(2, bias_boost)
        details["bias_boost"] = bias_boost

    limit = max(min_limit, min(limit, max_limit))
    details["resolved_limit"] = limit
    details["adjusted"] = limit != base_limit
    return limit, details
