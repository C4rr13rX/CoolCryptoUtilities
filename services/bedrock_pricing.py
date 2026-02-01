from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import json
import datetime as _dt


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float
    source: str
    as_of: str


PRICING_TABLE = {
    # AWS Bedrock pricing page (Claude 3.5 Sonnet, on-demand)
    "claude-3-5-sonnet": ModelPricing(
        0.006, 0.03, "https://aws.amazon.com/bedrock/pricing/", "2025-12-01"
    ),
    "claude-3-5-sonnet-v2": ModelPricing(
        0.006, 0.03, "https://aws.amazon.com/bedrock/pricing/", "2025-12-01"
    ),
}

CACHE_PATH = Path("runtime/c0d3r/pricing_cache.json")


def _load_cache() -> Dict[str, Any]:
    try:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"entries": []}


def _save_cache(data: Dict[str, Any]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _cache_entries() -> list:
    data = _load_cache()
    return data.get("entries") or []


def _normalize_as_of(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    try:
        if len(raw) == 7:
            _dt.datetime.strptime(raw, "%Y-%m")
            return raw
        if len(raw) == 10:
            _dt.datetime.strptime(raw, "%Y-%m-%d")
            return raw
    except Exception:
        return raw
    return raw


def cache_pricing(needle: str, pricing: ModelPricing) -> None:
    data = _load_cache()
    entries = data.get("entries") or []
    entry = {
        "needle": needle,
        "input_per_1k": pricing.input_per_1k,
        "output_per_1k": pricing.output_per_1k,
        "source": pricing.source,
        "as_of": _normalize_as_of(pricing.as_of),
        "cached_at": _dt.datetime.utcnow().strftime("%Y-%m-%d"),
    }
    entries = [e for e in entries if e.get("needle") != needle]
    entries.append(entry)
    data["entries"] = entries
    _save_cache(data)


def lookup_pricing(model_id: str) -> Optional[ModelPricing]:
    key = (model_id or "").lower()
    for needle, pricing in PRICING_TABLE.items():
        if needle in key:
            return pricing
    for entry in _cache_entries():
        needle = str(entry.get("needle") or "").lower()
        if needle and needle in key:
            try:
                return ModelPricing(
                    float(entry.get("input_per_1k")),
                    float(entry.get("output_per_1k")),
                    str(entry.get("source") or ""),
                    str(entry.get("as_of") or ""),
                )
            except Exception:
                continue
    return None


def estimate_cost(model_id: str, input_tokens: float, output_tokens: float) -> Tuple[Optional[float], Optional[float]]:
    pricing = lookup_pricing(model_id)
    if not pricing:
        return None, None
    in_cost = (input_tokens / 1000.0) * pricing.input_per_1k
    out_cost = (output_tokens / 1000.0) * pricing.output_per_1k
    return in_cost, out_cost
