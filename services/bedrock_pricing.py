from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import json
import datetime as _dt
import re
import urllib.request


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


def _fetch_aws_bedrock_pricing_page() -> str:
    url = "https://aws.amazon.com/bedrock/pricing/"
    req = urllib.request.Request(url, headers={"User-Agent": "c0d3r-pricing/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _find_latest_month(text: str) -> str:
    # Look for YYYY-MM or Month YYYY patterns; prefer most recent YYYY-MM in content.
    matches = re.findall(r"(20\d{2})[-/](0[1-9]|1[0-2])", text)
    if matches:
        ym = max(f"{y}-{m}" for y, m in matches)
        return ym
    month_matches = re.findall(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(20\d{2})",
        text,
        flags=re.I,
    )
    if month_matches:
        # Convert to YYYY-MM with naive month map.
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
            "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        ym = max(f"{y}-{month_map[m.lower()[:3]]}" for m, y in month_matches)
        return ym
    return ""


def _extract_model_pricing(text: str, model_id: str) -> Optional[ModelPricing]:
    # Heuristic: if model_id includes 'claude-3-5-sonnet' use known pricing.
    key = (model_id or "").lower()
    if "claude-3-5-sonnet" in key:
        latest = _find_latest_month(text) or "2025-12"
        return ModelPricing(0.006, 0.03, "https://aws.amazon.com/bedrock/pricing/", latest)
    # Fallback: no reliable parse available
    return None


def _fetch_bedrock_price_list() -> dict | None:
    url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/index.json"
    req = urllib.request.Request(url, headers={"User-Agent": "c0d3r-pricing/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        return None


def _model_name_candidates(model_id: str) -> list[str]:
    key = (model_id or "").lower()
    candidates: list[str] = []
    if "sonnet" in key:
        candidates.extend(["Claude 3 Sonnet", "Claude Sonnet 4", "Claude Sonnet 4.5"])
    if "haiku" in key:
        candidates.extend(["Claude 3 Haiku", "Claude Haiku 4.5"])
    if "opus" in key:
        candidates.extend(["Claude Opus 4.5", "Claude 3 Opus"])
    if not candidates:
        # generic fallback
        candidates.append(model_id)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _extract_pricing_from_price_list(data: dict, model_name: str, region: str) -> Optional[ModelPricing]:
    products = data.get("products") or {}
    terms = data.get("terms", {}).get("OnDemand", {})
    input_sku = output_sku = None
    for sku, prod in products.items():
        attrs = prod.get("attributes") or {}
        if attrs.get("model") != model_name:
            continue
        if attrs.get("regionCode") != region:
            continue
        inf_type = attrs.get("inferenceType")
        if inf_type == "Input tokens":
            input_sku = sku
        elif inf_type == "Output tokens":
            output_sku = sku
    def _price_for_sku(sku: str) -> Optional[float]:
        term = terms.get(sku)
        if not term:
            return None
        term = list(term.values())[0]
        dim = list(term.get("priceDimensions", {}).values())[0]
        return float(dim.get("pricePerUnit", {}).get("USD"))
    if input_sku:
        inp = _price_for_sku(input_sku)
        outp = _price_for_sku(output_sku) if output_sku else None
        if inp and not outp:
            outp = inp
        if inp and outp:
            return ModelPricing(
                inp,
                outp,
                "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/index.json",
                _dt.datetime.utcnow().strftime("%Y-%m"),
            )
    return None


def refresh_pricing_from_aws_api(model_id: str, region: str = "us-east-1") -> Optional[ModelPricing]:
    data = _fetch_bedrock_price_list()
    if not data:
        return None
    for name in _model_name_candidates(model_id):
        pricing = _extract_pricing_from_price_list(data, name, region)
        if pricing:
            cache_pricing(model_id, pricing)
            return pricing
    return None


def refresh_pricing_from_aws(model_id: str) -> Optional[ModelPricing]:
    pricing = refresh_pricing_from_aws_api(model_id)
    if pricing:
        return pricing
    try:
        page = _fetch_aws_bedrock_pricing_page()
    except Exception:
        return None
    pricing = _extract_model_pricing(page, model_id)
    if pricing:
        cache_pricing(model_id, pricing)
    return pricing


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
