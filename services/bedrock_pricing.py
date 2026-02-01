from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


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


def lookup_pricing(model_id: str) -> Optional[ModelPricing]:
    key = (model_id or "").lower()
    for needle, pricing in PRICING_TABLE.items():
        if needle in key:
            return pricing
    return None


def estimate_cost(model_id: str, input_tokens: float, output_tokens: float) -> Tuple[Optional[float], Optional[float]]:
    pricing = lookup_pricing(model_id)
    if not pricing:
        return None, None
    in_cost = (input_tokens / 1000.0) * pricing.input_per_1k
    out_cost = (output_tokens / 1000.0) * pricing.output_per_1k
    return in_cost, out_cost
