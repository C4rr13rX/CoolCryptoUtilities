"""Discover callable free chat models not already present in ATF's catalog.

Provider rules deliberately use machine-readable first-party model endpoints
where available. Every candidate is deduplicated by provider/model identity;
``--exclude-model-ids`` also suppresses the same normalized model across other
providers when the goal is model diversity rather than quota diversity.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs" / "free_ai_model_catalog.csv"
OUTPUT = ROOT / "runtime" / "agent_the_freeloader" / "discovered_models.json"


def _get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "ATF-FreeModelDiscovery/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _get_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ATF-FreeModelDiscovery/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def _normalized(model_id: str) -> str:
    value = model_id.lower().removesuffix(":free")
    value = value.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def _row(provider: str, model_id: str, best_at: str, quota: str, limits: str,
         base_url: str, key_name: str, catalog_url: str, quota_url: str,
         notes: str = "Recurring free tier") -> dict[str, str]:
    return {
        "provider": provider, "model_id": model_id,
        "modalities": "text to text", "best_at": best_at,
        "free_quota": quota, "rate_limits": limits,
        "base_url": base_url, "endpoint": "/v1/chat/completions",
        "api_style": "OpenAI-compatible", "authentication": f"Bearer {key_name}",
        "catalog_url": catalog_url, "quota_source_url": quota_url,
        "verified_on": date.today().isoformat(), "notes": notes,
    }


def discover() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    routeway = _get_json("https://api.routeway.ai/v1/models").get("data", [])
    for model in routeway:
        pricing = model.get("pricing") or {}
        input_price = (pricing.get("input") or {}).get("price_per_million_t")
        output_price = (pricing.get("output") or {}).get("price_per_million_t")
        if not model.get("available") or input_price != 0 or output_price != 0:
            continue
        caps = model.get("capabilities") or {}
        strengths = ["general chat"]
        if caps.get("function_call"): strengths.append("tools")
        if caps.get("reasoning"): strengths.append("reasoning")
        rows.append(_row(
            "Routeway", str(model["id"]), "; ".join(strengths),
            "200 requests/day shared across the free account",
            "5 RPM; 200 RPD shared across all Routeway free models",
            "https://api.routeway.ai", "ROUTEWAY_API_KEY",
            "https://api.routeway.ai/v1/models",
            "https://docs.routeway.ai/getting-started/rate-limits",
        ))

    logfare = _get_json("https://logfare.ai/v1/models").get("data", [])
    for model in logfare:
        if model.get("tier") != 1 or model.get("requires_training_optin"):
            continue
        rows.append(_row(
            "Logfare", str(model["id"]), "general chat; coding; reasoning",
            "Free tier; provider currently states no rate limits",
            "Provider does not publish a numeric limit",
            "https://logfare.ai", "LOGFARE_API_KEY",
            "https://logfare.ai/v1/models", "https://logfare.ai/",
            "Recurring free access claimed; review provider data-use terms before sensitive work",
        ))

    for model_id, best_at in (("auto", "fast routed chat"), ("hrLLM", "general chat")):
        rows.append(_row(
            "LLM.kiwi", model_id, best_at,
            "Free account access", "Baseline free-account limits; numeric quota unpublished",
            "https://api.llm.kiwi", "LLM_KIWI_API_KEY",
            "https://api.llm.kiwi/v1/models", "https://llm.kiwi/pricing",
        ))

    for model_id, best_at in (
        ("llama-3.1-8b", "general chat"), ("mistral-7b", "fast chat"),
        ("qwen-2.5-7b", "multilingual chat and coding"),
        ("phi-3-mini", "fast simple classification and chat"),
    ):
        rows.append(_row(
            "LLMRack", model_id, best_at,
            "10,000 tokens/day shared across the free account",
            "10 RPM; 10,000 TPD shared across all LLMRack models",
            "https://llmrack.com", "LLMRACK_API_KEY",
            "https://llmrack.com/", "https://llmrack.com/",
        ))

    # Cloudflare publishes its live catalog as first-party Markdown. Extract
    # only non-deprecated Text Generation cards, then resolve exact @cf IDs
    # from each model page concurrently. All models share one neuron pool.
    cf_index = _get_text("https://developers.cloudflare.com/workers-ai/models/index.md")
    link_pattern = re.compile(
        r"https://developers\.cloudflare\.com/workers-ai/models/([^/]+)/"
    )
    slugs: list[str] = []
    previous_end = 0
    for match in link_pattern.finditer(cf_index):
        card = cf_index[previous_end:match.end()]
        previous_end = match.end()
        if "Text Generation" in card and "Deprecated" not in card:
            slugs.append(match.group(1))
    slugs = list(dict.fromkeys(slugs))

    def cloudflare_row(slug: str) -> dict[str, str] | None:
        if any(token in slug.lower() for token in ("guard", "safety", "moderation")):
            return None
        page = _get_text(
            f"https://developers.cloudflare.com/workers-ai/models/{slug}/index.md"
        )
        model_match = re.search(r"`(@cf/[^`]+)`", page)
        if "Text Generation" not in page or not model_match:
            return None
        strengths = ["general chat"]
        if re.search(r"Function calling\s+.*Yes", page): strengths.append("tools")
        if re.search(r"Reasoning\s+.*Yes", page): strengths.append("reasoning")
        return _row(
            "Cloudflare Workers AI", model_match.group(1), "; ".join(strengths),
            "Shared 10000 neurons/day", "Shared 10000 neurons/day",
            "https://api.cloudflare.com", "CLOUDFLARE_API_TOKEN",
            f"https://developers.cloudflare.com/workers-ai/models/{slug}/",
            "https://developers.cloudflare.com/workers-ai/platform/pricing/",
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        for item in executor.map(cloudflare_row, slugs):
            if item:
                rows.append(item)

    kilo_models = _get_json("https://api.kilo.ai/api/gateway/models").get("data", [])
    for model in kilo_models:
        model_id = str(model.get("id") or "")
        if not (model.get("isFree") or model_id.endswith(":free") or model_id == "kilo-auto/free"):
            continue
        if any(token in model_id.lower() for token in ("safety", "guard", "moderation")):
            continue
        architecture = model.get("architecture") or {}
        if "text" not in (architecture.get("output_modalities") or ["text"]):
            continue
        kilo_row = _row(
            "Kilo Gateway", model_id, "general chat; coding; routed free inference",
            "200 requests/hour shared by IP across all free models",
            "200 requests/hour shared by IP across all Kilo free models",
            "https://api.kilo.ai/api/gateway", "optional KILO_API_KEY",
            "https://api.kilo.ai/api/gateway/models",
            "https://kilo.ai/docs/gateway/usage-and-billing",
            "Anonymous recurring free access; upstream providers may log prompts",
        )
        kilo_row["endpoint"] = "/chat/completions"
        rows.append(kilo_row)

    zen_models = _get_json("https://opencode.ai/zen/v1/models").get("data", [])
    for model in zen_models:
        model_id = str(model.get("id") or "")
        pricing = model.get("pricing") or {}
        prompt_price = pricing.get("prompt", pricing.get("input"))
        completion_price = pricing.get("completion", pricing.get("output"))
        is_free = (
            model_id.endswith("-free") or
            (str(prompt_price or "").strip() in {"0", "0.0", "0.000000000000"} and
             str(completion_price or "").strip() in {"0", "0.0", "0.000000000000"})
        )
        if not model_id or not is_free:
            continue
        rows.append(_row(
            "OpenCode Zen", model_id, "coding; agentic software engineering",
            "Limited-time free model access", "Numeric limits unpublished; shared account capacity",
            "https://opencode.ai/zen", "OPENCODE_ZEN_API_KEY",
            "https://opencode.ai/zen/v1/models", "https://dev.opencode.ai/docs/zen/",
            "Limited-time free access; some providers retain prompts for model improvement",
        ))

    naga_models = _get_json("https://api.naga.ac/v1/models").get("data", [])
    for model in naga_models:
        pricing = model.get("pricing") or {}
        architecture = model.get("architecture") or {}
        if pricing.get("per_input_token") != 0 or pricing.get("per_output_token") != 0:
            continue
        if "text" not in (architecture.get("output_modalities") or []):
            continue
        model_id = str(model.get("id") or "")
        if not model_id or any(token in model_id.lower() for token in ("guard", "safety", "moderation")):
            continue
        rows.append(_row(
            "NagaAI", model_id, "general chat; coding; reasoning",
            "100 requests/day shared across the free account",
            "10 RPM; 100 RPD shared across all NagaAI free models",
            "https://api.naga.ac", "NAGA_API_KEY",
            "https://api.naga.ac/v1/models", "https://naga.ac/pricing",
        ))

    airforce_models = _get_json("https://api.airforce/v1/models").get("data", [])
    for model in airforce_models:
        if model.get("tier") != "free" or not model.get("supports_chat"):
            continue
        if str(model.get("status") or "") not in {"operational", "stable"}:
            continue
        if "text" not in (model.get("output_modalities") or ["text"]):
            continue
        model_id = str(model.get("id") or "")
        if not model_id or any(token in model_id.lower() for token in ("guard", "safety", "moderation")):
            continue
        rows.append(_row(
            "Api.Airforce", model_id, "general chat; coding; reasoning",
            "1,000 requests/day shared across free-tier models",
            "1 RPM; 1,000 RPD shared across all Airforce free models",
            "https://api.airforce", "AIRFORCE_API_KEY",
            "https://api.airforce/v1/models", "https://api.airforce/pricing/",
        ))

    void_models = _get_json("https://api.voidai.app/v1/models").get("data", [])
    for model in void_models:
        if "free" not in (model.get("plan_requirements") or []):
            continue
        if "/v1/chat/completions" not in (model.get("endpoints") or []):
            continue
        model_id = str(model.get("id") or "")
        if not model_id or any(token in model_id.lower() for token in ("guard", "safety", "moderation")):
            continue
        rows.append(_row(
            "VoidAI", model_id, "general chat; coding; tools; reasoning",
            "125,000 credits/day shared across the free account",
            "Daily shared credit pool; token multipliers vary by model",
            "https://api.voidai.app", "VOIDAI_API_KEY",
            "https://api.voidai.app/v1/models", "https://voidai.app/pricing",
        ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--exclude-model-ids", action="store_true")
    args = parser.parse_args()
    with args.catalog.open(encoding="utf-8-sig", newline="") as handle:
        existing = list(csv.DictReader(handle))
    identities = {(r["provider"].lower(), r["model_id"].lower()) for r in existing}
    model_ids = {_normalized(r["model_id"]) for r in existing}
    candidates = [r for r in discover() if
                  (r["provider"].lower(), r["model_id"].lower()) not in identities and
                  (not args.exclude_model_ids or _normalized(r["model_id"]) not in model_ids)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"count": len(candidates), "models": candidates}, indent=2), encoding="utf-8")
    if args.merge and candidates:
        fields = list(existing[0])
        with args.catalog.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(existing + candidates)
    print(json.dumps({"found": len(candidates), "merged": bool(args.merge), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
