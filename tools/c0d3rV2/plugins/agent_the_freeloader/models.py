from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PoolLimit:
    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    requests_per_month: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model_id: str
    modalities: frozenset[str]
    best_at: str
    base_url: str
    endpoint: str
    api_style: str
    api_key_env: str
    pool_ids: tuple[str, ...]
    limits: PoolLimit = field(default_factory=PoolLimit)
    capabilities: dict[str, float] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return f"{self.provider}:{self.model_id}"


_PROVIDER_KEYS = {
    "Google Gemini API": "GEMINI_API_KEY",
    "Groq": "GROQ_API_KEY",
    "Cerebras Inference": "CEREBRAS_API_KEY",
    "SambaNova Cloud": "SAMBANOVA_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "GitHub Models": "GITHUB_TOKEN",
    "Cloudflare Workers AI": "CLOUDFLARE_API_TOKEN",
    "Mistral AI Studio": "MISTRAL_API_KEY",
    "Cohere": "COHERE_API_KEY",
    "Hugging Face Inference Providers": "HF_TOKEN",
    "NVIDIA API Catalog": "NVIDIA_API_KEY",
    "Fireworks AI": "FIREWORKS_API_KEY",
    "BlockRun Free": "",
    "Pollinations.AI": "",
    "Vercel AI Gateway": "AI_GATEWAY_API_KEY",
    "ModelScope API-Inference": "MODELSCOPE_ACCESS_TOKEN",
    "Speka": "SPEKA_API_KEY",
    "Scaleway Generative APIs": "SCW_SECRET_KEY",
    "Zhipu BigModel": "ZHIPU_API_KEY",
    "IO Intelligence": "IOINTELLIGENCE_API_KEY",
    "Alibaba Cloud Model Studio": "DASHSCOPE_API_KEY",
    "SiliconFlow": "SILICONFLOW_API_KEY",
    "Hyperbolic": "HYPERBOLIC_API_KEY",
    "Routeway": "ROUTEWAY_API_KEY",
    "Logfare": "LOGFARE_API_KEY",
    "LLM.kiwi": "LLM_KIWI_API_KEY",
    "LLMRack": "LLMRACK_API_KEY",
    "Kilo Gateway": "",
    "OpenCode Zen": "OPENCODE_ZEN_API_KEY",
    "NagaAI": "NAGA_API_KEY",
    "Api.Airforce": "AIRFORCE_API_KEY",
    "VoidAI": "VOIDAI_API_KEY",
}


def default_catalog_path() -> Path:
    configured = os.getenv("AGENT_FREELOADER_CATALOG", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[4] / "docs" / "free_ai_model_catalog.csv"


def load_catalog(path: str | Path | None = None) -> list[ModelSpec]:
    catalog_path = Path(path) if path else default_catalog_path()
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    specs: list[ModelSpec] = []
    for row in rows:
        provider = (row.get("provider") or "").strip()
        model_id = (row.get("model_id") or "").strip()
        if not _is_callable_model(provider, model_id, row):
            continue
        pool_ids = _pool_ids(provider, model_id)
        specs.append(
            ModelSpec(
                provider=provider,
                model_id=model_id,
                modalities=_modalities(row.get("modalities") or ""),
                best_at=(row.get("best_at") or "").strip(),
                base_url=(row.get("base_url") or "").strip().rstrip("/"),
                endpoint=(row.get("endpoint") or "").strip(),
                api_style=(row.get("api_style") or "").strip(),
                api_key_env=_PROVIDER_KEYS[provider],
                pool_ids=pool_ids,
                limits=_parse_limits(row),
                capabilities=_infer_capabilities(model_id, row),
            )
        )
    return specs


def _is_callable_model(provider: str, model_id: str, row: dict[str, str]) -> bool:
    if provider not in _PROVIDER_KEYS or not model_id:
        return False
    lowered = model_id.lower()
    placeholders = ("catalog model", "catalog ids", "low-tier", "high-tier", "embedding-tier")
    if any(token in lowered for token in placeholders):
        return False
    # C0d3rV2's session contract is conversational text.  Realtime audio,
    # embedding, image-generation, transcription, and rerank endpoints belong
    # behind tools, not behind session.send().
    modalities = (row.get("modalities") or "").lower()
    input_side = modalities.split(" to ", 1)[0]
    if "to text" not in modalities or "text" not in input_side:
        return False
    if provider == "Google Gemini Live API":
        return False
    return True


def _modalities(raw: str) -> frozenset[str]:
    left = raw.lower().split(" to ", 1)[0]
    return frozenset(part.strip() for part in left.split("/") if part.strip()) or frozenset({"text"})


def _pool_ids(provider: str, model_id: str) -> tuple[str, ...]:
    shared = {
        "OpenRouter": "openrouter:free",
        "SambaNova Cloud": "sambanova:free-tier",
        "Cohere": "cohere:trial",
        "Hugging Face Inference Providers": "huggingface:monthly-credit",
        "Cloudflare Workers AI": "cloudflare:neurons",
        "Mistral AI Studio": "mistral:organization",
        "GitHub Models": "github:free",
        "NVIDIA API Catalog": "nvidia:developer",
        "Fireworks AI": "fireworks:trial-credit",
        "BlockRun Free": "blockrun:free",
        "Pollinations.AI": "pollinations:anonymous",
        "Vercel AI Gateway": "vercel:monthly-credit",
        "ModelScope API-Inference": "modelscope:free",
        "Speka": "speka:monthly-credit",
        "Scaleway Generative APIs": "scaleway:new-customer-credit",
        "IO Intelligence": "io-intelligence:daily-credit",
        "Hyperbolic": "hyperbolic:promotional-credit",
        "Routeway": "routeway:free-account",
        "Logfare": "logfare:free-account",
        "LLM.kiwi": "llm-kiwi:free-account",
        "LLMRack": "llmrack:free-account",
        "Kilo Gateway": "kilo:free-ip-pool",
        "OpenCode Zen": "opencode-zen:free-account",
        "NagaAI": "naga:free-account",
        "Api.Airforce": "airforce:free-account",
        "VoidAI": "voidai:free-account",
    }
    # GitHub documents its free API limits per model/tier. Treating all models
    # as one provider-wide pool would discard legitimate independent capacity.
    if provider == "GitHub Models":
        return (f"github:{model_id}",)
    if provider in {"Zhipu BigModel", "Alibaba Cloud Model Studio", "SiliconFlow"}:
        slug = re.sub(r"[^a-z0-9]+", "-", provider.lower()).strip("-")
        return (f"{slug}:{model_id}",)
    if provider in shared:
        return (shared[provider],)
    slug = re.sub(r"[^a-z0-9]+", "-", provider.lower()).strip("-")
    return (f"{slug}:{model_id}",)


def _number(raw: str, suffix: str) -> int | None:
    match = re.search(rf"(\d[\d,]*(?:\.\d+)?)\s*([km]?)\s*{suffix}\b", raw, re.I)
    if not match:
        return None
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000}[match.group(2).lower()]
    return int(float(match.group(1).replace(",", "")) * multiplier)


def _parse_limits(row: dict[str, str]) -> PoolLimit:
    raw = row.get("rate_limits") or ""
    quota = row.get("free_quota") or ""
    monthly = None
    match = re.search(r"(\d[\d,]*)\s*(?:calls|requests)/month", quota, re.I)
    if match:
        monthly = int(match.group(1).replace(",", ""))
    return PoolLimit(
        requests_per_minute=_number(raw, "RPM"),
        requests_per_day=_number(raw, "RPD"),
        requests_per_month=monthly,
        tokens_per_minute=_number(raw, "TPM"),
        tokens_per_day=_number(raw, "TPD"),
    )


def _infer_capabilities(model_id: str, row: dict[str, str]) -> dict[str, float]:
    text = " ".join((model_id, row.get("best_at") or "", row.get("modalities") or "")).lower()
    caps = {
        "general": 0.68,
        "reasoning": 0.58,
        "coding": 0.48,
        "tools": 0.45,
        "multimodal": 0.3,
        "speed": 0.55,
        "multilingual": 0.55,
        "structured": 0.55,
    }
    strength_tokens = ("120b", "235b", "550b", "70b", "pro", "large", "ultra", "r1")
    if any(token in text for token in strength_tokens):
        caps["general"] += 0.17
        caps["reasoning"] += 0.2
        caps["speed"] -= 0.12
    if any(token in text for token in ("reason", "math", "thinking", "deepseek", "qwen3")):
        caps["reasoning"] += 0.2
    if any(token in text for token in ("code", "coding", "software", "devstral", "codestral", "gpt-oss")):
        caps["coding"] += 0.35
    if any(token in text for token in ("agent", "tool", "compound", "orchestration")):
        caps["tools"] += 0.35
    if any(token in text for token in ("image", "multimodal", "vision")):
        caps["multimodal"] += 0.6
    if any(token in text for token in ("instant", "flash", "lite", "mini", "8b", "3b")):
        caps["speed"] += 0.3
    if any(token in text for token in ("multilingual", "translation", "qwen", "gemma", "llama")):
        caps["multilingual"] += 0.2
    if any(token in text for token in ("json", "extract", "classif", "structured")):
        caps["structured"] += 0.25
    return {key: max(0.0, min(1.0, value)) for key, value in caps.items()}


def merge_pool_limits(specs: Iterable[ModelSpec]) -> dict[str, PoolLimit]:
    result: dict[str, PoolLimit] = {}
    for spec in specs:
        for pool_id in spec.pool_ids:
            current = result.get(pool_id)
            result[pool_id] = _conservative_merge(current, spec.limits)
    return result


def _conservative_merge(left: PoolLimit | None, right: PoolLimit) -> PoolLimit:
    if left is None:
        return right

    def minimum(a: int | None, b: int | None) -> int | None:
        values = [value for value in (a, b) if value is not None]
        return min(values) if values else None

    return PoolLimit(
        requests_per_minute=minimum(left.requests_per_minute, right.requests_per_minute),
        requests_per_day=minimum(left.requests_per_day, right.requests_per_day),
        requests_per_month=minimum(left.requests_per_month, right.requests_per_month),
        tokens_per_minute=minimum(left.tokens_per_minute, right.tokens_per_minute),
        tokens_per_day=minimum(left.tokens_per_day, right.tokens_per_day),
    )
