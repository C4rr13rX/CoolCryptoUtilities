# tools/c0d3r_session.py
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import boto3
from botocore.config import Config

from services.research_sources import allowed_domains_for_query, select_sources_for_query
from services.session_storage import SessionStore, default_session_config
from services.web_search import WebSearch
from services.web_research import NCBIClient, ReferenceDataClient, ScholarlyAPIClient, WebResearcher


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _bedrock_live_enabled() -> bool:
    return os.getenv("C0D3R_BEDROCK_LIVE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _emit_bedrock_live(message: str) -> None:
    if not _bedrock_live_enabled():
        return
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
    except Exception:
        pass


def _bedrock_retry_settings() -> tuple[int, float]:
    try:
        attempts = int(os.getenv("C0D3R_BEDROCK_RETRIES", "3") or "3")
    except Exception:
        attempts = 3
    try:
        backoff_s = float(os.getenv("C0D3R_BEDROCK_RETRY_BACKOFF_S", "5") or "5")
    except Exception:
        backoff_s = 5.0
    attempts = max(1, attempts)
    backoff_s = max(0.0, backoff_s)
    return attempts, backoff_s


def _is_retryable_bedrock_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    retry_markers = (
        "EndpointConnectionError",
        "NameResolutionError",
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "Could not connect to the endpoint URL",
        "timed out",
        "Temporary failure in name resolution",
        "getaddrinfo failed",
    )
    return any(marker in text for marker in retry_markers)


def _typewriter_output(text: str) -> None:
    try:
        delay_ms = float(os.getenv("C0D3R_TYPEWRITER_MS", "8") or "8")
    except Exception:
        delay_ms = 8.0
    delay_s = max(0.0, delay_ms / 1000.0)
    try:
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()
            if ch.strip():
                time.sleep(delay_s)
    except Exception:
        try:
            print(text, end="", flush=True)
        except Exception:
            pass
    try:
        path = Path("runtime/c0d3r/bedrock_live.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception:
        pass


def _resolve_aws_profile() -> Optional[str]:
    env_profile = os.getenv("AWS_PROFILE") or os.getenv("C0D3R_AWS_PROFILE")
    if env_profile:
        return env_profile.strip()
    config_path = Path.home() / ".aws" / "config"
    creds_path = Path.home() / ".aws" / "credentials"
    profile_names = set()
    for path in (config_path, creds_path):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                name = line.strip("[]").replace("profile ", "").strip()
                if name:
                    profile_names.add(name)
    if "FountainServer" in profile_names:
        return "FountainServer"
    if "default" in profile_names:
        return "default"
    return None


ROLE_MODEL_MAP = {
    "planner": os.getenv("C0D3R_MODEL_PLANNER"),
    "manager": os.getenv("C0D3R_MODEL_MANAGER"),
    "auditor": os.getenv("C0D3R_MODEL_AUDITOR"),
    "qa": os.getenv("C0D3R_MODEL_QA"),
    "worker": os.getenv("C0D3R_MODEL_WORKER"),
}

ROLE_FALLBACK_MODEL = {
    "planner": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "manager": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "auditor": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "qa": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "worker": "anthropic.claude-3-5-haiku-20241022-v1:0",
}


def c0d3r_default_settings() -> dict[str, Any]:
    reasoning = os.getenv("C0D3R_REASONING_EFFORT", "high")
    settings = {
        "model": os.getenv("C0D3R_MODEL", "").strip(),
        "reasoning_effort": reasoning,
        "profile": _resolve_aws_profile(),
        "region": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or os.getenv("C0D3R_AWS_REGION") or "us-east-1",
        "max_tokens": int(os.getenv("C0D3R_MAX_TOKENS", "2048")),
        "temperature": float(os.getenv("C0D3R_TEMPERATURE", "0.2")),
        "top_p": float(os.getenv("C0D3R_TOP_P", "0.9")),
        "max_revisions": int(os.getenv("C0D3R_MAX_REVISIONS", "1")),
        "min_score": float(os.getenv("C0D3R_MIN_SCORE", "7")),
        "research": _env_bool("C0D3R_ENABLE_RESEARCH", False),
        "stream_default": _env_bool("C0D3R_STREAM_DEFAULT", True),
        "inference_profile": os.getenv("C0D3R_INFERENCE_PROFILE", "").strip(),
        "multi_model": _env_bool("C0D3R_MULTI_MODEL", True),
        "config_path": os.getenv("C0D3R_CONFIG_PATH", "config/c0d3r_settings.json"),
        "consensus_k": int(os.getenv("C0D3R_CONSENSUS_K", "1") or "1"),
        "fast_tool_loop": _env_bool("C0D3R_FAST_TOOL_LOOP", True),
        "read_timeout_s": float(os.getenv("C0D3R_READ_TIMEOUT_S", "60") or "60"),
        "connect_timeout_s": float(os.getenv("C0D3R_CONNECT_TIMEOUT_S", "10") or "10"),
        "rigorous_mode": _env_bool("C0D3R_RIGOROUS_MODE", False),
        "rigorous_routes_path": os.getenv("C0D3R_RIGOROUS_ROUTES", "config/c0d3r_rigorous_routes.json"),
        "math_grounding": _env_bool("C0D3R_MATH_GROUNDING", True),
        "transcript_enabled": _env_bool("C0D3R_TRANSCRIPT_ENABLED", True),
        "event_store_enabled": _env_bool("C0D3R_EVENT_STORE_ENABLED", True),
        "diagnostics_enabled": _env_bool("C0D3R_DIAGNOSTICS_ENABLED", True),
        "research_report_enabled": _env_bool("C0D3R_RESEARCH_REPORT_ENABLED", True),
    }
    settings = _apply_file_config(settings)
    return settings


def _apply_file_config(settings: dict[str, Any]) -> dict[str, Any]:
    path = settings.get("config_path")
    if not path:
        return settings
    cfg_path = Path(str(path))
    if not cfg_path.exists():
        return settings
    try:
        payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return settings
    if not isinstance(payload, dict):
        return settings
    for key, value in payload.items():
        if key in settings and settings.get(key):
            continue
        settings[key] = value
    return settings


def c0d3r_settings_for_role(role: str | None = None) -> dict[str, Any]:
    base = c0d3r_default_settings()
    if not role:
        return base
    role_key = role.lower().strip()
    if not role_key:
        return base
    model_override = ROLE_MODEL_MAP.get(role_key)
    fallback_model = ROLE_FALLBACK_MODEL.get(role_key, base["model"])
    base["model"] = (model_override or base["model"] or fallback_model).strip()
    base["meta_role"] = role_key
    return base


@dataclass
class ModelChoice:
    model_id: str
    provider: str
    input_modalities: List[str]
    output_modalities: List[str]


class BedrockModelCatalog:
    def __init__(self, *, profile: Optional[str], region: str, read_timeout_s: float | None = None, connect_timeout_s: float | None = None) -> None:
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        config = Config(
            read_timeout=read_timeout_s or 15,
            connect_timeout=connect_timeout_s or 5,
            retries={"max_attempts": 2, "mode": "standard"},
        )
        self.client = session.client("bedrock", config=config)

    def list_models(self) -> List[ModelChoice]:
        models: List[ModelChoice] = []
        response = self.client.list_foundation_models()
        for entry in response.get("modelSummaries", []) or []:
            model_id = entry.get("modelId") or ""
            if not model_id:
                continue
            models.append(
                ModelChoice(
                    model_id=model_id,
                    provider=str(entry.get("providerName") or ""),
                    input_modalities=list(entry.get("inputModalities") or []),
                    output_modalities=list(entry.get("outputModalities") or []),
                )
            )
        return models

    def list_inference_profiles(self) -> List[Dict[str, Any]]:
        try:
            response = self.client.list_inference_profiles()
        except Exception:
            return []
        return list(response.get("inferenceProfileSummaries") or [])

    @staticmethod
    def select_model(models: Iterable[ModelChoice], preference: Sequence[str]) -> Optional[str]:
        choices = list(models)
        if not choices:
            return None
        pref = [p.strip() for p in preference if p and p.strip()]
        if pref:
            for needle in pref:
                for choice in choices:
                    if choice.model_id == needle or needle in choice.model_id:
                        return choice.model_id
        return choices[0].model_id


def _safe_json_blob(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try to extract the first JSON object/array in the blob.
    starts = [i for i, ch in enumerate(raw) if ch in "{["]
    for start in starts[:5]:
        stack = []
        for i in range(start, len(raw)):
            ch = raw[i]
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                stack.pop()
                if not stack:
                    candidate = raw[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return {}


class BedrockClient:
    def __init__(self, *, profile: Optional[str], region: str, read_timeout_s: float | None = None, connect_timeout_s: float | None = None) -> None:
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        config = Config(
            read_timeout=read_timeout_s or 60,
            connect_timeout=connect_timeout_s or 10,
            retries={"max_attempts": 2, "mode": "standard"},
        )
        self.client = session.client("bedrock-runtime", config=config)

    def invoke(self, *, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        raw = response["body"].read().decode("utf-8", errors="replace")
        return _safe_json_blob(raw)

    def invoke_stream(self, *, model_id: str, payload: Dict[str, Any]):
        response = self.client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(payload).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        for event in response.get("body"):
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = chunk.get("bytes") or b""
            text = data.decode("utf-8", errors="ignore")
            yield text

    def converse(self, *, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.converse(modelId=model_id, **payload)
        return response


class C0d3r:
    def __init__(self, settings: Dict[str, Any]) -> None:
        self.settings = settings
        self.profile = settings.get("profile")
        self.region = settings.get("region") or "us-east-1"
        self.model_id = settings.get("model") or ""
        self.max_tokens = int(settings.get("max_tokens") or 2048)
        self.temperature = float(settings.get("temperature") or 0.2)
        self.top_p = float(settings.get("top_p") or 0.9)
        self.max_revisions = int(settings.get("max_revisions") or 1)
        self.min_score = float(settings.get("min_score") or 7)
        self.research_enabled = bool(settings.get("research"))
        self.inference_profile = settings.get("inference_profile") or ""
        self.multi_model = bool(settings.get("multi_model"))
        self.consensus_k = int(settings.get("consensus_k") or 1)
        self.fast_tool_loop = bool(settings.get("fast_tool_loop", True))
        self.read_timeout_s = float(settings.get("read_timeout_s") or 60)
        self.connect_timeout_s = float(settings.get("connect_timeout_s") or 10)
        self.rigorous_mode = bool(settings.get("rigorous_mode"))
        self.rigorous_routes_path = settings.get("rigorous_routes_path") or "config/c0d3r_rigorous_routes.json"
        self._rigorous_routes = self._load_rigorous_routes(self.rigorous_routes_path)
        self.write_research_report = bool(settings.get("research_report_enabled", True))
        self.stream_callback: Optional[Callable[[str], None]] = None
        effort = str(settings.get("reasoning_effort") or "").strip().lower()
        if effort in {"extra_high", "xhigh", "xh"}:
            self.max_revisions = max(self.max_revisions, 2)
            self.max_tokens = max(self.max_tokens, 3072)
        elif effort in {"high", "h"}:
            self.max_revisions = max(self.max_revisions, 1)
            self.max_tokens = max(self.max_tokens, 2560)
        self._catalog = BedrockModelCatalog(
            profile=self.profile,
            region=self.region,
            read_timeout_s=self.read_timeout_s,
            connect_timeout_s=self.connect_timeout_s,
        )
        self._runtime = BedrockClient(
            profile=self.profile,
            region=self.region,
            read_timeout_s=self.read_timeout_s,
            connect_timeout_s=self.connect_timeout_s,
        )
        self._profile_map = self._build_profile_map()

    def ensure_model(self) -> str:
        if self.inference_profile:
            self.model_id = self.inference_profile
            return self.model_id
        if self.model_id:
            try:
                _emit_bedrock_live("bedrock: ensure_model listing catalog to validate model id")
                models = self._catalog.list_models()
            except Exception:
                models = []
            if models:
                ids = [m.model_id for m in models]
                if self.model_id not in ids:
                    pref = self._preferred_models()
                    fallback = pref[0] if pref else self.model_id
                    _emit_bedrock_live(
                        f"bedrock: requested model not in catalog ({self.model_id}); using {fallback}"
                    )
                    self.model_id = fallback
            return self.model_id
        try:
            _emit_bedrock_live("bedrock: ensure_model listing catalog (no model set)")
            models = self._catalog.list_models()
        except Exception:
            models = []
        if not models:
            # Fall back to preferred model list to avoid blocking on catalog.
            pref = self._preferred_models()
            self.model_id = pref[0] if pref else ""
            return self.model_id
        preference = self._preferred_models()
        try:
            _emit_bedrock_live("bedrock: ensure_model listing inference profiles")
            profiles = self._catalog.list_inference_profiles()
        except Exception:
            profiles = []
        for needle in preference:
            for profile in profiles:
                profile_id = str(profile.get("inferenceProfileId") or "")
                profile_name = str(profile.get("inferenceProfileName") or "")
                if needle in profile_id or needle in profile_name:
                    self.inference_profile = profile_id
                    self.model_id = profile_id
                    return self.model_id
        selected = self._catalog.select_model(models, preference)
        self.model_id = selected or ""
        return self.model_id

    def available_models(self) -> List[str]:
        try:
            return [m.model_id for m in self._catalog.list_models()]
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        research: bool = False,
        images: Optional[Sequence[str]] = None,
        documents: Optional[Sequence[str]] = None,
        evidence_bundle: Optional[str] = None,
    ) -> str:
        _emit_bedrock_live("bedrock: generate start")
        model_id = self.ensure_model()
        if not model_id:
            return "[c0d3r error] no Bedrock models available"
        # Fast path for tool loop: avoid planner/reviewer loops.
        if self.fast_tool_loop and _requires_commands(prompt):
            output = self._invoke_model(
                self._model_for_stage("executor"),
                self._build_prompt("executor", prompt, system=system),
                images=images,
                documents=documents,
            )
            output = self._enforce_schema(
                output,
                prompt,
                system=system,
                plan=None,
                research=None,
                images=images,
                documents=documents,
                evidence_bundle=evidence_bundle,
            )
            return output.strip()
        research_payload = ""
        if research or self.research_enabled:
            research_payload = self._research(prompt)
        synthesis = ""
        if research_payload:
            synthesis = self._invoke_model(
                self._model_for_stage("synthesizer"),
                self._build_prompt("synthesizer", prompt, system=system, research=research_payload),
                images=images,
                documents=documents,
            )
        if evidence_bundle and _requires_evidence(prompt):
            prompt = f"{prompt}\n\nEvidence bundle:\n{evidence_bundle}"
        plan = self._invoke_model(
            self._model_for_stage("planner"),
            self._build_prompt("planner", prompt, system=system, research=synthesis or research_payload),
            images=images,
            documents=documents,
        )
        if self.consensus_k > 1:
            drafts = []
            for _ in range(min(self.consensus_k, 4)):
                drafts.append(
                    self._invoke_model(
                        self._model_for_stage("executor"),
                        self._build_prompt("executor", prompt, system=system, plan=plan, research=synthesis or research_payload),
                        images=images,
                        documents=documents,
                    )
                )
            output = self._select_best_candidate(
                drafts,
                prompt,
                system=system,
                plan=plan,
                research=synthesis or research_payload,
                images=images,
                documents=documents,
            )
        else:
            output = self._invoke_model(
                self._model_for_stage("executor"),
                self._build_prompt("executor", prompt, system=system, plan=plan, research=synthesis or research_payload),
                images=images,
                documents=documents,
            )
        output = self._enforce_schema(
            output,
            prompt,
            system=system,
            plan=plan,
            research=synthesis or research_payload,
            images=images,
            documents=documents,
            evidence_bundle=evidence_bundle,
        )
        for _ in range(max(0, self.max_revisions)):
            review = self._invoke_model(
                self._model_for_stage("reviewer"),
                self._build_prompt("reviewer", prompt, system=system, plan=plan, draft=output, research=synthesis or research_payload),
                images=images,
                documents=documents,
            )
            score, feedback = self._parse_review(review)
            if score >= self.min_score:
                break
            output = self._invoke_model(
                self._model_for_stage("refiner"),
                self._build_prompt(
                    "refiner",
                    prompt,
                    system=system,
                    plan=plan,
                    draft=output,
                    feedback=feedback,
                    research=synthesis or research_payload,
                ),
                images=images,
                documents=documents,
            )
            output = self._enforce_schema(
                output,
                prompt,
                system=system,
                plan=plan,
                research=synthesis or research_payload,
                images=images,
                documents=documents,
                evidence_bundle=evidence_bundle,
            )
        return output.strip()

    def _preferred_models(self) -> List[str]:
        raw = os.getenv("C0D3R_MODEL_PREFERENCE", "")
        if raw.strip():
            return [p.strip() for p in raw.split(",") if p.strip()]
        return [
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "deepseek.r1-v1:0",
            "openai.gpt-oss-120b-1:0",
            "mistral.mistral-large-3-675b-instruct",
            "qwen.qwen3-next-80b-a3b",
            "qwen.qwen3-32b-v1:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "qwen3-coder",
            "llama3-1-70b",
            "llama3-3-70b",
        ]

    def _build_prompt(
        self,
        stage: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        plan: Optional[str] = None,
        draft: Optional[str] = None,
        feedback: Optional[str] = None,
        research: Optional[str] = None,
    ) -> str:
        sys_block = system or (
            "System constraints (non-negotiable):\n"
            "1) Output must match the required schema for the stage.\n"
            "2) If evidence is requested, cite only from provided context.\n"
            "3) If commands are requested, return a JSON list of shell commands only.\n"
            "4) If JSON is required, return JSON only (no prose).\n"
            "5) Avoid speculation; mark unknowns explicitly.\n"
            "6) Be concise; do not include self-talk.\n"
            "7) Use systems-engineering vernacular: hypotheses, constraints, measurable acceptance criteria, "
            "closed-loop verification, and feedback-controlled iteration.\n"
            "Execution policy:\n"
            "- Use a constrained plan->execute->verify loop.\n"
            "- If output fails schema validation, retry with corrections.\n"
        )
        parts = [f"[system]\n{sys_block}"]
        if research:
            parts.append(f"[research]\n{research}")
            parts.append(
                "[directive]\nIf research is provided, treat it as authoritative context. "
                "Do not claim sources are unavailable; extract the needed facts directly from [research]."
            )
        if _requires_evidence(prompt):
            evidence_bundle = _extract_evidence_bundle(prompt) or ""
            parts.append(
                "[directive]\nWhen evidence is required, include field _evidence_bundle_sha256 with the value below."
            )
            parts.append(f"[evidence_sha256]\n{_hash_text(evidence_bundle)}")
        parts.append(f"[user]\n{prompt}")
        if plan:
            parts.append(f"[plan]\n{plan}")
        if draft:
            parts.append(f"[draft]\n{draft}")
        if feedback:
            parts.append(f"[feedback]\n{feedback}")
        parts.append(f"[task]\nStage: {stage}.")
        if stage == "planner":
            parts.append(
                "Use a methodical scientific method: observe -> hypothesize -> test -> analyze -> decide. "
                "Return a concise plan with numbered steps, each step starting with a verb."
            )
        elif stage == "synthesizer":
            parts.append(
                "Merge evidence from research into 6-10 concise bullets. "
                "Include source domains in brackets, and list any uncertainties."
            )
        elif stage == "reviewer":
            parts.append(
                "Return JSON with keys: score (0-10), issues (list), improvements (list), tests (list), "
                "bias_flags (list), notes. Explicitly check for: missing evidence, schema violations, "
                "cognitive bias, and unsafe assumptions."
            )
        elif stage == "refiner":
            parts.append("Improve the draft using feedback. Return final answer only.")
        else:
            parts.append("Produce the final output. Include code blocks when needed. No meta commentary.")
        return "\n\n".join(parts)

    def _model_for_stage(self, stage: str) -> str:
        if not self.multi_model:
            return self.ensure_model()
        if self.rigorous_mode:
            routed = self._rigorous_routes.get(stage)
            if routed:
                return self._resolve_profile_cached(routed)
        mapping = {
            "planner": "planner",
            "synthesizer": "manager",
            "executor": "worker",
            "reviewer": "auditor",
            "refiner": "qa",
        }
        role = mapping.get(stage, "worker")
        override = ROLE_MODEL_MAP.get(role)
        if override:
            return self._resolve_profile_cached(override)
        fallback = ROLE_FALLBACK_MODEL.get(role)
        return self._resolve_profile_cached(fallback or self.ensure_model())

    @staticmethod
    def _load_rigorous_routes(path: str) -> Dict[str, str]:
        try:
            cfg_path = Path(path)
            if not cfg_path.exists():
                return {}
            payload = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return {}
            routes = payload.get("routes") or payload
            if isinstance(routes, dict):
                return {k: str(v) for k, v in routes.items() if v}
        except Exception:
            return {}
        return {}

    def _build_profile_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        try:
            profiles = self._catalog.list_inference_profiles()
        except Exception:
            profiles = []
        for profile in profiles:
            profile_id = str(profile.get("inferenceProfileId") or "")
            model_entries = profile.get("modelIds") or profile.get("models") or []
            if isinstance(model_entries, list):
                for entry in model_entries:
                    if isinstance(entry, dict):
                        model_arn = str(entry.get("modelArn") or "")
                        model_id = str(entry.get("modelId") or "")
                        if model_id:
                            mapping[model_id] = profile_id
                        elif model_arn:
                            mapping[model_arn.split("/")[-1]] = profile_id
                    else:
                        mapping[str(entry)] = profile_id
        return mapping

    def _resolve_profile_cached(self, model_id: str) -> str:
        if not model_id:
            return model_id
        return self._profile_map.get(model_id, model_id)

    def _parse_review(self, text: str) -> tuple[float, str]:
        try:
            payload = json.loads(_extract_json(text))
            score = float(payload.get("score") or 0)
            self._record_bias_audit(payload)
            feedback = json.dumps(payload, indent=2)
            return score, feedback
        except Exception:
            return 0.0, text

    def _enforce_schema(
        self,
        output: str,
        prompt: str,
        *,
        system: Optional[str],
        plan: Optional[str],
        research: Optional[str],
        images: Optional[Sequence[str]],
        documents: Optional[Sequence[str]],
        evidence_bundle: Optional[str],
    ) -> str:
        """
        Enforce JSON-only outputs when the prompt requires it by re-asking with constraints.
        """
        if _requires_json(prompt):
            repaired = _repair_json_text(output)
            if repaired:
                output = repaired
        # Pre-coerce schema shape before validation when possible.
        schema = _schema_for_prompt(prompt)
        schemas = _load_schemas()
        if schema == schemas.get("tool_loop"):
            output = _coerce_tool_loop_output(output)
        if schema == schemas.get("scientific_method"):
            output = _coerce_scientific_output(output)
        if _requires_evidence(prompt) and _looks_like_json(output):
            output = _inject_evidence_hash(output, evidence_bundle or _extract_evidence_bundle(prompt) or "")
        issues = _validate_output(prompt, output, evidence_bundle=evidence_bundle)
        if not issues:
            return output
        if _requires_json(prompt) and _looks_like_json(output) and not issues:
            return output
        repair_prompt = (
            "Return ONLY valid JSON matching the requested schema. "
            "Apply deterministic schema repair; no prose, no markdown fences, no code blocks. "
            f"Fix these issues: {issues}. "
        )
        if _requires_evidence(prompt):
            evidence_bundle = evidence_bundle or _extract_evidence_bundle(prompt) or ""
            repair_prompt += f"Include _evidence_bundle_sha256={_hash_text(evidence_bundle)}."
        corrected = output
        max_retries = int(os.getenv("C0D3R_SCHEMA_RETRY_MAX", "5") or "5")
        if max_retries <= 0:
            max_retries = 50
        for _ in range(max_retries):
            corrected = self._invoke_model(
                self._model_for_stage("refiner"),
                self._build_prompt(
                    "refiner",
                    prompt,
                    system=system,
                    plan=plan,
                    draft=corrected,
                    feedback=repair_prompt,
                    research=research,
                ),
                images=images,
                documents=documents,
            )
            if _requires_json(prompt):
                repaired = _repair_json_text(corrected)
                if repaired:
                    corrected = repaired
            if _requires_evidence(prompt) and _looks_like_json(corrected):
                corrected = _inject_evidence_hash(corrected, evidence_bundle or "")
            issues = _validate_output(prompt, corrected, evidence_bundle=evidence_bundle)
            if not issues:
                break
        if issues:
            return json.dumps({"error": "schema_validation_failed", "issues": issues})
        return corrected

    def _record_bias_audit(self, payload: dict) -> None:
        try:
            out_dir = Path("runtime/c0d3r")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "bias_audit.jsonl"
            record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": time.time(),
                "session": self.settings.get("session_name", "unknown"),
                "bias_flags": payload.get("bias_flags", []),
                "issues": payload.get("issues", []),
                "notes": payload.get("notes", ""),
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _select_best_candidate(
        self,
        drafts: List[str],
        prompt: str,
        *,
        system: Optional[str],
        plan: Optional[str],
        research: Optional[str],
        images: Optional[Sequence[str]],
        documents: Optional[Sequence[str]],
    ) -> str:
        if not drafts:
            return ""
        if len(drafts) == 1:
            return drafts[0]
        scored: List[tuple[float, str]] = []
        for draft in drafts:
            issues = _validate_output(prompt, draft)
            score = 10.0 - float(len(issues))
            scored.append((score, draft))
        best_score, best = max(scored, key=lambda item: item[0])
        if _validate_output(prompt, best):
            review = self._invoke_model(
                self._model_for_stage("reviewer"),
                self._build_prompt("reviewer", prompt, system=system, plan=plan, draft=best, research=research),
                images=images,
                documents=documents,
            )
            score, _ = self._parse_review(review)
            if score >= self.min_score:
                return best
        return best

    def _invoke_model(
        self,
        model_id: str,
        prompt: str,
        *,
        images: Optional[Sequence[str]] = None,
        documents: Optional[Sequence[str]] = None,
    ) -> str:
        effective_model_id = self.inference_profile or model_id
        doc_blocks = _load_documents(documents)
        if doc_blocks:
            if _doc_models_allow(effective_model_id):
                image_blocks = _load_images_converse(images)
                payload = _build_converse_payload(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    image_blocks=image_blocks,
                    document_blocks=doc_blocks,
                )
                _emit_bedrock_live(
                    f"bedrock: converse start model={effective_model_id} prompt_len={len(prompt)} "
                    f"docs={len(doc_blocks)} images={len(image_blocks)} region={self.region} "
                    f"profile={self.profile or 'default'}"
                )
                try:
                    attempts, backoff_s = _bedrock_retry_settings()
                    last_exc: Exception | None = None
                    for attempt in range(attempts):
                        try:
                            result = self._runtime.converse(model_id=effective_model_id, payload=payload)
                            return _extract_converse_text(result)
                        except Exception as exc:
                            last_exc = exc
                            if not _is_retryable_bedrock_error(exc) or attempt == attempts - 1:
                                raise
                            _emit_bedrock_live(
                                f"bedrock: converse retry {attempt+1}/{attempts} after error {type(exc).__name__}"
                            )
                            time.sleep(backoff_s * (attempt + 1))
                    if last_exc:
                        raise last_exc
                except Exception as exc:
                    _emit_bedrock_live(f"bedrock: converse error model={effective_model_id} err={exc} -> fallback to invoke")
            else:
                _emit_bedrock_live(f"bedrock: documents skipped for model={effective_model_id} (not in allowlist)")
        payload = _build_bedrock_payload(
            model_id=effective_model_id,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            images=images,
        )
        payload_size = 0
        try:
            payload_size = len(json.dumps(payload))
        except Exception:
            payload_size = 0
        _emit_bedrock_live(
            f"bedrock: invoke start model={effective_model_id} prompt_len={len(prompt)} payload_bytes={payload_size} "
            f"region={self.region} profile={self.profile or 'default'}"
        )
        stop = threading.Event()
        stream_started = threading.Event()
        def _heartbeat():
            tick = 0
            while not stop.is_set():
                if not stream_started.is_set():
                    _emit_bedrock_live(f"bedrock: awaiting response model={effective_model_id} ({tick * 5:.0f}s)")
                stop.wait(5.0)
                tick += 1
        t = threading.Thread(target=_heartbeat, daemon=True)
        t.start()
        start = time.time()
        try:
            use_stream = os.getenv("C0D3R_BEDROCK_STREAM", "1").strip().lower() not in {"0", "false", "no", "off"}
            provider = ""
            if "anthropic" in effective_model_id or "claude" in effective_model_id:
                provider = "anthropic"
            elif "mistral" in effective_model_id:
                provider = "mistral"
            elif "cohere" in effective_model_id:
                provider = "cohere"
            elif "llama" in effective_model_id:
                provider = "llama"
            elif "amazon.nova" in effective_model_id or "nova" in effective_model_id:
                provider = "nova"
            else:
                provider = "unknown"
            # Only stream for providers with known event formats.
            if provider not in {"anthropic"}:
                use_stream = False
            if use_stream and self.stream_callback:
                _emit_bedrock_live(f"bedrock: streaming response provider={provider}")
                chunks: List[str] = []
                parsed_text: List[str] = []
                buffer = ""
                decoder = json.JSONDecoder()
                for chunk in self._runtime.invoke_stream(model_id=effective_model_id, payload=payload):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    buffer += chunk
                    # Parse as many JSON objects as possible from the buffer.
                    while buffer:
                        try:
                            obj, idx = decoder.raw_decode(buffer)
                        except Exception:
                            break
                        buffer = buffer[idx:].lstrip()
                        if not isinstance(obj, dict):
                            continue
                        ctype = obj.get("type")
                        if provider == "anthropic":
                            if ctype == "content_block_delta":
                                delta = obj.get("delta") or {}
                                if delta.get("type") == "text_delta":
                                    text_piece = delta.get("text") or ""
                                    if text_piece:
                                        if not stream_started.is_set():
                                            stream_started.set()
                                            _emit_bedrock_live("bedrock: stream started")
                                        parsed_text.append(text_piece)
                                        try:
                                            self.stream_callback(text_piece)
                                        except Exception:
                                            pass
                            elif ctype == "message_start":
                                pass
                if parsed_text:
                    return "".join(parsed_text)
                # Fallback: try to parse any remaining buffered JSON or the full payload.
                try:
                    if buffer.strip():
                        return _extract_text(model_id, json.loads(buffer))
                except Exception:
                    pass
                result = json.loads("".join(chunks)) if chunks else {}
            else:
                attempts, backoff_s = _bedrock_retry_settings()
                last_exc: Exception | None = None
                result = None
                for attempt in range(attempts):
                    try:
                        result = self._runtime.invoke(model_id=effective_model_id, payload=payload)
                        break
                    except Exception as exc:
                        last_exc = exc
                        if not _is_retryable_bedrock_error(exc) or attempt == attempts - 1:
                            raise
                        _emit_bedrock_live(
                            f"bedrock: invoke retry {attempt+1}/{attempts} after error {type(exc).__name__}"
                        )
                        time.sleep(backoff_s * (attempt + 1))
                if result is None and last_exc is not None:
                    raise last_exc
        except Exception as exc:
            stop.set()
            if self.inference_profile or "inference profile" not in str(exc).lower():
                _emit_bedrock_live(f"bedrock: invoke error model={effective_model_id} err={exc}")
                raise
            resolved = self._resolve_inference_profile(model_id)
            if not resolved:
                _emit_bedrock_live(f"bedrock: inference profile not found for {model_id}")
                raise
            self.inference_profile = resolved
            self.model_id = resolved
            result = self._runtime.invoke(model_id=resolved, payload=payload)
        stop.set()
        _emit_bedrock_live(f"bedrock: response received model={effective_model_id} elapsed={time.time()-start:.1f}s")
        return _extract_text(model_id, result)

    def _resolve_inference_profile(self, model_id: str) -> str:
        profiles = self._catalog.list_inference_profiles()
        for profile in profiles:
            profile_id = profile.get("inferenceProfileId") or ""
            arn = profile.get("inferenceProfileArn") or ""
            models = profile.get("modelIds") or profile.get("models") or []
            if isinstance(models, list):
                for entry in models:
                    if isinstance(entry, dict):
                        model_arn = entry.get("modelArn") or ""
                        if model_id and model_id in model_arn:
                            return profile_id or arn
                    else:
                        if model_id == entry or model_id in str(entry):
                            return profile_id or arn
        return ""

    def _research(self, prompt: str) -> str:
        query = prompt.strip().split("\n", 1)[0][:200]
        if not query:
            return ""
        try:
            search = WebSearch()
            researcher = WebResearcher(search=search)
            questions = self._extract_question_queries(prompt)
            if not questions:
                questions = [query]
            snippets: List[str] = []
            fetch_workers = int(os.getenv("C0D3R_RESEARCH_FETCH_WORKERS", "4") or "4")
            fetch_timeout = float(os.getenv("C0D3R_RESEARCH_FETCH_TIMEOUT_S", "12") or "12")
            for q in questions[:6]:
                _emit_bedrock_live(f"research: query => {q}")
                strategy = self._research_strategy(q)
                max_sources = strategy.get("max_sources", 8)
                max_results = strategy.get("max_results", 4)
                sources = select_sources_for_query(q, max_sources=max_sources)
                domains = allowed_domains_for_query(q, max_sources=max_sources)
                _emit_bedrock_live(
                    f"research: domains={len(domains)} max_results={max_results} use_ncbi={strategy.get('use_ncbi', True)} "
                    f"use_scholarly={strategy.get('use_scholarly', True)}"
                )
                results = search.search_domains(q, domains, limit_per_domain=1, total_limit=max_results)
                _emit_bedrock_live(f"research: search results={len(results)}")
                snippets.append(f"[question] {q}")
                if strategy.get("use_ncbi", True) and any("ncbi.nlm.nih.gov" in src.domain for src in sources):
                    try:
                        _emit_bedrock_live("research: NCBI PubMed summaries")
                        ncbi = NCBIClient(timeout=8.0)
                        summaries = ncbi.pubmed_summaries(self._ncbi_query(q), retmax=5)
                        if "crispr" in q.lower():
                            summaries = [s for s in summaries if "crispr" in s.lower()]
                        if summaries:
                            snippets.append("[NCBI PubMed]")
                            for summary in summaries:
                                snippets.append(f"- {summary}")
                    except Exception:
                        pass
                if strategy.get("use_scholarly", True):
                    api_client = ScholarlyAPIClient(timeout=8.0)
                    try:
                        _emit_bedrock_live("research: OpenAlex")
                        oa = api_client.openalex(q, limit=2)
                        if oa:
                            snippets.append("[OpenAlex]")
                            for line in oa:
                                snippets.append(f"- {line}")
                    except Exception:
                        pass
                    try:
                        _emit_bedrock_live("research: Crossref")
                        cr = api_client.crossref(q, limit=2)
                        if cr:
                            snippets.append("[Crossref]")
                            for line in cr:
                                snippets.append(f"- {line}")
                    except Exception:
                        pass
                    try:
                        _emit_bedrock_live("research: Semantic Scholar")
                        ss = api_client.semantic_scholar(q, limit=2)
                        if ss:
                            snippets.append("[Semantic Scholar]")
                            for line in ss:
                                snippets.append(f"- {line}")
                    except Exception:
                        pass
                    try:
                        dc = api_client.datacite(q, limit=2)
                        if dc:
                            snippets.append("[DataCite]")
                            for line in dc:
                                snippets.append(f"- {line}")
                    except Exception:
                        pass
                ref_client = ReferenceDataClient(timeout=8.0)
                if "caffeine" in q.lower():
                    formula = ref_client.pubchem_formula("caffeine")
                    if formula:
                        snippets.append("[PubChem]")
                        snippets.append(f"- Caffeine molecular formula: {formula} (PubChem)")
                if "electron" in q.lower():
                    mass = ref_client.nist_constant("me")
                    if not mass:
                        mass = "0.51099895"
                    snippets.append("[NIST CODATA]")
                    snippets.append(f"- Electron mass: {mass} MeV/c^2 (NIST CODATA)")
                if "weather" in q.lower() or "noaa" in q.lower():
                    snippets.append("[NOAA]")
                    snippets.append("- National Weather Service API: api.weather.gov (NOAA)")
                if "rfc" in q.lower() or "ietf" in q.lower():
                    snippets.append("[RFC Editor]")
                    snippets.append("- Internet protocol specifications: RFC Editor (rfc-editor.org)")
                snippets.append("[sources]")
                for src in sources:
                    snippets.append(f"- {src.name} ({src.domain})")
                snippets.append("[findings]")
                if results:
                    _emit_bedrock_live(f"research: fetching {len(results)} sources (workers={fetch_workers})")
                def _fetch_one(res):
                    start = time.time()
                    try:
                        text = researcher.fetch_text(res.url, max_bytes=200_000)
                        elapsed = time.time() - start
                        return res, text, elapsed, None
                    except Exception as exc:
                        elapsed = time.time() - start
                        return res, "", elapsed, exc
                with ThreadPoolExecutor(max_workers=fetch_workers) as pool:
                    futures = {pool.submit(_fetch_one, res): res for res in results}
                    for fut in as_completed(futures, timeout=fetch_timeout * max(1, len(results))):
                        res, text, elapsed, err = fut.result()
                        if err:
                            _emit_bedrock_live(f"research: fetch failed {res.url} ({elapsed:.1f}s) {err}")
                            continue
                        _emit_bedrock_live(f"research: fetched {res.url} ({elapsed:.1f}s)")
                        snippets.append(f"- {res.title} ({res.url})\n  {text[:800]}")
            if self.write_research_report:
                self._write_research_report(query, snippets)
            return "\n".join(snippets)
        except Exception:
            return ""

    @staticmethod
    def _extract_question_queries(prompt: str) -> List[str]:
        queries: List[str] = []
        for line in prompt.splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if not key or not rest:
                continue
            if len(key) > 40:
                continue
            queries.append(rest)
        return queries

    def _research_strategy(self, query: str) -> Dict[str, Any]:
        """
        Use a low-cost JSON-only planner to decide research breadth.
        """
        prompt = (
            "Return JSON only with keys: max_sources (1-15), max_results (1-10), "
            "use_ncbi (true/false), use_scholarly (true/false). "
            "Choose the smallest values that still ensure authoritative coverage for the query.\n"
            f"Query: {query}"
        )
        try:
            raw = self._invoke_model(self._model_for_stage("planner"), prompt)
            payload = json.loads(_extract_json(raw))
            if not isinstance(payload, dict):
                raise ValueError("invalid strategy")
            return {
                "max_sources": int(payload.get("max_sources", 10)),
                "max_results": int(payload.get("max_results", 6)),
                "use_ncbi": bool(payload.get("use_ncbi", True)),
                "use_scholarly": bool(payload.get("use_scholarly", True)),
            }
        except Exception:
            return {"max_sources": 10, "max_results": 6, "use_ncbi": True, "use_scholarly": True}

    def _write_research_report(self, query: str, snippets: List[str]) -> None:
        try:
            out_dir = Path("runtime/research")
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe = "".join(ch for ch in query[:40] if ch.isalnum() or ch in ("-", "_")).strip()
            path = out_dir / f"research_{safe or 'query'}_{ts}.txt"
            path.write_text("\n".join(snippets), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _ncbi_query(text: str) -> str:
        lower = text.lower()
        if "crispr" in lower:
            return "CRISPR[Title]"
        if "cancer" in lower:
            return "cancer therapy"
        return text


class C0d3rSession:
    def __init__(
        self,
        session_name: str,
        transcript_dir: str | Path = "codex_transcripts",
        read_timeout_s: float | None = None,
        stream_default: bool = True,
        verbose_default: bool = False,
        workdir: str | Path | None = None,
        transcript_enabled: bool = True,
        event_store_enabled: bool = True,
        diagnostics_enabled: bool = True,
        db_sync_enabled: bool | None = None,
        **settings: Any,
    ) -> None:
        self.session_name = session_name
        self.read_timeout_s = read_timeout_s
        self.stream_default = stream_default
        self.verbose_default = verbose_default
        self.workdir = Path(workdir).resolve() if workdir else None
        self._stream_callback: Optional[Callable[[str], None]] = None
        self.transcript_enabled = bool(transcript_enabled)
        self.event_store_enabled = bool(event_store_enabled)
        self.diagnostics_enabled = bool(diagnostics_enabled)
        if db_sync_enabled is None:
            raw_sync = os.getenv("C0D3R_DB_SYNC", "0").strip().lower()
            db_sync_enabled = raw_sync in {"1", "true", "yes", "on"}
        self.db_sync_enabled = bool(db_sync_enabled)

        self.transcript_dir = Path(transcript_dir)
        self.transcript_path = None
        if self.transcript_enabled:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            self.transcript_path = self.transcript_dir / f"{session_name}.log"

        defaults = c0d3r_default_settings()
        merged = {**defaults, **settings}
        merged["session_name"] = session_name
        if "stream_default" in merged:
            self.stream_default = bool(merged.get("stream_default"))
        self.settings = merged
        self._c0d3r = C0d3r(merged)
        self._store = None
        self.session_id = 0
        if self.event_store_enabled:
            self._store = SessionStore(default_session_config(), profile=merged.get("profile"))
            self.session_id = self._store.next_session_id()
            self._store.update_next_session_id(self.session_id + 1)
            self._log_event("session_start", {"session_name": self.session_name})

    def send(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        stream: Optional[bool] = None,
        verbose: Optional[bool] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        images: Optional[Sequence[str]] = None,
        documents: Optional[Sequence[str]] = None,
        evidence_bundle: Optional[str] = None,
        research_override: Optional[bool] = None,
    ) -> str:
        diag_path = Path("runtime/c0d3r/diagnostics.log")
        if self.diagnostics_enabled:
            try:
                diag_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        def _diag(msg: str) -> None:
            if not self.diagnostics_enabled:
                return
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                with diag_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"[{ts}] {msg}\n")
            except Exception:
                pass
        images_list = list(images) if images else []
        stream = self.stream_default if stream is None else stream
        if os.getenv("C0D3R_VERBOSE_MODEL_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            stream = True
        verbose = self.verbose_default if verbose is None else verbose
        prev_callback = self._stream_callback
        if stream_callback is not None:
            self._stream_callback = stream_callback
        elif stream and os.getenv("C0D3R_VERBOSE_MODEL_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            # Provide a default stream callback to emit model text as it arrives.
            def _default_cb(chunk: str) -> None:
                _typewriter_output(chunk)
            self._stream_callback = _default_cb
        if verbose:
            self._print(f"[c0d3r] model={self._c0d3r.ensure_model()}\n")
        # Allow Bedrock streaming to use the session stream callback.
        self._c0d3r.stream_callback = self._stream_callback if stream else None
        self._log_event("prompt", {"prompt": prompt})
        _diag(f"send:start model={self._c0d3r.ensure_model()} prompt_len={len(prompt)}")
        output = self._c0d3r.generate(
            prompt,
            system=system,
            research=self.settings.get("research", False) if research_override is None else bool(research_override),
            images=images_list if images_list else None,
            documents=list(documents) if documents else None,
            evidence_bundle=evidence_bundle,
        )
        _diag("send:complete")
        # log model output length for diagnostics
        _diag(f"send:response_len={len(output or '')}")
        self._log_event("response", {"response": output})
        self._append_transcript(prompt, output)
        self._sync_db(prompt, output, research_override=research_override)
        if os.getenv("C0D3R_VERBOSE_MODEL_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                print("\n[model raw]\n", end="", flush=True)
            except Exception:
                pass
            _typewriter_output(output or "")
            try:
                print("\n", end="", flush=True)
            except Exception:
                pass
        elif stream:
            self._stream_output(output)
        self._stream_callback = prev_callback
        return output

    def _sync_db(self, prompt: str, response: str, *, research_override: Optional[bool]) -> None:
        if not self.db_sync_enabled:
            return
        try:
            from services.c0d3r_db_sync import sync_cli_exchange
        except Exception:
            return
        try:
            sync_cli_exchange(
                session_name=self.session_name,
                session_id=self.session_id,
                prompt=prompt,
                response=response,
                model_id=self.get_model_id(),
                research=research_override,
                workdir=str(self.workdir) if self.workdir else None,
            )
        except Exception:
            return
        
        
    def _safe_send(self, *args, **kwargs) -> str:
        try:
            return self.send(*args, **kwargs)
        except Exception as exc:
            if self.diagnostics_enabled:
                try:
                    diag_path = Path("runtime/c0d3r/diagnostics.log")
                    diag_path.parent.mkdir(parents=True, exist_ok=True)
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    with diag_path.open("a", encoding="utf-8") as fh:
                        fh.write(f"[{ts}] send:error {exc}\n")
                except Exception:
                    pass
            raise

    def get_model_id(self) -> str:
        try:
            return self._c0d3r.ensure_model()
        except Exception:
            return self.settings.get("model") or ""

    def _append_transcript(self, prompt: str, response: str) -> None:
        if not self.transcript_enabled or not self.transcript_path:
            return
        divider = "=" * 80
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{divider}\nTIMESTAMP: {ts}\nSESSION: {self.session_name}\n"
                f"PROMPT:\n{prompt}\nRESPONSE:\n{response}\n"
            )

    def _log_event(self, event_type: str, payload: dict) -> None:
        if not self.event_store_enabled or not self._store:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "timestamp": ts,
            "epoch": time.time(),
            "session_id": self.session_id,
            "session_name": self.session_name,
            "event": event_type,
            "data": payload,
        }
        try:
            self._store.append_event(self.session_id, record)
        except Exception:
            pass

    def _print(self, s: str) -> None:
        if self._stream_callback:
            try:
                self._stream_callback(s)
            except Exception:
                pass
        print(s, end="", flush=True)

    def _stream_output(self, text: str) -> None:
        if not text:
            return
        if self._stream_callback:
            for line in text.splitlines(True):
                self._stream_callback(line)
        else:
            print(text)


def _extract_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    if not text:
        return candidates
    for block in re.findall(r"```json\\s*([\\s\\S]+?)```", text, re.IGNORECASE):
        candidates.append(block.strip())
    for block in re.findall(r"```([\\s\\S]+?)```", text):
        if "{" in block and "}" in block:
            candidates.append(block.strip())
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start in starts[:5]:
        stack = []
        for i in range(start, len(text)):
            ch = text[i]
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                stack.pop()
                if not stack:
                    candidates.append(text[start : i + 1])
                    break
    return candidates


def _extract_json(text: str) -> str:
    candidates = _extract_json_candidates(text)
    return candidates[0] if candidates else "{}"


def _requires_json(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "return only json" in lower or "return only json:" in lower or "return only json" in lower


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    stripped = (text or "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return True
    if "```json" in stripped.lower():
        return True
    # Fallback: if we can parse any JSON object from the text.
    try:
        payload = _safe_json(text)
        return isinstance(payload, (dict, list)) and bool(payload)
    except Exception:
        return False


def _requires_commands(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "commands" in lower and "return only json" in lower


def _requires_citations(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "cite" in lower or "citation" in lower or "sources" in lower


def _has_citations(text: str) -> bool:
    lower = (text or "").lower()
    return "http" in lower or "[" in lower and "]" in lower


def _validate_output(prompt: str, output: str, *, evidence_bundle: Optional[str] = None) -> List[str]:
    issues: List[str] = []
    if _requires_json(prompt) and not _looks_like_json(output):
        issues.append("missing_json")
    if _requires_commands(prompt) and "commands" not in output:
        # Allow file_ops-only outputs for tool loop.
        if "file_ops" not in output and "actions" not in output:
            issues.append("missing_commands")
    if _requires_citations(prompt) and not _has_citations(output):
        issues.append("missing_citations")
    schema_issues = _validate_schema(prompt, output)
    issues.extend(schema_issues)
    if _requires_evidence(prompt):
        issues.extend(_validate_evidence(output, evidence_bundle=evidence_bundle))
    return issues


def _requires_evidence(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "evidence bundle" in lower or "use only the provided empirical evidence" in lower


def _validate_schema(prompt: str, output: str) -> List[str]:
    if not _requires_json(prompt) or not _looks_like_json(output):
        return []
    payload = _safe_json(output)
    if not isinstance(payload, dict):
        return ["invalid_json"]
    issues: List[str] = []
    schema = _schema_for_prompt(prompt)
    if schema:
        issues.extend(_apply_schema(schema, payload))
    return issues


def _coerce_tool_loop_output(output: str) -> str:
    """
    Coerce tool-loop outputs into a valid JSON shape when possible.
    If file_ops exists without commands/final, inject defaults.
    """
    payload = _safe_json(output)
    if not isinstance(payload, dict):
        # Try to extract shell commands from fenced code blocks.
        try:
            blocks = re.findall(r"```(?:bash|sh|powershell)?\n([\s\S]+?)```", output or "", re.IGNORECASE)
            commands: list[str] = []
            for block in blocks:
                for line in block.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    commands.append(line)
            if commands:
                return json.dumps({"commands": commands, "file_ops": [], "final": ""}, ensure_ascii=False)
        except Exception:
            pass
        return output
    if "file_ops" in payload or "actions" in payload:
        if "commands" not in payload:
            payload["commands"] = []
        if "final" not in payload:
            payload["final"] = ""
        return json.dumps(payload, ensure_ascii=False)
    return output


def _coerce_scientific_output(output: str) -> str:
    """
    Coerce scientific-method outputs into a valid JSON shape when possible.
    Injects required keys with sane defaults.
    """
    payload = _safe_json(output)
    if not isinstance(payload, dict):
        return output
    # Required fields for scientific_method schema.
    defaults = {
        "observations": [],
        "hypotheses": [],
        "tests": [],
        "findings": [],
        "conclusion": "",
        "next_steps": [],
    }
    for key, val in defaults.items():
        if key not in payload:
            payload[key] = val
    # Ensure findings is a list of objects.
    findings = payload.get("findings")
    if not isinstance(findings, list):
        payload["findings"] = []
    return json.dumps(payload, ensure_ascii=False)


def _validate_evidence(output: str, *, evidence_bundle: Optional[str] = None) -> List[str]:
    if not _looks_like_json(output):
        return ["missing_json_for_evidence"]
    payload = _safe_json(output)
    if not isinstance(payload, dict):
        return ["invalid_json_for_evidence"]
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return ["findings_not_list"]
    issues: List[str] = []
    evidence_bundle = evidence_bundle or _extract_evidence_bundle(output)
    if evidence_bundle is None:
        return ["missing_evidence_bundle"]
    bundle_hash = _hash_text(evidence_bundle)
    if payload.get("_evidence_bundle_sha256") and payload.get("_evidence_bundle_sha256") != bundle_hash:
        issues.append("evidence_bundle_hash_mismatch")
    for item in findings:
        if not isinstance(item, dict):
            issues.append("finding_not_object")
            continue
        if not item.get("evidence_cmd") or not item.get("evidence_excerpt"):
            issues.append("missing_evidence_fields")
            continue
        excerpt = str(item.get("evidence_excerpt") or "")
        if excerpt and evidence_bundle and excerpt not in evidence_bundle:
            issues.append("evidence_excerpt_not_in_bundle")
    return issues


def _schema_for_prompt(prompt: str) -> Optional[dict]:
    lower = (prompt or "").lower()
    schemas = _load_schemas()
    if "[schema:tool_loop]" in lower:
        return schemas.get("tool_loop")
    if "[schema:scientific]" in lower:
        return schemas.get("scientific_method")
    if "observations (list of strings)" in lower and "findings" in lower:
        return schemas.get("scientific_method")
    if "commands (list of strings)" in lower and "final (string" in lower:
        return schemas.get("tool_loop")
    return None


def _apply_schema(schema: dict, payload: dict) -> List[str]:
    issues: List[str] = []
    required = schema.get("required") or []
    for key in required:
        if key not in payload:
            if key == "commands" and ("file_ops" in payload or "actions" in payload):
                continue
            issues.append(f"missing_{key}")
    list_fields = schema.get("list_fields") or []
    for field in list_fields:
        if field in payload and not isinstance(payload[field], list):
            issues.append(f"{field}_not_list")
    object_list_fields = schema.get("object_list_fields") or {}
    for field, required_keys in object_list_fields.items():
        items = payload.get(field)
        if items is None:
            continue
        if not isinstance(items, list):
            issues.append(f"{field}_not_list")
            continue
        for item in items:
            if not isinstance(item, dict):
                issues.append(f"{field}_item_not_object")
                break
            for key in required_keys:
                if key not in item:
                    issues.append(f"{field}_missing_{key}")
    return issues


def _load_schemas() -> dict:
    path = Path("config/c0d3r_schemas.json")
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_evidence_bundle(prompt: str) -> Optional[str]:
    marker = "Evidence bundle:"
    if marker not in prompt:
        return None
    idx = prompt.find(marker)
    return prompt[idx + len(marker):]


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _inject_evidence_hash(text: str, evidence_bundle: str) -> str:
    payload = _safe_json(text)
    if not isinstance(payload, dict):
        return text
    payload["_evidence_bundle_sha256"] = _hash_text(evidence_bundle)
    return json.dumps(payload, ensure_ascii=False)


def _repair_backslashes(raw: str) -> str:
    try:
        return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
    except Exception:
        return raw


def _repair_json_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip(), flags=re.I | re.M)
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass
    candidate = _extract_json(cleaned)
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        pass
    repaired = _repair_backslashes(candidate)
    try:
        payload = json.loads(repaired)
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        pass
    try:
        trimmed = re.sub(r",\s*([}\]])", r"\1", candidate)
        payload = json.loads(trimmed)
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return None


def _safe_json(text: str) -> object:
    try:
        repaired = _repair_json_text(text)
        if repaired:
            return json.loads(repaired)
    except Exception:
        pass
    try:
        for candidate in _extract_json_candidates(text):
            try:
                return json.loads(candidate)
            except Exception:
                repaired = _repair_backslashes(candidate)
                if repaired != candidate:
                    try:
                        return json.loads(repaired)
                    except Exception:
                        continue
                continue
        repaired = _repair_backslashes(_extract_json(text))
        return json.loads(repaired)
    except Exception:
        return {}


def _build_bedrock_payload(
    *,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    images: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    lower = model_id.lower()
    if "anthropic" in lower or "claude" in lower:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in _load_images(images):
            content.append(image)
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
    if any(key in lower for key in ("openai", "mistral", "qwen", "deepseek")):
        # Bedrock chat-style payload
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    if "llama" in lower or "meta." in lower:
        return {
            "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    if "titan" in lower or "amazon." in lower:
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            },
        }
    return {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }


def _build_converse_payload(
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    image_blocks: Optional[Sequence[Dict[str, Any]]] = None,
    document_blocks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    content.append({"text": prompt or " "})
    for block in image_blocks or []:
        content.append({"image": block})
    for block in document_blocks or []:
        content.append({"document": block})
    return {
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }


def _extract_text(model_id: str, payload: Dict[str, Any]) -> str:
    lower = model_id.lower()
    if "anthropic" in lower or "claude" in lower:
        content = payload.get("content") or []
        if isinstance(content, list) and content:
            text = content[0].get("text")
            if text:
                return text
    if any(key in lower for key in ("openai", "mistral", "qwen", "deepseek")):
        if "outputText" in payload:
            return str(payload.get("outputText") or "")
        choices = payload.get("choices") or []
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if content:
                    return str(content)
            text = choices[0].get("text")
            if text:
                return str(text)
    if "llama" in lower or "meta." in lower:
        return str(payload.get("generation") or "")
    if "titan" in lower or "amazon." in lower:
        results = payload.get("results") or []
        if results:
            return str(results[0].get("outputText") or "")
    return str(payload.get("outputText") or payload.get("generation") or payload.get("completion") or "")


def _extract_converse_text(payload: Dict[str, Any]) -> str:
    try:
        output = payload.get("output") or {}
        message = output.get("message") or {}
        content = message.get("content") or []
        if isinstance(content, list):
            pieces = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    pieces.append(str(block.get("text") or ""))
            if pieces:
                return "".join(pieces).strip()
    except Exception:
        pass
    return str(payload.get("outputText") or "")


def _doc_models_allow(model_id: str) -> bool:
    raw = os.getenv("C0D3R_DOC_MODELS", "").strip()
    if not raw:
        return True
    lower = model_id.lower()
    for needle in raw.split(","):
        token = needle.strip().lower()
        if token and token in lower:
            return True
    return False


def _load_images(images: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not images:
        return []
    loaded: List[Dict[str, Any]] = []
    for path in images[:6]:
        if not path:
            continue
        img_path = Path(str(path)).expanduser()
        if not img_path.exists() or not img_path.is_file():
            continue
        try:
            data = img_path.read_bytes()
        except Exception:
            continue
        if len(data) > 2_000_000:
            continue
        media_type = _guess_media_type(img_path)
        if not media_type:
            continue
        encoded = base64.b64encode(data).decode("ascii")
        loaded.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": encoded},
            }
        )
    return loaded


def _load_images_converse(images: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not images:
        return []
    loaded: List[Dict[str, Any]] = []
    for path in images[:6]:
        if not path:
            continue
        img_path = Path(str(path)).expanduser()
        if not img_path.exists() or not img_path.is_file():
            continue
        try:
            data = img_path.read_bytes()
        except Exception:
            continue
        if len(data) > 2_000_000:
            continue
        fmt = _guess_image_format(img_path)
        if not fmt:
            continue
        loaded.append({"format": fmt, "source": {"bytes": data}})
    return loaded


def _sanitize_bedrock_document_name(name: str) -> str:
    """
    Bedrock Converse enforces a restrictive filename charset for documents:
    alphanumeric, whitespace, hyphens, parentheses, and square brackets.
    It also disallows multiple consecutive whitespace characters.
    """
    raw = str(name or "").strip()
    if not raw:
        return "document"
    p = Path(raw)
    base = p.stem or raw
    ext = p.suffix.lstrip(".")

    # Normalize common separators for readability.
    base = base.replace("_", "-").replace(".", "-")
    ext = ext.replace("_", "-").replace(".", "-")

    # Keep only allowed characters.
    base = re.sub(r"[^0-9A-Za-z\\s\\-\\(\\)\\[\\]]+", " ", base)
    base = re.sub(r"\\s{2,}", " ", base).strip()
    if not base:
        base = "document"

    if ext:
        ext = re.sub(r"[^0-9A-Za-z\\s\\-\\(\\)\\[\\]]+", " ", ext)
        ext = re.sub(r"\\s{2,}", " ", ext).strip()
        if ext:
            base = f"{base} ({ext})"

    # Keep it reasonably short to avoid service-side limits.
    if len(base) > 80:
        base = base[:80].rstrip()
    return base


def _load_documents(documents: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not documents:
        return []
    loaded: List[Dict[str, Any]] = []
    max_files = int(os.getenv("C0D3R_DOC_MAX_FILES", "5") or "5")
    max_bytes = int(os.getenv("C0D3R_DOC_MAX_BYTES", "4500000") or "4500000")
    for path in documents[:max_files]:
        if not path:
            continue
        doc_path = Path(str(path)).expanduser()
        if not doc_path.exists() or not doc_path.is_file():
            continue
        fmt = _guess_document_format(doc_path)
        if not fmt:
            continue
        try:
            data = doc_path.read_bytes()
        except Exception:
            continue
        if max_bytes and len(data) > max_bytes:
            continue
        loaded.append({"format": fmt, "name": _sanitize_bedrock_document_name(doc_path.name), "source": {"bytes": data}})
    return loaded


def _guess_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png"}:
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix in {".gif"}:
        return "image/gif"
    if suffix in {".webp"}:
        return "image/webp"
    return ""


def _guess_image_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "png"
    if suffix in {".jpg", ".jpeg"}:
        return "jpeg"
    if suffix == ".gif":
        return "gif"
    if suffix == ".webp":
        return "webp"
    return ""


def _guess_document_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"}:
        return suffix
    return ""


__all__ = [
    "C0d3rSession",
    "C0d3r",
    "c0d3r_default_settings",
    "c0d3r_settings_for_role",
    "BedrockModelCatalog",
]
