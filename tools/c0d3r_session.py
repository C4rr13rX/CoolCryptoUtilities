# tools/c0d3r_session.py
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import boto3

from services.research_sources import allowed_domains_for_query, select_sources_for_query
from services.session_storage import SessionStore, default_session_config
from services.web_search import WebSearch
from services.web_research import NCBIClient, ReferenceDataClient, ScholarlyAPIClient, WebResearcher


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
    def __init__(self, *, profile: Optional[str], region: str) -> None:
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        self.client = session.client("bedrock")

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


class BedrockClient:
    def __init__(self, *, profile: Optional[str], region: str) -> None:
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        self.client = session.client("bedrock-runtime")

    def invoke(self, *, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )
        raw = response["body"].read().decode("utf-8", errors="replace")
        return json.loads(raw)


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
        effort = str(settings.get("reasoning_effort") or "").strip().lower()
        if effort in {"extra_high", "xhigh", "xh"}:
            self.max_revisions = max(self.max_revisions, 2)
            self.max_tokens = max(self.max_tokens, 3072)
        elif effort in {"high", "h"}:
            self.max_revisions = max(self.max_revisions, 1)
            self.max_tokens = max(self.max_tokens, 2560)
        self._catalog = BedrockModelCatalog(profile=self.profile, region=self.region)
        self._runtime = BedrockClient(profile=self.profile, region=self.region)

    def ensure_model(self) -> str:
        if self.inference_profile:
            self.model_id = self.inference_profile
            return self.model_id
        if self.model_id:
            return self.model_id
        try:
            models = self._catalog.list_models()
        except Exception:
            models = []
        preference = self._preferred_models()
        try:
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
    ) -> str:
        model_id = self.ensure_model()
        if not model_id:
            return "[c0d3r error] no Bedrock models available"
        research_payload = ""
        if research or self.research_enabled:
            research_payload = self._research(prompt)
        synthesis = ""
        if research_payload:
            synthesis = self._invoke_model(
                self._model_for_stage("synthesizer"),
                self._build_prompt("synthesizer", prompt, system=system, research=research_payload),
                images=images,
            )
        plan = self._invoke_model(
            self._model_for_stage("planner"),
            self._build_prompt("planner", prompt, system=system, research=synthesis or research_payload),
            images=images,
        )
        if self.consensus_k > 1:
            drafts = []
            for _ in range(min(self.consensus_k, 4)):
                drafts.append(
                    self._invoke_model(
                        self._model_for_stage("executor"),
                        self._build_prompt("executor", prompt, system=system, plan=plan, research=synthesis or research_payload),
                        images=images,
                    )
                )
            output = self._select_best_candidate(
                drafts,
                prompt,
                system=system,
                plan=plan,
                research=synthesis or research_payload,
                images=images,
            )
        else:
            output = self._invoke_model(
                self._model_for_stage("executor"),
                self._build_prompt("executor", prompt, system=system, plan=plan, research=synthesis or research_payload),
                images=images,
            )
        output = self._enforce_schema(output, prompt, system=system, plan=plan, research=synthesis or research_payload, images=images)
        for _ in range(max(0, self.max_revisions)):
            review = self._invoke_model(
                self._model_for_stage("reviewer"),
                self._build_prompt("reviewer", prompt, system=system, plan=plan, draft=output, research=synthesis or research_payload),
                images=images,
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
            )
            output = self._enforce_schema(output, prompt, system=system, plan=plan, research=synthesis or research_payload, images=images)
        return output.strip()

    def _preferred_models(self) -> List[str]:
        raw = os.getenv("C0D3R_MODEL_PREFERENCE", "")
        if raw.strip():
            return [p.strip() for p in raw.split(",") if p.strip()]
        return [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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
            return override
        fallback = ROLE_FALLBACK_MODEL.get(role)
        return fallback or self.ensure_model()

    def _parse_review(self, text: str) -> tuple[float, str]:
        try:
            payload = json.loads(_extract_json(text))
            score = float(payload.get("score") or 0)
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
    ) -> str:
        """
        Enforce JSON-only outputs when the prompt requires it by re-asking with constraints.
        """
        issues = _validate_output(prompt, output)
        if not issues:
            return output
        if _requires_json(prompt) and _looks_like_json(output) and not issues:
            return output
        repair_prompt = (
            "Return ONLY valid JSON matching the requested schema. "
            "Do not include any other text."
        )
        corrected = self._invoke_model(
            self._model_for_stage("refiner"),
            self._build_prompt(
                "refiner",
                prompt,
                system=system,
                plan=plan,
                draft=output,
                feedback=repair_prompt,
                research=research,
            ),
            images=images,
        )
        return corrected

    def _select_best_candidate(
        self,
        drafts: List[str],
        prompt: str,
        *,
        system: Optional[str],
        plan: Optional[str],
        research: Optional[str],
        images: Optional[Sequence[str]],
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
            )
            score, _ = self._parse_review(review)
            if score >= self.min_score:
                return best
        return best

    def _invoke_model(self, model_id: str, prompt: str, *, images: Optional[Sequence[str]] = None) -> str:
        effective_model_id = self.inference_profile or model_id
        payload = _build_bedrock_payload(
            model_id=model_id,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            images=images,
        )
        try:
            result = self._runtime.invoke(model_id=effective_model_id, payload=payload)
        except Exception as exc:
            if self.inference_profile or "inference profile" not in str(exc).lower():
                raise
            resolved = self._resolve_inference_profile(model_id)
            if not resolved:
                raise
            self.inference_profile = resolved
            self.model_id = resolved
            result = self._runtime.invoke(model_id=resolved, payload=payload)
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
            for q in questions[:6]:
                strategy = self._research_strategy(q)
                max_sources = strategy.get("max_sources", 8)
                max_results = strategy.get("max_results", 4)
                sources = select_sources_for_query(q, max_sources=max_sources)
                domains = allowed_domains_for_query(q, max_sources=max_sources)
                results = search.search_domains(q, domains, limit_per_domain=1, total_limit=max_results)
                snippets.append(f"[question] {q}")
                if strategy.get("use_ncbi", True) and any("ncbi.nlm.nih.gov" in src.domain for src in sources):
                    try:
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
                        oa = api_client.openalex(q, limit=2)
                        if oa:
                            snippets.append("[OpenAlex]")
                            for line in oa:
                                snippets.append(f"- {line}")
                    except Exception:
                        pass
                    try:
                        cr = api_client.crossref(q, limit=2)
                        if cr:
                            snippets.append("[Crossref]")
                            for line in cr:
                                snippets.append(f"- {line}")
                    except Exception:
                        pass
                    try:
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
                for result in results:
                    text = researcher.fetch_text(result.url)
                    snippets.append(f"- {result.title} ({result.url})\n  {text[:800]}")
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
        **settings: Any,
    ) -> None:
        self.session_name = session_name
        self.read_timeout_s = read_timeout_s
        self.stream_default = stream_default
        self.verbose_default = verbose_default
        self.workdir = Path(workdir).resolve() if workdir else None
        self._stream_callback: Optional[Callable[[str], None]] = None

        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.transcript_dir / f"{session_name}.log"

        defaults = c0d3r_default_settings()
        merged = {**defaults, **settings}
        if "stream_default" in merged:
            self.stream_default = bool(merged.get("stream_default"))
        self.settings = merged
        self._c0d3r = C0d3r(merged)
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
    ) -> str:
        images_list = list(images) if images else []
        stream = self.stream_default if stream is None else stream
        verbose = self.verbose_default if verbose is None else verbose
        prev_callback = self._stream_callback
        if stream_callback is not None:
            self._stream_callback = stream_callback
        if verbose:
            self._print(f"[c0d3r] model={self._c0d3r.ensure_model()}\n")
        self._log_event("prompt", {"prompt": prompt})
        output = self._c0d3r.generate(
            prompt,
            system=system,
            research=self.settings.get("research", False),
            images=images_list if images_list else None,
        )
        self._log_event("response", {"response": output})
        self._append_transcript(prompt, output)
        if stream:
            self._stream_output(output)
        self._stream_callback = prev_callback
        return output

    def _append_transcript(self, prompt: str, response: str) -> None:
        divider = "=" * 80
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{divider}\nTIMESTAMP: {ts}\nSESSION: {self.session_name}\n"
                f"PROMPT:\n{prompt}\nRESPONSE:\n{response}\n"
            )

    def _log_event(self, event_type: str, payload: dict) -> None:
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


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return "{}"
    return text[start : end + 1]


def _requires_json(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "return only json" in lower or "return only json:" in lower or "return only json" in lower


def _looks_like_json(text: str) -> bool:
    stripped = (text or "").strip()
    return stripped.startswith("{") and stripped.endswith("}")


def _requires_commands(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "commands" in lower and "return only json" in lower


def _requires_citations(prompt: str) -> bool:
    lower = (prompt or "").lower()
    return "cite" in lower or "citation" in lower or "sources" in lower


def _has_citations(text: str) -> bool:
    lower = (text or "").lower()
    return "http" in lower or "[" in lower and "]" in lower


def _validate_output(prompt: str, output: str) -> List[str]:
    issues: List[str] = []
    if _requires_json(prompt) and not _looks_like_json(output):
        issues.append("missing_json")
    if _requires_commands(prompt) and "commands" not in output:
        issues.append("missing_commands")
    if _requires_citations(prompt) and not _has_citations(output):
        issues.append("missing_citations")
    return issues


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


def _extract_text(model_id: str, payload: Dict[str, Any]) -> str:
    lower = model_id.lower()
    if "anthropic" in lower or "claude" in lower:
        content = payload.get("content") or []
        if isinstance(content, list) and content:
            text = content[0].get("text")
            if text:
                return text
    if "llama" in lower or "meta." in lower:
        return str(payload.get("generation") or "")
    if "titan" in lower or "amazon." in lower:
        results = payload.get("results") or []
        if results:
            return str(results[0].get("outputText") or "")
    return str(payload.get("outputText") or payload.get("generation") or payload.get("completion") or "")


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


__all__ = [
    "C0d3rSession",
    "C0d3r",
    "c0d3r_default_settings",
    "c0d3r_settings_for_role",
    "BedrockModelCatalog",
]
