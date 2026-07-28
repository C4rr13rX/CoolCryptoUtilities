from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .adapters import ProviderError, ProviderResponse, has_credential, invoke
from .models import ModelSpec
from .quota import QuotaLedger
from .feedback import ModelFeedbackStore


@dataclass(frozen=True)
class RequestProfile:
    weights: dict[str, float]
    estimated_tokens: int


@dataclass(frozen=True)
class RankedCandidate:
    spec: ModelSpec
    quality: float
    quality_tier: int
    headroom: float
    health: float
    uses: int
    semantic_health: float = 1.0


class FreeloaderRouter:
    """Select, invoke, and fail over between free hosted models."""

    def __init__(
        self,
        specs: list[ModelSpec],
        ledger: QuotaLedger,
        *,
        max_attempts: int = 6,
        timeout_s: float = 60.0,
        invoker: Callable[..., ProviderResponse] = invoke,
        feedback: ModelFeedbackStore | None = None,
        allowed_models: set[str] | None = None,
    ) -> None:
        self.specs = list(specs)
        self.ledger = ledger
        self.max_attempts = max(1, max_attempts)
        self.timeout_s = timeout_s
        self.invoker = invoker
        self.feedback = feedback or ModelFeedbackStore()
        self.allowed_models = {item.lower() for item in (allowed_models or set())}
        self._lock = threading.RLock()
        self._health: dict[str, float] = {}
        self._uses: dict[str, int] = {}
        self._provider_blocked_until: dict[str, float] = {}
        self.last_trace: list[dict] = []
        self.last_model_id = "agent-the-freeloader"

    def send(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
        preferred_identity: str = "",
        excluded_identities: set[str] | None = None,
    ) -> str:
        profile = classify_request(prompt, system, max_tokens=max_tokens)
        ranked = self.rank(profile)
        excluded = set(excluded_identities or ())
        if excluded:
            alternatives = [item for item in ranked if item.spec.identity not in excluded]
            if alternatives:
                ranked = alternatives
        if not ranked:
            raise RuntimeError(
                "AgentTheFreeloader has no eligible model. Configure at least one provider "
                "credential and ensure its shared quota pool is not exhausted."
            )
        if preferred_identity:
            preferred_index = next((
                index for index, item in enumerate(ranked)
                if item.spec.identity == preferred_identity
            ), -1)
            if preferred_index > 0 and ranked[preferred_index].quality_tier >= ranked[0].quality_tier - 1:
                ranked.insert(0, ranked.pop(preferred_index))

        attempt_list = _diversified_attempt_list(ranked)
        trace: list[dict] = []
        failures: list[str] = []
        for candidate in attempt_list[: self.max_attempts]:
            spec = candidate.spec
            # A prior candidate may have revealed that a shared provider pool
            # is exhausted.  Re-check here because the ranked list is a
            # snapshot taken before attempts began.
            if not self.ledger.available(spec.pool_ids, profile.estimated_tokens):
                trace.append({
                    "provider": spec.provider,
                    "model": spec.model_id,
                    "outcome": "skipped_shared_quota",
                })
                continue
            try:
                reservation = self.ledger.reserve(spec.pool_ids, profile.estimated_tokens)
            except RuntimeError:
                trace.append({
                    "provider": spec.provider,
                    "model": spec.model_id,
                    "outcome": "skipped_quota_race",
                })
                continue
            trace_item = {
                "provider": spec.provider,
                "model": spec.model_id,
                "quality": round(candidate.quality, 4),
                "quality_tier": candidate.quality_tier,
                "quota_headroom": round(candidate.headroom, 4),
                "semantic_health": round(candidate.semantic_health, 4),
            }
            try:
                response = self._invoke_with_deadline(
                    spec,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if not response.text:
                    raise ProviderError(502, "provider returned an empty response")
                self.ledger.reconcile(reservation, response.total_tokens)
                self.ledger.observe_headers(spec.pool_ids, response.headers)
                self._record_success(spec)
                trace_item["outcome"] = "selected"
                trace.append(trace_item)
                self.last_trace = trace
                self.last_model_id = spec.model_id
                return response.text
            except ProviderError as exc:
                self.ledger.release(reservation)
                self._record_failure(spec, exc)
                if exc.is_quota:
                    self.ledger.block(spec.pool_ids, exc.retry_after)
                trace_item.update({"outcome": "failed", "status": exc.status, "error": str(exc)[:300]})
                trace.append(trace_item)
                failures.append(f"{spec.provider}/{spec.model_id}: {exc}")
            except Exception as exc:
                self.ledger.release(reservation)
                wrapped = ProviderError(503, str(exc))
                self._record_failure(spec, wrapped)
                trace_item.update({"outcome": "failed", "status": 503, "error": str(exc)[:300]})
                trace.append(trace_item)
                failures.append(f"{spec.provider}/{spec.model_id}: {exc}")

        self.last_trace = trace
        detail = " | ".join(failures[-4:])
        raise RuntimeError(f"AgentTheFreeloader exhausted eligible fallbacks: {detail}")

    def _invoke_with_deadline(
        self,
        spec: ModelSpec,
        *,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
    ) -> ProviderResponse:
        """Enforce a wall-clock deadline even when an HTTP adapter stalls.

        Socket timeouts are not total request deadlines: DNS, TLS, or a server
        that trickles bytes can outlive them. A daemon worker lets routing move
        to the next provider without blocking process shutdown. The underlying
        adapter still receives the same timeout so healthy calls clean up
        normally.
        """
        result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def invoke_one() -> None:
            try:
                value = self.invoker(
                    spec,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=self.timeout_s,
                )
                result.put_nowait((True, value))
            except BaseException as exc:  # propagate adapter errors on caller thread
                result.put_nowait((False, exc))

        worker = threading.Thread(
            target=invoke_one,
            name=f"atf-provider-{spec.provider}-{spec.model_id}"[:120],
            daemon=True,
        )
        worker.start()
        worker.join(max(0.1, float(self.timeout_s)))
        if worker.is_alive():
            raise ProviderError(
                504,
                f"provider exceeded hard wall-clock deadline ({self.timeout_s:g}s)",
            )
        try:
            ok, value = result.get_nowait()
        except queue.Empty as exc:
            raise ProviderError(503, "provider invocation ended without a result") from exc
        if not ok:
            raise value  # type: ignore[misc]
        if not isinstance(value, ProviderResponse):
            raise ProviderError(502, "provider adapter returned an invalid response type")
        return value

    def rank(self, profile: RequestProfile) -> list[RankedCandidate]:
        allowed_providers = _csv_env("AGENT_FREELOADER_PROVIDERS")
        denied_providers = _csv_env("AGENT_FREELOADER_DENY_PROVIDERS")
        allowed_models = _csv_env("AGENT_FREELOADER_MODELS")
        if self.allowed_models:
            allowed_models = self.allowed_models
        now = time.time()
        candidates: list[RankedCandidate] = []
        credential_cache: dict[str, bool] = {}
        with self._lock:
            for spec in self.specs:
                if allowed_providers and spec.provider.lower() not in allowed_providers:
                    continue
                if spec.provider.lower() in denied_providers:
                    continue
                if allowed_models and spec.model_id.lower() not in allowed_models:
                    continue
                if self._provider_blocked_until.get(spec.provider, 0.0) > now:
                    continue
                if spec.api_key_env not in credential_cache:
                    credential_cache[spec.api_key_env] = has_credential(spec)
                configured = credential_cache[spec.api_key_env]
                if not configured:
                    continue
                if not self.ledger.available(spec.pool_ids, profile.estimated_tokens):
                    continue
                quality = _quality(spec, profile)
                semantic_health = self.feedback.factor(spec.identity)
                quality = max(0.0, min(1.25, quality * semantic_health))
                candidates.append(
                    RankedCandidate(
                        spec=spec,
                        quality=quality,
                        # Models within the same tenth are an equivalent quality
                        # tier; quota and health then decide, enabling rotation.
                        quality_tier=int(quality * 10),
                        headroom=self.ledger.headroom(spec.pool_ids, profile.estimated_tokens),
                        health=self._health.get(spec.identity, 1.0),
                        uses=self._uses.get(spec.identity, 0),
                        semantic_health=semantic_health,
                    )
                )
        candidates.sort(
            key=lambda item: (
                item.quality_tier,
                item.headroom,
                item.health,
                -item.uses,
                item.quality,
            ),
            reverse=True,
        )
        return candidates

    def report_outcome(
        self,
        provider: str,
        model_id: str,
        *,
        success: bool,
        reason: str = "",
    ) -> None:
        self.feedback.record(provider, model_id, success=success, reason=reason)

    def _record_success(self, spec: ModelSpec) -> None:
        with self._lock:
            self._uses[spec.identity] = self._uses.get(spec.identity, 0) + 1
            previous = self._health.get(spec.identity, 1.0)
            self._health[spec.identity] = min(1.0, previous * 0.8 + 0.2)

    def _record_failure(self, spec: ModelSpec, error: ProviderError) -> None:
        with self._lock:
            previous = self._health.get(spec.identity, 1.0)
            self._health[spec.identity] = max(0.05, previous * 0.55)
            if error.is_auth:
                self._provider_blocked_until[spec.provider] = time.time() + 300.0
        if not spec.api_key_env.startswith("TEST_"):
            self.feedback.record(spec.provider, spec.model_id, success=False, reason=f"transport:{error}")


def _diversified_attempt_list(ranked: list[RankedCandidate]) -> list[RankedCandidate]:
    """Prefer breadth across shared quota pools before depth inside one pool."""
    buckets: dict[str, list[RankedCandidate]] = {}
    order: list[str] = []
    for item in ranked:
        pool = item.spec.pool_ids[0] if item.spec.pool_ids else item.spec.provider
        if pool not in buckets:
            buckets[pool] = []
            order.append(pool)
        buckets[pool].append(item)

    diversified: list[RankedCandidate] = []
    index = 0
    while True:
        added = False
        for pool in order:
            bucket = buckets[pool]
            if index < len(bucket):
                diversified.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return diversified


def classify_request(prompt: str, system: str, *, max_tokens: int) -> RequestProfile:
    text = f"{system}\n{prompt}".lower()
    weights = {"general": 1.0}
    bounded_research = "bounded read-only archival-research role" in text

    def add(name: str, weight: float, markers: tuple[str, ...]) -> None:
        if any(marker in text for marker in markers):
            weights[name] = weight

    if not bounded_research:
        add("coding", 2.4, ("code", "repository", "function", "class ", "pytest", "debug", "refactor", "implement"))
        add("tools", 2.5, ("tool_calls", "available tools", "function calling", "call a tool", "executor", "file_write"))
    add("reasoning", 2.0, ("reason", "analy", "plan", "prove", "tradeoff", "architecture", "diagnose"))
    add("structured", 1.8, ("return only json", "json object", "schema", "structured", "extract"))
    if bounded_research:
        weights["reasoning"] = max(weights.get("reasoning", 0.0), 2.8)
        weights["structured"] = max(weights.get("structured", 0.0), 2.4)
    add("multimodal", 2.2, ("image", "screenshot", "visual", "video"))
    add("multilingual", 1.6, ("translate", "multilingual", "language"))
    add("speed", 1.4, ("fast", "quick", "low latency", "classify", "route"))
    input_tokens = max(1, (len(prompt) + len(system) + 3) // 4)
    # Reserve the full requested output budget. Providers that return usage
    # metadata reconcile this downward; providers that omit it retain the
    # conservative reservation so workday token limits cannot be bypassed.
    estimated_output = max(128, max_tokens)
    return RequestProfile(weights=weights, estimated_tokens=input_tokens + estimated_output)


def _quality(spec: ModelSpec, profile: RequestProfile) -> float:
    weighted = 0.0
    total = 0.0
    for capability, weight in profile.weights.items():
        weighted += spec.capabilities.get(capability, spec.capabilities.get("general", 0.5)) * weight
        total += weight
    return weighted / total if total else 0.0


def _csv_env(name: str) -> set[str]:
    return {value.strip().lower() for value in os.getenv(name, "").split(",") if value.strip()}
