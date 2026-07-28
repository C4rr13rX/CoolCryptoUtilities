from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

try:
    from model_response_normalizer import ModelResponseNormalizer
except ModuleNotFoundError:  # package import in tests/library consumers
    from .model_response_normalizer import ModelResponseNormalizer


BUILD_MARKERS = (
    "build", "create", "make", "implement", "develop", "design", "write",
    "scaffold", "set up", "setup", "produce", "generate", "add", "upgrade",
)

EXPANSION_DOMAINS = {
    "blockchain", "crypto", "mobile", "android", "ios", "cloud", "saas",
    "multiplayer", "social network", "machine learning", "artificial intelligence",
    "subscription", "marketplace", "microservice", "kubernetes", "ar/vr",
}


def is_creation_request(request: str) -> bool:
    text = " ".join(str(request or "").lower().split())
    return any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in BUILD_MARKERS)


class OutlineRefiner:
    """Scope-locked, model-assisted, deterministic outline refinement."""

    def __init__(
        self,
        *,
        send: Callable[..., str] | None = None,
        market_search: Callable[[str], dict] | None = None,
        workdir: Path | None = None,
        passes: int | None = None,
    ) -> None:
        self.send = send
        self.market_search = market_search
        self.workdir = workdir
        self.max_passes = max(1, min(int(passes or os.getenv("C0D3R_OUTLINE_REFINEMENT_PASSES", "4")), 8))

    def refine(self, request: str, scientific_request: str = "") -> dict[str, Any]:
        boundary = self._boundary(request)
        market = self._market_context(request)
        outline = self._seed(boundary, scientific_request or request)
        contract_ready = self._contract_ready(request)
        history = []
        stable_count = 0
        prior_score = 0.0
        for pass_number in range(1, self.max_passes + 1):
            outline = self._deterministic_pass(outline, boundary, market, pass_number)
            # An explicit single-artifact contract already contains the model's
            # reasoning target. Four deterministic passes still enrich and
            # scope-lock it, without wasting four scarce calls paraphrasing it.
            candidate = {} if contract_ready else self._model_pass(outline, boundary, market, pass_number)
            if candidate and not self._scope_violations(candidate, boundary):
                outline = self._safe_merge(outline, candidate)
            score = self._score(outline, market)
            gain = round(score - prior_score, 2)
            history.append({"pass": pass_number, "score": score, "gain": gain})
            stable_count = stable_count + 1 if gain < 2.0 else 0
            prior_score = score
            if score >= 92.0 and stable_count >= 1 and pass_number >= 3:
                break
        outline["quality"] = {
            "score": prior_score,
            "threshold": 92.0,
            "passed": prior_score >= 92.0 and (not self._market_required(request) or bool(market)),
            "refinement_passes": len(history),
            "history": history,
            "market_evidence_count": len(market),
            "market_evidence_required": self._market_required(request),
            "contract_ready_fast_path": contract_ready,
        }
        outline["scope_boundary"] = boundary
        outline["market_evidence"] = market
        outline["created_at"] = time.time()
        if self.workdir:
            path = self.workdir / ".c0d3r" / "refined-outline.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(outline, indent=2, ensure_ascii=True), encoding="utf-8")
            outline["path"] = str(path)
        return outline

    @staticmethod
    def _market_required(request: str) -> bool:
        text = request.lower()
        return any(term in text for term in ("market", "sell", "sale", "buyer", "customer demand", "commercial product"))

    @staticmethod
    def _contract_ready(request: str) -> bool:
        text = request.lower()
        artifact = bool(re.search(r"\b[\w.-]+\.(?:py|ts|tsx|js|jsx|rs|go|java|cpp|c|h|php|pl)\b", text))
        validator = any(term in text for term in ("run:", "test_", "pytest", "acceptance", "contract.json"))
        bounded = any(term in text for term in (
            "one class", "the class", "public class", "class described", "class named",
            "single file", "currently locked", "atomic contract",
        ))
        return artifact and validator and bounded

    @staticmethod
    def _boundary(request: str) -> dict[str, Any]:
        normalized = " ".join(request.split())
        explicit_constraints = [
            part.strip() for part in re.split(r"[.;]\s+", normalized)
            if any(term in part.lower() for term in ("must", "should", "only", "without", "do not", "don't", "needs to", "has to"))
        ]
        return {
            "original_request": request,
            "goal": normalized,
            "explicit_constraints": explicit_constraints,
            "allowed_supporting_detail": [
                "implementation architecture", "interfaces and data shapes", "testing and validation",
                "accessibility, security, reliability and performance needed for the requested result",
                "packaging, documentation and local operation needed to use the requested result",
            ],
            "forbidden": [
                "new business goals, audiences, platforms, integrations, monetization, or features not requested",
                "working outside the authorized project root",
                "weakening or silently dropping an explicit constraint",
            ],
        }

    def _market_context(self, request: str) -> list[dict[str, str]]:
        if not self.market_search:
            return []
        words = [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9+-]{2,}", request) if word.lower() not in BUILD_MARKERS]
        query = " ".join(words[:12]) + " professional best practices quality standards"
        try:
            result = self.market_search(query) or {}
        except Exception:
            return []
        evidence = []
        for item in (result.get("results") or [])[:5]:
            if not isinstance(item, dict):
                continue
            evidence.append({
                "title": str(item.get("title") or "")[:160],
                "url": str(item.get("url") or "")[:300],
                "snippet": str(item.get("snippet") or "")[:400],
            })
        return evidence

    @staticmethod
    def _seed(boundary: dict, scientific_request: str) -> dict[str, Any]:
        return {
            "title": "Scope-locked delivery outline",
            "goal": boundary["goal"],
            "scientific_request": scientific_request,
            "users_and_use_cases": [],
            "deliverables": [],
            "functional_requirements": [],
            "quality_requirements": [],
            "architecture": [],
            "interfaces": [],
            "data_contracts": [],
            "validation": [],
            "risks": [],
            "release_and_operation": [],
            "atomic_work": [],
            "non_goals": list(boundary["forbidden"]),
        }

    def _deterministic_pass(self, outline: dict, boundary: dict, market: list[dict], pass_number: int) -> dict:
        value = json.loads(json.dumps(outline))
        goal = boundary["goal"]
        if pass_number == 1:
            value["users_and_use_cases"] = [f"Primary user completes the requested outcome: {goal}"]
            value["deliverables"] = ["A complete usable implementation of the requested result", "Usage and operation instructions"]
            value["functional_requirements"] = [goal, *boundary["explicit_constraints"]]
        elif pass_number == 2:
            value["architecture"] = ["Separate domain behavior, adapters/integrations, presentation, and validation where applicable", "Preserve existing project conventions discovered by the project mapper"]
            value["interfaces"] = ["Every module/class task declares named inputs, input shape, outputs, output shape, errors, and invariants"]
            value["data_contracts"] = ["Machine-readable schemas validate at system boundaries", "No implicit shape changes between dependent work contracts"]
            value["atomic_work"] = ["Map only in-scope files and interfaces", "Implement one bounded file/class/artifact contract at a time", "Integrate completed contracts"]
        elif pass_number == 3:
            value["quality_requirements"] = ["No known critical correctness, security, accessibility, or reliability defects", "Requested workflows succeed from a clean local start", "Performance is measured against an explicit use-appropriate bound"]
            value["validation"] = ["Run deterministic syntax/schema/parser checks", "Run focused unit or artifact tests after every change", "Run integration and user-flow acceptance checks before completion", "Record failures and repeat test-fix-test until all required checks pass"]
            value["risks"] = ["Scope drift", "Interface mismatch", "Model hallucination", "False-positive completion", "Missing runtime dependency"]
        else:
            value["release_and_operation"] = ["Provide a reproducible start/open command", "Record dependency and environment requirements", "Preserve a validated working version before further development"]
            value["validation"].append("Compare the completed result with relevant professional/market expectations without adding unrequested features")
            if market:
                value["quality_requirements"].append("Use cited market/professional evidence only to improve quality attributes inside the requested scope")
        return self._dedupe(value)

    def _model_pass(self, outline: dict, boundary: dict, market: list[dict], pass_number: int) -> dict:
        if not self.send:
            return {}
        prompt = json.dumps({
            "instruction": "Improve this outline one level. Preserve scope exactly. Add missing implementation detail, interfaces, input/output shapes, validation, risks, and professional quality constraints. Do not implement anything. Return one JSON object using the same keys.",
            "pass": pass_number,
            "scope_boundary": boundary,
            "market_evidence": market,
            "outline": outline,
        }, ensure_ascii=True)
        try:
            raw = self.send(prompt=prompt, stream=False, system="You refine plans without expanding user intent. JSON only; no tool calls or markdown.")
            parsed = ModelResponseNormalizer().parse(raw)
            if parsed.valid and isinstance(parsed.value, dict):
                return parsed.value
        except Exception:
            return {}
        return {}

    @staticmethod
    def _safe_merge(base: dict, candidate: dict) -> dict:
        merged = json.loads(json.dumps(base))
        allowed = set(base)
        for key, value in candidate.items():
            if key not in allowed or key in {"goal", "scientific_request", "non_goals"}:
                continue
            if isinstance(base.get(key), list) and isinstance(value, list):
                merged[key] = [*base[key], *[str(item) for item in value if isinstance(item, (str, int, float))]][:40]
            elif isinstance(base.get(key), dict) and isinstance(value, dict):
                merged[key] = {**base[key], **value}
            elif isinstance(value, str):
                merged[key] = value
        return OutlineRefiner._dedupe(merged)

    @staticmethod
    def _scope_violations(candidate: dict, boundary: dict) -> list[str]:
        original = boundary["original_request"].lower()
        text = json.dumps(candidate, ensure_ascii=True).lower()
        return sorted(domain for domain in EXPANSION_DOMAINS if domain in text and domain not in original)

    @staticmethod
    def _dedupe(value: dict) -> dict:
        for key, items in list(value.items()):
            if isinstance(items, list):
                seen = set(); cleaned = []
                for item in items:
                    marker = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item).strip().lower()
                    if marker and marker not in seen:
                        seen.add(marker); cleaned.append(item)
                value[key] = cleaned
        return value

    @staticmethod
    def _score(outline: dict, market: list[dict]) -> float:
        sections = ("users_and_use_cases", "deliverables", "functional_requirements", "quality_requirements", "architecture", "interfaces", "data_contracts", "validation", "risks", "release_and_operation", "atomic_work", "non_goals")
        coverage = sum(1 for key in sections if outline.get(key)) / len(sections) * 60
        validation = min(15, len(outline.get("validation") or []) * 3)
        contracts = 10 if outline.get("interfaces") and outline.get("data_contracts") else 0
        scope = 10 if outline.get("non_goals") else 0
        evidence = 5 if market else 3  # market evidence is useful but not mandatory for every artifact
        return round(min(100.0, coverage + validation + contracts + scope + evidence), 2)
