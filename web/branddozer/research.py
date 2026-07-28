"""Evidence-first archival research workflow for Brand Dozer."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from django.db import close_old_connections
from django.utils import timezone

from branddozer.models import (
    BacklogItem,
    DeliveryArtifact,
    DeliveryProject,
    DeliveryRun,
    DeliverySession,
    ResearchClaim,
    ResearchPaper,
    ResearchPaperRevision,
    ResearchSource,
    Sprint,
    SprintItem,
)
from tools.ai_session import (
    default_settings,
    get_session_class,
    session_provider_from_context,
)
from tools.c0d3rV2.delivery_runner import run_delivery_turn_detailed
from tools.c0d3rV2.plugins.research_harvester import HarvestConfig, ResearchHarvester


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "branddozer" / "research_papers"
REQUIRED_SECTIONS = (
    "abstract",
    "keywords",
    "introduction",
    "methodology",
    "literature review",
    "findings",
    "discussion",
    "limitations",
    "conclusion",
    "references",
)
CITATION_RE = re.compile(r"\[@([A-Za-z0-9_.:-]+)\]")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")
ROLE_SCHEMA_KEYS: dict[str, set[str]] = {
    "research_planner": {
        "title", "research_question", "keywords", "scope",
        "search_strategy", "work_packages",
    },
    "literature_reviewer": {"findings", "sources", "claims"},
    "methods_reviewer": {"method", "bias_risks", "validity_limits"},
    "research_writer": {"title", "abstract", "markdown", "claims"},
    "citation_auditor": {"claims", "blocking_issues"},
    "peer_reviewer": {"recommendation", "blocking_issues"},
}


@dataclass(frozen=True)
class ResearchPolicy:
    min_words: int = 5000
    min_sources: int = 12
    min_verified_sources: int = 10
    min_high_authority_sources: int = 6
    min_source_domains: int = 4
    max_revision_rounds: int = 4
    max_parallel_agents: int = 4

    @classmethod
    def from_context(cls, context: dict[str, Any] | None) -> "ResearchPolicy":
        raw = (context or {}).get("research_config") or {}

        def bounded(name: str, default: int, low: int, high: int) -> int:
            try:
                value = int(raw.get(name, default))
            except (TypeError, ValueError):
                value = default
            return max(low, min(high, value))

        return cls(
            min_words=bounded("min_words", 5000, 500, 30000),
            min_sources=bounded("min_sources", 12, 3, 100),
            min_verified_sources=bounded("min_verified_sources", 10, 2, 100),
            min_high_authority_sources=bounded(
                "min_high_authority_sources", 6, 1, 100
            ),
            min_source_domains=bounded("min_source_domains", 4, 2, 30),
            max_revision_rounds=bounded("max_revision_rounds", 4, 1, 10),
            max_parallel_agents=bounded("max_parallel_agents", 4, 1, 8),
        )


def _json_object_candidates(text: str) -> list[dict[str, Any]]:
    """Recover every complete object from a C0D3R multi-branch response."""
    raw = str(text or "").strip()
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    index = 0
    while index < len(raw):
        start = raw.find("{", index)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            index = start + 1
            continue
        index = start + consumed
        if isinstance(value, dict):
            found.append(value)

    expanded: list[dict[str, Any]] = []
    queue = list(found)
    seen: set[str] = set()
    while queue:
        value = queue.pop(0)
        fingerprint = json.dumps(value, sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        expanded.append(value)
        for key in ("answer", "output", "result", "payload"):
            nested = value.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
            elif isinstance(nested, str) and "{" in nested:
                queue.extend(_json_object_candidates(nested))
    return expanded


def _extract_json(
    text: str, *, expected_keys: set[str] | None = None
) -> dict[str, Any]:
    raw = str(text or "").strip()
    candidates = _json_object_candidates(raw)
    if not candidates:
        raise RuntimeError("research agent returned no complete JSON object")
    required = set(expected_keys or ())

    def score(value: dict[str, Any]) -> tuple[int, int, int]:
        matched = len(required.intersection(value))
        return (
            matched,
            len(value),
            len(json.dumps(value, default=str)),
        )

    selected = max(candidates, key=score)
    if required and not required.issubset(selected):
        missing = sorted(required.difference(selected))
        raise RuntimeError(
            "research agent JSON did not satisfy the role schema; missing "
            + ", ".join(missing)
        )
    return selected


def _clean_key(value: Any, fallback: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value or "")).strip("-")
    return (key or fallback)[:120]


def _word_count(markdown: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", markdown or ""))


def _authority_tier(source: dict[str, Any]) -> int:
    host = (
        urllib.parse.urlparse(str(source.get("url") or "")).hostname or ""
    ).lower()
    doi = str(source.get("doi") or "").strip()
    publisher = str(source.get("publisher") or "").lower()
    if host.endswith((".gov", ".edu", ".ac.uk")) or any(
        marker in host
        for marker in (
            "doi.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
            "nist.gov", "nasa.gov", "ieee.org", "acm.org", "springer.com",
            "sciencedirect.com", "nature.com", "science.org",
        )
    ):
        return 3
    if doi or bool(source.get("peer_reviewed")) or any(
        marker in publisher for marker in ("university", "institute", "journal")
    ):
        return 2
    return 1


def validate_paper_payload(
    *,
    markdown: str,
    sources: Iterable[dict[str, Any]],
    claims: Iterable[dict[str, Any]],
    policy: ResearchPolicy,
    peer_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic publication-readiness gates over a candidate paper."""
    source_list = list(sources)
    claim_list = list(claims)
    source_by_key = {
        str(item.get("citation_key") or ""): item
        for item in source_list
        if item.get("citation_key")
    }
    headings = {
        re.sub(r"[^a-z0-9 ]+", "", heading.lower()).strip()
        for heading in HEADING_RE.findall(markdown or "")
    }
    missing_sections = [
        required
        for required in REQUIRED_SECTIONS
        if not any(required == heading or required in heading for heading in headings)
    ]
    used_keys = set(CITATION_RE.findall(markdown or ""))
    unknown_citations = sorted(used_keys - set(source_by_key))
    uncited_sources = sorted(set(source_by_key) - used_keys)
    verified = [
        source
        for source in source_list
        if source.get("verification_status") == "verified"
        and source.get("content_sha256")
    ]
    high_authority = [
        source for source in verified if int(source.get("authority_tier") or 0) >= 2
    ]
    domains = {
        (urllib.parse.urlparse(str(source.get("url") or "")).hostname or "").lower()
        for source in verified
        if source.get("url")
    }
    unsupported_claims = []
    for index, claim in enumerate(claim_list):
        keys = {
            str(key)
            for key in (claim.get("source_keys") or [])
            if str(key).strip()
        }
        known_verified = {
            key
            for key in keys
            if key in source_by_key
            and source_by_key[key].get("verification_status") == "verified"
        }
        status = str(claim.get("verification_status") or "pending")
        if status not in {"supported", "qualified"} or not known_verified:
            unsupported_claims.append(
                {
                    "index": index,
                    "claim": str(claim.get("claim_text") or "")[:300],
                    "source_keys": sorted(keys),
                    "status": status,
                }
            )
    peer_review = peer_review or {}
    peer_blockers = [
        str(item)
        for item in (peer_review.get("blocking_issues") or [])
        if str(item).strip()
    ]
    checks = {
        "required_sections": not missing_sections,
        "minimum_word_count": _word_count(markdown) >= policy.min_words,
        "minimum_sources": len(source_list) >= policy.min_sources,
        "verified_sources": len(verified) >= policy.min_verified_sources,
        "authoritative_sources": (
            len(high_authority) >= policy.min_high_authority_sources
        ),
        "source_diversity": len(domains) >= policy.min_source_domains,
        "known_citations": not unknown_citations,
        "claims_supported": bool(claim_list) and not unsupported_claims,
        "peer_review": not peer_blockers
        and str(peer_review.get("recommendation") or "").lower()
        in {"accept", "minor_revision", "minor revision"},
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "word_count": _word_count(markdown),
            "sources": len(source_list),
            "verified_sources": len(verified),
            "high_authority_sources": len(high_authority),
            "source_domains": len(domains),
            "claims": len(claim_list),
            "citations_used": len(used_keys),
        },
        "missing_sections": missing_sections,
        "unknown_citations": unknown_citations,
        "uncited_sources": uncited_sources,
        "unsupported_claims": unsupported_claims,
        "peer_review_blockers": peer_blockers,
    }


def _verify_source(
    harvester: ResearchHarvester, source: dict[str, Any], question: str
) -> dict[str, Any]:
    candidate = dict(source)
    candidate["authority_tier"] = _authority_tier(candidate)
    url = str(candidate.get("url") or "").strip()
    if not url:
        candidate.update(
            verification_status="rejected",
            verification_detail="source URL is missing",
        )
        return candidate
    result = harvester.crawl(
        [url],
        query=question,
        config=HarvestConfig(
            max_pages=1,
            max_depth=0,
            max_bytes_per_page=2_000_000,
            delay_seconds=0,
            same_origin=True,
            respect_robots=True,
        ),
    )
    stored = result.get("stored") or []
    if not stored:
        candidate.update(
            verification_status="rejected",
            verification_detail=(
                (result.get("errors") or [{}])[-1].get("error")
                or "source could not be independently retrieved"
            ),
        )
        return candidate
    record = stored[0]
    document = harvester.document(str(record.get("url") or url))
    passage = re.sub(
        r"\s+", " ", str(candidate.get("verified_passage") or "")
    ).strip()
    content = re.sub(
        r"\s+", " ", str((document or {}).get("content") or "")
    ).strip()
    if len(passage) < 40 or passage.casefold() not in content.casefold():
        candidate.update(
            verification_status="rejected",
            verification_detail=(
                "source was retrieved, but the agent's purported supporting passage "
                "was not found verbatim in the fetched document"
            ),
            content_sha256=str(record.get("sha256") or ""),
        )
        return candidate
    doi = str(candidate.get("doi") or "").strip()
    if doi and doi.casefold() not in content.casefold() and doi.casefold() not in url.casefold():
        candidate["doi"] = ""
    candidate.update(
        url=record.get("url") or url,
        title=record.get("title") or candidate.get("title") or url,
        content_sha256=record.get("sha256") or "",
        retrieved_at=timezone.now().isoformat(),
        verification_status="verified",
        verification_detail="retrieved and content-hashed by ResearchHarvester",
    )
    return candidate


def _render_pdf(path: Path, title: str, markdown: str) -> None:
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ResearchTitle", parent=styles["Title"], alignment=TA_CENTER, spaceAfter=18
    )
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=title,
    )
    story: list[Any] = [Paragraph(escape(title), title_style), Spacer(1, 10)]
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 8))
        elif stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            style = styles["Heading1" if level <= 2 else "Heading2"]
            story.append(Paragraph(escape(stripped[level:].strip()), style))
        elif stripped == "---":
            story.append(PageBreak())
        else:
            story.append(Paragraph(escape(stripped), styles["BodyText"]))
            story.append(Spacer(1, 4))
    document.build(story)


def persist_paper_files(paper: ResearchPaper) -> None:
    directory = RUNTIME_ROOT / str(paper.id)
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / f"paper-v{paper.version}.md"
    pdf_path = directory / f"paper-v{paper.version}.pdf"
    markdown_path.write_text(paper.content_markdown, encoding="utf-8")
    _render_pdf(pdf_path, paper.title, paper.content_markdown)
    paper.markdown_path = str(markdown_path)
    paper.pdf_path = str(pdf_path)
    paper.save(update_fields=["markdown_path", "pdf_path", "updated_at"])


class ResearchWorkflow:
    """Bounded multi-agent Scrum workflow for one archival research run."""

    def __init__(
        self,
        run: DeliveryRun,
        root: Path,
        *,
        source_verifier: Callable[
            [ResearchHarvester, dict[str, Any], str], dict[str, Any]
        ] = _verify_source,
    ) -> None:
        self.run = run
        self.root = root
        self.policy = ResearchPolicy.from_context(run.context or {})
        context = run.context or {}
        self.agent_provider = str(context.get("agent_provider") or "c0d3r").strip().lower()
        self.model_provider = str(
            context.get("model_provider")
            or session_provider_from_context(context)
            or "wizard"
        ).strip().lower()
        # Kept as a compatibility alias for existing diagnostics.
        self.provider = self.model_provider
        self.source_verifier = source_verifier
        self.transcript_root = (
            PROJECT_ROOT / "runtime" / "branddozer" / "transcripts" / str(run.id)
        )
        self.transcript_root.mkdir(parents=True, exist_ok=True)
        self.harvester = ResearchHarvester(PROJECT_ROOT / "runtime" / "branddozer")

    def _session(self, role: str, name: str) -> tuple[DeliverySession, Any | None]:
        record = DeliverySession.objects.create(
            project=self.run.project,
            run=self.run,
            role=role,
            name=name,
            status="running",
            workspace_path=str(self.root),
            last_heartbeat=timezone.now(),
            meta={
                "agent_provider": self.agent_provider,
                "model_provider": self.model_provider,
                "research": True,
            },
        )
        log_path = (
            PROJECT_ROOT
            / "runtime"
            / "branddozer"
            / "sessions"
            / f"{record.id}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record.log_path = str(log_path)
        record.save(update_fields=["log_path"])
        client = None
        if self.agent_provider not in {"c0d3r", "coder", "c0d3rv2"}:
            SessionClass = get_session_class(self.model_provider)
            client = SessionClass(
                session_name=f"research-{role}-{record.id}",
                transcript_dir=self.transcript_root,
                read_timeout_s=None,
                workdir=str(self.root),
                **default_settings(self.model_provider),
            )
        return record, client

    @staticmethod
    def _write_log(record: DeliverySession, text: str) -> None:
        if record.log_path:
            path = Path(record.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(text.rstrip() + "\n")

    def _call(
        self, role: str, name: str, prompt: str, *, system: str
    ) -> dict[str, Any]:
        record, client = self._session(role, name)
        self._write_log(record, f"PROMPT\n{prompt}")
        try:
            if self.agent_provider in {"c0d3r", "coder", "c0d3rv2"}:
                routed = run_delivery_turn_detailed(
                    prompt=(
                        "This is a bounded, read-only archival-research role. Do not "
                        "modify files or run project mutations. Complete the assigned "
                        "role and return the requested strict JSON object as the final "
                        f"answer.\n\n{prompt}"
                    ),
                    session_key=f"branddozer-research:{self.run.id}:{record.id}",
                    workdir=self.root,
                    backend=self.model_provider,
                    system_context=(
                        "Brand Dozer selected C0D3R V2 as the research agent. "
                        f"C0D3R V2 selected {self.model_provider} as its model backend. "
                        "Execution mode: bounded read-only archival-research role. "
                        f"{system}"
                    ),
                    reset=True,
                )
                output = str(routed.get("output") or "")
                record.meta = {
                    **(record.meta or {}),
                    "route_history": routed.get("route_history") or [],
                    "models": routed.get("models") or [],
                    "turn_model_calls": routed.get("turn_model_calls") or 0,
                    "tool_events": routed.get("tool_events") or [],
                }
            else:
                if client is None:
                    raise RuntimeError("research model client was not initialized")
                output = client.send(prompt, stream=False, system=system)
            self._write_log(record, f"OUTPUT\n{output}")
            payload = _extract_json(
                output, expected_keys=ROLE_SCHEMA_KEYS.get(role)
            )
            record.status = "done"
            record.completed_at = timezone.now()
            record.save(update_fields=["status", "completed_at", "meta"])
            return payload
        except Exception as exc:
            self._write_log(record, f"ERROR\n{exc}")
            record.status = "error"
            record.completed_at = timezone.now()
            record.meta = {**(record.meta or {}), "error": str(exc)}
            record.save(update_fields=["status", "completed_at", "meta"])
            raise

    def _plan(self) -> dict[str, Any]:
        requested_config = (self.run.context or {}).get("research_config") or {}
        return self._call(
            "research_planner",
            "Research protocol and Scrum planner",
            (
                "Convert the user's goal into a rigorous archival-research protocol. "
                "Return JSON with title, research_question, keywords, target_journal, "
                "scope, exclusion_criteria, search_strategy, and work_packages. "
                "work_packages must contain 3-8 independent objects with title, query, "
                "angle, deliverable, and acceptance_criteria. Include competing "
                "interpretations and negative evidence. Do not invent sources.\n\n"
                f"GOAL:\n{self.run.prompt}\n\n"
                f"REQUESTED PUBLICATION SETTINGS:\n{json.dumps(requested_config)}"
            ),
            system=(
                "You are a PhD research director and Scrum product owner. Return strict "
                "JSON only. This is archival research; do not propose experiments as if "
                "they were performed."
            ),
        )

    def _create_scrum(self, plan: dict[str, Any]) -> list[BacklogItem]:
        items: list[BacklogItem] = []
        for index, package in enumerate((plan.get("work_packages") or [])[:8], start=1):
            if not isinstance(package, dict):
                continue
            item = BacklogItem.objects.create(
                project=self.run.project,
                run=self.run,
                kind="story",
                title=str(package.get("title") or f"Evidence package {index}")[:240],
                description=json.dumps(package, indent=2),
                acceptance_criteria=package.get("acceptance_criteria") or [
                    "Identify authoritative archival sources",
                    "Extract traceable findings and counterevidence",
                    "Return claims with citation keys",
                ],
                priority=index,
                estimate_points=3,
                status="todo",
                source="research",
                meta={"research_package": package},
            )
            items.append(item)
        if not items:
            raise RuntimeError("research planner returned no executable work packages")
        sprint = Sprint.objects.create(
            project=self.run.project,
            run=self.run,
            number=self.run.sprint_count + 1,
            goal=f"Evidence synthesis for {str(plan.get('title') or self.run.prompt)[:180]}",
            status="active",
            started_at=timezone.now(),
            meta={"research": True, "packages": len(items)},
        )
        self.run.sprint_count += 1
        self.run.save(update_fields=["sprint_count"])
        for item in items:
            SprintItem.objects.create(
                sprint=sprint,
                backlog_item=item,
                status="todo",
                owner="literature_reviewer",
            )
        return items

    def _review_package(
        self, item: BacklogItem, plan: dict[str, Any]
    ) -> dict[str, Any]:
        package = (item.meta or {}).get("research_package") or {}
        discovered: list[dict[str, Any]] = []
        try:
            from tools.c0d3rV2.web_search import WebSearch

            discovered = WebSearch(
                None, delay_s=0.5, max_results=12
            ).discover(str(package.get("query") or item.title))
        except Exception:
            discovered = []
        result = self._call(
            "literature_reviewer",
            f"Literature review: {item.title[:80]}",
            (
                "Perform the assigned archival literature work package. Return strict "
                "JSON with findings, counterevidence, uncertainties, sources, and claims. "
                "Every source needs citation_key, exact title, authors, publication_year, "
                "publisher, URL, DOI when available, peer_reviewed, and a short "
                "verified_passage to locate. Every claim needs claim_text, section, and "
                "source_keys. Use only the independently discovered candidate URLs "
                "provided below. Copy a short supporting passage verbatim so it can be "
                "checked against the fetched document. Never fabricate a citation; omit "
                "anything you cannot identify precisely.\n\n"
                f"RESEARCH QUESTION: {plan.get('research_question')}\n"
                f"SEARCH STRATEGY: {json.dumps(plan.get('search_strategy') or {})}\n"
                f"WORK PACKAGE: {json.dumps(package)}\n"
                f"INDEPENDENTLY DISCOVERED CANDIDATES: {json.dumps(discovered)}"
            ),
            system=(
                "You are an archival literature-review agent. Distinguish source facts, "
                "interpretation, and uncertainty. Return JSON only."
            ),
        )
        item.status = "done"
        item.meta = {**(item.meta or {}), "result_summary": result.get("summary", "")}
        item.save(update_fields=["status", "meta", "updated_at"])
        SprintItem.objects.filter(backlog_item=item).update(status="done")
        return result

    def _collect_evidence(
        self, items: list[BacklogItem], plan: dict[str, Any]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        workers = min(self.policy.max_parallel_agents, len(items))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(self._review_package, item, plan): item for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    item.status = "blocked"
                    item.meta = {**(item.meta or {}), "error": str(exc)}
                    item.save(update_fields=["status", "meta", "updated_at"])
                    SprintItem.objects.filter(backlog_item=item).update(status="blocked")
                    raise
        return results

    def _verify_sources(
        self, evidence: list[dict[str, Any]], question: str
    ) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for package in evidence:
            for index, raw in enumerate(package.get("sources") or []):
                if not isinstance(raw, dict):
                    continue
                key = _clean_key(
                    raw.get("citation_key"), f"source-{len(unique) + index + 1}"
                )
                raw = {**raw, "citation_key": key}
                if key in unique and unique[key].get("url") != raw.get("url"):
                    key = _clean_key(f"{key}-{len(unique) + 1}", "source")
                    raw["citation_key"] = key
                unique[key] = raw
        verified: list[dict[str, Any]] = []
        workers = min(self.policy.max_parallel_agents, max(1, len(unique)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.source_verifier, self.harvester, source, question
                ): key
                for key, source in unique.items()
            }
            for future in as_completed(futures):
                verified.append(future.result())
        return sorted(verified, key=lambda item: str(item.get("citation_key") or ""))

    def _methods_review(
        self, plan: dict[str, Any], evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self._call(
            "methods_reviewer",
            "Methodology, bias, and validity review",
            (
                "Audit the archival research protocol and gathered evidence. Return JSON "
                "with method, inclusion_decisions, exclusion_decisions, bias_risks, "
                "validity_limits, unresolved_questions, and synthesis_rules. Do not "
                "claim that experiments or measurements were performed.\n\n"
                f"PROTOCOL: {json.dumps(plan)}\n"
                f"EVIDENCE SUMMARIES: {json.dumps(evidence)[:80000]}"
            ),
            system=(
                "You are a research-methods and systematic-review specialist. Return "
                "strict JSON only and expose limitations rather than hiding them."
            ),
        )

    def _write_candidate(
        self,
        plan: dict[str, Any],
        evidence: list[dict[str, Any]],
        sources: list[dict[str, Any]],
        methods: dict[str, Any],
        *,
        previous: str = "",
        revision_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instruction = (
            "Write a complete journal-style archival research paper. Return JSON with "
            "title, abstract, keywords, markdown, and claims. markdown must contain "
            "Abstract, Keywords, Introduction, Methodology, Literature Review, Findings, "
            "Discussion, Limitations, Conclusion, and References. Cite only the supplied "
            "citation keys using [@key]. Each claims item must have section, claim_text, "
            "source_keys, verification_status ('supported' or 'qualified'), and rationale. "
            "Clearly label inference, disagreement, uncertainty, and the absence of "
            "primary evidence. Never report original experiments."
        )
        if previous:
            instruction += (
                " Rewrite the full prior paper to resolve every supplied deterministic "
                "gate and peer-review issue; do not merely append an errata section."
            )
        return self._call(
            "research_writer",
            "Research synthesis and paper revision" if previous else "Research synthesis",
            (
                f"{instruction}\n\nPROTOCOL: {json.dumps(plan)}\n"
                f"METHODS REVIEW: {json.dumps(methods)}\n"
                f"VERIFIED SOURCE REGISTRY: {json.dumps(sources)}\n"
                f"EVIDENCE: {json.dumps(evidence)[:120000]}\n"
                f"PREVIOUS PAPER: {previous[:120000]}\n"
                f"REVISION FEEDBACK: {json.dumps(revision_feedback or {})}"
            ),
            system=(
                "You are the senior academic author. Return strict JSON only. Citation "
                "precision and qualified conclusions take priority over persuasive prose."
            ),
        )

    def _audit(
        self,
        candidate: dict[str, Any],
        sources: list[dict[str, Any]],
        methods: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        citation_review = self._call(
            "citation_auditor",
            "Claim and citation audit",
            (
                "Audit each submitted claim against the verified source registry and "
                "paper context. Return JSON with claims, where every item repeats "
                "claim_text and source_keys and sets verification_status to supported, "
                "qualified, or rejected with rationale. Also return fabricated_or_unknown "
                "citation keys and blocking_issues. A source URL alone is not support.\n\n"
                f"SOURCES: {json.dumps(sources)}\n"
                f"CLAIMS: {json.dumps(candidate.get('claims') or [])}\n"
                f"PAPER: {str(candidate.get('markdown') or '')[:120000]}"
            ),
            system=(
                "You are a hostile citation auditor. Return strict JSON only. Reject "
                "claims whose supplied evidence does not actually support them."
            ),
        )
        claims = [
            claim
            for claim in (citation_review.get("claims") or [])
            if isinstance(claim, dict)
        ]
        peer_review = self._call(
            "peer_reviewer",
            "Independent journal peer review",
            (
                "Perform an independent journal peer review. Return JSON with "
                "recommendation (accept, minor_revision, major_revision, reject), "
                "blocking_issues, nonblocking_issues, validity_assessment, novelty, "
                "methods_assessment, evidence_assessment, and required_rewrites. Evaluate "
                "only archival claims; do not demand invented experiments.\n\n"
                f"METHODS: {json.dumps(methods)}\n"
                f"CITATION AUDIT: {json.dumps(citation_review)}\n"
                f"PAPER: {str(candidate.get('markdown') or '')[:120000]}"
            ),
            system=(
                "You are an independent senior peer reviewer. Return strict JSON only. "
                "Surface hallucinations, overclaiming, missing counterevidence, and "
                "methodological weaknesses."
            ),
        )
        citation_blockers = list(citation_review.get("blocking_issues") or [])
        if citation_review.get("fabricated_or_unknown"):
            citation_blockers.append(
                f"Unknown citations: {citation_review['fabricated_or_unknown']}"
            )
        if citation_blockers:
            peer_review["blocking_issues"] = list(
                dict.fromkeys(
                    list(peer_review.get("blocking_issues") or [])
                    + [str(item) for item in citation_blockers]
                )
            )
        return claims, peer_review

    def _persist_sources(
        self, paper: ResearchPaper, sources: list[dict[str, Any]]
    ) -> None:
        paper.sources.all().delete()
        for source in sources:
            try:
                year = int(source.get("publication_year") or 0) or None
            except (TypeError, ValueError):
                year = None
            ResearchSource.objects.create(
                paper=paper,
                citation_key=_clean_key(source.get("citation_key"), "source"),
                title=str(source.get("title") or source.get("url") or "Untitled source"),
                authors=(
                    source.get("authors")
                    if isinstance(source.get("authors"), list)
                    else []
                ),
                publication_year=year,
                publisher=str(source.get("publisher") or ""),
                url=str(source.get("url") or ""),
                doi=str(source.get("doi") or ""),
                retrieved_at=timezone.now() if source.get("retrieved_at") else None,
                content_sha256=str(source.get("content_sha256") or ""),
                authority_tier=int(source.get("authority_tier") or 0),
                peer_reviewed=bool(source.get("peer_reviewed")),
                archival=True,
                verified_passage=str(source.get("verified_passage") or "")[:12000],
                verification_status=str(source.get("verification_status") or "pending"),
                verification_detail=str(source.get("verification_detail") or ""),
            )

    def _persist_claims(
        self, paper: ResearchPaper, claims: list[dict[str, Any]]
    ) -> None:
        paper.claims.all().delete()
        for claim in claims:
            ResearchClaim.objects.create(
                paper=paper,
                section=str(claim.get("section") or "")[:160],
                claim_text=str(claim.get("claim_text") or ""),
                source_keys=[
                    str(key) for key in (claim.get("source_keys") or []) if str(key)
                ],
                verification_status=str(
                    claim.get("verification_status") or "pending"
                ),
                rationale=str(claim.get("rationale") or ""),
            )

    def execute(self) -> ResearchPaper:
        self.run.status = "running"
        self.run.phase = "research_planning"
        self.run.started_at = self.run.started_at or timezone.now()
        self.run.error = ""
        self.run.save(update_fields=["status", "phase", "started_at", "error"])
        plan = self._plan()
        items = self._create_scrum(plan)
        self.run.phase = "research_evidence"
        self.run.save(update_fields=["phase"])
        evidence = self._collect_evidence(items, plan)
        question = str(plan.get("research_question") or self.run.prompt)
        sources = self._verify_sources(evidence, question)
        methods = self._methods_review(plan, evidence)
        paper = ResearchPaper.objects.create(
            project=self.run.project,
            run=self.run,
            title=str(plan.get("title") or self.run.project.name)[:500],
            research_question=question,
            keywords=(
                plan.get("keywords")
                if isinstance(plan.get("keywords"), list)
                else []
            ),
            target_journal=str(plan.get("target_journal") or "")[:255],
            citation_style=str(
                ((self.run.context or {}).get("research_config") or {}).get(
                    "citation_style", "apa"
                )
            )[:32],
            status="draft",
        )
        self._persist_sources(paper, sources)

        candidate: dict[str, Any] = {}
        report: dict[str, Any] = {}
        previous = ""
        feedback: dict[str, Any] = {}
        for revision in range(1, self.policy.max_revision_rounds + 1):
            self.run.phase = f"research_revision_{revision}"
            self.run.iteration = revision
            self.run.save(update_fields=["phase", "iteration"])
            candidate = self._write_candidate(
                plan,
                evidence,
                sources,
                methods,
                previous=previous,
                revision_feedback=feedback,
            )
            markdown = str(candidate.get("markdown") or "")
            claims, peer_review = self._audit(candidate, sources, methods)
            report = validate_paper_payload(
                markdown=markdown,
                sources=sources,
                claims=claims,
                policy=self.policy,
                peer_review=peer_review,
            )
            report.update(
                {
                    "revision": revision,
                    "peer_review": peer_review,
                    "policy": self.policy.__dict__,
                }
            )
            ResearchPaperRevision.objects.create(
                paper=paper,
                version=revision,
                content_markdown=markdown,
                change_summary=str(candidate.get("change_summary") or ""),
                validation_report=report,
            )
            paper.version = revision
            paper.title = str(candidate.get("title") or paper.title)[:500]
            paper.abstract = str(candidate.get("abstract") or "")
            if isinstance(candidate.get("keywords"), list):
                paper.keywords = candidate["keywords"]
            paper.content_markdown = markdown
            paper.word_count = _word_count(markdown)
            paper.content_sha256 = hashlib.sha256(
                markdown.encode("utf-8")
            ).hexdigest()
            paper.validation_report = report
            paper.status = "validated" if report["passed"] else "revision_required"
            paper.validated_at = timezone.now() if report["passed"] else None
            paper.save()
            self._persist_claims(paper, claims)
            if report["passed"]:
                break
            previous = markdown
            feedback = report

        persist_paper_files(paper)
        DeliveryArtifact.objects.create(
            project=self.run.project,
            run=self.run,
            kind="research_paper",
            title=paper.title[:200],
            path=paper.pdf_path,
            content=paper.abstract,
            data={
                "paper_id": str(paper.id),
                "version": paper.version,
                "status": paper.status,
                "validation": report,
                "markdown_path": paper.markdown_path,
                "pdf_path": paper.pdf_path,
            },
        )
        sprint = Sprint.objects.filter(run=self.run).order_by("-number").first()
        if sprint:
            sprint.status = "complete" if report.get("passed") else "review"
            sprint.completed_at = timezone.now() if report.get("passed") else None
            sprint.retrospective = json.dumps(
                {
                    "paper_id": str(paper.id),
                    "revision": paper.version,
                    "validation": report,
                },
                indent=2,
            )
            sprint.save(update_fields=["status", "completed_at", "retrospective"])
        if report.get("passed"):
            self.run.status = (
                "awaiting_acceptance"
                if self.run.acceptance_required
                else "complete"
            )
            self.run.phase = (
                "awaiting_acceptance"
                if self.run.acceptance_required
                else "research_complete"
            )
            self.run.completed_at = (
                None if self.run.acceptance_required else timezone.now()
            )
        else:
            self.run.status = "blocked"
            self.run.phase = "research_validation"
            self.run.error = (
                "Paper did not satisfy publication-readiness gates after "
                f"{self.policy.max_revision_rounds} revisions."
            )
            self.run.completed_at = timezone.now()
        context = dict(self.run.context or {})
        context["research_paper_id"] = str(paper.id)
        context["research_validation"] = report
        self.run.context = context
        self.run.save(
            update_fields=[
                "status", "phase", "error", "completed_at", "context",
            ]
        )
        delivery = DeliveryProject.objects.filter(project=self.run.project).first()
        if delivery:
            delivery.status = (
                "complete"
                if self.run.status in {"complete", "awaiting_acceptance"}
                else "blocked"
            )
            delivery.active_run = self.run
            delivery.save(update_fields=["status", "active_run", "updated_at"])
        return paper


def run_research_workflow(run: DeliveryRun, root: Path) -> ResearchPaper:
    """Entry point used by the background delivery worker."""
    close_old_connections()
    try:
        return ResearchWorkflow(run, root).execute()
    finally:
        close_old_connections()
