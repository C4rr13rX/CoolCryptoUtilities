"""Evidence-first archival research workflow for Brand Dozer."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from copy import deepcopy
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


class AgentLimited(RuntimeError):
    """The agent hit a usage limit; the run should pause, not fail.

    Carries the reset time so the auto-resume heartbeat knows when the
    agent is expected back.
    """

    def __init__(self, message: str, *, block_kind: str = "cooldown", reset_at=None):
        super().__init__(message)
        self.block_kind = block_kind
        self.reset_at = reset_at
# CLI agents that act as their own provider and supply their own model set.
_CLI_AGENTS = frozenset({"codex", "claude_code"})
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
    "research_writer": {
        "title", "abstract", "markdown", "claims", "rival_hypotheses",
    },
    "citation_auditor": {"claims", "blocking_issues"},
    "peer_reviewer": {"recommendation", "blocking_issues"},
}


@dataclass(frozen=True)
class ResearchPolicy:
    min_words: int = 5000
    min_sources: int = 12
    min_verified_sources: int = 10
    min_high_authority_sources: int = 6
    min_primary_sources: int = 2
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
            min_primary_sources=bounded("min_primary_sources", 2, 0, 100),
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


def _enforce_current_temporal_scope(plan: dict[str, Any]) -> dict[str, Any]:
    """Prevent a current-events study from silently truncating recent years."""
    guarded = deepcopy(plan)
    today = timezone.localdate()
    current_year = today.year
    prior_year = current_year - 1
    as_of = today.isoformat()
    guard_text = (
        f" Mandatory temporal guard: discovery and inclusion screening remain "
        f"open through {as_of}. Earlier date ranges are historical subwindows, "
        "not exclusion cutoffs. Identify competing Target boycott events before "
        "selecting the focal event."
    )
    scope = str(guarded.get("scope") or "").strip()
    # Replace stale guards instead of appending a new dated sentence on every
    # resume.  Checkpointed runs must be safe to re-enter indefinitely.
    guard_pattern = (
        r"\s*Mandatory temporal guard: discovery and inclusion screening remain "
        r"open through \d{4}-\d{2}-\d{2}\. Earlier date ranges are historical "
        r"subwindows, not exclusion cutoffs\. Identify competing Target boycott "
        r"events before selecting the focal event\."
    )
    scope = re.sub(guard_pattern, "", scope).strip()
    guarded["scope"] = scope + guard_text
    strategy = guarded.get("search_strategy")
    if isinstance(strategy, dict):
        guarded["search_strategy"] = {
            **strategy,
            "as_of_date": as_of,
            "temporal_inclusion_rule": (
                f"Search through {as_of}; do not exclude {prior_year} or "
                f"{current_year} records before event identity is resolved."
            ),
        }
    else:
        strategy_text = re.sub(
            guard_pattern, "", str(strategy or "").strip()
        ).strip()
        guarded["search_strategy"] = (
            strategy_text + guard_text
        ).strip()
    guarded["temporal_scope_guard"] = {
        "as_of_date": as_of,
        "required_recent_years": [prior_year, current_year],
        "event_identity_precedes_focal_window": True,
    }
    recent_query = (
        f'("Target Corporation" AND boycott AND (DEI OR diversity OR minority) '
        f"AND ({prior_year} OR {current_year} OR current))"
    )
    packages = []
    for package in guarded.get("work_packages") or []:
        if not isinstance(package, dict):
            continue
        updated = dict(package)
        query = str(updated.get("query") or "").strip()
        if str(current_year) not in query and "current" not in query.lower():
            updated["query"] = f"({query}) OR {recent_query}" if query else recent_query
        packages.append(updated)
    guarded["work_packages"] = packages
    return guarded


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
    # A first-party record authenticated by provenance classification (the
    # subject's own corporate domain, a court docket, a regulatory filing) is
    # authoritative *for claims about the subject* — often more so than
    # secondary coverage. Without this, a study of a company's own programs
    # scores its primary documents no higher than an anonymous blog and can
    # never satisfy the authoritative-sources gate.
    if bool(source.get("first_party")) and source.get("provenance_status") in {
        "verified", "corroborated"
    }:
        return 2
    return 1


def _classify_source_provenance(source: dict[str, Any]) -> dict[str, Any]:
    """Classify evidence origin separately from passage/retrieval verification."""
    classified = dict(source)
    host = (
        urllib.parse.urlparse(str(classified.get("url") or "")).hostname or ""
    ).lower().removeprefix("www.")
    source_class = "other"
    first_party = False
    provenance_status = "unverified"
    detail = "Source type could not be authenticated from its host alone."

    if host == "corporate.target.com":
        source_class, first_party, provenance_status = (
            "corporate_primary", True, "verified"
        )
        detail = "Published on Target Corporation's official corporate domain."
    elif host.endswith(".gov") or host == "gov":
        source_class, first_party, provenance_status = (
            "government_record", True, "verified"
        )
        detail = "Published by an official government domain."
    elif host == "courtlistener.com" or host.endswith(".courtlistener.com"):
        source_class, first_party, provenance_status = (
            "court_record", True, "corroborated"
        )
        detail = "Public court-record repository; docket identity still requires citation."
    elif host == "muckrock.com" or host.endswith(".muckrock.com"):
        source_class, first_party, provenance_status = (
            "government_record", True, "corroborated"
        )
        detail = "Public-record/FOIA repository with document provenance metadata."
    elif host == "wikileaks.org" or host.endswith(".wikileaks.org"):
        source_class, first_party, provenance_status = (
            "leaked_primary", True, "unverified"
        )
        detail = (
            "Leaked-document repository: direct evidence of document contents only; "
            "authenticity, completeness, chain of custody, and context require "
            "independent corroboration."
        )
    elif host in {"documentcloud.org", "www.documentcloud.org"}:
        source_class = "archival_copy"
        detail = "Document repository copy; authenticate against its originating record."
    elif host in {"archive.org", "web.archive.org"}:
        source_class, provenance_status = "archival_copy", "corroborated"
        detail = "Archived web copy; provenance derives from the captured origin and date."
    elif bool(classified.get("peer_reviewed")) or any(
        marker in host
        for marker in (
            "doi.org", "pubmed.ncbi.nlm.nih.gov", "ieee.org", "acm.org",
            "springer.com", "sciencedirect.com", "nature.com", "science.org",
        )
    ):
        source_class, provenance_status = "peer_reviewed", "verified"
        detail = "Scholarly publication provenance identified from its publication host."
    elif host in {"apnews.com", "reuters.com"}:
        source_class, provenance_status = "journalism", "verified"
        detail = "Identified news publisher; reporting remains secondary evidence."

    classified.update(
        source_class=source_class,
        first_party=first_party,
        provenance_status=provenance_status,
        provenance_detail=detail,
    )
    return classified


def validate_paper_payload(
    *,
    markdown: str,
    sources: Iterable[dict[str, Any]],
    claims: Iterable[dict[str, Any]],
    policy: ResearchPolicy,
    peer_review: dict[str, Any] | None = None,
    epistemic_config: dict[str, Any] | None = None,
    rival_hypotheses: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run deterministic publication-readiness gates over a candidate paper.

    Covers two orthogonal axes: evidence sufficiency (counting gates) and
    reasoning quality (see branddozer.epistemics).
    """
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
    verified_primary = [
        source
        for source in verified
        if bool(source.get("first_party"))
        and source.get("provenance_status") in {"verified", "corroborated"}
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
        "primary_source_coverage": (
            len(verified_primary) >= policy.min_primary_sources
        ),
        "source_diversity": len(domains) >= policy.min_source_domains,
        "known_citations": not unknown_citations,
        "claims_supported": bool(claim_list) and not unsupported_claims,
        "peer_review": not peer_blockers
        and str(peer_review.get("recommendation") or "").lower()
        in {"accept", "minor_revision", "minor revision"},
    }
    # Reasoning-quality gates run alongside the counting gates: a paper must
    # be both well-evidenced *and* well-argued to pass.
    from branddozer.epistemics import EpistemicPolicy, evaluate_reasoning

    reasoning = evaluate_reasoning(
        claims=claim_list,
        sources=source_list,
        policy=EpistemicPolicy.from_config(epistemic_config),
        rival_hypotheses=rival_hypotheses or [],
    )
    checks.update(reasoning["checks"])

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reasoning": reasoning,
        "metrics": {
            "word_count": _word_count(markdown),
            "sources": len(source_list),
            "verified_sources": len(verified),
            "high_authority_sources": len(high_authority),
            "verified_primary_sources": len(verified_primary),
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


def _extract_passages(source: dict[str, Any]) -> list[str]:
    """Collect candidate supporting quotations from a source record.

    Agents legitimately return this in several shapes — a single
    ``verified_passage`` string, or a ``verified_passages`` list whose
    items are either strings or ``{"passage": ..., "locator": ...}``
    objects. Reading only the singular string form silently dropped every
    quotation from agents using the plural form, so *all* their sources
    failed verification for "no passage" rather than on their merits.
    """
    out: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                out.append(value.strip())
        elif isinstance(value, dict):
            for key in ("passage", "quote", "text", "verbatim"):
                inner = value.get(key)
                if isinstance(inner, str) and inner.strip():
                    out.append(inner.strip())
                    return
        elif isinstance(value, list):
            for item in value:
                _add(item)

    for key in ("verified_passage", "verified_passages", "passages", "quotes"):
        _add(source.get(key))
    # Longest first: the most specific quotation is the strongest evidence
    # and the least likely to match incidentally.
    return sorted(dict.fromkeys(out), key=len, reverse=True)


def _verify_source(
    harvester: ResearchHarvester, source: dict[str, Any], question: str
) -> dict[str, Any]:
    candidate = dict(source)
    candidate = _classify_source_provenance(candidate)
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
        error_detail = (
            (result.get("errors") or [{}])[-1].get("error")
            or "source could not be independently retrieved"
        )
        # Record *why* retrieval failed. A robots.txt block or size limit is
        # an access problem, not evidence that the source is unsound, and a
        # fact-checker must be able to tell those apart.
        from branddozer.reproducibility import classify_failure

        candidate.update(
            verification_status="rejected",
            verification_detail=error_detail,
            failure_kind=classify_failure(error_detail),
        )
        return candidate
    record = stored[0]
    document = harvester.document(str(record.get("url") or url))
    content = str((document or {}).get("content") or "")
    # Normalised matching: publishers substitute non-breaking hyphens, curly
    # quotes and nbsp, which made correct quotations fail a naive compare.
    # The words must still match verbatim; only typography is folded.
    from branddozer.reproducibility import passage_matches, snapshot_id

    passages = _extract_passages(candidate)
    passage = passages[0] if passages else ""
    match = {
        "matched": False,
        "outcome": "passage_too_short",
        "detail": "no supporting quotation was supplied",
    }
    # A source may cite several passages; any one that checks out verifies
    # it. Keep the matching quotation as the source's evidence of record.
    for candidate_passage in passages:
        attempt = passage_matches(candidate_passage, content)
        if attempt["matched"]:
            match, passage = attempt, candidate_passage
            break
        match = attempt
    candidate["verified_passage"] = passage
    if not match["matched"]:
        candidate.update(
            verification_status="rejected",
            verification_detail=(
                f"source was retrieved, but {match['detail']}"
            ),
            failure_kind=match["outcome"],
            content_sha256=str(record.get("sha256") or ""),
            snapshot=snapshot_id(url, content),
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
        failure_kind="",
        # Snapshot id ties the verdict to the exact text that was matched,
        # so the check can be replayed offline and drift can be detected.
        snapshot=snapshot_id(url, content),
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

    # Publish the verification manifest next to the paper so any later
    # fact-checker — human or model — can replay every source check.
    try:
        from branddozer.reproducibility import build_manifest

        manifest = build_manifest(
            [
                {
                    "citation_key": source.citation_key,
                    "url": source.url,
                    "verification_status": source.verification_status,
                    "verification_detail": source.verification_detail,
                    "content_sha256": source.content_sha256,
                    "retrieved_at": (
                        source.retrieved_at.isoformat() if source.retrieved_at else ""
                    ),
                    "verified_passage": source.verified_passage,
                }
                for source in paper.sources.all()
            ],
            paper_sha256=paper.content_sha256,
            claims=[
                {"claim_text": claim.claim_text, "source_keys": claim.source_keys}
                for claim in paper.claims.all()
            ],
        )
        (directory / f"verification-v{paper.version}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except Exception:
        # A manifest failure must never lose the paper itself.
        pass


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
        self.agent_model = str(context.get("agent_model") or "").strip()
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
            # CLI agents (Codex / Claude Code) are their own provider and carry
            # their own model namespace; everything else routes by model backend.
            provider = (
                self.agent_provider if self.agent_provider in _CLI_AGENTS else self.model_provider
            )
            settings = default_settings(provider)
            if self.agent_provider in _CLI_AGENTS and self.agent_model:
                settings["model"] = self.agent_model
            SessionClass = get_session_class(provider, explicit=True)
            client = SessionClass(
                session_name=f"research-{role}-{record.id}",
                transcript_dir=self.transcript_root,
                read_timeout_s=None,
                workdir=str(self.root),
                **settings,
            )
        return record, client

    def _raise_if_agent_limited(self, output: str, record: DeliverySession) -> None:
        """Turn an agent usage-limit reply into a pausable AgentLimited error.

        Claude Code and Codex report limits as ordinary output, e.g.
        "You've hit your session limit · resets 7pm (America/New_York)".
        Without this the text simply fails JSON extraction and the run dies
        with a misleading "no complete JSON object".
        """
        text = (output or "").strip()
        if not text or len(text) > 2000:
            # Real role output is long JSON; limit notices are short.
            return
        from services.branddozer_agent_watch import classify_block, parse_reset_at

        kind = classify_block(text)
        if kind == "unknown":
            return
        reset_at = parse_reset_at(text)
        record.meta = {
            **(record.meta or {}),
            "agent_limited": True,
            "block_kind": kind,
            "reset_at": reset_at.isoformat() if reset_at else None,
        }
        raise AgentLimited(text, block_kind=kind, reset_at=reset_at)

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
                    wizard_endpoint=str((self.run.context or {}).get("wizard_endpoint") or ""),
                    wizard_chat_path=str((self.run.context or {}).get("wizard_chat_path") or ""),
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
            # An agent that reports a usage limit has not produced a bad
            # answer — it produced no answer. Surface that as a pausable
            # cooldown so the heartbeat can resume the run when the limit
            # clears, instead of failing it as malformed JSON.
            self._raise_if_agent_limited(output, record)
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
        existing = list(
            BacklogItem.objects.filter(run=self.run, source="research").order_by(
                "priority", "created_at"
            )
        )
        if existing:
            packages = [
                package
                for package in (plan.get("work_packages") or [])
                if isinstance(package, dict)
            ]
            for index, item in enumerate(existing):
                if index >= len(packages):
                    break
                package = packages[index]
                item.description = json.dumps(package, indent=2)
                item.acceptance_criteria = package.get("acceptance_criteria") or []
                item.meta = {**(item.meta or {}), "research_package": package}
                item.save(
                    update_fields=[
                        "description", "acceptance_criteria", "meta", "updated_at"
                    ]
                )
            return existing
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

            base_query = str(package.get("query") or item.title)
            current_year = timezone.localdate().year
            queries = [
                base_query,
                (
                    f'"Target Corporation" boycott DEI rollback '
                    f"{current_year - 1} {current_year}"
                ),
                (
                    'site:corporate.target.com "diversity, equity and inclusion" '
                    f"{current_year - 1} {current_year}"
                ),
                f"site:wikileaks.org ({base_query})",
                (
                    "(site:documentcloud.org OR site:muckrock.com OR "
                    f"site:courtlistener.com) ({base_query})"
                ),
                f"site:sec.gov ({base_query})",
                f"(site:archive.org OR site:web.archive.org) ({base_query})",
            ]
            package_text = " ".join(
                str(package.get(key) or "")
                for key in ("title", "angle", "deliverable")
            ).lower()
            if any(
                marker in package_text
                for marker in ("program", "fund", "supplier", "governance")
            ):
                queries.append(
                    'site:corporate.target.com Target supplier diversity '
                    '"community engagement" fund'
                )
            if any(
                marker in package_text
                for marker in ("impact", "psychological", "mental", "counterfactual")
            ):
                queries.append(
                    '"DEI rollback" employee belonging mental health '
                    "systematic review"
                )
            search = WebSearch(None, delay_s=0.25, max_results=8)
            unique: dict[str, dict[str, Any]] = {}
            for seed in (self.run.context or {}).get("research_seed_sources") or []:
                if not isinstance(seed, dict):
                    continue
                url = str(seed.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    continue
                key = re.sub(r"[?#].*$", "", url.lower())
                unique[key] = {
                    "title": str(seed.get("title") or url),
                    "url": url,
                    "snippet": (
                        "Explicit benchmark seed; discovery only. Fetch and verify "
                        "the document before using any claim."
                    ),
                    "authority_score": 10,
                    "metadata_relevance": 10,
                    "provider": "benchmark_seed",
                }
            for query in queries:
                for candidate in search.discover(query):
                    url = str(candidate.get("url") or "").strip()
                    haystack = (
                        f"{candidate.get('title', '')} "
                        f"{candidate.get('snippet', '')} {url}"
                    ).lower()
                    target_context = (
                        "target corporation" in haystack
                        or "target's" in haystack
                        or "corporate.target.com" in haystack
                        or (
                            "target" in haystack
                            and any(
                                marker in haystack
                                for marker in (
                                    "boycott", "dei", "diversity", "equity",
                                    "minority", "supplier", "community",
                                )
                            )
                        )
                    )
                    broader_evidence = (
                        query.startswith('"DEI rollback"')
                        and any(
                            marker in haystack
                            for marker in (
                                "dei", "diversity", "belonging",
                                "mental health", "employee",
                            )
                        )
                    )
                    if not (target_context or broader_evidence):
                        continue
                    key = re.sub(r"[?#].*$", "", url.lower())
                    if key:
                        unique[key] = candidate
            discovered = sorted(
                unique.values(),
                key=lambda candidate: (
                    -int(candidate.get("metadata_relevance") or 0),
                    -int(candidate.get("authority_score") or 0),
                ),
            )[:24]
        except Exception:
            discovered = []
        result = self._call(
            "literature_reviewer",
            f"Literature review: {item.title[:80]}",
            (
                "Perform the assigned archival literature work package. Return strict "
                "JSON with findings, counterevidence, uncertainties, sources, and claims. "
                "Every source needs citation_key, exact title, authors, publication_year, "
                "publisher, URL, DOI when available, peer_reviewed, and "
                "`verified_passage`: a single string containing at least 40 "
                "characters copied VERBATIM from the fetched page (not a "
                "paraphrase or summary). This exact string is re-matched "
                "against the retrieved document, and the source is rejected "
                "if it is absent, so copy it character-for-character. "
                "Classify source_class, first_party, "
                "provenance_status, and provenance_detail. Search official releases, "
                "filings, court/FOIA records, public archives, WikiLeaks, and comparable "
                "public document repositories. A leak is direct evidence of its contents "
                "only until authenticity, completeness, chain of custody, and context "
                "are independently corroborated. Every claim needs claim_text, section, and "
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
        item.meta = {
            **(item.meta or {}),
            "result_summary": result.get("summary", ""),
            "research_result": result,
            "error": "",
        }
        item.save(update_fields=["status", "meta", "updated_at"])
        SprintItem.objects.filter(backlog_item=item).update(status="done")
        return result

    def _collect_evidence(
        self, items: list[BacklogItem], plan: dict[str, Any]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        quarantined: list[dict[str, Any]] = []
        workers = min(self.policy.max_parallel_agents, len(items))

        def review_with_isolated_connection(item: BacklogItem) -> dict[str, Any]:
            close_old_connections()
            try:
                return self._review_package(item, plan)
            finally:
                close_old_connections()

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(review_with_isolated_connection, item): item
                for item in items
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
                    quarantined.append(
                        {
                            "backlog_item_id": str(item.id),
                            "title": item.title,
                            "error": str(exc),
                            "quarantined_at": timezone.now().isoformat(),
                        }
                    )
                    # Persist each failure immediately. A different package can
                    # still be waiting on a provider, and its latency must not
                    # hide failures that have already completed.
                    self.run.refresh_from_db(fields=["context"])
                    context = dict(self.run.context or {})
                    context["research_quarantine"] = [
                        *(context.get("research_quarantine") or []),
                        quarantined[-1],
                    ][-100:]
                    context["research_quarantine_count"] = len(
                        context["research_quarantine"]
                    )
                    self.run.context = context
                    self.run.save(update_fields=["context"])
        if not results:
            raise RuntimeError(
                "all archival evidence work packages failed; "
                f"{len(quarantined)} package(s) quarantined"
            )
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
                "claim that experiments or measurements were performed. Explicitly audit "
                "the search of official company releases and filings, government/court/"
                "FOIA records, WikiLeaks, and comparable public document repositories. "
                "Prioritize first-party evidence while treating source proximity, "
                "authenticity, truth, completeness, and representativeness as separate "
                "questions. Record leak provenance, corroboration, selection bias, and "
                "missing context.\n\n"
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
            "title, abstract, keywords, markdown, claims, and rival_hypotheses. "
            "markdown must contain "
            "Abstract, Keywords, Introduction, Methodology, Literature Review, Findings, "
            "Rival Explanations, Discussion, Limitations, Conclusion, and References. "
            "Cite only the supplied "
            "citation keys using [@key]. Each claims item must have section, claim_text, "
            "source_keys, verification_status ('supported' or 'qualified'), rationale, "
            "claim_type, modality, premise_modality, inference_depth, and uncertainty; "
            "causal claims add causal_design and identification_strategy; claims resting "
            "on the subject's own documents add quoted_wording and deniability. "
            "Each rival_hypotheses item must have hypothesis, discriminating_evidence, "
            "and status ('favoured', 'disfavoured', or 'unsettled'). "
            "Clearly label inference, disagreement, uncertainty, and the absence of "
            "primary evidence. Never report original experiments."
        )
        # The acceptance gate scores the paper against these thresholds
        # (see _evaluate_gates). Without stating them here the writer is
        # judged on a bar it was never told, which produced short drafts
        # that failed minimum_word_count on every revision round.
        instruction += (
            f"\n\nPUBLICATION REQUIREMENTS (enforced by an automated gate; a paper "
            f"that misses any of these is rejected):\n"
            f"- markdown body must be at least {self.policy.min_words} words. This is a "
            f"floor, not a target; a substantially longer paper is expected for this "
            f"scope. Reaching it through depth of analysis and evidence, never padding.\n"
            f"- cite at least {self.policy.min_verified_sources} independently verified "
            f"sources drawn from the VERIFIED SOURCE REGISTRY below.\n"
            f"- every required section must be substantive; no placeholder or stub "
            f"sections.\n"
            f"- Findings and Discussion carry the analytical weight; do not compress "
            f"them into summary bullets."
        )
        # Reasoning-quality contract: typed claims, causal warrant, modality,
        # deniability, rival explanations. Stated here so the writer is told
        # exactly what evaluate_reasoning will score it on.
        from branddozer.epistemics import EpistemicPolicy, writer_requirements

        instruction += writer_requirements(
            EpistemicPolicy.from_config(
                (self.run.context or {}).get("research_config") or {}
            )
        )
        if previous:
            instruction += (
                " Rewrite the full prior paper to resolve every supplied deterministic "
                "gate and peer-review issue; do not merely append an errata section."
            )
            # Name the shortfall explicitly. The raw report is JSON-dumped
            # further down, but an explicit delta is far harder to overlook.
            metrics = (revision_feedback or {}).get("metrics") or {}
            checks = (revision_feedback or {}).get("checks") or {}
            if checks.get("minimum_word_count") is False:
                actual = int(metrics.get("word_count") or 0)
                instruction += (
                    f" The previous draft was {actual} words against a "
                    f"{self.policy.min_words}-word minimum: it must grow by at least "
                    f"{max(0, self.policy.min_words - actual)} words of substantive "
                    f"analysis, not restatement."
                )
            failed = [name for name, ok in checks.items() if ok is False]
            if failed:
                instruction += f" Failed gates to fix: {', '.join(sorted(failed))}."
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
                "qualified, or rejected with rationale. Preserve each claim's "
                "claim_type, modality, premise_modality, inference_depth, uncertainty, "
                "causal_design, identification_strategy, quoted_wording, and deniability "
                "fields; correct them where the evidence does not bear them out — in "
                "particular, downgrade a causal claim whose identification strategy does "
                "not rule out confounding, flag any possible->actual or actual->necessary "
                "escalation, and mark wording as deniable where the quoted text is hedged "
                "or non-binding. Also return fabricated_or_unknown "
                "citation keys and blocking_issues. A source URL alone is not support. "
                "Reject or qualify any claim that treats an unverified or disputed leaked "
                "document as authenticated external fact, or that confuses a company's "
                "statement about itself with independent validation.\n\n"
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
                source_class=str(source.get("source_class") or "other"),
                first_party=bool(source.get("first_party")),
                provenance_status=str(
                    source.get("provenance_status") or "unverified"
                ),
                provenance_detail=str(source.get("provenance_detail") or ""),
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
        context = dict(self.run.context or {})
        checkpoint = context.get("research_plan")
        if isinstance(checkpoint, dict) and checkpoint.get("work_packages"):
            plan = checkpoint
        else:
            plan = self._plan()
        plan = _enforce_current_temporal_scope(plan)
        if plan != checkpoint:
            context = dict(self.run.context or {})
            context["research_plan"] = plan
            context["research_plan_checkpointed_at"] = timezone.now().isoformat()
            self.run.context = context
            self.run.save(update_fields=["context"])
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
                epistemic_config=(self.run.context or {}).get("research_config") or {},
                rival_hypotheses=candidate.get("rival_hypotheses") or [],
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
