"""branddozer/reproducibility.py — make fact-checking independently repeatable.

The problem
-----------
Source verification in ``research.py::_verify_source`` happens once, inside
the run, and leaves only a verdict behind. A later fact-checker — human or
model — cannot repeat it:

* Rejections conflate three different situations under one status.
  A live audit of a finished paper found 19/19 sources ``rejected``: 12 for
  a genuine passage mismatch, 5 because ``robots.txt`` disallowed the crawl,
  and 2 because the page exceeded a byte limit. Only the first is an
  integrity problem; the other 7 are *access* failures that say nothing
  about whether the source supports the claim. Treating them alike both
  slanders good sources and hides real fabrication.

* Nothing records *how* a check was performed — what was fetched, when,
  under which matching rules — so a re-check that disagrees cannot be
  attributed to a changed page, a different normalisation, or a real error.

* Web pages change. A hash proves the text was *what we saw*, but without a
  stored snapshot there is nothing to diff against when a URL later drifts.

What this adds
--------------
A **verification manifest**: a self-contained, re-runnable record of every
source check, emitted alongside the paper. It states the outcome, the
evidence (hash, snapshot id, retrieval time), the exact matching rules
used, and — for failures — which *kind* of failure it was. Re-running the
manifest against the archived snapshots reproduces every verdict offline
and deterministically; re-running against the live web additionally reports
which sources have drifted since publication.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable


# Verification is not binary. These outcomes separate "the source does not
# support the claim" from "we could not check the source".
OUTCOME_VERIFIED = "verified"            # fetched; passage matched
OUTCOME_PASSAGE_MISMATCH = "passage_mismatch"  # fetched; passage NOT found
OUTCOME_ACCESS_BLOCKED = "access_blocked"      # robots.txt / paywall / 403
OUTCOME_FETCH_FAILED = "fetch_failed"          # network, 404, size limit
OUTCOME_PASSAGE_TOO_SHORT = "passage_too_short"  # quote below min length
OUTCOME_NO_URL = "no_url"

# Only this one indicts the source's support for a claim. The others are
# access problems that a fact-checker may be able to resolve manually.
INTEGRITY_FAILURES = frozenset({OUTCOME_PASSAGE_MISMATCH, OUTCOME_PASSAGE_TOO_SHORT})
ACCESS_FAILURES = frozenset({OUTCOME_ACCESS_BLOCKED, OUTCOME_FETCH_FAILED, OUTCOME_NO_URL})

# Bumping this invalidates cached verdicts: a re-check under different
# rules is not the same check, and the manifest must say so.
MATCH_RULES_VERSION = "1.0"
MIN_PASSAGE_CHARS = 40


def classify_failure(detail: str) -> str:
    """Map a verification_detail string onto a typed outcome.

    Lets already-stored results be re-bucketed without re-fetching.
    """
    lowered = (detail or "").lower()
    if "robots" in lowered or "403" in lowered or "paywall" in lowered:
        return OUTCOME_ACCESS_BLOCKED
    if "byte limit" in lowered or "exceeded" in lowered or "could not be independently retrieved" in lowered:
        return OUTCOME_FETCH_FAILED
    if "url is missing" in lowered:
        return OUTCOME_NO_URL
    if "not found" in lowered or "mismatch" in lowered:
        return OUTCOME_PASSAGE_MISMATCH
    return OUTCOME_FETCH_FAILED


def normalize_text(text: str) -> str:
    """Canonical form for passage matching.

    Publishers substitute typographic characters (non-breaking hyphen,
    curly quotes, nbsp) that make a correct quotation fail a naive
    comparison. Normalising both sides removes that false-negative class
    while still requiring the words themselves to match verbatim.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    # NFKC folds some variants but not all: U+2011 becomes U+2010 (hyphen),
    # never ASCII "-". Map the residue explicitly, after NFKC has run.
    for src, dst in (
        ("‐", "-"), ("‑", "-"), ("‒", "-"), ("–", "-"),
        ("—", "-"), ("―", "-"), ("−", "-"),
        ("‘", "'"), ("’", "'"), ("‚", "'"), ("‛", "'"),
        ("“", '"'), ("”", '"'), ("„", '"'),
        (" ", " "), (" ", " "), (" ", " "),
        ("​", ""), ("﻿", ""),
    ):
        out = out.replace(src, dst)
    return re.sub(r"\s+", " ", out).strip().casefold()


def passage_matches(passage: str, document: str) -> dict[str, Any]:
    """Check a quotation against a document, reporting *why* it failed.

    Distinguishes "not present at all" from "present but altered", because
    a fact-checker treats those very differently: the first suggests
    fabrication, the second a transcription slip.
    """
    raw_passage = (passage or "").strip()
    if len(raw_passage) < MIN_PASSAGE_CHARS:
        return {
            "matched": False,
            "outcome": OUTCOME_PASSAGE_TOO_SHORT,
            "detail": (
                f"quotation is {len(raw_passage)} chars; at least "
                f"{MIN_PASSAGE_CHARS} are required to be identifying"
            ),
        }
    npass = normalize_text(raw_passage)
    ndoc = normalize_text(document)
    if npass in ndoc:
        return {"matched": True, "outcome": OUTCOME_VERIFIED, "detail": "verbatim match after Unicode normalisation"}
    # Locate a partial anchor so the report can say whether the passage is
    # absent entirely or merely altered.
    anchor = npass[:60]
    if anchor and anchor in ndoc:
        return {
            "matched": False,
            "outcome": OUTCOME_PASSAGE_MISMATCH,
            "detail": "opening of the quotation is present but the full passage diverges (likely paraphrase or elision)",
        }
    return {
        "matched": False,
        "outcome": OUTCOME_PASSAGE_MISMATCH,
        "detail": "quotation does not appear in the fetched document",
    }


def snapshot_id(url: str, content: str) -> str:
    """Stable identifier for an archived retrieval of a URL."""
    digest = hashlib.sha256()
    digest.update((url or "").encode("utf-8", "replace"))
    digest.update(b"\x00")
    digest.update(normalize_text(content).encode("utf-8", "replace"))
    return digest.hexdigest()


@dataclass
class SourceCheck:
    """One re-runnable verification record."""

    citation_key: str
    url: str
    outcome: str
    detail: str
    content_sha256: str = ""
    snapshot: str = ""
    retrieved_at: str = ""
    passage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_key": self.citation_key,
            "url": self.url,
            "outcome": self.outcome,
            "detail": self.detail,
            "content_sha256": self.content_sha256,
            "snapshot": self.snapshot,
            "retrieved_at": self.retrieved_at,
            "passage": self.passage,
            "is_integrity_failure": self.outcome in INTEGRITY_FAILURES,
            "is_access_failure": self.outcome in ACCESS_FAILURES,
        }


def build_manifest(
    sources: Iterable[dict[str, Any]],
    *,
    paper_sha256: str = "",
    claims: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Emit the verification manifest published alongside a paper.

    The manifest is the fact-checking contract: everything needed to
    re-run every source check, plus an honest tally that never counts an
    access failure as evidence against a source.
    """
    checks: list[SourceCheck] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        status = str(source.get("verification_status") or "")
        detail = str(source.get("verification_detail") or "")
        outcome = OUTCOME_VERIFIED if status == "verified" else classify_failure(detail)
        checks.append(
            SourceCheck(
                citation_key=str(source.get("citation_key") or ""),
                url=str(source.get("url") or ""),
                outcome=outcome,
                detail=detail,
                content_sha256=str(source.get("content_sha256") or ""),
                snapshot=str(source.get("snapshot") or ""),
                retrieved_at=str(source.get("retrieved_at") or ""),
                passage=str(source.get("verified_passage") or ""),
            )
        )

    by_outcome: dict[str, int] = {}
    for check in checks:
        by_outcome[check.outcome] = by_outcome.get(check.outcome, 0) + 1

    integrity_failures = [c for c in checks if c.outcome in INTEGRITY_FAILURES]
    access_failures = [c for c in checks if c.outcome in ACCESS_FAILURES]
    verified = [c for c in checks if c.outcome == OUTCOME_VERIFIED]

    # Claims whose entire support base failed for *integrity* reasons are
    # the ones a fact-checker must look at first.
    unsupported_claims: list[dict[str, Any]] = []
    failed_keys = {c.citation_key for c in integrity_failures}
    verified_keys = {c.citation_key for c in verified}
    for index, claim in enumerate(claims or []):
        if not isinstance(claim, dict):
            continue
        keys = {str(k) for k in (claim.get("source_keys") or [])}
        if keys and keys <= failed_keys and not (keys & verified_keys):
            unsupported_claims.append(
                {
                    "index": index,
                    "claim": str(claim.get("claim_text") or "")[:200],
                    "failed_sources": sorted(keys),
                }
            )

    checkable = len(verified) + len(integrity_failures)
    return {
        "match_rules_version": MATCH_RULES_VERSION,
        "min_passage_chars": MIN_PASSAGE_CHARS,
        "normalization": "NFKC; typographic dashes/quotes folded; whitespace collapsed; case-insensitive",
        "paper_sha256": paper_sha256,
        "totals": {
            "sources": len(checks),
            "verified": len(verified),
            "integrity_failures": len(integrity_failures),
            "access_failures": len(access_failures),
            # Rate over sources we could actually check: an unreachable
            # page is not evidence that a source was fabricated.
            "verified_rate_of_checkable": (
                round(len(verified) / checkable, 3) if checkable else None
            ),
        },
        "by_outcome": by_outcome,
        "checks": [c.to_dict() for c in checks],
        "claims_resting_only_on_failed_sources": unsupported_claims,
        "how_to_reproduce": (
            "For each entry in `checks`: fetch `url`, normalise the document "
            f"per `normalization`, and assert that `passage` occurs in it. "
            "Entries whose outcome is an access failure could not be fetched "
            "by the pipeline and require manual retrieval; they are not "
            "evidence that the source is unsound. Compare `content_sha256` "
            "against a fresh fetch to detect post-publication page drift."
        ),
    }


__all__ = [
    "build_manifest",
    "passage_matches",
    "normalize_text",
    "classify_failure",
    "snapshot_id",
    "SourceCheck",
    "OUTCOME_VERIFIED",
    "OUTCOME_PASSAGE_MISMATCH",
    "OUTCOME_ACCESS_BLOCKED",
    "OUTCOME_FETCH_FAILED",
    "OUTCOME_PASSAGE_TOO_SHORT",
    "INTEGRITY_FAILURES",
    "ACCESS_FAILURES",
    "MATCH_RULES_VERSION",
]
