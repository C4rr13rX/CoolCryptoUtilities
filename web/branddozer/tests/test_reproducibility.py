"""Verification must be independently repeatable and honestly reported."""
from __future__ import annotations

from django.test import TestCase

from branddozer.reproducibility import (
    ACCESS_FAILURES,
    INTEGRITY_FAILURES,
    OUTCOME_ACCESS_BLOCKED,
    OUTCOME_FETCH_FAILED,
    OUTCOME_PASSAGE_MISMATCH,
    OUTCOME_PASSAGE_TOO_SHORT,
    OUTCOME_VERIFIED,
    build_manifest,
    classify_failure,
    normalize_text,
    passage_matches,
    snapshot_id,
)


DOC = (
    "Target Commits to Spending More Than $2 Billion with Black-Owned "
    "Businesses by 2025. The company said it would add products from more "
    "than 500 Black-owned brands."
)


class NormalizationTests(TestCase):
    def test_folds_typographic_variants(self):
        self.assertEqual(
            normalize_text("Black‑Owned  “Businesses”"),
            normalize_text("black-owned \"businesses\""),
        )

    def test_collapses_whitespace_and_case(self):
        self.assertEqual(normalize_text("  A   B \n C "), "a b c")

    def test_handles_empty(self):
        self.assertEqual(normalize_text(""), "")


class PassageMatchTests(TestCase):
    def test_exact_quotation_matches(self):
        got = passage_matches("add products from more than 500 Black-owned brands", DOC)
        self.assertTrue(got["matched"])
        self.assertEqual(got["outcome"], OUTCOME_VERIFIED)

    def test_typographic_variant_still_matches(self):
        """A correct quote must not fail because of a non-breaking hyphen."""
        got = passage_matches(
            "add products from more than 500 Black‑owned brands", DOC
        )
        self.assertTrue(got["matched"])

    def test_absent_quotation_is_a_mismatch(self):
        got = passage_matches(
            "Target promised to double its charitable giving next year", DOC
        )
        self.assertFalse(got["matched"])
        self.assertEqual(got["outcome"], OUTCOME_PASSAGE_MISMATCH)

    def test_short_quote_is_reported_distinctly(self):
        """Too-short quotes are unidentifying, not fabricated."""
        got = passage_matches("Target", DOC)
        self.assertEqual(got["outcome"], OUTCOME_PASSAGE_TOO_SHORT)

    def test_altered_quotation_is_distinguished_from_absent(self):
        """A quote that starts right but drifts reads as paraphrase, not fabrication."""
        # Must share the first 60 normalised chars with the document so the
        # anchor lands, then diverge.
        got = passage_matches(
            "Target Commits to Spending More Than $2 Billion with Black-Owned "
            "Businesses by 2099 according to purple elephants",
            DOC,
        )
        self.assertFalse(got["matched"])
        self.assertIn("diverges", got["detail"])


class FailureClassificationTests(TestCase):
    def test_robots_block_is_an_access_failure(self):
        self.assertEqual(
            classify_failure("robots.txt disallows crawl"), OUTCOME_ACCESS_BLOCKED
        )

    def test_size_limit_is_a_fetch_failure(self):
        self.assertEqual(
            classify_failure("page exceeded byte limit"), OUTCOME_FETCH_FAILED
        )

    def test_passage_problem_is_an_integrity_failure(self):
        self.assertEqual(
            classify_failure("purported supporting passage was not found"),
            OUTCOME_PASSAGE_MISMATCH,
        )

    def test_access_and_integrity_sets_are_disjoint(self):
        self.assertFalse(INTEGRITY_FAILURES & ACCESS_FAILURES)


class SnapshotTests(TestCase):
    def test_same_content_gives_same_id(self):
        self.assertEqual(snapshot_id("https://x/y", DOC), snapshot_id("https://x/y", DOC))

    def test_changed_content_changes_id(self):
        self.assertNotEqual(
            snapshot_id("https://x/y", DOC), snapshot_id("https://x/y", DOC + " edited")
        )

    def test_whitespace_only_change_does_not_change_id(self):
        """Re-fetch reflow must not look like content drift."""
        self.assertEqual(
            snapshot_id("https://x/y", DOC),
            snapshot_id("https://x/y", DOC.replace(" ", "  ")),
        )


class ManifestTests(TestCase):
    def _sources(self):
        return [
            {
                "citation_key": "OK",
                "url": "https://a.example/1",
                "verification_status": "verified",
                "verification_detail": "retrieved and content-hashed",
                "content_sha256": "a" * 64,
                "verified_passage": "a sufficiently long verbatim passage here",
            },
            {
                "citation_key": "BLOCKED",
                "url": "https://b.example/2",
                "verification_status": "rejected",
                "verification_detail": "robots.txt disallows crawl",
            },
            {
                "citation_key": "BAD",
                "url": "https://c.example/3",
                "verification_status": "rejected",
                "verification_detail": "purported supporting passage was not found",
            },
        ]

    def test_separates_access_from_integrity_failures(self):
        m = build_manifest(self._sources())
        self.assertEqual(m["totals"]["verified"], 1)
        self.assertEqual(m["totals"]["integrity_failures"], 1)
        self.assertEqual(m["totals"]["access_failures"], 1)

    def test_verified_rate_excludes_unreachable_sources(self):
        """An unfetchable page is not evidence the source was fabricated."""
        m = build_manifest(self._sources())
        # 1 verified of 2 checkable (verified + integrity), not of 3.
        self.assertEqual(m["totals"]["verified_rate_of_checkable"], 0.5)

    def test_flags_claims_resting_only_on_failed_sources(self):
        claims = [
            {"claim_text": "Rests on a bad source", "source_keys": ["BAD"]},
            {"claim_text": "Rests on a good source", "source_keys": ["OK"]},
        ]
        m = build_manifest(self._sources(), claims=claims)
        flagged = m["claims_resting_only_on_failed_sources"]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["failed_sources"], ["BAD"])

    def test_claim_with_one_good_source_is_not_flagged(self):
        claims = [{"claim_text": "Mixed support", "source_keys": ["OK", "BAD"]}]
        m = build_manifest(self._sources(), claims=claims)
        self.assertEqual(m["claims_resting_only_on_failed_sources"], [])

    def test_access_failure_alone_does_not_flag_a_claim(self):
        """A blocked fetch must not be reported as an unsupported claim."""
        claims = [{"claim_text": "Cites a blocked page", "source_keys": ["BLOCKED"]}]
        m = build_manifest(self._sources(), claims=claims)
        self.assertEqual(m["claims_resting_only_on_failed_sources"], [])

    def test_manifest_states_its_matching_rules(self):
        m = build_manifest(self._sources())
        self.assertIn("match_rules_version", m)
        self.assertIn("normalization", m)
        self.assertIn("how_to_reproduce", m)

    def test_every_check_is_replayable(self):
        m = build_manifest(self._sources())
        for check in m["checks"]:
            for field in ("citation_key", "url", "outcome", "passage"):
                self.assertIn(field, check)

    def test_empty_input_does_not_crash(self):
        m = build_manifest([])
        self.assertEqual(m["totals"]["sources"], 0)
        self.assertIsNone(m["totals"]["verified_rate_of_checkable"])
