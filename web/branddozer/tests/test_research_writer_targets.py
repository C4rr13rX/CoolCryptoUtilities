"""The writer must be told the thresholds its output is graded against.

Regression guard: min_words/min_verified_sources were only ever used to
*fail* the acceptance gate, never stated in the writer prompt, so drafts
came back far under the floor and burned every revision round.
"""
from __future__ import annotations

from unittest.mock import patch

from django.test import TestCase

from branddozer.research import ResearchPolicy, ResearchWorkflow


class _StubRun:
    """Minimal stand-in for DeliveryRun used by _write_candidate."""

    def __init__(self):
        self.id = "00000000-0000-0000-0000-000000000000"
        self.prompt = "goal"
        self.context = {}


class WriterInstructionTests(TestCase):
    def _workflow(self, **policy_kwargs) -> ResearchWorkflow:
        workflow = ResearchWorkflow.__new__(ResearchWorkflow)
        workflow.run = _StubRun()
        workflow.policy = ResearchPolicy(**policy_kwargs)
        return workflow

    def _capture_prompt(self, workflow, **kwargs) -> str:
        seen = {}

        def _fake_call(_self, role, name, prompt, *, system):
            seen["prompt"] = prompt
            return {}

        with patch.object(ResearchWorkflow, "_call", _fake_call):
            workflow._write_candidate({}, [], [], {}, **kwargs)
        return seen["prompt"]

    def test_first_draft_states_word_floor_and_source_minimum(self):
        workflow = self._workflow(min_words=5000, min_verified_sources=10)
        prompt = self._capture_prompt(workflow)
        self.assertIn("5000", prompt)
        self.assertIn("10", prompt)
        self.assertIn("at least 5000 words", prompt)

    def test_thresholds_track_policy(self):
        workflow = self._workflow(min_words=1234, min_verified_sources=7)
        prompt = self._capture_prompt(workflow)
        self.assertIn("at least 1234 words", prompt)

    def test_revision_names_the_word_shortfall(self):
        workflow = self._workflow(min_words=5000)
        feedback = {
            "checks": {"minimum_word_count": False, "required_sections": True},
            "metrics": {"word_count": 690},
        }
        prompt = self._capture_prompt(
            workflow, previous="# old paper", revision_feedback=feedback
        )
        self.assertIn("690 words", prompt)
        # 5000 - 690 = 4310 words of growth required.
        self.assertIn("4310", prompt)

    def test_revision_lists_failed_gates(self):
        workflow = self._workflow(min_words=5000)
        feedback = {
            "checks": {
                "minimum_word_count": False,
                "verified_sources": False,
                "required_sections": True,
            },
            "metrics": {"word_count": 100},
        }
        prompt = self._capture_prompt(
            workflow, previous="# old", revision_feedback=feedback
        )
        # Isolate the failed-gates sentence; the raw report is JSON-dumped
        # later in the prompt and legitimately names passing gates too.
        sentence = prompt.split("Failed gates to fix:")[1].split(".")[0]
        self.assertIn("minimum_word_count", sentence)
        self.assertIn("verified_sources", sentence)
        self.assertNotIn("required_sections", sentence)

    def test_first_draft_has_no_revision_shortfall_text(self):
        workflow = self._workflow(min_words=5000)
        prompt = self._capture_prompt(workflow)
        self.assertNotIn("Failed gates to fix", prompt)


class AuthorityTierTests(TestCase):
    """First-party records must count as authoritative for the subject.

    Regression guard: corporate.target.com was classified first-party and
    provenance-verified, yet scored tier 1 (same as an anonymous blog), so a
    study of a company's own programs could never satisfy the
    authoritative-sources gate.
    """

    def _tier(self, source):
        from branddozer.research import _authority_tier, _classify_source_provenance

        return _authority_tier(_classify_source_provenance(source))

    def test_corporate_primary_domain_is_authoritative(self):
        tier = self._tier({"url": "https://corporate.target.com/press/releases/x"})
        self.assertGreaterEqual(tier, 2)

    def test_government_domain_stays_top_tier(self):
        self.assertEqual(self._tier({"url": "https://cbc.house.gov/news/x"}), 3)

    def test_doi_source_is_authoritative(self):
        self.assertGreaterEqual(self._tier({"url": "https://doi.org/10.1234/x"}), 2)

    def test_unaffiliated_blog_stays_lowest_tier(self):
        self.assertEqual(self._tier({"url": "https://some-random-blog.example/post"}), 1)

    def test_encyclopedia_is_not_authoritative(self):
        self.assertEqual(self._tier({"url": "https://en.wikipedia.org/wiki/Target"}), 1)
