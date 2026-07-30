"""Reasoning-quality gates: causality, corroboration, modality, deniability.

These gates are orthogonal to the counting gates in research.py — a paper
can have plenty of sources and words and still reason badly.
"""
from __future__ import annotations

from django.test import TestCase

from branddozer.epistemics import (
    EpistemicPolicy,
    asserts_causation,
    deniability_level,
    evaluate_reasoning,
    independent_support,
    inference_depth,
    modal_escalation,
    modality,
    wording_is_deniable,
    writer_requirements,
)


FIRST_PARTY = {
    "citation_key": "TGT",
    "url": "https://corporate.target.com/press/a",
    "verification_status": "verified",
    "first_party": True,
}
NEWSWIRE = {
    "citation_key": "AP",
    "url": "https://apnews.com/article/b",
    "verification_status": "verified",
    "first_party": False,
}
JOURNAL = {
    "citation_key": "DOI",
    "url": "https://doi.org/10.1000/c",
    "verification_status": "verified",
    "first_party": False,
}
UNVERIFIED = {
    "citation_key": "BLOG",
    "url": "https://blog.example/d",
    "verification_status": "rejected",
    "first_party": False,
}
SOURCES = [FIRST_PARTY, NEWSWIRE, JOURNAL, UNVERIFIED]
BY_KEY = {s["citation_key"]: s for s in SOURCES}


class IndependentSupportTests(TestCase):
    def test_two_independent_domains_corroborate(self):
        got = independent_support({"source_keys": ["AP", "DOI"]}, BY_KEY)
        self.assertTrue(got["corroborated"])
        self.assertEqual(got["independent_domains"], 2)

    def test_single_source_is_not_corroboration(self):
        self.assertFalse(independent_support({"source_keys": ["AP"]}, BY_KEY)["corroborated"])

    def test_first_party_alone_is_not_independent(self):
        """A company confirming its own effects is not corroboration."""
        got = independent_support({"source_keys": ["TGT"]}, BY_KEY)
        self.assertFalse(got["corroborated"])
        self.assertEqual(got["third_party_domains"], 0)

    def test_unverified_sources_do_not_count(self):
        got = independent_support({"source_keys": ["AP", "BLOG"]}, BY_KEY)
        self.assertEqual(got["verified_count"], 1)
        self.assertFalse(got["corroborated"])


class LanguageDetectionTests(TestCase):
    def test_detects_causal_assertion(self):
        self.assertTrue(asserts_causation("The rollback caused a decline"))

    def test_plain_description_is_not_causal(self):
        self.assertFalse(asserts_causation("The program was announced in 2020"))

    def test_detects_deniable_wording(self):
        self.assertTrue(wording_is_deniable("We aim to, where feasible, increase spend"))
        self.assertTrue(wording_is_deniable("up to $2 billion"))

    def test_firm_wording_is_not_deniable(self):
        self.assertFalse(wording_is_deniable("Target terminated the program on 24 Jan"))

    def test_unlabelled_inference_depth_defaults_to_one(self):
        self.assertEqual(inference_depth({}), 1)
        self.assertEqual(inference_depth({"inference_depth": 4}), 4)


class ModalityTests(TestCase):
    def test_unknown_modality_is_blank(self):
        self.assertEqual(modality({"modality": "vibes"}), "")
        self.assertEqual(modality({"modality": "actual"}), "actual")

    def test_possible_to_actual_is_escalation(self):
        got = modal_escalation({"premise_modality": "possible", "modality": "actual"})
        self.assertIsNotNone(got)
        self.assertEqual((got["from"], got["to"]), ("possible", "actual"))

    def test_actual_to_necessary_is_escalation(self):
        self.assertIsNotNone(
            modal_escalation({"premise_modality": "actual", "modality": "necessary"})
        )

    def test_downgrade_is_legitimate(self):
        """Concluding *less* than the premise supports is always allowed."""
        self.assertIsNone(
            modal_escalation({"premise_modality": "actual", "modality": "probable"})
        )

    def test_same_modality_is_not_escalation(self):
        self.assertIsNone(
            modal_escalation({"premise_modality": "actual", "modality": "actual"})
        )


class DeniabilityTests(TestCase):
    def test_levels_are_validated(self):
        self.assertEqual(deniability_level({"deniability": "deniable"}), "deniable")
        self.assertEqual(deniability_level({"deniability": "nonsense"}), "")

    def test_deniable_wording_asserted_as_fact_is_flagged(self):
        claim = {
            "claim_text": "Target committed to spending $2 billion.",
            "claim_type": "descriptive",
            "modality": "actual",
            "deniability": "deniable",
            "quoted_wording": "we aim to spend up to $2 billion where feasible",
            "source_keys": ["TGT"],
        }
        report = evaluate_reasoning(
            claims=[claim], sources=SOURCES, policy=EpistemicPolicy(),
            rival_hypotheses=[{"hypothesis": "x"}, {"hypothesis": "y"}],
        )
        self.assertFalse(report["checks"]["no_deniable_wording_as_fact"])

    def test_hedged_prose_over_deniable_wording_is_acceptable(self):
        claim = {
            "claim_text": "Target's wording suggests an aspiration, not a binding commitment.",
            "claim_type": "descriptive",
            "modality": "actual",
            "deniability": "deniable",
            "quoted_wording": "we aim to spend up to $2 billion",
            "uncertainty": "non-binding language",
            "source_keys": ["TGT"],
        }
        report = evaluate_reasoning(
            claims=[claim], sources=SOURCES, policy=EpistemicPolicy(),
            rival_hypotheses=[{"hypothesis": "x"}, {"hypothesis": "y"}],
        )
        self.assertTrue(report["checks"]["no_deniable_wording_as_fact"])


class CausalWarrantTests(TestCase):
    def _causal(self, **overrides):
        claim = {
            "claim_text": "The rollback was followed by a sales decline.",
            "claim_type": "causal",
            "modality": "probable",
            "premise_modality": "probable",
            "inference_depth": 2,
            "causal_design": "interrupted_time_series",
            "identification_strategy": "pre/post against control chains",
            "uncertainty": "macro confounding",
            "source_keys": ["AP", "DOI"],
        }
        claim.update(overrides)
        return claim

    def _run(self, claims):
        return evaluate_reasoning(
            claims=claims,
            sources=SOURCES,
            policy=EpistemicPolicy(),
            rival_hypotheses=[{"hypothesis": "macro"}, {"hypothesis": "seasonal"}],
        )

    def test_well_warranted_causal_claim_passes(self):
        report = self._run([self._causal()])
        self.assertTrue(report["checks"]["causal_claims_have_design"])
        self.assertTrue(report["checks"]["causal_claims_corroborated"])
        self.assertTrue(report["checks"]["no_causal_overclaiming"])

    def test_causal_claim_without_design_fails(self):
        report = self._run([self._causal(causal_design="", identification_strategy="")])
        self.assertFalse(report["checks"]["causal_claims_have_design"])

    def test_causal_claim_on_one_source_is_uncorroborated(self):
        report = self._run([self._causal(source_keys=["AP"])])
        self.assertFalse(report["checks"]["causal_claims_corroborated"])

    def test_causal_prose_typed_descriptive_is_overclaiming(self):
        claim = {
            "claim_text": "The DEI rollback caused the sales decline.",
            "claim_type": "descriptive",
            "modality": "actual",
            "source_keys": ["AP", "DOI"],
        }
        self.assertFalse(self._run([claim])["checks"]["no_causal_overclaiming"])

    def test_hedged_causal_prose_is_not_overclaiming(self):
        claim = {
            "claim_text": "The rollback may have caused the decline.",
            "claim_type": "correlational",
            "modality": "possible",
            "source_keys": ["AP", "DOI"],
        }
        self.assertTrue(self._run([claim])["checks"]["no_causal_overclaiming"])


class ReachAndDepthTests(TestCase):
    def _run(self, claims, **policy_kwargs):
        return evaluate_reasoning(
            claims=claims,
            sources=SOURCES,
            policy=EpistemicPolicy(**policy_kwargs),
            rival_hypotheses=[{"hypothesis": "a"}, {"hypothesis": "b"}],
        )

    def test_all_deep_speculation_fails_depth_gate(self):
        claims = [
            {
                "claim_text": f"Speculative inference {i}",
                "claim_type": "counterfactual",
                "modality": "counterfactual",
                "inference_depth": 4,
                "source_keys": ["AP", "DOI"],
                "causal_design": "process_tracing",
                "identification_strategy": "chain",
                "uncertainty": "high",
            }
            for i in range(4)
        ]
        self.assertFalse(self._run(claims)["checks"]["inference_depth_bounded"])

    def test_purely_descriptive_paper_lacks_analytic_reach(self):
        claims = [
            {
                "claim_text": f"Document {i} states a fact.",
                "claim_type": "descriptive",
                "modality": "actual",
                "inference_depth": 0,
                "uncertainty": "none",
                "source_keys": ["AP"],
            }
            for i in range(5)
        ]
        self.assertFalse(self._run(claims)["checks"]["analytic_reach"])

    def test_missing_rival_hypotheses_fails(self):
        report = evaluate_reasoning(
            claims=[{"claim_text": "x", "modality": "actual", "uncertainty": "y"}],
            sources=SOURCES,
            policy=EpistemicPolicy(),
            rival_hypotheses=[],
        )
        self.assertFalse(report["checks"]["rival_hypotheses_considered"])

    def test_missing_modality_is_flagged(self):
        report = self._run([{"claim_text": "x", "source_keys": ["AP"]}])
        self.assertFalse(report["checks"]["modality_declared"])

    def test_empty_paper_does_not_crash(self):
        report = evaluate_reasoning(
            claims=[], sources=[], policy=EpistemicPolicy(), rival_hypotheses=[]
        )
        self.assertIn("checks", report)


class WriterContractTests(TestCase):
    def test_contract_states_every_scored_dimension(self):
        text = writer_requirements(EpistemicPolicy())
        for token in (
            "causal_design", "identification_strategy", "modality",
            "premise_modality", "inference_depth", "deniability",
            "quoted_wording", "Rival Explanations",
        ):
            self.assertIn(token, text)

    def test_contract_tracks_policy_values(self):
        text = writer_requirements(EpistemicPolicy(min_rival_hypotheses=5))
        self.assertIn("5", text)
