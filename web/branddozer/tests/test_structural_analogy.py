"""Rules-as-attractor comparison must actually search history.

Regression guard for a live failure: a run asked for exactly this method,
produced a 12,428-word paper whose every finding was a statement that the
work was "unexecutable", cited no case earlier than 2020, and never used
the word "attractor". A blocked focal case must not stop comparative work.
"""
from __future__ import annotations

from django.test import TestCase

from branddozer.structural_analogy import (
    MIN_COMPARISON_CASES,
    MIN_ERAS,
    RULE_DIMENSIONS,
    RuleSystem,
    compare_prompt,
    extract_rules_prompt,
    find_analogues_prompt,
    inconclusive_report,
    validate_case_set,
)


def _case(name, era, strength="strong", domain="x", disanalogies="differs on scale"):
    return {
        "name": name,
        "era": era,
        "domain": domain,
        "strength": strength,
        "disanalogies": disanalogies,
    }


GOOD_SET = [
    _case("Guild ordinance", "14th century", domain="crafts"),
    _case("New Deal lending rule", "1930s", domain="finance"),
    _case("Affirmative action policy", "1970s", domain="education"),
    _case("Corporate quota", "1990s", strength="failed", domain="corporate"),
]


class RuleSystemTests(TestCase):
    def test_coverage_reports_missing_dimensions(self):
        system = RuleSystem(
            name="P", era="2020s", context="c",
            dimensions={"eligibility": "x", "allocation": "y"},
        )
        self.assertLess(system.coverage(), 1.0)

    def test_full_coverage(self):
        system = RuleSystem(
            name="P", era="2020s", context="c",
            dimensions={d: "stated" for d in RULE_DIMENSIONS},
        )
        self.assertEqual(system.coverage(), 1.0)


class ExtractionPromptTests(TestCase):
    def test_prompt_names_every_dimension(self):
        text = extract_rules_prompt("A programme", "evidence")
        for dimension in RULE_DIMENSIONS:
            self.assertIn(dimension, text)

    def test_partial_evidence_must_not_abort_extraction(self):
        """Rules survive weak evidence; that is the point of the method."""
        text = extract_rules_prompt("A programme", "evidence")
        self.assertIn("do not abandon the whole extraction", text)


class AnaloguePromptTests(TestCase):
    def test_prompt_demands_cross_era_search(self):
        text = find_analogues_prompt({"name": "P"})
        self.assertIn("ANY era", text)
        self.assertIn("not restrict to the same population", text.lower())

    def test_prompt_demands_negative_cases(self):
        text = find_analogues_prompt({"name": "P"})
        self.assertIn("NEGATIVE case", text)

    def test_prompt_treats_leads_as_unverified(self):
        text = find_analogues_prompt({"name": "P"})
        self.assertIn("verified, not findings", text)


class CaseSetValidationTests(TestCase):
    def test_broad_honest_set_passes(self):
        report = validate_case_set(GOOD_SET)
        self.assertTrue(report["passed"], report)

    def test_too_few_cases_fails(self):
        report = validate_case_set(GOOD_SET[:2])
        self.assertFalse(report["checks"]["enough_cases"])

    def test_modern_only_set_is_not_a_historical_search(self):
        """The failing run cited nothing before 2020."""
        modern = [
            _case("A", "2021"), _case("B", "2022"),
            _case("C", "2023"), _case("D", "2024"),
        ]
        report = validate_case_set(modern)
        self.assertFalse(report["checks"]["spans_multiple_eras"])

    def test_set_without_negative_cases_fails(self):
        allpos = [_case(f"C{i}", f"19{i}0s") for i in range(4)]
        report = validate_case_set(allpos)
        self.assertFalse(report["checks"]["has_negative_cases"])

    def test_case_without_disanalogies_fails(self):
        cases = list(GOOD_SET)
        cases[0] = _case("Guild", "14th century", disanalogies="")
        report = validate_case_set(cases)
        self.assertFalse(report["checks"]["every_case_states_disanalogies"])
        self.assertIn("Guild", report["cases_missing_disanalogies"])

    def test_empty_set_fails_rather_than_passing_vacuously(self):
        report = validate_case_set([])
        self.assertFalse(report["passed"])

    def test_thresholds_are_stated(self):
        self.assertGreaterEqual(MIN_COMPARISON_CASES, 3)
        self.assertGreaterEqual(MIN_ERAS, 2)


class ComparisonPromptTests(TestCase):
    def test_prompt_forbids_concluding_from_analogy(self):
        text = compare_prompt({"name": "P"}, GOOD_SET)
        self.assertIn("never a conclusion", text)
        self.assertIn("falsification test", text.lower())

    def test_prompt_requires_reporting_failed_analogies(self):
        text = compare_prompt({"name": "P"}, GOOD_SET)
        self.assertIn("cherry-picked", text)

    def test_prompt_forbids_averaging_disagreement(self):
        text = compare_prompt({"name": "P"}, GOOD_SET)
        self.assertIn("Do not average", text)


class InconclusiveTests(TestCase):
    def test_inconclusive_is_explicit_and_unpublishable(self):
        """A failed run must say so, not ship a paper about impossibility."""
        report = inconclusive_report("Subject", "no sources", ["archival", "structural"])
        self.assertEqual(report["status"], "inconclusive")
        self.assertFalse(report["publishable"])
        self.assertIn("Inconclusive", report["headline"])
        self.assertEqual(len(report["methods_attempted"]), 2)
