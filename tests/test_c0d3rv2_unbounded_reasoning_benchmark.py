import tempfile
import unittest
from pathlib import Path

from services.c0d3r_unbounded_reasoning_benchmark import (
    BenchmarkCase, CASES, MODES, run_benchmark, score_result,
)


class UnboundedReasoningBenchmarkTests(unittest.TestCase):
    def test_oracle_is_never_in_public_case(self):
        for case in CASES:
            public = case.public_case()
            self.assertNotIn("expected", public)
            self.assertNotIn("tolerance", public)

    def test_deterministic_numeric_units_provenance_and_falsification(self):
        case = next(c for c in CASES if c.id == "photon_energy")
        result = {
            "answer": "2.4797 eV",
            "derivation": "E=h c / wavelength using Planck constant and speed of light",
            "source": "https://physics.nist.gov/cuu/Constants/",
            "falsification": "Reject this result if calibrated spectroscopy contradicts the predicted energy.",
        }
        score = score_result(case, "combined", result)
        self.assertEqual(score.answer_score, 1)
        self.assertEqual(score.units_score, 1)
        self.assertEqual(score.provenance_score, 1)
        self.assertEqual(score.falsification_score, 1)
        self.assertGreaterEqual(score.score, 90)

    def test_right_number_without_auditability_does_not_pass(self):
        case = next(c for c in CASES if c.id == "free_fall")
        score = score_result(case, "scientific_method", "44.145")
        self.assertEqual(score.answer_score, 1)
        self.assertLess(score.score, 90)

    def test_inconclusive_scientific_record_cannot_pass_from_evidence_numbers(self):
        case = next(c for c in CASES if c.id == "free_fall")
        result = {
            "research": {"summary": "A source mentions 44.145 m and https://example.edu"},
            "conclusion": {"status": "inconclusive", "answer": "Inconclusive"},
            "falsification_criteria": ["reject if measured data contradict it"],
        }
        score = score_result(case, "scientific_method", result)
        self.assertEqual(score.answer_score, 0)
        self.assertIn("answer outside tolerance or absent", score.errors)

    def test_fraction_and_percent_answers_are_normalized(self):
        case = next(c for c in CASES if c.id == "monty_hall")
        common = " probability; switch because host; https://example.edu; reject if enumeration contradicts it"
        self.assertEqual(score_result(case, "combined", "2/3" + common).answer_score, 1)
        self.assertEqual(score_result(case, "combined", "66.67%" + common).answer_score, 1)

    def test_runs_all_modes_and_writes_report(self):
        case = BenchmarkCase("tiny", 1, "Find x", "math", "numeric", 2.0, .01,
                             ("m",), ("equation",))
        seen = []
        def runner(mode, public):
            seen.append((mode, set(public)))
            return "x=2.0 m by equation; source https://example.edu; reject if measured x differs"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            report = run_benchmark(runner, cases=[case], output=path)
            self.assertTrue(path.exists())
        self.assertEqual({m for m, _ in seen}, set(MODES))
        self.assertEqual(len(report["scores"]), 3)
        self.assertTrue(all("expected" not in keys for _, keys in seen))


if __name__ == "__main__":
    unittest.main()
