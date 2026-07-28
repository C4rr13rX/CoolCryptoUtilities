from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.c0d3rV2.outline_refiner import OutlineRefiner, is_creation_request


class OutlineRefinerTests(unittest.TestCase):
    def test_creation_detection_does_not_capture_conversation(self):
        self.assertTrue(is_creation_request("Build a Django inventory app"))
        self.assertFalse(is_creation_request("How are you today?"))

    def test_four_passes_clear_quality_threshold_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = OutlineRefiner(workdir=Path(tmp), passes=4).refine(
                "Build a Django inventory app with CSV export and do not add payment processing."
            )
            self.assertTrue(result["quality"]["passed"])
            self.assertGreaterEqual(result["quality"]["score"], 92)
            self.assertEqual(result["quality"]["refinement_passes"], 4)
            self.assertTrue((Path(tmp) / ".c0d3r" / "refined-outline.json").exists())

    def test_model_scope_expansion_is_rejected(self):
        def expanding_model(**_kwargs):
            return '{"deliverables":["Add a blockchain marketplace and mobile app"]}'

        result = OutlineRefiner(send=expanding_model, passes=4).refine(
            "Create a printable bicycle maintenance checklist."
        )
        text = str(result).lower()
        self.assertNotIn("blockchain marketplace", text)
        self.assertNotIn("mobile app", text)

    def test_commercial_plan_requires_market_evidence(self):
        result = OutlineRefiner(passes=4).refine("Create a digital product to sell to customers.")
        self.assertFalse(result["quality"]["passed"])
        self.assertTrue(result["quality"]["market_evidence_required"])


if __name__ == "__main__":
    unittest.main()
