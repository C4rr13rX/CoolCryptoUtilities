"""Script compression: fit the slot without flattening the findings."""
from __future__ import annotations

from django.test import TestCase

from branddozer.presentation_script import (
    WORDS_PER_MINUTE,
    ScriptRequest,
    script_prompt,
    script_to_markdown,
    validate_script,
    write_script,
)


REQUEST = ScriptRequest(
    title="A Paper",
    abstract="An abstract.",
    markdown="# A Paper\n\nBody text.",
    target_minutes=15,
    findings_tone="negative / inconclusive",
)


def _script(sections=3, words_each=100):
    return {
        "title": "Spoken Title",
        "hook": "What does this mean?",
        "sections": [
            {"heading": f"Section {i}", "narration": " ".join(["word"] * words_each)}
            for i in range(sections)
        ],
        "closing": "Much remains unresolved.",
    }


class BudgetTests(TestCase):
    def test_budget_follows_measured_narration_speed(self):
        self.assertEqual(REQUEST.word_budget(), 15 * WORDS_PER_MINUTE)

    def test_target_minutes_are_clamped(self):
        self.assertGreater(ScriptRequest("T", "A", "M", target_minutes=1).word_budget(), 0)
        self.assertLessEqual(
            ScriptRequest("T", "A", "M", target_minutes=999).word_budget(),
            30 * WORDS_PER_MINUTE,
        )

    def test_overlong_script_drops_whole_sections(self):
        """Truncating prose could cut a hedge in half; drop sections instead."""
        result = validate_script(_script(sections=40, words_each=200), REQUEST)
        self.assertLessEqual(result["word_count"], result["word_budget"])
        self.assertGreater(result["sections_dropped"], 0)
        # Surviving sections are intact, never truncated mid-sentence.
        for section in result["sections"]:
            self.assertEqual(len(section["narration"].split()), 200)

    def test_at_least_one_section_always_survives(self):
        result = validate_script(_script(sections=5, words_each=99999), REQUEST)
        self.assertGreaterEqual(len(result["sections"]), 1)

    def test_short_script_is_flagged_as_under_budget(self):
        """An under-run means the video is far shorter than its slot."""
        result = validate_script(_script(sections=1, words_each=20), REQUEST)
        self.assertTrue(result["under_budget"])
        self.assertLess(result["budget_use"], 0.85)

    def test_well_sized_script_is_not_flagged(self):
        # Sized to land just inside the 1,950-word budget: 10 x 190 = 1,900
        # plus headings, so nothing is dropped and use is above 85%.
        result = validate_script(_script(sections=10, words_each=190), REQUEST)
        self.assertTrue(result["within_budget"])
        self.assertEqual(result["sections_dropped"], 0)
        self.assertFalse(result["under_budget"])

    def test_estimated_minutes_tracks_word_count(self):
        result = validate_script(_script(sections=2, words_each=130), REQUEST)
        self.assertAlmostEqual(
            result["estimated_minutes"], result["word_count"] / WORDS_PER_MINUTE, places=1
        )


class PromptTests(TestCase):
    def test_prompt_states_a_floor_not_only_a_ceiling(self):
        """A ceiling alone produced 267 words against a 1,950 budget."""
        text = script_prompt(REQUEST)
        self.assertIn("1950", text.replace(",", ""))
        self.assertIn("far under is as much a failure", text)

    def test_prompt_protects_hedged_findings(self):
        text = script_prompt(REQUEST)
        self.assertIn("CUT THE CLAIM", text)
        self.assertIn("negative / inconclusive", text)

    def test_prompt_forbids_written_only_artefacts(self):
        text = script_prompt(REQUEST)
        for banned in ("citation markers", "URLs", "tables"):
            self.assertIn(banned, text)


class MarkdownTests(TestCase):
    def test_script_renders_as_chunkable_markdown(self):
        md = script_to_markdown(validate_script(_script(), REQUEST))
        self.assertTrue(md.startswith("# "))
        self.assertIn("## Section 0", md)
        self.assertIn("## In closing", md)

    def test_empty_sections_do_not_produce_stray_headings(self):
        script = validate_script(
            {"title": "T", "hook": "", "sections": [], "closing": ""}, REQUEST
        )
        self.assertEqual(script_to_markdown(script).strip(), "# T")


class WriteScriptTests(TestCase):
    def test_non_json_response_raises(self):
        with self.assertRaises(ValueError):
            write_script(REQUEST, agent_send=lambda p, system="": "not json")

    def test_agent_output_is_clamped(self):
        raw = (
            '{"title":"T","hook":"H","closing":"C","sections":['
            + ",".join(
                f'{{"heading":"S{i}","narration":"{" ".join(["w"] * 500)}"}}'
                for i in range(20)
            )
            + "]}"
        )
        result = write_script(REQUEST, agent_send=lambda p, system="": raw)
        self.assertLessEqual(result["word_count"], result["word_budget"])
