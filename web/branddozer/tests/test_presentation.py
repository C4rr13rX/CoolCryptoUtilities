"""The adjacent presentation schema: mobile-legible slides, exact timings."""
from __future__ import annotations

from django.test import TestCase

from branddozer.presentation import (
    MAX_SLIDE_CHARS,
    SCHEMA_VERSION,
    attach_word_timings,
    build_presentation,
    build_slides,
)
from branddozer.presentation_media import (
    COLOR_RATIOS,
    COLOR_SCHEMES,
    TRANSITIONS,
    WORD_ANIMATIONS,
    background_prompt,
    palette_prompt,
    score_sync_points,
)


PAPER = """# A Long Research Title About Something

## Abstract

This paper reports a negative result. The archival record does not support
the stronger claim, and we say so plainly rather than overstating it.

## Findings

- The first documented program ended in January.
- No successor arrangement appears in the record.

> The company said it would add products from Black-owned brands.

https://example.com/a/very/long/reference/url/that/cannot/be/split/sensibly
"""


class ChunkingTests(TestCase):
    def setUp(self):
        self.slides = build_slides(PAPER, title="A Long Research Title About Something")

    def test_every_prose_slide_fits_the_mobile_budget(self):
        """Nothing but an unsplittable URL may exceed the character budget."""
        for slide in self.slides:
            if slide.notes == "url":
                continue
            self.assertLessEqual(
                len(slide.text), MAX_SLIDE_CHARS, f"over budget: {slide.text!r}"
            )

    def test_title_is_not_duplicated_by_the_h1(self):
        titles = [s for s in self.slides if s.kind == "title"]
        joined = " ".join(s.text for s in titles).lower()
        self.assertEqual(joined.count("a long research title"), 1)

    def test_structure_is_preserved(self):
        kinds = {s.kind for s in self.slides}
        self.assertIn("heading", kinds)
        self.assertIn("body", kinds)
        self.assertIn("list_item", kinds)
        self.assertIn("quote", kinds)

    def test_headings_carry_their_section(self):
        body = [s for s in self.slides if s.kind == "body" and s.section]
        self.assertTrue(body, "body slides should inherit a section")

    def test_bare_url_is_a_citation_and_stays_whole(self):
        urls = [s for s in self.slides if s.notes == "url"]
        self.assertEqual(len(urls), 1)
        self.assertEqual(urls[0].kind, "citation")
        self.assertIn("example.com", urls[0].text)

    def test_markdown_emphasis_is_stripped(self):
        slides = build_slides("Some **bold** and _italic_ and `code` text.")
        self.assertNotIn("**", slides[0].text)
        self.assertNotIn("`", slides[0].text)

    def test_abbreviations_do_not_split_sentences(self):
        slides = build_slides("Work by Smith et al. supports this reading.")
        self.assertEqual(len(slides), 1)

    def test_empty_paper_yields_no_slides(self):
        self.assertEqual(build_slides(""), [])

    def test_code_blocks_are_skipped(self):
        slides = build_slides("Before.\n\n```\nnot narrated\n```\n\nAfter.")
        joined = " ".join(s.text for s in slides)
        self.assertNotIn("not narrated", joined)


class PresentationDocumentTests(TestCase):
    def test_document_shape(self):
        deck = build_presentation(
            paper_id="p1", title="T", markdown=PAPER, abstract="A"
        )
        self.assertEqual(deck["schema_version"], SCHEMA_VERSION)
        for key in ("slides", "slide_count", "estimated_duration_ms", "narrated"):
            self.assertIn(key, deck)
        self.assertEqual(deck["slide_count"], len(deck["slides"]))

    def test_unnarrated_deck_still_has_durations(self):
        """A deck must be playable before narration exists."""
        deck = build_presentation(paper_id="p1", title="T", markdown=PAPER)
        self.assertFalse(deck["narrated"])
        self.assertTrue(all(s["duration_ms"] > 0 for s in deck["slides"]))


class WordTimingTests(TestCase):
    MARKS = [
        {"type": "word", "time": 125, "value": "Archival"},
        {"type": "word", "time": 712, "value": "evidence"},
        {"type": "word", "time": 1137, "value": "constrains"},
    ]

    def test_each_word_gets_a_span(self):
        slide = {"index": 0, "text": "Archival evidence constrains"}
        attach_word_timings(slide, self.MARKS, audio_ms=1800)
        self.assertEqual(len(slide["words"]), 3)
        self.assertEqual(slide["words"][0]["start_ms"], 125)
        self.assertEqual(slide["words"][0]["end_ms"], 712)

    def test_last_word_runs_to_the_clip_end(self):
        """Otherwise the final word's highlight ends early."""
        slide = {"index": 0, "text": "x"}
        attach_word_timings(slide, self.MARKS, audio_ms=1800)
        self.assertEqual(slide["words"][-1]["end_ms"], 1800)

    def test_spans_are_ordered_and_non_negative(self):
        slide = {"index": 0, "text": "x"}
        attach_word_timings(slide, self.MARKS, audio_ms=1800)
        words = slide["words"]
        for word in words:
            self.assertGreaterEqual(word["end_ms"], word["start_ms"])
        for i in range(len(words) - 1):
            self.assertLessEqual(words[i]["start_ms"], words[i + 1]["start_ms"])

    def test_narration_overrides_the_estimate(self):
        slide = {"index": 0, "text": "x", "duration_ms": 9999}
        attach_word_timings(slide, self.MARKS, audio_ms=1800)
        self.assertEqual(slide["duration_ms"], 1800)

    def test_non_word_marks_are_ignored(self):
        slide = {"index": 0, "text": "x"}
        marks = [{"type": "sentence", "time": 0, "value": "s"}, *self.MARKS]
        attach_word_timings(slide, marks, audio_ms=1800)
        self.assertEqual(len(slide["words"]), 3)


class MediaPromptTests(TestCase):
    def test_palette_prompt_names_every_scheme(self):
        text = palette_prompt("Title", "Abstract", ratio="60_30_10")
        for scheme in COLOR_SCHEMES:
            self.assertIn(scheme, text)

    def test_palette_prompt_forbids_random_choice(self):
        text = palette_prompt("Title", "Abstract", ratio="60_30_10")
        self.assertIn("not pick colours at random", text)
        self.assertIn("60", text)

    def test_background_prompt_excludes_text_in_the_image(self):
        text = background_prompt("Some slide", "Findings", {"scheme": "triadic"})
        self.assertIn("No text", text)
        self.assertIn("Findings", text)

    def test_known_ratios_and_effects_are_declared(self):
        self.assertIn("60_30_10", COLOR_RATIOS)
        self.assertIn("crossfade", TRANSITIONS)
        self.assertIn("highlight", WORD_ANIMATIONS)


class ScoreSyncTests(TestCase):
    def test_sync_points_accumulate_along_the_timeline(self):
        slides = [
            {"index": 0, "kind": "title", "duration_ms": 2000},
            {"index": 1, "kind": "body", "duration_ms": 3000},
            {"index": 2, "kind": "heading", "duration_ms": 1500},
        ]
        points = score_sync_points(slides)
        self.assertEqual([p["at_ms"] for p in points], [0, 2000, 5000])

    def test_section_changes_are_accented(self):
        slides = [
            {"index": 0, "kind": "body", "duration_ms": 1000},
            {"index": 1, "kind": "heading", "duration_ms": 1000},
        ]
        points = score_sync_points(slides)
        self.assertFalse(points[0]["accent"])
        self.assertTrue(points[1]["accent"])
