"""MP4 export: portrait by default, and audio that stays in sync."""
from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import TestCase

from branddozer.presentation_video import (
    ASPECT_RATIOS,
    DEFAULT_ASPECT,
    DEFAULT_TRANSITION,
    DEFAULT_WORD_ANIMATION,
    TRANSITION_MS,
    VideoConfig,
    attach_backgrounds,
    build_audio_track,
    export_mp4,
    render_slide_frame,
)


def _slides(count=4, duration=1500):
    return [
        {
            "index": i,
            "kind": "body",
            "text": f"Slide {i} text here",
            "section": "Findings",
            "words": [],
            "duration_ms": duration,
        }
        for i in range(count)
    ]


class DefaultsTests(TestCase):
    def test_portrait_is_the_default(self):
        self.assertEqual(DEFAULT_ASPECT, "9:16")
        self.assertEqual(VideoConfig().size(), (1080, 1920))

    def test_frame_is_taller_than_wide(self):
        """A landscape export would mean the default silently changed."""
        width, height = VideoConfig().size()
        self.assertGreater(height, width)

    def test_cube_flip_and_fade_in_are_the_defaults(self):
        self.assertEqual(DEFAULT_TRANSITION, "cube_flip")
        self.assertEqual(DEFAULT_WORD_ANIMATION, "fade_in")
        self.assertEqual(VideoConfig().transition, "cube_flip")
        self.assertEqual(VideoConfig().word_animation, "fade_in")

    def test_every_ratio_is_a_valid_size(self):
        for name, (w, h) in ASPECT_RATIOS.items():
            self.assertGreater(w, 0, name)
            self.assertGreater(h, 0, name)


class AudioSyncTests(TestCase):
    """Regression: transitions added 420ms of picture per slide but no
    audio, so narration ended 64 seconds behind the video."""

    def _build(self, transition_ms):
        out = Path(tempfile.mkdtemp()) / "a.wav"
        slides = _slides(count=5, duration=2000)
        # No audio files exist, so every slide pads to silence — which is
        # exactly what makes the length arithmetic testable.
        result = build_audio_track(
            slides, Path(tempfile.mkdtemp()), out, fps=30,
            transition_ms=transition_ms,
        )
        return result, slides

    def test_audio_covers_transition_gaps(self):
        result, slides = self._build(TRANSITION_MS)
        expected = sum(s["duration_ms"] for s in slides) + 4 * TRANSITION_MS
        self.assertAlmostEqual(result["duration_ms"], expected, delta=60)

    def test_without_transitions_audio_matches_slide_durations(self):
        result, slides = self._build(0)
        expected = sum(s["duration_ms"] for s in slides)
        self.assertAlmostEqual(result["duration_ms"], expected, delta=60)

    def test_transition_padding_is_what_drifted(self):
        """The unpadded track is measurably shorter — the original bug."""
        padded, _ = self._build(TRANSITION_MS)
        unpadded, _ = self._build(0)
        self.assertGreater(padded["duration_ms"], unpadded["duration_ms"])


class FrameTests(TestCase):
    def test_frame_matches_configured_size(self):
        frame = render_slide_frame(_slides(1)[0], 0, VideoConfig())
        self.assertEqual(frame.size, (1080, 1920))

    def test_landscape_config_produces_landscape_frame(self):
        config = VideoConfig(aspect="16:9")
        frame = render_slide_frame(_slides(1)[0], 0, config)
        self.assertEqual(frame.size, (1920, 1080))

    def test_missing_background_does_not_fail_the_frame(self):
        slide = _slides(1)[0]
        slide["background_path"] = "/nonexistent/image.png"
        frame = render_slide_frame(slide, 0, VideoConfig())
        self.assertEqual(frame.size, (1080, 1920))

    def test_empty_slide_still_renders(self):
        frame = render_slide_frame(
            {"index": 0, "kind": "body", "text": "", "words": []}, 0, VideoConfig()
        )
        self.assertEqual(frame.size, (1080, 1920))


class BackgroundTests(TestCase):
    def test_backgrounds_are_generated_per_section_not_per_slide(self):
        """154 slides must not mean 154 image calls."""
        deck = {"slides": _slides(count=20)}
        calls = []

        def fake_generate(prompt, *, out_dir, name, api_key=""):
            calls.append(name)
            return ""

        import branddozer.presentation_media as media

        original = media.generate_background
        media.generate_background = fake_generate
        try:
            result = attach_backgrounds(
                deck, out_dir=Path(tempfile.mkdtemp()), api_key="k"
            )
        finally:
            media.generate_background = original
        # All 20 slides share one section.
        self.assertEqual(result["sections"], 1)
        self.assertEqual(len(calls), 1)


class ExportTests(TestCase):
    def test_short_deck_exports_a_playable_file(self):
        out = Path(tempfile.mkdtemp()) / "deck.mp4"
        result = export_mp4(_slides(count=2, duration=500), out_path=out) \
            if False else export_mp4({"slides": _slides(2, 500)}, out_path=out)
        self.assertTrue(Path(result["path"]).is_file())
        self.assertGreater(result["bytes"], 1000)
        self.assertEqual(result["resolution"], "1080x1920")

    def test_empty_deck_raises(self):
        with self.assertRaises(ValueError):
            export_mp4({"slides": []}, out_path=Path(tempfile.mkdtemp()) / "x.mp4")
