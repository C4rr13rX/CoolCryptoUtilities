"""Symbolic score: composed to constraints, clamped, rendered locally.

The point of composing symbolically rather than generating audio is that
slide transitions and narration timings are known *before* a note exists,
so alignment is specified rather than hoped for. These tests hold that
guarantee, and the safety clamps that protect narration intelligibility.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import TestCase

from branddozer.presentation_score import (
    KEY_AFFECT,
    MAX_PITCH_UNDER_SPEECH,
    NARRATION_SAFE_PROGRAMS,
    ScoreRequest,
    alignment_report,
    compose_prompt,
    compose_score,
    export_midi,
    render_wav,
    validate_score,
)


REQUEST = ScoreRequest(
    title="A Paper",
    abstract="An abstract.",
    duration_ms=8000,
    transitions_ms=[0, 3200, 5900],
    findings_tone="negative / inconclusive",
    mood=["sober"],
)


def _score(**overrides):
    score = {
        "key": "A minor",
        "scale": "natural minor",
        "bpm": 60,
        "tracks": [
            {
                "name": "pad",
                "program": 89,
                "notes": [
                    {"pitch": 57, "start": 0.0, "dur": 3.2, "vel": 48},
                    {"pitch": 60, "start": 3.2, "dur": 2.7, "vel": 44},
                    {"pitch": 62, "start": 5.9, "dur": 2.1, "vel": 40},
                ],
            }
        ],
    }
    score.update(overrides)
    return score


class PromptTests(TestCase):
    def test_prompt_states_the_hard_constraints(self):
        text = compose_prompt(REQUEST)
        self.assertIn("8.0 seconds", text)
        self.assertIn("3.2", text)          # a transition time
        self.assertIn(str(MAX_PITCH_UNDER_SPEECH), text)

    def test_prompt_offers_keys_with_stated_affect(self):
        text = compose_prompt(REQUEST)
        for key in ("A minor", "C major", "D dorian"):
            self.assertIn(key, text)

    def test_prompt_demands_honesty_about_findings(self):
        text = compose_prompt(REQUEST)
        self.assertIn("negative / inconclusive", text)
        self.assertIn("must not sound", text)

    def test_long_decks_do_not_swamp_the_prompt(self):
        """Hundreds of transitions must not blow up the prompt."""
        request = ScoreRequest(
            title="T", abstract="A", duration_ms=600000,
            transitions_ms=list(range(0, 600000, 1000)),
        )
        self.assertLess(len(compose_prompt(request)), 6000)


class ValidationTests(TestCase):
    def test_clean_score_passes_untouched(self):
        result = validate_score(_score(), REQUEST)
        self.assertEqual(result["issues"], [])
        self.assertEqual(len(result["tracks"][0]["notes"]), 3)

    def test_pitch_above_speech_range_is_transposed_not_dropped(self):
        """Keep the harmony; only fix the register."""
        score = _score()
        score["tracks"][0]["notes"][0]["pitch"] = 96
        result = validate_score(score, REQUEST)
        pitches = [n["pitch"] for n in result["tracks"][0]["notes"]]
        self.assertTrue(all(p <= MAX_PITCH_UNDER_SPEECH for p in pitches))
        self.assertEqual(len(pitches), 3)

    def test_note_running_past_the_deck_is_truncated(self):
        score = _score()
        score["tracks"][0]["notes"][2]["dur"] = 99.0
        result = validate_score(score, REQUEST)
        end = max(n["start"] + n["dur"] for n in result["tracks"][0]["notes"])
        self.assertLessEqual(end, REQUEST.duration_ms / 1000)

    def test_unsafe_instrument_falls_back_to_a_pad(self):
        score = _score()
        score["tracks"][0]["program"] = 127  # gunshot
        result = validate_score(score, REQUEST)
        self.assertIn(result["tracks"][0]["program"], NARRATION_SAFE_PROGRAMS)

    def test_malformed_notes_are_skipped(self):
        score = _score()
        score["tracks"][0]["notes"].append({"pitch": "x"})
        score["tracks"][0]["notes"].append({"pitch": 60, "start": -1, "dur": 1})
        result = validate_score(score, REQUEST)
        self.assertEqual(len(result["tracks"][0]["notes"]), 3)

    def test_empty_score_does_not_crash(self):
        result = validate_score({}, REQUEST)
        self.assertEqual(result["tracks"], [])


class AlignmentTests(TestCase):
    def test_onsets_on_transitions_score_full_alignment(self):
        report = alignment_report(_score(), [0, 3200, 5900])
        self.assertEqual(report["alignment_rate"], 1.0)

    def test_missed_transitions_are_reported(self):
        report = alignment_report(_score(), [0, 3200, 5900, 7500])
        self.assertLess(report["alignment_rate"], 1.0)
        self.assertEqual(report["aligned"], 3)

    def test_small_drift_is_tolerated(self):
        """A few ms of rounding must not read as a miss."""
        report = alignment_report(_score(), [50, 3250, 5950], tolerance_ms=120)
        self.assertEqual(report["alignment_rate"], 1.0)


class RenderTests(TestCase):
    def _out(self):
        return Path(tempfile.mkdtemp())

    def test_wav_is_written_and_non_silent(self):
        score = validate_score(_score(), REQUEST)
        score["duration_ms"] = 8000
        result = render_wav(score, self._out() / "s.wav")
        path = Path(result["path"])
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 44)   # bigger than a header
        self.assertGreater(result["peak"], 0.0)

    def test_wav_length_matches_the_deck(self):
        score = validate_score(_score(), REQUEST)
        score["duration_ms"] = 8000
        result = render_wav(score, self._out() / "s.wav")
        self.assertEqual(result["duration_ms"], 8000)

    def test_silent_score_still_produces_a_file(self):
        result = render_wav({"duration_ms": 2000, "tracks": []}, self._out() / "s.wav")
        self.assertTrue(Path(result["path"]).is_file())
        self.assertEqual(result["peak"], 0.0)

    def test_midi_export_is_a_valid_header(self):
        score = validate_score(_score(), REQUEST)
        path = Path(export_midi(score, self._out() / "s.mid"))
        data = path.read_bytes()
        self.assertTrue(data.startswith(b"MThd"))
        self.assertIn(b"MTrk", data)


class ComposeTests(TestCase):
    def test_compose_clamps_whatever_the_model_returns(self):
        """A model breaking every constraint must still yield a safe score."""
        bad = (
            '{"key":"A minor","bpm":60,"tracks":[{"name":"lead","program":127,'
            '"notes":[{"pitch":110,"start":0.0,"dur":999,"vel":200}]}]}'
        )
        result = compose_score(REQUEST, agent_send=lambda p, system="": bad)
        note = result["tracks"][0]["notes"][0]
        self.assertLessEqual(note["pitch"], MAX_PITCH_UNDER_SPEECH)
        self.assertLessEqual(note["start"] + note["dur"], 8.0)
        self.assertLessEqual(note["vel"], 127)
        self.assertIn(result["tracks"][0]["program"], NARRATION_SAFE_PROGRAMS)

    def test_non_json_response_raises(self):
        with self.assertRaises(ValueError):
            compose_score(REQUEST, agent_send=lambda p, system="": "no json here")

    def test_every_offered_key_has_stated_affect(self):
        for key, affect in KEY_AFFECT.items():
            self.assertTrue(affect.strip(), f"{key} has no affect description")
