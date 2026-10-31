from __future__ import annotations

import copy

import pytest

from web.wizard_chat.eye_signal import normalize_eye_signal, response_eye_fields


def trajectory():
    return {"eye_signal": {"version": 1, "coordinate_space": "coding/semantic-v1",
                           "frames": [{"at_ms": 0, "vector": [0, -1], "word_index": 0},
                                      {"at_ms": 250, "vector": [1, 0], "word_index": 1}]}}


def test_native_vector_values_and_word_timing_survive_adapter():
    signal = normalize_eye_signal(trajectory())
    assert signal["source"] == "model-vector"
    assert signal["coordinate_space"] == "coding/semantic-v1"
    assert signal["frames"] == trajectory()["eye_signal"]["frames"]


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.update(coordinate_space=""),
    lambda raw: raw["frames"][1].update(at_ms=0),
    lambda raw: raw["frames"][1].update(at_ms=-1),
    lambda raw: raw["frames"][1].update(vector=[1]),
    lambda raw: raw["frames"][0].update(vector=[float("nan"), 0]),
    lambda raw: raw["frames"][0].update(vector=[True, 0]),
    lambda raw: raw["frames"][0].update(vector=[float("inf"), 0]),
    lambda raw: raw["frames"][0].update(vector=[0] * 257),
    lambda raw: raw.update(frames=raw["frames"] * 129),
])
def test_invalid_trajectory_never_becomes_semantic_telemetry(mutation):
    payload = copy.deepcopy(trajectory())
    mutation(payload["eye_signal"])
    payload["activated_concepts"] = ["moon"]
    signal = normalize_eye_signal(payload)
    assert signal["source"] == "model-concepts"
    assert signal["frames"] == []


def test_reply_text_does_not_manufacture_semantic_vectors():
    assert normalize_eye_signal({"answer": "The universe is vast"})["source"] == "none"
    assert normalize_eye_signal({"activated_concepts": ["moon", "moon", None]})["concepts"] == ["moon"]


def test_augmented_answer_does_not_use_original_audio_or_trajectory():
    payload = {**trajectory(), "audio_b64": "VOICE", "images": ["IMAGE"],
               "core_prerogatives": {"status": "compliant", "compliant": True}}
    native = response_eye_fields(payload, answer_matches_model=True)
    assert native["audio_b64"] == "VOICE"
    augmented = response_eye_fields(payload, answer_matches_model=False)
    assert "audio_b64" not in augmented and "images" not in augmented
    assert augmented["eye_signal"]["source"] == "none"
    assert augmented["core_prerogatives"]["compliant"] is True
