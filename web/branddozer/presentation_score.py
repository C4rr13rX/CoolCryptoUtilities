"""branddozer/presentation_score.py — symbolic score, timed to the narration.

Why symbolic rather than generative audio
-----------------------------------------
An audio model (MusicGen and friends) returns a waveform: you cannot ask
it to place a chord change at 3.200s, and on this machine (i5-8500, no
CUDA) it runs ~50x slower than realtime — one 81-minute deck measured out
at 7+ hours.

So the score is written *symbolically* by the site's selected model —
notes, durations, key, instrument — and rendered locally with numpy. That
inverts the problem:

* **Exact sync.** Slide boundaries and Polly word timings are known before
  a note is written, so they are passed in as constraints and the composer
  places chord changes on them. Alignment is specified, not hoped for.
* **Seconds, not beats.** Note times are absolute seconds, so a score
  never drifts against narration regardless of tempo.
* **Cheap.** Rendering is arithmetic over a numpy array: a minute of audio
  costs milliseconds, and nothing needs a GPU.

The rendered result is a plain 16-bit WAV, and the same JSON exports to a
standard MIDI file for editing in a DAW.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SAMPLE_RATE = 44100

# Keys carry conventional affect. Stated explicitly so the composer is
# choosing from a described palette rather than guessing, and so the
# choice can be justified against the paper's findings.
KEY_AFFECT = {
    "A minor": "sober, unresolved, neutral gravity",
    "D minor": "grave, weighty, elegiac",
    "E minor": "restless, searching",
    "C minor": "severe, formal",
    "F# minor": "tense, unsettled",
    "C major": "plain, declarative, open",
    "G major": "affirmative, warm",
    "F major": "calm, pastoral, settled",
    "Bb major": "measured, institutional",
    "D dorian": "modal, analytic, non-committal",
    "A aeolian": "archival, cool, observational",
    "E phrygian": "austere, foreboding",
}

# Instruments that sit under speech without masking it. General MIDI
# program numbers; the renderer maps these onto its own timbres.
NARRATION_SAFE_PROGRAMS = {
    89: "warm pad",
    90: "polysynth pad",
    49: "string ensemble",
    0: "grand piano",
    11: "vibraphone",
    46: "orchestral harp",
    52: "choir aahs",
}

# Speech occupies roughly 85-255 Hz fundamental with formants above; a
# score staying under C5 avoids competing with intelligibility.
MAX_PITCH_UNDER_SPEECH = 72  # MIDI C5


@dataclass
class ScoreRequest:
    """Everything the composer needs to write a synchronised cue."""

    title: str
    abstract: str
    duration_ms: int
    transitions_ms: list[int]
    findings_tone: str = "neutral"
    mood: list[str] | None = None


def compose_prompt(request: ScoreRequest) -> str:
    """Prompt the selected model for a symbolic, pre-synchronised score."""
    seconds = round(request.duration_ms / 1000, 3)
    marks = [round(t / 1000, 3) for t in request.transitions_ms]
    # A long deck has hundreds of transitions; naming them all would swamp
    # the prompt, so constrain against the ones a listener actually hears.
    shown = marks[:40]
    keys = "\n".join(f"  - {k}: {v}" for k, v in KEY_AFFECT.items())
    programs = "\n".join(
        f"  - {num}: {name}" for num, name in NARRATION_SAFE_PROGRAMS.items()
    )
    mood = ", ".join(request.mood or []) or "measured, evidential"
    return (
        "Compose an instrumental cue for a research-paper video presentation. "
        "Return STRICT JSON only.\n\n"
        f"PAPER: {request.title}\n"
        f"ABSTRACT: {request.abstract[:1200]}\n"
        f"FINDINGS TONE: {request.findings_tone} — the music must match this "
        "honestly. A negative or inconclusive result must not sound "
        "triumphant or resolved.\n"
        f"VISUAL MOOD: {mood}\n\n"
        f"KEYS (choose one and justify it):\n{keys}\n\n"
        f"INSTRUMENTS (General MIDI, all safe under speech):\n{programs}\n\n"
        "HARD CONSTRAINTS:\n"
        f"- Total length exactly {seconds} seconds. The last note must end "
        "at or before that time.\n"
        f"- A chord change must land on each of these slide-transition "
        f"times (seconds): {shown}\n"
        f"- No pitch above MIDI {MAX_PITCH_UNDER_SPEECH} (C5): the cue sits "
        "beneath spoken narration and must not mask it.\n"
        "- Sparse. Long sustained notes, few onsets, no percussion, no "
        "sudden dynamics, no melody that draws attention from the words.\n\n"
        "Return JSON with: key, scale, bpm, time_signature, rationale (why "
        "this key and register suit THIS paper's subject and the honesty of "
        "its findings), and tracks — a list of {name, program, notes}, where "
        "each note is {pitch, start, dur, vel} with start and dur in "
        "SECONDS as floats, pitch a MIDI number, and vel 1-127."
    )


def _parse(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("composer returned no JSON object")
    return json.loads(raw[start:end])


def validate_score(score: dict[str, Any], request: ScoreRequest) -> dict[str, Any]:
    """Clamp a composed score to the constraints the renderer relies on.

    The model is reliable but not guaranteed: a single out-of-range pitch
    or an overrunning note would either mask the narration or desync the
    deck, so both are corrected here rather than trusted.
    """
    limit = request.duration_ms / 1000
    issues: list[str] = []
    tracks: list[dict[str, Any]] = []

    for track in score.get("tracks") or []:
        if not isinstance(track, dict):
            continue
        program = int(track.get("program") or 89)
        if program not in NARRATION_SAFE_PROGRAMS:
            issues.append(f"program {program} not narration-safe; using pad")
            program = 89
        notes: list[dict[str, Any]] = []
        for note in track.get("notes") or []:
            try:
                pitch = int(note["pitch"])
                start = float(note["start"])
                dur = float(note["dur"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or dur <= 0 or start >= limit:
                continue
            if pitch > MAX_PITCH_UNDER_SPEECH:
                # Drop by octaves rather than discarding the note, so the
                # harmony survives even when the register was wrong.
                while pitch > MAX_PITCH_UNDER_SPEECH:
                    pitch -= 12
                issues.append("pitch above C5 transposed down")
            if start + dur > limit:
                dur = limit - start
                issues.append("note truncated to fit the deck length")
            notes.append(
                {
                    "pitch": max(0, min(127, pitch)),
                    "start": start,
                    "dur": dur,
                    "vel": max(1, min(127, int(note.get("vel") or 48))),
                }
            )
        if notes:
            tracks.append(
                {"name": str(track.get("name") or "score"), "program": program, "notes": notes}
            )

    return {
        "key": str(score.get("key") or "A minor"),
        "scale": str(score.get("scale") or "natural minor"),
        "bpm": int(score.get("bpm") or 60),
        "time_signature": str(score.get("time_signature") or "4/4"),
        "rationale": str(score.get("rationale") or ""),
        "tracks": tracks,
        "duration_ms": request.duration_ms,
        "issues": issues,
    }


def alignment_report(
    score: dict[str, Any], transitions_ms: Iterable[int], tolerance_ms: int = 120
) -> dict[str, Any]:
    """How well the score's onsets land on slide transitions."""
    onsets = sorted(
        {
            round(float(note["start"]) * 1000)
            for track in score.get("tracks") or []
            for note in track.get("notes") or []
        }
    )
    targets = list(transitions_ms)
    hits = 0
    for target in targets:
        if any(abs(onset - target) <= tolerance_ms for onset in onsets):
            hits += 1
    return {
        "transitions": len(targets),
        "aligned": hits,
        "alignment_rate": round(hits / len(targets), 3) if targets else 1.0,
        "onsets": len(onsets),
    }


def _adsr(length: int, attack: float, release: float) -> Any:
    """Gentle envelope. Sharp edges click and draw attention to the score."""
    import numpy as np

    envelope = np.ones(length, dtype="float32")
    attack_n = min(int(SAMPLE_RATE * attack), length // 2)
    release_n = min(int(SAMPLE_RATE * release), length // 2)
    if attack_n > 0:
        envelope[:attack_n] = np.linspace(0.0, 1.0, attack_n, dtype="float32")
    if release_n > 0:
        envelope[-release_n:] = np.linspace(1.0, 0.0, release_n, dtype="float32")
    return envelope


def render_wav(score: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Render a validated score to a 16-bit WAV using numpy only.

    Additive synthesis with a mild harmonic series: enough timbre to sound
    intentional, cheap enough that a full deck renders in well under a
    second, and dependency-free (no GPU, no ffmpeg, no soundfont).
    """
    import numpy as np

    duration_s = max(0.1, score.get("duration_ms", 0) / 1000)
    total = int(SAMPLE_RATE * duration_s)
    buffer = np.zeros(total, dtype="float32")

    for track in score.get("tracks") or []:
        for note in track.get("notes") or []:
            start_n = int(float(note["start"]) * SAMPLE_RATE)
            length = int(float(note["dur"]) * SAMPLE_RATE)
            if length <= 0 or start_n >= total:
                continue
            length = min(length, total - start_n)
            freq = 440.0 * (2.0 ** ((int(note["pitch"]) - 69) / 12.0))
            t = np.arange(length, dtype="float32") / SAMPLE_RATE
            wave = (
                np.sin(2 * math.pi * freq * t)
                + 0.28 * np.sin(2 * math.pi * freq * 2 * t)
                + 0.12 * np.sin(2 * math.pi * freq * 3 * t)
            ).astype("float32")
            amplitude = (int(note.get("vel") or 48) / 127.0) * 0.22
            buffer[start_n:start_n + length] += (
                wave * _adsr(length, 0.35, 0.6) * amplitude
            )

    peak = float(np.max(np.abs(buffer))) if total else 0.0
    if peak > 0:
        # Leave headroom so the narration always sits above the score.
        buffer = buffer / peak * 0.5

    pcm = (buffer * 32767).astype("<i2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(out_path, pcm.tobytes())
    return {
        "path": str(out_path),
        "duration_ms": int(duration_s * 1000),
        "peak": round(peak, 4),
        "samples": total,
    }


def _write_wav(path: Path, pcm: bytes) -> None:
    """Minimal RIFF writer; avoids a soundfile/pydub dependency."""
    byte_rate = SAMPLE_RATE * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, byte_rate, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    path.write_bytes(header + pcm)


def export_midi(score: dict[str, Any], out_path: Path, *, ticks: int = 480) -> str:
    """Write the score as a type-1 MIDI file, for editing in a DAW."""
    bpm = max(1, int(score.get("bpm") or 60))
    us_per_beat = int(60_000_000 / bpm)

    def varint(value: int) -> bytes:
        buf = value & 0x7F
        value >>= 7
        while value:
            buf <<= 8
            buf |= ((value & 0x7F) | 0x80)
            value >>= 7
        out = b""
        while True:
            out += bytes([buf & 0xFF])
            if buf & 0x80:
                buf >>= 8
            else:
                break
        return out

    def sec_to_ticks(seconds: float) -> int:
        return int(round(seconds * (bpm / 60.0) * ticks))

    tracks_data: list[bytes] = []
    tempo = b"\x00\xff\x51\x03" + us_per_beat.to_bytes(3, "big")
    tracks_data.append(tempo + b"\x00\xff\x2f\x00")

    for channel, track in enumerate(score.get("tracks") or []):
        events: list[tuple[int, bytes]] = []
        program = int(track.get("program") or 89)
        events.append((0, bytes([0xC0 | (channel & 0x0F), program & 0x7F])))
        for note in track.get("notes") or []:
            on = sec_to_ticks(float(note["start"]))
            off = sec_to_ticks(float(note["start"]) + float(note["dur"]))
            pitch = int(note["pitch"]) & 0x7F
            vel = int(note.get("vel") or 48) & 0x7F
            events.append((on, bytes([0x90 | (channel & 0x0F), pitch, vel])))
            events.append((off, bytes([0x80 | (channel & 0x0F), pitch, 0])))
        events.sort(key=lambda item: item[0])
        chunk = b""
        previous = 0
        for at, payload in events:
            chunk += varint(max(0, at - previous)) + payload
            previous = at
        tracks_data.append(chunk + b"\x00\xff\x2f\x00")

    out = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks_data), ticks)
    for data in tracks_data:
        out += b"MTrk" + struct.pack(">I", len(data)) + data
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(out)
    return str(out_path)


def compose_score(
    request: ScoreRequest, *, agent_send
) -> dict[str, Any]:
    """Ask the selected model for a score, then clamp it to constraints."""
    raw = agent_send(
        compose_prompt(request),
        system="You are a film composer. Return strict JSON only.",
    )
    return validate_score(_parse(raw), request)


__all__ = [
    "SAMPLE_RATE",
    "KEY_AFFECT",
    "NARRATION_SAFE_PROGRAMS",
    "MAX_PITCH_UNDER_SPEECH",
    "ScoreRequest",
    "compose_prompt",
    "compose_score",
    "validate_score",
    "alignment_report",
    "render_wav",
    "export_midi",
]
