"""branddozer/presentation_media.py — narration, artwork and palette.

Turns the adjacent presentation schema into playable media:

* **Narration** — AWS Polly (the `FountainServer` profile), using *speech
  marks* so every word carries a real millisecond offset. Word highlighting
  and word-synced animation are therefore exact, not interpolated.

* **Backgrounds** — OpenAI image generation, one per slide group, prompted
  from the slide's own content so the artwork is about the subject.

* **Palette** — chosen by the site's currently selected agent under stated
  colour-theory rules (complementary, analogous, triadic, …) rather than
  sampled at random, and justified against the paper's subject so the
  choice carries the intended psychological reading.

No background score
-------------------
Music generation is deliberately absent. This machine has no CUDA GPU
(i5-8500 / Intel UHD 630), and MusicGen-class models run roughly 50x
slower than realtime on CPU — scoring one 81-minute deck measured out at
7+ hours. ``score_sync_points`` is kept because the timeline marks it
produces are what drive *visual* transitions; if a scoring provider is
added later it can consume the same marks unchanged.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# --- Colour theory --------------------------------------------------------
# Named schemes the agent may choose from. Ratios are the share of the
# frame each role should occupy; these are the conventional splits used in
# editorial and UI design, not arbitrary numbers.
COLOR_SCHEMES = {
    "complementary": "Two hues opposite on the wheel; maximum contrast, high tension.",
    "split_complementary": "A base hue plus the two neighbours of its complement; contrast without harshness.",
    "analogous": "Three adjacent hues; calm, cohesive, low conflict.",
    "triadic": "Three hues evenly spaced; vivid and balanced.",
    "tetradic": "Two complementary pairs; rich but needs one dominant hue.",
    "monochromatic": "One hue across values; austere, authoritative, documentary.",
    "achromatic_accent": "Neutral field with a single saturated accent; forensic, evidential.",
}

# Distribution presets in common editorial use. The 60/30/10 rule is the
# web default; the others suit denser or more austere subjects.
COLOR_RATIOS = {
    "60_30_10": [60, 30, 10],
    "70_20_10": [70, 20, 10],
    "50_30_20": [50, 30, 20],
    "80_15_5": [80, 15, 5],
    "40_30_20_10": [40, 30, 20, 10],
}

TRANSITIONS = (
    "cut", "fade", "crossfade", "slide_left", "slide_up", "push",
    "zoom_in", "zoom_out", "wipe", "dissolve", "blur_through",
)

WORD_ANIMATIONS = (
    "none", "highlight", "fade_in", "pop", "rise", "typewriter", "underline",
)


@dataclass
class MediaConfig:
    voice_id: str = "Joanna"
    engine: str = "neural"
    aws_profile: str = "FountainServer"
    color_scheme: str = ""          # blank -> the agent chooses
    color_ratio: str = "60_30_10"
    transition: str = "crossfade"
    word_animation: str = "highlight"
    generate_backgrounds: bool = True


# --- Narration ------------------------------------------------------------

def synthesize_slide(
    text: str, *, config: MediaConfig, out_dir: Path, name: str
) -> dict[str, Any]:
    """Synthesise one slide and return its audio path plus word timings.

    Two Polly calls are required: one for the audio, one for the speech
    marks. Polly will not return both in a single response.
    """
    import boto3

    out_dir.mkdir(parents=True, exist_ok=True)
    session = boto3.Session(profile_name=config.aws_profile)
    polly = session.client("polly")

    audio = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=config.voice_id,
        Engine=config.engine,
    )
    audio_path = out_dir / f"{name}.mp3"
    audio_path.write_bytes(audio["AudioStream"].read())

    marks_response = polly.synthesize_speech(
        Text=text,
        OutputFormat="json",
        SpeechMarkTypes=["word"],
        VoiceId=config.voice_id,
        Engine=config.engine,
    )
    marks = [
        json.loads(line)
        for line in marks_response["AudioStream"].read().decode("utf-8").splitlines()
        if line.strip()
    ]

    # Polly marks give each word's start only. The clip runs a little past
    # the final word, so estimate the tail from the average word gap rather
    # than cutting the last word's highlight short.
    word_marks = [m for m in marks if m.get("type") == "word"]
    if len(word_marks) >= 2:
        gaps = [
            int(word_marks[i + 1]["time"]) - int(word_marks[i]["time"])
            for i in range(len(word_marks) - 1)
        ]
        tail = int(sum(gaps) / len(gaps))
    else:
        tail = 600
    audio_ms = (int(word_marks[-1]["time"]) + tail) if word_marks else 800

    return {
        "audio_path": str(audio_path),
        "marks": word_marks,
        "audio_ms": audio_ms,
    }


# --- Palette --------------------------------------------------------------

def palette_prompt(title: str, abstract: str, *, ratio: str) -> str:
    """Ask the selected agent to choose a scheme and justify it."""
    ratios = COLOR_RATIOS.get(ratio, COLOR_RATIOS["60_30_10"])
    schemes = "\n".join(f"  - {k}: {v}" for k, v in COLOR_SCHEMES.items())
    return (
        "Choose a colour scheme for a research-paper video presentation. "
        "Return STRICT JSON only.\n\n"
        f"PAPER TITLE: {title}\n"
        f"ABSTRACT: {abstract[:1500]}\n\n"
        f"AVAILABLE SCHEMES (pick exactly one and name it):\n{schemes}\n\n"
        f"DISTRIBUTION: assign the hues to these shares of the frame: {ratios}.\n\n"
        "Return JSON with: scheme (one of the names above), rationale "
        "(why this scheme suits THIS subject psychologically — tone, "
        "seriousness, whether the findings are affirmative or negative), "
        "colors (a list matching the distribution, each with hex, role "
        "['dominant','secondary','accent','support'], share (int), and "
        "meaning), text_color (hex, must meet WCAG AA against the dominant "
        "colour), and mood (three adjectives).\n"
        "Do not pick colours at random and do not default to blue: the "
        "scheme must follow the named colour-theory relationship exactly."
    )


def choose_palette(
    title: str, abstract: str, *, config: MediaConfig, agent_send
) -> dict[str, Any]:
    """Run the palette prompt through the site's selected agent."""
    if config.color_scheme:
        # An explicit user choice overrides the agent's judgement.
        return {
            "scheme": config.color_scheme,
            "rationale": "explicitly selected",
            "colors": [],
            "ratio": config.color_ratio,
        }
    raw = agent_send(
        palette_prompt(title, abstract, ratio=config.color_ratio),
        system="You are an art director. Return strict JSON only.",
    )
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        payload = json.loads(raw[start:end])
    except Exception:
        payload = {}
    payload.setdefault("scheme", "achromatic_accent")
    payload.setdefault("ratio", config.color_ratio)
    return payload


# --- Backgrounds ----------------------------------------------------------

def background_prompt(slide_text: str, section: str, palette: dict[str, Any]) -> str:
    """Compose an image prompt tied to the slide and the chosen palette."""
    hexes = ", ".join(
        str(c.get("hex")) for c in (palette.get("colors") or []) if c.get("hex")
    )
    mood = ", ".join(palette.get("mood") or []) or "measured, evidential"
    return (
        "Abstract editorial background for a research presentation slide. "
        f"Section: {section or 'introduction'}. "
        f"Slide content: {slide_text[:220]}. "
        f"Palette ({palette.get('scheme', 'achromatic accent')}): {hexes or 'muted neutrals with one accent'}. "
        f"Mood: {mood}. "
        "No text, no words, no letters, no charts, no logos. Leave the "
        "centre visually quiet so large type remains legible. Subtle grain, "
        "high production value, 16:9."
    )


def generate_background(
    prompt: str, *, out_dir: Path, name: str, api_key: str = ""
) -> str:
    """Generate one background image. Returns a path, or '' on failure."""
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1536x1024",
            n=1,
        )
        payload = result.data[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.png"
        if getattr(payload, "b64_json", None):
            path.write_bytes(base64.b64decode(payload.b64_json))
            return str(path)
        url = getattr(payload, "url", "")
        if url:
            import urllib.request

            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
            return str(path)
    except Exception:
        return ""
    return ""


# --- Timeline -------------------------------------------------------------

def score_sync_points(slides: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cumulative timeline marks for slide changes.

    Gives each slide its absolute start time, so the player can drive
    transitions and word animation from one timeline. Section boundaries
    are marked `accent` — the natural place for a stronger transition.
    """
    points: list[dict[str, Any]] = []
    elapsed = 0
    for slide in slides:
        points.append(
            {
                "slide_index": slide.get("index"),
                "at_ms": elapsed,
                "kind": slide.get("kind"),
                "accent": slide.get("kind") in {"title", "heading", "subtitle"},
            }
        )
        elapsed += int(slide.get("duration_ms") or 0)
    return points


__all__ = [
    "MediaConfig",
    "COLOR_SCHEMES",
    "COLOR_RATIOS",
    "TRANSITIONS",
    "WORD_ANIMATIONS",
    "synthesize_slide",
    "palette_prompt",
    "choose_palette",
    "background_prompt",
    "generate_background",
    "score_sync_points",
]
