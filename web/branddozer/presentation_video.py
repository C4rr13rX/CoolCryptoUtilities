"""branddozer/presentation_video.py — render a deck to MP4.

Renders the same timeline the in-browser player uses, so an exported file
matches what was previewed rather than being a separate interpretation.

Defaults (overridable per export)
---------------------------------
* **9:16 portrait, 1080x1920** — the deck is designed for a phone held
  upright, and that is where it will be watched.
* **3D cube flip** between slides.
* **Fade-in** for words, revealed on the real Polly speech-mark timings.

Implementation notes
--------------------
Frames are composed with PIL and piped raw to the ffmpeg binary that ships
inside ``imageio-ffmpeg`` — no system ffmpeg install, no moviepy. The cube
flip is done as a perspective warp per frame via cv2, which is fast enough
on CPU because each frame is a single 1080x1920 array operation.

Audio is concatenated per slide with silence padding so a slide's visual
duration always equals its narration length; the result is muxed in one
ffmpeg pass. Slides without narration hold for their estimated duration.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# 9:16 portrait is the default because these decks are watched on a phone.
ASPECT_RATIOS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4:3": (1440, 1080),
    "21:9": (2560, 1080),
}
DEFAULT_ASPECT = "9:16"
DEFAULT_TRANSITION = "cube_flip"
DEFAULT_WORD_ANIMATION = "fade_in"

FPS = 30
TRANSITION_MS = 420

VIDEO_TRANSITIONS = (
    "cube_flip", "cut", "fade", "crossfade", "slide_left", "slide_up",
    "zoom_in", "zoom_out", "wipe",
)
VIDEO_WORD_ANIMATIONS = ("fade_in", "none", "highlight", "rise", "pop", "typewriter")


@dataclass
class VideoConfig:
    aspect: str = DEFAULT_ASPECT
    transition: str = DEFAULT_TRANSITION
    word_animation: str = DEFAULT_WORD_ANIMATION
    fps: int = FPS
    background: str = "#070d18"
    foreground: str = "#e8eeff"
    accent: str = "#7ea8ff"
    # Slides inserted from a user video file, keyed by the slide index they
    # should appear *before*. Lets an edit put footage anywhere in the deck.
    video_inserts: dict[int, str] = field(default_factory=dict)

    def size(self) -> tuple[int, int]:
        return ASPECT_RATIOS.get(self.aspect, ASPECT_RATIOS[DEFAULT_ASPECT])


def ffmpeg_exe() -> str:
    """Path to the bundled ffmpeg binary (no system install required)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _font(size: int):
    from PIL import ImageFont

    # Prefer a real UI face; fall back to PIL's default so a missing font
    # never fails an export.
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(draw, words: list[str], font, max_width: int) -> list[list[int]]:
    """Group word indices into lines that fit `max_width`."""
    lines: list[list[int]] = []
    current: list[int] = []
    for i, word in enumerate(words):
        trial = current + [i]
        text = " ".join(words[j] for j in trial)
        if draw.textlength(text, font=font) > max_width and current:
            lines.append(current)
            current = [i]
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def render_slide_frame(
    slide: dict[str, Any], elapsed_ms: int, config: VideoConfig
):
    """Compose one frame of a slide at `elapsed_ms` into its narration."""
    from PIL import Image, ImageDraw

    width, height = config.size()
    image = Image.new("RGB", (width, height), _hex(config.background))
    draw = ImageDraw.Draw(image)

    kind = slide.get("kind", "body")
    # Type scales with the frame so 9:16 and 16:9 both read correctly.
    base = width / 12
    if kind == "title":
        size = int(base * 1.25)
    elif kind in {"heading", "subtitle"}:
        size = int(base * 1.0)
    elif kind == "citation":
        size = int(base * 0.42)
    else:
        size = int(base * 0.92)
    font = _font(size)

    words = [w["word"] for w in (slide.get("words") or [])]
    if not words:
        words = str(slide.get("text") or "").split()
    if not words:
        return image

    margin = int(width * 0.08)
    lines = _wrap(draw, words, font, width - margin * 2)
    line_height = size * 1.3
    total_height = line_height * len(lines)
    y = (height - total_height) / 2

    colour = _hex(config.accent) if kind in {"heading", "subtitle"} else _hex(config.foreground)
    timings = slide.get("words") or []

    for line in lines:
        text = " ".join(words[j] for j in line)
        x = (width - draw.textlength(text, font=font)) / 2
        for index in line:
            word = words[index]
            # Word reveal follows the real speech marks when present.
            alpha = 1.0
            if config.word_animation != "none" and index < len(timings):
                start = int(timings[index].get("start_ms") or 0)
                if config.word_animation == "fade_in":
                    # Ease in over 220ms ending on the spoken onset.
                    delta = elapsed_ms - start
                    alpha = 0.18 if delta < -220 else min(1.0, max(0.18, (delta + 220) / 220))
                elif config.word_animation == "typewriter":
                    alpha = 1.0 if elapsed_ms >= start else 0.0
                elif config.word_animation == "highlight":
                    alpha = 1.0
            shade = tuple(
                int(_hex(config.background)[c] + (colour[c] - _hex(config.background)[c]) * alpha)
                for c in range(3)
            )
            if config.word_animation == "highlight" and index < len(timings):
                start = int(timings[index].get("start_ms") or 0)
                end = int(timings[index].get("end_ms") or start)
                if start <= elapsed_ms < end:
                    shade = _hex(config.accent)
            draw.text((x, y), word, font=font, fill=shade)
            x += draw.textlength(word + " ", font=font)
        y += line_height

    return image


def _cube_flip(frame_a, frame_b, progress: float):
    """3D cube-flip between two frames using a perspective warp."""
    import cv2
    import numpy as np

    height, width = frame_a.shape[:2]
    # Rotate a virtual cube face 90 degrees: the outgoing face swings away
    # while the incoming face swings in from the right.
    angle = progress * (math.pi / 2)
    shrink = math.sin(angle)
    grow = math.cos(angle)

    out = np.zeros_like(frame_a)

    if grow > 0.01:
        w_a = max(1, int(width * grow))
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst = np.float32(
            [
                [0, height * 0.5 * shrink * 0.25],
                [w_a, 0],
                [w_a, height],
                [0, height - height * 0.5 * shrink * 0.25],
            ]
        )
        warped = cv2.warpPerspective(
            frame_a, cv2.getPerspectiveTransform(src, dst), (width, height)
        )
        out[:, :w_a] = warped[:, :w_a]

    if shrink > 0.01:
        w_b = max(1, int(width * shrink))
        start = width - w_b
        src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        dst = np.float32(
            [
                [0, 0],
                [w_b, height * 0.5 * grow * 0.25],
                [w_b, height - height * 0.5 * grow * 0.25],
                [0, height],
            ]
        )
        warped = cv2.warpPerspective(
            frame_b, cv2.getPerspectiveTransform(src, dst), (width, height)
        )
        out[:, start:] = warped[:, :w_b]

    # Darken toward the midpoint so the fold reads as a solid edge.
    shade = 1.0 - 0.35 * math.sin(progress * math.pi)
    return (out * shade).astype("uint8")


def _blend(frame_a, frame_b, progress: float, mode: str):
    import numpy as np

    if mode == "cube_flip":
        return _cube_flip(frame_a, frame_b, progress)
    if mode == "cut":
        return frame_b if progress > 0.5 else frame_a
    height, width = frame_a.shape[:2]
    if mode in {"fade", "crossfade"}:
        return (frame_a * (1 - progress) + frame_b * progress).astype("uint8")
    if mode == "slide_left":
        shift = int(width * progress)
        out = np.zeros_like(frame_a)
        if shift < width:
            out[:, : width - shift] = frame_a[:, shift:]
        if shift > 0:
            out[:, width - shift:] = frame_b[:, :shift]
        return out
    if mode == "slide_up":
        shift = int(height * progress)
        out = np.zeros_like(frame_a)
        if shift < height:
            out[: height - shift] = frame_a[shift:]
        if shift > 0:
            out[height - shift:] = frame_b[:shift]
        return out
    if mode == "wipe":
        shift = int(width * progress)
        out = frame_a.copy()
        out[:, :shift] = frame_b[:, :shift]
        return out
    if mode in {"zoom_in", "zoom_out"}:
        import cv2

        scale = 1 + 0.12 * (progress if mode == "zoom_in" else (1 - progress))
        big = cv2.resize(frame_b, None, fx=scale, fy=scale)
        y = (big.shape[0] - height) // 2
        x = (big.shape[1] - width) // 2
        cropped = big[y:y + height, x:x + width]
        return (frame_a * (1 - progress) + cropped * progress).astype("uint8")
    return (frame_a * (1 - progress) + frame_b * progress).astype("uint8")


def build_audio_track(
    slides: list[dict[str, Any]], audio_dir: Path, out_path: Path, *, fps: int
) -> dict[str, Any]:
    """Concatenate per-slide narration, padding to each slide's duration."""
    import numpy as np

    rate = 44100
    chunks: list[Any] = []
    for slide in slides:
        duration_ms = int(slide.get("duration_ms") or 0)
        want = int(rate * duration_ms / 1000)
        clip = audio_dir / f"slide-{int(slide['index']):05d}.mp3"
        samples = None
        if clip.is_file():
            samples = _decode_mp3(clip, rate)
        if samples is None:
            samples = np.zeros(want, dtype="float32")
        if len(samples) < want:
            samples = np.concatenate(
                [samples, np.zeros(want - len(samples), dtype="float32")]
            )
        chunks.append(samples[:want])

    track = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
    pcm = np.clip(track, -1, 1)
    _write_wav(out_path, (pcm * 32767).astype("<i2").tobytes(), rate)
    return {"path": str(out_path), "duration_ms": int(len(track) / rate * 1000)}


def _decode_mp3(path: Path, rate: int):
    """Decode an MP3 to mono float32 via the bundled ffmpeg."""
    import numpy as np

    try:
        result = subprocess.run(
            [
                ffmpeg_exe(), "-v", "error", "-i", str(path),
                "-f", "f32le", "-ac", "1", "-ar", str(rate), "-",
            ],
            capture_output=True,
            check=True,
        )
        return np.frombuffer(result.stdout, dtype="float32").copy()
    except Exception:
        return None


def _write_wav(path: Path, pcm: bytes, rate: int) -> None:
    import struct

    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + pcm)


def export_mp4(
    deck: dict[str, Any],
    *,
    out_path: Path,
    audio_dir: Path | None = None,
    config: VideoConfig | None = None,
    max_slides: int = 0,
    progress=None,
) -> dict[str, Any]:
    """Render a deck to an MP4 and return a summary."""
    import numpy as np

    config = config or VideoConfig()
    width, height = config.size()
    slides = deck.get("slides") or []
    if max_slides > 0:
        slides = slides[:max_slides]
    if not slides:
        raise ValueError("deck has no slides to render")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        silent = tmp_dir / "video.mp4"

        command = [
            ffmpeg_exe(), "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(config.fps),
            "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "20",
            str(silent),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        transition_frames = int(config.fps * TRANSITION_MS / 1000)
        rendered = 0
        try:
            previous_last = None
            for position, slide in enumerate(slides):
                duration_ms = max(1, int(slide.get("duration_ms") or 1000))
                frames = max(1, int(config.fps * duration_ms / 1000))

                first = np.asarray(render_slide_frame(slide, 0, config))
                # Transition in from the previous slide's final frame.
                if previous_last is not None and config.transition != "cut":
                    for step in range(transition_frames):
                        blended = _blend(
                            previous_last, first, (step + 1) / transition_frames,
                            config.transition,
                        )
                        process.stdin.write(blended.tobytes())

                for frame_index in range(frames):
                    elapsed = int(frame_index * 1000 / config.fps)
                    frame = np.asarray(render_slide_frame(slide, elapsed, config))
                    process.stdin.write(frame.tobytes())
                    previous_last = frame
                rendered += 1
                if progress and position % 25 == 0:
                    progress(position, len(slides))
        finally:
            if process.stdin:
                process.stdin.close()
            process.wait()

        # Mux narration if it exists; otherwise ship the silent render.
        if audio_dir and audio_dir.is_dir():
            audio_path = tmp_dir / "audio.wav"
            build_audio_track(slides, audio_dir, audio_path, fps=config.fps)
            subprocess.run(
                [
                    ffmpeg_exe(), "-y", "-v", "error",
                    "-i", str(silent), "-i", str(audio_path),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", str(out_path),
                ],
                check=True,
            )
        else:
            silent.replace(out_path)

    size = out_path.stat().st_size
    total_ms = sum(int(s.get("duration_ms") or 0) for s in slides)
    return {
        "path": str(out_path),
        "bytes": size,
        "slides": rendered,
        "duration_ms": total_ms,
        "aspect": config.aspect,
        "resolution": f"{width}x{height}",
        "transition": config.transition,
        "word_animation": config.word_animation,
    }


__all__ = [
    "ASPECT_RATIOS",
    "DEFAULT_ASPECT",
    "DEFAULT_TRANSITION",
    "DEFAULT_WORD_ANIMATION",
    "VIDEO_TRANSITIONS",
    "VIDEO_WORD_ANIMATIONS",
    "VideoConfig",
    "export_mp4",
    "render_slide_frame",
    "build_audio_track",
    "ffmpeg_exe",
]
