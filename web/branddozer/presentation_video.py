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
import shutil
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
    """
    Group word indices into lines that fit `max_width`.

    A word that is wider than `max_width` on its own gets a line to itself
    rather than being merged into a neighbour: it will still overflow, but by
    less, and the caller shrinks the font until it does not. Silently packing
    it next to another word is what pushed text off the frame edge.
    """
    lines: list[list[int]] = []
    current: list[int] = []
    for i, word in enumerate(words):
        word_width = draw.textlength(word, font=font)
        if word_width > max_width and current:
            # Too wide even alone -- do not compound it by appending.
            lines.append(current)
            lines.append([i])
            current = []
            continue
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


def _fit_font(draw, words: list[str], size: int, max_width: int,
              max_height: int, line_ratio: float = 1.3):
    """
    Shrink the font until the wrapped text fits the safe area.

    Without this a long word or a dense slide simply overflowed the frame.
    Ten steps at 6% each covers a ~45% reduction, which is enough for the
    longest headings these decks produce while staying readable.
    """
    for _ in range(10):
        font = _font(size)
        lines = _wrap(draw, words, font, max_width)
        widest = max(
            (draw.textlength(" ".join(words[j] for j in line), font=font)
             for line in lines),
            default=0,
        )
        if widest <= max_width and len(lines) * size * line_ratio <= max_height:
            return font, lines, size
        size = max(12, int(size * 0.94))
    # Floor reached: return the smallest attempt rather than looping forever.
    font = _font(size)
    return font, _wrap(draw, words, font, max_width), size


_BACKGROUND_CACHE: dict[str, Any] = {}


def _load_background(path: str, size: tuple[int, int]):
    """Load, crop-to-fill, blur and dim a background image.

    Cached: a section's slides share one image, and re-decoding a 1.5MB
    PNG for every one of 30 frames per second would dominate render time.
    """
    key = f"{path}|{size[0]}x{size[1]}"
    if key in _BACKGROUND_CACHE:
        return _BACKGROUND_CACHE[key]
    try:
        from PIL import Image, ImageEnhance, ImageFilter

        source = Image.open(path).convert("RGB")
        target_ratio = size[0] / size[1]
        ratio = source.width / source.height
        # Crop to fill rather than stretch: a squashed background reads as
        # a mistake even when the type on top is correct.
        if ratio > target_ratio:
            new_width = int(source.height * target_ratio)
            left = (source.width - new_width) // 2
            source = source.crop((left, 0, left + new_width, source.height))
        else:
            new_height = int(source.width / target_ratio)
            top = (source.height - new_height) // 2
            source = source.crop((0, top, source.width, top + new_height))
        source = source.resize(size, Image.LANCZOS)
        source = source.filter(ImageFilter.GaussianBlur(radius=size[0] / 180))
        # Dim hard: the text is the content, the artwork is atmosphere.
        source = ImageEnhance.Brightness(source).enhance(0.42)
        _BACKGROUND_CACHE[key] = source
        return source
    except Exception:
        _BACKGROUND_CACHE[key] = None
        return None


def render_slide_frame(
    slide: dict[str, Any], elapsed_ms: int, config: VideoConfig
):
    """Compose one frame of a slide at `elapsed_ms` into its narration."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

    width, height = config.size()
    image = Image.new("RGB", (width, height), _hex(config.background))

    # Section artwork, dimmed and blurred so large type stays legible on
    # top of it. Cached per path because consecutive slides in a section
    # share one background.
    art_path = slide.get("background_path") or ""
    if art_path:
        art = _load_background(art_path, (width, height))
        if art is not None:
            image.paste(art, (0, 0))

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

    # 12% rather than 8%: the cube flip warps the frame inward, so text laid
    # out to the old margin was sliced at the panel edge mid-transition.
    # This is the "title safe" area broadcast has used for the same reason.
    margin = int(width * 0.12)
    max_text_width = width - margin * 2
    max_text_height = int(height * 0.72)

    # Shrink to fit rather than overflow. A single long word (these decks are
    # academic titles) would otherwise run off both sides.
    font, lines, size = _fit_font(
        draw, words, size, max_text_width, max_text_height
    )
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
    slides: list[dict[str, Any]],
    audio_dir: Path,
    out_path: Path,
    *,
    # Needed to quantise the inter-slide silence to whole video frames; a
    # mismatch here is what put the narration out of sync with the words.
    fps: int = FPS,
    transition_ms: int = 0,
    score_path: str = "",
    score_gain: float = 0.16,
) -> dict[str, Any]:
    """Concatenate per-slide narration, padding to each slide's duration.

    `transition_ms` must match what the video renderer inserts between
    slides. Omitting it desynchronises progressively: at 420ms across 153
    transitions the narration ended 64 seconds behind the picture.
    """
    import numpy as np

    rate = 44100
    chunks: list[Any] = []
    for position, slide in enumerate(slides):
        duration_ms = int(slide.get("duration_ms") or 0)
        # Silence covering the transition that precedes this slide.
        #
        # Quantised to whole frames, because that is what the video writes:
        # int(fps * 420/1000) = 12 frames = 400ms, not 420ms. Using the
        # nominal duration here instead makes every slide drift 20ms and the
        # error accumulates across the deck.
        if position > 0 and transition_ms > 0:
            transition_frames = int(fps * transition_ms / 1000)
            silence_ms = transition_frames * 1000 / fps
            chunks.append(np.zeros(int(rate * silence_ms / 1000), dtype="float32"))
        want = int(rate * duration_ms / 1000)
        clip = audio_dir / f"slide-{int(slide['index']):05d}.mp3"
        samples = None
        if clip.is_file():
            samples = _decode_audio(clip, rate)
        if samples is None:
            samples = np.zeros(want, dtype="float32")
        if len(samples) < want:
            samples = np.concatenate(
                [samples, np.zeros(want - len(samples), dtype="float32")]
            )
        chunks.append(samples[:want])

    track = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")

    # Mix the score underneath at low gain. Narration intelligibility wins
    # every time: the music is atmosphere, not a duet.
    if score_path and Path(score_path).is_file():
        music = _decode_audio(Path(score_path), rate)
        if music is not None and len(music) > 0:
            if len(music) < len(track):
                # Loop the cue to cover the full deck.
                repeats = int(np.ceil(len(track) / len(music)))
                music = np.tile(music, repeats)
            music = music[: len(track)] * score_gain
            # Short fades so the loop seam and the ending are not abrupt.
            fade = min(rate * 2, len(music) // 4)
            if fade > 0:
                music[:fade] *= np.linspace(0, 1, fade, dtype="float32")
                music[-fade:] *= np.linspace(1, 0, fade, dtype="float32")
            track = track + music

    pcm = np.clip(track, -1, 1)
    _write_wav(out_path, (pcm * 32767).astype("<i2").tobytes(), rate)
    return {"path": str(out_path), "duration_ms": int(len(track) / rate * 1000)}


def _decode_audio(path: Path, rate: int):
    """Decode any audio file to mono float32 via the bundled ffmpeg."""
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


def attach_backgrounds(
    deck: dict[str, Any],
    *,
    out_dir: Path,
    palette: dict[str, Any] | None = None,
    api_key: str = "",
    max_images: int = 12,
) -> dict[str, Any]:
    """Generate one background per section and attach it to that section's slides.

    Per *section*, not per slide: 154 slides would mean 154 image calls,
    and the artwork is atmosphere that should persist while a section is
    being read rather than flickering on every sentence.
    """
    from branddozer.presentation_media import background_prompt, generate_background

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = palette or {"scheme": "achromatic_accent", "mood": ["sober", "evidential"]}

    sections: list[str] = []
    for slide in deck.get("slides") or []:
        name = str(slide.get("section") or "").strip() or "opening"
        if name not in sections:
            sections.append(name)

    generated: dict[str, str] = {}
    failures: list[dict[str, str]] = []
    for index, name in enumerate(sections[:max_images]):
        sample = next(
            (
                s["text"]
                for s in deck["slides"]
                if (s.get("section") or "opening") == name and s.get("kind") == "body"
            ),
            name,
        )
        path = generate_background(
            background_prompt(sample, name, palette),
            out_dir=out_dir,
            name=f"bg-{index:02d}",
            api_key=api_key,
        )
        if path:
            generated[name] = path
        else:
            failures.append({"section": name, "error": "generation failed"})

    for slide in deck.get("slides") or []:
        name = str(slide.get("section") or "").strip() or "opening"
        if name in generated:
            slide["background_path"] = generated[name]

    return {
        "sections": len(sections),
        "generated": len(generated),
        "failures": failures,
    }


def export_mp4(
    deck: dict[str, Any],
    *,
    out_path: Path,
    audio_dir: Path | None = None,
    config: VideoConfig | None = None,
    max_slides: int = 0,
    score_path: str = "",
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
            # Square pixels + an explicit display aspect ratio: without
            # these some players infer landscape from the codec defaults
            # and letterbox a portrait video sideways.
            "-aspect", f"{width}:{height}",
            "-vf", "setsar=1:1",
            str(silent),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)

        transition_frames = int(config.fps * TRANSITION_MS / 1000)
        rendered = 0
        try:
            previous_last = None
            for position, slide in enumerate(slides):
                duration_ms = max(1, int(slide.get("duration_ms") or 1000))
                # round(), not int(): truncating loses up to a frame per slide
                # and the narration slides progressively later against the
                # words. build_audio_track pads each slide to the same length.
                frames = max(1, round(config.fps * duration_ms / 1000))

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
            # Must match the frames actually written below, or narration
            # drifts behind the picture by one transition per slide.
            build_audio_track(
                slides,
                audio_dir,
                audio_path,
                fps=config.fps,
                transition_ms=0 if config.transition == "cut" else TRANSITION_MS,
                score_path=score_path,
            )
            subprocess.run(
                [
                    ffmpeg_exe(), "-y", "-v", "error",
                    "-i", str(silent), "-i", str(audio_path),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    # Move the index to the front so a phone can start
                    # playing before the whole file has downloaded.
                    "-movflags", "+faststart",
                    "-shortest", str(out_path),
                ],
                check=True,
            )
        else:
            # shutil.move, not Path.replace: the temp dir and the output often
            # live on different drives (TEMP on C:, the project on D:), and
            # os.replace cannot cross a filesystem -- it fails with
            # "[WinError 17] The system cannot move the file to a different
            # disk drive" after the whole render has already succeeded.
            shutil.move(str(silent), str(out_path))

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
    "TRANSITION_MS",
    "attach_backgrounds",
    "export_mp4",
    "render_slide_frame",
    "build_audio_track",
    "ffmpeg_exe",
]
