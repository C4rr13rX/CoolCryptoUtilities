"""branddozer/presentation.py — the adjacent, presentation-ready view of a paper.

Why this exists
---------------
A validated research paper is one long markdown document: correct for
publication, unreadable on a phone. This module emits an *adjacent*
schema — the same content, restructured into bite-sized slides that carry
their own semantics (title / heading / body / quote / citation), sized so
each one renders in large type, centred, on a mobile screen.

The schema is deliberately renderer-agnostic. It is consumed by the
in-browser player, and the same timeline can later drive a rendered video
without regenerating anything.

Design decisions worth knowing
------------------------------
* **Sentence-level slides, not word-level.** One word per slide would make
  a 12,000-word paper into 12,000 slides and destroy sentence context. A
  slide holds one sentence (split further when it exceeds the mobile
  budget), and *words carry their own timings* — so a renderer can still
  reveal or highlight word by word while the reader keeps the sentence.

* **Timings come from AWS Polly speech marks, not estimates.** Polly
  returns a millisecond offset per spoken word, so word highlighting is
  exact rather than interpolated. Slides carry `audio_ms` only once
  narration has been synthesised; before that they are `None` and a
  renderer falls back to reading-speed pacing.

* **Character budget, not word count.** Legibility on a phone is bounded
  by characters at a given font size, not by words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"

# Mobile legibility budget. At ~34px on a 390pt-wide viewport roughly 90
# characters still fit centred without shrinking the type. Slides above
# this are split at clause boundaries.
MAX_SLIDE_CHARS = 90
# Below this a fragment is merged into its neighbour rather than flashing
# past as its own slide.
MIN_SLIDE_CHARS = 18

# Reading pace used only when narration has not been generated yet.
FALLBACK_MS_PER_WORD = 380
MIN_SLIDE_MS = 1200

SLIDE_KINDS = (
    "title",
    "subtitle",
    "heading",
    "body",
    "quote",
    "citation",
    "list_item",
)


def _norm(text: str) -> str:
    """Loose comparison key for detecting a repeated title."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking common abbreviations."""
    protected = text
    for abbr in ("e.g.", "i.e.", "cf.", "et al.", "vs.", "Dr.", "Prof.", "No."):
        protected = protected.replace(abbr, abbr.replace(".", "\x00"))
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def _split_long(sentence: str, limit: int = MAX_SLIDE_CHARS) -> list[str]:
    """Break an over-long sentence at clause boundaries, then at words.

    Clause breaks are preferred because they land where a speaker would
    pause anyway, so the slide change matches the narration's rhythm.
    """
    if len(sentence) <= limit:
        return [sentence]
    chunks: list[str] = []
    for clause in re.split(r"(?<=[,;:—])\s+", sentence):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= limit:
            chunks.append(clause)
            continue
        # A clause can still exceed the budget (long parentheticals, lists
        # of source keys). Fall through to a word-level pack so no slide
        # ever ships over budget.
        current = ""
        for word in clause.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > limit and current:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
    # Re-merge fragments too short to deserve their own slide — but only
    # when the merge stays inside the budget, or we would reintroduce the
    # over-long slides this function exists to prevent.
    merged: list[str] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk) < MIN_SLIDE_CHARS
            and len(merged[-1]) + len(chunk) + 1 <= limit
        ):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


@dataclass
class Slide:
    """One bite-sized unit of the presentation."""

    index: int
    kind: str
    text: str
    section: str = ""
    # Word-level timing, filled in by narration. Each entry is
    # {"word": str, "start_ms": int, "end_ms": int}.
    words: list[dict[str, Any]] = field(default_factory=list)
    audio_ms: int | None = None
    audio_url: str = ""
    background_url: str = ""
    notes: str = ""

    def estimated_ms(self) -> int:
        """Duration to use when narration has not been generated."""
        if self.audio_ms:
            return self.audio_ms
        count = max(1, len(self.text.split()))
        return max(MIN_SLIDE_MS, count * FALLBACK_MS_PER_WORD)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "text": self.text,
            "section": self.section,
            "words": self.words,
            "audio_ms": self.audio_ms,
            "duration_ms": self.estimated_ms(),
            "audio_url": self.audio_url,
            "background_url": self.background_url,
            "notes": self.notes,
        }


def _classify_line(line: str) -> tuple[str, str]:
    """Return (kind, cleaned_text) for one markdown line."""
    stripped = line.strip()
    if stripped.startswith("#"):
        depth = len(stripped) - len(stripped.lstrip("#"))
        text = stripped.lstrip("#").strip()
        if depth == 1:
            return "title", text
        if depth == 2:
            return "heading", text
        return "subtitle", text
    if stripped.startswith(">"):
        return "quote", stripped.lstrip(">").strip()
    if re.match(r"^([-*+]|\d+\.)\s+", stripped):
        return "list_item", re.sub(r"^([-*+]|\d+\.)\s+", "", stripped)
    return "body", stripped


def _strip_markup(text: str) -> str:
    """Remove markdown emphasis so slides display clean prose."""
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"(\*\*|__)(.*?)\1", r"\2", out)
    out = re.sub(r"(\*|_)(.*?)\1", r"\2", out)
    out = re.sub(r"`([^`]*)`", r"\1", out)
    return out.strip()


def build_slides(markdown: str, *, title: str = "") -> list[Slide]:
    """Chunk a paper's markdown into presentation slides."""
    slides: list[Slide] = []
    section = ""
    index = 0

    if title:
        # Research titles are routinely 150+ characters, so the opening
        # slide needs the same budget treatment as everything else.
        for part in _split_long(_strip_markup(title)):
            slides.append(Slide(index=index, kind="title", text=part))
            index += 1

    in_code = False
    for raw_line in (markdown or "").splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line.strip():
            continue

        kind, text = _classify_line(line)
        text = _strip_markup(text)
        if not text:
            continue

        if kind in {"title", "heading", "subtitle"}:
            section = text
            # A paper's H1 repeats the title we already showed; showing it
            # twice opens the deck on a duplicate. Compare against the whole
            # opening title run, since a long title is itself split across
            # several slides.
            if kind == "title" and slides and slides[0].kind == "title":
                opening = " ".join(
                    slide.text for slide in slides if slide.kind == "title"
                )
                if _norm(text) == _norm(opening):
                    continue
            # Headings obey the same mobile budget as prose — an unsplit
            # title was the single widest slide in the deck.
            for part in _split_long(text):
                slides.append(Slide(index=index, kind=kind, text=part, section=section))
                index += 1
            continue

        # A bare URL is a reference, not prose: it must not be split across
        # slides (meaningless) nor read out word by word. It is shown whole
        # and marked so the renderer can shrink the type to fit.
        if re.fullmatch(r"<?https?://\S+>?", text):
            slides.append(
                Slide(
                    index=index,
                    kind="citation",
                    text=text.strip("<>"),
                    section=section,
                    notes="url",
                )
            )
            index += 1
            continue

        # Citation-only lines read as references, not prose.
        if re.fullmatch(r"[\[\(]?@?[\w.-]+[\]\)]?[.,;]?", text):
            slides.append(
                Slide(index=index, kind="citation", text=text, section=section)
            )
            index += 1
            continue

        for sentence in _sentences(text):
            for chunk in _split_long(sentence):
                slides.append(
                    Slide(index=index, kind=kind, text=chunk, section=section)
                )
                index += 1

    return slides


def build_presentation(
    *,
    paper_id: str,
    title: str,
    markdown: str,
    abstract: str = "",
    theme: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit the full adjacent presentation document for a paper."""
    slides = build_slides(markdown, title=title)
    total_ms = sum(slide.estimated_ms() for slide in slides)
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "theme": theme or {},
        "slide_count": len(slides),
        "estimated_duration_ms": total_ms,
        "narrated": any(slide.audio_ms for slide in slides),
        "slides": [slide.to_dict() for slide in slides],
    }


def attach_word_timings(
    slide: dict[str, Any], marks: Iterable[dict[str, Any]], *, audio_ms: int
) -> dict[str, Any]:
    """Attach Polly speech marks to a slide.

    Polly gives the *start* of each word only, so each word's end is the
    next word's start (and the last word runs to the clip end). Without
    this a renderer cannot know how long to hold a highlight.
    """
    ordered = sorted(
        (m for m in marks if m.get("type") == "word"),
        key=lambda m: int(m.get("time") or 0),
    )
    words: list[dict[str, Any]] = []
    for position, mark in enumerate(ordered):
        start = int(mark.get("time") or 0)
        if position + 1 < len(ordered):
            end = int(ordered[position + 1].get("time") or start)
        else:
            end = audio_ms
        words.append(
            {"word": str(mark.get("value") or ""), "start_ms": start, "end_ms": end}
        )
    slide["words"] = words
    slide["audio_ms"] = audio_ms
    slide["duration_ms"] = audio_ms
    return slide


__all__ = [
    "SCHEMA_VERSION",
    "MAX_SLIDE_CHARS",
    "SLIDE_KINDS",
    "Slide",
    "build_slides",
    "build_presentation",
    "attach_word_timings",
]
