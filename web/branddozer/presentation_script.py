"""branddozer/presentation_script.py — compress a paper into a spoken script.

Why this stage exists
---------------------
Reading a 12,428-word paper aloud produces an 81-minute video. Nobody
watches that. A presentation is a *different artefact* from the paper: it
has to carry the argument in about fifteen minutes and survive being
watched on a phone.

So before slides are built, the site's selected agent rewrites the paper
as a narration script under a hard word budget derived from measured
narration speed (~130 wpm on Polly neural, measured, not assumed). The
script is returned as structured sections so the existing chunker can turn
it into slides without re-parsing prose.

What the budget protects
------------------------
Compression is where a paper's honesty usually dies: nuance gets dropped,
hedges get flattened into assertions, and a negative result turns into a
positive-sounding summary. The prompt therefore treats qualifications as
*load-bearing* — a claim that cannot be stated with its uncertainty inside
the budget must be cut entirely rather than stated bare.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Measured on AWS Polly neural (Joanna): 5 words in 2.312s.
WORDS_PER_MINUTE = 130

DEFAULT_TARGET_MINUTES = 15
MIN_TARGET_MINUTES = 3
MAX_TARGET_MINUTES = 30


@dataclass
class ScriptRequest:
    title: str
    abstract: str
    markdown: str
    target_minutes: int = DEFAULT_TARGET_MINUTES
    findings_tone: str = "neutral"

    def word_budget(self) -> int:
        minutes = max(MIN_TARGET_MINUTES, min(MAX_TARGET_MINUTES, self.target_minutes))
        return int(minutes * WORDS_PER_MINUTE)


def script_prompt(request: ScriptRequest) -> str:
    budget = request.word_budget()
    return (
        "Rewrite this research paper as a spoken narration script for a "
        "video presentation. Return STRICT JSON only.\n\n"
        # A ceiling alone gets badly under-used: an early run returned 267
        # words against a 1,950 budget (2 minutes instead of 15). State it
        # as a target band with an explicit floor and a per-section quota.
        f"LENGTH TARGET: {budget} words total across all sections "
        f"(~{request.target_minutes} minutes at {WORDS_PER_MINUTE} words per "
        f"minute).\n"
        f"- Write between {int(budget * 0.85)} and {budget} words. Coming in "
        f"far under is as much a failure as going over: a {request.target_minutes}"
        "-minute slot filled with two minutes of narration is a broken "
        "deliverable.\n"
        f"- Produce 8 to 10 sections. Each section's `narration` must be at "
        f"least {int(budget * 0.85 / 10)} words — roughly a full spoken "
        "paragraph, not a sentence. Develop each point with its evidence "
        "and its limits rather than summarising it in one line.\n"
        "- Use the room to explain *why*, not to pad. If you find yourself "
        "short, you are omitting reasoning the viewer needs: what the "
        "evidence was, what was checked, what failed, and what follows.\n\n"
        f"TITLE: {request.title}\n"
        f"FINDINGS TONE: {request.findings_tone}\n\n"
        "RULES:\n"
        "- This is spoken, not read. Short sentences. No parentheticals, no "
        "citation markers like [@key], no URLs, no footnotes, no tables.\n"
        "- Qualifications are load-bearing. If a claim cannot be stated "
        "together with its uncertainty inside the budget, CUT THE CLAIM "
        "rather than state it bare. Never let compression turn a hedged or "
        "negative finding into a confident one.\n"
        "- Say plainly what was NOT established, and why. A negative or "
        "inconclusive result is the honest headline when that is the "
        "finding.\n"
        "- Lead with the question, then what the evidence permits, then what "
        "it does not, then what would settle it.\n"
        "- Every section needs a heading of at most 6 words.\n\n"
        "Return JSON with:\n"
        "  title: a short spoken title, at most 12 words\n"
        "  hook: one sentence that states the question\n"
        "  sections: list of {heading, narration} — narration is plain "
        "spoken prose\n"
        "  closing: one or two sentences on what remains unresolved\n"
        "  word_count: your own count of the total narration words\n\n"
        f"PAPER:\n{request.markdown[:90000]}"
    )


def _count(text: str) -> int:
    return len([w for w in str(text or "").split() if w.strip()])


def validate_script(script: dict[str, Any], request: ScriptRequest) -> dict[str, Any]:
    """Enforce the budget the model was asked to respect.

    The model's self-reported `word_count` is not trusted; sections are
    counted and dropped from the end until the script fits. Dropping whole
    sections (rather than truncating prose) keeps every surviving sentence
    intact, so a hedge is never cut off halfway.
    """
    budget = request.word_budget()
    sections: list[dict[str, str]] = []
    for section in script.get("sections") or []:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        narration = str(section.get("narration") or "").strip()
        if narration:
            sections.append({"heading": heading, "narration": narration})

    hook = str(script.get("hook") or "").strip()
    closing = str(script.get("closing") or "").strip()

    # Reserve room for the hook and closing so the script always lands.
    reserved = _count(hook) + _count(closing)
    running = reserved
    kept: list[dict[str, str]] = []
    dropped = 0
    for section in sections:
        cost = _count(section["narration"]) + _count(section["heading"])
        if running + cost > budget and kept:
            dropped += 1
            continue
        running += cost
        kept.append(section)

    return {
        "title": str(script.get("title") or request.title)[:200],
        "hook": hook,
        "sections": kept,
        "closing": closing,
        "word_count": running,
        "word_budget": budget,
        "target_minutes": request.target_minutes,
        "estimated_minutes": round(running / WORDS_PER_MINUTE, 1),
        "sections_dropped": dropped,
        "within_budget": running <= budget,
        # An under-run is a real defect, not a success: it means the video
        # is far shorter than the slot it was written for.
        "under_budget": running < budget * 0.85,
        "budget_use": round(running / budget, 2) if budget else 0.0,
    }


def script_to_markdown(script: dict[str, Any]) -> str:
    """Render a validated script as markdown the slide chunker understands."""
    parts: list[str] = [f"# {script.get('title') or ''}".strip()]
    hook = str(script.get("hook") or "").strip()
    if hook:
        parts.append(hook)
    for section in script.get("sections") or []:
        heading = str(section.get("heading") or "").strip()
        if heading:
            parts.append(f"## {heading}")
        narration = str(section.get("narration") or "").strip()
        if narration:
            parts.append(narration)
    closing = str(script.get("closing") or "").strip()
    if closing:
        parts.append("## In closing")
        parts.append(closing)
    return "\n\n".join(part for part in parts if part)


def write_script(request: ScriptRequest, *, agent_send) -> dict[str, Any]:
    """Ask the selected agent for a script, then clamp it to the budget."""
    raw = agent_send(
        script_prompt(request),
        system=(
            "You are a science communicator writing narration for a video. "
            "Return strict JSON only. Never overstate a hedged finding."
        ),
    )
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("script writer returned no JSON object")
    return validate_script(json.loads(raw[start:end]), request)


__all__ = [
    "WORDS_PER_MINUTE",
    "DEFAULT_TARGET_MINUTES",
    "ScriptRequest",
    "script_prompt",
    "validate_script",
    "script_to_markdown",
    "write_script",
]
