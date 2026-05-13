"""wizard_chat/conversation.py — rolling conversation context + summary.

Per-session memory for wizard-chat.  Each session_id keeps an ordered
turn log; on every new user message we:

  1. Append the turn
  2. Build a context blob the backend prepends to the next prompt:
       • a one-line "active topic" (what problem we're working on)
       • the rolling summary of older turns
       • the verbatim last-N turns
  3. After a configurable threshold of turns, summarise the
     oldest-falling-out-of-window turns and roll them into the summary
     string so the prompt size stays bounded

The summariser is **pluggable**.  The default is a self-contained
heuristic that extracts the first declarative sentence of each
message — works without any external dependency.  Operators can swap
in `OpenAISummariser` (requires OPENAI_API_KEY) or
`WizardNodeSummariser` (requires the local node to recall well enough)
without changing call sites.

Topic tracking is similarly heuristic: noun phrases that appear
across multiple recent turns become the "active topic," and a
user-initiated switch ("now let's talk about X") is detected by
imperative-frame patterns.  Good enough for the rolling context to
feel coherent; refinable later.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Protocol


# ── Configuration ──────────────────────────────────────────────────────────

# Hold the last N turns verbatim.  Anything older is rolled into the
# running summary on the next message.
DEFAULT_VERBATIM_WINDOW = 8

# Maximum total characters of context we'll prepend to the backend
# prompt.  Truncates from the OLDEST end of the verbatim window
# if needed; the summary stays.
DEFAULT_MAX_CONTEXT_CHARS = 4000

# Idle TTL for sessions (seconds).  Cleared on next access if expired.
SESSION_IDLE_TTL_SECS = 6 * 3600


# ── Data ────────────────────────────────────────────────────────────────────


class Turn:
    __slots__ = ("role", "text", "ts")

    def __init__(self, role: str, text: str, ts: float | None = None):
        self.role = role           # "user" or "wizard"
        self.text = text.strip()
        self.ts   = ts if ts is not None else time.time()


class ConversationState:
    """Live state for one session — held in-process under a lock."""
    __slots__ = ("session_id", "turns", "summary", "active_topic", "last_touch")

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.turns: list[Turn] = []
        self.summary      = ""           # running summary of OLDER turns
        self.active_topic = ""           # current problem/topic line
        self.last_touch   = time.time()


# ── Summariser interface + heuristic default ───────────────────────────────


class Summariser(Protocol):
    def summarise(self, prior_summary: str, dropped_turns: list[Turn]) -> str: ...


class HeuristicSummariser:
    """Standalone summariser — no external API.  Extracts the first
    declarative sentence of each dropped turn, prepends a role marker,
    and concatenates onto the prior running summary.  Trims to a soft
    cap.  Reliable; not eloquent."""

    MAX_SUMMARY_CHARS = 1200

    @staticmethod
    def _first_sentence(text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return ""
        # Split on ., !, ? followed by space or end-of-string.
        m = re.match(r"^(.+?[\.\!\?])(?:\s|$)", text)
        s = m.group(1) if m else text
        # Cap individual sentences so a giant paragraph doesn't blow
        # the summary.
        return s[:240]

    def summarise(self, prior_summary: str, dropped_turns: list[Turn]) -> str:
        new_lines = []
        for t in dropped_turns:
            tag = "you asked" if t.role == "user" else "I said"
            sent = self._first_sentence(t.text)
            if sent:
                new_lines.append(f"{tag}: {sent}")
        appended = "\n".join(new_lines)
        full = (prior_summary + ("\n" if prior_summary and appended else "") + appended).strip()
        if len(full) > self.MAX_SUMMARY_CHARS:
            # Keep the tail (most recent gets best fidelity).
            full = "...\n" + full[-self.MAX_SUMMARY_CHARS:]
        return full


# Optional OpenAI summariser — instantiates only if OPENAI_API_KEY is
# present.  Import is lazy so module load doesn't pull openai when
# not used.
class OpenAISummariser:
    MODEL = os.getenv("WIZARD_CHAT_SUMMARY_MODEL", "gpt-4o-mini")

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI()  # picks up OPENAI_API_KEY from env

    def summarise(self, prior_summary: str, dropped_turns: list[Turn]) -> str:
        dropped_lines = []
        for t in dropped_turns:
            tag = "USER" if t.role == "user" else "ASSISTANT"
            dropped_lines.append(f"{tag}: {t.text}")
        prompt = (
            "You are maintaining a rolling summary of a conversation between a "
            "user and an AI named W1z4rD.  The user is testing the system and "
            "iterating on it.  Below is the prior summary followed by new "
            "turns that need to be folded in.  Update the summary to be "
            "concise (under 400 words), prioritising:\n"
            "  - the active problem or topic being worked on\n"
            "  - decisions or conclusions reached\n"
            "  - open questions or pending work\n"
            "Drop pleasantries and verbatim chatter.\n\n"
            f"PRIOR SUMMARY:\n{prior_summary or '(none yet)'}\n\n"
            f"NEW TURNS:\n" + "\n".join(dropped_lines)
        )
        try:
            resp = self._client.chat.completions.create(
                model=self.MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.2,
                timeout=20,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            # Fall through to heuristic so the chat keeps working.
            return HeuristicSummariser().summarise(prior_summary, dropped_turns)


def _make_default_summariser() -> Summariser:
    if os.getenv("OPENAI_API_KEY") and os.getenv("WIZARD_CHAT_USE_OPENAI_SUMMARY") == "1":
        try:
            return OpenAISummariser()
        except Exception:
            pass
    return HeuristicSummariser()


# ── Topic tracking ─────────────────────────────────────────────────────────


# Very common stop-words / function words that shouldn't anchor a topic.
_TOPIC_STOPWORDS = {
    "the","a","an","is","are","was","were","be","being","been","do","does",
    "did","have","has","had","what","why","how","when","where","who","which",
    "this","that","these","those","i","you","we","they","it","my","your","our",
    "their","me","us","them","to","of","in","on","at","for","with","about",
    "and","or","but","not","no","yes","please","can","could","would","should",
    "going","make","made","just","like","want","need","get","got",
}

_NEW_TOPIC_PATTERNS = [
    re.compile(r"\b(?:now|let'?s|let us|switch(?:ing)? to|change topic|new topic)\b", re.I),
    re.compile(r"\b(?:talk about|tell me about|explain|teach me)\s+(?:the )?([a-z][\w\- ]{2,40})", re.I),
]


def _content_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    return [w for w in words if w not in _TOPIC_STOPWORDS and len(w) >= 3]


def update_topic(prev_topic: str, new_user_msg: str, recent_user_msgs: list[str]) -> str:
    """Recompute the active topic.  Cheap heuristic:
    - explicit switch keyword → carve out the suffix as the new topic
    - else find the noun-like words present in the new msg AND in at
      least one of the recent prior user msgs → those become the topic
    - else keep prev_topic
    """
    # Explicit switch.
    for pat in _NEW_TOPIC_PATTERNS:
        m = pat.search(new_user_msg)
        if m and m.lastindex:
            candidate = m.group(m.lastindex).strip().rstrip("?.!")
            if len(candidate) >= 3:
                return candidate[:80]
    # Content-word overlap with recent turns.
    new_words = set(_content_words(new_user_msg))
    if not new_words:
        return prev_topic
    overlap: dict[str, int] = {}
    for prior in recent_user_msgs[-4:]:
        prior_words = set(_content_words(prior))
        for w in new_words & prior_words:
            overlap[w] = overlap.get(w, 0) + 1
    if overlap:
        # Pick the top-3 most-shared content words.
        ranked = sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0]))
        return " ".join(w for w, _ in ranked[:3])
    return prev_topic


# ── Session store ──────────────────────────────────────────────────────────


class ConversationStore:
    """Process-local; cleared on Django restart.  For multi-worker
    deployments use a Redis-backed variant — interface is identical."""

    def __init__(self):
        self._sessions: dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        self._summariser_factory: Callable[[], Summariser] = _make_default_summariser

    def get_or_create(self, session_id: str) -> ConversationState:
        now = time.time()
        with self._lock:
            # Evict stale sessions opportunistically.
            stale = [sid for sid, s in self._sessions.items()
                     if now - s.last_touch > SESSION_IDLE_TTL_SECS]
            for sid in stale:
                self._sessions.pop(sid, None)
            s = self._sessions.get(session_id)
            if s is None:
                s = ConversationState(session_id)
                self._sessions[session_id] = s
            s.last_touch = now
            return s

    def append_turn(self, session_id: str, role: str, text: str) -> ConversationState:
        s = self.get_or_create(session_id)
        s.turns.append(Turn(role, text))
        if role == "user":
            # Update topic on every user message — that's when the
            # conversation direction is set.
            user_msgs = [t.text for t in s.turns if t.role == "user"]
            s.active_topic = update_topic(s.active_topic, text, user_msgs[:-1])
        # Roll oldest verbatim turns into the summary if we've exceeded
        # the window.
        if len(s.turns) > DEFAULT_VERBATIM_WINDOW:
            drop_count = len(s.turns) - DEFAULT_VERBATIM_WINDOW
            dropped = s.turns[:drop_count]
            s.turns = s.turns[drop_count:]
            summariser = self._summariser_factory()
            s.summary = summariser.summarise(s.summary, dropped)
        return s

    def build_context_blob(self, session_id: str,
                            max_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> str:
        """Render the context the backend should see along with the
        next user message.  Returned string is empty if there's no
        meaningful history."""
        s = self.get_or_create(session_id)
        if not s.turns and not s.summary:
            return ""
        lines: list[str] = []
        if s.active_topic:
            lines.append(f"[Active topic: {s.active_topic}]")
        if s.summary:
            lines.append(f"[Earlier in this conversation]\n{s.summary}")
        if s.turns:
            lines.append("[Recent turns]")
            for t in s.turns:
                tag = "User" if t.role == "user" else "W1z4rD"
                lines.append(f"{tag}: {t.text}")
        blob = "\n".join(lines)
        # Hard cap: if over budget, drop oldest verbatim turns first.
        while len(blob) > max_chars and s.turns:
            # Drop one turn from the head of recent and rebuild.
            s.turns.pop(0)
            lines = []
            if s.active_topic:
                lines.append(f"[Active topic: {s.active_topic}]")
            if s.summary:
                lines.append(f"[Earlier in this conversation]\n{s.summary}")
            if s.turns:
                lines.append("[Recent turns]")
                for t in s.turns:
                    tag = "User" if t.role == "user" else "W1z4rD"
                    lines.append(f"{tag}: {t.text}")
            blob = "\n".join(lines)
        return blob

    def snapshot(self, session_id: str) -> dict:
        s = self.get_or_create(session_id)
        return {
            "session_id":   session_id,
            "active_topic": s.active_topic,
            "summary":      s.summary,
            "turn_count":   len(s.turns),
            "recent_turns": [{"role": t.role, "text": t.text} for t in s.turns],
        }


# Module-level singleton — Django views import this directly.
STORE = ConversationStore()
