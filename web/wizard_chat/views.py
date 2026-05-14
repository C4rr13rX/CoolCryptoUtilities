"""wizard_chat/views.py — API backend for the W1z4rD Vision multimodal chat."""
from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import re
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

WIZARD_ENDPOINT = os.getenv("WIZARD_NODE_URL", "http://localhost:8090").rstrip("/")
WIZARD_TIMEOUT = float(os.getenv("WIZARD_TIMEOUT_S", "8"))
WEB_SEARCH_TIMEOUT = float(os.getenv("WIZARD_WEB_SEARCH_TIMEOUT_S", "3"))
MAX_UPLOAD_MB = int(os.getenv("WIZARD_CHAT_MAX_MB", "50"))

# Human corrections use higher LR + multiple passes so the Hebbian association
# actually dominates over the background noise from general training.
TRAIN_REPEATS = 5    # sequential STDP passes per correction
TRAIN_LR      = 0.40 # vs 0.14 for background; corrections are deliberate
TRAIN_SURPRISE = 0.5  # non-zero → ACh/NE neuromodulator gate amplifies LTP


# ── /sensor/observe routing for uploaded files ──────────────────────────────


def _sensor_kind_for_file(
    name: str, content_type: str, raw: bytes, extracted_text: str,
) -> tuple[str | None, dict]:
    """Decide which /sensor/observe `kind` to send a file under, and
    build the body dict.  Returns (kind, body) or (None, {}) to skip.

    Mapping:
      image/*  → kind=image,   bytes_b64=<raw>
      audio/*  → kind=audio,   bytes_b64=<raw>   (encoder expects WAV)
      pdf/*    → kind=pdf_text, text=<extracted_text>
      text/*   → kind=text,    text=<extracted_text>
      else     → None (skip — node has no encoder for it)
    """
    ct = (content_type or "").lower()
    lname = name.lower()
    if ct.startswith("image/") or lname.endswith((".jpg", ".jpeg", ".png",
                                                   ".bmp", ".gif", ".webp")):
        return "image", {"bytes_b64": base64.b64encode(raw).decode("ascii")}
    if ct.startswith("audio/") or lname.endswith((".wav",)):
        return "audio", {"bytes_b64": base64.b64encode(raw).decode("ascii")}
    if ct == "application/pdf" or lname.endswith(".pdf"):
        if extracted_text:
            return "pdf_text", {"text": extracted_text}
        return None, {}
    if ct.startswith("text/") or lname.endswith((".txt", ".md", ".csv", ".log")):
        if extracted_text:
            return "text", {"text": extracted_text}
        return None, {}
    return None, {}


def _post_sensor_observe(kind: str, body: dict, timeout: float = 15.0
                          ) -> tuple[bool, str]:
    """POST one observation to the node.  Returns (ok, error_msg)."""
    body = {"kind": kind, **body}
    try:
        raw = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{WIZARD_ENDPOINT}/sensor/observe",
            data=raw, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True, ""
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wizard_ask(text: str, session_id: str = "") -> dict:
    """POST to /chat; always returns a normalised dict.

    /chat routes through multi_pool first (where Phase 29 concept bindings
    live) and falls back to the slow-pool char_chain decoder when
    multi_pool confidence is below threshold.  Previously this called
    /neuro/ask which is slow-pool-only and never benefits from multi_pool
    even after the curriculum's concept-binding phase finishes.
    """
    payload = json.dumps({
        "text": text,
        "session_id": session_id or str(uuid.uuid4()),
        "hops": 2,
        "min_strength": 0.05,
    }).encode()
    req = urllib.request.Request(
        f"{WIZARD_ENDPOINT}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=WIZARD_TIMEOUT) as resp:
            data = json.loads(resp.read())
        # /chat / /neuro/ask shape normalisation for the downstream display.
        if "activated_concepts" not in data:
            wa = data.get("word_activations") or []
            data["activated_concepts"] = [
                f"txt:word_{w['word']}" for w in wa if isinstance(w, dict) and w.get("word")
            ]
        if "answer" not in data:
            data["answer"] = ""
        if data.get("answer") is None:
            data["answer"] = ""
        # /chat reports `decoder` = "multi_pool" when the concept-binding
        # path was confident enough, "char_chain" when it fell through to
        # the slow pool, or "eem" when the equation matrix supplied the
        # answer.  Map that to the legacy hypothesis/tier flags the chat
        # UI already renders so users see the same UNCERTAIN badge for
        # char_chain fallbacks as before.
        decoder = data.get("decoder")
        if decoder and "hypothesis" not in data:
            data["hypothesis"] = decoder == "char_chain"
        if decoder and "confidence_tier" not in data:
            data["confidence_tier"] = {
                "multi_pool": "high",
                "eem":        "medium",
                "char_chain": "low",
            }.get(decoder, "low")
        return data
    except urllib.error.URLError as exc:
        return {"error": str(exc), "answer": "", "hypothesis": True,
                "confidence_tier": "offline", "activated_concepts": []}
    except Exception as exc:
        return {"error": str(exc), "answer": "", "hypothesis": True,
                "confidence_tier": "error", "activated_concepts": []}


def _wizard_train(question: str, answer: str, session_id: str = "") -> dict:
    """
    Train the Hebbian graph with a Q->A human correction.

    Returns a structured progress log with per-step status + final
    recall verification so the UI can render a progress bar AND tell
    the user whether the chat will actually return the trained answer
    next time they ask.

    Pipeline (every step exercises a different architectural layer):
      1. /multi_pool/train_pair   high-confidence concept binding
                                  (the path the /chat endpoint routes
                                  through first — this is what makes
                                  a single correction immediately
                                  recallable)
      2. /media/train_sequence ×N slow-pool character-bigram
                                  reinforcement — feeds the motif
                                  runtime so the pattern enters the
                                  global Hebbian graph too
      3. /neuro/reinforce         dopamine flush + three-factor LTP
                                  capture; consolidates the trace
                                  tags left by step 2
      4. /knowledge/ingest        structured-knowledge store entry
                                  for the QA fast-path
      5. verify via /chat         immediately query the question and
                                  return the actual answer so the
                                  caller knows whether the training
                                  took
    """
    import base64

    def _text_b64(s: str) -> str:
        return base64.b64encode(s.encode("utf-8")).decode("ascii")

    def _spans(text: str, role: str, y: float, idx: int, total: int) -> list:
        return [{"text": text, "role": role, "bold": False, "italic": False,
                 "size_ratio": 1.0, "x_frac": 0.5, "y_frac": y,
                 "seq_index": idx, "seq_total": total}]

    def _post(path: str, payload: bytes, timeout: float) -> dict:
        req = urllib.request.Request(
            f"{WIZARD_ENDPOINT}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    steps: list[dict] = []
    errors: list[str] = []

    def _step(label: str, ok: bool, detail: str = "", err: str = "") -> None:
        steps.append({"label": label, "ok": ok, "detail": detail, "error": err})
        if not ok and err:
            errors.append(err)

    # 1. Multi-pool concept binding — the path /chat routes through.
    #    Without this, the slow-pool training (step 2) only feeds the
    #    motif graph and the character-chain fallback; the user won't
    #    see their answer come back via the multi_pool decoder.
    try:
        mp_payload = json.dumps({
            "src_pool": "in",  "src": question,
            "tgt_pool": "out", "tgt": answer,
            "passes":   35,  # GA-tuned saturation default
        }).encode()
        mp_result = _post("/multi_pool/train_pair", mp_payload, timeout=60)
        x = (mp_result.get("stats") or {}).get("cross_edges", 0)
        _step("Concept binding (multi_pool)", True,
              f"35 passes through /multi_pool/train_pair; cross_edges={x}")
    except Exception as exc:
        _step("Concept binding (multi_pool)", False, "",
              f"/multi_pool/train_pair failed: {exc}")

    # 2. Slow-pool sequence training — feeds the global Hebbian graph
    #    and the motif runtime so the pattern is reachable from related
    #    queries even when the exact question phrasing isn't asked.
    last_seq_result: dict = {}
    seq_passes_ok = 0
    for i in range(TRAIN_REPEATS):
        sid = str(uuid.uuid4())  # fresh per pass — prevents STDP suppression
        seq_payload = json.dumps({
            "session_id": sid,
            "base_lr":    TRAIN_LR,
            "tau_secs":   2.0,
            "frames": [
                {"modality": "text", "t_secs": 0.0, "lr_scale": 1.0,
                 "data_b64": _text_b64(question), "text": question,
                 "spans": _spans(question, "body", 0.0, 0, 2)},
                {"modality": "text", "t_secs": 1.0, "lr_scale": 1.0,
                 "data_b64": _text_b64(answer), "text": answer,
                 "spans": _spans(answer, "body", 1.0, 1, 2)},
            ],
        }).encode()
        try:
            last_seq_result = _post("/media/train_sequence", seq_payload, timeout=20)
            seq_passes_ok += 1
        except Exception as exc:
            errors.append(f"train_sequence pass {i+1}: {exc}")
    _step("Slow-pool sequence training", seq_passes_ok > 0,
          f"{seq_passes_ok}/{TRAIN_REPEATS} passes at lr={TRAIN_LR}",
          "" if seq_passes_ok > 0 else "all train_sequence passes failed")

    # 3. Dopamine flush — captures the trace tags step 2 left behind
    #    into permanent weight boosts via three-factor Hebbian LTP.
    try:
        _post("/neuro/record_episode",
              json.dumps({
                  "context_labels": [f"txt:word_{w}" for w in question.lower().split()[:12]],
                  "predicted": question,
                  "actual":    answer,
                  "streams":   [],
                  "surprise":  TRAIN_SURPRISE,
              }).encode(), timeout=6)
        _post("/neuro/reinforce",
              json.dumps({"confidence": TRAIN_SURPRISE}).encode(),
              timeout=6)
        _step("Dopamine consolidation (/neuro/reinforce)", True,
              f"three-factor LTP capture at confidence={TRAIN_SURPRISE}")
    except Exception as exc:
        _step("Dopamine consolidation (/neuro/reinforce)", False, "",
              f"reinforce failed: {exc}")

    # 4. Knowledge ingest — structured-recall fast-path.  Best-effort:
    #    if the knowledge runtime isn't available the rest of the
    #    pipeline still trained the pool correctly.
    #
    #    Schema MUST match `KnowledgeDocument` in
    #    crates/core/src/streaming/knowledge.rs:
    #      { doc_id: str (required), source: str (required),
    #        title: Optional[str], text_blocks: [TextBlock] }
    #    where TextBlock = { block_id, text, ... }.
    #    Wrong schema (e.g. {body, tags}) returns HTTP 422 and the
    #    step shows a non-fatal red X.
    try:
        doc_id = "wizard-chat-" + uuid.uuid4().hex[:16]
        _post("/knowledge/ingest",
              json.dumps({"document": {
                  "doc_id":  doc_id,
                  "source":  "wizard_chat_correction",
                  "title":   question[:80],
                  "text_blocks": [
                      {"block_id": doc_id + "-q",
                       "text": f"Question: {question}",
                       "section": "question",
                       "order": 0,
                       "source": "wizard_chat",
                       "confidence": 1.0},
                      {"block_id": doc_id + "-a",
                       "text": f"Answer: {answer}",
                       "section": "answer",
                       "order": 1,
                       "source": "wizard_chat",
                       "confidence": 1.0},
                  ],
              }}).encode(), timeout=6)
        _step("Knowledge ingest", True,
              f"doc_id={doc_id} indexed with Q + A text_blocks")
    except Exception as exc:
        _step("Knowledge ingest", False, "(non-fatal)", f"knowledge/ingest: {exc}")

    # 5. Recall verification — immediately query /chat and report the
    #    actual returned answer.  The user will see exactly what the
    #    system will say next time they ask, so they know whether the
    #    training took.
    verify_actual = ""
    verify_decoder = ""
    verify_conf = None
    verify_match = False
    try:
        chat_result = _post("/chat", json.dumps({
            "text": question, "hops": 2, "min_strength": 0.05,
        }).encode(), timeout=10)
        verify_actual  = (chat_result.get("answer") or "").strip()
        verify_decoder = chat_result.get("decoder", "?")
        verify_conf    = chat_result.get("confidence")
        # Match check: the trained answer appears in /chat output
        # (substring; lets the multi-pool concept decoder produce
        # whitespace-normalised output and still pass).
        wanted = re.sub(r"\s+", " ", answer.strip().lower())
        got    = re.sub(r"\s+", " ", verify_actual.lower())
        verify_match = wanted in got or got in wanted or wanted == got
        _step("Recall verification (/chat)", verify_match,
              f"decoder={verify_decoder} conf={verify_conf} "
              f"answer={verify_actual!r}",
              "" if verify_match else "trained answer NOT returned by /chat")
    except Exception as exc:
        _step("Recall verification (/chat)", False, "", f"/chat: {exc}")

    overall_ok = verify_match  # final criterion the user cares about
    return {
        "ok":              overall_ok,
        "verify_match":    verify_match,
        "verify_answer":   verify_actual,
        "verify_decoder":  verify_decoder,
        "verify_confidence": verify_conf,
        "steps":           steps,
        "repeats":         TRAIN_REPEATS,
        "error":           errors[-1] if errors and not overall_ok else "",
    }


def _web_search(query: str) -> str:
    """DuckDuckGo instant answer search; returns best available snippet."""
    params = urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1"
    })
    url = f"https://api.duckduckgo.com/?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "wizard-chat/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            return abstract[:1200]
        topics = data.get("RelatedTopics") or []
        snippets = [
            t["Text"] for t in topics[:5]
            if isinstance(t, dict) and t.get("Text")
        ]
        if snippets:
            return " ".join(snippets)[:1200]
        answer_box = (data.get("Answer") or "").strip()
        return answer_box[:800] if answer_box else ""
    except Exception:
        return ""


def _extract_file_text(name: str, data: bytes, mime: str) -> str:
    """Extract plain-text content from an uploaded file."""
    name_lower = name.lower()
    try:
        # Plain text variants
        if (name_lower.endswith((".txt", ".md", ".csv", ".log", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env"))
                or mime.startswith("text/")):
            return data.decode("utf-8", errors="replace")[:12000]

        # PDF
        if name_lower.endswith(".pdf") or "pdf" in mime:
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    pages = [(p.extract_text() or "") for p in pdf.pages[:30]]
                return "\n".join(pages)[:14000]
            except ImportError:
                pass
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                texts = [(p.extract_text() or "") for p in reader.pages[:30]]
                return "\n".join(texts)[:14000]
            except ImportError:
                pass
            return f"[PDF: {name} — install pdfplumber or pypdf to extract text]"

        # Word docx
        if name_lower.endswith(".docx") or "wordprocessingml" in mime:
            try:
                from docx import Document
                doc = Document(io.BytesIO(data))
                paras = [p.text for p in doc.paragraphs if p.text.strip()]
                return "\n".join(paras)[:14000]
            except ImportError:
                return f"[Word document: {name} — install python-docx to extract text]"

        # Legacy .doc (very basic — extract readable ASCII)
        if name_lower.endswith(".doc"):
            text = "".join(chr(b) for b in data if 32 <= b < 127 or b in (9, 10, 13))
            return text[:8000] if text.strip() else f"[Binary .doc file: {name}]"

        # Images — no vision on the wizard node yet
        if name_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg")):
            return f"[Image file: {name}, {len(data):,} bytes — image analysis not yet available on this node]"

        # Video
        if name_lower.endswith((".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv", ".wmv")):
            return f"[Video file: {name}, {len(data):,} bytes — video analysis not yet available on this node]"

        # Audio
        if name_lower.endswith((".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".opus")):
            return f"[Audio file: {name}, {len(data):,} bytes — audio transcription not yet available on this node]"

        # Generic: try UTF-8 decoding
        try:
            text = data.decode("utf-8")
            if sum(1 for c in text[:200] if c.isprintable() or c in "\n\r\t") > 100:
                return text[:10000]
        except Exception:
            pass

        return f"[Binary file: {name}, {len(data):,} bytes — no text extraction available]"

    except Exception as exc:
        return f"[Error extracting {name}: {exc}]"


def _node_online() -> bool:
    try:
        with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/health", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def _build_answer(wizard_data: dict, web_snippet: str, question: str) -> dict:
    """
    Combine wizard node answer + optional web snippet into a final chat response.
    Returns: {answer, is_hypothesis, confidence_tier, web_used, concepts}
    """
    raw_answer = (wizard_data.get("answer") or "").strip()
    is_hypothesis = bool(wizard_data.get("hypothesis", False))
    tier = wizard_data.get("confidence_tier", "uncertain")
    concepts = wizard_data.get("activated_concepts") or []

    # Node is completely offline
    if wizard_data.get("confidence_tier") == "offline":
        answer = (
            "The W1z4rD Vision Node is currently offline. "
            "Start it with: `cd D:\\Projects\\W1z4rDV1510n && .\\bin\\w1z4rd_node.exe`"
        )
        if web_snippet:
            answer += f"\n\n**Web search result:**\n{web_snippet}"
        return {"answer": answer, "is_hypothesis": True, "confidence_tier": "offline",
                "web_used": bool(web_snippet), "concepts": []}

    # Good answer from node
    if raw_answer and not is_hypothesis:
        return {"answer": raw_answer, "is_hypothesis": False,
                "confidence_tier": tier, "web_used": False, "concepts": concepts}

    # Hypothesis / low confidence — enrich with web search
    if web_snippet:
        answer = (
            f"{web_snippet}\n\n"
            f"*(Source: web search — the node is still learning about this topic. "
            f"You can correct this answer below to help it train.)*"
        )
        if raw_answer:
            answer = f"{raw_answer}\n\n---\n**Web search supplement:**\n{web_snippet}"
        return {"answer": answer, "is_hypothesis": True, "confidence_tier": tier,
                "web_used": True, "concepts": concepts}

    # No answer, no web results
    concept_str = ""
    if concepts:
        clean = [c.replace("txt:word_", "") for c in concepts[:8]]
        concept_str = f"\n\nRelated concepts the node activated: *{', '.join(clean)}*"
    answer = (
        "The W1z4rD node doesn't have a confident answer for this yet. "
        "It has been added to the hypothesis queue for future training."
        f"{concept_str}\n\n"
        "You can provide the correct answer below to train the node."
    )
    return {"answer": answer, "is_hypothesis": True, "confidence_tier": tier,
            "web_used": False, "concepts": concepts}


# ---------------------------------------------------------------------------
# API Views
# ---------------------------------------------------------------------------

@method_decorator(csrf_exempt, name="dispatch")
class WizardChatMessageView(View):
    """
    POST /api/wizard-chat/message/
    Body: {text, session_id?, attachments?: [{name, content (base64 or text)}]}
    Returns: {answer, is_hypothesis, confidence_tier, web_used, concepts, node_online}
    """

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        text = (body.get("text") or "").strip()
        session_id = (body.get("session_id") or str(uuid.uuid4()))
        attachment_texts: list[str] = body.get("attachment_texts") or []

        if not text and not attachment_texts:
            return JsonResponse({"error": "No message content"}, status=400)

        # Rolling conversation context — prepended to the prompt so the
        # backend sees what we've been working on, what the active topic
        # is, and the last N turns verbatim.  Without this every /chat
        # call is a goldfish.
        from .conversation import STORE as CONV_STORE
        if text:
            CONV_STORE.append_turn(session_id, "user", text)
        context_blob = CONV_STORE.build_context_blob(session_id)

        # Build full prompt: context blob (if any) + attachments + new turn.
        parts = []
        if context_blob:
            parts.append(context_blob)
        if attachment_texts:
            for i, at in enumerate(attachment_texts, 1):
                parts.append(f"[Attachment {i}]\n{at[:6000]}")
        # The user's bare turn is already embedded in the context blob;
        # append a "Now answer:" cue plus the bare text so the backend
        # knows what to respond to (not just summarise).
        if text:
            parts.append(f"[Now answer concisely]\n{text}")
        full_prompt = "\n\n".join(parts)

        wizard_data = _wizard_ask(full_prompt, session_id)

        # Derive online status from the response — no separate health ping needed.
        node_up = wizard_data.get("confidence_tier") not in ("offline", "error")

        # Web-search only when the node returned NO answer at all — if there's
        # any Hebbian output (even uncertain), show it immediately without waiting
        # for an external search to complete.
        web_snippet = ""
        is_hypothesis = bool(wizard_data.get("hypothesis", False)) or not (wizard_data.get("answer") or "").strip()
        answer_empty = not (wizard_data.get("answer") or "").strip()
        if answer_empty and node_up:
            search_query = text if text else full_prompt[:200]
            fut = _thread_pool.submit(_web_search, search_query)
            try:
                web_snippet = fut.result(timeout=WEB_SEARCH_TIMEOUT)
            except Exception:
                web_snippet = ""

        result = _build_answer(wizard_data, web_snippet, full_prompt)
        result["node_online"] = node_up
        result["session_id"] = session_id
        # Record the wizard's response into the rolling context so the
        # next turn sees it.  Use the final composed answer (after
        # web-snippet augmentation), not the raw node response.
        answer_to_store = (result.get("answer") or "").strip()
        if answer_to_store:
            CONV_STORE.append_turn(session_id, "wizard", answer_to_store)
        # Expose the active topic + turn count so the UI can render
        # a "what we're working on" chip.
        snap = CONV_STORE.snapshot(session_id)
        result["active_topic"] = snap["active_topic"]
        result["turn_count"]   = snap["turn_count"]
        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatSessionView(View):
    """GET /api/wizard-chat/session/?session_id=... — current rolling
    context snapshot (active topic, summary, turn count, last N turns)."""

    def get(self, request):
        from .conversation import STORE as CONV_STORE
        sid = request.GET.get("session_id", "").strip()
        if not sid:
            return JsonResponse({"error": "session_id required"}, status=400)
        return JsonResponse(CONV_STORE.snapshot(sid))


# Path of the W1z4rD node's training event log.  The runner writes
# every benchmark + train_start/train_end + regression_alert event
# here as JSONL.  Configurable via env so the UI follows the same
# convention as the runner (W1Z4RD_TRAINING_EVENTS).
import os as _os
_TRAINING_EVENTS_PATH = _os.environ.get(
    "W1Z4RD_TRAINING_EVENTS",
    str(_os.environ.get("W1Z4RDV1510N_DATA_DIR", "D:/w1z4rdv1510n-data"))
    + "/training/training_events.jsonl",
)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatTrainingLiveView(View):
    """GET /api/wizard-chat/training/live/?since_ts=...&limit=80

    Returns the most recent training events + a brain snapshot so the
    UI's bottom-of-chat panel can render per-pool tiles AND a scrolling
    list of training activity without two separate fetches.

    Query params:
      since_ts:  ISO8601 timestamp — only return events strictly after this
      limit:     max events returned (default 80, capped at 500)

    Response:
      brain:        current /brain snapshot
      events:       array of event dicts, OLDEST → NEWEST
      latest_ts:    ISO8601 of the newest event returned, for next-poll cursor
      events_count: int
    """

    MAX_LIMIT = 500

    def get(self, request):
        since_ts = (request.GET.get("since_ts") or "").strip()
        try:
            limit = int(request.GET.get("limit") or "80")
        except ValueError:
            limit = 80
        limit = max(1, min(self.MAX_LIMIT, limit))

        events = self._read_events_tail(_TRAINING_EVENTS_PATH, limit, since_ts)
        # Generous timeouts under training contention — see status view.
        brain = {}
        try:
            with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/brain", timeout=10) as r:
                brain = json.loads(r.read())
        except Exception:
            pass
        multi_pool_stats: dict = {}
        try:
            with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/multi_pool/stats", timeout=10) as r:
                multi_pool_stats = json.loads(r.read())
        except Exception:
            pass

        latest_ts = events[-1].get("ts", "") if events else since_ts
        return JsonResponse({
            "brain":            brain,
            "multi_pool_stats": multi_pool_stats,
            "events":           events,
            "latest_ts":        latest_ts,
            "events_count":     len(events),
        })

    @staticmethod
    def _read_events_tail(path: str, limit: int, since_ts: str) -> list[dict]:
        """Read the last N events (or events after since_ts) from the JSONL log.
        Robust to a missing file (returns empty)."""
        try:
            import pathlib
            p = pathlib.Path(path)
            if not p.exists():
                return []
            # Tail: read up to 2 MB from the end and parse lines.  The
            # training event log is small (typical event << 1 KB) so
            # this is bounded.  Larger logs: switch to a true reverse
            # reader, but 2 MB covers months of typical activity.
            size = p.stat().st_size
            with p.open("rb") as fh:
                if size > 2_000_000:
                    fh.seek(-2_000_000, _os.SEEK_END)
                    fh.readline()  # discard partial line
                raw = fh.read().decode("utf-8", errors="replace")
        except Exception:
            return []
        events: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since_ts and ev.get("ts", "") <= since_ts:
                continue
            events.append(ev)
        # Cap to last `limit`.
        if len(events) > limit:
            events = events[-limit:]
        return events


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatUploadView(View):
    """
    POST /api/wizard-chat/upload/  (multipart/form-data)
    Field: files[]  (multiple files allowed)
    Returns: [{name, text, size, type}] — extracted text per file
    """

    def post(self, request):
        files = request.FILES.getlist("files") or request.FILES.getlist("files[]")
        if not files:
            return JsonResponse({"error": "No files provided"}, status=400)

        results = []
        for f in files[:20]:  # cap at 20 files per upload
            size = f.size
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                results.append({
                    "name": f.name,
                    "text": f"[File too large: {size:,} bytes. Max {MAX_UPLOAD_MB} MB]",
                    "size": size,
                    "type": f.content_type or "",
                    "error": True,
                })
                continue
            try:
                data = f.read()
                text = _extract_file_text(f.name, data, f.content_type or "")
                # Route raw bytes to the node's /sensor/observe so the
                # corresponding modality pool gets trained.  Text-only
                # files (PDFs, plaintext) go to pdf_text/keyboard_text;
                # images go to image_pixels; audio to audio_features.
                # The extracted text is still returned to the frontend
                # so the chat prompt can be augmented with it — but the
                # original bytes also reach the node now.
                sensor_kind, sensor_body = _sensor_kind_for_file(
                    f.name, f.content_type or "", data, text,
                )
                sensor_ok = False
                sensor_err = ""
                if sensor_kind is not None:
                    sensor_ok, sensor_err = _post_sensor_observe(
                        sensor_kind, sensor_body,
                    )
                results.append({
                    "name": f.name,
                    "text": text,
                    "size": size,
                    "type": f.content_type or "",
                    "error": False,
                    "sensor_kind":  sensor_kind,
                    "sensor_ok":    sensor_ok,
                    "sensor_error": sensor_err,
                })
            except Exception as exc:
                results.append({
                    "name": f.name,
                    "text": f"[Upload error: {exc}]",
                    "size": size,
                    "type": f.content_type or "",
                    "error": True,
                })

        return JsonResponse({"files": results})


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatTrainView(View):
    """
    POST /api/wizard-chat/train/
    Body: {question, answer, session_id?}
    Ingest a corrected answer into the wizard node's training data.
    """

    def post(self, request):
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        question = (body.get("question") or "").strip()
        answer = (body.get("answer") or "").strip()
        session_id = (body.get("session_id") or "")

        if not question or not answer:
            return JsonResponse({"error": "Both question and answer are required"}, status=400)

        result = _wizard_train(question, answer, session_id)
        return JsonResponse(result)


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatStatusView(View):
    """GET /api/wizard-chat/status/ — node health + brain snapshot.

    Returns:
      online:     bool — node reachable
      endpoint:   str  — wizard node URL
      health:     dict — node's /health response (uptime, node_id, status)
      brain:      dict — node's /brain response (pools, cross_edges,
                          neuromodulators, motifs, feedback_recipes).
                          Used by the UI's top status strip to show rich
                          live state without an extra round-trip.
    """

    def get(self, request):
        try:
            with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/health", timeout=5) as r:
                health = json.loads(r.read())
        except Exception as exc:
            return JsonResponse({
                "online": False,
                "endpoint": WIZARD_ENDPOINT,
                "error": str(exc),
                "health": {},
                "brain": {},
                "multi_pool_stats": {},
            })
        # Generous timeouts: under heavy training load (drive_corpora
        # POSTing observations several times per second) the node's
        # inner lock can hold for >4s and both /brain and stats fetches
        # would time out, leaving the UI with empty payloads.  10s is
        # well under any reasonable polling cadence and covers the
        # worst observed contention.
        brain = {}
        try:
            with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/brain", timeout=10) as r:
                brain = json.loads(r.read())
        except Exception:
            pass
        multi_pool_stats: dict = {}
        try:
            with urllib.request.urlopen(f"{WIZARD_ENDPOINT}/multi_pool/stats", timeout=10) as r:
                multi_pool_stats = json.loads(r.read())
        except Exception:
            pass
        return JsonResponse({
            "online": True,
            "endpoint": WIZARD_ENDPOINT,
            "health": health,
            "brain": brain,
            "multi_pool_stats": multi_pool_stats,
        })


def _node_fetch(path: str, timeout: float = 5.0) -> dict:
    """GET from the wizard node; returns {} on error."""
    try:
        with urllib.request.urlopen(f"{WIZARD_ENDPOINT}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def _qa_query(pool_name: str = "knowledge", top_k: int = 50) -> dict:
    payload = json.dumps({"query": "", "top_k": top_k, "pool": pool_name}).encode()
    req = urllib.request.Request(
        f"{WIZARD_ENDPOINT}/qa/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read())
    except Exception:
        return {}


@method_decorator(csrf_exempt, name="dispatch")
class WizardChatPoolsView(View):
    """
    GET /api/wizard-chat/pools/          — discover + return all pool data
    GET /api/wizard-chat/pools/?pool=qa  — single pool by key

    Returns { pools: { <key>: { label, type, data, count } }, discovered: [...keys] }
    """

    # All endpoints to probe; each produces one or more named pools.
    _POOL_ENDPOINTS = [
        ("neuro_snapshot",  "GET",  "/neuro/snapshot",    None),
        ("knowledge",       "GET",  "/knowledge/queue",   None),
        ("equations",       "GET",  "/equations/report",  None),
        ("streaming_labels","GET",  "/streaming/labels",  None),
        ("subnet_report",   "GET",  "/streaming/subnets", None),
        ("causal_graph",    "GET",  "/causal/graph",      None),
        ("health",          "GET",  "/health",            None),
    ]

    def get(self, request):
        only = request.GET.get("pool")  # optional filter

        def _fetch_one(endpoint_key, method, path, payload):
            if method == "GET":
                return endpoint_key, _node_fetch(path, timeout=6)
            payload_bytes = json.dumps(payload).encode() if payload else b"{}"
            req = urllib.request.Request(
                f"{WIZARD_ENDPOINT}{path}",
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method=method,
            )
            try:
                with urllib.request.urlopen(req, timeout=6) as r:
                    return endpoint_key, json.loads(r.read())
            except Exception:
                return endpoint_key, {}

        # Submit all in parallel
        endpoint_list = [
            e for e in self._POOL_ENDPOINTS if not only or e[0] == only
        ]
        # Always include both QA pools (knowledge + corrections)
        qa_futures = {}
        if not only or only in ("qa_knowledge", "qa_corrections"):
            qa_futures["qa_knowledge"]    = _thread_pool.submit(_qa_query, "knowledge", 50)
            qa_futures["qa_corrections"]  = _thread_pool.submit(_qa_query, "corrections", 50)

        futs = {key: _thread_pool.submit(_fetch_one, key, m, p, body)
                for key, m, p, body in endpoint_list}

        raw: dict[str, dict] = {}
        for key, fut in futs.items():
            try:
                _, data = fut.result(timeout=7)
                raw[key] = data
            except Exception:
                raw[key] = {}

        for key, fut in qa_futures.items():
            try:
                raw[key] = fut.result(timeout=7)
            except Exception:
                raw[key] = {}

        # Shape into named pool entries with metadata
        pools: dict[str, dict] = {}

        def _add(key: str, label: str, pool_type: str, data: dict, count_hint=None):
            if not data:
                return
            count = count_hint
            if count is None:
                for field in ("total", "count", "pairs_ingested", "size", "length"):
                    if isinstance(data.get(field), int):
                        count = data[field]
                        break
            pools[key] = {"label": label, "type": pool_type, "data": data, "count": count}

        _add("qa_knowledge",    "QA — Knowledge",     "qa",         raw.get("qa_knowledge", {}),
             len(raw.get("qa_knowledge", {}).get("matches") or []))
        _add("qa_corrections",  "QA — Corrections",   "qa",         raw.get("qa_corrections", {}),
             len(raw.get("qa_corrections", {}).get("matches") or []))
        _add("knowledge",       "Knowledge Docs",      "knowledge",  raw.get("knowledge", {}),
             len(raw.get("knowledge", {}).get("queue") or []))
        _add("equations",       "Equations (EEM)",     "equations",  raw.get("equations", {}))
        _add("neuro_snapshot",  "Hebbian State",       "neuro",      raw.get("neuro_snapshot", {}))
        _add("streaming_labels","Streaming Labels",    "labels",     raw.get("streaming_labels", {}))
        _add("subnet_report",   "Subnet Report",       "subnets",    raw.get("subnet_report", {}))
        _add("causal_graph",    "Causal Graph",        "causal",     raw.get("causal_graph", {}))

        # Expose pool stats from health endpoint as a meta pool
        health = raw.get("health", {})
        if health:
            _add("node_health", "Node Health", "meta", health)

        return JsonResponse({
            "pools":      pools,
            "discovered": list(pools.keys()),
        })
